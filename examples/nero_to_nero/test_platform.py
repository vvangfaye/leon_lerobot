"""第一轮回归检查：模拟硬件，真实 LeRobot 临时数据；不会打开 CAN 或相机。"""

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import verify_platform_dataset
from nero_platform import camera_io, cameras, config, devices, recorder
from nero_platform.app import create_app
from nero_platform.runtime import DEFAULT_REPO_ID, Runtime

from lerobot.datasets import LeRobotDataset


class Clock:
    def __init__(self, events: list):
        self.now = 0.0
        self.events = events

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.events.append(("sleep", duration))
        self.now += duration


class FakeArm:
    def __init__(self, name: str, events: list):
        self.name = name
        self.events = events
        self.feedback_timestamp = 1.0
        self.enabled = True
        self.ctrl_mode = 1

    def feedback(self, positions: list[float]) -> SimpleNamespace:
        self.feedback_timestamp += 1
        return SimpleNamespace(msg=positions, timestamp=self.feedback_timestamp)

    def enable(self) -> bool:
        self.events.append((self.name, "enable"))
        self.enabled = True
        self.ctrl_mode = 1
        return True

    def set_leader_mode(self) -> None:
        self.events.append((self.name, "leader"))
        self.ctrl_mode = 6

    def set_follower_mode(self) -> None:
        self.events.append((self.name, "follower"))
        # 实机从臂在调用 set_follower_mode 后可上报联动模式 0x06。
        self.ctrl_mode = 6

    def get_arm_status(self) -> SimpleNamespace:
        self.feedback_timestamp += 1
        return SimpleNamespace(
            timestamp=self.feedback_timestamp,
            msg=SimpleNamespace(ctrl_mode=self.ctrl_mode, arm_status=0, err_code=0, motion_status=0),
        )

    def get_driver_states(self, joint: int) -> SimpleNamespace:
        self.feedback_timestamp += 1
        return SimpleNamespace(
            timestamp=self.feedback_timestamp,
            msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.enabled)),
        )

    def get_leader_joint_angles(self) -> SimpleNamespace:
        self.events.append((self.name, "read_leader"))
        return self.feedback([0.1 * i for i in range(7)])

    def get_joint_angles(self) -> SimpleNamespace:
        self.events.append((self.name, "read_follower"))
        return self.feedback([0.1 * i for i in range(7)])

    def move_js(self, angles: list) -> None:
        self.events.append((self.name, "move_js", list(angles)))


class FakeGripper:
    def __init__(self, name: str, events: list):
        self.name = name
        self.events = events
        self.feedback_timestamp = 1.0
        self.enabled = True

    def disable_gripper(self) -> bool:
        self.events.append((self.name, "disable_gripper"))
        self.enabled = False
        return True

    def set_gripper_teaching_pendant_param(self, **kwargs) -> bool:
        self.events.append((self.name, "teaching", kwargs))
        return True

    def get_gripper_teaching_pendant_param(self, **kwargs) -> SimpleNamespace | None:
        self.events.append((self.name, "read_teaching", kwargs))
        return None

    def get_gripper_ctrl_states(self) -> SimpleNamespace:
        self.events.append((self.name, "read_input"))
        self.feedback_timestamp += 1
        return SimpleNamespace(msg=SimpleNamespace(value=0.025), timestamp=self.feedback_timestamp)

    def get_gripper_status(self) -> SimpleNamespace:
        self.events.append((self.name, "read_status"))
        self.feedback_timestamp += 1
        return SimpleNamespace(
            msg=SimpleNamespace(value=0.02, foc_status=SimpleNamespace(driver_enable_status=self.enabled)),
            timestamp=self.feedback_timestamp,
        )

    def move_gripper_m(self, **kwargs) -> None:
        self.events.append((self.name, "move", kwargs))


class FakeCamera:
    def __init__(self, cfg=None, events: list | None = None):
        self.events = events if events is not None else []
        # 彩色非对称图案也经过真实视频编码和解码。
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame[:, :320, 0] = 220
        self.frame[:, 320:, 1] = 180

    def connect(self) -> None:
        self.events.append(("camera", "connect"))

    def disconnect(self) -> None:
        self.events.append(("camera", "disconnect"))

    def read_latest(self) -> np.ndarray:
        self.events.append(("camera", "read"))
        return self.frame.copy()

    def read_latest_with_timestamp(self) -> tuple[np.ndarray, float, float]:
        return self.read_latest(), time.perf_counter(), 0.0


class MemoryDataset:
    def __init__(self, events: list):
        self.events = events
        self.frames = []
        self.meta = SimpleNamespace(total_episodes=0)

    def add_frame(self, frame: dict) -> None:
        self.events.append(("dataset", "add_frame"))
        self.frames.append(copy.deepcopy(frame))

    def has_pending_frames(self) -> bool:
        return bool(self.frames)

    def save_episode(self) -> None:
        self.events.append(("dataset", "save_episode"))
        self.meta.total_episodes += 1

    def finalize(self) -> None:
        self.events.append(("dataset", "finalize"))


def attach_fake_pair(runtime: Runtime, events: list) -> None:
    for ch, label in (("can0", "主左"), ("can1", "从左")):
        runtime.channels[ch].update(
            arm=FakeArm(label, events), gripper=FakeGripper(label, events), label=label, detected=True
        )
    runtime.state.update(leader_label="主左", follower_label="从左")


class PlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        # 一旦任何测试意外走到真实设备创建入口，立即失败。
        self.arm_guard = patch.object(
            devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")
        )
        self.camera_guard = patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用"))
        self.collection_guard = patch.object(
            camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")
        )
        for guard in (self.arm_guard, self.camera_guard, self.collection_guard):
            guard.start()
            self.addCleanup(guard.stop)

    def test_import_and_read_only_routes_do_not_open_hardware(self) -> None:
        runtime = Runtime()
        with create_app(runtime).test_client() as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertFalse(client.get("/api/status").json["connected"])
            self.assertEqual(client.get("/api/config").json["repo_id"], DEFAULT_REPO_ID)
        self.assertIsNone(runtime.dataset)

    def test_config_roundtrip_and_legacy_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "collect_config.json"
            original = json.dumps(
                {"arms": {"can2": "主右"}, "fps": 15, "task": "抓取方块", "repo_id": "nero/old"}
            )
            legacy.write_text(original)
            runtime = Runtime(config_path=root / "platform.json")
            config.load_saved(runtime, legacy)
            self.assertEqual(runtime.state["repo_id"], DEFAULT_REPO_ID)
            runtime.state["repo_id"] = "nero/new"
            config.save_config(runtime, {})
            restarted = Runtime(config_path=runtime.config_path)
            config.load_saved(restarted, legacy)
            self.assertEqual(restarted.state["repo_id"], "nero/new")
            self.assertEqual(restarted.channels["can2"]["label"], "主右")
            self.assertEqual(restarted.state["task"], "抓取方块")
            self.assertEqual(legacy.read_text(), original)
            with create_app(restarted).test_client() as client:
                self.assertEqual(client.get("/api/config").json["fps"], 15)

    def test_commands_match_recorded_action_and_rgb_frames(self) -> None:
        events = []
        runtime = Runtime()
        runtime.state.update(fps=10, episode_sec=0.25)
        attach_fake_pair(runtime, events)
        runtime.dataset = MemoryDataset(events)
        camera = FakeCamera(events=events)
        runtime.collection_cams["left"] = camera
        clock = Clock(events)
        with patch.object(recorder, "time", clock):
            recorder.record_worker(runtime, ["left"])
        self.assertEqual(runtime.state["episode_count"], 1)
        moves = [event[2] for event in events if event[1] == "move_js"]
        self.assertEqual(len(moves), 25)
        for command, frame in zip(moves[::10], runtime.dataset.frames, strict=True):
            np.testing.assert_allclose(frame["action"][:7], command, atol=1e-6)
            self.assertAlmostEqual(float(frame["action"][7]), 0.025)
            np.testing.assert_allclose(frame["observation.state"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02])
            np.testing.assert_array_equal(frame["observation.images.left"], camera.frame)

    def test_disconnect_waits_for_episode_before_finalizing(self) -> None:
        runtime = Runtime()
        events = []
        runtime.dataset = MemoryDataset(events)
        entered = threading.Event()
        allow_save = threading.Event()

        def worker() -> None:
            entered.set()
            allow_save.wait(5)
            runtime.dataset.save_episode()

        runtime.record_thread = threading.Thread(target=worker)
        runtime.record_thread.start()
        self.assertTrue(entered.wait(2))
        disconnect = threading.Thread(target=recorder.do_disconnect, args=(runtime, False))
        disconnect.start()
        try:
            self.assertNotIn(("dataset", "finalize"), events)
        finally:
            allow_save.set()
            disconnect.join(5)
        self.assertFalse(disconnect.is_alive())
        self.assertEqual(events, [("dataset", "save_episode"), ("dataset", "finalize")])

    def test_shutdown_keeps_joint_and_gripper_power(self) -> None:
        runtime = Runtime()
        events = []
        attach_fake_pair(runtime, events)
        runtime.state["connected"] = True
        runtime.dataset = MemoryDataset(events)
        with patch.object(recorder, "time", Clock(events)):
            recorder.do_disconnect(runtime, reopen_preview=False)
        self.assertFalse(runtime.state["connected"])
        self.assertTrue(runtime.channels["can0"]["arm"].enabled)
        self.assertTrue(runtime.channels["can1"]["arm"].enabled)
        self.assertTrue(
            all(
                e[1] in ("finalize", "follower", "role_request", "read_follower") or e[0] == "sleep"
                for e in events
            )
        )

    def test_empty_episode_reports_error_and_resets_recording(self) -> None:
        runtime = Runtime()
        attach_fake_pair(runtime, [])
        runtime.state.update(episode_sec=0, recording=True)
        runtime.dataset = MemoryDataset([])
        with patch.object(recorder.traceback, "print_exc"):
            recorder.record_worker(runtime, [])
        self.assertFalse(runtime.state["recording"])
        self.assertEqual(runtime.state["episode_count"], 0)
        self.assertIn("没有有效帧", runtime.state["last_error"])

    def test_config_changes_are_rejected_while_connected(self) -> None:
        runtime = Runtime()
        runtime.state["connected"] = True
        with create_app(runtime).test_client() as client:
            result = client.post("/api/save", json={"fps": 15})
            self.assertEqual(result.status_code, 409)
            self.assertEqual(runtime.state["fps"], 30)
            self.assertEqual(client.post("/api/arm/move/can0").status_code, 409)

    def test_incompatible_resume_does_not_change_hardware_modes(self) -> None:
        runtime = Runtime()
        runtime.state.update(record_camera="right", camera_map={"fake-right": "right"})
        events = []
        attach_fake_pair(runtime, events)
        with (
            patch.object(recorder, "validate_resume", side_effect=ValueError("帧率不一致")),
            patch.object(cameras, "detect_cameras", return_value={"fake-right": "/fake/right"}),
            patch.object(cameras, "open_preview_cams") as open_preview,
            patch.object(recorder.traceback, "print_exc"),
        ):
            ok, msg = recorder.do_connect(runtime)
        self.assertFalse(ok)
        self.assertIn("帧率不一致", msg)
        self.assertEqual(events, [])
        open_preview.assert_not_called()

    def test_real_lerobot_two_episodes_restart_and_resume_with_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nero-step1-") as directory:
            root = Path(directory)
            runtime = Runtime(config_path=root / "config.json", dataset_root=root / "datasets")
            runtime.state.update(
                fps=10, episode_sec=0.25, camera_map={"fake-left": "left", "fake-right": "right"}
            )
            attach_fake_pair(runtime, [])
            detected = {"fake-left": "/fake/left", "fake-right": "/fake/right"}
            saved_videos = {}
            with (
                patch.object(cameras, "detect_cameras", return_value=detected),
                patch.object(camera_io, "RecordingCamera", FakeCamera),
            ):
                for session_index, episode_count in enumerate((2, 1)):
                    clock = Clock([])
                    with patch.object(recorder, "time", clock), patch.object(devices, "time", clock):
                        ok, msg = recorder.do_connect(runtime)
                        self.assertTrue(ok, msg)
                        self.assertEqual(runtime.state["episode_count"], 0 if session_index == 0 else 2)
                        for _ in range(episode_count):
                            recorder.record_worker(runtime, ["left", "right"])
                            self.assertIsNone(runtime.state["last_error"])
                            for path, content in saved_videos.items():
                                self.assertEqual(
                                    path.read_bytes(), content, "后续录制不能拼接或覆盖已保存视频"
                                )
                            saved_videos = {
                                path: path.read_bytes()
                                for path in (recorder.dataset_path(runtime) / "videos").rglob("*.mp4")
                            }
                            self.assertEqual(len(saved_videos), runtime.state["episode_count"] * 2)
                        config.save_config(runtime, detected)
                        recorder.do_disconnect(runtime, reopen_preview=False)
                    dataset_root = recorder.dataset_path(runtime)
                    loaded = LeRobotDataset(runtime.state["repo_id"], root=dataset_root, video_backend="pyav")
                    self.assertEqual(loaded.num_episodes, 2 if session_index == 0 else 3)
                    for role in ("left", "right"):
                        key = f"observation.images.{role}"
                        paths = [loaded.meta.get_video_file_path(i, key) for i in range(loaded.num_episodes)]
                        self.assertEqual(len(set(paths)), loaded.num_episodes)
                        for index, path in enumerate(paths):
                            episode = loaded.meta.episodes[index]
                            self.assertEqual(episode[f"videos/{key}/from_timestamp"], 0)
                            with av.open(str(dataset_root / path)) as container:
                                self.assertEqual(sum(1 for _ in container.decode(video=0)), episode["length"])
                    for frame_index in range(loaded.num_frames):
                        frame = loaded[frame_index]
                        np.testing.assert_allclose(
                            frame["action"].numpy(), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.025], atol=1e-6
                        )
                        self.assertEqual(tuple(frame["observation.state"].shape), (8,))
                        for role in ("left", "right"):
                            picture = frame[f"observation.images.{role}"]
                            self.assertEqual(tuple(picture.shape), (3, 480, 640))
                            self.assertGreater(float(picture[0, :, :320].mean()), 0.7)
                    if session_index == 0:
                        first_frames = loaded.num_frames
                        first_actions = np.asarray(loaded.hf_dataset[:]["action"]).copy()
                        restarted = Runtime(
                            config_path=runtime.config_path, dataset_root=runtime.dataset_root
                        )
                        config.load_saved(restarted)
                        attach_fake_pair(restarted, [])
                        runtime = restarted
                    else:
                        self.assertGreater(loaded.num_frames, first_frames)
                        np.testing.assert_equal(
                            np.asarray(loaded.hf_dataset[:first_frames]["action"]), first_actions
                        )
                        incompatible = recorder.make_features({"left": {"type": "rgb", "path": "/l"}})
                        with self.assertRaises(ValueError):
                            recorder.validate_resume(dataset_root, 10, incompatible)
                        with self.assertRaises(ValueError):
                            recorder.validate_resume(
                                dataset_root,
                                15,
                                recorder.make_features(
                                    {
                                        "left": {"type": "rgb", "path": "/l"},
                                        "right": {"type": "rgb", "path": "/r"},
                                    }
                                ),
                            )
                    with patch(
                        "sys.argv",
                        [
                            "verify_platform_dataset.py",
                            "--repo-id",
                            runtime.state["repo_id"],
                            "--dataset-root",
                            str(runtime.dataset_root),
                            "--expected-episodes",
                            str(2 if session_index == 0 else 3),
                            "--require-timing",
                            "--last-episodes",
                            "2",
                        ],
                    ):
                        verify_platform_dataset.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
