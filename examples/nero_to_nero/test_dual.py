"""第四步：四臂均为模拟设备，验证配对隔离、共同时间轴与真实 16 维数据读写。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import verify_platform_dataset
from nero_platform import camera_io, cameras, config, devices, pairing, recorder, timing
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_disable import DisableArm, DisableGripper
from test_home import HomeArm
from test_platform import Clock, FakeCamera, MemoryDataset

from lerobot.datasets import LeRobotDataset


class ValueGripper(DisableGripper):
    def __init__(self, name: str, events: list, value: float):
        super().__init__(name, events)
        self.value = value

    def get_gripper_ctrl_states(self) -> SimpleNamespace:
        feedback = super().get_gripper_ctrl_states()
        feedback.msg.value = self.value
        return feedback

    def get_gripper_status(self) -> SimpleNamespace:
        feedback = super().get_gripper_status()
        feedback.msg.value = self.value
        return feedback


class DualTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-dual-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.events = []
        self.clock = Clock(self.events)
        self.runtime = Runtime(config_path=self.root / "config.json", dataset_root=self.root / "datasets")
        self.attach(self.runtime)
        self.runtime.state.update(
            pair_mode="dual",
            connected=True,
            fps=10,
            episode_sec=0.3,
            repo_id="nero/dual_test",
            record_camera="right",
            camera_map={"fake-right": "right"},
        )
        self.runtime.dataset = MemoryDataset(self.events)
        self.runtime.dataset.root = self.root / "memory"
        self.client = create_app(self.runtime).test_client()
        for guard in (
            patch.object(devices, "time", self.clock),
            patch.object(recorder, "time", self.clock),
            patch.object(cameras, "detect_cameras", return_value={"fake-right": "/fake/right"}),
            patch.object(cameras, "open_preview_cams"),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(recorder.traceback, "print_exc"),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.addCleanup(self.finish)

    def attach(self, runtime: Runtime) -> None:
        # 故意打乱 CAN 顺序，禁止实现用通道编号猜左右。
        for ch, label in zip(runtime.channels, ("从右", "主左", "从左", "主右"), strict=True):
            arm = HomeArm(label, self.events)
            arm.positions = [0.1 if label.endswith("左") else 0.4] * 7
            grip = {"主左": 0.015, "从左": 0.01, "主右": 0.05, "从右": 0.045}[label]
            runtime.channels[ch].update(
                label=label, detected=True, arm=arm, gripper=ValueGripper(label, self.events, grip)
            )

    def finish(self) -> None:
        for thread in (self.runtime.record_thread, self.runtime.home_thread):
            if thread is not None:
                thread.join(5)
                self.assertFalse(thread.is_alive())

    def arm(self, label: str) -> HomeArm | DisableArm:
        return self.runtime.channels[devices.channel_by_label(self.runtime, label)]["arm"]

    def record(self) -> None:
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()

    def targets(self) -> dict:
        return pairing.targets(pairing.resolve(self.runtime))

    def report(self) -> dict:
        return json.loads(timing.report_path(self.runtime.dataset.root, 0).read_text())

    def test_config_roundtrip_and_old_config_defaults_to_single(self) -> None:
        self.runtime.state["connected"] = False
        config.save_config(self.runtime, {})
        restarted = Runtime(config_path=self.runtime.config_path)
        config.load_saved(restarted)
        self.assertEqual(restarted.state["pair_mode"], "dual")
        saved = json.loads(self.runtime.config_path.read_text())
        del saved["pair_mode"]
        self.runtime.config_path.write_text(json.dumps(saved))
        legacy = Runtime(config_path=self.runtime.config_path)
        config.load_saved(legacy)
        self.assertEqual(legacy.state["pair_mode"], "single")
        self.assertIsNone(legacy.state["config_error"])
        for value in ("both", "", None, 2):
            with self.assertRaises(ValueError):
                config.validate_settings(self.runtime, {"pair_mode": value})

    def test_feature_order_and_active_config_lock(self) -> None:
        ft = recorder.make_features({"right": {"type": "rgb", "path": "/fake/right"}}, "dual")
        self.assertEqual(ft["action"]["shape"], (16,))
        self.assertEqual(ft["action"]["names"], pairing.joint_keys("dual"))
        self.assertEqual(ft["observation.state"]["names"], ft["action"]["names"])
        self.assertEqual(self.client.post("/api/save", json={"pair_mode": "single"}).status_code, 409)
        self.assertEqual(self.client.get("/api/status").json["pair_mode"], "dual")

    def test_missing_or_duplicate_right_role_rejected_before_any_enable(self) -> None:
        for missing in (True, False):
            with self.subTest(missing=missing):
                right = self.runtime.channels["can0"]
                right.update(label="从右", detected=True)
                right["detected" if missing else "label"] = False if missing else "从左"
                self.events.clear()
                with self.assertRaises(ValueError):
                    devices.transition_to_collection(self.runtime)
                self.assertEqual(self.events, [])
                self.assertEqual(self.client.post("/api/record/start").status_code, 409)
                self.assertEqual(self.events, [])

    def test_transition_switches_both_pairs_without_crossing_roles(self) -> None:
        devices.transition_to_collection(self.runtime)
        self.assertEqual([e[0] for e in self.events if e[1] == "leader"], ["主左", "主右"])
        self.assertEqual([e[0] for e in self.events if e[1] == "disable_gripper"], ["主左", "主右"])
        self.assertEqual([e[0] for e in self.events if e[1] == "enable"], ["主左", "从左", "主右", "从右"])
        self.assertFalse(any(e[1] in ("move_js", "move", "disable") for e in self.events))

    def test_left_motion_keeps_right_values_and_commands_independent(self) -> None:
        leader = self.arm("主左")
        read = leader.get_leader_joint_angles

        def moving() -> SimpleNamespace:
            leader.positions = [0.1 + 0.001 * len(self.runtime.dataset.frames)] * 7
            return read()

        with patch.object(leader, "get_leader_joint_angles", side_effect=moving):
            self.record()
        self.assertEqual(self.runtime.state["record_phase"], "saved")
        frames = self.runtime.dataset.frames
        self.assertEqual(len(frames), 3)
        moves = [e for e in self.events if e[1] == "move_js"]
        self.assertEqual([e[0] for e in moves], ["从左", "从右"] * 30)
        for i, frame in enumerate(frames):
            np.testing.assert_allclose(frame["action"][:8], [0.1 + 0.001 * i] * 7 + [0.015])
            np.testing.assert_allclose(frame["action"][8:], [0.4] * 7 + [0.05])
            np.testing.assert_allclose(frame["observation.state"][8:], [0.4] * 7 + [0.045])
            np.testing.assert_allclose(moves[2 * i * 10][2], frame["action"][:7])
            np.testing.assert_allclose(moves[2 * i * 10 + 1][2], frame["action"][8:15])
        report = self.report()
        self.assertEqual([p["leader"] for p in report["pairs"]], ["can1", "can3"])
        self.assertEqual(report["joint_names"], pairing.joint_keys("dual"))
        self.assertFalse(report["sensor_sync_verified"])
        verify_platform_dataset.verify_timing(self.runtime.dataset.root, 0, 3, 10)

    def test_right_feedback_failure_stops_both_before_another_left_command(self) -> None:
        leader = self.arm("主右")
        read = leader.get_leader_joint_angles
        with patch.object(
            leader,
            "get_leader_joint_angles",
            side_effect=lambda: None if self.runtime.dataset.frames else read(),
        ):
            self.record()
        self.assertEqual(self.runtime.state["record_phase"], "saved_review")
        self.assertEqual(len(self.runtime.dataset.frames), 1)
        self.assertEqual([e[0] for e in self.events if e[1] == "move_js"], ["从左", "从右"])
        self.assertIn("主右→从右", self.report()["termination_error"])

    def test_raw_targets_keep_left_and_right_pairs_independent(self) -> None:
        self.arm("主左").positions = [0.18] * 7
        self.arm("主右").positions = [0.32] * 7
        recorder.record_worker(self.runtime, [])
        self.assertEqual(self.runtime.state["record_phase"], "saved")
        frames = self.runtime.dataset.frames
        self.assertEqual(len(frames), 3)
        for frame in frames:
            np.testing.assert_allclose(frame["action"][:7], 0.18)
            np.testing.assert_allclose(frame["action"][8:15], 0.32)
            self.assertAlmostEqual(frame["action"][7], 0.015)
            self.assertAlmostEqual(frame["action"][15], 0.05)

    def test_right_first_frame_misalignment_sends_no_left_command(self) -> None:
        with patch.object(
            devices,
            "align_follower_before_recording",
            side_effect=lambda _: setattr(self.arm("主右"), "positions", [1.0] * 7),
        ):
            self.record()
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assertFalse(any(e[1] in ("move_js", "move", "add_frame") for e in self.events))
        self.assertIn("主右→从右", self.runtime.state["last_error"])

    def test_right_command_failure_does_not_save_partial_frame_or_retry(self) -> None:
        follower = self.arm("从右")
        move = follower.move_js

        def fail(target: list) -> None:
            if self.runtime.dataset.frames:
                raise OSError("模拟 CAN 发送失败")
            move(target)

        with patch.object(follower, "move_js", side_effect=fail) as send:
            self.record()
        self.assertEqual(send.call_count, 2)
        self.assertEqual(len(self.runtime.dataset.frames), 1)
        self.assertEqual(sum(e[0] == "从左" and e[1] == "move_js" for e in self.events), 2)
        self.assertEqual(self.runtime.state["record_phase"], "saved_review")
        self.assertIn("不重试指令", self.report()["termination_error"])
        self.assertIsNone(self.report()["control"]["samples"][-1]["frame_index"])

    def test_cancel_left_alignment_never_starts_right_or_records(self) -> None:
        self.arm("主左").positions = [0.4] * 7
        original = self.arm("从左").move_js

        def cancel(target: list) -> None:
            original(target)
            self.runtime.state["stop_requested"] = True

        with patch.object(self.arm("从左"), "move_js", side_effect=cancel):
            self.record()
        self.assertEqual(self.runtime.state["record_phase"], "idle")
        self.assertFalse(any(e[0] == "从右" and e[1] == "move_js" for e in self.events))
        self.assertEqual(self.runtime.state["episode_count"], 0)

    def test_all_alignments_finish_before_first_frame(self) -> None:
        self.arm("主左").positions = [0.2] * 7
        self.arm("主右").positions = [0.5] * 7
        self.record()
        self.assertEqual(self.runtime.state["record_phase"], "saved")
        first = self.events.index(("dataset", "add_frame"))
        moves = [e for e in self.events[:first] if e[1] == "move_js"]
        for role, value in (("从左", 0.2), ("从右", 0.5)):
            selected = [e for e in moves if e[0] == role]
            self.assertGreater(len(selected), 100)
            self.assertEqual(selected[-1][2], [value] * 7)
        self.assertEqual(len(self.runtime.dataset.frames), 3)

    def test_four_arm_home_requires_four_confirmed_targets_and_preserves_enable(self) -> None:
        targets = self.targets()
        stale = {ch: label for ch, label in targets.items() if label.endswith("左")}
        self.assertEqual(self.client.post("/api/pair/home", json={"targets": stale}).status_code, 409)
        self.assertEqual(self.events, [])
        self.assertEqual(self.client.post("/api/pair/home", json={"targets": targets}).status_code, 202)
        self.finish()
        self.assertEqual([e[0] for e in self.events if e[1] == "move_j"], ["从左", "从右", "主左", "主右"])
        self.assertTrue(all(info["arm"].enabled for info in self.runtime.channels.values()))
        self.assertIsNone(self.runtime.state["home_error"])

    def test_end_and_disable_cover_four_arms_and_retry_only_failed_one(self) -> None:
        targets = self.targets()
        for info in self.runtime.channels.values():
            info["arm"] = DisableArm(info["label"], self.events)
        result = self.client.post("/api/collection/end", json={"targets": targets})
        self.assertEqual(result.status_code, 200)
        self.assertEqual([e[0] for e in self.events if e[1] == "follower"], ["主左", "主右"])
        self.assertFalse(any(e[1] == "disable" for e in self.events))
        self.arm("从右").accept_disable = False
        self.assertEqual(self.client.post("/api/pair/disable", json={"targets": targets}).status_code, 409)
        result = self.runtime.state["close_result"]
        self.assertEqual(len(result["arms"]), 4)
        self.assertEqual([ch for ch, e in result["arms"].items() if e["state"] == "error"], ["can0"])
        self.events.clear()
        self.arm("从右").accept_disable = True
        self.assertEqual(
            self.client.post(
                "/api/close/retry", json={"targets": targets, "stage": "arm", "channel": "can0"}
            ).status_code,
            200,
        )
        self.assertEqual({e[0] for e in self.events if e[1] in ("disable", "disable_gripper")}, {"从右"})
        self.assertFalse(any(e[0] == "dataset" for e in self.events))

    def test_shutdown_leaves_both_leaders_enabled(self) -> None:
        recorder.do_disconnect(self.runtime, reopen_preview=False)
        self.assertEqual([e[0] for e in self.events if e[1] == "follower"], ["主左", "主右"])
        self.assertFalse(any(e[1] in ("disable", "disable_gripper") for e in self.events))

    def test_incompatible_eight_dimensional_dataset_rejected_before_enable(self) -> None:
        self.runtime.state["connected"] = False
        path = recorder.dataset_path(self.runtime) / "meta/info.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "fps": 10,
                    "features": recorder.make_features({"right": {"type": "rgb", "path": "/fake/right"}}),
                }
            )
        )
        ok, message = recorder.do_connect(self.runtime)
        self.assertFalse(ok)
        self.assertIn("续录字段不一致", message)
        self.assertEqual(self.events, [])

    def test_real_dual_dataset_video_roundtrip_and_resume(self) -> None:
        self.runtime.state["connected"] = False
        self.runtime.dataset = None
        with patch.object(camera_io, "RecordingCamera", side_effect=lambda cfg: FakeCamera(cfg, self.events)):
            for iteration in range(2):
                self.assertTrue(recorder.do_connect(self.runtime)[0])
                self.record()
                self.assertEqual(self.runtime.state["record_phase"], "saved")
                config.save_config(self.runtime, {"fake-right": "/fake/right"})
                recorder.do_disconnect(self.runtime, reopen_preview=False)
                root = recorder.dataset_path(self.runtime)
                loaded = LeRobotDataset("nero/dual_test", root=root, video_backend="pyav")
                self.assertEqual(loaded.num_episodes, iteration + 1)
                frame = loaded[loaded.num_frames - 1]
                np.testing.assert_allclose(frame["action"].numpy(), [0.1] * 7 + [0.015] + [0.4] * 7 + [0.05])
                self.assertEqual(tuple(frame["observation.images.right"].shape), (3, 480, 640))
                self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), iteration + 1)
                with patch(
                    "sys.argv",
                    [
                        "verify",
                        "--repo-id",
                        "nero/dual_test",
                        "--dataset-root",
                        str(self.runtime.dataset_root),
                        "--require-timing",
                        "--expect-mode",
                        "dual",
                    ],
                ):
                    verify_platform_dataset.main()
                if iteration == 0:
                    self.runtime = Runtime(
                        config_path=self.runtime.config_path, dataset_root=self.runtime.dataset_root
                    )
                    config.load_saved(self.runtime)
                    self.attach(self.runtime)
                    self.client = create_app(self.runtime).test_client()


if __name__ == "__main__":
    unittest.main()
