"""注入反馈丢失/冻结及重复图像；不连接实际硬件。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import verify_platform_dataset
from nero_platform import recorder, sampling, timing
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


class SamplingTests(unittest.TestCase):
    def make_runtime(self) -> tuple[Runtime, Clock, list]:
        directory = tempfile.TemporaryDirectory(prefix="nero-quality-")
        self.addCleanup(directory.cleanup)
        runtime = Runtime(dataset_root=Path(directory.name))
        events = []
        attach_fake_pair(runtime, events)
        runtime.state.update(connected=True, fps=10, episode_sec=1)
        runtime.dataset = MemoryDataset(events)
        runtime.dataset.root = Path(directory.name)
        return runtime, Clock(events), events

    def record(self, runtime: Runtime, clock: Clock, cameras: list[str] | None = None) -> None:
        with patch.object(recorder, "time", clock), patch.object(recorder.traceback, "print_exc"):
            recorder.record_worker(runtime, cameras or [])

    def test_bad_feedback_stops_before_command_and_saves_only_prior_valid_frames(self) -> None:
        for channel, kind, method, malformed in (
            ("can0", "arm", "get_leader_joint_angles", None),
            ("can1", "arm", "get_joint_angles", SimpleNamespace(msg=[0] * 6, timestamp=99)),
            ("can0", "gripper", "get_gripper_ctrl_states", None),
            (
                "can1",
                "gripper",
                "get_gripper_status",
                SimpleNamespace(msg=SimpleNamespace(value=float("nan")), timestamp=99),
            ),
            ("can0", "arm", "get_leader_joint_angles", SimpleNamespace(msg=[0] * 7, timestamp=0.5)),
        ):
            with self.subTest(channel=channel, method=method):
                runtime, clock, events = self.make_runtime()
                obj = runtime.channels[channel][kind]
                original = getattr(obj, method)
                with patch.object(
                    obj,
                    method,
                    side_effect=lambda malformed=malformed, runtime=runtime, original=original: (
                        malformed if runtime.dataset.frames else original()
                    ),
                ):
                    self.record(runtime, clock)
                self.assertEqual(len(runtime.dataset.frames), 1)
                self.assertEqual(sum(e[1] == "move_js" for e in events), 1)
                self.assertEqual(runtime.state["episode_count"], 1)
                self.assertEqual(runtime.state["record_phase"], "saved_review")
                self.assertIsNone(runtime.state["last_error"])
                self.assertIn("可重新准备下一集", runtime.state["record_message"])
                self.assertIsNone(runtime.state["writer_error"])
                self.assertTrue(runtime.state["last_episode"]["needs_review"])
                report = json.loads(timing.report_path(runtime.dataset.root, 0).read_text())
                self.assertTrue(report["termination_error"])
                self.assertIsNone(report["control"]["samples"][-1]["frame_index"])
                self.assertFalse(any(e[1] in ("disable", "disable_gripper") for e in events))

    def test_missing_or_frozen_initial_feedback_never_sends_a_command(self) -> None:
        for value in (None, SimpleNamespace(msg=[0.1] * 7, timestamp=10.0)):
            runtime, clock, events = self.make_runtime()
            with patch.object(runtime.channels["can0"]["arm"], "get_leader_joint_angles", return_value=value):
                self.record(runtime, clock)
            self.assertEqual(runtime.state["episode_count"], 0)
            self.assertEqual(runtime.state["record_phase"], "error")
            self.assertFalse(any(e[1] in ("move_js", "move", "add_frame", "save_episode") for e in events))
            self.assertLessEqual(clock.now, sampling.FEEDBACK_TIMEOUT_SECONDS + 0.02)

    def test_feedback_freezing_mid_episode_stops_after_timeout(self) -> None:
        runtime, clock, events = self.make_runtime()
        gripper = runtime.channels["can0"]["gripper"]
        original = gripper.get_gripper_ctrl_states
        frozen = None

        def read():
            nonlocal frozen
            if runtime.dataset.frames:
                if frozen is None:
                    frozen = original()
                return frozen
            return original()

        with patch.object(gripper, "get_gripper_ctrl_states", side_effect=read):
            self.record(runtime, clock)
        self.assertIn("主臂夹爪反馈超过", runtime.state["last_episode"]["termination_error"])
        self.assertEqual(runtime.state["record_phase"], "saved_review")
        self.assertIsNone(runtime.state["last_error"])
        self.assertLess(len(runtime.dataset.frames), 10)
        commands = sum(e[1] == "move_js" for e in events)
        self.assertEqual(commands, runtime.state["last_episode"]["control_timing"]["cycles"])
        self.assertGreater(commands, len(runtime.dataset.frames))

    def test_recorded_gripper_action_matches_clamped_command(self) -> None:
        runtime, clock, events = self.make_runtime()
        gripper = runtime.channels["can0"]["gripper"]
        original = gripper.get_gripper_ctrl_states

        def read():
            sample = original()
            sample.msg.value = 0.09
            return sample

        with patch.object(gripper, "get_gripper_ctrl_states", side_effect=read):
            self.record(runtime, clock)
        self.assertTrue(all(e[2]["value"] == 0.07 for e in events if e[1] == "move"))
        self.assertTrue(all(abs(float(f["action"][7]) - 0.07) < 1e-6 for f in runtime.dataset.frames))

    def test_duplicate_camera_frames_are_counted_per_camera_and_written_to_report(self) -> None:
        runtime, clock, _ = self.make_runtime()
        for role in ("left", "right"):
            runtime.collection_cams[role] = FakeCamera()
        left = runtime.collection_cams["left"]
        with patch.object(left, "read_latest_with_timestamp", return_value=(left.frame, 50.0, 100.0)):
            self.record(runtime, clock, ["left", "right"])
        stats = runtime.state["last_episode"]["timing"]["cameras"]
        self.assertEqual(stats["left"]["repeated_frames"], 9)
        self.assertEqual(stats["right"]["repeated_frames"], 0)
        self.assertEqual(stats["left"]["max_age_ms"], 100)
        self.assertTrue(runtime.state["last_episode"]["needs_review"])
        report = json.loads(timing.report_path(runtime.dataset.root, 0).read_text())
        self.assertEqual(sum(s["cameras"]["left"]["repeated"] for s in report["samples"]), 9)
        self.assertFalse(report["sensor_sync_verified"])
        verify_platform_dataset.verify_timing(runtime.dataset.root, 0, 10, 10)
        report["summary"]["cameras"]["left"]["repeated_frames"] = 0
        timing.report_path(runtime.dataset.root, 0).write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "统计与采样明细不一致"):
            verify_platform_dataset.verify_timing(runtime.dataset.root, 0, 10, 10)

    def test_opencv_snapshot_has_matching_timestamp_and_retains_stale_frame_check(self) -> None:
        camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0))
        camera.thread = SimpleNamespace(is_alive=lambda: True)
        camera.frame_lock = threading.Lock()
        camera.latest_frame = np.zeros((2, 2, 3), np.uint8)
        camera.latest_timestamp = 100.0
        with (
            patch("lerobot.cameras.opencv.camera_opencv.time.perf_counter", return_value=100.025),
            patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True),
        ):
            frame, stamp, age = camera.read_latest_with_timestamp()
            self.assertIs(frame, camera.latest_frame)
            self.assertEqual(stamp, 100)
            self.assertAlmostEqual(age, 25)
            self.assertIs(camera.read_latest(), frame)
        with (
            patch("lerobot.cameras.opencv.camera_opencv.time.perf_counter", return_value=100.6),
            self.assertRaises(TimeoutError),
        ):
            camera.read_latest_with_timestamp()
        camera.thread = None


if __name__ == "__main__":
    unittest.main()
