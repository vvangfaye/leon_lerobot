"""第三轮第一步：可控时钟和模拟设备验证采样节拍、诊断文件与续录。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import verify_platform_dataset
from nero_platform import camera_io, devices, recorder, timing
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair


class TimingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-timing-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "dataset"
        self.events = []
        self.clock = Clock(self.events)
        self.runtime = Runtime(dataset_root=Path(directory.name))
        attach_fake_pair(self.runtime, self.events)
        self.runtime.state.update(connected=True, fps=30, episode_sec=10)
        self.runtime.dataset = MemoryDataset(self.events)
        self.runtime.dataset.root = self.root
        self.camera = FakeCamera(events=self.events)
        self.camera.frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.runtime.collection_cams["right"] = self.camera
        for guard in (
            patch.object(recorder, "time", self.clock),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(recorder.traceback, "print_exc"),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def record(self) -> dict:
        recorder.record_worker(self.runtime, ["right"])
        self.assertIsNone(self.runtime.state["writer_error"])
        self.assertEqual(
            self.runtime.state["record_phase"],
            "saved_review" if self.runtime.state["last_episode"]["needs_review"] else "saved",
        )
        return self.runtime.state["last_episode"]["timing"]

    def report(self, index: int = 0) -> dict:
        return json.loads(timing.report_path(self.root, index).read_text(encoding="utf-8"))

    def test_processing_time_is_subtracted_and_ten_seconds_yields_300_frames(self) -> None:
        read, add = self.camera.read_latest, self.runtime.dataset.add_frame

        def read_with_delay():
            self.clock.now += 0.012
            return read()

        def add_with_delay(frame: dict) -> None:
            self.clock.now += 0.008
            add(frame)

        with (
            patch.object(self.camera, "read_latest", side_effect=read_with_delay),
            patch.object(self.runtime.dataset, "add_frame", side_effect=add_with_delay),
        ):
            summary = self.record()
        self.assertEqual(summary["frames"], 300)
        self.assertEqual(summary["effective_fps"], 30)
        self.assertEqual(summary["elapsed_seconds"], 10)
        self.assertEqual(summary["mean_processing_ms"], 20)
        self.assertEqual(summary["skipped_slots"], 0)
        self.assertEqual(summary["warnings"], [])
        starts = [sample["started_seconds"] for sample in self.report()["samples"]]
        np.testing.assert_allclose(starts, np.arange(300) / 30, atol=1e-9)
        self.assertEqual(verify_platform_dataset.verify_timing(self.root, 0, 300, 30), summary)

    def test_control_stays_100_hz_with_10_30_or_60_fps_data_sampling(self) -> None:
        for episode, fps in enumerate((10, 30, 60)):
            with self.subTest(fps=fps):
                self.runtime.state.update(fps=fps, episode_sec=1)
                self.runtime.dataset.frames.clear()
                self.events.clear()
                summary = self.record()
                report = self.report(episode)
                self.assertEqual(summary["frames"], fps)
                self.assertEqual(summary["effective_fps"], fps)
                controls = report["control"]
                self.assertEqual(controls["summary"]["cycles"], 100)
                self.assertEqual(controls["summary"]["effective_hz"], 100)
                self.assertEqual(controls["summary"]["skipped_slots"], 0)
                self.assertEqual(sum(e[1] == "move_js" for e in self.events), 100)
                np.testing.assert_allclose(
                    [c["started_seconds"] for c in controls["samples"]], np.arange(100) / 100, atol=1e-9
                )
                self.assertLess(max(s["control_age_ms"] for s in report["samples"]), 10)
                verify_platform_dataset.verify_timing(self.root, episode, fps, fps)

    def test_all_control_frequency_choices_keep_data_at_30_fps(self) -> None:
        for episode, setting in enumerate(("capture", 50, 100, 150, 200)):
            with self.subTest(control_frequency=setting):
                expected = 30 if setting == "capture" else setting
                self.runtime.state.update(control_frequency=setting, fps=30, episode_sec=1)
                self.runtime.dataset.frames.clear()
                self.events.clear()
                summary = self.record()
                report = self.report(episode)
                self.assertEqual(summary["frames"], 30)
                self.assertEqual(summary["effective_fps"], 30)
                self.assertEqual(report["control_frequency"], setting)
                self.assertEqual(self.runtime.state["last_episode"]["control_frequency"], setting)
                control = report["control"]
                self.assertEqual(control["summary"]["cycles"], expected)
                self.assertEqual(control["summary"]["effective_hz"], expected)
                self.assertEqual(sum(e[1] == "move_js" for e in self.events), expected)
                np.testing.assert_allclose(
                    [c["started_seconds"] for c in control["samples"]],
                    np.arange(expected) / expected,
                    atol=1e-9,
                )
                self.assertEqual(control["summary"]["skipped_slots"], 0)
                verify_platform_dataset.verify_timing(self.root, episode, 30, 30)

    def test_capture_choice_issues_one_command_per_sampling_attempt_after_slow_write(self) -> None:
        self.runtime.state.update(control_frequency="capture", fps=30, episode_sec=0.3)
        add = self.runtime.dataset.add_frame

        def slow_write(frame: dict) -> None:
            add(frame)
            self.clock.now += 0.11

        with patch.object(self.runtime.dataset, "add_frame", side_effect=slow_write):
            self.record()
        report = self.report()
        self.assertEqual(len(self.runtime.dataset.frames), 3)
        self.assertEqual(sum(e[1] == "move_js" for e in self.events), 3)
        self.assertEqual(report["control"]["scheduler"], "capture_cadence")
        self.assertEqual([s["control_index"] for s in report["samples"]], [0, 1, 2])
        self.assertTrue(report["control"]["summary"]["warnings"])
        verify_platform_dataset.verify_timing(self.root, 0, 3, 30)

    def test_sample_links_to_latest_sent_action_not_next_leader_target(self) -> None:
        self.runtime.state.update(episode_sec=1)
        leader = self.runtime.channels["can0"]["arm"]
        with patch.object(
            leader,
            "get_leader_joint_angles",
            side_effect=lambda: leader.feedback([0.1 * i + self.clock.now / 10 for i in range(7)]),
        ):
            self.record()
        report = self.report()
        commands = [e[2] for e in self.events if e[1] == "move_js"]
        indices = []
        for frame, sample in zip(self.runtime.dataset.frames, report["samples"], strict=True):
            index = sample["control_index"]
            indices.append(index)
            cycle = report["control"]["samples"][index]
            self.assertEqual(sample["robots"], cycle["robots"])
            np.testing.assert_allclose(frame["action"][:7], commands[index])
            np.testing.assert_allclose(frame["action"], cycle["robots"]["single"]["commanded_action"])
            self.assertLessEqual(cycle["completed_seconds"], sample["started_seconds"])
        self.assertEqual(indices[:6], [0, 3, 6, 10, 13, 16])
        # 篡改映射不能继续通过质检。
        report["samples"][1]["control_index"] = 2
        timing.report_path(self.root, 0).write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "控制周期"):
            verify_platform_dataset.verify_timing(self.root, 0, 30, 30)

    def test_slow_camera_exposes_control_gap_without_burst_replay(self) -> None:
        self.runtime.state.update(episode_sec=0.3)
        read = self.camera.read_latest
        calls = 0

        def slow_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                self.clock.now += 0.11
            return read()

        with patch.object(self.camera, "read_latest", side_effect=slow_once):
            self.record()
        control = self.report()["control"]
        self.assertEqual([c["slot"] for c in control["samples"]][:3], [0, 11, 12])
        self.assertEqual(control["summary"]["skipped_slots"], 10)
        self.assertEqual(control["summary"]["max_interval_ms"], 110)
        self.assertTrue(control["summary"]["warnings"])
        self.assertTrue(self.runtime.state["last_episode"]["needs_review"])
        self.assertEqual(len({c["started_seconds"] for c in control["samples"]}), len(control["samples"]))

    def test_stop_between_data_frames_stops_control_before_saving(self) -> None:
        self.runtime.state.update(episode_sec=1)
        sleep = self.clock.sleep
        save = self.runtime.dataset.save_episode

        def stop_before_fifth_control_tick(duration: float) -> None:
            sleep(duration)
            if self.runtime.state["record_started_at"] is not None and self.clock.now >= 0.049:
                recorder.request_stop(self.runtime)

        def slow_save() -> None:
            count = sum(e[1] == "move_js" for e in self.events)
            self.clock.now += 5
            save()
            self.assertEqual(sum(e[1] == "move_js" for e in self.events), count)

        with (
            patch.object(self.clock, "sleep", side_effect=stop_before_fifth_control_tick),
            patch.object(self.runtime.dataset, "save_episode", side_effect=slow_save),
        ):
            self.record()
        self.assertEqual(len(self.runtime.dataset.frames), 2)
        self.assertEqual(sum(e[1] == "move_js" for e in self.events), 4)
        self.assertLess(self.runtime.state["last_episode"]["elapsed_seconds"], 0.1)

    def test_long_frame_skips_elapsed_slots_without_replaying_them(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=1)
        read = self.camera.read_latest
        attempts = 0

        def stall_once():
            nonlocal attempts
            self.clock.now += 0.35 if attempts == 0 else 0.01
            attempts += 1
            return read()

        with patch.object(self.camera, "read_latest", side_effect=stall_once):
            summary = self.record()
        samples = self.report()["samples"]
        self.assertEqual([s["slot"] for s in samples], [0, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(summary["skipped_slots"], 2)
        self.assertEqual(summary["frames"], 8)
        self.assertEqual(summary["over_budget_samples"], 1)
        self.assertEqual(summary["max_interval_ms"], 350)
        self.assertEqual(summary["nominal_playback_seconds"], 0.8)
        self.assertTrue(summary["warnings"])

    def test_tail_overrun_is_counted_and_not_hidden_in_nominal_fps(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.3)
        read = self.camera.read_latest

        def slow():
            self.clock.now += 0.35
            return read()

        with patch.object(self.camera, "read_latest", side_effect=slow):
            summary = self.record()
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["skipped_slots"], 2)
        self.assertIsNone(summary["effective_fps"])
        self.assertEqual(summary["max_processing_ms"], 350)

    def test_failed_sample_has_no_frame_index_and_does_not_inflate_effective_fps(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.5)
        frame = self.camera.frame
        with patch.object(
            self.camera, "read_latest", side_effect=[frame, OSError("读图失败"), frame, frame, frame]
        ):
            summary = self.record()
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["failed_samples"], 1)
        self.assertEqual(summary["attempts"], 5)
        self.assertEqual(summary["effective_fps"], 7.5)
        self.assertEqual(summary["max_interval_ms"], 200)
        self.assertEqual([s["frame_index"] for s in self.report()["samples"]], [0, None, 1, 2, 3])
        self.assertEqual(verify_platform_dataset.verify_timing(self.root, 0, 4, 10), summary)

    def test_save_time_is_excluded_from_timing_and_summary(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.5)
        save = self.runtime.dataset.save_episode

        def slow_save():
            self.clock.now += 20
            save()

        with patch.object(self.runtime.dataset, "save_episode", side_effect=slow_save):
            summary = self.record()
        self.assertEqual(summary["elapsed_seconds"], 0.5)
        self.assertEqual(summary["effective_fps"], 10)
        self.assertAlmostEqual(self.clock.now, 20.51)  # 额外 10 ms 准备不计入录制时长。
        self.assertEqual(recorder.recording_status(self.runtime)["timing"], summary)

    def test_raw_leader_targets_are_sent_and_recorded_without_legacy_filter(self) -> None:
        self.runtime.state.update(episode_sec=0.1)
        leader = self.runtime.channels["can0"]["arm"]
        start = [0.1 * i for i in range(7)]
        target = [value + 0.08 for value in start]
        with patch.object(leader, "get_leader_joint_angles", side_effect=lambda: leader.feedback(target[:])):
            self.record()
        commands = [e[2] for e in self.events if e[1] == "move_js"]
        self.assertEqual(len(commands), 10)
        for command in commands:
            np.testing.assert_allclose(command, target)
        for frame in self.runtime.dataset.frames:
            np.testing.assert_allclose(frame["action"][:7], target)
            np.testing.assert_allclose(frame["observation.state"][:7], start)
        report = self.report()
        self.assertEqual(report["action_source"], "commanded_joint_targets_and_leader_gripper")
        self.assertNotIn("follow_smoothing", report)

    def test_first_target_still_requires_joint_alignment(self) -> None:
        self.runtime.state.update(episode_sec=0.1)
        leader = self.runtime.channels["can0"]["arm"]
        with patch.object(leader, "get_leader_joint_angles", side_effect=lambda: leader.feedback([1.0] * 7)):
            recorder.record_worker(self.runtime, ["right"])
        self.assertFalse(any(event[1] == "move_js" for event in self.events))
        self.assertEqual(self.runtime.dataset.frames, [])
        self.assertIn("未对齐", self.runtime.state["record_message"])

    def test_stop_during_wait_does_not_send_an_extra_frame_or_wait_a_full_low_fps_period(self) -> None:
        self.runtime.state.update(fps=1, episode_sec=10)
        sleep = self.clock.sleep

        def stop_during_sleep(duration: float) -> None:
            if self.runtime.dataset.frames:
                recorder.request_stop(self.runtime)
            sleep(duration)

        with patch.object(self.clock, "sleep", side_effect=stop_during_sleep):
            summary = self.record()
        self.assertEqual(summary["frames"], 1)
        self.assertLessEqual(summary["elapsed_seconds"], 0.05)
        self.assertEqual(sum(e[1] == "move_js" for e in self.events), 1)

    def test_report_failure_preserves_saved_episode_and_allows_next_episode(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.2)
        with patch.object(timing, "save_report", side_effect=OSError("报告目录不可写")):
            self.record()
        self.assertEqual(self.runtime.state["episode_count"], 1)
        self.assertIsNone(self.runtime.state["writer_error"])
        self.assertIn("数据已保存", self.runtime.state["timing_report_error"])
        self.record()
        self.assertIsNone(self.runtime.state["timing_report_error"])
        self.assertEqual(self.runtime.state["episode_count"], 2)
        self.assertEqual(self.report(1)["episode_index"], 1)

    def test_failed_dataset_save_never_creates_success_report(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.2)
        with patch.object(self.runtime.dataset, "save_episode", side_effect=OSError("磁盘错误")):
            recorder.record_worker(self.runtime, ["right"])
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assertFalse(timing.report_path(self.root, 0).exists())
        self.assertIn("磁盘错误", self.runtime.state["writer_error"])

    def test_timing_reports_append_without_overwriting_existing_episode(self) -> None:
        self.runtime.state.update(fps=10, episode_sec=0.2)
        self.record()
        first = timing.report_path(self.root, 0).read_bytes()
        self.record()
        self.assertEqual(timing.report_path(self.root, 0).read_bytes(), first)
        self.assertEqual(self.report(1)["summary"]["frames"], 2)
        with self.assertRaises(FileExistsError):
            timing.save_report(self.root, 0, {"wrong": True})
        self.assertEqual(timing.report_path(self.root, 0).read_bytes(), first)
        self.assertEqual(len(list(timing.report_path(self.root, 0).parent.iterdir())), 2)

    def test_verifier_distinguishes_missing_corrupt_and_mismatched_timing(self) -> None:
        self.assertIsNone(verify_platform_dataset.verify_timing(self.root, 0, 2, 10))
        self.runtime.state.update(fps=10, episode_sec=0.2)
        self.record()
        with self.assertRaisesRegex(ValueError, "不一致"):
            verify_platform_dataset.verify_timing(self.root, 0, 3, 10)
        report = self.report()
        report["samples"][1]["started_seconds"] = -1
        timing.report_path(self.root, 0).write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "采样时间"):
            verify_platform_dataset.verify_timing(self.root, 0, 2, 10)


if __name__ == "__main__":
    unittest.main()
