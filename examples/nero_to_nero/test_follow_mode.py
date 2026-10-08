"""遥操作模式：只用模拟硬件验证 J 跟随、动作保存及退出时的保持请求。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from nero_platform import camera_io, devices, recorder, timing
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeArm, FakeCamera, FakeGripper, MemoryDataset, attach_fake_pair


class PlannedArm(FakeArm):
    def move_j(self, angles: list[float]) -> None:
        # 模拟未追上目标：实测位置保持独立，不能以最后指令冒充反馈。
        self.events.append((self.name, "move_j", list(angles)))

    def electronic_emergency_stop(self) -> None:
        self.events.append((self.name, "emergency_stop"))


class FollowModeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-follow-mode-")
        self.addCleanup(directory.cleanup)
        self.events = []
        self.clock = Clock(self.events)
        self.runtime = Runtime(dataset_root=Path(directory.name))
        attach_fake_pair(self.runtime, self.events)
        self.follower = PlannedArm("从左", self.events)
        self.runtime.channels["can1"]["arm"] = self.follower
        self.runtime.state.update(connected=True, fps=30, episode_sec=0.1, follow_motion_mode="j")
        self.dataset = MemoryDataset(self.events)
        self.dataset.root = Path(directory.name) / "dataset"
        self.runtime.dataset = self.dataset
        camera = FakeCamera(events=self.events)
        camera.frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.runtime.collection_cams["right"] = camera
        self.current = [0.1 * i for i in range(7)]
        self.leader = self.runtime.channels["can0"]["arm"]
        for guard in (
            patch.object(recorder, "time", self.clock),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(recorder.traceback, "print_exc"),
            patch.object(
                self.leader,
                "get_leader_joint_angles",
                side_effect=lambda: self.leader.feedback([v + 0.08 for v in self.current]),
            ),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def record(self) -> None:
        recorder.record_worker(self.runtime, ["right"])

    def commands(self) -> list:
        return [event for event in self.events if event[1] in ("move_j", "move_js")]

    def report(self) -> dict:
        return json.loads(timing.report_path(self.dataset.root, 0).read_text())

    def assert_held_before_save(self) -> None:
        commands = self.commands()
        np.testing.assert_allclose(commands[-1][2], self.current)
        self.assertLess(
            max(i for i, event in enumerate(self.events) if event[1] == "move_j"),
            self.events.index(("dataset", "save_episode")),
        )
        self.assertTrue(self.follower.enabled)
        self.assertIsNone(self.runtime.state["recovery_required"])

    def test_j_sends_raw_targets_and_saves_actions_then_holds_actual_position(self) -> None:
        self.record()
        commands = self.commands()
        self.assertEqual([event[1] for event in commands], ["move_j"] * 11)
        self.assertEqual(len(self.dataset.frames), 3)
        for i in range(10):
            step = 0.08
            np.testing.assert_allclose(commands[i][2], np.array(self.current) + step)
        for i, command_index in enumerate((0, 3, 6)):
            np.testing.assert_allclose(self.dataset.frames[i]["action"][:7], commands[command_index][2])
            self.assertAlmostEqual(self.dataset.frames[i]["action"][7], 0.025)
        self.assert_held_before_save()
        report = self.report()
        self.assertEqual(report["summary"]["effective_fps"], 30)
        self.assertEqual(report["follow_motion_mode"], "j")
        self.assertNotIn("follow_smoothing", report)
        self.assertIn("已请求", report["follow_stop_message"])
        self.assertEqual(self.runtime.state["last_episode"]["follow_motion_mode"], "j")
        self.assertIsNone(report["termination_error"])

    def test_j_frequency_choices_route_all_commands_and_stop_before_saving(self) -> None:
        for frequency, expected_cycles in (("capture", 3), (50, 5), (100, 10), (150, 15), (200, 20)):
            with self.subTest(control_frequency=frequency):
                self.runtime.state["control_frequency"] = frequency
                self.dataset.frames.clear()
                self.events.clear()
                self.record()
                self.assertEqual(len(self.dataset.frames), 3)
                self.assertEqual([e[1] for e in self.commands()], ["move_j"] * (expected_cycles + 1))
                self.assertEqual(self.runtime.state["control_timing"]["cycles"], expected_cycles)
                self.assert_held_before_save()

    def test_js_has_no_extra_hold_command(self) -> None:
        self.runtime.state["follow_motion_mode"] = "js"
        self.record()
        self.assertEqual([event[1] for event in self.commands()], ["move_js"] * 10)
        self.assertEqual(len(self.dataset.frames), 3)
        self.assertEqual(self.runtime.state["follow_stop_message"], "")

    def test_dual_pair_records_both_targets_and_holds_each_at_its_own_position(self) -> None:
        self.runtime.state["pair_mode"] = "dual"
        leader_right, follower_right = PlannedArm("主右", self.events), PlannedArm("从右", self.events)
        for channel, arm in (("can2", leader_right), ("can3", follower_right)):
            self.runtime.channels[channel].update(
                arm=arm, gripper=FakeGripper(arm.name, self.events), label=arm.name, detected=True
            )
        right_current = [0.4] * 7
        with (
            patch.object(
                leader_right, "get_leader_joint_angles", side_effect=lambda: leader_right.feedback([0.46] * 7)
            ),
            patch.object(
                follower_right, "get_joint_angles", side_effect=lambda: follower_right.feedback(right_current)
            ),
        ):
            self.record()
        left_commands = [event[2] for event in self.commands() if event[0] == "从左"]
        right_commands = [event[2] for event in self.commands() if event[0] == "从右"]
        self.assertEqual(len(left_commands), 11)
        self.assertEqual(len(right_commands), 11)
        for i, frame in enumerate(self.dataset.frames):
            command_index = (0, 3, 6)[i]
            self.assertEqual(frame["action"].shape, (16,))
            np.testing.assert_allclose(frame["action"][:7], left_commands[command_index])
            np.testing.assert_allclose(frame["action"][8:15], right_commands[command_index])
        np.testing.assert_allclose(left_commands[-1], self.current)
        np.testing.assert_allclose(right_commands[-1], right_current)
        self.assertEqual(self.events[-1], ("dataset", "save_episode"))
        self.assertIsNone(self.runtime.state["recovery_required"])

    def test_manual_stop_sends_only_hold_after_last_saved_frame(self) -> None:
        self.runtime.state.update(fps=1, episode_sec=10)
        sleep = self.clock.sleep

        def stop_during_sleep(duration: float) -> None:
            if self.dataset.frames:
                recorder.request_stop(self.runtime)
            sleep(duration)

        with patch.object(self.clock, "sleep", side_effect=stop_during_sleep):
            self.record()
        self.assertEqual(len(self.dataset.frames), 1)
        self.assertEqual(len(self.commands()), 2)
        self.assertLessEqual(self.report()["summary"]["elapsed_seconds"], 0.05)
        self.assert_held_before_save()

    def test_hold_delay_is_not_counted_as_recording_time(self) -> None:
        send = self.follower.move_j

        def delayed_hold(angles: list[float]) -> None:
            if angles == self.current:
                self.clock.now += 1
            send(angles)

        with patch.object(self.follower, "move_j", side_effect=delayed_hold):
            self.record()
        self.assertAlmostEqual(self.report()["summary"]["elapsed_seconds"], 0.1)
        self.assert_held_before_save()

    def test_cancel_before_start_and_misalignment_do_not_send_hold_or_follow(self) -> None:
        self.runtime.state["stop_requested"] = True
        self.record()
        self.assertEqual(self.commands(), [])
        self.runtime.state["stop_requested"] = False
        with patch.object(
            self.leader, "get_leader_joint_angles", side_effect=lambda: self.leader.feedback([2] * 7)
        ):
            self.record()
        self.assertEqual(self.commands(), [])
        self.assertEqual(self.dataset.frames, [])
        self.assertIn("未对齐", self.runtime.state["record_message"])

    def test_leader_feedback_failure_holds_and_saves_valid_frames(self) -> None:
        def read_leader() -> SimpleNamespace:
            if self.dataset.frames:
                raise OSError("主臂反馈中断")
            return self.leader.feedback([v + 0.08 for v in self.current])

        with patch.object(self.leader, "get_leader_joint_angles", side_effect=read_leader):
            self.record()
        self.assertEqual(len(self.dataset.frames), 1)
        self.assert_held_before_save()
        self.assertEqual(self.runtime.state["record_phase"], "saved_review")
        self.assertIn("主臂反馈中断", self.report()["termination_error"])

    def test_writer_failure_still_holds_and_does_not_claim_save(self) -> None:
        with patch.object(self.dataset, "add_frame", side_effect=OSError("写入失败")):
            self.record()
        self.assertEqual(len(self.commands()), 2)
        np.testing.assert_allclose(self.commands()[-1][2], self.current)
        self.assertNotIn(("dataset", "save_episode"), self.events)
        self.assertIn("写入失败", self.runtime.state["writer_error"])
        self.assertEqual(self.runtime.state["episode_count"], 0)

    def test_partial_command_failure_also_holds_attempted_follower(self) -> None:
        send = self.follower.move_j

        def partial_send(angles: list[float]) -> None:
            send(angles)
            if len(self.commands()) == 1:
                raise OSError("发送中途失败")

        with patch.object(self.follower, "move_j", side_effect=partial_send):
            self.record()
        self.assertEqual(len(self.commands()), 2)
        np.testing.assert_allclose(self.commands()[-1][2], self.current)
        self.assertEqual(self.dataset.frames, [])
        self.assertIn("发送中途失败", self.runtime.state["record_message"])

    def test_hold_failure_locks_motion_but_preserves_valid_recording(self) -> None:
        send = self.follower.move_j

        def reject_hold(angles: list[float]) -> None:
            if angles == self.current:
                raise OSError("保持失败")
            send(angles)

        with patch.object(self.follower, "move_j", side_effect=reject_hold):
            self.record()
        self.assertEqual(len(self.dataset.frames), 3)
        self.assertIn(("从左", "emergency_stop"), self.events)
        self.assertEqual(self.runtime.state["record_phase"], "saved_review")
        self.assertIn("保持失败", self.report()["termination_error"])
        self.assertIn("运动操作已锁定", self.runtime.state["recovery_required"])
        self.assertNotIn("可重新准备下一集", self.runtime.state["record_message"])

    def test_invalid_stop_feedback_never_becomes_a_hold_target(self) -> None:
        for sample in (
            None,
            SimpleNamespace(timestamp=1, msg=self.current),  # 倒退
            SimpleNamespace(timestamp=2, msg=self.current),  # 0.5 秒未更新
            SimpleNamespace(timestamp=float("nan"), msg=self.current),
            SimpleNamespace(timestamp=3, msg=[float("nan")] * 7),
            SimpleNamespace(timestamp=3, msg=[0.0] * 6),
        ):
            with (
                self.subTest(sample=sample),
                patch.object(self.follower, "get_joint_angles", return_value=sample),
            ):
                self.events.clear()
                error = devices.hold_planned_followers(self.runtime, {"can1": (2, 0, True)}, 0.5)
                self.assertIsNotNone(error)
                self.assertEqual(self.commands(), [])
                self.assertIn(("从左", "emergency_stop"), self.events)

    def test_recent_cached_feedback_can_hold_without_waiting_for_a_new_tick(self) -> None:
        sample = SimpleNamespace(timestamp=2, msg=self.current)
        with patch.object(self.follower, "get_joint_angles", return_value=sample):
            error = devices.hold_planned_followers(self.runtime, {"can1": (2, 0, True)}, 0.1)
        self.assertIsNone(error)
        self.assertEqual(self.commands(), [("从左", "move_j", self.current)])

    def test_one_failed_stop_does_not_skip_the_other_follower(self) -> None:
        right = PlannedArm("从右", self.events)
        self.runtime.channels["can2"]["arm"] = right
        right_positions = [0.4] * 7
        with (
            patch.object(self.follower, "get_joint_angles", return_value=None),
            patch.object(self.follower, "electronic_emergency_stop", side_effect=OSError("急停发送失败")),
            patch.object(
                right, "get_joint_angles", return_value=SimpleNamespace(timestamp=3, msg=right_positions)
            ),
        ):
            error = devices.hold_planned_followers(
                self.runtime, {"can1": (2, 0, True), "can2": (2, 0, True)}, 0.1
            )
        self.assertIn("急停指令发送失败", error)
        self.assertEqual(self.commands(), [("从右", "move_j", right_positions)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
