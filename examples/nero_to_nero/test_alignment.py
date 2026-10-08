"""每集开录前对齐门禁：模拟位置变化、陈旧反馈和普通续录，不操作真机。"""

import threading
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform import camera_io, cameras, devices, hardware_status, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeArm, MemoryDataset, attach_fake_pair


class PositionedArm(FakeArm):
    def __init__(self, name: str, events: list):
        super().__init__(name, events)
        self.positions = [0.0] * 7
        self.command = None
        self.follow_commands = True
        self.tracking_error = 0.0

    def get_joint_angles(self) -> SimpleNamespace:
        self.events.append((self.name, "read_follower"))
        return self.feedback(self.positions[:])

    def get_leader_joint_angles(self) -> SimpleNamespace:
        self.events.append((self.name, "read_leader"))
        return self.feedback((self.command if self.command is not None else self.positions)[:])

    def move_js(self, angles: list) -> None:
        super().move_js(angles)
        if self.follow_commands:
            self.positions = [v - self.tracking_error for v in angles]

    def electronic_emergency_stop(self) -> None:
        self.events.append((self.name, "emergency_stop"))


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Runtime()
        self.events = []
        attach_fake_pair(self.runtime, self.events)
        self.leader = PositionedArm("主左", self.events)
        self.follower = PositionedArm("从左", self.events)
        self.runtime.channels["can0"]["arm"] = self.leader
        self.runtime.channels["can1"]["arm"] = self.follower
        self.runtime.state.update(connected=True, fps=10, episode_sec=0.25)
        self.runtime.dataset = MemoryDataset(self.events)
        self.client = create_app(self.runtime).test_client()
        self.clock = Clock(self.events)
        self.releases = []
        for guard in (
            patch.object(devices, "time", self.clock),
            patch.object(recorder, "time", self.clock),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(recorder.traceback, "print_exc"),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.addCleanup(self.finish)

    def finish(self) -> None:
        for release in self.releases:
            release.set()
        if self.runtime.record_thread is not None:
            self.runtime.record_thread.join(5)
            self.assertFalse(self.runtime.record_thread.is_alive())

    def assert_no_control_or_data(self) -> None:
        self.assertFalse(
            any(
                event[1] in ("enable", "move_js", "move_j", "move", "add_frame", "save_episode")
                for event in self.events
            )
        )

    def test_first_episode_moves_only_follower_gradually_before_recording(self) -> None:
        self.runtime.state["connected"] = False
        self.leader.positions[3] = 0.3

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            runtime.state["connected"] = True
            return True, "已就绪"

        with patch.object(recorder, "do_connect", side_effect=prepare):
            result = self.client.post("/api/record/start")
            self.finish()
        self.assertEqual(result.status_code, 202)
        self.assertEqual(self.runtime.state["episode_count"], 1)
        first_frame = next(i for i, e in enumerate(self.events) if e[1] == "add_frame")
        moves = [e for e in self.events[:first_frame] if e[1] == "move_js"]
        self.assertGreater(len(moves), 100)
        self.assertTrue(all(e[0] == "从左" for e in moves))
        previous = [0.0] * 7
        for _, _, command in moves:
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(command, previous, strict=True)), 0.25 * 0.02 + 1e-9
            )
            previous = command
        self.assertEqual(moves[-1][2], self.leader.positions)
        self.assertNotEqual(self.leader.positions, devices.HOME_POS)
        self.assertAlmostEqual(self.runtime.state["last_episode"]["timing"]["elapsed_seconds"], 0.25)
        self.assertEqual(self.runtime.state["last_episode"]["frames"], 3)
        self.assertGreater(self.clock.now, 3.5)

    def test_every_next_episode_automatically_realigns(self) -> None:
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()
        self.assertEqual(self.runtime.state["episode_count"], 1)
        self.events.clear()
        self.leader.positions[6] = 0.3
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()
        self.assertEqual(self.follower.positions, self.leader.positions)
        self.assertEqual(self.runtime.state["episode_count"], 2)
        self.assertIsNone(self.runtime.state["last_error"])

    def test_linkage_follower_without_teaching_broadcast_can_prepare_align_and_record_again(self) -> None:
        self.runtime.state["connected"] = False

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            devices.transition_to_collection(runtime)
            runtime.state["connected"] = True
            return True, "已就绪"

        with (
            patch.object(recorder, "do_connect", side_effect=prepare),
            patch.object(self.follower, "get_leader_joint_angles", return_value=None),
            patch.object(hardware_status, "time", self.clock),
        ):
            for number in (1, 2):
                self.leader.positions[6] = number * 0.2
                self.assertEqual(self.client.post("/api/record/start").status_code, 202)
                self.finish()
                self.assertEqual(self.runtime.state["episode_count"], number)
                self.assertEqual(self.follower.ctrl_mode, 6)
                self.assertEqual(self.follower.positions, self.leader.positions)
                self.assertIsNone(self.runtime.state["last_error"])
                self.assertEqual(self.runtime.state["record_phase"], "saved")

    def test_unconfirmed_linkage_role_blocks_recording_without_motion_or_frames(self) -> None:
        self.follower.ctrl_mode = 6
        self.leader.positions[6] = 0.2
        with patch.object(hardware_status, "time", self.clock):
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.finish()
        self.assertEqual(self.runtime.state["record_frames"], 0)
        self.assertIn("角色未确认", self.runtime.state["last_error"])
        self.assert_no_control_or_data()

    def test_invalid_feedback_and_missing_channels_cannot_pass(self) -> None:
        for positions in ([0.0] * 6, [float("nan")] * 7, [float("inf")] * 7):
            with self.subTest(positions=positions):
                self.leader.positions = positions
                self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        with patch.object(self.leader, "get_leader_joint_angles", return_value=None):
            self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.runtime.channels["can0"]["detected"] = False
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assert_no_control_or_data()

    def test_duplicate_arm_role_cannot_bypass_check(self) -> None:
        self.runtime.channels["can3"]["label"] = "主左"
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assert_no_control_or_data()

    def test_alignment_uses_positions_after_preparation(self) -> None:
        self.runtime.state["connected"] = False

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            runtime.state["connected"] = True
            self.follower.positions[0] = 0.3
            return True, "已就绪"

        with patch.object(recorder, "do_connect", side_effect=prepare):
            self.client.post("/api/record/start")
            self.finish()
        self.assertIsNone(self.runtime.state["last_error"])
        self.assertEqual(self.runtime.state["episode_count"], 1)
        self.assertEqual(self.follower.positions, self.leader.positions)

    def test_target_is_teaching_broadcast_instead_of_old_ordinary_feedback(self) -> None:
        self.leader.command = [0.3] * 7
        result = self.client.post("/api/record/start")
        self.finish()
        self.assertEqual(result.status_code, 202)
        self.assertEqual(self.runtime.state["record_phase"], "saved")
        self.assertEqual(self.follower.positions, self.leader.command)

    def test_same_position_but_stale_feedback_cannot_pass(self) -> None:
        for arm, reader, source in (
            (self.leader, "get_leader_joint_angles", "主臂示教"),
            (self.follower, "get_joint_angles", "从臂"),
        ):
            with (
                self.subTest(source=source),
                patch.object(arm, reader, return_value=SimpleNamespace(msg=[0.0] * 7, timestamp=1.0)),
            ):
                self.assertEqual(self.client.post("/api/record/start").status_code, 202)
                self.finish()
                self.assertIn(f"{source}关节反馈未更新", self.runtime.state["last_error"])
                self.assert_no_control_or_data()

    def test_frozen_ordinary_feedback_after_mode_switch_allows_first_and_next_episode(self) -> None:
        self.runtime.state["connected"] = False
        frozen = SimpleNamespace(msg=[0.0] * 7, timestamp=1.0)

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            runtime.state["connected"] = True
            # 模拟切到示教后普通反馈冻结，两臂在示教广播中已移到相同的新位置。
            self.leader.positions = self.follower.positions = [0.4] * 7
            return True, "已就绪"

        with (
            patch.object(self.leader, "get_joint_angles", return_value=frozen) as ordinary,
            patch.object(recorder, "do_connect", side_effect=prepare),
        ):
            for expected_count in (1, 2):
                self.assertEqual(self.client.post("/api/record/start").status_code, 202)
                self.finish()
                self.assertEqual(self.runtime.state["record_phase"], "saved")
                self.assertEqual(self.runtime.state["episode_count"], expected_count)
                self.assertIsNone(self.runtime.state["last_error"])
            ordinary.assert_called_once()

    def test_ordinary_feedback_can_be_absent_while_teaching_broadcast_is_valid(self) -> None:
        with patch.object(self.leader, "get_joint_angles", return_value=None) as ordinary:
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.finish()
            self.assertEqual(self.runtime.state["record_phase"], "saved")
            ordinary.assert_not_called()

    def test_missing_teaching_broadcast_never_falls_back_to_ordinary_feedback(self) -> None:
        with patch.object(self.leader, "get_leader_joint_angles", return_value=None):
            self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assertIn("主臂示教", self.runtime.state["last_error"])
        self.assert_no_control_or_data()

    def test_preparation_must_wait_for_valid_teaching_source(self) -> None:
        self.runtime.state["connected"] = False

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            runtime.state["connected"] = True
            return True, "已就绪"

        with (
            patch.object(recorder, "do_connect", side_effect=prepare),
            patch.object(self.leader, "get_leader_joint_angles", return_value=None),
        ):
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.finish()
        self.assertIn("主臂示教", self.runtime.state["last_error"])
        self.assertEqual(self.runtime.state["record_phase"], "error")
        self.assert_no_control_or_data()

    def test_ordinary_feedback_used_before_teaching_and_after_home(self) -> None:
        # 主臂还未进入示教或刚退出示教回位时，旧示教广播不能参与对齐判断。
        self.leader.command = [0.3] * 7
        for connected, needs_restore in ((False, False), (True, True)):
            with self.subTest(connected=connected, needs_restore=needs_restore):
                self.runtime.state.update(connected=connected, teleop_needs_restore=needs_restore)
                self.assertEqual(devices.check_alignment(self.runtime), (True, ""))
                self.leader.positions[2] = 0.3
                self.assertFalse(devices.check_alignment(self.runtime)[0])
                self.leader.positions[2] = 0.0
        self.assert_no_control_or_data()

    def test_status_displays_mode_appropriate_joint_positions(self) -> None:
        self.leader.positions = [0.1] * 7
        self.leader.command = [0.3] * 7
        self.runtime.channels["can0"]["requested_role"] = "leader"
        self.client.get("/api/status")  # 第一份快照建立时间戳基线。
        self.events.clear()
        info = self.client.get("/api/status").json["arms"]["can0"]
        self.assertEqual(info["hardware"]["angles"], [17.2] * 7)
        self.assertNotIn("data", info)
        self.assertEqual(self.events.count(("主左", "read_leader")), 1)
        self.assertEqual(self.events.count(("主左", "read_follower")), 1)
        self.runtime.state["teleop_needs_restore"] = True
        self.runtime.channels["can0"]["requested_role"] = "follower"
        self.assertEqual(self.client.get("/api/status").json["arms"]["can0"]["hardware"]["angles"], [5.7] * 7)

    def assert_no_episode(self) -> None:
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assertIsNone(self.runtime.state["writer_error"])
        self.assertFalse(any(e[1] in ("add_frame", "save_episode") for e in self.events))

    def test_cancel_during_alignment_holds_follower_and_never_starts_episode(self) -> None:
        self.leader.positions[1] = 0.3
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        move = self.follower.move_js

        def wait(command: list) -> None:
            move(command)
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试等待超时")

        with patch.object(self.follower, "move_js", side_effect=wait):
            self.client.post("/api/record/start")
            self.assertTrue(entered.wait(2))
            status = self.client.get("/api/status").json
            self.assertEqual(status["record"]["phase"], "aligning")
            self.assertEqual(status["record"]["frames"], 0)
            self.assertEqual(status["record"]["elapsed_seconds"], 0)
            for route, payload in (
                ("/api/record/start", {}),
                ("/api/pair/home", {"targets": {"can0": "主左", "can1": "从左"}}),
                ("/api/disconnect", {"targets": {"can0": "主左", "can1": "从左"}}),
            ):
                self.assertFalse(self.client.post(route, json=payload).json["ok"])
            self.assertTrue(self.client.post("/api/record/stop").json["ok"])
            self.finish()
        commands = [e[2] for e in self.events if e[1] == "move_js"]
        self.assertEqual(len(commands), 2)  # 第一条渐进指令 + 取消后原地保持。
        self.assertEqual(commands[0], commands[1])
        self.assertNotEqual(commands[-1], self.leader.positions)
        self.assertEqual(self.runtime.state["record_phase"], "idle")
        self.assert_no_episode()

    def test_leader_moving_during_alignment_stops_instead_of_chasing(self) -> None:
        self.leader.positions[1] = 0.3
        move = self.follower.move_js

        def move_then_change_leader(command: list) -> None:
            move(command)
            self.leader.positions[1] += 0.05

        with patch.object(self.follower, "move_js", side_effect=move_then_change_leader):
            self.client.post("/api/record/start")
            self.finish()
        self.assertIn("主臂位置发生变化", self.runtime.state["last_error"])
        self.assert_no_episode()

    def test_stalled_follower_cannot_accumulate_large_commands_or_record(self) -> None:
        self.leader.positions[1] = 0.3
        self.follower.follow_commands = False
        self.client.post("/api/record/start")
        self.finish()
        self.assertIn("未能跟上", self.runtime.state["last_error"])
        commands = [e[2] for e in self.events if e[1] == "move_js"]
        self.assertTrue(
            all(max(abs(v) for v in command) <= devices.ALIGN_TOLERANCE_RAD for command in commands)
        )
        self.assertEqual(commands[-1], [0.0] * 7)
        self.assert_no_episode()

    def test_settle_timeout_never_records_even_if_within_old_alignment_tolerance(self) -> None:
        self.leader.positions = [0.3] * 7
        self.follower.tracking_error = 0.05
        self.client.post("/api/record/start")
        self.finish()
        self.assertIn("到位超时", self.runtime.state["last_error"])
        self.assert_no_episode()

    def test_stale_feedback_during_motion_stops_without_recording(self) -> None:
        for arm, reader, source in (
            (self.leader, "get_leader_joint_angles", "主臂示教"),
            (self.follower, "get_joint_angles", "从臂"),
        ):
            with self.subTest(source=source):
                self.events.clear()
                self.leader.positions = [0.3] * 7
                self.follower.positions = [0.0] * 7
                read = getattr(arm, reader)
                frozen = None

                def freeze_after_moving(read: Callable[[], SimpleNamespace] = read) -> SimpleNamespace:
                    nonlocal frozen
                    if any(e[1] == "move_js" for e in self.events):
                        if frozen is None:
                            frozen = read()
                        return frozen
                    return read()

                with patch.object(arm, reader, side_effect=freeze_after_moving):
                    self.client.post("/api/record/start")
                    self.finish()
                self.assertEqual(self.runtime.state["record_phase"], "error")
                if source == "从臂":
                    self.assertIn(("从左", "emergency_stop"), self.events)
                    self.assertIn("运动操作已锁定", self.runtime.state["recovery_required"])
                    before = len(self.events)
                    self.assertFalse(self.client.post("/api/record/start").json["ok"])
                    self.assertFalse(
                        self.client.post(
                            "/api/pair/home", json={"targets": {"can0": "主左", "can1": "从左"}}
                        ).json["ok"]
                    )
                    self.assertFalse(self.client.post("/api/arm/move/can1").json["ok"])
                    self.assertEqual(len(self.events), before)
                else:
                    self.assertIn("反馈未更新", self.runtime.state["last_error"])
                self.assert_no_episode()

    def test_failed_emergency_stop_request_still_locks_motion_and_reports_failure(self) -> None:
        self.leader.positions = [0.3] * 7
        read = self.follower.get_joint_angles

        def lose_feedback():
            return None if any(e[1] == "move_js" for e in self.events) else read()

        with (
            patch.object(self.follower, "get_joint_angles", side_effect=lose_feedback),
            patch.object(self.follower, "electronic_emergency_stop", side_effect=OSError("CAN 发送失败")),
        ):
            self.client.post("/api/record/start")
            self.finish()
        self.assertIn("急停指令发送失败", self.runtime.state["recovery_required"])
        self.assertIn("以实际反馈为准", self.runtime.state["last_error"])
        self.assertFalse(self.client.post("/api/record/start").json["ok"])
        self.assert_no_episode()

    def test_large_distance_extends_ramp_duration_and_limits_each_step(self) -> None:
        self.leader.positions[0] = 2.0
        devices.align_follower_before_recording(self.runtime)
        commands = [e[2] for e in self.events if e[1] == "move_js"]
        self.assertGreaterEqual(self.clock.now, 1.5 * 2.0 / devices.ALIGN_MAX_SPEED_RAD_S)
        previous = [0.0] * 7
        for command in commands:
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(command, previous, strict=True)), 0.25 * 0.02 + 1e-9
            )
            previous = command
        self.assertEqual(commands[-1], self.leader.positions)
        self.assert_no_episode()

    def test_aligned_but_moving_pair_cannot_pass(self) -> None:
        sleep = self.clock.sleep

        def move_both(duration: float) -> None:
            self.leader.positions[0] += 0.04
            self.follower.positions[0] += 0.04
            sleep(duration)

        with patch.object(self.clock, "sleep", side_effect=move_both):
            self.client.post("/api/record/start")
            self.finish()
        self.assertIn("仍在移动", self.runtime.state["last_error"])
        self.assert_no_control_or_data()

    def test_stability_check_is_excluded_from_recorded_time_and_frames(self) -> None:
        self.client.post("/api/record/start")
        self.finish()
        self.assertEqual(self.runtime.state["record_phase"], "saved")
        self.assertAlmostEqual(self.clock.now, 0.46)  # 含 10 ms 全部反馈更新检查，不计入录制。
        self.assertEqual(self.runtime.state["last_episode"]["timing"]["elapsed_seconds"], 0.25)
        self.assertEqual(self.runtime.state["last_episode"]["frames"], 3)

    def test_cancel_during_stability_check_never_starts_recording(self) -> None:
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        sleep = self.clock.sleep

        def wait(duration: float) -> None:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试等待超时")
            sleep(duration)

        with patch.object(self.clock, "sleep", side_effect=wait):
            self.client.post("/api/record/start")
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.runtime.state["record_phase"], "preparing")
            self.client.post("/api/record/stop")
            self.finish()
        self.assertEqual(self.runtime.state["record_phase"], "idle")
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assert_no_control_or_data()

    def test_first_command_rechecks_if_leader_moves_after_gate(self) -> None:
        reads = 0
        read = self.leader.get_leader_joint_angles

        def change_after_checks() -> SimpleNamespace:
            nonlocal reads
            reads += 1
            # 续录预检查一次，准备完成后三次稳定检查，然后才发送第一帧。
            if reads > devices.ALIGN_SAMPLES + 1:
                self.leader.command = [0.3] * 7
            return read()

        with patch.object(self.leader, "get_leader_joint_angles", side_effect=change_after_checks):
            self.client.post("/api/record/start")
            self.finish()
        self.assertIn("第一帧", self.runtime.state["last_error"])
        self.assertIsNone(self.runtime.state["writer_error"])
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assert_no_control_or_data()


if __name__ == "__main__":
    unittest.main(verbosity=2)
