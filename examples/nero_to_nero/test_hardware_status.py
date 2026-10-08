"""当前硬件状态与运动前检查：只用模拟反馈，不连接 CAN。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform import devices, hardware_status
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeArm


class StatusArm(FakeArm):
    def __init__(self, events: list):
        super().__init__("can0", events)
        self.stamps = {}
        self.frozen = set()
        self.missing = set()
        self.joints = [True] * 7
        self.arm_status = 0
        self.err_code = 0
        self.angles = [0.0] * 7

    def message(self, key: str, msg: object) -> SimpleNamespace | None:
        if key in self.missing:
            return None
        stamp = self.stamps.setdefault(key, 1.0)
        if key not in self.frozen:
            stamp += 1
            self.stamps[key] = stamp
        return SimpleNamespace(timestamp=stamp, msg=msg)

    def get_joint_angles(self) -> SimpleNamespace | None:
        return self.message("joint", self.angles)

    def get_leader_joint_angles(self) -> SimpleNamespace | None:
        return self.message("teaching", [0.1] * 7)

    def get_arm_status(self) -> SimpleNamespace | None:
        return self.message(
            "controller",
            SimpleNamespace(ctrl_mode=self.ctrl_mode, arm_status=self.arm_status, err_code=self.err_code),
        )

    def get_driver_states(self, joint: int) -> SimpleNamespace | None:
        return self.message(
            str(joint),
            SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.joints[joint - 1])),
        )


class HardwareStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = []
        self.runtime = Runtime()
        self.arm = StatusArm(self.events)
        self.runtime.channels["can0"].update(arm=self.arm, detected=True)
        self.clock = Clock(self.events)
        clock_patch = patch.object(hardware_status, "time", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def read(self) -> dict:
        return hardware_status.read_status(self.runtime, "can0")

    def inspect(self) -> dict:
        return hardware_status.inspect_status(self.runtime, "can0")

    def test_first_cached_sample_is_unknown_then_updates_confirm_current_state(self) -> None:
        self.assertEqual(self.read()["enabled"]["state"], "unknown")
        result = self.read()
        self.assertEqual(result["communication"]["state"], "live")
        self.assertEqual(result["enabled"]["state"], "enabled")
        self.assertEqual(result["health"]["state"], "normal")
        self.assertEqual(result["angles"], [0.0] * 7)

    def test_stale_driver_is_unknown_even_while_joint_positions_update(self) -> None:
        self.inspect()
        self.arm.frozen.add("3")
        self.clock.now += 1.1
        result = self.read()
        self.assertEqual(result["communication"]["state"], "live")
        self.assertIsNone(result["enabled"]["joints"][2])
        self.assertEqual(result["enabled"]["state"], "unknown")

    def test_partial_disabled_and_missing_are_different(self) -> None:
        self.arm.joints[1] = False
        self.assertEqual(self.inspect()["enabled"]["state"], "partial")
        self.arm.joints = [False] * 7
        self.assertEqual(self.read()["enabled"]["state"], "disabled")
        self.arm.missing.add("1")
        self.assertEqual(self.read()["enabled"]["state"], "unknown")

    def test_teaching_broadcast_does_not_prove_enable_or_health(self) -> None:
        self.arm.missing.update({"controller", "joint", *map(str, range(1, 8))})
        result = self.inspect()
        self.assertEqual(result["communication"]["state"], "live")
        self.assertIn("示教广播", result["mode"]["label"])
        self.assertEqual(result["health"]["state"], "unknown")
        self.assertEqual(result["enabled"]["state"], "unknown")

    def test_hardware_mode_is_independent_of_last_commanded_mode(self) -> None:
        self.runtime.channels["can0"]["feedback_mode"] = "leader"
        self.assertEqual(self.inspect()["mode"]["code"], 1)
        self.arm.ctrl_mode = 6
        self.assertEqual(self.read()["mode"]["code"], 6)

    def test_reconnect_and_timestamp_reset_require_new_samples(self) -> None:
        self.inspect()
        self.arm.stamps = {}
        self.assertEqual(self.read()["enabled"]["state"], "unknown")
        self.assertEqual(self.read()["enabled"]["state"], "enabled")
        self.runtime.channels["can0"]["arm"] = StatusArm(self.events)
        self.assertEqual(self.read()["enabled"]["state"], "unknown")

    def test_missing_or_malformed_feedback_never_means_normal(self) -> None:
        self.arm.err_code = None
        self.arm.angles = ["bad"] * 7
        result = self.inspect()
        self.assertEqual(result["health"]["state"], "unknown")
        self.assertIsNone(result["angles"])
        with patch.object(self.arm, "get_arm_status", side_effect=OSError("读取失败")):
            self.assertTrue(self.read()["errors"])

    def test_read_only_inspection_is_available_during_recording_and_close_failure(self) -> None:
        self.runtime.state.update(recording=True, disable_error="失能未确认")
        client = create_app(self.runtime).test_client()
        result = client.get("/api/arm/status/can0")
        self.assertTrue(result.json["ok"])
        self.assertEqual(result.json["hardware"]["enabled"]["state"], "enabled")
        self.assertEqual(client.get("/api/arm/status/can9").status_code, 404)
        self.assertFalse(any(e[0] == "can0" for e in self.events))

    def test_ready_guard_rejects_fault_missing_partial_or_wrong_mode(self) -> None:
        hardware_status.require_ready_for_motion(self.runtime, "can0")
        for field, value in (
            ("arm_status", 1),
            ("err_code", 8),
            ("ctrl_mode", 6),
            ("joints", [True] * 6 + [None]),
        ):
            old = getattr(self.arm, field)
            setattr(self.arm, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "暂不执行运动"):
                hardware_status.require_ready_for_motion(self.runtime, "can0")
            setattr(self.arm, field, old)

    def test_known_fault_blocks_enable_and_motion(self) -> None:
        self.arm.arm_status = 1
        devices.micro_move(self.runtime, "can0")
        self.assertIn("急停", self.runtime.channels["can0"]["status"])
        self.assertFalse(any(e[1] in ("enable", "move_j", "move_js", "reset") for e in self.events))

    def test_disabled_brake_status_is_not_ready_not_a_physical_lock_diagnosis(self) -> None:
        self.arm.arm_status = 6
        self.arm.joints = [False] * 7
        result = self.inspect()
        self.assertEqual(result["health"]["state"], "not_ready")
        self.assertEqual(result["health"]["code"], 6)
        self.assertEqual(result["health"]["error_code"], 0)
        self.assertIn("关节未使能", result["health"]["label"])
        with self.assertRaisesRegex(RuntimeError, "暂不执行运动"):
            hardware_status.require_ready_for_motion(self.runtime, "can0")
        self.assertFalse(any(e[1] in ("enable", "move_j", "move_js", "reset") for e in self.events))

    def test_brake_status_with_fault_enabled_partial_or_missing_is_not_relaxed(self) -> None:
        self.arm.arm_status = 6
        for joints, err_code in (
            ([False] * 7, 1),
            ([True] * 7, 0),
            ([False] * 6 + [True], 0),
            ([False] * 6 + [None], 0),
        ):
            self.arm.joints = joints
            self.arm.err_code = err_code
            with self.subTest(joints=joints, err_code=err_code):
                self.assertEqual(self.inspect()["health"]["state"], "fault")
                with self.assertRaisesRegex(RuntimeError, "暂不执行运动"):
                    hardware_status.require_ready_for_motion(self.runtime, "can0")

    def test_disabled_brake_status_allows_requested_enable_then_requires_normal_status(self) -> None:
        for clears in (True, False):
            self.events.clear()
            self.arm.joints = [False] * 7
            self.arm.arm_status = 6

            def enable(clears: bool = clears) -> bool:
                self.events.append(("can0", "enable"))
                self.arm.joints = [True] * 7
                if clears:
                    self.arm.arm_status = 0
                return True

            with (
                self.subTest(clears=clears),
                patch.object(devices, "time", self.clock),
                patch.object(self.arm, "enable", side_effect=enable),
                patch.object(self.arm, "move_j", create=True) as move,
            ):
                devices.micro_move(self.runtime, "can0")
                self.assertIn(("can0", "enable"), self.events)
                self.assertEqual(move.call_count, 2 if clears else 0)
                if not clears:
                    self.assertIn("暂不执行运动", self.runtime.channels["can0"]["status"])

    def test_controller_may_report_brake_status_briefly_after_drivers_enable(self) -> None:
        self.arm.arm_status = 6
        self.inspect()
        original = self.arm.get_arm_status

        def status():
            if self.clock.now > 0.2:
                self.arm.arm_status = 0
            return original()

        with patch.object(self.arm, "get_arm_status", side_effect=status):
            hardware_status.require_ready_for_motion(self.runtime, "can0")
        self.assertGreater(self.clock.now, 0.2)

    def test_gripper_is_reported_separately(self) -> None:
        grip = SimpleNamespace(
            get_gripper_status=lambda: self.arm.message(
                "grip", SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=False))
            )
        )
        self.runtime.channels["can0"]["gripper"] = grip
        result = self.inspect()
        self.assertTrue(result["gripper"]["present"])
        self.assertFalse(result["gripper"]["enabled"])
        self.assertEqual(result["enabled"]["state"], "enabled")

    def test_linkage_follower_uses_actual_joints_without_teaching_broadcast(self) -> None:
        self.arm.missing.add("teaching")
        devices.set_control_role(self.runtime, "can0", "follower")
        result = self.inspect()
        self.assertEqual(result["mode"]["code"], 6)
        self.assertIn("从臂受控", result["mode"]["label"])
        self.assertEqual(result["angles"], [0.0] * 7)
        self.assertEqual(result["angles_source"], "joint_feedback")
        hardware_status.require_ready_for_motion(self.runtime, "can0")

    def test_linkage_role_cannot_be_inferred_from_channel_label_or_stream_alone(self) -> None:
        self.arm.ctrl_mode = 6
        self.runtime.channels["can0"].update(label="从左", feedback_mode="follower")
        self.assertIn("角色未确认", self.inspect()["mode"]["label"])
        with self.assertRaisesRegex(RuntimeError, "控制模式和角色"):
            hardware_status.require_ready_for_motion(self.runtime, "can0")

    def test_linkage_leader_never_passes_follower_motion_check(self) -> None:
        devices.set_control_role(self.runtime, "can0", "leader")
        result = self.inspect()
        self.assertEqual(result["angles_source"], "teaching_broadcast")
        with self.assertRaisesRegex(RuntimeError, "暂不执行运动"):
            hardware_status.require_ready_for_motion(self.runtime, "can0")

    def test_follower_cannot_substitute_teaching_broadcast_for_missing_actual_position(self) -> None:
        devices.set_control_role(self.runtime, "can0", "follower")
        self.arm.missing.add("joint")
        result = self.inspect()
        self.assertIsNone(result["angles"])
        with self.assertRaisesRegex(RuntimeError, "实际关节位置无有效新反馈"):
            hardware_status.require_ready_for_motion(self.runtime, "can0")

    def test_role_switch_requires_updated_feedback_and_failed_request_clears_role(self) -> None:
        self.inspect()
        devices.set_control_role(self.runtime, "can0", "follower")
        self.assertEqual(self.read()["enabled"]["state"], "unknown")
        self.arm.frozen.add("joint")
        with self.assertRaisesRegex(RuntimeError, "实际关节位置无有效新反馈"):
            hardware_status.require_ready_for_motion(self.runtime, "can0")
        with (
            patch.object(self.arm, "set_follower_mode", side_effect=OSError("发送失败")),
            self.assertRaises(OSError),
        ):
            devices.set_control_role(self.runtime, "can0", "follower")
        self.assertIsNone(self.runtime.channels["can0"]["requested_role"])

    def test_linkage_follower_still_requires_health_and_all_enabled(self) -> None:
        devices.set_control_role(self.runtime, "can0", "follower")
        for field, value in (
            ("arm_status", 1),
            ("err_code", 8),
            ("joints", [True] * 6 + [False]),
            ("joints", [True] * 6 + [None]),
        ):
            old = getattr(self.arm, field)
            setattr(self.arm, field, value)
            with self.subTest(field=field, value=value), self.assertRaisesRegex(RuntimeError, "暂不执行运动"):
                hardware_status.require_ready_for_motion(self.runtime, "can0")
            setattr(self.arm, field, old)


if __name__ == "__main__":
    unittest.main()
