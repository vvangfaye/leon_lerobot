"""连接只建立通信；用模拟设备复现残留示教模式与固件查询失败。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import can
from nero_platform import devices, hardware_status
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from pyAgxArm.protocols.can_protocol.drivers.core.driver_context import DriverContext
from pyAgxArm.protocols.can_protocol.drivers.effector.agx_gripper.default.driver import (
    Driver as GripperDriver,
)
from test_platform import Clock, FakeArm, attach_fake_pair


def teaching_feedback(timestamp: float, **overrides) -> SimpleNamespace:
    values = {"teaching_range_per": 100, "max_range_config": 0.07, "teaching_friction": 8}
    values.update(overrides)
    return SimpleNamespace(timestamp=timestamp, msg=SimpleNamespace(**values))


class ConnectionArm(FakeArm):
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="gripper"))

    def __init__(self, events: list):
        super().__init__("can0", events)
        self.mode = "leader"
        self.fresh = True
        self.firmware = None
        self.transport_error = None
        self.honor_mode_switch = True

    def connect(self) -> None:
        self.events.append((self.name, "connect"))
        if self.transport_error:
            raise self.transport_error

    def disconnect(self) -> None:
        self.events.append((self.name, "disconnect"))

    def init_effector(self, effector: str) -> None:
        return None

    def get_firmware(self, timeout: float) -> dict | None:
        self.events.append((self.name, "firmware", timeout))
        return self.firmware

    def read(self, mode: str) -> SimpleNamespace | None:
        if mode != self.mode:
            return None
        if self.fresh:
            self.feedback_timestamp += 1
        return SimpleNamespace(msg=[0.0] * 7, timestamp=self.feedback_timestamp)

    def get_joint_angles(self) -> SimpleNamespace | None:
        return self.read("follower")

    def get_leader_joint_angles(self) -> SimpleNamespace | None:
        return self.read("leader")

    def set_follower_mode(self) -> None:
        super().set_follower_mode()
        if self.honor_mode_switch:
            self.mode = "follower"

    def set_leader_mode(self) -> None:
        super().set_leader_mode()
        self.mode = "leader"

    def enable(self) -> bool:
        self.events.append((self.name, "enable"))
        return self.mode == "follower"

    def get_arm_status(self) -> SimpleNamespace:
        self.feedback_timestamp += 1
        return SimpleNamespace(
            timestamp=self.feedback_timestamp,
            msg=SimpleNamespace(ctrl_mode=6 if self.mode == "leader" else 1, arm_status=0, err_code=0),
        )

    def get_driver_states(self, joint: int) -> SimpleNamespace:
        self.feedback_timestamp += 1
        return SimpleNamespace(
            timestamp=self.feedback_timestamp,
            msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.mode == "follower")),
        )

    def move_j(self, angles: list) -> None:
        self.events.append((self.name, "move_j", angles[:]))


class ConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Runtime()
        self.runtime.channels["can0"]["label"] = "主左"
        self.events = []
        self.arm = ConnectionArm(self.events)
        self.clock = Clock(self.events)
        for guard in (
            patch.object(devices, "time", self.clock),
            patch.object(devices.AgxArmFactory, "create_arm", return_value=self.arm),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def test_live_teaching_feedback_connects_even_without_firmware_reply(self) -> None:
        devices.connect_channel(self.runtime, "can0")
        info = self.runtime.channels["can0"]
        self.assertTrue(info["detected"])
        self.assertIn("固件查询未响应", info["status"])
        self.assertIn("示教反馈正常", info["status"])
        self.assertEqual(info["feedback_mode"], "leader")
        self.assertFalse(self.runtime.state["connected"])
        self.assertTrue(devices._leader_in_teaching_mode(self.runtime))
        hardware_status.read_status(self.runtime, "can0")
        self.assertEqual(hardware_status.read_status(self.runtime, "can0")["angles"], [0.0] * 7)
        self.assertTrue(all(e[1] in ("connect", "firmware", 0.05) for e in self.events))

    def test_live_ordinary_feedback_connects_without_firmware_reply(self) -> None:
        self.arm.mode = "follower"
        devices.connect_channel(self.runtime, "can0")
        self.assertTrue(self.runtime.channels["can0"]["detected"])
        self.assertEqual(self.runtime.channels["can0"]["feedback_mode"], "follower")
        self.assertFalse(devices._leader_in_teaching_mode(self.runtime))

    def test_stale_feedback_never_confirms_connection(self) -> None:
        self.arm.fresh = False
        devices.connect_channel(self.runtime, "can0")
        self.assertFalse(self.runtime.channels["can0"]["detected"])
        self.assertIn("通信未确认", self.runtime.channels["can0"]["status"])
        self.assertLess(self.clock.now, 1.1)

    def test_firmware_response_still_confirms_connection_without_joint_stream(self) -> None:
        self.arm.mode = None
        self.arm.firmware = {"software_version": "1.21"}
        devices.connect_channel(self.runtime, "can0")
        self.assertTrue(self.runtime.channels["can0"]["detected"])
        self.assertIn("等待关节反馈", self.runtime.channels["can0"]["status"])

    def test_firmware_query_exception_does_not_hide_live_teaching_stream(self) -> None:
        with patch.object(self.arm, "get_firmware", side_effect=RuntimeError("固件读取错误")):
            devices.connect_channel(self.runtime, "can0")
        self.assertTrue(self.runtime.channels["can0"]["detected"])
        self.assertIn("固件读取错误", self.runtime.channels["can0"]["status"])

    def test_reconnect_closes_previous_reader_before_creating_new_one(self) -> None:
        previous = ConnectionArm(self.events)
        previous.name = "old"
        self.runtime.channels["can0"]["arm"] = previous
        self.runtime.channels["can0"]["requested_role"] = "follower"
        devices.connect_channel(self.runtime, "can0")
        self.assertLess(self.events.index(("old", "disconnect")), self.events.index(("can0", "connect")))
        self.assertIs(self.runtime.channels["can0"]["arm"], self.arm)
        self.assertIsNone(self.runtime.channels["can0"]["requested_role"])

    def test_failed_connection_releases_partial_resources_and_clears_detection(self) -> None:
        self.arm.transport_error = RuntimeError("CAN 连接失败")
        self.runtime.channels["can0"]["detected"] = True
        devices.connect_channel(self.runtime, "can0")
        self.assertFalse(self.runtime.channels["can0"]["detected"])
        self.assertIsNone(self.runtime.channels["can0"]["arm"])
        self.assertIn(("can0", "disconnect"), self.events)

    def test_micro_move_and_home_time_out_instead_of_leaving_device_busy(self) -> None:
        self.runtime.channels["can0"].update(arm=self.arm, detected=True)
        client = create_app(self.runtime).test_client()

        class InlineThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        with (
            patch.object(self.arm, "enable", return_value=False),
            patch("nero_platform.app.threading.Thread", InlineThread),
        ):
            for route in ("/api/arm/move/can0", "/api/arm/disable/can0"):
                with self.subTest(route=route):
                    self.clock.now = 0
                    self.assertTrue(client.post(route).json["ok"])
                    self.assertIn("使能超时", self.runtime.channels["can0"]["status"])
                    self.assertFalse(self.runtime.device_busy)
                    # 初次读取到残留联动模式时，额外等待 0.2 秒退出示教；使能仍有固定超时。
                    self.assertLess(self.clock.now, devices.ENABLE_TIMEOUT_SECONDS + 0.3)

    def test_micro_move_exits_residual_teaching_before_enable(self) -> None:
        devices.connect_channel(self.runtime, "can0")
        devices.micro_move(self.runtime, "can0")
        self.assertIn("微动完成", self.runtime.channels["can0"]["status"])
        self.assertLess(self.events.index(("can0", "follower")), self.events.index(("can0", "enable")))
        moves = [event for event in self.events if event[1] == "move_j"]
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0][2], [0.0] * 6 + [0.0873])
        self.assertEqual(moves[1][2], [0.0] * 7)
        self.assertEqual(self.runtime.channels["can0"]["feedback_mode"], "follower")

    def test_failed_mode_switch_times_out_with_details_without_moving(self) -> None:
        devices.connect_channel(self.runtime, "can0")
        self.arm.honor_mode_switch = False
        devices.micro_move(self.runtime, "can0")
        status = self.runtime.channels["can0"]["status"]
        self.assertIn("未执行微动", status)
        self.assertIn("控制模式=6", status)
        self.assertIn("未使能关节=J1,J2,J3,J4,J5,J6,J7", status)
        self.assertFalse(any(event[1] in ("move_j", "move_js", "reset", "disable") for event in self.events))

    def test_missing_feedback_is_reported_separately_from_disabled_joints(self) -> None:
        with (
            patch.object(self.arm, "get_arm_status", return_value=None),
            patch.object(self.arm, "get_driver_states", return_value=None),
        ):
            message = devices._enable_failure_details(self.arm)
        self.assertIn("未收到机械臂控制状态反馈", message)
        self.assertIn("驱动反馈缺失=J1,J2,J3,J4,J5,J6,J7", message)
        self.assertNotIn("未使能关节", message)

    def test_collection_restores_teaching_only_after_enabling(self) -> None:
        attach_fake_pair(self.runtime, self.events)
        self.runtime.channels["can0"].update(arm=self.arm, feedback_mode="leader")
        devices.transition_to_collection(self.runtime)
        self.assertLess(self.events.index(("can0", "follower")), self.events.index(("can0", "enable")))
        self.assertLess(self.events.index(("can0", "enable")), self.events.index(("can0", "leader")))
        self.assertEqual(self.runtime.channels["can0"]["feedback_mode"], "leader")
        self.assertFalse(any(event[1] in ("move_j", "move_js", "reset", "disable") for event in self.events))

    def test_leader_gripper_releases_residual_hold_before_teaching(self) -> None:
        attach_fake_pair(self.runtime, self.events)
        gripper = self.runtime.channels["can0"]["gripper"]
        configure = gripper.set_gripper_teaching_pendant_param

        def require_released(**kwargs) -> bool:
            self.assertFalse(gripper.enabled, "残留的位置保持控制尚未释放")
            return configure(**kwargs)

        with patch.object(gripper, "set_gripper_teaching_pendant_param", side_effect=require_released):
            devices.transition_to_collection(self.runtime)
        names = [event[1] for event in self.events if event[0] == "主左"]
        self.assertLess(names.index("enable"), names.index("disable_gripper"))
        self.assertLess(names.index("disable_gripper"), names.index("teaching"))
        self.assertLess(names.index("teaching"), names.index("leader"))
        self.assertNotIn(("从左", "disable_gripper"), self.events)
        self.assertTrue(all(self.runtime.channels[ch]["arm"].enabled for ch in ("can0", "can1")))
        self.assertFalse(any(e[1] in ("disable", "move", "move_j", "move_js") for e in self.events))

    def test_false_disable_success_cannot_enter_teaching(self) -> None:
        attach_fake_pair(self.runtime, self.events)
        gripper = self.runtime.channels["can0"]["gripper"]
        # SDK 返回成功，但持续更新的反馈仍显示使能，不能切换示教。
        with (
            patch.object(gripper, "disable_gripper", return_value=True),
            self.assertRaisesRegex(RuntimeError, "主左.*can0.*新反馈仍显示使能"),
        ):
            devices.transition_to_collection(self.runtime)
        self.assertFalse(any(e[1] in ("teaching", "leader", "disable", "move") for e in self.events))
        self.assertNotIn(("从左", "enable"), self.events)

    def test_release_requires_fresh_and_valid_gripper_feedback(self) -> None:
        for value in (
            None,
            SimpleNamespace(
                timestamp=1.0, msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=False))
            ),
            SimpleNamespace(
                timestamp=0.0, msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=False))
            ),
        ):
            with self.subTest(feedback=value):
                attach_fake_pair(self.runtime, self.events)
                self.events.clear()
                gripper = self.runtime.channels["can0"]["gripper"]
                with (
                    patch.object(gripper, "get_gripper_status", return_value=value),
                    self.assertRaisesRegex(RuntimeError, "夹爪准备失败.*失能确认超时"),
                ):
                    devices.transition_to_collection(self.runtime)
                self.assertFalse(any(e[1] in ("teaching", "leader", "disable", "move") for e in self.events))
                self.assertTrue(self.runtime.channels["can0"]["arm"].enabled)

    def test_unconfirmed_teaching_parameters_stop_before_leader_mode(self) -> None:
        for result in (False, None):
            with self.subTest(result=result):
                attach_fake_pair(self.runtime, self.events)
                self.events.clear()
                gripper = self.runtime.channels["can0"]["gripper"]
                with (
                    patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=result),
                    self.assertRaisesRegex(RuntimeError, "夹爪示教参数未确认"),
                ):
                    devices.transition_to_collection(self.runtime)
                self.assertFalse(any(e[1] in ("leader", "disable", "move", "move_js") for e in self.events))
                self.assertNotIn(("从左", "enable"), self.events)
                self.assertFalse(gripper.enabled)
                self.assertTrue(self.runtime.channels["can0"]["arm"].enabled)

    def test_missing_set_ack_accepts_matching_fresh_parameter_readback(self) -> None:
        for old in (None, teaching_feedback(1.0, teaching_friction=1)):
            with self.subTest(previous=old):
                attach_fake_pair(self.runtime, self.events)
                self.events.clear()
                gripper = self.runtime.channels["can0"]["gripper"]
                with (
                    patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=False),
                    patch.object(
                        gripper,
                        "get_gripper_teaching_pendant_param",
                        side_effect=[old, teaching_feedback(2.0)],
                    ) as query,
                ):
                    devices.transition_to_collection(self.runtime)
                self.assertEqual(query.call_count, 2)
                self.assertIn(("主左", "leader"), self.events)
                self.assertEqual(self.events.count(("主左", "disable_gripper")), 1)

    def test_parameter_readback_rejects_stale_missing_and_invalid_timestamps(self) -> None:
        for stamp in (None, 0.0, 1.0, 0.5, float("nan"), float("inf"), True):
            with self.subTest(timestamp=stamp):
                attach_fake_pair(self.runtime, self.events)
                self.events.clear()
                gripper = self.runtime.channels["can0"]["gripper"]
                current = None if stamp is None else teaching_feedback(stamp)
                with (
                    patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=False),
                    patch.object(
                        gripper,
                        "get_gripper_teaching_pendant_param",
                        side_effect=[teaching_feedback(1.0), current],
                    ),
                    self.assertRaisesRegex(RuntimeError, "未收到新的有效参数反馈"),
                ):
                    devices.transition_to_collection(self.runtime)
                self.assertFalse(any(e[1] in ("leader", "disable", "move", "move_js") for e in self.events))

    def test_parameter_readback_rejects_mismatched_and_incomplete_parameters(self) -> None:
        for values in (
            {"teaching_range_per": 150},
            {"max_range_config": 0.1},
            {"teaching_friction": 1},
            {"teaching_friction": None},
            {"max_range_config": float("nan")},
        ):
            with self.subTest(parameters=values):
                attach_fake_pair(self.runtime, self.events)
                self.events.clear()
                gripper = self.runtime.channels["can0"]["gripper"]
                with (
                    patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=False),
                    patch.object(
                        gripper,
                        "get_gripper_teaching_pendant_param",
                        side_effect=[teaching_feedback(1.0), teaching_feedback(2.0, **values)],
                    ),
                    self.assertRaisesRegex(RuntimeError, "读回值不匹配"),
                ):
                    devices.transition_to_collection(self.runtime)
                self.assertFalse(any(e[1] in ("leader", "disable", "move", "move_js") for e in self.events))

    def test_readback_transport_error_does_not_enter_teaching(self) -> None:
        attach_fake_pair(self.runtime, self.events)
        gripper = self.runtime.channels["can0"]["gripper"]
        with (
            patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=False),
            patch.object(gripper, "get_gripper_teaching_pendant_param", side_effect=OSError("CAN 查询失败")),
            self.assertRaisesRegex(RuntimeError, "夹爪准备失败.*CAN 查询失败"),
        ):
            devices.transition_to_collection(self.runtime)
        self.assertNotIn(("主左", "leader"), self.events)

    def test_installed_sdk_missing_ack_with_real_parameter_payload(self) -> None:
        # 使用已安装 SDK 的设置/查询/解析实现，只替换传输；不创建 CAN socket。
        context = DriverContext({})
        gripper = GripperDriver({}, context)
        sent = []

        def send(frame: can.Message) -> None:
            sent.append(frame)
            if frame.arbitration_id == 0x477:
                gripper._parser.parse_packet(
                    can.Message(
                        arbitration_id=0x47E,
                        data=bytes.fromhex("6446080000000000"),
                        timestamp=float(len(sent)),
                        is_extended_id=False,
                    )
                )
            # 固件已保存参数，但不返回 0x476 设置 ACK。

        context.comm = SimpleNamespace(send=send, last_error=None)
        self.assertTrue(devices._set_gripper_teaching(gripper))
        self.assertEqual([frame.arbitration_id for frame in sent], [0x47D, 0x477, 0x477])
        self.assertEqual(bytes(sent[0].data), bytes.fromhex("6446080000000000"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
