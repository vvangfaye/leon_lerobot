"""返回默认位置的模拟硬件检查，不连接机械臂或相机。"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform import camera_io, devices, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeArm, MemoryDataset, attach_fake_pair


class HomeArm(FakeArm):
    def __init__(self, name: str, events: list):
        super().__init__(name, events)
        self.positions = [0.0] * 7
        self.reach_home = True
        self.teaching = True

    def get_joint_angles(self) -> SimpleNamespace:
        return self.feedback(self.positions[:])

    def get_leader_joint_angles(self) -> SimpleNamespace:
        if not self.teaching:
            return super().get_leader_joint_angles()
        self.events.append((self.name, "read_leader"))
        return self.feedback(self.positions[:])

    def set_leader_mode(self) -> None:
        super().set_leader_mode()
        self.teaching = True

    def set_follower_mode(self) -> None:
        super().set_follower_mode()
        self.teaching = False

    def get_arm_status(self) -> SimpleNamespace:
        return super().get_arm_status()

    def move_j(self, target: list[float]) -> None:
        self.events.append((self.name, "move_j", target[:]))
        if self.reach_home:
            self.positions = target[:]

    def move_js(self, target: list[float]) -> None:
        super().move_js(target)
        self.positions = target[:]


class HomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = []
        self.runtime = Runtime()
        attach_fake_pair(self.runtime, self.events)
        self.arm = HomeArm("从左", self.events)
        self.leader = HomeArm("主左", self.events)
        self.runtime.channels["can0"]["arm"] = self.leader
        self.runtime.channels["can1"]["arm"] = self.arm
        self.runtime.state.update(connected=True, episode_count=10)
        self.runtime.dataset = MemoryDataset(self.events)
        self.client = create_app(self.runtime).test_client()
        self.clock = Clock(self.events)
        clock_patch = patch.object(devices, "time", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.payload = {"targets": {"can0": "主左", "can1": "从左"}}
        for guard in (
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def finish_home(self) -> None:
        self.runtime.home_thread.join(3)
        self.assertFalse(self.runtime.home_thread.is_alive())

    def test_selected_pair_moves_in_order_and_dataset_stays_open(self) -> None:
        dataset = self.runtime.dataset
        other = HomeArm("从右", self.events)
        self.runtime.channels["can3"].update(arm=other, label="从右", detected=True)
        response = self.client.post("/api/pair/home", json=self.payload)
        self.assertEqual(response.status_code, 202)
        self.finish_home()
        self.assertIn(("从左", "move_j", devices.HOME_POS), self.events)
        self.assertIn(("主左", "move_j", devices.HOME_POS), self.events)
        self.assertLess(
            self.events.index(("从左", "move_j", devices.HOME_POS)),
            self.events.index(("主左", "move_j", devices.HOME_POS)),
        )
        self.assertFalse(any(event[0] in ("从右", "dataset") for event in self.events))
        self.assertFalse(any(event[1] in ("move", "teaching") for event in self.events))
        self.assertFalse(any("disable" in str(event) for event in self.events))
        self.assertIs(self.runtime.dataset, dataset)
        self.assertEqual(self.runtime.state["episode_count"], 10)
        self.assertTrue(self.runtime.state["connected"])
        self.assertTrue(self.runtime.state["teleop_needs_restore"])
        self.assertFalse(self.runtime.state["homing"])
        self.assertIsNone(self.runtime.state["home_error"])
        self.assertIn("已回默认位置", self.client.get("/api/status").json["home_status"])

    def test_recording_and_device_work_reject_home_without_motion(self) -> None:
        self.runtime.state["recording"] = True
        self.assertEqual(self.client.post("/api/pair/home", json=self.payload).status_code, 409)
        self.runtime.state["recording"] = False
        self.runtime.device_busy.add("can0")
        self.assertEqual(self.client.post("/api/pair/home", json=self.payload).status_code, 409)
        self.assertEqual(self.events, [])

    def test_unknown_stale_or_ambiguous_selection_is_rejected(self) -> None:
        for payload in (
            {"targets": {"can0": "主左", "can3": "从右"}},
            {"targets": {"can0": "从左", "can1": "主左"}},
            {"targets": {"can1": "从左"}},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/pair/home", json=payload).status_code, 409)
        self.runtime.channels["can3"]["label"] = "从左"
        self.assertEqual(self.client.post("/api/pair/home", json=self.payload).status_code, 409)
        self.assertEqual(self.events, [])

    def test_no_feedback_sends_no_enable_or_motion(self) -> None:
        with patch.object(self.arm, "get_joint_angles", return_value=None):
            self.client.post("/api/pair/home", json=self.payload)
            self.finish_home()
        self.assertEqual(self.events, [])
        self.assertIn("读不到有效关节位置", self.runtime.state["home_error"])

    def test_enable_timeout_sends_no_motion(self) -> None:
        with patch.object(self.arm, "enable", return_value=False):
            self.client.post("/api/pair/home", json=self.payload)
            self.finish_home()
        self.assertIn("使能超时", self.runtime.state["home_error"])
        self.assertFalse(any(event[1] in ("move_j", "move_js", "move") for event in self.events))

    def test_idle_status_without_home_position_times_out_then_retry_recovers(self) -> None:
        self.arm.reach_home = False
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        self.assertFalse(self.runtime.state["homing"])
        self.assertIn("默认位置超时", self.runtime.state["home_error"])
        self.assertNotIn(("主左", "move_j", devices.HOME_POS), self.events)
        ok, message = recorder.start_recording(self.runtime)
        self.assertFalse(ok)
        self.assertIn("回位未成功", message)
        self.arm.reach_home = True
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        self.assertIsNone(self.runtime.state["home_error"])

    def test_homing_blocks_repeat_record_disconnect_and_config_commands(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def slow_move(target: list[float]) -> None:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试等待超时")
            self.arm.positions = target[:]

        with patch.object(self.arm, "move_j", side_effect=slow_move):
            self.client.post("/api/pair/home", json=self.payload)
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(self.client.get("/api/status").json["homing"])
                self.assertEqual(self.client.post("/api/pair/home", json=self.payload).status_code, 409)
                self.assertFalse(self.client.post("/api/record/start").json["ok"])
                self.assertFalse(self.client.post("/api/connect").json["ok"])
                self.assertEqual(self.client.post("/api/disconnect").status_code, 409)
                self.assertEqual(self.client.post("/api/arm/move/can1").status_code, 409)
                self.assertEqual(self.client.post("/api/arm/disable/can1").status_code, 409)
                self.assertEqual(self.client.post("/api/save", json={"fps": 15}).status_code, 409)
                self.assertEqual(self.client.post("/api/label/can1", json={"label": "从右"}).status_code, 409)
            finally:
                release.set()
                self.finish_home()

    def test_recording_automatically_aligns_if_leader_moved_after_home(self) -> None:
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        self.leader.positions[0] = 0.3
        with (
            patch.object(recorder, "time", self.clock),
            patch.object(recorder, "record_worker") as record,
        ):
            ok, message = recorder.start_recording(self.runtime)
            self.assertTrue(ok, message)
            self.runtime.record_thread.join(3)
            record.assert_called_once()
        self.assertEqual(self.arm.positions, self.leader.positions)
        self.assertEqual(devices.check_alignment(self.runtime), (True, ""))

    def test_button_is_above_disconnect_and_explains_target(self) -> None:
        html = self.client.get("/").get_data(as_text=True)
        self.assertLess(html.index('id="btn-home"'), html.index('id="btn-disconnect"'))
        self.assertIn("主从臂返回默认位置", html)
        self.assertIn("保持使能和夹爪状态", html)

    def test_alignment_before_teleop_uses_actual_leader_positions(self) -> None:
        self.runtime.state["teleop_needs_restore"] = True
        self.arm.positions = devices.HOME_POS[:]
        leader = self.runtime.channels["can0"]["arm"]
        with (
            patch.object(leader, "get_leader_joint_angles", return_value=None),
            patch.object(leader, "get_joint_angles", return_value=SimpleNamespace(msg=devices.HOME_POS[:])),
        ):
            self.assertEqual(devices.check_alignment(self.runtime), (True, ""))

    def test_resume_restores_teaching_before_recording_without_reopening_dataset(self) -> None:
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        dataset = self.runtime.dataset
        # HomeArm 继承的旧主臂广播与 HOME_POS 不同，必须使用实际反馈做对齐检查。
        with (
            patch.object(recorder, "time", self.clock),
            patch.object(
                recorder, "record_worker", side_effect=lambda *_: self.events.append(("record", "start"))
            ),
        ):
            ok, message = recorder.start_recording(self.runtime)
            self.assertTrue(ok, message)
            self.runtime.record_thread.join(3)
        self.assertLess(self.events.index(("主左", "leader")), self.events.index(("record", "start")))
        self.assertEqual(self.events.count(("主左", "disable_gripper")), 1)
        self.assertLess(self.events.index(("主左", "disable_gripper")), self.events.index(("主左", "leader")))
        self.assertIs(self.runtime.dataset, dataset)
        self.assertNotIn(("dataset", "finalize"), self.events)
        self.assertFalse(self.runtime.state["teleop_needs_restore"])

    def test_second_arm_failure_reports_channel_and_blocks_recording(self) -> None:
        self.leader.reach_home = False
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        self.assertIn("主左 (can0)", self.runtime.state["home_error"])
        self.assertIn("已回默认位置", self.runtime.channels["can1"]["status"])
        self.assertFalse(recorder.start_recording(self.runtime)[0])

    def test_restore_failure_does_not_start_recording(self) -> None:
        self.client.post("/api/pair/home", json=self.payload)
        self.finish_home()
        with (
            patch.object(devices, "transition_to_collection", side_effect=RuntimeError("模式错误")),
            patch.object(recorder.traceback, "print_exc"),
        ):
            self.assertTrue(recorder.start_recording(self.runtime)[0])
            self.runtime.record_thread.join(3)
        self.assertFalse(self.runtime.state["recording"])
        self.assertEqual(self.runtime.state["record_phase"], "error")
        self.assertIn("模式错误", self.runtime.state["last_error"])
        self.assertTrue(self.runtime.state["teleop_needs_restore"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
