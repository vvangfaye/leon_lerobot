"""结束采集并失能的模拟检查；不连接硬件，不使用正式数据集。"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform import camera_io, cameras, devices, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_home import HomeArm
from test_platform import Clock, FakeArm, FakeCamera, FakeGripper, MemoryDataset


def feedback(timestamp: float, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=timestamp,
        msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=enabled)),
    )


class DisableArm(FakeArm):
    def __init__(self, name: str, events: list):
        super().__init__(name, events)
        self.timestamp = 1.0
        self.enabled = True
        self.accept_disable = True
        self.fresh = True
        self.missing = False

    def disable(self, timeout: float = 0) -> bool:
        self.events.append((self.name, "disable"))
        if self.fresh:
            self.timestamp += 1
        if self.accept_disable:
            self.enabled = False
        return True  # 即使 SDK 报成功，也必须检查反馈。

    def get_driver_states(self, joint_index: int) -> SimpleNamespace | None:
        if self.fresh:
            self.timestamp += 1
        return None if self.missing else feedback(self.timestamp, self.enabled)


class DisableGripper(FakeGripper):
    def __init__(self, name: str, events: list):
        super().__init__(name, events)
        self.timestamp = 1.0
        self.enabled = True
        self.accept_disable = True
        self.fresh = True

    def disable_gripper(self) -> bool:
        self.events.append((self.name, "disable_gripper"))
        if self.fresh:
            self.timestamp += 1
        if self.accept_disable:
            self.enabled = False
        return True

    def get_gripper_status(self) -> SimpleNamespace:
        if self.fresh:
            self.timestamp += 1
        result = feedback(self.timestamp, self.enabled)
        result.msg.value = 0.02
        return result


class DisableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = []
        self.runtime = Runtime()
        self.runtime.state.update(connected=True, episode_count=3)
        self.runtime.dataset = MemoryDataset(self.events)
        self.runtime.collection_cams["left"] = FakeCamera(events=self.events)
        for ch, label in zip(self.runtime.channels, ("主左", "从左", "主右", "从右"), strict=True):
            self.runtime.channels[ch].update(
                arm=DisableArm(label, self.events),
                gripper=DisableGripper(label, self.events),
                label=label,
                detected=True,
            )
        self.client = create_app(self.runtime).test_client()
        self.payload = {"targets": {"can0": "主左", "can1": "从左"}}
        clock = Clock(self.events)
        for guard in (
            patch.object(devices, "time", clock),
            patch.object(recorder, "time", clock),
            patch.object(cameras, "open_preview_cams"),
            patch.object(cameras, "detect_cameras", return_value={}),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def disable(self):
        return self.client.post("/api/disconnect", json=self.payload)

    def test_end_session_keeps_motors_enabled_and_does_not_move(self) -> None:
        response = self.client.post("/api/collection/end", json=self.payload)
        self.assertTrue(response.json["ok"], response.json)
        self.assertIn(("dataset", "finalize"), self.events)
        self.assertFalse(
            any(
                event[1] in ("disable", "disable_gripper", "enable", "move_js", "move_j", "move")
                for event in self.events
            )
        )
        self.assertFalse(self.runtime.state["connected"])
        self.assertIsNone(self.runtime.dataset)
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIn("未执行机械臂失能", self.runtime.state["end_status"])
        for ch in self.payload["targets"]:
            self.assertTrue(self.runtime.channels[ch]["arm"].enabled)
            self.assertTrue(self.runtime.channels[ch]["gripper"].enabled)
            self.assertEqual(self.runtime.state["close_result"]["arms"][ch]["state"], "not_requested")

    def test_disable_pair_requires_end_session_and_never_repeats_data_cleanup(self) -> None:
        response = self.client.post("/api/pair/disable", json=self.payload)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.events, [])
        self.assertTrue(self.client.post("/api/collection/end", json=self.payload).json["ok"])
        self.events.clear()
        with patch.object(cameras, "open_preview_cams") as preview:
            response = self.client.post("/api/pair/disable", json=self.payload)
        self.assertTrue(response.json["ok"], response.json)
        preview.assert_not_called()
        self.assertFalse(any(event[0] in ("dataset", "camera") for event in self.events))
        for ch in self.payload["targets"]:
            self.assertFalse(self.runtime.channels[ch]["arm"].enabled)
            self.assertFalse(self.runtime.channels[ch]["gripper"].enabled)

    def test_end_data_failure_still_allows_separate_disable_without_retrying_data(self) -> None:
        dataset = self.runtime.dataset
        with patch.object(dataset, "finalize", side_effect=OSError("磁盘错误")) as finalize:
            self.assertFalse(self.client.post("/api/collection/end", json=self.payload).json["ok"])
            self.assertFalse(any(event[1] in ("disable", "disable_gripper") for event in self.events))
            self.assertIsNone(self.runtime.state["disable_error"])
            self.assertTrue(self.client.post("/api/pair/disable", json=self.payload).json["ok"])
            self.assertEqual(finalize.call_count, 1)
        self.assertIs(self.runtime.dataset, dataset)
        self.assertIn("磁盘错误", self.runtime.state["close_error"])
        self.events.clear()
        self.assertTrue(self.retry("data").json["ok"])
        self.assertEqual(self.events, [("dataset", "finalize")])
        self.assertIsNone(self.runtime.state["close_error"])

    def test_end_data_retry_does_not_disable_unrequested_arms(self) -> None:
        with patch.object(self.runtime.dataset, "finalize", side_effect=OSError("磁盘错误")):
            self.assertFalse(self.client.post("/api/collection/end", json=self.payload).json["ok"])
        self.events.clear()
        self.assertTrue(self.retry("data").json["ok"])
        self.assertEqual(self.events, [("dataset", "finalize")])
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertEqual(self.runtime.state["close_result"]["arms"]["can0"]["state"], "not_requested")

    def test_ending_blocks_parallel_motion_config_start_and_disable(self) -> None:
        self.runtime.state["ending"] = True
        for url, payload in (
            ("/api/collection/end", self.payload),
            ("/api/pair/disable", self.payload),
            ("/api/pair/home", self.payload),
            ("/api/record/start", {}),
            ("/api/arm/move/can0", {}),
            ("/api/label/can0", {"label": "主右"}),
            ("/api/save", {"fps": 15}),
            ("/api/connect", {}),
        ):
            self.assertFalse(self.client.post(url, json=payload).json["ok"], url)
        self.assertEqual(self.events, [])

    def retry(self, stage: str, channel: str | None = None):
        return self.client.post("/api/close/retry", json={**self.payload, "stage": stage, "channel": channel})

    def test_selected_pair_only_saved_before_disable_without_motion(self) -> None:
        self.assertTrue(self.disable().json["ok"])
        for label in ("主左", "从左"):
            self.assertIn((label, "disable"), self.events)
            self.assertIn((label, "disable_gripper"), self.events)
            self.assertLess(self.events.index(("dataset", "finalize")), self.events.index((label, "disable")))
            self.assertLess(
                self.events.index((label, "disable_gripper")), self.events.index((label, "disable"))
            )
        self.assertFalse(any(event[0] in ("主右", "从右") for event in self.events))
        self.assertFalse(any(event[1] in ("enable", "move_j", "move_js", "move") for event in self.events))
        self.assertIsNone(self.runtime.dataset)
        self.assertFalse(self.runtime.state["connected"])
        self.assertEqual(self.runtime.state["episode_count"], 3)
        self.assertIn("已失能", self.client.get("/api/status").json["disable_status"])
        self.assertTrue(self.runtime.channels["can0"]["detected"])

    def test_gripper_controller_ignores_disable_after_arm_is_disabled(self) -> None:
        # 模拟 Nero 控制器：整臂失能后忽略夹爪控制。旧的关节优先顺序会一直超时。
        arm = self.runtime.channels["can1"]["arm"]
        gripper = self.runtime.channels["can1"]["gripper"]
        original = gripper.disable_gripper

        def disable_gripper() -> bool:
            gripper.accept_disable = arm.enabled
            return original()

        with patch.object(gripper, "disable_gripper", side_effect=disable_gripper):
            self.assertTrue(self.disable().json["ok"])
        self.assertFalse(gripper.enabled)
        self.assertFalse(arm.enabled)
        self.assertEqual(self.runtime.state["close_result"]["arms"]["can1"]["state"], "done")

    def test_late_gripper_disable_is_reconciled_in_same_close_operation(self) -> None:
        arm = self.runtime.channels["can1"]["arm"]
        gripper = self.runtime.channels["can1"]["gripper"]
        gripper.accept_disable = False
        original = arm.disable

        def disable_arm(timeout: float = 0) -> bool:
            result = original(timeout)
            # 初次等待夹爪超时，之后才收到失能回报；无需用户再次点只读复核。
            gripper.enabled = False
            return result

        with patch.object(arm, "disable", side_effect=disable_arm):
            self.assertTrue(self.disable().json["ok"])
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertEqual(self.runtime.channels["can1"]["disable_result"]["gripper"], "disabled")
        joint_disabled_at = self.events.index(("从左", "disable"))
        self.assertNotIn(("从左", "disable_gripper"), self.events[joint_disabled_at:])
        self.assertFalse(any(event[1] in ("enable", "move", "move_j", "move_js") for event in self.events))

    def test_final_feedback_reenable_or_staleness_keeps_failure_visible(self) -> None:
        arm = self.runtime.channels["can1"]["arm"]
        gripper = self.runtime.channels["can1"]["gripper"]
        original = arm.disable
        for enabled, fresh, reason in ((True, True, "仍显示使能"), (False, False, "时间戳未更新")):
            with self.subTest(enabled=enabled, fresh=fresh):
                gripper.fresh = True

                def disable_arm(timeout: float = 0, enabled: bool = enabled, fresh: bool = fresh) -> bool:
                    result = original(timeout)
                    gripper.enabled, gripper.fresh = enabled, fresh
                    return result

                with patch.object(arm, "disable", side_effect=disable_arm):
                    response = (
                        self.retry("arm", "can1") if self.runtime.state["close_result"] else self.disable()
                    )
                self.assertFalse(response.json["ok"])
                self.assertIn(reason, self.runtime.state["disable_error"])
                self.assertEqual(self.runtime.channels["can1"]["disable_result"]["gripper"], "unknown")

    def test_gripper_read_error_still_disables_joints_and_other_arm(self) -> None:
        with patch.object(
            self.runtime.channels["can0"]["gripper"],
            "get_gripper_status",
            side_effect=OSError("夹爪通信错误"),
        ):
            self.assertFalse(self.disable().json["ok"])
        self.assertIn(("主左", "disable"), self.events)
        self.assertIn(("从左", "disable_gripper"), self.events)
        self.assertIn(("从左", "disable"), self.events)
        self.assertIn("夹爪通信错误", self.runtime.state["disable_error"])

    def test_waits_for_record_save_and_rejects_concurrent_commands(self) -> None:
        saving = threading.Event()
        release = threading.Event()

        def save() -> None:
            saving.set()
            if not release.wait(3):
                raise TimeoutError("测试等待保存超时")
            self.runtime.dataset.save_episode()

        self.runtime.state["recording"] = True
        self.runtime.record_thread = threading.Thread(target=save)
        self.runtime.record_thread.start()
        self.assertTrue(saving.wait(2))
        # 在 join 开始时通知测试，保证禁用检查发生于收尾过程中。
        joining = threading.Event()
        original_join = self.runtime.record_thread.join

        def join() -> None:
            joining.set()
            original_join()

        with patch.object(self.runtime.record_thread, "join", side_effect=join):
            ending = threading.Thread(
                target=recorder.end_collection_and_disable, args=(self.runtime, self.payload["targets"])
            )
            ending.start()
            try:
                self.assertTrue(joining.wait(2))
                self.assertTrue(self.client.get("/api/status").json["disabling"])
                self.assertNotIn(("主左", "disable"), self.events)
                for url, data in (
                    ("/api/disconnect", self.payload),
                    ("/api/record/start", {}),
                    ("/api/connect", {}),
                    ("/api/pair/home", self.payload),
                    ("/api/save", {"fps": 15}),
                    ("/api/label/can0", {"label": "主右"}),
                    ("/api/arm/move/can0", {}),
                ):
                    self.assertFalse(self.client.post(url, json=data).json["ok"], url)
            finally:
                release.set()
                ending.join(5)
        self.assertFalse(ending.is_alive())
        self.assertLess(
            self.events.index(("dataset", "save_episode")), self.events.index(("dataset", "finalize"))
        )
        self.assertLess(self.events.index(("dataset", "finalize")), self.events.index(("主左", "disable")))

    def test_failed_joint_still_disables_gripper_and_other_arm_and_can_retry(self) -> None:
        arm = self.runtime.channels["can0"]["arm"]
        arm.accept_disable = False
        self.assertEqual(self.disable().status_code, 500)
        self.assertIn("关节", self.runtime.state["disable_error"])
        self.assertIn(("主左", "disable_gripper"), self.events)
        self.assertIn(("从左", "disable"), self.events)
        self.assertFalse(recorder.start_recording(self.runtime)[0])
        self.assertEqual(self.client.post("/api/label/can0", json={"label": "主右"}).status_code, 409)
        arm.accept_disable = True
        self.events.clear()
        self.assertTrue(self.retry("arm", "can0").json["ok"])
        self.assertFalse(any(e[0] in ("dataset", "从左") for e in self.events))
        self.assertFalse(any(e[1] in ("enable", "move_j", "move_js", "move") for e in self.events))
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIsNone(self.runtime.state["last_error"])

    def test_missing_or_stale_feedback_never_reports_success(self) -> None:
        arm = self.runtime.channels["can0"]["arm"]
        for missing in (True, False):
            with self.subTest(missing=missing):
                arm.missing = missing
                arm.fresh = False
                arm.enabled = False
                response = self.retry("arm", "can0") if self.runtime.state["close_result"] else self.disable()
                self.assertFalse(response.json["ok"])
                self.assertIn("失能确认超时", self.runtime.state["disable_error"])
                self.assertIn(
                    "未收到有效反馈" if missing else "时间戳未更新", self.runtime.state["disable_error"]
                )

    def test_gripper_failure_is_reported(self) -> None:
        self.runtime.channels["can1"]["gripper"].accept_disable = False
        self.assertFalse(self.disable().json["ok"])
        self.assertIn("夹爪", self.runtime.state["disable_error"])
        self.assertIn("失能未确认", self.runtime.channels["can1"]["status"])
        self.assertIn("新反馈仍显示使能", self.runtime.state["disable_error"])
        self.assertEqual(self.runtime.state["disable_error"].count("can1"), 1)

    def test_recheck_late_disable_feedback_without_commands_or_data_writes(self) -> None:
        # 同时覆盖主臂：只读复核不能再次退出示教或发送任何失能命令。
        for ch in ("can0", "can1"):
            self.runtime.channels[ch]["gripper"].accept_disable = False
        self.assertFalse(self.disable().json["ok"])
        for ch in ("can0", "can1"):
            self.runtime.channels[ch]["gripper"].enabled = False
        self.events.clear()
        self.assertFalse(self.retry("status", "can0").json["ok"])  # 另一条仍待确认。
        self.assertEqual(self.runtime.state["close_result"]["arms"]["can0"]["state"], "done")
        self.assertTrue(self.retry("status", "can1").json["ok"])
        self.assertEqual(self.events, [])
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIsNone(self.runtime.state["last_error"])
        self.assertIn("新反馈已确认", self.runtime.channels["can1"]["status"])

    def test_recheck_still_enabled_or_stale_gripper_remains_unconfirmed(self) -> None:
        gripper = self.runtime.channels["can1"]["gripper"]
        gripper.accept_disable = False
        self.assertFalse(self.disable().json["ok"])
        for enabled, fresh, reason in ((True, True, "仍显示使能"), (False, False, "时间戳未更新")):
            with self.subTest(enabled=enabled, fresh=fresh):
                gripper.enabled, gripper.fresh = enabled, fresh
                self.events.clear()
                self.assertFalse(self.retry("status", "can1").json["ok"])
                self.assertTrue(all(event[0] == "sleep" for event in self.events))
                result = self.runtime.state["close_result"]["arms"]["can1"]
                self.assertEqual(result["details"]["joints"], "disabled")
                self.assertEqual(result["details"]["gripper"], "unknown")
                self.assertIn(reason, result["message"])

    def test_recheck_rejects_active_busy_completed_or_changed_targets(self) -> None:
        gripper = self.runtime.channels["can1"]["gripper"]
        gripper.accept_disable = False
        self.disable()
        self.events.clear()
        for key in ("connected", "recording", "disabling", "homing"):
            self.runtime.state[key] = True
            self.assertFalse(self.retry("status", "can1").json["ok"])
            self.runtime.state[key] = False
        self.assertFalse(self.retry("status", "can0").json["ok"])
        self.payload["targets"] = {"can2": "主右", "can3": "从右"}
        self.assertFalse(self.retry("status", "can1").json["ok"])
        self.assertEqual(self.events, [])

    def test_finalize_failure_keeps_dataset_for_retry_but_attempts_disable(self) -> None:
        dataset = self.runtime.dataset
        with patch.object(dataset, "finalize", side_effect=OSError("磁盘错误")):
            self.assertFalse(self.disable().json["ok"])
        self.assertIs(self.runtime.dataset, dataset)
        self.assertIn(("主左", "disable"), self.events)
        self.assertIn(("从左", "disable"), self.events)
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIn("磁盘错误", self.runtime.state["close_error"])
        self.assertFalse(recorder.do_connect(self.runtime)[0])
        self.events.clear()
        self.assertTrue(self.retry("data").json["ok"])
        self.assertEqual(self.events, [("dataset", "finalize")])
        self.assertIsNone(self.runtime.dataset)

    def test_mode_failure_does_not_skip_disable(self) -> None:
        with patch.object(
            self.runtime.channels["can0"]["arm"], "set_follower_mode", side_effect=OSError("模式错误")
        ):
            self.assertTrue(self.disable().json["ok"])
        self.assertIn(("主左", "disable"), self.events)
        self.assertIn(("从左", "disable"), self.events)
        self.assertIn("模式错误", self.runtime.state["close_result"]["warnings"][0])

    def test_stale_confirmation_and_busy_rejected_without_actions(self) -> None:
        self.payload["targets"] = {"can2": "主右", "can3": "从右"}
        self.assertEqual(self.disable().status_code, 409)
        self.assertEqual(self.events, [])
        self.runtime.state["homing"] = True
        self.assertEqual(self.disable().status_code, 409)
        self.assertEqual(self.events, [])

    def test_no_active_session_can_still_disable_connected_pair(self) -> None:
        self.runtime.state["connected"] = False
        self.runtime.dataset = None
        self.assertTrue(self.disable().json["ok"])
        self.assertIn(("从左", "disable"), self.events)

    def test_shutdown_cleanup_keeps_existing_torque_behavior(self) -> None:
        recorder.do_disconnect(self.runtime, reopen_preview=False)
        self.assertFalse(any(event[1] in ("disable", "disable_gripper") for event in self.events))

    def test_restarting_updates_enabled_status_and_uses_actual_alignment(self) -> None:
        self.assertTrue(self.disable().json["ok"])
        # 退出示教之后，旧主臂广播与实际姿态可能不同，检查实际关节反馈。
        self.assertEqual(devices.check_alignment(self.runtime), (True, ""))
        devices.transition_to_collection(self.runtime)
        for ch in ("can0", "can1"):
            self.assertIn("已使能", self.runtime.channels[ch]["status"])

    def test_episode_save_failure_is_reported_and_still_disables(self) -> None:
        self.runtime.record_thread = threading.Thread(target=lambda: None)
        self.runtime.record_thread.start()
        self.runtime.state["last_error"] = "本集保存失败: 磁盘错误"
        self.runtime.state["writer_error"] = "本集保存失败: 磁盘错误"
        self.assertFalse(self.disable().json["ok"])
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIn("本集保存失败", self.runtime.state["close_error"])
        self.assertEqual(self.runtime.state["close_result"]["data"]["state"], "needs_review")
        self.assertIn(("从左", "disable"), self.events)
        self.events.clear()
        self.assertFalse(self.retry("data").json["ok"])
        self.assertEqual(self.events, [])

    def test_historical_alignment_error_is_not_a_data_or_disable_failure(self) -> None:
        self.runtime.state["last_error"] = "主从对齐失败，尚未录制"
        self.assertTrue(self.disable().json["ok"])
        self.assertIsNone(self.runtime.state["disable_error"])
        self.assertIsNone(self.runtime.state["close_error"])

    def test_preview_failure_is_separate_from_data_and_disable_results(self) -> None:
        with patch.object(cameras, "open_preview_cams", side_effect=OSError("相机忙")):
            self.assertTrue(self.disable().json["ok"])
        result = self.runtime.state["close_result"]
        self.assertEqual(result["data"]["state"], "done")
        self.assertTrue(all(entry["state"] == "done" for entry in result["arms"].values()))
        self.assertEqual(result["preview"]["state"], "warning")
        self.assertIn("相机忙", result["preview"]["message"])

    def test_failed_gripper_does_not_hide_successful_joint_disable(self) -> None:
        self.runtime.channels["can1"]["gripper"].accept_disable = False
        self.disable()
        result = self.runtime.state["close_result"]["arms"]["can1"]
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["details"]["joints"], "disabled")
        self.assertEqual(result["details"]["gripper"], "unknown")

    def test_retry_refuses_completed_or_stale_targets_without_commands(self) -> None:
        self.assertTrue(self.disable().json["ok"])
        self.events.clear()
        self.assertFalse(self.retry("arm", "can0").json["ok"])
        self.assertFalse(self.retry("data").json["ok"])
        self.assertFalse(self.retry("bad").json["ok"])
        self.payload["targets"] = {"can2": "主右", "can3": "从右"}
        self.assertFalse(self.retry("arm", "can2").json["ok"])
        self.assertEqual(self.events, [])

    def test_data_retry_keeps_unrelated_arm_failure_visible(self) -> None:
        self.runtime.channels["can1"]["arm"].accept_disable = False
        with patch.object(self.runtime.dataset, "finalize", side_effect=OSError("磁盘错误")):
            self.disable()
        self.events.clear()
        self.assertFalse(self.retry("data").json["ok"])
        self.assertEqual(self.events, [("dataset", "finalize")])
        self.assertIsNone(self.runtime.state["close_error"])
        self.assertIn("can1", self.runtime.state["disable_error"])

    def test_pending_data_blocks_motion_and_config_but_not_status(self) -> None:
        with patch.object(self.runtime.dataset, "finalize", side_effect=OSError("磁盘错误")):
            self.disable()
        self.events.clear()
        for url, payload in (
            ("/api/record/start", {}),
            ("/api/pair/home", self.payload),
            ("/api/arm/move/can0", {}),
            ("/api/save", {}),
            ("/api/label/can0", {}),
            ("/api/disconnect", self.payload),
        ):
            self.assertFalse(self.client.post(url, json=payload).json["ok"], url)
        self.assertEqual(self.events, [])
        self.assertTrue(self.client.get("/api/arm/status/can0").json["ok"])

    def test_single_arm_home_confirms_arrival_then_disable(self) -> None:
        arm = HomeArm("主左", self.events)
        self.runtime.channels["can0"]["arm"] = arm

        def disable(timeout: float = 0) -> bool:
            self.events.append(("主左", "disable"))
            arm.enabled = False
            return True

        with patch.object(arm, "disable", create=True, side_effect=disable):
            devices.home_and_disable(self.runtime, "can0")
        self.assertIn("已失能", self.runtime.channels["can0"]["status"])
        self.assertLess(
            self.events.index(("主左", "move_j", devices.HOME_POS)), self.events.index(("主左", "disable"))
        )

    def test_single_arm_sdk_success_without_disabled_feedback_is_not_success(self) -> None:
        arm = HomeArm("主左", self.events)
        self.runtime.channels["can0"]["arm"] = arm
        with patch.object(arm, "disable", create=True, return_value=True):
            devices.home_and_disable(self.runtime, "can0")
        self.assertIn("失能未完成", self.runtime.channels["can0"]["status"])
        self.assertIn("新反馈仍显示使能", self.runtime.channels["can0"]["status"])

    def test_single_arm_does_not_disable_before_arrival(self) -> None:
        arm = HomeArm("主左", self.events)
        arm.reach_home = False
        self.runtime.channels["can0"]["arm"] = arm
        with patch.object(arm, "disable", create=True) as disable:
            devices.home_and_disable(self.runtime, "can0")
            disable.assert_not_called()
        self.assertIn("默认位置超时", self.runtime.channels["can0"]["status"])
        self.assertNotIn(("主左", "disable_gripper"), self.events)


class DisableFeedbackTests(unittest.TestCase):
    def test_async_reply_checked_before_resending_including_timeout_boundary(self) -> None:
        for arrival in (0.01, devices.DISABLE_TIMEOUT_SECONDS):
            with self.subTest(arrival=arrival):
                clock = Clock([])
                commands = []
                cached = feedback(1.0, True)

                def read(
                    clock: Clock = clock, arrival: float = arrival, cached: SimpleNamespace = cached
                ) -> list:
                    if clock.now >= arrival:
                        cached.timestamp = 2.0
                        cached.msg.foc_status.driver_enable_status = False
                    return [cached]

                with patch.object(devices, "time", clock):
                    devices._disable_and_confirm(
                        lambda commands=commands, clock=clock: commands.append(clock.now), read
                    )
                self.assertLess(commands[-1], arrival)
                if arrival == 0.01:
                    self.assertEqual(commands, [0.0])
                else:
                    self.assertEqual(clock.now, devices.DISABLE_TIMEOUT_SECONDS)

    def test_invalid_timestamps_never_confirm_disabled(self) -> None:
        for stamp in (None, 0, -1, float("nan"), float("inf")):
            with (
                self.subTest(stamp=stamp),
                patch.object(devices, "time", Clock([])),
                self.assertRaisesRegex(TimeoutError, "未收到有效反馈"),
            ):
                devices._disable_and_confirm(None, lambda stamp=stamp: [feedback(stamp, False)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
