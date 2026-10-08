"""第二轮录制状态检查：用事件阻塞模拟准备/保存，验证并发操作不会串入下一集。"""

import threading
import unittest
from unittest.mock import patch

from nero_platform import camera_io, cameras, devices, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair


class RecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Runtime()
        self.events = []
        attach_fake_pair(self.runtime, self.events)
        self.runtime.state.update(connected=True, fps=10, episode_sec=0.25, camera_names=["right"])
        self.runtime.dataset = MemoryDataset(self.events)
        self.runtime.collection_cams["right"] = FakeCamera(events=self.events)
        self.client = create_app(self.runtime).test_client()
        self.releases = []
        self.clock = Clock(self.events)
        for guard in (
            patch.object(recorder, "time", self.clock),
            patch.object(devices, "time", self.clock),
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(cameras, "open_preview_cams"),
            patch.object(recorder.traceback, "print_exc"),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.addCleanup(self.finish)

    def finish(self) -> None:
        for event in self.releases:
            event.set()
        if self.runtime.record_thread is not None:
            self.runtime.record_thread.join(5)
            self.assertFalse(self.runtime.record_thread.is_alive(), "录制线程未结束")

    def gate(self) -> tuple[threading.Event, threading.Event]:
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        return entered, release

    def status(self) -> dict:
        return self.client.get("/api/status").json

    def test_preparation_is_visible_cancellable_and_rejects_other_actions(self) -> None:
        self.runtime.state["connected"] = False
        self.runtime.record_preview = {"right": (object(), object(), 0.0)}
        entered, release = self.gate()

        def prepare(runtime: Runtime) -> tuple[bool, str]:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试准备超时")
            runtime.state["connected"] = True
            return True, "已就绪"

        with patch.object(recorder, "do_connect", side_effect=prepare):
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.runtime.record_preview, {}, "新一集准备时清除上集预览")
            self.assertEqual(self.status()["record"]["phase"], "preparing")
            for url, data in (
                ("/api/record/start", {}),
                ("/api/connect", {}),
                ("/api/save", {"fps": 15}),
                ("/api/label/can0", {"label": "主右"}),
                ("/api/arm/move/can0", {}),
                ("/api/pair/home", {"targets": {"can0": "主左", "can1": "从左"}}),
                ("/api/disconnect", {"targets": {"can0": "主左", "can1": "从左"}}),
            ):
                self.assertFalse(self.client.post(url, json=data).json["ok"], url)
            self.assertTrue(self.client.post("/api/record/stop").json["ok"])
            self.assertTrue(self.client.post("/api/record/stop").json["ok"])
            self.finish()
        state = self.status()
        self.assertFalse(state["recording"])
        self.assertEqual(state["record"]["phase"], "idle")
        self.assertIn("准备已取消", state["record"]["message"])
        self.assertEqual(state["episode_count"], 0)
        self.assertNotIn(("dataset", "save_episode"), self.events)
        self.assertFalse(any(event[1] == "move_js" for event in self.events))

    def test_saving_freezes_elapsed_and_increments_count_only_after_success(self) -> None:
        entered, release = self.gate()
        save_episode = self.runtime.dataset.save_episode

        def save() -> None:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试保存超时")
            save_episode()

        with patch.object(self.runtime.dataset, "save_episode", side_effect=save) as save_mock:
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.assertTrue(entered.wait(2))
            state = self.status()
            self.assertEqual(state["record"]["phase"], "saving")
            self.assertEqual(state["record"]["frames"], 3)
            self.assertEqual(state["episode_count"], 0)
            self.assertIsNone(state["record"]["last_episode"])
            elapsed = state["record"]["elapsed_seconds"]
            self.clock.now += 10
            self.assertEqual(self.status()["record"]["elapsed_seconds"], elapsed)
            for _ in range(3):
                self.assertEqual(self.client.post("/api/record/start").status_code, 409)
                self.assertIn("正在保存", self.client.post("/api/record/stop").json["msg"])
            self.assertEqual(self.client.post("/api/pair/home", json={"targets": {}}).status_code, 409)
            self.assertEqual(self.client.post("/api/save", json={"task": "新任务"}).status_code, 409)
            self.finish()
            self.assertEqual(save_mock.call_count, 1)
        state = self.status()
        self.assertEqual(state["record"]["phase"], "saved")
        self.assertFalse(state["recording"])
        self.assertEqual(state["episode_count"], 1)
        self.assertEqual(state["record"]["last_episode"]["number"], 1)
        self.assertEqual(state["record"]["last_episode"]["frames"], 3)
        self.assertEqual(state["record"]["last_episode"]["cameras"], ["right"])
        self.assertEqual(state["record"]["last_episode"]["elapsed_seconds"], elapsed)
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()
        self.assertEqual(self.status()["episode_count"], 2)
        self.assertEqual(self.status()["record"]["last_episode"]["frames"], 3)

    def test_manual_stop_saves_exactly_once_after_current_frame(self) -> None:
        entered, release = self.gate()
        add_frame = self.runtime.dataset.add_frame

        def add(frame: dict) -> None:
            add_frame(frame)
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试采样超时")

        with patch.object(self.runtime.dataset, "add_frame", side_effect=add):
            self.client.post("/api/record/start")
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.status()["record"]["phase"], "recording")
            self.client.post("/api/record/stop")
            self.assertEqual(self.status()["record"]["phase"], "stopping")
            self.client.post("/api/record/stop")
            self.finish()
        self.assertEqual(self.events.count(("dataset", "save_episode")), 1)
        self.assertEqual(self.status()["record"]["last_episode"]["frames"], 1)
        self.assertEqual(self.status()["episode_count"], 1)
        self.assertEqual(self.status()["record"]["phase"], "saved")
        self.assertFalse(self.status()["record"]["last_episode"]["needs_review"])

    def test_save_failure_blocks_new_episode_and_is_not_cleared_by_disabling(self) -> None:
        with patch.object(self.runtime.dataset, "save_episode", side_effect=OSError("磁盘错误")) as save:
            self.client.post("/api/record/start")
            self.finish()
            state = self.status()
            self.assertEqual(state["record"]["phase"], "error")
            self.assertIn("磁盘错误", state["record"]["writer_error"])
            self.assertEqual(state["episode_count"], 0)
            self.assertIsNone(state["record"]["last_episode"])
            self.assertEqual(self.client.post("/api/record/start").status_code, 409)
            self.assertEqual(save.call_count, 1)
        with patch.object(devices, "disable_channel") as disable:
            self.client.post("/api/disconnect", json={"targets": {"can0": "主左", "can1": "从左"}})
            self.assertEqual(disable.call_count, 2)
        self.assertIn("磁盘错误", self.status()["record"]["writer_error"])
        self.assertFalse(recorder.do_connect(self.runtime)[0])

    def test_add_frame_failure_stops_instead_of_reusing_partially_written_buffer(self) -> None:
        with patch.object(self.runtime.dataset, "add_frame", side_effect=OSError("图像写入失败")) as add:
            self.client.post("/api/record/start")
            self.finish()
        self.assertEqual(add.call_count, 1)
        self.assertNotIn(("dataset", "save_episode"), self.events)
        self.assertEqual(self.status()["record"]["phase"], "error")
        self.assertIn("图像写入失败", self.status()["record"]["writer_error"])
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)

    def test_read_errors_are_counted_and_success_summary_reports_skipped_frames(self) -> None:
        camera = self.runtime.collection_cams["right"]
        frame = camera.frame.copy()
        with patch.object(camera, "read_latest", side_effect=[RuntimeError("相机未就绪"), frame, frame]):
            self.client.post("/api/record/start")
            self.finish()
        state = self.status()
        self.assertEqual(state["record"]["phase"], "saved_review")
        self.assertEqual(state["record"]["frame_errors"], 1)
        self.assertEqual(state["record"]["last_episode"]["frames"], 2)
        self.assertEqual(state["record"]["last_episode"]["frame_errors"], 1)
        self.assertIn("相机未就绪", state["record"]["frame_error"])
        self.assertTrue(state["record"]["last_episode"]["needs_review"])
        self.assertIsNone(state["record"]["writer_error"])
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()
        self.assertEqual(self.status()["record"]["phase"], "saved")
        self.assertEqual(self.status()["episode_count"], 2)

    def test_thread_start_failure_clears_busy_state(self) -> None:
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("无法创建线程")):
            self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assertFalse(self.status()["recording"])
        self.assertEqual(self.status()["record"]["phase"], "error")
        self.assertIsNone(self.runtime.record_thread)
        self.assertTrue(
            all(event[1] in ("read_leader", "read_follower", "read_status") for event in self.events)
        )

    def test_preparation_failure_can_retry_without_incrementing_count(self) -> None:
        self.runtime.state["connected"] = False
        with patch.object(recorder, "do_connect", return_value=(False, "相机未检测到")):
            self.client.post("/api/record/start")
            self.finish()
        self.assertEqual(self.status()["record"]["phase"], "error")
        self.assertIn("相机未检测到", self.status()["last_error"])
        self.assertEqual(self.status()["episode_count"], 0)
        self.assertIsNone(self.status()["record"]["writer_error"])
        self.runtime.state["connected"] = True
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.finish()
        self.assertEqual(self.status()["record"]["phase"], "saved")

    def test_idle_stop_does_not_change_state(self) -> None:
        self.assertTrue(self.client.post("/api/record/stop").json["ok"])
        self.assertEqual(self.status()["record"]["phase"], "idle")
        self.assertFalse(self.runtime.state["stop_requested"])
        self.assertTrue(
            all(event[1] in ("read_leader", "read_follower", "read_status") for event in self.events)
        )

    def test_prepare_enable_has_timeout(self) -> None:
        with (
            patch.object(self.runtime.channels["can0"]["arm"], "enable", return_value=False),
            self.assertRaisesRegex(TimeoutError, "主臂使能超时"),
        ):
            devices.transition_to_collection(self.runtime)
        self.assertFalse(any(event[1] in ("leader", "move_js") for event in self.events))

    def test_next_episodes_do_not_reset_grippers_without_mode_restore(self) -> None:
        with patch.object(recorder, "record_worker") as record:
            for _ in range(2):
                self.assertEqual(self.client.post("/api/record/start").status_code, 202)
                self.finish()
            self.assertEqual(record.call_count, 2)
        self.assertFalse(
            any(e[1] in ("disable_gripper", "teaching", "leader", "disable") for e in self.events)
        )

    def test_gripper_prepare_failure_aborts_restoration_before_recording(self) -> None:
        self.runtime.state["teleop_needs_restore"] = True
        gripper = self.runtime.channels["can0"]["gripper"]
        dataset = self.runtime.dataset
        with (
            patch.object(gripper, "set_gripper_teaching_pendant_param", return_value=False),
            patch.object(recorder, "record_worker") as record,
        ):
            self.assertEqual(self.client.post("/api/record/start").status_code, 202)
            self.finish()
            record.assert_not_called()
        self.assertEqual(self.runtime.state["record_phase"], "error")
        self.assertIn("主左 (can0) 夹爪准备失败", self.runtime.state["last_error"])
        self.assertTrue(self.runtime.state["teleop_needs_restore"])
        self.assertIs(self.runtime.dataset, dataset)
        self.assertFalse(
            any(e[1] in ("leader", "disable", "move", "move_js", "add_frame") for e in self.events)
        )

    def test_unexpected_failure_after_a_frame_also_blocks_reusing_buffer(self) -> None:
        sleep = self.clock.sleep

        def fail_after_frame(duration: float) -> None:
            if ("dataset", "add_frame") in self.events:
                raise RuntimeError("采样循环中止")
            sleep(duration)

        with patch.object(self.clock, "sleep", side_effect=fail_after_frame):
            self.client.post("/api/record/start")
            self.finish()
        self.assertEqual(self.status()["record"]["frames"], 1)
        self.assertIn("已有帧未确认保存", self.status()["record"]["writer_error"])
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)

    def test_shutdown_rejects_new_commands(self) -> None:
        self.runtime.state["shutting_down"] = True
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assertEqual(self.client.post("/api/arm/move/can0").status_code, 409)
        self.assertFalse(recorder.start_recording(self.runtime)[0])
        self.assertIsNone(self.runtime.record_thread)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
