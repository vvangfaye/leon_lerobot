"""只使用模拟设备验证取消边界，禁止连接真实硬件。"""

import unittest
from unittest.mock import patch

import test_recording
from nero_platform import recorder


class CancelRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = test_recording.RecordingTests(
            "test_manual_stop_saves_exactly_once_after_current_frame"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime = self.fixture.runtime
        self.client = self.fixture.client
        self.cleared = 0

        def clear() -> None:
            self.cleared += 1
            self.runtime.dataset.frames.clear()

        guard = patch.object(self.runtime.dataset, "clear_episode_buffer", side_effect=clear, create=True)
        guard.start()
        self.addCleanup(guard.stop)

    def test_cancel_at_save_boundary_discards_and_can_record_next_episode(self) -> None:
        pending = self.runtime.dataset.has_pending_frames

        def cancel() -> bool:
            self.assertTrue(recorder.request_cancel(self.runtime)[0])
            self.assertTrue(recorder.request_cancel(self.runtime)[0])
            recorder.request_stop(self.runtime)
            self.assertEqual(self.runtime.state["record_phase"], "cancelling")
            return pending()

        with patch.object(self.runtime.dataset, "has_pending_frames", side_effect=cancel):
            recorder.record_worker(self.runtime, ["right"])
        self.assertEqual(self.cleared, 1)
        self.assertEqual(self.runtime.state["episode_count"], 0)
        self.assertEqual(self.runtime.state["record_phase"], "cancelled")
        self.assertFalse(self.runtime.state["cancel_requested"])
        self.assertNotIn(("dataset", "save_episode"), self.fixture.events)
        self.assertEqual(self.client.post("/api/record/start").status_code, 202)
        self.fixture.finish()
        self.assertEqual(self.runtime.state["episode_count"], 1)

    def test_cancel_rejected_once_save_has_started(self) -> None:
        save = self.runtime.dataset.save_episode

        def saving() -> None:
            self.assertEqual(self.client.post("/api/record/cancel").status_code, 409)
            save()

        with patch.object(self.runtime.dataset, "save_episode", side_effect=saving):
            recorder.record_worker(self.runtime, ["right"])
        self.assertEqual(self.cleared, 0)
        self.assertEqual(self.runtime.state["episode_count"], 1)

    def test_cancel_cannot_override_manual_save_or_preparation(self) -> None:
        for phase in ("idle", "preparing", "aligning", "stopping", "saving", "saved", "cancelled"):
            with self.subTest(phase=phase):
                self.runtime.state.update(recording=True, record_phase=phase)
                self.assertEqual(self.client.post("/api/record/cancel").status_code, 409)
                self.assertFalse(self.runtime.state["cancel_requested"])

    def test_cleanup_failure_blocks_next_recording(self) -> None:
        pending = self.runtime.dataset.has_pending_frames

        def cancel() -> bool:
            recorder.request_cancel(self.runtime)
            return pending()

        with (
            patch.object(self.runtime.dataset, "has_pending_frames", side_effect=cancel),
            patch.object(self.runtime.dataset, "clear_episode_buffer", side_effect=OSError("暂存帧清理失败")),
        ):
            recorder.record_worker(self.runtime, ["right"])
        self.assertEqual(self.runtime.state["record_phase"], "error")
        self.assertFalse(self.runtime.state["cancel_requested"])
        self.assertEqual(self.client.post("/api/record/start").status_code, 409)
        self.assertNotIn(("dataset", "save_episode"), self.fixture.events)


if __name__ == "__main__":
    unittest.main()
