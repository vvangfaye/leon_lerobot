"""RTC scheduling tests with simulated devices; no CAN or cameras are opened."""

import time
import unittest

import numpy as np
import test_inference as baseline
from inference_platform.rtc import RTCBuffer, RTCOptions


class RTCBufferTests(unittest.TestCase):
    @staticmethod
    def planner(actions, anchor):
        return (
            np.concatenate([anchor[None], actions.reshape(-1, *anchor.shape)]),
            np.full(len(actions), 1 / 30),
            [],
        )

    def test_keeps_committed_endpoint_and_discards_already_elapsed_prefix(self):
        queue = RTCBuffer(np.zeros((1, 2)))
        queue.merge(np.array([[1, 1], [2, 2], [3, 3]]), 0, self.planner)
        committed = queue.pop()
        merged, skipped, _ = queue.merge(np.array([[8, 8], [9, 9], [10, 10]]), 0, self.planner)
        self.assertTrue(merged)
        self.assertEqual(skipped, 1)
        np.testing.assert_equal(committed[1], [[1, 1]])
        np.testing.assert_equal(queue.pop()[0], [[1, 1]])
        np.testing.assert_equal(queue.anchor, [[9, 9]])

    def test_stale_result_does_not_replay_executed_actions(self):
        queue = RTCBuffer(np.zeros((1, 2)))
        actions = np.array([[1, 1], [2, 2]])
        queue.merge(actions, 0, self.planner)
        queue.pop()
        queue.pop()
        self.assertEqual(queue.merge(actions, 0, self.planner), (False, 2, []))
        self.assertIsNone(queue.prefix(10))

    def test_prefix_is_owned_copy_of_planned_future(self):
        queue = RTCBuffer(np.zeros((1, 2)))
        queue.merge(np.array([[1, 1], [2, 2]]), 0, self.planner)
        prefix = queue.prefix(1)
        prefix[:] = 99
        np.testing.assert_equal(queue.prefix(1), [[1, 1]])

    def test_invalid_horizons_are_rejected(self):
        for value in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                RTCOptions.from_settings({"rtcEnabled": True, "rtcHorizon": value}, 3)

    def test_training_checkpoint_selects_trained_rtc(self):
        options = RTCOptions.from_settings({"rtcEnabled": True, "rtcHorizon": 10}, 50, 10)
        self.assertEqual(options.policy_config()["mode"], "trained")
        for horizon in (9, 41):
            with self.assertRaises(ValueError):
                RTCOptions.from_settings({"rtcEnabled": True, "rtcHorizon": horizon}, 50, 10)


class RTCRuntimeTests(unittest.TestCase):
    setUp = baseline.RuntimeTests.setUp
    checkpoint = baseline.RuntimeTests.checkpoint
    wait = baseline.RuntimeTests.wait
    load = baseline.RuntimeTests.load
    run_once = baseline.RuntimeTests.run_once

    def test_rtc_load_warms_first_and_guided_chunks_without_motion(self):
        self.settings.update(rtcEnabled=True, rtcHorizon=1)
        self.load()
        self.assertEqual(self.runtime.policy.calls, ["predict", "predict", "reset"])
        self.assertEqual(self.runtime.rtc_warmup_ms, [10, 10])
        self.assertEqual(self.readers["arm"].commands, [])

    def test_predicts_while_control_keeps_sending_and_uses_prefix(self):
        self.load()
        self.settings.update(rtcEnabled=True, rtcHorizon=1)
        policy = self.runtime.policy
        policy.delay = 0.07
        original_call = policy.call
        requests, overlap = [], []
        active = [False]

        def call(request, check):
            if request["op"] == "predict":
                requests.append(request)
                active[0] = True
            try:
                return original_call(request, check)
            finally:
                active[0] = False

        policy.call = call
        reader = self.readers["arm"]
        original_control = reader.control

        def control(operation, value, check):
            if operation == "send":
                overlap.append(active[0])
            return original_control(operation, value, check)

        reader.control = control
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "done", result)
        self.assertGreaterEqual(len(requests), 2)
        self.assertIsNone(requests[0]["prefix"])
        self.assertTrue(any(request["prefix"] is not None for request in requests[1:]))
        self.assertTrue(any(overlap), "control paused while the next chunk was being predicted")
        self.assertEqual(result["rtc"]["mode"], "guided")
        self.assertEqual(reader.commands[-1], "hold")

    def test_stop_holds_before_slow_prediction_returns(self):
        self.load()
        self.settings.update(rtcEnabled=True, rtcHorizon=1)
        policy = self.runtime.policy
        policy.delay = 0.07
        self.runtime.start("continuous", self.settings, self.runtime.revision)
        self.wait(lambda: self.runtime.chunks >= 1)
        policy.delay = 0.8
        self.wait(lambda: policy.predicts >= 2)
        started = time.monotonic()
        self.runtime.stop()
        self.wait(lambda: self.readers["arm"].commands[-1] == "hold", timeout=0.5)
        self.assertLess(time.monotonic() - started, 0.5)
        self.wait(lambda: not self.runtime.thread.is_alive())
        sent = self.readers["arm"].commands.count("send")
        time.sleep(0.9)
        self.assertEqual(self.readers["arm"].commands.count("send"), sent)

    def test_invalid_rtc_settings_send_no_motor_commands(self):
        self.load()
        self.settings.update(rtcEnabled=True, rtcHorizon=3)
        with self.assertRaises(ValueError):
            self.runtime.start("single", self.settings, self.runtime.revision)
        self.assertEqual(self.readers["arm"].commands, [])


if __name__ == "__main__":
    unittest.main()
