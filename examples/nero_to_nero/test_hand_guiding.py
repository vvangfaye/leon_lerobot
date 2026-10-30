"""Direct hand guiding never records limits or sends a joint-position command."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform import hand_guiding
from nero_platform.app import create_app
from nero_platform.runtime import Runtime


class HandGuidingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Runtime()
        self.events = []
        arm = SimpleNamespace()
        grip = SimpleNamespace(
            get_gripper_status=lambda: SimpleNamespace(msg=SimpleNamespace(value=0.06)),
            disable_gripper=lambda: self.events.append("release_gripper"),
        )
        self.runtime.channels["can1"].update(label="从右", detected=True, arm=arm, gripper=grip)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.ready = self.stack.enter_context(
            patch.object(hand_guiding.hardware_status, "require_ready_for_motion")
        )
        self.wait = self.stack.enter_context(patch.object(hand_guiding, "_wait_for_drag_feedback"))
        self.stack.enter_context(
            patch.object(hand_guiding.devices, "_disable_and_confirm", side_effect=lambda fn, read: fn())
        )
        self.role = self.stack.enter_context(
            patch.object(
                hand_guiding.devices,
                "set_control_role",
                side_effect=lambda rt, ch, role: self.events.append((ch, role)),
            )
        )

    def test_start_stop_only_releases_gripper_and_changes_modes(self) -> None:
        hand_guiding.start(self.runtime, "can1")
        self.assertEqual(self.events, ["release_gripper", ("can1", "leader")])
        self.assertEqual(self.runtime.state["hand_guiding_phase"], "active")
        self.assertTrue(self.runtime.operation_busy())
        self.assertFalse(self.runtime.state["measuring"])
        hand_guiding.stop(self.runtime, "can1")
        self.assertEqual(self.events[-1], ("can1", "follower"))
        self.assertIsNone(self.runtime.state["hand_guiding_channel"])
        self.assertNotIn("can1", self.runtime.device_busy)

    def test_wrong_arm_busy_or_unconnected_refused_before_actuation(self) -> None:
        for setting in ("wrong_arm", "teleop", "unconnected"):
            with self.subTest(setting=setting):
                self.runtime.state["teleop_test_active"] = setting == "teleop"
                self.runtime.channels["can1"]["detected"] = setting != "unconnected"
                with self.assertRaises((RuntimeError, ValueError)):
                    hand_guiding.start(self.runtime, "can0" if setting == "wrong_arm" else "can1")
                self.assertEqual(self.events, [])

    def test_preflight_failure_does_not_send_mode_or_gripper_command(self) -> None:
        self.ready.side_effect = RuntimeError("not ready")
        with self.assertRaises(RuntimeError):
            hand_guiding.start(self.runtime, "can1")
        self.assertEqual(self.events, [])
        self.assertIsNone(self.runtime.state["hand_guiding_channel"])

    def test_missing_teaching_feedback_restores_follower(self) -> None:
        self.wait.side_effect = TimeoutError("no feedback")
        with self.assertRaises(TimeoutError):
            hand_guiding.start(self.runtime, "can1")
        self.assertEqual(self.events[-1], ("can1", "follower"))
        self.assertIsNone(self.runtime.state["hand_guiding_channel"])

    def test_stop_failure_stays_locked_for_retry(self) -> None:
        hand_guiding.start(self.runtime, "can1")
        self.ready.side_effect = RuntimeError("stale")
        with self.assertRaises(RuntimeError):
            hand_guiding.stop(self.runtime, "can1")
        self.assertEqual(self.runtime.state["hand_guiding_phase"], "error")
        self.assertTrue(self.runtime.operation_busy())
        self.ready.side_effect = None
        hand_guiding.stop(self.runtime, "can1")
        self.assertFalse(self.runtime.operation_busy())

    def test_http_and_other_commands_are_interlocked(self) -> None:
        client = create_app(self.runtime).test_client()
        self.assertEqual(client.post("/api/arm/guide/start/can1").status_code, 200)
        self.assertEqual(client.post("/api/arm/move/can1").status_code, 409)
        self.assertEqual(client.post("/api/arm/measure/can1").status_code, 409)
        self.assertEqual(client.get("/api/status").json["hand_guiding_channel"], "can1")
        self.assertEqual(client.post("/api/arm/guide/stop/can0").status_code, 409)
        self.assertEqual(client.post("/api/arm/guide/stop/can1").status_code, 200)


if __name__ == "__main__":
    unittest.main()
