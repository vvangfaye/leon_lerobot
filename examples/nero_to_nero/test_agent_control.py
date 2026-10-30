"""Offline tests for NERO MCP motion boundaries and lifecycle."""

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from nero_platform.agent_control import Controller, MockBackend, load_config


class SimpleKinematics:
    """A deterministic test model, not a simulation of real NERO geometry."""

    def forward(self, joints: list[float], width: float) -> list[float]:
        return [*joints[:3], 1, 0, 0, 0, 1, 0, width]

    def inverse(self, target: list[float], seed: list[float]) -> list[float]:
        return [*target[:3], *seed[3:]]


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockBackend()
        self.controller = Controller(
            self.backend, SimpleKinematics(), mock=True, allow_motion=True, sleep=lambda _: None
        )

    def test_read_only_and_unprepared_servers_reject_all_actuation(self) -> None:
        self.controller.allow_motion = False
        with self.assertRaisesRegex(RuntimeError, "Read-only"):
            self.controller.prepare("left")
        self.controller.allow_motion = True
        with self.assertRaisesRegex(RuntimeError, "prepare_arm"):
            self.controller.move_joints("left", [1] * 7)
        self.assertEqual(self.backend.commands, [])

    def test_invalid_inputs_never_write(self) -> None:
        self.controller.prepare("left")
        for target in ([100] * 7, [float("nan")] * 7, [True] * 7, [0] * 6):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.controller.move_joints("left", target)
        for duration in (0, 6, float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.controller.move_joints("left", [1] * 7, duration)
        with self.assertRaises(ValueError):
            self.controller.set_gripper("left", 70)
        with self.assertRaises(ValueError):
            self.controller.move_ee_by("left", [10, 10, 0])
        self.assertEqual(self.backend.commands, [])

    def test_only_selected_follower_moves_and_arrival_uses_feedback(self) -> None:
        self.controller.prepare("left")
        result = self.controller.move_joints("left", [1, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(result["joints_deg"][0], 1)
        self.assertTrue(result["mock"])
        self.assertTrue(all(side == "left" for side, _, _ in self.backend.commands))
        self.assertEqual(self.backend.states["right"].joints, [0] * 7)

    def test_ik_failure_never_sends_a_partial_motion(self) -> None:
        self.controller.prepare("left")
        with (
            patch.object(self.controller.kinematics, "inverse", side_effect=ValueError("unreachable")),
            self.assertRaisesRegex(ValueError, "unreachable"),
        ):
            self.controller.move_ee_by("left", [1, 0, 0])
        self.assertEqual(self.backend.commands, [])

    def test_tracking_failure_is_not_reported_as_success(self) -> None:
        self.controller.prepare("left")
        with patch.object(self.backend, "write_joints"), self.assertRaises(TimeoutError):
            self.controller.move_joints("left", [2, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "prepare_arm"):
            self.controller.move_joints("left", [1] * 7)

    def test_stop_remains_available_during_motion_and_blocks_next_move(self) -> None:
        self.controller.prepare("left")
        entered, release = threading.Event(), threading.Event()
        failures = []

        def sleep(_: float) -> None:
            entered.set()
            release.wait(timeout=2)

        def move() -> None:
            try:
                self.controller.move_joints("left", [1, 0, 0, 0, 0, 0, 0])
            except RuntimeError as exc:
                failures.append(str(exc))

        self.controller.sleep = sleep
        worker = threading.Thread(target=move)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaisesRegex(RuntimeError, "Another arm"):
                self.controller.prepare("right")
            self.assertTrue(self.controller.stop()["cancel_requested"])
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, ["Motion cancelled"])
        self.assertEqual(len(self.backend.commands), 2)  # one step, one measured hold
        with self.assertRaisesRegex(RuntimeError, "prepare_arm"):
            self.controller.move_joints("left", [1] * 7)

    def test_lost_feedback_never_produces_a_zero_hold_target(self) -> None:
        self.controller.prepare("left")
        count = 0
        original = self.backend.read

        def failing_read(side: str):
            nonlocal count
            count += 1
            if count > 2:
                raise RuntimeError("stale feedback")
            return original(side)

        with (
            patch.object(self.backend, "read", side_effect=failing_read),
            self.assertLogs(level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "stale"),
        ):
            self.controller.move_joints("left", [1] * 7)
        self.assertEqual(len(self.backend.commands), 1)

    def test_closed_controller_cannot_reconnect(self) -> None:
        self.controller.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.controller.state("left")

    def test_encoder_noise_at_limit_does_not_widen_commanded_targets(self) -> None:
        from nero_platform.agent_control import JOINT_LIMITS

        self.backend.states["left"].joints[1] = JOINT_LIMITS[1][0] - math.radians(0.02)
        self.controller.prepare("left")
        self.controller.move_joints("left", [0, -99, 0, 0, 0, 0, 0])
        self.assertTrue(all(cmd[2][1] >= JOINT_LIMITS[1][0] for cmd in self.backend.commands))

    def test_joint_six_uses_the_more_conservative_installed_sdk_limit(self) -> None:
        self.backend.states["right"].joints[5] = math.radians(49)
        self.controller.prepare("right")
        with self.assertRaisesRegex(ValueError, "Joint 6"):
            self.controller.move_joints("right", [0, 0, 0, 0, 0, 51, 0])
        self.assertEqual(self.backend.commands, [])

    def test_lagging_stream_keeps_sending_endpoint_until_feedback_arrives(self) -> None:
        self.controller.prepare("right")

        def lagging_write(side: str, goal: list[float]) -> None:
            self.backend.commands.append((side, "joints", list(goal)))
            current = self.backend.states[side].joints
            self.backend.states[side].joints = [
                a + 0.05 * (b - a) for a, b in zip(current, goal, strict=True)
            ]

        with patch.object(self.backend, "write_joints", side_effect=lagging_write):
            result = self.controller.move_joints("right", [5, 0, 0, 0, 0, 0, 0], duration_s=1)
        self.assertGreater(len(self.backend.commands), 50)
        self.assertLess(abs(result["joints_deg"][0] - 5), 0.5)
        self.assertTrue(all(0 <= command[2][0] <= math.radians(5) for command in self.backend.commands))


class ConfigTests(unittest.TestCase):
    def test_followers_are_selected_by_role_not_old_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "arms": {
                            "can0": "从左",
                            "can1": "从右",
                            "can2": "主左",
                            "can3": "主右",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_config(path)["channels"], {"left": "can0", "right": "can1"})
            path.write_text(json.dumps({"arms": {"can0": "从左", "can2": "从左"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
