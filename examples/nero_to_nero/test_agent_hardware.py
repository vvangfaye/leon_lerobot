"""Verify read-only SDK access and partial-feedback rejection without CAN."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nero_platform.agent_hardware import LiveBackend


class FakeParser:
    """Emit independently timestamped joint packets."""

    def __init__(self) -> None:
        self.missing: set[str] = set()
        self.ticks: dict[str, int] = {}

    def __getattr__(self, name: str) -> SimpleNamespace | None:
        if name in self.missing:
            return None
        self.ticks[name] = self.ticks.get(name, 1) + 1
        return SimpleNamespace(timestamp=self.ticks[name], msg=SimpleNamespace())


class FakeArm:
    """Expose feedback and record any operation that could change hardware."""

    def __init__(self) -> None:
        self._parser = FakeParser()
        self.tick = 1
        self.commands: list[str] = []
        self.connected = False
        self.fault = 0
        self.OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="gripper"))

    def packet(self, msg: object) -> SimpleNamespace:
        self.tick += 1
        return SimpleNamespace(timestamp=self.tick, msg=msg)

    def init_effector(self, kind: str) -> "FakeArm":
        return self

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_angles(self) -> SimpleNamespace:
        return self.packet([0.0] * 7)

    def get_leader_joint_angles(self) -> None:
        return None

    def get_arm_status(self) -> SimpleNamespace:
        return self.packet(SimpleNamespace(ctrl_mode=1, arm_status=self.fault, err_code=0))

    def get_driver_states(self, joint: int) -> SimpleNamespace:
        return self.packet(SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=True)))

    def get_gripper_status(self) -> SimpleNamespace:
        return self.packet(SimpleNamespace(value=0.04, foc_status=SimpleNamespace(driver_enable_status=True)))


class HardwareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = FakeArm()
        self.backend = LiveBackend({"channels": {"left": "can0", "right": "can1"}})
        factory = patch("nero_platform.agent_hardware.AgxArmFactory.create_arm", return_value=self.arm)
        factory.start()
        self.addCleanup(factory.stop)
        self.addCleanup(self.backend.close)

    def test_construct_read_close_never_enable_or_move(self) -> None:
        self.assertFalse(self.arm.connected)
        state = self.backend.read("left")
        self.assertTrue(state.ready)
        self.assertEqual(state.joints, [0.0] * 7)
        self.backend.close()
        self.assertFalse(self.arm.connected)
        self.assertEqual(self.arm.commands, [])

    def test_missing_joint_packet_rejected_even_with_valid_aggregate(self) -> None:
        self.arm._parser.missing.add("joint_7")
        with self.assertRaisesRegex(RuntimeError, "incomplete or stale"):
            self.backend.read("left")
        self.assertEqual(self.arm.commands, [])

    def test_controller_fault_makes_arm_not_ready(self) -> None:
        self.arm.fault = 7
        self.assertFalse(self.backend.read("left").ready)


if __name__ == "__main__":
    unittest.main()
