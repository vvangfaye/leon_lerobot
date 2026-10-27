"""Bounded, single-arm-at-a-time control shared by MCP and offline tests."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SIDES = ("left", "right")
# Conservative SDK limits; do not widen these from an LLM request.
JOINT_LIMITS = (
    (-2.705261, 2.705261),
    (-1.74533, 1.74533),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, math.radians(50)),
    (-1.570797, 1.570797),
)
MAX_JOINT_DELTA = math.radians(5)
MAX_JOINT_SPEED = math.radians(5)
MAX_CARTESIAN_DELTA_M = 0.01
MAX_GRIPPER_DELTA_M = 0.01
MAX_GRIPPER_WIDTH_M = 0.07
# Measured encoder noise at a joint limit is not a wider command range.
FEEDBACK_LIMIT_TOLERANCE_RAD = math.radians(0.1)
PERIOD_S = 0.02
logger = logging.getLogger(__name__)


def finite_vector(values: Any, size: int) -> list[float]:
    """Reject malformed vectors, including bools, NaNs and infinities."""
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"Expected exactly {size} numeric values")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("All values must be finite numbers")
    return [float(v) for v in values]


def load_config(path: Path) -> dict[str, Any]:
    """Use the collection platform's explicit follower and camera mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))
    channels = {}
    for side, label in (("left", "从左"), ("right", "从右")):
        matches = [ch for ch, role in data.get("arms", {}).items() if role == label]
        if len(matches) != 1:
            raise ValueError(f"Configuration must assign exactly one CAN channel to {label}")
        channels[side] = matches[0]
    if channels["left"] == channels["right"]:
        raise ValueError("Follower channels must be different")
    return {
        "channels": channels,
        "cameras": data.get("cameras", {}),
        "camera_resolution": data.get("camera_resolution", {}),
    }


@dataclass
class ArmState:
    """A validated snapshot; unknown hardware state is an error, never zero."""

    joints: list[float]
    gripper_m: float
    ready: bool


class Backend(Protocol):
    """Hardware contract. Reading must never enable or move an actuator."""

    def read(self, side: str) -> ArmState: ...
    def prepare(self, side: str) -> None: ...
    def write_joints(self, side: str, joints: list[float]) -> None: ...
    def write_gripper(self, side: str, width_m: float) -> None: ...
    def close(self) -> None: ...


class Kinematics(Protocol):
    """Use the existing NERO model for task-space conversion."""

    def forward(self, joints: list[float], width_m: float) -> list[float]: ...
    def inverse(self, target: list[float], seed: list[float]) -> list[float]: ...


class MockBackend:
    """Deterministic mock, explicitly identified in every MCP response."""

    def __init__(self) -> None:
        self.states = {side: ArmState([0.0] * 7, 0.04, False) for side in SIDES}
        self.commands: list[tuple[str, str, Any]] = []

    def read(self, side: str) -> ArmState:
        state = self.states[side]
        return ArmState(list(state.joints), state.gripper_m, state.ready)

    def prepare(self, side: str) -> None:
        self.states[side].ready = True

    def write_joints(self, side: str, joints: list[float]) -> None:
        self.commands.append((side, "joints", list(joints)))
        self.states[side].joints = list(joints)

    def write_gripper(self, side: str, width_m: float) -> None:
        self.commands.append((side, "gripper", width_m))
        self.states[side].gripper_m = width_m

    def close(self) -> None:
        pass


class Controller:
    """Validate whole requests before writing; execute bounded local trajectories."""

    def __init__(
        self,
        backend: Backend,
        kinematics: Kinematics,
        *,
        allow_motion: bool = False,
        mock: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.kinematics = kinematics
        self.allow_motion = allow_motion
        self.mock = mock
        self.sleep = sleep
        self._motion_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._generation = 0
        self._prepared: set[str] = set()
        self._closed = False

    def _side(self, side: str) -> None:
        if side not in SIDES:
            raise ValueError("side must be 'left' or 'right'")
        if self._closed:
            raise RuntimeError("Controller is closed")

    def _state(self, side: str, *, ready: bool = False) -> ArmState:
        self._side(side)
        state = self.backend.read(side)
        state.joints = finite_vector(state.joints, 7)
        finite_vector([state.gripper_m], 1)
        if ready and not state.ready:
            raise RuntimeError("Arm is not enabled, healthy and in follower control mode")
        return state

    def state(self, side: str) -> dict[str, Any]:
        """Read measured state in physical units, without enabling the arm."""
        state = self._state(side)
        eef = self.kinematics.forward(state.joints, state.gripper_m)
        return {
            "mock": self.mock,
            "side": side,
            "ready": state.ready,
            "joints_deg": [math.degrees(q) for q in state.joints],
            "gripper_mm": state.gripper_m * 1000,
            "eef_xyz_m": eef[:3],
            "eef_rotation_6d": eef[3:9],
            "timestamp_monotonic": time.monotonic(),
            "outside_conservative_limits": [
                i + 1
                for i, (q, limits) in enumerate(zip(state.joints, JOINT_LIMITS, strict=True))
                if not limits[0] <= q <= limits[1]
            ],
        }

    def _begin(self, side: str, *, preparing: bool = False) -> int:
        self._side(side)
        if not self.allow_motion:
            raise RuntimeError("Read-only server: restart with --allow-motion to enable action tools")
        if not self._motion_lock.acquire(blocking=False):
            raise RuntimeError("Another arm operation is running; wait or call stop_motion")
        with self._state_lock:
            if not preparing and side not in self._prepared:
                self._motion_lock.release()
                raise RuntimeError("Call prepare_arm for this side before requesting motion")
            return self._generation

    def _check_generation(self, generation: int) -> None:
        with self._state_lock:
            if generation != self._generation or self._closed:
                raise RuntimeError("Motion cancelled")

    def prepare(self, side: str) -> dict[str, Any]:
        """Explicitly enable only the selected follower, without a home move."""
        generation = self._begin(side, preparing=True)
        try:
            self._state(side)
            self._check_generation(generation)
            self.backend.prepare(side)
            self._check_generation(generation)
            self._state(side, ready=True)
            with self._state_lock:
                if generation != self._generation:
                    raise RuntimeError("Preparation cancelled")
                self._prepared.add(side)
            return self.state(side)
        finally:
            self._motion_lock.release()

    def _validate_joints(self, start: list[float], target: list[float], duration_s: float) -> None:
        finite_vector([duration_s], 1)
        if not 1.0 <= duration_s <= 5.0:
            raise ValueError("duration_s must be between 1 and 5 seconds")
        for index, (q0, q1, bounds) in enumerate(zip(start, target, JOINT_LIMITS, strict=True)):
            if (
                not bounds[0] - FEEDBACK_LIMIT_TOLERANCE_RAD <= q0 <= bounds[1] + FEEDBACK_LIMIT_TOLERANCE_RAD
                or not bounds[0] <= q1 <= bounds[1]
            ):
                raise ValueError(f"Joint {index + 1} start or target exceeds conservative SDK limits")
            delta = abs(q1 - q0)
            if delta > MAX_JOINT_DELTA + 1e-9 or delta / duration_s > MAX_JOINT_SPEED + 1e-9:
                raise ValueError("Maximum change is 5 degrees per call and 5 degrees/second")

    def _recover(self, side: str, *, gripper: bool = False) -> None:
        # Only a fresh healthy position can be used as a hold target. Never disable torque.
        with self._state_lock:
            self._prepared.discard(side)
        try:
            state = self._state(side, ready=True)
            if gripper:
                self.backend.write_gripper(side, state.gripper_m)
            else:
                self.backend.write_joints(side, state.joints)
        except Exception as exc:
            # Lost feedback: do not fabricate a target or replay a stale one.
            logger.error("Could not confirm a fresh-position hold for %s: %s", side, exc)

    def _move(self, side: str, target: list[float], duration_s: float, generation: int) -> dict[str, Any]:
        start = self._state(side, ready=True).joints
        self._validate_joints(start, target, duration_s)
        steps = math.ceil(duration_s / PERIOD_S)
        try:
            for step in range(1, steps + 1):
                self._check_generation(generation)
                current = self._state(side, ready=True)
                goal = [a + (b - a) * step / steps for a, b in zip(start, target, strict=True)]
                goal = [max(lo, min(hi, q)) for q, (lo, hi) in zip(goal, JOINT_LIMITS, strict=True)]
                if max(abs(a - b) for a, b in zip(goal, current.joints, strict=True)) > MAX_JOINT_DELTA:
                    raise RuntimeError("Measured tracking error exceeded 5 degrees")
                self._check_generation(generation)
                self.backend.write_joints(side, goal)
                self.sleep(duration_s / steps)
            for _ in range(100):
                self._check_generation(generation)
                current = self._state(side, ready=True)
                if max(abs(a - b) for a, b in zip(target, current.joints, strict=True)) <= math.radians(0.5):
                    return self.state(side)
                self.sleep(PERIOD_S)
            raise TimeoutError("Arm did not reach the target within 0.5 degrees")
        except Exception:
            self._recover(side)
            raise

    def move_joints(self, side: str, target_deg: list[float], duration_s: float = 2.0) -> dict[str, Any]:
        """Interpolate a small joint move locally, checking feedback every tick."""
        target = [math.radians(q) for q in finite_vector(target_deg, 7)]
        generation = self._begin(side)
        try:
            return self._move(side, target, duration_s, generation)
        finally:
            self._motion_lock.release()

    def move_ee_by(self, side: str, delta_mm: list[float], duration_s: float = 2.0) -> dict[str, Any]:
        """Translate up to 10 mm in this arm's base frame, preserving orientation."""
        delta = finite_vector(delta_mm, 3)
        if math.sqrt(sum(v * v for v in delta)) > MAX_CARTESIAN_DELTA_M * 1000 + 1e-9:
            raise ValueError("Maximum Cartesian translation is 10 mm per call")
        generation = self._begin(side)
        try:
            state = self._state(side, ready=True)
            pose = self.kinematics.forward(state.joints, state.gripper_m)
            pose[:3] = [a + b / 1000 for a, b in zip(pose[:3], delta, strict=True)]
            target = finite_vector(self.kinematics.inverse(pose, state.joints), 7)
            return self._move(side, target, duration_s, generation)
        finally:
            self._motion_lock.release()

    def set_gripper(self, side: str, width_mm: float) -> dict[str, Any]:
        """Change the selected gripper by at most 10 mm with fixed low force."""
        width = finite_vector([width_mm], 1)[0] / 1000
        if not 0 <= width <= MAX_GRIPPER_WIDTH_M:
            raise ValueError("Gripper width must be between 0 and 70 mm")
        generation = self._begin(side)
        try:
            state = self._state(side, ready=True)
            if abs(width - state.gripper_m) > MAX_GRIPPER_DELTA_M + 1e-9:
                raise ValueError("Maximum gripper change is 10 mm per call")
            try:
                self._check_generation(generation)
                self.backend.write_gripper(side, width)
                for _ in range(100):
                    self._check_generation(generation)
                    if abs(self._state(side, ready=True).gripper_m - width) <= 0.002:
                        return self.state(side)
                    self.sleep(PERIOD_S)
                raise TimeoutError("Gripper did not reach the requested width within 2 mm")
            except Exception:
                self._recover(side, gripper=True)
                raise
        finally:
            self._motion_lock.release()

    def stop(self) -> dict[str, Any]:
        """Cancel active work; the worker attempts a fresh-position hold."""
        with self._state_lock:
            self._generation += 1
            self._prepared.clear()
        return {
            "mock": self.mock,
            "cancel_requested": True,
            "message": "Cancellation requested; this is not a hardware emergency stop. Re-prepare before moving.",
        }

    def close(self) -> None:
        """Cancel and finish the worker before releasing SDK connections."""
        self.stop()
        with self._motion_lock:
            self._closed = True
            self.backend.close()
