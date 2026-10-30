"""Joint-relative human takeover using the unchanged native VLA sender."""

import time
from typing import Any

import numpy as np

from examples.nero_to_nero.nero_platform import sampling

from .config import vector
from .hardware import NeroHardware
from .master_input import MasterInput


class NativeTakeover:
    """Anchor both arms once, then translate master deltas into absolute targets."""

    def __init__(self, readings: dict, follower_target: np.ndarray):
        self.master_start = vector(readings["主臂关节"] + readings["主臂夹爪"], 8)
        self.follower_start = vector(follower_target, 8).copy()

    def action(self, readings: dict) -> list[float]:
        master = vector(readings["主臂关节"] + readings["主臂夹爪"], 8)
        target = self.follower_start + master - self.master_start
        target[7] = np.clip(target[7], 0, 0.09)
        return target.tolist()


def engage_takeover(
    backend: NeroHardware, master: MasterInput, keys: Any
) -> tuple[sampling.RobotFeedback, np.ndarray, NativeTakeover] | None:
    """Hold the follower and capture fresh references, without an alignment move."""
    cancel = getattr(keys, "cancel", keys.abort)
    cancel.clear()
    if not backend.prepared:
        backend.prepare()
    backend.hold()
    master.prepare()
    if cancel.is_set() or keys.abort.is_set():
        return None
    feedback = sampling.RobotFeedback(
        master.backend.runtime.channels["can3"], backend.runtime.channels["can1"]
    )
    deadline = time.monotonic() + 1
    while True:
        if cancel.is_set() or keys.abort.is_set():
            return None
        readings = feedback.read(time.monotonic())
        if feedback.ready:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Takeover master/follower feedback did not become ready")
        time.sleep(0.01)
    # The paired feedback captures both references in one acquisition. Retain
    # the gripper command (not measured contact width), so takeover cannot open it.
    previous = np.r_[readings["从臂关节"], backend.last_servo_width]
    backend.last_servo_target = previous[:7].copy()
    return feedback, previous, NativeTakeover(readings, previous)
