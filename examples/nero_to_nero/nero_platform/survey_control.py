"""Slow, operator-only joint increments for measuring a task workspace.

This is not an RL environment and generates no SAC demonstrations. Cartesian
workspace limits are the output of the survey, not assumed to be known here.
"""

import math

import numpy as np

from .agent_control import JOINT_LIMITS as SDK_JOINT_LIMITS, finite_vector

JOINT_LIMITS = np.array(SDK_JOINT_LIMITS)


def vector(values: object, size: int) -> np.ndarray:
    return np.asarray(finite_vector(values, size))


class SurveyAlignmentError(ValueError):
    """Recoverable mismatch: hold the follower and let the operator reposition the master."""


class SurveyControl:
    def __init__(self, reading: dict, now: float, *, absolute: bool = False, allow_catchup: bool = False):
        self.previous = vector(reading["主臂关节"], 7)
        self.target = vector(reading["从臂关节"], 7)
        self.width = float(reading["从臂夹爪"][0])
        self.previous_width = float(reading["主臂夹爪"][0])
        if not math.isfinite(self.previous_width) or not 0 <= self.previous_width <= 0.105:
            raise ValueError("主臂夹爪反馈无效")
        self.time = now
        self.absolute = absolute
        self.lag_deg = 0.0
        if np.any(self.target < JOINT_LIMITS[:, 0]) or np.any(self.target > JOINT_LIMITS[:, 1]):
            raise ValueError("当前从臂超出 RL 保守关节限位，未开始范围测量")
        if not 0 <= self.width <= 0.07:
            raise ValueError("当前从臂夹爪不在 0–70 mm 范围，未开始范围测量")
        if absolute and (
            np.any(self.previous < JOINT_LIMITS[:, 0]) or np.any(self.previous > JOINT_LIMITS[:, 1])
        ):
            raise ValueError("主臂目标超出从臂保守关节限位")
        if absolute and not allow_catchup and np.max(np.abs(self.previous - self.target)) > math.radians(0.5):
            raise SurveyAlignmentError("主从关节姿态尚未对齐（需要每关节差不超过 0.5°）")
        if absolute and not allow_catchup and abs(self.previous_width - self.width) > 0.001:
            raise SurveyAlignmentError("主从夹爪开度差超过 1 mm，先对齐开度")

    def action(self, reading: dict, now: float) -> list[float]:
        current = vector(reading["主臂关节"], 7)
        actual = vector(reading["从臂关节"], 7)
        width = float(reading["主臂夹爪"][0])
        if not math.isfinite(width) or not 0 <= width <= 0.105:
            raise ValueError("主臂夹爪反馈无效")
        if np.any(actual < JOINT_LIMITS[:, 0] - 1e-3) or np.any(actual > JOINT_LIMITS[:, 1] + 1e-3):
            raise RuntimeError("从臂反馈超出保守关节限位，已停止跟随")
        dt = now - self.time
        if not 0 < dt <= 0.2:
            raise RuntimeError("范围测量控制循环延迟，已停止跟随")
        if np.max(np.abs(self.target - actual)) > math.radians(3):
            raise RuntimeError("从臂未跟上范围测量指令，已停止跟随")
        if self.absolute:
            if np.any(current < JOINT_LIMITS[:, 0]) or np.any(current > JOINT_LIMITS[:, 1]):
                raise RuntimeError("主臂目标超出从臂保守关节限位，已停止跟随")
            # The operator's desired pose can lead a rate-limited command.
            # This is not the tracking error tested above (command vs feedback).
            # Approach the desired pose through bounded increments, including
            # after an explicitly requested catch-up; never jump to it.
            delta, grip_delta = current - self.target, width - self.width
        else:
            # Consume relative deltas once; clipped motion never becomes a backlog.
            delta, grip_delta = (current - self.previous) * 0.5, width - self.previous_width
        change = np.clip(delta, -math.radians(5) * dt, math.radians(5) * dt)
        self.target = np.clip(self.target + change, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.width = float(np.clip(self.width + np.clip(grip_delta, -0.01 * dt, 0.01 * dt), 0, 0.07))
        self.lag_deg = float(np.rad2deg(np.max(np.abs(current - actual))))
        self.previous, self.previous_width, self.time = current, width, now
        return [*self.target.tolist(), self.width]
