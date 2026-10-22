"""SDK operations executed only by each arm's existing device owner thread."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .jog import ArmJog


class ArmMotion(ArmJog):
    """Reuse jog's fresh controller/driver checks, enabling and limit queries."""

    def __init__(self, reader: object, home: bool = False) -> None:
        from .hardware import FreshFeedback

        super().__init__(
            reader.arm, reader.robot, reader.limits, FreshFeedback(),
            label=getattr(reader, "info", {}).get("label"),
        )
        self.gripper = reader.gripper
        self.home_mode = home
        self.last_target = None

    def _identify_rest_pose(self, pose: list) -> None:
        # The wrist-test exception must never widen inference targets.
        pass

    def _bounded(self, pose: list) -> list:
        if self.home_mode:
            # Match the existing workbench's bounded return from folded feedback.
            if any(
                value < lo - math.radians(10) or value > hi + math.radians(10)
                for value, (lo, hi) in zip(pose, self.limits, strict=True)
            ):
                raise ValueError("回位起点超出限位 10°，请现场检查")
            return list(pose)
        return super()._bounded(pose)

    def prepare(self, check: Callable) -> dict:
        origin = self._prepare(check, lambda message: None)
        self.arm.set_speed_percent(10)
        self.last_target = np.r_[origin, self.feedback_gripper()]
        return {"point": self.last_target.tolist(), "limits": self.limits}

    def feedback_gripper(self) -> float:
        # ArmReader has already established fresh gripper feedback. Establish our
        # own timestamp history before using a gripper as a command anchor.
        deadline = time.monotonic() + 0.55
        while True:
            try:
                grip = self.fresh.read("gripper", self.gripper.get_gripper_status(), time.monotonic())
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        if (
            not math.isfinite(grip.value)
            or not -0.002 <= grip.value <= 0.102
            or getattr(getattr(grip, "foc_status", None), "driver_error_status", False)
        ):
            raise ValueError("夹爪反馈异常")
        return float(np.clip(grip.value, 0, 0.1))

    def read(self) -> dict:
        observation = self._observe()
        pose = self._ready(observation)
        return {"point": [*pose, self.feedback_gripper()], "stamp": observation[3]}

    def send(self, point: list, check: Callable) -> None:
        point = np.asarray(point, dtype=float)
        if point.shape != (self.count + 1,) or not np.isfinite(point).all() or not 0 <= point[-1] <= 0.1:
            raise ValueError("控制目标维度/数值无效")
        # 与 LeRobot 的 send_action 一致：目标已在 plan() 内钳制到限位、并按 0.25 rad/s /
        # 0.02 m/s 限速插值（等价于 ActionInterpolator），此处只发平滑目标，不再做限位/跳变检查。
        check()
        self.arm.move_js(point[:-1].tolist())
        check()
        self.gripper.move_gripper_m(value=float(point[-1]), force=1.0)
        self.last_target = point

    def home(self, target: list, check: Callable) -> None:
        if len(target) != self.count or any(
            not lo <= q <= hi for q, (lo, hi) in zip(target, self.limits, strict=True)
        ):
            raise ValueError("初始位置超出本机限位")
        self.read()
        check()
        self.arm.set_speed_percent(10)
        check()
        self.arm.move_j(target)

    def hold(self) -> None:
        if not self.touched:
            return
        try:
            observation = self._observe()
            previous = observation[3]
            deadline = time.monotonic() + 0.3
            while observation[3] == previous:
                if time.monotonic() >= deadline:
                    raise ValueError("停止时关节反馈未更新")
                time.sleep(0.01)
                observation = self._observe()
            pose = self._ready(observation)
            # A folded home origin cannot be emitted as a general target.
            if any(
                not lo - math.radians(0.15) <= q <= hi + math.radians(0.15)
                for q, (lo, hi) in zip(pose, self.limits, strict=True)
            ):
                raise ValueError("停止位置超限，使用电子急停")
            target = [max(lo, min(hi, q)) for q, (lo, hi) in zip(pose, self.limits, strict=True)]
            self.arm.move_j(target)
            if not self.home_mode:
                self.gripper.move_gripper_m(value=self.feedback_gripper(), force=1.0)
            deadline, stamp, stable, last = time.monotonic() + 1, observation[3], 0, pose
            while time.monotonic() < deadline:
                observation = self._observe()
                pose = self._ready(observation)
                if observation[3] != stamp:
                    stable = (
                        stable + 1
                        if max(abs(a - b) for a, b in zip(pose, last, strict=True)) < math.radians(0.05)
                        and max(abs(a - b) for a, b in zip(pose, target, strict=True)) < math.radians(0.8)
                        else 0
                    )
                    if stable >= 3:
                        return
                    stamp, last = observation[3], pose
                time.sleep(0.02)
            raise ValueError("未确认停止后关节稳定")
        except Exception:
            self.arm.electronic_emergency_stop()
            raise


def home_target(robot: str) -> list:
    if robot == "nero":
        from .hardware import camera_module

        return list(camera_module("devices").HOME_POS)
    return [0.0] * 6


def plan(
    actions: np.ndarray, contract: object, initial: np.ndarray, limits: np.ndarray, check: Callable
) -> tuple:
    from .kinematics import solve

    points = [initial.copy()]
    width = contract.count + 1 if contract.config["space"] == "joint" else 10
    warnings: list[str] = []
    for action in contract.actions(actions):
        check()
        targets = []
        for index, target in enumerate(action.reshape(-1, width)):
            if contract.config["space"] == "joint":
                q, grip = target[:-1], target[-1]
            else:
                q = solve(contract.config["robot"], target, points[-1][index, :-1], limits[index])
                grip = (target[-1] + 1) / 20
            if not -0.0001 <= grip <= 0.1001:
                clamped = float(np.clip(grip, 0, 0.1))
                warnings.append(f"夹爪 {grip:.4f} m 越界，钳制到 {clamped:.4f} m")
                grip = clamped
            # 模型外推越界：像 pyAgxArm SDK 的 clamp_joints 一样逐关节钳制到限位，而不是报错拒绝。
            # 钳制只作用在越界关节，钳制值落在限位边界，仍受相邻步 0.20 rad 与速度上限约束。
            overflow = (q < limits[index, :, 0]) | (q > limits[index, :, 1])
            if np.any(overflow):
                for j in np.flatnonzero(overflow):
                    bound = limits[index, j, 0] if q[j] < limits[index, j, 0] else limits[index, j, 1]
                    warnings.append(f"J{j + 1} {math.degrees(q[j]):.2f}° 越界，钳制到 {math.degrees(bound):.2f}°")
                q = np.clip(q, limits[index, :, 0], limits[index, :, 1])
            if np.max(np.abs(q - points[-1][index, :-1])) > 0.20:
                raise ValueError("动作相邻关节变化超过 0.20 rad，未执行本段")
            targets.append(np.r_[q, np.clip(grip, 0, 0.1)])
        points.append(np.asarray(targets))
    points = np.asarray(points)
    # Retain every prediction, slow down only segments exceeding conservative
    # physical rates: 0.25 rad/s and 0.02 m/s. 30 Hz nominal action timing.
    delta = np.abs(np.diff(points, axis=0))
    durations = np.maximum(
        1 / 30, np.maximum(delta[..., :-1].max(axis=(1, 2)) / 0.25, delta[..., -1].max(axis=1) / 0.02)
    )
    return points, durations, warnings
