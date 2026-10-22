"""Small identification motion, executed by the device's existing owner thread."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

# 反馈噪声容差：回位 J7=90° 正好压在控制器 +90° 上限上，编码器噪声会读到 90.00x°。
# 允许反馈略超限位并夹回，避免边界位姿被误判为越界；模型目标仍严格检查。
FEEDBACK_LIMIT_TOLERANCE_RAD = math.radians(0.1)

# 控制器/SDK 软限位比机械行程窄。本机实测手动拖动范围（度，can1）：
# J1 [-157.1, 157.4]  J2 [-101.0, 104.8]  J3 [-160.8, 159.8]  J4 [-59.7, 126.3]
# J5 [-161.0, 159.9]  J6 [-43.1, 57.3]  J7 [-101.8, 98.1]
# 只有贴近训练数据边界的关节，模型才会外推越界，需放宽：J2 下界、J4 上界、J7 双侧。
# 放宽值取实测范围并略向内收（留安全余量），只扩不缩；Piper 不适用。
# 模型目标越界会被钳制到限位（不再拒绝），且仍受相邻步 0.20 rad 与速度上限约束。
NERO_LIMIT_WIDEN_DEG = {2: (-101.0, None), 4: (None, 126.2), 7: (-101.8, 98.1)}


def _joint_limits() -> Any:
    import importlib
    import sys
    from pathlib import Path

    collection = Path(__file__).resolve().parents[2] / "nero_to_nero"
    if str(collection) not in sys.path:
        sys.path.insert(0, str(collection))
    return importlib.import_module("nero_platform.joint_limits")


class JogStopped(Exception):
    pass


class FeedbackPending(ValueError):
    pass


class JogJob:
    """A browser must receive the job ID and keep renewing its short lease."""

    def __init__(self, identity: str) -> None:
        self.id, self.identity = uuid4().hex, identity
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.created = time.monotonic()
        self.heartbeat_at = 0.0
        self.active, self.recovery = True, False
        self.phase, self.message = "queued", "等待页面确认微动请求…"

    def heartbeat(self) -> None:
        with self.lock:
            # A delayed HTTP renewal must not revive an expired lease.
            if self.active and not self.cancel.is_set() \
                    and time.monotonic() - (self.heartbeat_at or self.created) <= 1.5:
                self.heartbeat_at = time.monotonic()

    def check(self) -> None:
        with self.lock:
            if self.cancel.is_set():
                raise JogStopped("已请求停止")
            if time.monotonic() - (self.heartbeat_at or self.created) > 1.5:
                raise JogStopped("页面心跳中断")

    def wait_for_browser(self) -> None:
        while True:
            self.check()
            with self.lock:
                if self.heartbeat_at:
                    return
            self.cancel.wait(0.02)

    def progress(self, message: str) -> None:
        with self.lock:
            self.phase, self.message = "running", message

    def finish(self, phase: str, message: str, recovery: bool = False) -> None:
        with self.lock:
            self.phase, self.message = phase, message
            self.recovery, self.active = recovery, False

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id, "device": self.identity, "active": self.active,
                "phase": self.phase, "message": self.message, "recovery": self.recovery,
            }


class ArmJog:
    """5° wrist round trip at 5% planned joint speed. Never uses instantaneous JS."""

    def __init__(self, arm: Any, robot: str, preset: list, fresh: Any, label: str | None = None) -> None:
        self.arm, self.robot, self.preset, self.fresh = arm, robot, preset, fresh
        self.label = label
        self.count = {"nero": 7, "piper": 6}[robot]
        self.limits: list[tuple[float, float]] = []
        self.touched = False
        self.following = False
        self.rest_origin: list[float] | None = None

    def _observe(self) -> tuple[list[float], Any, list[bool], float]:
        values, errors = {}, []
        reads = [
            ("joint", self.arm.get_joint_angles, ()),
            ("controller", self.arm.get_arm_status, ()),
            *((f"J{i}", self.arm.get_driver_states, (i,)) for i in range(1, self.count + 1)),
        ]
        for key, method, args in reads:
            try:
                sample = method(*args)
                values[key] = self.fresh.read(key, sample, time.monotonic())
                if key == "joint":
                    stamp = float(sample.timestamp)
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append(f"{key}: {exc}")
        if errors:
            raise FeedbackPending("等待完整新反馈：" + "；".join(errors))
        pose = [float(value) for value in values["joint"]]
        if len(pose) != self.count or not all(math.isfinite(value) for value in pose):
            raise ValueError("普通关节反馈维数或数值无效")
        controller = values["controller"]
        enabled = []
        for i in range(1, self.count + 1):
            foc = values[f"J{i}"].foc_status
            if getattr(foc, "driver_error_status", True):
                raise ValueError(f"J{i} 驱动故障，未复位")
            flag = foc.driver_enable_status
            if not isinstance(flag, bool):
                raise ValueError(f"J{i} 使能状态无效")
            enabled.append(flag)
        if controller.err_code != 0 or controller.arm_status not in (0, 6):
            raise ValueError(f"控制器异常：状态 {controller.arm_status} / 错误 {controller.err_code}，未复位")
        if controller.ctrl_mode not in (0, 1, 2, 6):
            raise ValueError(f"当前控制模式 {controller.ctrl_mode} 不支持微动")
        return pose, controller, enabled, stamp

    def _read_initial(self, check: Callable) -> tuple:
        deadline = time.monotonic() + 0.8
        while True:
            check()
            try:
                return self._observe()
            except FeedbackPending:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.02)

    def _read_limits(self, check: Callable) -> None:
        if len(self.preset) != self.count:
            raise ValueError("SDK 关节限位缺失")
        for i, preset in enumerate(self.preset, 1):
            check()
            reply = self.arm.get_joint_angle_vel_limits(i, timeout=0.15, min_interval=0)
            check()
            msg = getattr(reply, "msg", None)
            lo, hi = getattr(msg, "min_angle_limit", None), getattr(msg, "max_angle_limit", None)
            if lo is None or hi is None or not all(math.isfinite(x) for x in (*preset, lo, hi)):
                raise ValueError(f"J{i} 控制器限位未确认")
            if lo >= hi or preset[0] >= preset[1]:
                raise ValueError(f"J{i} 关节限位无效")
            common = max(lo, preset[0]), min(hi, preset[1])
            if common[0] >= common[1]:
                raise ValueError(f"J{i} SDK 与控制器限位无交集")
            self.limits.append(common)
        if self.robot == "nero":
            # 优先用按角色标签保存的实测范围；没有保存则回退到内置的 can1 实测值。
            widen = NERO_LIMIT_WIDEN_DEG
            if self.label:
                try:
                    widen = _joint_limits().widen_for_label(self.label, NERO_LIMIT_WIDEN_DEG)
                except Exception:
                    widen = NERO_LIMIT_WIDEN_DEG
            for index, (lo_deg, hi_deg) in widen.items():
                lo, hi = self.limits[index - 1]
                self.limits[index - 1] = (
                    math.radians(lo_deg) if lo_deg is not None and math.radians(lo_deg) < lo else lo,
                    math.radians(hi_deg) if hi_deg is not None and math.radians(hi_deg) > hi else hi,
                )

    def _identify_rest_pose(self, pose: list[float]) -> None:
        """Recognize the local NERO folded pose, not a wider planning workspace.

        These feedback windows cover the four installed NERO arms at rest. They
        are identification tolerances, not manufacturer travel limits. Only the
        wrist gesture may use this envelope; policy/home targets may not.
        """
        if all(lo <= value <= hi for value, (lo, hi) in zip(pose, self.limits, strict=True)):
            return
        if self.robot != "nero":
            return
        reference = (0, -100, 0, 123, 0, 0, 95)
        tolerance = (5, 2, 5, 4, 5, 5, 3)
        degrees = [math.degrees(value) for value in pose]
        if any(abs(value - center) > width for value, center, width in zip(
            degrees, reference, tolerance, strict=True
        )):
            return
        # Only the observed outward offsets of J2/J4/J7 are admitted. A narrower
        # controller range or an unrelated out-of-range axis is still rejected.
        for i, (value, (lo, hi)) in enumerate(zip(pose, self.limits, strict=True)):
            lower_margin = math.radians(2) if i == 1 else 0
            upper_margin = math.radians({3: 4, 6: 8}.get(i, 0))
            if not lo - lower_margin <= value <= hi + upper_margin:
                return
        # Do not turn off SDK checks. Refuse if this SDK would alter the measured
        # whole-arm target: that could move non-wrist joints.
        if self.arm.get_joint_limits_enabled() is not False:
            raise ValueError("NERO 收拢姿态：SDK 已开启目标裁剪，不能保证其余关节原位，未执行")
        self.rest_origin = list(pose)

    def _bounded(self, pose: list[float]) -> list[float]:
        if self.rest_origin is not None:
            for i, (value, origin) in enumerate(zip(pose, self.rest_origin, strict=True)):
                lo, hi = (origin - math.radians(5.5), origin + math.radians(0.3)) if i == 6 \
                    else (origin - math.radians(0.5), origin + math.radians(0.5))
                if not lo <= value <= hi:
                    raise ValueError(f"J{i + 1} 偏离收拢姿态微动范围，已中止")
            return list(pose)
        bounded = []
        for i, (value, (lo, hi)) in enumerate(zip(pose, self.limits, strict=True), 1):
            excess = max(lo - value, value - hi, 0.0)
            if excess > FEEDBACK_LIMIT_TOLERANCE_RAD:
                raise ValueError(
                    f"J{i} 当前 {math.degrees(value):.2f}° 超出允许范围 "
                    f"[{math.degrees(lo):.2f}°, {math.degrees(hi):.2f}°]，超出 {math.degrees(excess):.4f}°；"
                    f"请核对零位与限位"
                )
            bounded.append(max(lo, min(hi, value)))
        return bounded

    def _ready(self, observation: tuple) -> list[float]:
        pose, controller, enabled, _ = observation
        if controller.arm_status != 0 or not all(enabled):
            raise ValueError("控制器尚未正常或关节未全部使能")
        if controller.ctrl_mode != 1 and not (controller.ctrl_mode == 6 and self.following):
            raise ValueError("未确认从臂受控模式")
        return self._bounded(pose)

    def _prepare(self, check: Callable, progress: Callable) -> list[float]:
        pose, controller, enabled, _ = self._read_initial(check)
        if controller.arm_status == 6 and any(enabled):
            raise ValueError("抱闸状态异常，请先检查机械臂")
        self._read_limits(check)
        self._identify_rest_pose(pose)
        self._bounded(pose)
        # Re-read after querying limits; enabling must not use an old pose.
        pose, controller, enabled, _ = self._read_initial(check)
        self._bounded(pose)
        progress("校验通过，正在切换从臂模式并使能…")
        check()
        self.touched = True
        self.arm.set_follower_mode()
        self.following = True
        self.fresh.seen.clear()
        self._read_initial(check)
        deadline = time.monotonic() + 3
        last_enable = 0.0
        while True:
            check()
            observation = self._observe()
            current, _, enabled, _ = observation
            self._bounded(current)
            if max(abs(a - b) for a, b in zip(current, pose, strict=True)) > math.radians(0.5):
                raise ValueError("准备期间机械臂位置变化，已取消微动")
            controlled = observation[1].ctrl_mode in (1, 6)
            if all(enabled) and observation[1].arm_status == 0 and controlled:
                return self._ready(observation)
            if time.monotonic() >= deadline:
                raise ValueError("使能超时，未开始微动")
            if time.monotonic() - last_enable >= 0.2:
                check()
                if self.robot == "nero":
                    self.arm.enable(timeout=0.05)
                else:
                    self.arm.enable()  # Standard Piper has no timeout parameter.
                last_enable = time.monotonic()
            time.sleep(0.02)

    def _wait_target(self, target: list[float], check: Callable, timeout: float = 5) -> None:
        deadline, last_stamp, settled = time.monotonic() + timeout, None, 0
        while True:
            check()
            observation = self._observe()
            current = self._ready(observation)
            # A wrist identification move must not drift the other axes.
            if any(abs(a - b) > math.radians(1) for a, b in zip(current[:-1], target[:-1], strict=True)):
                raise ValueError("其他关节偏移超过 1°，已中止微动")
            if observation[3] != last_stamp:
                settled = settled + 1 if max(abs(a - b) for a, b in zip(current, target, strict=True)) \
                    <= math.radians(0.3) else 0
                last_stamp = observation[3]
            if settled >= 3:
                return
            if time.monotonic() >= deadline:
                raise ValueError("微动到位超时")
            time.sleep(0.02)

    def _emergency(self, message: str) -> tuple[str, str, bool]:
        try:
            self.arm.electronic_emergency_stop()
            detail = "已发送电子急停；检查实物并处理故障后重启平台"
        except Exception as exc:
            detail = f"电子急停发送失败：{exc}；请使用实体急停并检查设备"
        return "error", f"{message}。{detail}", True

    def run(self, check: Callable, progress: Callable) -> tuple[str, str, bool]:
        try:
            progress("正在检查普通关节、驱动、控制器与关节限位…")
            origin = self._prepare(check, progress)
            target = list(origin)
            lo, hi = self.limits[-1]
            delta = math.radians(5)
            # Choose a direction towards the middle of the permitted range.
            target[-1] += -delta if origin[-1] >= (lo + hi) / 2 else delta
            self._bounded(target)
            check()
            self.arm.set_speed_percent(5)
            for pose, text in ((target, f"J{self.count} 微动 5°…"), (origin, "正在返回测试前位置…")):
                self._ready(self._observe())
                check()
                progress(text)
                self.arm.move_j(pose)
                self._wait_target(pose, check)
            prefix = "NERO 收拢姿态 · " if self.rest_origin is not None else ""
            return "done", f"{prefix}J{self.count} 微动完成，已回到测试前位置；请按实物分配用途", False
        except JogStopped as exc:
            if not self.touched:
                return "stopped", f"{exc}；未开始运动", False
            try:
                # Cancellation holds the current measured position; it must not
                # continue the automatic return leg. Feedback is still required.
                hold = self._ready(self._observe())
                self.arm.set_speed_percent(5)
                self.arm.move_j(hold)
                self._wait_target(hold, lambda: None, timeout=1)
                return "stopped", f"{exc}；已在当前位置停止，未执行自动返回", False
            except Exception as failure:
                return self._emergency(f"{exc}；无法确认停稳：{failure}")
        except Exception as exc:
            return self._emergency(str(exc)) if self.touched else ("error", f"未执行：{exc}", False)
