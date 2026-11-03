"""Asynchronous RTC playback for the platform's existing device owners."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.policies.rtc.configuration_rtc import validate_trained_rtc_horizon

from .motion import plan


@dataclass(frozen=True)
class RTCOptions:
    enabled: bool = False
    horizon: int = 10
    mode: str = "guided"

    @classmethod
    def from_settings(cls, settings: dict, steps: int, training_max_delay: int = 0) -> RTCOptions:
        enabled = settings.get("rtcEnabled", False)
        horizon = settings.get("rtcHorizon", min(10, max(1, steps - 1)))
        if not isinstance(enabled, bool) or type(horizon) is not int:
            raise ValueError("RTC 开关或前缀长度无效")
        if enabled and not 1 <= horizon < steps:
            raise ValueError("RTC 前缀长度必须小于模型动作段长度")
        mode = "trained" if training_max_delay > 0 else "guided"
        if enabled and mode == "trained":
            validate_trained_rtc_horizon(horizon, steps, training_max_delay)
        return cls(enabled, horizon, mode)

    def policy_config(self) -> dict:
        return {"enabled": True, "mode": self.mode, "execution_horizon": self.horizon}


class RTCBuffer:
    """A popped endpoint is committed; prediction can replace only future endpoints."""

    def __init__(self, initial: np.ndarray) -> None:
        self.anchor = initial.copy()
        self.cursor = 0
        self.future: deque[tuple[np.ndarray, float]] = deque()

    def prefix(self, horizon: int) -> np.ndarray | None:
        items = list(self.future)[:horizon]
        return np.stack([item[0].reshape(-1) for item in items]) if items else None

    def merge(self, actions: np.ndarray, requested_cursor: int, planner: Callable) -> tuple[bool, int, list]:
        skipped = self.cursor - requested_cursor
        if skipped < 0:
            raise ValueError("RTC 动作游标倒退")
        if skipped >= len(actions):
            return False, skipped, []
        points, durations, warnings = planner(actions[skipped:], self.anchor)
        self.future = deque(
            (point.copy(), float(duration)) for point, duration in zip(points[1:], durations, strict=True)
        )
        return True, skipped, warnings

    def pop(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not self.future:
            return None
        previous = self.anchor.copy()
        target, duration = self.future.popleft()
        self.anchor = target.copy()
        self.cursor += 1
        return previous, target, duration


def run_rtc(runtime: Any, kind: str, settings: dict, limits: np.ndarray, options: RTCOptions) -> list[str]:
    """Keep motor interpolation on the control thread; only policy RPC runs in background."""
    initial = np.asarray([value["point"] for value in runtime._commands("read").values()])
    buffer = RTCBuffer(initial)
    runtime._rtc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-rtc")
    runtime._rtc_future = None
    requested_cursor, requested_at, last_delay = 0, 0.0, 0
    accepted_cursor = 0
    segment = None
    segment_start = 0.0
    last_send = None
    starving_since = None
    warnings: list[str] = []
    runtime.rtc_stats = {
        "enabled": True,
        "mode": options.mode,
        "horizon": options.horizon,
        "updates": 0,
        "discarded": 0,
        "skipped_actions": 0,
        "starvations": 0,
    }

    while True:
        runtime._check()
        now = time.monotonic()
        pending = runtime._rtc_future
        if pending is not None and pending.done():
            result = pending.result()
            runtime._rtc_future = None
            if now - requested_at > 10:
                raise ValueError("RTC 预测结果已过期")
            actions = runtime.contract.actions(result["actions"])
            merged, skipped, notes = buffer.merge(
                actions,
                requested_cursor,
                lambda values, anchor: plan(values, runtime.contract, anchor, limits, runtime._check),
            )
            warnings.extend(notes)
            last_delay = skipped
            runtime.rtc_stats["skipped_actions"] += skipped
            runtime.inference_ms = result["inference_ms"]
            if merged:
                runtime.chunks += 1
                runtime.rtc_stats["updates"] += 1
                accepted_cursor = buffer.cursor
                if runtime.recorder:
                    runtime.recorder.chunk(actions, runtime.inference_ms)
                runtime.job.progress(f"RTC 连续执行 · 已更新 {runtime.chunks} 段")
            else:
                runtime.rtc_stats["discarded"] += 1

        if runtime._rtc_future is None and (
            not runtime.chunks
            or buffer.cursor - accepted_cursor >= options.horizon
            or len(buffer.future) <= options.horizon
        ):
            observation = runtime._capture()
            prefix = buffer.prefix(options.horizon)
            if prefix is not None and runtime.contract.config["space"] == "eef":
                # The queue contains planned joints; EEF policies need those
                # executed targets mapped back through the existing FK contract.
                canonical = []
                for points in prefix.reshape(-1, *initial.shape):
                    samples = {
                        identity: {"joint_radians": point[:-1], "gripper_m": float(point[-1])}
                        for identity, point in zip(runtime.arms.values(), points, strict=True)
                    }
                    canonical.append(runtime.contract.state(samples, runtime.arms))
                prefix = np.asarray(canonical)
            requested_cursor, requested_at = buffer.cursor, time.monotonic()
            delay = min(len(prefix), max(1, last_delay + 1)) if prefix is not None else 0
            request = {
                "op": "predict",
                "state": observation["state"],
                "images": observation["images"],
                "task": settings["task"],
                "rtc": options.policy_config(),
                "prefix": prefix,
                "inference_delay": delay,
            }
            runtime._rtc_future = runtime._rtc_executor.submit(
                runtime.policy.call, request, runtime._check_policy
            )
        if runtime._rtc_future is not None and now - requested_at > 10:
            raise ValueError("RTC 预测超时，停止控制")

        if segment is None and buffer.future:
            segment = buffer.pop()
            segment_start = now
        while segment is not None and now >= segment_start + segment[2]:
            segment_start += segment[2]
            segment = buffer.pop()

        target = (
            buffer.anchor
            if segment is None
            else (segment[0] + (segment[1] - segment[0]) * np.clip((now - segment_start) / segment[2], 0, 1))
        )
        if runtime.chunks:
            if last_send is not None and now - last_send > 0.15:
                raise ValueError("RTC 控制循环超过 150 ms 未更新")
            runtime._commands("send", dict(zip(runtime.arms, target.tolist(), strict=True)))
            runtime._set_command(target)
            last_send = time.monotonic()
            if segment is None:
                if starving_since is None:
                    starving_since = now
                    runtime.rtc_stats["starvations"] += 1
                elif now - starving_since > 1:
                    raise ValueError("RTC 动作缓冲区持续为空，已停止")
            else:
                starving_since = None
        if kind == "single" and time.monotonic() - runtime.started >= settings.get("duration", 60):
            return warnings
        runtime.job.cancel.wait(max(0, 1 / 30 - (time.monotonic() - now)))
