"""RGBD 时间戳有效性；时差检查不等同于同步曝光验证。"""

import math
from typing import Any

CLOCK_FIELDS = {
    "device": ("rgb_timestamp_us", "depth_timestamp_us"),
    "global": ("rgb_global_timestamp_us", "depth_global_timestamp_us"),
}
CHECK_FIELDS = ("timestamp_check_version", "timestamp_status", "timestamp_source", "rgb_depth_skew_ms")
STATUS_TEXT = {
    "valid": "已检查有效时间戳时差，未验证同步曝光",
    "warming_up": "等待连续时间戳，时间同步未验证",
    "unavailable": "时间戳不可用，时间同步未验证",
    "stalled": "时间戳未推进，时间同步未验证",
    "regressed": "时间戳倒退，时间同步未验证",
}


def positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def receive_skew_ms(sample: dict) -> float | None:
    """同一主机时钟的接收时差，用于识别队列积压，不作为曝光同步证据。"""
    rgb, depth = (sample.get(f"{name}_system_timestamp_us") for name in ("rgb", "depth"))
    return abs(rgb - depth) / 1000 if positive(rgb) and positive(depth) else None


def receive_skew_limit_ms(fps: float) -> float:
    # 容忍正常 USB 抖动；低帧率至少允许三个周期，不要求亚帧同步。
    return max(100.0, 3000 / fps)


def add_receive_stats(stats: dict, sample: dict) -> None:
    skew = receive_skew_ms(sample)
    if skew is not None:
        stats["max_rgb_depth_receive_skew_ms"] = max(stats.get("max_rgb_depth_receive_skew_ms", 0), skew)


def pair(sample: dict, source: str) -> tuple[int | float, int | float] | None:
    values = tuple(sample.get(key) for key in CLOCK_FIELDS[source])
    return values if all(positive(value) for value in values) else None


def checked_sample(sample: dict) -> dict | None:
    """读取新版后台检查结果；旧报告交给 Tracker 按采样顺序重算。"""
    if sample.get("timestamp_check_version") != 1:
        return None
    status, source = sample.get("timestamp_status"), sample.get("timestamp_source")
    values = pair(sample, source) if source in CLOCK_FIELDS else None
    if status == "valid" and values is not None:
        skew = abs(values[0] - values[1]) / 1000
    else:
        status = status if status in STATUS_TEXT and status != "valid" else "unavailable"
        source, skew = None, None
    return {
        "timestamp_check_version": 1,
        "timestamp_status": status,
        "timestamp_source": source,
        "rgb_depth_skew_ms": skew,
    }


class TimestampTracker:
    def __init__(self) -> None:
        self.previous: dict[str, tuple[int | float, int | float]] = {}
        self.last_capture: float | None = None
        self.last_result: dict | None = None

    def assess(self, sample: dict) -> dict:
        # 录制循环重复取用同一缓存帧不算设备时钟停滞，重复帧另行统计。
        capture = sample.get("capture_timestamp")
        if capture is not None and capture == self.last_capture and self.last_result is not None:
            return self.last_result.copy()
        candidates = []
        for source in CLOCK_FIELDS:
            values = pair(sample, source)
            if values is None:
                self.previous.pop(source, None)
                continue
            previous = self.previous.get(source)
            self.previous[source] = values
            if previous is None:
                status = "warming_up"
            elif any(value < old for value, old in zip(values, previous, strict=True)):
                status = "regressed"
            elif any(value == old for value, old in zip(values, previous, strict=True)):
                status = "stalled"
            else:
                status = "valid"
            candidates.append((source, status, values))
        source, skew = None, None
        status = "unavailable"
        for name, candidate, values in candidates:
            if candidate == "valid":
                source, status = name, candidate
                skew = abs(values[0] - values[1]) / 1000
                break
        else:
            if candidates:
                status = candidates[0][1]
        result = {
            "timestamp_check_version": 1,
            "timestamp_status": status,
            "timestamp_source": source,
            "rgb_depth_skew_ms": skew,
        }
        self.last_capture, self.last_result = capture, result
        return result.copy()


def add_stats(stats: dict, check: dict) -> None:
    stats.setdefault("timestamp_checked_frames", 0)
    stats.setdefault("timestamp_unverified_frames", 0)
    stats.setdefault("max_rgb_depth_skew_ms", None)
    for source in CLOCK_FIELDS:
        stats.setdefault(f"timestamp_{source}_frames", 0)
    if check["timestamp_status"] == "valid":
        stats["timestamp_checked_frames"] += 1
        stats[f"timestamp_{check['timestamp_source']}_frames"] += 1
        stats["max_rgb_depth_skew_ms"] = max(stats["max_rgb_depth_skew_ms"] or 0, check["rgb_depth_skew_ms"])
    else:
        stats["timestamp_unverified_frames"] += 1


def warnings(stats: dict, fps: float, label: str = "头部") -> list[str]:
    messages = []
    receive_skew = stats.get("max_rgb_depth_receive_skew_ms")
    if receive_skew is not None and receive_skew > receive_skew_limit_ms(fps):
        messages.append(
            f"{label} RGB/深度主机接收时差达 {receive_skew:.1f} ms，可能存在旧帧积压；接收时差不等于曝光时差"
        )
    if stats.get("timestamp_unverified_frames"):
        messages.append(
            f"{label} RGB/深度有 {stats['timestamp_unverified_frames']} 帧时间戳不可用或未通过有效性检查，时间同步未验证；不代表图像保存失败"
        )
    skew = stats.get("max_rgb_depth_skew_ms")
    if skew is not None and skew > 1000 / fps:
        messages.append(f"{label} RGB/深度有效时间戳差超过一个采样周期，请复核配对")
    return messages
