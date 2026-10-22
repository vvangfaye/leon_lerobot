"""关节限位测量与持久化。

每台臂按角色标签（主左/主右/从左/从右）把实测拖动范围保存到
``~/.cache/lerobot/limits/<标签>.json``，后续采集/推理平台按标签读取，
不用每次重测，也避免把「这台臂」的限位误用到「那台臂」上。

测量只读，不切模式、不发送控制；调用方负责先使能并切到可拖动模式。
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

NAMES = tuple(f"J{i}" for i in range(1, 8))
PACKETS = ("12", "34", "56", "7")
STALE_SECONDS = 0.5

LIMITS_DIR = Path.home() / ".cache" / "lerobot" / "limits"

# 关节名（1-based）→ 放宽方向。只放宽实测明显超出 SDK 预设、且训练数据会逼近的关节。
# 采集平台只存原始实测范围；推理平台用它来替换硬编码的 NERO_LIMIT_WIDEN_DEG。
WIDEN_JOINTS = {2: ("lo", None), 4: (None, "hi"), 7: ("lo", "hi")}

# 推理平台的角色 → 采集平台的角色标签（限位文件按标签命名）。
ROLE_TO_LABEL = {
    "left_leader": "主左", "right_leader": "主右",
    "left_follower": "从左", "right_follower": "从右",
}


class AngleReader:
    """V1.21 的两种反馈，只接受完整且持续更新的七关节数据。

    SDK 聚合 getter 的 timestamp 只代表最后一组，并可能用零补齐缺失组。
    因此检查 SDK parser 的四组时间戳，避免把旧值/补零当极值。
    """

    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self.seen: dict[str, tuple[float, float, bool]] = {}
        self.accepted: dict[str, tuple] = {}
        self.diagnostics = "等待关节反馈"

    def _fresh(self, key: str, packet: Any, now: float) -> float | None:
        stamp = float(getattr(packet, "timestamp", 0))
        if not math.isfinite(stamp) or stamp <= 0 or getattr(packet, "msg", None) is None:
            return None
        previous = self.seen.get(key)
        if previous is None or stamp < previous[0]:
            self.seen[key] = (stamp, now, False)
        elif stamp > previous[0]:
            self.seen[key] = (stamp, now, True)
        current = self.seen[key]
        return stamp if current[2] and now - current[1] <= STALE_SECONDS else None

    def read(self, now: float) -> tuple[tuple[float, ...], str] | None:
        parser = getattr(self.arm, "_parser", None)
        if parser is None:
            raise RuntimeError("当前 SDK 缺少 V1.21 反馈解析器，无法确认七关节反馈完整性")
        candidates, details = [], []
        for source, prefix, method in (
            ("拖动反馈", "leader_joint_", "get_leader_joint_angles"),
            ("普通反馈", "joint_", "get_joint_angles"),
        ):
            stamps, missing = [], []
            try:
                for suffix in PACKETS:
                    key = prefix + suffix
                    stamp = self._fresh(key, getattr(parser, key, None), now)
                    stamps.append(stamp)
                    if stamp is None:
                        missing.append("/".join(f"J{i}" for i in suffix))
                if missing:
                    details.append(f"{source}：等待 {'、'.join(missing)} 的新反馈")
                    continue
                sample = getattr(self.arm, method)()
                angles = tuple(float(v) for v in sample.msg)
                if len(angles) != 7 or not all(math.isfinite(v) for v in angles):
                    raise ValueError("关节角数量或数值无效")
                signature = tuple(stamps)
                details.append(f"{source}：7/7 正在更新")
                if signature != self.accepted.get(source):
                    candidates.append((angles, source, signature))
            except (AttributeError, TypeError, ValueError, OSError) as exc:
                details.append(f"{source}：{exc}")
        self.diagnostics = "；".join(details)
        if not candidates:
            return None
        angles, source, _ = candidates[0]
        for _, name, stamps in candidates:
            self.accepted[name] = stamps
        return angles, source


class JointRanges:
    def __init__(self) -> None:
        self.minimum = [math.inf] * 7
        self.maximum = [-math.inf] * 7
        self.count = 0
        self.sources: set[str] = set()

    def add(self, angles: tuple[float, ...], source: str) -> list[float]:
        degrees = [math.degrees(value) for value in angles]
        for i, value in enumerate(degrees):
            self.minimum[i] = min(self.minimum[i], value)
            self.maximum[i] = max(self.maximum[i], value)
        self.count += 1
        self.sources.add(source)
        return degrees

    def report(self, channel: str) -> dict:
        return {
            "channel": channel,
            "unit": "degrees",
            "valid_samples": self.count,
            "sources": sorted(self.sources),
            "kind": "observed_manual_range",
            "note": "仅为本次手动拖动的观测范围，未修改 SDK 或控制器限位。",
            "joints": {
                name: ({"min": self.minimum[i], "max": self.maximum[i], "span": self.maximum[i] - self.minimum[i]}
                       if self.count else None)
                for i, name in enumerate(NAMES)
            },
        }


def measure(
    arm: Any,
    stop: threading.Event,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """循环读数直到 stop 置位；返回 JointRanges.report。"""
    reader, ranges = AngleReader(arm), JointRanges()
    last_update = None
    last_progress = 0.0
    while not stop.is_set():
        now = time.monotonic()
        sample = reader.read(now)
        if sample is not None:
            angles, source = sample
            ranges.add(angles, source)
            last_update = now
        if progress is not None and now - last_progress >= 0.5:
            last_progress = now
            if last_update is None or now - last_update > STALE_SECONDS:
                progress(f"等待有效角度，暂停累计：{reader.diagnostics}")
            else:
                progress(
                    "  ".join(
                        f"{n}[{lo:.1f}, {hi:.1f}]°"
                        for n, lo, hi in zip(NAMES, ranges.minimum, ranges.maximum, strict=True)
                    )
                )
        time.sleep(0.02)
    return ranges.report(getattr(arm, "_channel", "?"))


def limits_path(label: str) -> Path:
    return LIMITS_DIR / f"{label}.json"


def load_limits(label: str) -> dict | None:
    """读取已保存的实测范围；不存在或损坏返回 None。"""
    path = limits_path(label)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_limits(label: str, report: dict) -> Path:
    """按标签保存实测范围，附带测量时间与通道（用于换臂提醒）。"""
    LIMITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "channel": report.get("channel"),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "valid_samples": report.get("valid_samples", 0),
        "unit": "degrees",
        "joints": report.get("joints", {}),
    }
    path = limits_path(label)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def widen_from_report(report: dict) -> dict[int, tuple[float | None, float | None]]:
    """从实测范围生成推理平台的放宽字典（J2 下界 / J4 上界 / J7 双侧）。

    放宽值向内收 0.05° 留安全余量；只返回实测确有数据的关节。
    """
    joints = report.get("joints", {})
    widen: dict[int, tuple[float | None, float | None]] = {}
    margin = 0.05
    for index, (lo_want, hi_want) in WIDEN_JOINTS.items():
        entry = joints.get(NAMES[index - 1])
        if not entry:
            continue
        # 向内收余量：下界取 min+margin（更靠内、不太负），上界取 max-margin（更靠内、不太正）。
        lo = round(entry["min"] + margin, 1) if lo_want and math.isfinite(entry["min"]) else None
        hi = round(entry["max"] - margin, 1) if hi_want and math.isfinite(entry["max"]) else None
        if lo is None and hi is None:
            continue
        widen[index] = (lo, hi)
    return widen


def widen_for_label(label: str, fallback: dict | None = None) -> dict:
    """按角色标签读取已保存实测范围并转成放宽字典；无保存或无效时用 fallback。"""
    saved = load_limits(label)
    if not saved or not saved.get("valid_samples") or not saved.get("joints"):
        return fallback or {}
    return widen_from_report(saved) or fallback or {}
