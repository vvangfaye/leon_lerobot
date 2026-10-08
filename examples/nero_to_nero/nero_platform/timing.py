"""采样调度和主机侧节拍报告；不把主机循环时间当作传感器同步时间。"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import rgbd_timing
from .camera_model import is_depth, stream_label


@dataclass
class EpisodeTiming:
    fps: int
    started_at: float
    next_slot: int = 0
    skipped_slots: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    frames: int = 0
    first_sample: float | None = None
    last_sample: float | None = None
    max_interval: float = 0.0
    total_work: float = 0.0
    max_work: float = 0.0
    over_budget: int = 0
    camera_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    camera_stamps: dict[str, float] = field(default_factory=dict)
    timestamp_trackers: dict[str, rgbd_timing.TimestampTracker] = field(default_factory=dict)

    @property
    def deadline(self) -> float:
        return self.started_at + self.next_slot / self.fps

    def begin(self, now: float) -> int:
        """只执行当前采样时刻，错过的时刻计数，不循环补发旧帧。"""
        slot = max(self.next_slot, math.floor((now - self.started_at) * self.fps + 1e-7))
        self.skipped_slots += slot - self.next_slot
        self.next_slot = slot + 1
        return slot

    def finish(
        self,
        slot: int,
        started: float,
        completed: float,
        frame_index: int | None,
        error: str | None,
        cameras: dict[str, dict[str, Any]] | None = None,
        robots: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        work = max(0.0, completed - started)
        self.total_work += work
        self.max_work = max(self.max_work, work)
        self.over_budget += int(work > 1 / self.fps + 1e-9)
        if frame_index is not None:
            if self.last_sample is not None:
                self.max_interval = max(self.max_interval, started - self.last_sample)
            else:
                self.first_sample = started
            self.last_sample = started
            self.frames += 1
            for name, sample in (cameras or {}).items():
                stats = self.camera_stats.setdefault(
                    name, {"frames": 0, "repeated_frames": 0, "max_age_ms": 0.0}
                )
                stamp = sample["capture_timestamp"]
                repeated = self.camera_stamps.get(name) == stamp
                sample["repeated"] = repeated
                stats["frames"] += 1
                stats["repeated_frames"] += int(repeated)
                stats["max_age_ms"] = max(stats["max_age_ms"], sample["age_ms"])
                if "invalid_fraction" in sample:
                    stats["max_invalid_fraction"] = max(
                        stats.get("max_invalid_fraction", 0), sample["invalid_fraction"]
                    )
                    stats["all_invalid_frames"] = stats.get("all_invalid_frames", 0) + int(
                        sample["invalid_fraction"] >= 1
                    )
                    stats["above_range_frames"] = stats.get("above_range_frames", 0) + int(
                        sample["above_range_fraction"] > 0
                    )
                    check = rgbd_timing.checked_sample(sample)
                    if check is None:
                        tracker = self.timestamp_trackers.setdefault(name, rgbd_timing.TimestampTracker())
                        check = tracker.assess(sample)
                    sample.update(check)
                    rgbd_timing.add_stats(stats, check)
                    rgbd_timing.add_receive_stats(stats, sample)
                self.camera_stamps[name] = stamp
        self.samples.append(
            {
                "slot": slot,
                "scheduled_seconds": slot / self.fps,
                "started_seconds": started - self.started_at,
                "completed_seconds": completed - self.started_at,
                "frame_index": frame_index,
                "error": error,
                "cameras": cameras or {},
                "robots": robots or {},
            }
        )

    def finish_duration(self, duration: float) -> None:
        """到时停止时也统计被最后一次慢采样占用的剩余时刻。"""
        slots = math.ceil(duration * self.fps - 1e-7)
        self.skipped_slots += max(0, slots - self.next_slot)
        self.next_slot = max(slots, self.next_slot)

    def summary(self, elapsed: float) -> dict[str, Any]:
        span = (self.last_sample - self.first_sample) if self.frames > 1 else 0.0
        rate = (self.frames - 1) / span if span > 0 else None
        warnings = []
        if rate is not None and rate < self.fps * 0.95:
            warnings.append("实测采样 FPS 低于目标的 95%")
        if self.skipped_slots:
            warnings.append(f"错过 {self.skipped_slots} 个计划采样时刻")
        if self.over_budget:
            warnings.append(f"有 {self.over_budget} 次处理耗时超过一个采样周期")
        if self.max_interval > 1.5 / self.fps:
            warnings.append("有效帧之间出现超过 1.5 个采样周期的间隔")
        failed = len(self.samples) - self.frames
        if failed:
            warnings.append(f"有 {failed} 次采样未写入有效帧")
        for name, stats in self.camera_stats.items():
            if stats["repeated_frames"]:
                warnings.append(
                    f"{name} 相机重复取用缓存帧 {stats['repeated_frames']} 次，请复核图像更新情况"
                )
            if is_depth(name):
                if stats.get("all_invalid_frames"):
                    warnings.append(
                        f"{stream_label(name)}有 {stats['all_invalid_frames']} 帧全部无效，请检查测距"
                    )
                if stats.get("above_range_frames"):
                    warnings.append(
                        f"{stream_label(name)}有 {stats['above_range_frames']} 帧含超过 10 米的值，编码会饱和到 10 米"
                    )
                warnings.extend(rgbd_timing.warnings(stats, self.fps, stream_label(name)))
        return {
            "target_fps": self.fps,
            "effective_fps": round(rate, 2) if rate is not None else None,
            "frames": self.frames,
            "attempts": len(self.samples),
            "elapsed_seconds": round(max(0.0, elapsed), 6),
            "nominal_playback_seconds": round(self.frames / self.fps, 6),
            "max_interval_ms": round(self.max_interval * 1000, 2) if self.frames > 1 else None,
            "mean_processing_ms": round(self.total_work / len(self.samples) * 1000, 2)
            if self.samples
            else None,
            "max_processing_ms": round(self.max_work * 1000, 2) if self.samples else None,
            "over_budget_samples": self.over_budget,
            "skipped_slots": self.skipped_slots,
            "failed_samples": failed,
            "warnings": warnings,
            "cameras": {name: dict(stats) for name, stats in self.camera_stats.items()},
        }


class ControlTiming(EpisodeTiming):
    """控制周期单独计时；不将每次指令下发计作数据集帧。"""

    def control_summary(self, elapsed: float) -> dict[str, Any]:
        stats = self.summary(elapsed)
        warnings = []
        rate = stats["effective_fps"]
        if rate is not None and rate < self.fps * 0.95:
            warnings.append("实测跟随控制频率低于目标的 95%")
        if self.skipped_slots:
            warnings.append(f"错过 {self.skipped_slots} 个计划控制时刻")
        if self.max_interval > 1.5 / self.fps:
            warnings.append("控制指令间隔超过 1.5 个周期")
        if stats["failed_samples"]:
            warnings.append("存在未完成的控制周期")
        return {
            "target_hz": self.fps,
            "effective_hz": rate,
            "cycles": self.frames,
            "attempts": len(self.samples),
            "max_interval_ms": stats["max_interval_ms"],
            "mean_processing_ms": stats["mean_processing_ms"],
            "max_processing_ms": stats["max_processing_ms"],
            "skipped_slots": self.skipped_slots,
            "warnings": warnings,
        }


def report_path(root: Path, episode_index: int) -> Path:
    return root / "meta" / "nero_timing" / f"episode_{episode_index:06d}.json"


def save_report(root: Path, episode_index: int, report: dict[str, Any]) -> Path:
    """保存已成功写入的数据集对应的诊断文件；原子发布，不覆盖已有报告。"""
    path = report_path(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
