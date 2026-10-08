#!/usr/bin/env python
"""只读验收本地采集数据：索引、状态/动作和每集首中尾视频帧。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from nero_platform import pairing
from nero_platform.runtime import DEFAULT_REPO_ID
from nero_platform.timing import report_path

from lerobot.datasets import LeRobotDataset


def verify_timing(root: Path, episode_index: int, frames: int, fps: int) -> dict | None:
    """核对节拍报告与数据集帧索引，避免用名义 timestamp 推断真实采样频率。"""
    path = report_path(root, episode_index)
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["summary"]
    samples = report["samples"]
    if (
        report.get("schema_version") not in (1, 2, 3)
        or report.get("episode_index") != episode_index
        or summary["frames"] != frames
        or summary["target_fps"] != fps
        or summary["attempts"] != len(samples)
    ):
        raise ValueError("节拍报告与数据集集号、帧数或 FPS 不一致")
    indices = [sample["frame_index"] for sample in samples if sample["frame_index"] is not None]
    if indices != list(range(frames)):
        raise ValueError("节拍报告中的有效帧索引不连续或与数据集不一致")
    previous = 0.0
    previous_slot = -1
    starts = []
    for sample in samples:
        begin, end, slot = sample["started_seconds"], sample["completed_seconds"], sample["slot"]
        if (
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (begin, end, slot))
            or begin < previous
            or end < begin
            or slot <= previous_slot
            or slot != int(slot)
        ):
            raise ValueError("节拍报告含无效或乱序的采样时间")
        previous, previous_slot = end, slot
        if sample["frame_index"] is not None:
            starts.append(begin)
    measured = (frames - 1) / (starts[-1] - starts[0]) if frames > 1 and starts[-1] > starts[0] else None
    stored = summary["effective_fps"]
    if (measured is None and stored is not None) or (
        measured is not None
        and (stored is None or not math.isfinite(stored) or abs(stored - measured) > 0.011)
    ):
        raise ValueError("实测 FPS 与节拍报告的帧时间不一致")
    if report["schema_version"] >= 2:
        valid = [s for s in samples if s["frame_index"] is not None]
        for name, stats in summary["cameras"].items():
            previous_stamp, repeated, max_age = None, 0, 0.0
            for sample in valid:
                camera = sample["cameras"][name]
                stamp, age = camera["capture_timestamp"], camera["age_ms"]
                if (
                    not all(math.isfinite(v) for v in (stamp, age))
                    or age < 0
                    or (previous_stamp is not None and stamp < previous_stamp)
                ):
                    raise ValueError(f"{name} 相机时间戳或帧龄无效")
                same = stamp == previous_stamp
                if camera["repeated"] != same:
                    raise ValueError(f"{name} 相机重复帧标记与时间戳不一致")
                repeated += int(same)
                max_age = max(max_age, age)
                previous_stamp = stamp
            if (
                stats["frames"] != frames
                or stats["repeated_frames"] != repeated
                or stats["max_age_ms"] != max_age
            ):
                raise ValueError(f"{name} 相机新鲜度统计与采样明细不一致")
    control = report.get("control")
    control_cycles = {}
    if control is not None:
        previous_end, previous_slot = 0.0, -1
        for cycle in control["samples"]:
            begin, end, slot = cycle["started_seconds"], cycle["completed_seconds"], cycle["slot"]
            if (
                not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (begin, end, slot))
                or begin < previous_end
                or end < begin
                or slot <= previous_slot
                or slot != int(slot)
            ):
                raise ValueError("控制节拍时间无效或乱序")
            previous_end, previous_slot = end, slot
            index = cycle["frame_index"]
            if index is not None:
                if index != len(control_cycles):
                    raise ValueError("成功控制周期编号不连续")
                control_cycles[index] = cycle
        if control["summary"]["cycles"] != len(control_cycles):
            raise ValueError("控制周期数与明细不一致")
        control_starts = [c["started_seconds"] for c in control_cycles.values()]
        hz = (
            (len(control_starts) - 1) / (control_starts[-1] - control_starts[0])
            if len(control_starts) > 1
            else None
        )
        stored = control["summary"]["effective_hz"]
        if (hz is None and stored is not None) or (
            hz is not None and (stored is None or not math.isfinite(stored) or abs(hz - stored) > 0.011)
        ):
            raise ValueError("实测控制频率与明细不一致")
    if "pair_mode" in report:
        mode = report["pair_mode"]
        keys = pairing.joint_keys(mode)
        expected = ["left", "right"] if mode == "dual" else ["single"]
        if report["joint_names"] != keys or [p["name"] for p in report["pairs"]] != expected:
            raise ValueError("节拍报告中的配对顺序或关节字段不一致")
        for sample in samples:
            if sample["frame_index"] is None:
                continue  # 异常帧可只完成部分读取或部分下发，不当成完整帧验收。
            robots = sample.get("robots", {})
            if set(robots) != set(expected):
                raise ValueError("有效帧缺少配对采样记录")
            bounds = sample
            if control is not None:
                bounds = control_cycles.get(sample.get("control_index"))
                if (
                    bounds is None
                    or robots != bounds["robots"]
                    or bounds["completed_seconds"] > sample["started_seconds"] + 1e-9
                ):
                    raise ValueError("数据帧未关联到已成功下发的控制周期")
                age = (sample["started_seconds"] - bounds["started_seconds"]) * 1000
                if abs(sample["control_age_ms"] - age) > 0.0011:
                    raise ValueError("数据帧中的控制反馈帧龄不一致")
                next_cycle = control_cycles.get(sample["control_index"] + 1)
                if (
                    next_cycle is not None
                    and next_cycle["completed_seconds"] <= sample["started_seconds"] - 1e-9
                ):
                    raise ValueError("数据帧没有使用最近已完成的控制周期")
            read_ends, command_starts = [], []
            for name in expected:
                trace = robots[name]
                times = [
                    trace[key]
                    for key in (
                        "read_started_seconds",
                        "read_completed_seconds",
                        "command_started_seconds",
                        "command_completed_seconds",
                    )
                ]
                if not all(math.isfinite(t) for t in times) or times != sorted(times):
                    raise ValueError("配对读取/下发时间无效或乱序")
                if times[0] < bounds["started_seconds"] or times[-1] > bounds["completed_seconds"]:
                    raise ValueError("配对采样不在共同采样周期内")
                stamps = trace["feedback_timestamps"]
                if set(stamps) != {"主臂关节", "主臂夹爪", "从臂关节", "从臂夹爪"} or not all(
                    math.isfinite(stamp) and stamp > 0 for stamp in stamps.values()
                ):
                    raise ValueError("配对反馈时间戳无效")
                read_ends.append(times[1])
                command_starts.append(times[2])
            if max(read_ends) > min(command_starts):
                raise ValueError("未完成全部反馈校验便开始下发指令")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / ".cache/huggingface/lerobot")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--last-episodes", type=int, help="只对最近 N 集执行视频和节拍检查")
    parser.add_argument("--require-timing", action="store_true", help="要求被检查的集都有实测节拍报告")
    parser.add_argument("--expect-mode", choices=("single", "dual"), help="验证预期的 8 维或 16 维模式")
    args = parser.parse_args()
    if args.last_episodes is not None and args.last_episodes <= 0:
        parser.error("--last-episodes 必须大于 0")
    root = args.dataset_root / args.repo_id
    if not (root / "meta/info.json").is_file():
        raise SystemExit(f"找不到本地数据集: {root}；请先结束采集并断开/退出程序。")
    dataset = LeRobotDataset(args.repo_id, root=root, video_backend="pyav", depth_output_unit="m")
    if args.expected_episodes is not None and dataset.num_episodes != args.expected_episodes:
        raise SystemExit(f"集数不一致：预期 {args.expected_episodes}，实际 {dataset.num_episodes}")
    if dataset.num_frames == 0:
        raise SystemExit("数据集没有帧。")
    table = dataset.hf_dataset
    indices = np.asarray(table["index"])
    episode_ids = np.asarray(table["episode_index"])
    frame_ids = np.asarray(table["frame_index"])
    if not np.array_equal(indices, np.arange(dataset.num_frames)):
        raise SystemExit("全局帧索引有缺失、重复或乱序。")
    if not np.array_equal(np.unique(episode_ids), np.arange(dataset.num_episodes)):
        raise SystemExit("集索引与元信息不一致。")
    mode = "dual" if tuple(dataset.features["action"]["shape"]) == (16,) else "single"
    names = pairing.joint_keys(mode)
    if args.expect_mode and args.expect_mode != mode:
        raise SystemExit(f"采集模式不一致：预期 {args.expect_mode}，实际 {mode}")
    for key in ("action", "observation.state"):
        values = np.asarray(table[key])
        if values.shape != (dataset.num_frames, len(names)) or not np.isfinite(values).all():
            raise SystemExit(f"{key} 维度或数值异常（{mode} 应为 {len(names)} 维）。")
        if dataset.features[key]["names"] != names:
            raise SystemExit(f"{key} 关节顺序不符合 Nero {mode} 数据定义。")
    print(f"数据目录: {root}")
    print(f"LeRobot {dataset.meta.info.codebase_version}，{dataset.num_episodes} 集，{dataset.num_frames} 帧")
    print(f"声明帧率: {dataset.fps} FPS；实测值来自独立的主机节拍报告。")
    print(f"相机字段: {', '.join(dataset.meta.video_keys) or '无'}")
    print(
        f"采集模式: {mode}；状态/动作各 {len(names)} 维"
        + ("，左 8 维在前、右 8 维在后" if mode == "dual" else "")
    )
    first = max(0, dataset.num_episodes - args.last_episodes) if args.last_episodes else 0
    for episode_id in range(first, dataset.num_episodes):
        positions = np.flatnonzero(episode_ids == episode_id)
        if not np.array_equal(frame_ids[positions], np.arange(len(positions))):
            raise SystemExit(f"第 {episode_id + 1} 集帧索引不连续。")
        for position in {positions[0], positions[len(positions) // 2], positions[-1]}:
            frame = dataset[int(position)]
            for key in dataset.meta.video_keys:
                feature = dataset.features[key]
                height, width, channels = feature["shape"]
                if tuple(frame[key].shape) != (channels, height, width):
                    raise SystemExit(f"第 {episode_id + 1} 集 {key} 视频解码尺寸异常。")
                values = frame[key].numpy()
                if not np.isfinite(values).all():
                    raise SystemExit(f"第 {episode_id + 1} 集 {key} 含 NaN/Inf。")
                if feature.get("info", {}).get("is_depth_map"):
                    info = feature["info"]
                    if values.min() < -1e-5 or values.max() > info["video.depth_max"] + 0.01:
                        raise SystemExit(f"第 {episode_id + 1} 集 {key} 深度解码值超出编码范围。")
                    valid = values[values > max(1e-5, info["video.depth_min"] + 1e-5)]
                    print(
                        f"  {key} 帧 {int(frame_ids[position])}：{channels}×{height}×{width}，单位 m，"
                        + (
                            f"非零深度 {valid.min():.3f}～{valid.max():.3f}，"
                            if valid.size
                            else "无有效深度，"
                        )
                        + f"零值 {np.count_nonzero(values <= 1e-5) / values.size:.1%}；12-bit 量化有误差"
                    )
        print(f"第 {episode_id + 1} 集: {len(positions)} 帧，状态/动作可读，视频首/中/尾帧可解码")
        try:
            timing = verify_timing(root, episode_id, len(positions), dataset.fps)
        except (ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"第 {episode_id + 1} 集节拍报告异常：{exc}") from exc
        if timing is None:
            if args.require_timing:
                raise SystemExit(f"第 {episode_id + 1} 集缺少节拍报告")
            print("  未记录实测节拍（旧版本或报告未保存），不能据此验证实际 FPS。")
        else:
            rate = timing["effective_fps"]
            rate_text = f"{rate:.2f} FPS" if rate is not None else "有效帧不足，无法估算"
            print(
                f"  实测采样: {rate_text}；最大帧间隔 {timing['max_interval_ms']} ms；错过时刻 {timing['skipped_slots']} 次"
            )
            print(
                f"  实际录制 {timing['elapsed_seconds']:.3f} 秒；按数据集 FPS 播放 {timing['nominal_playback_seconds']:.3f} 秒"
            )
            for warning in timing["warnings"]:
                print(f"  节拍提示：{warning}")
            for name, stats in timing.get("cameras", {}).items():
                print(
                    f"  {name} 相机：{stats['frames']} 帧，重复取用 {stats['repeated_frames']} 帧，最大帧龄 {stats['max_age_ms']} ms"
                )
    print("文件可读性检查通过；节拍提示见各集结果，画面内容与传感器真实同步仍需单独验证。")


if __name__ == "__main__":
    main()
