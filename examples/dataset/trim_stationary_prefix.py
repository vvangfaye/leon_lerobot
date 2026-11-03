"""Trim stationary starts from a joint-space LeRobot dataset into a new dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)


def trim_index(actions: np.ndarray, fps: int, context_s: float = 0.2) -> int:
    """Find a sustained 2-degree / 3-mm change, retaining a short lead-in."""
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise ValueError("Expected finite seven-joint plus gripper actions")
    if not len(actions):
        raise ValueError("Empty episode")
    changed = (np.abs(actions[:, :7] - actions[0, :7]).max(axis=1) > math.radians(2)) | (
        np.abs(actions[:, 7] - actions[0, 7]) > 0.003
    )
    sustain = max(1, round(0.1 * fps))
    if len(changed) < sustain:
        return 0
    starts = np.flatnonzero(np.convolve(changed.astype(int), np.ones(sustain, dtype=int), "valid") == sustain)
    return max(0, int(starts[0]) - round(context_s * fps)) if len(starts) else 0


def numeric_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
        **{f"q{int(q * 100):02d}": np.quantile(values, q, axis=0) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
    }


def trim_video(source: Path, output: Path, cut: int, original_length: int, fps: int) -> dict[str, np.ndarray]:
    """Decode by frame index, reset PTS, and recompute sampled RGB statistics."""
    output.parent.mkdir(parents=True, exist_ok=True)
    length = original_length - cut
    samples = set(sample_indices(length))
    images = []
    source_count = kept_count = 0
    with av.open(str(source)) as src, av.open(str(output), "w", options={"movflags": "faststart"}) as dst:
        input_stream = src.streams.video[0]
        input_stream.codec_context.thread_count = 2
        stream = dst.add_stream(
            "libx264",
            rate=Fraction(fps),
            options={
                "crf": "18",
                "preset": "veryfast",
                "g": "2",
                "threads": "2",
            },
        )
        stream.width, stream.height = input_stream.width, input_stream.height
        stream.pix_fmt, stream.time_base = "yuv420p", Fraction(1, fps)
        for frame in src.decode(input_stream):
            frame_index = source_count
            source_count += 1
            if frame_index < cut:
                continue
            if kept_count in samples:
                rgb = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                images.append(auto_downsample_height_width(rgb).copy())
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = kept_count, Fraction(1, fps)
            for packet in stream.encode(frame):
                dst.mux(packet)
            kept_count += 1
        for packet in stream.encode():
            dst.mux(packet)
    if source_count != original_length or kept_count != length:
        raise ValueError(
            f"Frame count mismatch in {source}: {source_count=}, {original_length=}, {kept_count=}"
        )
    stats = get_feature_stats(np.stack(images), axis=(0, 2, 3), keepdims=True)
    return {
        key: np.array([length]) if key == "count" else np.squeeze(value / 255, axis=0)
        for key, value in stats.items()
    }


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(source: Path, output: Path, audit: Path, *, plan_only: bool = False, workers: int = 2) -> None:
    if not 1 <= workers <= 16:
        raise ValueError("Video worker count must be between 1 and 16")
    source, output = source.resolve(), output.resolve()
    building = output.with_name(output.name + ".building")
    if source == output or output.is_relative_to(source) or output.exists() or building.exists():
        raise ValueError("Output must be new and different from source; no overwrite is supported")
    audit.mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text())
    fps = int(info["fps"])
    expected_names = [*[f"joint{i}.pos" for i in range(1, 8)], "gripper.pos"]
    if info["features"]["action"].get("names") != expected_names:
        raise ValueError("This trimmer requires explicit seven-joint / gripper action names")
    paths = sorted((source / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in paths]).sort_by("index")
    episode_table = pa.concat_tables(
        [pq.read_table(path) for path in sorted((source / "meta/episodes").rglob("*.parquet"))]
    )
    episodes = sorted(episode_table.to_pylist(), key=lambda row: row["episode_index"])
    columns = table.to_pydict()
    episode_indices = np.asarray(columns["episode_index"])
    camera_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    plan = []
    for episode in episodes:
        indices = np.flatnonzero(episode_indices == episode["episode_index"])
        actions = np.asarray([columns["action"][index] for index in indices])
        cut = trim_index(actions, fps)
        if len(indices) - cut < 50:
            raise ValueError("Trimming would leave fewer than 50 frames")
        plan.append(
            {
                "episode": episode["episode_index"],
                "original_frames": len(indices),
                "cut_frames": cut,
                "cut_seconds": cut / fps,
                "kept_frames": len(indices) - cut,
            }
        )
    report = {
        "source": str(source),
        "output": str(output),
        "fps": fps,
        "method": "2-degree / 3-mm cumulative action change sustained for 0.1 s; retain 0.2 s lead-in",
        "episodes": plan,
        "original_frames": table.num_rows,
        "removed_frames": sum(item["cut_frames"] for item in plan),
    }
    (audit / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)
    if plan_only:
        return
    source_files = [path for path in source.rglob("*") if path.is_file() and "nero_timing" not in path.parts]
    original_hashes = {str(path.relative_to(source)): digest(path) for path in source_files}
    building.mkdir(parents=True)
    kept_tables, episode_stats, new_rows, video_jobs = [], {}, [], []
    offset = 0
    seen_videos = set()
    for episode, item in zip(episodes, plan, strict=True):
        ep = item["episode"]
        indices = np.flatnonzero(episode_indices == ep)[item["cut_frames"] :]
        kept = table.take(pa.array(indices))
        length = kept.num_rows
        for name, values in {
            "index": np.arange(offset, offset + length),
            "frame_index": np.arange(length),
            "timestamp": np.arange(length) / fps,
        }.items():
            position = kept.schema.get_field_index(name)
            kept = kept.set_column(
                position, kept.schema.field(position), pa.array(values, type=kept.schema.field(position).type)
            )
        kept_tables.append(kept)
        stats = {key: numeric_stats(np.asarray(kept.column(key).to_pylist())) for key in kept.column_names}
        episode_stats[ep] = stats
        row = {key: value for key, value in episode.items() if not key.startswith("stats/")}
        row.update(length=length, dataset_from_index=offset, dataset_to_index=offset + length)
        row.update(
            {
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        for key in camera_keys:
            relative = info["video_path"].format(
                video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            if relative in seen_videos or episode[f"videos/{key}/from_timestamp"] != 0:
                raise ValueError("Expected a separate video file per episode; refusing shared-video trimming")
            seen_videos.add(relative)
            video_jobs.append((ep, key, relative, item))
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = length / fps
        new_rows.append(row)
        offset += length
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(
                trim_video, source / rel, building / rel, item["cut_frames"], item["original_frames"], fps
            ): (ep, key)
            for ep, key, rel, item in video_jobs
        }
        for completed, future in enumerate(as_completed(pending), 1):
            ep, key = pending[future]
            episode_stats[ep][key] = future.result()
            print(f"video {completed}/{len(video_jobs)}: episode {ep} {key}", flush=True)
    trimmed = pa.concat_tables(kept_tables)
    (building / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(trimmed, building / "data/chunk-000/file-000.parquet")
    for row in new_rows:
        for key, stats in episode_stats[row["episode_index"]].items():
            row.update({f"stats/{key}/{name}": value.tolist() for name, value in stats.items()})
    (building / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(new_rows, schema=episode_table.schema),
        building / "meta/episodes/chunk-000/file-000.parquet",
    )
    shutil.copy2(source / "meta/tasks.parquet", building / "meta/tasks.parquet")
    new_info = copy.deepcopy(info)
    new_info["total_frames"] = trimmed.num_rows
    for key in camera_keys:
        new_info["features"][key]["info"].update(
            {"video.codec": "h264", "video.g": 2, "video.crf": 18, "video.preset": "veryfast"}
        )
    (building / "meta/info.json").write_text(json.dumps(new_info, indent=2) + "\n")
    combined = aggregate_stats(list(episode_stats.values()))
    # Exact vector quantiles are inexpensive for this dataset and avoid merging
    # approximate per-episode quantiles for policy normalization.
    combined.update(
        {key: numeric_stats(np.asarray(trimmed.column(key).to_pylist())) for key in trimmed.column_names}
    )
    (building / "meta/stats.json").write_text(json.dumps(combined, default=as_json, indent=2) + "\n")
    if any(digest(source / relative) != value for relative, value in original_hashes.items()):
        raise RuntimeError("Source changed during the operation; output remains staged")
    report.update(
        kept_frames=trimmed.num_rows,
        source_unchanged=True,
        videos_reencoded=len(video_jobs),
        numeric_stats="exact",
        image_stats="sampled retained source frames; length-weighted aggregation",
    )
    (building / "trim_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    building.rename(output)
    (audit / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Completed", output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent video decoders/encoders (1–16)")
    args = parser.parse_args()
    run(args.source, args.output, args.audit, plan_only=args.plan_only, workers=args.workers)


if __name__ == "__main__":
    main()
