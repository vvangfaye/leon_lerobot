"""只编辑已收尾的本地 LeRobot v3 数据集，不访问机械臂或相机。

先校验独占视频与确认版本，再生成受影响分片和完整统计，最后交给事务层。
保留全部未删除集的文件位置、集号、帧号及视频字节，不合并分片。
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import write_stats

from . import dataset_transaction


@dataclass
class DeletePlan:
    root: Path
    info: dict[str, Any]
    revision: str
    rows: list[dict[str, Any]]
    episode_files: dict[Path, pa.Table]
    videos: list[Path]

    @property
    def last(self) -> dict[str, Any]:
        return self.rows[-1]

    def confirmation(self) -> dict[str, Any]:
        return {
            "episode_index": self.last["episode_index"],
            "frames": self.last["length"],
            "revision": self.revision,
        }


def _path(root: Path, relative: str) -> Path:
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("元数据包含数据集外部路径，不能删除")
    if not path.is_file():
        raise ValueError(f"数据文件缺失，不能删除：{relative}")
    return path


def _revision(root: Path) -> str:
    digest = hashlib.sha256((root / "meta/info.json").read_bytes())
    files = [
        root / "meta/info.json",
        *sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")),
        *sorted((root / "data").glob("chunk-*/file-*.parquet")),
    ]
    for path in files:
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def plan_delete(root: Path) -> DeletePlan:
    """只读预检；共用视频或不完整数据在任何正式文件修改前拒绝。"""
    root = root.resolve()
    if dataset_transaction.pending(root):
        raise ValueError("上次删除尚需恢复，请先重新打开删除确认")
    info = json.loads(_path(root, "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("仅支持 LeRobot v3.0 数据集删除")
    total = info.get("total_episodes")
    if type(total) is not int or total <= 0:
        raise ValueError("数据集没有已保存的集")
    revision = _revision(root)
    files = {
        _path(root, str(path.relative_to(root))): pq.read_table(path)
        for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    }
    rows = sorted(
        (row for table in files.values() for row in table.to_pylist()), key=lambda r: r["episode_index"]
    )
    if [row["episode_index"] for row in rows] != list(range(total)):
        raise ValueError("集索引与总集数不一致，不能删除")
    offset = 0
    for row in rows:
        length = row["length"]
        if type(length) is not int or length <= 0 or row["dataset_from_index"] != offset:
            raise ValueError("集长度或帧范围不连续，不能删除")
        offset += length
        if row["dataset_to_index"] != offset:
            raise ValueError("集帧范围不一致，不能删除")
        _path(
            root,
            info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]),
        )
    if offset != info["total_frames"]:
        raise ValueError("总帧数与集元数据不一致，不能删除")
    last = rows[-1]
    videos = []
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        fields = (f"videos/{key}/chunk_index", f"videos/{key}/file_index")
        reference = tuple(last[field] for field in fields)
        if any(tuple(row[field] for field in fields) == reference for row in rows[:-1]):
            raise ValueError(f"本集的 {key} 与其他集共用视频文件，不支持直接删除；数据未改动")
        videos.append(
            _path(
                root,
                info["video_path"].format(video_key=key, chunk_index=reference[0], file_index=reference[1]),
            )
        )
    if _revision(root) != revision:
        raise ValueError("数据集正在变化，请结束采集后重新确认")
    return DeletePlan(root, info, revision, rows, files, videos)


def _remaining_stats(plan: DeletePlan) -> dict:
    """沿用 LeRobot 的按集聚合，覆盖数值、RGB、深度和索引，绝不混入旧全局统计。"""
    expected = {
        key
        for key, feature in plan.info["features"].items()
        if feature["dtype"] not in ("string", "language")
    }
    all_stats = []
    for row in plan.rows[:-1]:
        stats: dict[str, dict[str, np.ndarray]] = {}
        for key, value in row.items():
            if key.startswith("stats/"):
                feature, stat = key.removeprefix("stats/").rsplit("/", 1)
                if feature in expected and value is not None:
                    stats.setdefault(feature, {})[stat] = np.atleast_1d(np.asarray(value))
        if set(stats) != expected or any(
            not {"min", "max", "mean", "std", "count"} <= set(s) for s in stats.values()
        ):
            raise ValueError(f"第 {row['episode_index'] + 1} 集统计不完整，不能安全重算；数据未改动")
        all_stats.append(stats)
    return aggregate_stats(all_stats) if all_stats else {}


def _stage(plan: DeletePlan, stage: Path) -> dict[str, Path | None]:
    """全部新内容先写入暂存区；此阶段失败不会修改原数据。"""
    replacements: dict[str, Path | None] = {}

    def parquet(path: Path, table: pa.Table) -> None:
        relative = str(path.relative_to(plan.root))
        if table.num_rows == 0 and plan.last["episode_index"] > 0:
            # 有剩余集时删除空尾分片，HF 的多分片读取器不接受零行批次。
            replacements[relative] = None
            return
        output = stage / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output)
        replacements[relative] = output

    last = plan.last
    index = last["episode_index"]
    data_path = _path(
        plan.root,
        plan.info["data_path"].format(
            chunk_index=last["data/chunk_index"], file_index=last["data/file_index"]
        ),
    )
    table = pq.read_table(data_path)
    removed = table.filter(pc.equal(table["episode_index"], index))
    if removed.num_rows != last["length"] or sorted(removed["index"].to_pylist()) != list(
        range(last["dataset_from_index"], last["dataset_to_index"])
    ):
        raise ValueError("最后一集的帧数或帧索引不一致，数据未改动")
    # 检查其他分片没有残留的同集数据；只读 episode_index，不加载图像或动作。
    for other in (plan.root / "data").glob("chunk-*/file-*.parquet"):
        if (
            other != data_path
            and pq.read_table(
                other, columns=["episode_index"], filters=[("episode_index", "=", index)]
            ).num_rows
        ):
            raise ValueError("最后一集分散在多个数据分片中，不能直接删除")
    parquet(data_path, table.filter(pc.not_equal(table["episode_index"], index)))
    for path, episodes in plan.episode_files.items():
        remaining = episodes.filter(pc.not_equal(episodes["episode_index"], index))
        if remaining.num_rows != episodes.num_rows:
            parquet(path, remaining)
    # 整个数据集删空时保留 schema；此时原生 resume 不加载旧集索引。
    info = {**plan.info, "total_episodes": index, "total_frames": last["dataset_from_index"]}
    info["splits"] = {}
    for name, bounds in plan.info.get("splits", {"train": f"0:{index + 1}"}).items():
        start, end = map(int, bounds.split(":"))
        info["splits"][name] = f"{min(start, index)}:{min(end, index)}"
    (stage / "meta").mkdir(exist_ok=True)
    (stage / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    replacements["meta/info.json"] = stage / "meta/info.json"
    stats = _remaining_stats(plan)
    if stats:
        write_stats(stats, stage)
        replacements["meta/stats.json"] = stage / "meta/stats.json"
    else:
        replacements["meta/stats.json"] = None  # 无历史统计，新录第一集直接建立统计。
    for video in plan.videos:
        replacements[str(video.relative_to(plan.root))] = None
    report = f"meta/nero_timing/episode_{index:06d}.json"
    if (plan.root / report).exists():
        _path(plan.root, report)
        replacements[report] = None
    return replacements


def delete_last(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    plan = plan_delete(root)
    if expected != plan.confirmation():
        raise ValueError("数据集或最后一集已变化，请重新打开确认框；未删除任何数据")
    # 暂存区清理失败不能把已提交的删除误报为失败；原件恢复由事务日志负责。
    with tempfile.TemporaryDirectory(
        prefix=".nero-delete-stage-", dir=plan.root, ignore_cleanup_errors=True
    ) as directory:
        replacements = _stage(plan, Path(directory))
        if _revision(plan.root) != plan.revision:
            raise ValueError("数据集在准备删除期间发生变化，请重新确认；未删除任何数据")
        dataset_transaction.replace_files(plan.root, replacements)
    return {"episode_count": plan.last["episode_index"], "frames": plan.last["length"]}
