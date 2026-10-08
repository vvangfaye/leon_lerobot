"""只读本地 LeRobot v3 文件；不创建数据写入器、不连接设备、不下载数据。"""

import bisect
import json
import math
import re
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from flask import Blueprint, Response, jsonify, request, send_file

from lerobot.datasets.depth_utils import dequantize_depth

from . import depth_images, pairing, rgbd_timing
from .camera_model import is_depth, stream_label
from .runtime import Runtime


class ReviewError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def inside(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ReviewError("数据路径超出当前数据集目录")
    return path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    # json.loads 默认接受 NaN / Infinity；禁止把这些值送到浏览器 JSON 解析器。
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ReviewError(f"无效的 JSON 对象：{path.name}")
    return value


class LocalReview:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def dataset_root(self, repo_id: str) -> Path:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id) or any(
            part in (".", "..") for part in repo_id.split("/")
        ):
            raise ReviewError("请选择有效的本地数据集名（用户/名称）")
        return inside(self.root, repo_id)

    def info(self, repo_id: str) -> tuple[Path, dict]:
        root = self.dataset_root(repo_id)
        path = inside(root, "meta/info.json")
        if not path.is_file():
            raise ReviewError("找不到本地数据集", 404)
        info = read_json(path)
        if info.get("codebase_version") != "v3.0":
            raise ReviewError("当前回放支持 LeRobot v3.0 数据集")
        fps = info.get("fps")
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ReviewError("数据集 FPS 无效")
        features = info["features"]
        if not isinstance(features, dict) or any(not isinstance(value, dict) for value in features.values()):
            raise ReviewError("数据集字段定义无效")
        for key in ("total_episodes", "total_frames"):
            if type(info[key]) is not int or info[key] < 0:
                raise ReviewError(f"数据集 {key} 无效")
        mode = "dual" if features["action"]["shape"] == [16] else "single"
        names = pairing.joint_keys(mode)
        for key in ("action", "observation.state"):
            if features[key]["shape"] != [len(names)] or features[key]["names"] != names:
                raise ReviewError("数据集状态/动作的维度或顺序不符合 Nero 单对/双对格式")
        return root, {**info, "pair_mode": mode, "joint_names": names}

    def datasets(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*/*/meta/info.json")):
            repo = path.parents[1].relative_to(self.root).as_posix()
            try:
                _, info = self.info(repo)
                result.append(
                    {
                        "repo_id": repo,
                        "episodes": info["total_episodes"],
                        "frames": info["total_frames"],
                        "fps": info["fps"],
                        "pair_mode": info["pair_mode"],
                        "error": None,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result.append({"repo_id": repo, "error": str(exc)})
        return result

    def metadata(self, root: Path, info: dict) -> list[dict]:
        files = [
            inside(root, p.relative_to(root)) for p in sorted((root / "meta/episodes").glob("*/*.parquet"))
        ]
        if not files:
            if info["total_episodes"] == 0:
                return []
            raise ReviewError("集索引尚未完成，请先结束采集并等待数据收尾，再刷新", 409)
        rows = []
        for path in files:
            # 不加载高维 stats 列，也不触发 Hugging Face 缓存、联网或视频解码。
            columns = [name for name in pq.read_schema(path).names if not name.startswith("stats/")]
            rows.extend(pq.read_table(path, columns=columns).to_pylist())
        rows.sort(key=lambda row: row["episode_index"])
        if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
            raise ReviewError("磁盘集索引与总集数不一致，请确认采集已收尾；原文件未改动", 409)
        for row in rows:
            if type(row["length"]) is not int or row["length"] <= 0:
                raise ReviewError("本集帧数无效")
            if not isinstance(row["tasks"], list) or any(not isinstance(task, str) for task in row["tasks"]):
                raise ReviewError("本集任务描述无效")
        return rows

    def report(self, root: Path, row: dict, fps: float) -> tuple[dict, dict | None]:
        index, length = row["episode_index"], row["length"]
        relative = f"meta/nero_timing/episode_{index:06d}.json"
        quality = {"state": "unknown", "warnings": [], "summary": None, "report_path": relative}
        try:
            path = inside(root, relative)
            if not path.exists():
                quality["warnings"] = ["未找到节拍报告，不能据此确认采样质量"]
                return quality, None
            report = read_json(path)
            summary = report["summary"]
            if (
                report.get("schema_version") not in (1, 2, 3)
                or report.get("episode_index") != index
                or summary["frames"] != length
                or summary["target_fps"] != fps
            ):
                raise ValueError("报告集号、帧数或 FPS 与数据不一致")
            samples = report["samples"]
            if not isinstance(samples, list):
                raise ValueError("报告采样列表无效")
            for sample in samples:
                start, end = sample["started_seconds"], sample["completed_seconds"]
                if not all(type(v) in (int, float) for v in (start, end)) or not 0 <= start <= end:
                    raise ValueError("报告采样起止时间无效")
                frame = sample["frame_index"]
                if frame is not None and (type(frame) is not int or not 0 <= frame < length):
                    raise ValueError("报告帧索引无效")
                cameras = sample.get("cameras", {})
                if not isinstance(cameras, dict) or any(not isinstance(v, dict) for v in cameras.values()):
                    raise ValueError("报告相机信息无效")
                for details in cameras.values():
                    if any(
                        key in details and type(details[key]) not in (int, float)
                        for key in (
                            "invalid_fraction",
                            "above_range_fraction",
                            "rgb_timestamp_us",
                            "depth_timestamp_us",
                        )
                        if details.get(key) is not None or key in ("invalid_fraction", "above_range_fraction")
                    ):
                        raise ValueError("报告 RGBD 数值无效")
            valid = [s for s in samples if s["frame_index"] is not None]
            if [s["frame_index"] for s in valid] != list(range(length)):
                raise ValueError("报告中的有效帧索引不连续")
            starts = [s["started_seconds"] for s in samples]
            if any(not math.isfinite(v) or v < 0 for v in starts) or starts != sorted(starts):
                raise ValueError("报告采样时间无效")
            warnings = summary.get("warnings", [])
            if not isinstance(warnings, list) or any(not isinstance(v, str) for v in warnings):
                raise ValueError("报告警告列表无效")
            warnings = list(warnings)
            for key, value in summary.items():
                if (
                    key not in ("warnings", "cameras")
                    and value is not None
                    and type(value) not in (int, float)
                ):
                    raise ValueError(f"报告统计项 {key} 无效")
            camera_stats = summary.get("cameras", {})
            if not isinstance(camera_stats, dict) or any(
                not isinstance(stats, dict)
                or any(
                    type(stats.get(key)) not in (int, float)
                    for key in ("frames", "repeated_frames", "max_age_ms")
                )
                for stats in camera_stats.values()
            ):
                raise ValueError("报告相机统计无效")
            # 旧报告的 0 - 0 = 0 不能沿用为同步证据；仅修正内存中的展示，不重写文件。
            trackers: dict[str, rgbd_timing.TimestampTracker] = {}
            timestamp_stats: dict[str, dict] = {}
            for sample in samples:
                for name, details in sample.get("cameras", {}).items():
                    if not (is_depth(name) or "rgb_timestamp_us" in details):
                        continue
                    check = rgbd_timing.checked_sample(details)
                    if check is None:
                        check = trackers.setdefault(name, rgbd_timing.TimestampTracker()).assess(details)
                    details.update(check)
                    if sample["frame_index"] is not None:
                        rgbd_timing.add_stats(timestamp_stats.setdefault(name, {}), check)
                        rgbd_timing.add_receive_stats(timestamp_stats[name], details)
            for name in set(timestamp_stats) | {name for name in camera_stats if is_depth(name)}:
                if name in camera_stats:
                    corrected = timestamp_stats.get(
                        name,
                        {
                            "timestamp_checked_frames": 0,
                            "timestamp_unverified_frames": camera_stats[name]["frames"],
                            "max_rgb_depth_skew_ms": None,
                        },
                    )
                    camera_stats[name].update(corrected)
            warnings = [message for message in warnings if not message.startswith("头部 RGB/深度")]
            for name, stats in camera_stats.items():
                if is_depth(name):
                    warnings.extend(rgbd_timing.warnings(stats, fps, stream_label(name)))
            warnings = list(dict.fromkeys(warnings))
            pairs = report.get("pairs", [])
            if not isinstance(pairs, list) or any(not isinstance(pair, dict) for pair in pairs):
                raise ValueError("报告机械臂配对无效")
            if report.get("termination_error"):
                warnings.append("提前结束：" + report["termination_error"])
            if summary.get("failed_samples") and not warnings:
                warnings.append("存在采样失败，请查看异常时刻")
            control = report.get("control")
            if control is not None:
                if not isinstance(control, dict) or not isinstance(control.get("summary"), dict):
                    raise ValueError("报告控制节拍无效")
                control_warnings = control["summary"].get("warnings", [])
                if not isinstance(control_warnings, list) or any(
                    not isinstance(w, str) for w in control_warnings
                ):
                    raise ValueError("报告控制节拍提示无效")
                warnings.extend(control_warnings)
                quality["control_timing"] = control["summary"]
            quality.update(state="review" if warnings else "ok", warnings=warnings, summary=summary)
            quality["pairs"] = pairs
            quality["camera_devices"] = report.get("camera_devices", {})
            legacy_rgbd = report.get("femto") or report.get("head_camera")
            if legacy_rgbd and not quality["camera_devices"]:
                quality["camera_devices"] = {"legacy_rgbd": legacy_rgbd}
            return quality, report
        except (OSError, ValueError, KeyError, TypeError) as exc:
            quality.update(state="review", warnings=[f"节拍报告不可用：{exc}"])
            return quality, None

    def episodes(self, repo_id: str, offset: int, limit: int) -> dict:
        root, info = self.info(repo_id)
        rows = list(reversed(self.metadata(root, info)))
        episodes = []
        for row in rows[offset : offset + limit]:
            quality, _ = self.report(root, row, info["fps"])
            episodes.append(
                {
                    "index": row["episode_index"],
                    "frames": row["length"],
                    "tasks": row["tasks"],
                    "duration": row["length"] / info["fps"],
                    "quality": quality,
                }
            )
        return {
            "repo_id": repo_id,
            "pair_mode": info["pair_mode"],
            "fps": info["fps"],
            "total": len(rows),
            "offset": offset,
            "episodes": episodes,
        }

    def episode_metadata(self, repo_id: str, index: int) -> tuple[Path, dict, dict]:
        root, info = self.info(repo_id)
        rows = self.metadata(root, info)
        if index < 0 or index >= len(rows):
            raise ReviewError("找不到这一集", 404)
        return root, info, rows[index]

    def file_path(self, root: Path, template: str, **values: Any) -> Path:
        path = inside(root, template.format(**values))
        if not path.is_file():
            raise ReviewError(f"数据文件缺失：{path.name}", 404)
        return path

    def video(self, repo_id: str, index: int, camera: str) -> tuple[Path, float, float]:
        root, info, row = self.episode_metadata(repo_id, index)
        return self.video_location(root, info, row, camera)

    def video_location(self, root: Path, info: dict, row: dict, camera: str) -> tuple[Path, float, float]:
        key = f"observation.images.{camera}"
        if info["features"].get(key, {}).get("dtype") != "video":
            raise ReviewError("这一集没有指定的相机", 404)
        start, end = (row[f"videos/{key}/{field}"] for field in ("from_timestamp", "to_timestamp"))
        if (
            not all(type(v) in (float, int) and math.isfinite(v) for v in (start, end))
            or not 0 <= start < end
        ):
            raise ReviewError("视频片段时间范围无效")
        path = self.file_path(
            root,
            info["video_path"],
            video_key=key,
            chunk_index=row[f"videos/{key}/chunk_index"],
            file_index=row[f"videos/{key}/file_index"],
        )
        return path, start, end

    def episode(self, repo_id: str, index: int) -> dict:
        root, info, row = self.episode_metadata(repo_id, index)
        path = self.file_path(
            root, info["data_path"], chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        table = pq.read_table(
            path,
            columns=["frame_index", "timestamp", "action", "observation.state"],
            filters=[("episode_index", "=", index)],
        )
        data = table.sort_by("frame_index").to_pydict()
        length, dimensions = row["length"], len(info["joint_names"])
        if length <= 0 or data["frame_index"] != list(range(length)):
            raise ReviewError("本集帧索引缺失或重复，无法可靠对齐视频与曲线")
        for name in ("action", "observation.state"):
            values = np.asarray(data[name], dtype=float)
            if values.shape != (length, dimensions) or not np.isfinite(values).all():
                raise ReviewError(f"{name} 含无效数值或维度不一致")
        timestamps = data["timestamp"]
        if not np.allclose(timestamps, np.arange(length) / info["fps"], atol=1e-4):
            raise ReviewError("数据时间戳与保存帧率不一致，无法可靠对齐视频")
        quality, report = self.report(root, row, info["fps"])
        videos = []
        for key, feature in info["features"].items():
            if not key.startswith("observation.images.") or feature["dtype"] != "video":
                continue
            camera = key.removeprefix("observation.images.")
            video_info = feature.get("info")
            codec = video_info.get("video.codec") if isinstance(video_info, dict) else None
            entry = {
                "camera": camera,
                "codec": codec,
                "error": None,
                "is_depth": isinstance(video_info, dict) and bool(video_info.get("is_depth_map")),
            }
            try:
                _, start, end = self.video_location(root, info, row, camera)
                if abs(end - start - length / info["fps"]) > 1.1 / info["fps"]:
                    raise ReviewError("视频片段时长与本集帧数不一致")
                entry.update(start=start, end=end)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                entry["error"] = str(exc)
            videos.append(entry)
        return {
            "repo_id": repo_id,
            "index": index,
            "tasks": row["tasks"],
            "frames": length,
            "fps": info["fps"],
            "duration": length / info["fps"],
            "pair_mode": info["pair_mode"],
            "names": info["joint_names"],
            "timestamps": timestamps,
            "state": data["observation.state"],
            "action": data["action"],
            "videos": videos,
            "quality": quality,
            "events": self.events(report, timestamps, info["fps"]),
        }

    def depth_frame(self, repo_id: str, index: int, camera: str, frame_index: int) -> np.ndarray:
        root, info, row = self.episode_metadata(repo_id, index)
        if not 0 <= frame_index < row["length"]:
            raise ReviewError("深度帧索引超出本集范围")
        feature = info["features"].get(f"observation.images.{camera}", {})
        settings = feature.get("info", {})
        if not settings.get("is_depth_map"):
            raise ReviewError("指定的视频不是深度数据")
        path, start, _ = self.video_location(root, info, row, camera)
        target = start + frame_index / info["fps"]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                stamp = float(frame.pts * frame.time_base)
                if stamp < target - 0.5 / info["fps"]:
                    continue
                if abs(stamp - target) > 0.51 / info["fps"]:
                    break
                depth = dequantize_depth(
                    frame,
                    depth_min=settings["video.depth_min"],
                    depth_max=settings["video.depth_max"],
                    shift=settings["video.shift"],
                    use_log=settings["video.use_log"],
                    output_unit="m",
                    output_tensor=False,
                    output_channel_last=True,
                )
                return np.asarray(depth)
        raise ReviewError("未能解码指定深度帧", 422)

    @staticmethod
    def events(report: dict | None, timestamps: list[float], fps: float) -> list[dict]:
        if report is None:
            return []
        samples = report["samples"]
        valid = [s for s in samples if s["frame_index"] is not None]
        starts = [s["started_seconds"] for s in valid]
        events = []
        previous = None
        timestamp_trackers: dict[str, rgbd_timing.TimestampTracker] = {}
        previous_timestamp_status: dict[str, str] = {}
        for sample in samples:
            messages = []
            if sample.get("error"):
                messages.append(str(sample["error"]))
            for camera, details in sample.get("cameras", {}).items():
                if details.get("repeated"):
                    messages.append(f"{camera} 相机重复取用缓存帧")
                if is_depth(camera):
                    label = stream_label(camera)
                    if details.get("invalid_fraction", 0) >= 1:
                        messages.append(f"{label}整帧无效")
                    if details.get("above_range_fraction", 0) > 0:
                        messages.append(f"{label}含超过 10 米的值，编码饱和")
                    check = rgbd_timing.checked_sample(details) or timestamp_trackers.setdefault(
                        camera, rgbd_timing.TimestampTracker()
                    ).assess(details)
                    status = check["timestamp_status"]
                    if status != "valid" and status != previous_timestamp_status.get(camera):
                        messages.append(f"{label} RGB/深度：" + rgbd_timing.STATUS_TEXT[status])
                    elif status == "valid" and check["rgb_depth_skew_ms"] > 1000 / fps:
                        messages.append(f"{label} RGB/深度有效时间戳差超过一个采样周期")
                    previous_timestamp_status[camera] = status
                    receive_skew = rgbd_timing.receive_skew_ms(details)
                    if receive_skew is not None and receive_skew > rgbd_timing.receive_skew_limit_ms(fps):
                        messages.append(f"{label} RGB/深度主机接收时差 {receive_skew:.1f} ms，疑似旧帧积压")
            if sample["completed_seconds"] - sample["started_seconds"] > 1.0 / fps + 1e-9:
                messages.append("本次处理超过一个采样周期")
            if previous is not None and sample["started_seconds"] - previous > 1.5 / fps:
                messages.append("采样间隔超过 1.5 个周期")
            previous = sample["started_seconds"]
            if messages:
                frame = sample["frame_index"]
                if frame is None:
                    frame = min(bisect.bisect_left(starts, previous), len(timestamps) - 1)
                    messages.append("未写入数据，定位到相邻有效帧")
                events.append(
                    {
                        "frame": frame,
                        "time": timestamps[frame],
                        "host_time": previous,
                        "message": "；".join(messages),
                    }
                )
        return events


def create_blueprint(runtime: Runtime) -> Blueprint:
    """只读路由独立于设备状态轮询，不持有采集锁执行磁盘读取。"""
    blueprint = Blueprint("review", __name__)
    reader = LocalReview(runtime.dataset_root)
    active_message = "该数据集采集会话尚未收尾，请先点击结束采集，再刷新回放；无需失能机械臂。"

    def active_repo() -> str | None:
        with runtime.lifecycle_lock:
            if runtime.dataset is not None or runtime.state["connected"] or runtime.state["ending"]:
                return runtime.state["repo_id"]
        return None

    def selection() -> str:
        repo = request.args.get("repo_id", "")
        reader.dataset_root(repo)
        if repo == active_repo():
            raise ReviewError(active_message, 409)
        return repo

    def integer(name: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
        try:
            value = int(request.args.get(name, str(default)))
        except ValueError as exc:
            raise ReviewError(f"{name} 必须是整数") from exc
        if value < minimum or (maximum is not None and value > maximum):
            raise ReviewError(f"{name} 超出允许范围")
        return value

    @blueprint.errorhandler(ReviewError)
    def review_error(exc: ReviewError) -> tuple[Response, int]:
        return jsonify(ok=False, msg=str(exc)), exc.status

    # 未收尾的 Parquet、损坏的本地元数据均给出可读错误，不改变任何原文件。
    def disk_error(exc: Exception) -> tuple[Response, int]:
        return jsonify(ok=False, msg=f"本地数据读取失败：{exc}。请确认采集已结束、文件完整，再刷新。"), 422

    for exception in (OSError, ValueError, KeyError, TypeError, IndexError):
        blueprint.register_error_handler(exception, disk_error)

    @blueprint.after_request
    def no_cache(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.get("/review")
    def page() -> str:
        return (Path(__file__).parent / "review.html").read_text(encoding="utf-8")

    @blueprint.get("/api/review/datasets")
    def datasets() -> Response:
        active = active_repo()
        entries = reader.datasets()
        for entry in entries:
            entry["blocked"] = active_message if entry["repo_id"] == active else None
        return jsonify(datasets=entries, selected_repo=runtime.state["repo_id"])

    @blueprint.get("/api/review/episodes")
    def episodes() -> Response:
        return jsonify(reader.episodes(selection(), integer("offset", 0), integer("limit", 20, 1, 100)))

    @blueprint.get("/api/review/episode")
    def episode() -> Response:
        return jsonify(reader.episode(selection(), integer("episode_index", -1)))

    @blueprint.get("/api/review/video")
    def video() -> Response:
        path, _, _ = reader.video(selection(), integer("episode_index", -1), request.args.get("camera", ""))
        return send_file(path, mimetype="video/mp4", conditional=True)

    @blueprint.get("/api/review/depth")
    def depth() -> Response:
        frame = reader.depth_frame(
            selection(),
            integer("episode_index", -1),
            request.args.get("camera", "depth"),
            integer("frame_index", -1),
        )
        color = depth_images.depth_preview(frame)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ReviewError("深度预览编码失败", 503)
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    return blueprint
