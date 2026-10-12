"""配置持久化，与原版 collect_config.json 隔离。"""

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from . import camera_focus, cameras, resolution
from .runtime import ARM_LABELS, CONFIG_FIELDS, DEFAULT_REPO_ID, Runtime, target_control_hz


def load_saved(runtime: Runtime, legacy_path: Path | None = None) -> None:
    source = runtime.config_path
    importing_legacy = not source.exists() and legacy_path is not None and legacy_path.exists()
    if importing_legacy:
        source = legacy_path
    if not source.exists():
        return
    try:
        cfg = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("配置顶层必须是对象")
        cfg, notice = migrate_camera_config(cfg)
        if "camera_focus_settings" not in cfg:
            # 仅迁移已在此设备上验证过的旧值；新型号不继承统一的 128。
            cfg["camera_focus_settings"] = {
                key: {"mode": "manual", "value": 128}
                for key in cfg.get("camera_map", {})
                if "1080P_USB_Camera_46435000_" in key
            }
        apply_settings(runtime, {**cfg, "labels": cfg.get("arms", {})})
        runtime.state["camera_config_notice"] = notice
        runtime.state["config_error"] = None
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        runtime.state["config_error"] = (
            f"配置读取失败：{source}：{exc}。原文件保留，设备操作已锁定；请修复配置后重启程序。"
        )
        return
    if importing_legacy:
        # 继承设备映射，但使用独立验收数据集，避免混入原版采集结果。
        runtime.state["repo_id"] = DEFAULT_REPO_ID


def migrate_camera_config(cfg: dict) -> tuple[dict, str]:
    """只迁移明确的设备身份；旧自动选择不能静默变成不录头部。"""
    mode = cfg.get("head_camera_mode", "off")
    if mode == "off":
        return cfg, ""
    mapping = dict(cfg.get("camera_map", {}))
    if "head" in mapping.values():
        return cfg, ""  # 已有统一角色映射优先。
    serial = cfg.get("head_camera_serial", "")
    device_id = f"femto:{serial}" if serial else "legacy-rgbd-selection"
    if mapping.get(device_id, "unused") != "unused":
        raise ValueError("旧头部设备已分配其他角色，请先统一 camera_map 与旧头部配置")
    mapping[device_id] = "head"
    notice = (
        "旧头部相机配置已转换。RGBD 现在保存该角色的 RGB 与深度；请核对相机分配并保存配置。"
        if serial
        else "旧头部配置未指定序列号：请将 legacy-rgbd-selection 设为未用，重新分配真实头部设备并保存。"
    )
    return {**cfg, "camera_map": mapping}, notice


def validate_settings(runtime: Runtime, data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("配置必须是对象")
    merged = {key: data.get(key, runtime.state[key]) for key in (*CONFIG_FIELDS, "camera_map")}
    fps, duration = merged["fps"], merged["episode_sec"]
    if type(fps) is not int or fps <= 0:
        raise ValueError("FPS 必须为正整数")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("每集时长必须为有限正数")
    repo = merged["repo_id"]
    if (
        not isinstance(repo, str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
        or any(part in (".", "..") for part in repo.split("/"))
    ):
        raise ValueError("数据集名必须为 用户/名称，不能包含路径跳转")
    if not isinstance(merged["task"], str) or not merged["task"].strip():
        raise ValueError("任务描述不能为空")
    if merged["leader_label"] not in ("主左", "主右") or merged["follower_label"] not in ("从左", "从右"):
        raise ValueError("请选择有效的主臂和从臂角色")
    if merged["pair_mode"] not in ("single", "dual"):
        raise ValueError("请选择单对或左右双对采集模式")
    if merged["follow_motion_mode"] not in ("js", "j"):
        raise ValueError("遥操作模式请选择分离控制或底层平滑")
    force = merged["gripper_force"]
    if type(force) not in (int, float) or not math.isfinite(force) or not 0.0 <= force <= 32.0:
        raise ValueError("夹持力必须在 0 到 32 N 之间")
    target_control_hz(merged)
    resolution.validate(merged["camera_resolution"])
    camera_focus.validate_settings(merged["camera_focus_settings"])
    labels = data.get("labels", {ch: info["label"] for ch, info in runtime.channels.items()})
    if not isinstance(labels, dict) or any(
        label not in ARM_LABELS for ch, label in labels.items() if ch in runtime.channels
    ):
        raise ValueError("机械臂通道或角色无效")
    merged["labels"] = labels
    cameras.validate_selection(
        merged["camera_map"],
        merged["record_camera"],
    )
    if any(not isinstance(key, str) or not key for key in merged["camera_map"]):
        raise ValueError("相机设备标识无效")
    return merged


def apply_settings(runtime: Runtime, data: dict[str, Any]) -> None:
    data = validate_settings(runtime, data)  # 完整校验后才修改内存，失败时不留下半份配置。
    if data.get("repo_id", runtime.state["repo_id"]) != runtime.state["repo_id"]:
        runtime.state.update(episode_count=0, last_episode=None)
        if not runtime.state["writer_error"]:
            runtime.state.update(
                record_phase="idle",
                record_message="等待开始录制",
                record_elapsed=0.0,
                record_frames=0,
                record_frame_errors=0,
                record_frame_error=None,
            )
    for ch, label in data.get("labels", {}).items():
        if ch in runtime.channels and label in ARM_LABELS:
            runtime.channels[ch]["label"] = label
    if "camera_map" in data:
        runtime.state["camera_map"] = data["camera_map"]
    for key in CONFIG_FIELDS:
        if key in data:
            runtime.state[key] = data[key]


def settings(runtime: Runtime, detected: dict[str, dict] | None = None) -> dict[str, Any]:
    detected = detected or {}
    return {
        "arms": {ch: runtime.channels[ch]["label"] for ch in runtime.channels},
        "camera_map": runtime.state["camera_map"],
        "cameras": {
            role: detected[cam_id]
            for cam_id, role in runtime.state["camera_map"].items()
            if role in ("left", "right", "head") and cam_id in detected
        },
        **{key: runtime.state[key] for key in CONFIG_FIELDS},
    }


def save_config(
    runtime: Runtime, detected: dict[str, dict], data: dict[str, Any] | None = None
) -> dict[str, Any]:
    if runtime.state["config_error"]:
        raise ValueError(runtime.state["config_error"])
    validated = validate_settings(runtime, data or {})
    cfg = {
        **{key: validated[key] for key in CONFIG_FIELDS},
        "arms": {**{ch: info["label"] for ch, info in runtime.channels.items()}, **validated["labels"]},
        "camera_map": validated["camera_map"],
        "cameras": {
            role: detected[cam_id]
            for cam_id, role in validated["camera_map"].items()
            if role in ("left", "right", "head") and cam_id in detected
        },
    }
    runtime.config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=runtime.config_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(cfg, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, runtime.config_path)
        apply_settings(runtime, validated)
        runtime.state["camera_config_notice"] = ""
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return cfg
