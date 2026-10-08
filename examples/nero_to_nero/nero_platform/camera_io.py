"""相机后端适配：驱动负责缓存帧，录制器只消费按角色命名的快照。

backend 表示驱动，type 表示输出能力；增加驱动时在 BACKENDS 注册适配器，
无需在录制循环、预览接口中增加厂商分支。设备句柄由 Runtime.collection_cams 唯一持有。
"""

from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig

from . import azure_kinect, camera_focus, femto, realsense, resolution, sampling
from .camera_model import stream_label, stream_names
from .cameras import RecordingCamera


class OpenCVBackend:
    kind = "rgb"

    @staticmethod
    def profiles(info: dict) -> dict:
        return resolution.v4l2_profiles(info["path"])

    @classmethod
    def create(cls, info: dict, setting: dict, fps: int) -> Any:
        try:
            profiles = cls.profiles(info)["profiles"]
        except (OSError, RuntimeError):
            profiles = []  # connect 仍会严格验证实际尺寸和 FPS。
        cam = RecordingCamera(
            OpenCVCameraConfig(
                index_or_path=Path(info["path"]),
                width=setting["width"],
                height=setting["height"],
                fps=fps,
                color_mode=ColorMode.RGB,
                fourcc=resolution.select_fourcc(profiles, setting, fps) if profiles else None,
            )
        )
        cam.output_size = resolution.output_size(setting)
        return cam

    @staticmethod
    def connect(cam: Any, info: dict, focus_setting: dict | None = None) -> None:
        cam.connect()
        cam.metadata = {
            **getattr(cam, "metadata", {}),
            "focus": camera_focus.configure_focus(info["path"], focus_setting),
        }

    @staticmethod
    def inspect_focus(info: dict, stage: str, setting: dict | None = None) -> dict:
        return camera_focus.inspect_focus(info["path"], stage=stage, setting=setting)

    @staticmethod
    def snapshot(cam: Any) -> tuple[dict[str, np.ndarray], dict]:
        image, stats = sampling.camera_sample(cam)
        return {"rgb": image}, stats

    @staticmethod
    def preview(cam: Any, depth: bool) -> np.ndarray:
        return cam.read_latest()


class RGBDBackend:
    kind = "rgbd"

    @staticmethod
    def connect(cam: Any, info: dict, focus_setting: dict | None = None) -> None:
        cam.connect()

    @staticmethod
    def inspect_focus(info: dict, stage: str, setting: dict | None = None) -> None:
        return None  # RGBD 不经 V4L2 写入对焦控制。

    @staticmethod
    def snapshot(cam: Any) -> tuple[dict[str, np.ndarray], dict]:
        sample = cam.read_snapshot()  # 每周期一次：RGB 与深度必须来自同一完整帧组。
        return (
            {key: sample[key] for key in ("rgb", "depth")},
            {key: value for key, value in sample.items() if key not in ("rgb", "depth")},
        )

    @staticmethod
    def preview(cam: Any, depth: bool) -> np.ndarray:
        return cam.read_snapshot()["depth" if depth else "rgb"]


class OrbbecBackend(RGBDBackend):
    @staticmethod
    def profiles(info: dict) -> dict:
        return femto.list_profiles(info["serial"])

    @staticmethod
    def create(info: dict, setting: dict, fps: int) -> Any:
        return femto.FemtoCamera(
            fps=fps,
            serial=info["serial"],
            color_size=(setting["width"], setting["height"]),
            output_size=resolution.output_size(setting),
        )


class AzureKinectBackend(RGBDBackend):
    @staticmethod
    def profiles(info: dict) -> dict:
        return azure_kinect.list_profiles(info["serial"])

    @staticmethod
    def create(info: dict, setting: dict, fps: int) -> Any:
        return azure_kinect.AzureKinectCamera(
            fps=fps,
            serial=info["serial"],
            color_size=(setting["width"], setting["height"]),
            output_size=resolution.output_size(setting),
        )


class RealSenseBackend(RGBDBackend):
    @staticmethod
    def profiles(info: dict) -> dict:
        return realsense.list_profiles(info["serial"])

    @staticmethod
    def create(info: dict, setting: dict, fps: int) -> Any:
        return realsense.RealSenseCamera(
            fps=fps,
            serial=info["serial"],
            color_size=(setting["width"], setting["height"]),
            output_size=resolution.output_size(setting),
        )


BACKENDS = {
    "opencv": OpenCVBackend,
    "orbbec": OrbbecBackend,
    "azure_kinect": AzureKinectBackend,
    "realsense": RealSenseBackend,
}


def backend(info: dict) -> type[OpenCVBackend] | type[RGBDBackend]:
    if info.get("recording_supported") is False:
        raise ValueError("此相机已支持检测，但 RGBD 录制驱动尚未接入，请暂设为未用")
    kind = info.get("type", "rgb")
    # 兼容重构初版的描述符；新发现的设备始终提供明确的 backend。
    name = info.get("backend", {"rgb": "opencv", "rgbd": "orbbec"}.get(kind))
    adapter = BACKENDS.get(name)
    if adapter is None or adapter.kind != kind:
        raise ValueError(f"不支持的相机后端/类型：{name} / {kind}")
    return adapter


def sample(role: str, info: dict, cam: Any) -> tuple[dict, dict]:
    try:
        images, stats = backend(info).snapshot(cam)
    except Exception as exc:
        identity = info.get("id") or info.get("serial") or info.get("path", "")
        raise RuntimeError(f"{stream_label(role)} 相机 {identity}：{exc}") from exc
    names = stream_names(role, info.get("type", "rgb"))
    frames, details = {}, {}
    for name, kind in zip(names, ("rgb", "depth"), strict=False):
        frames[name] = images[kind]
        # 每路单独的统计对象，避免质检写入 repeated 等字段时互相污染。
        details[name] = {**stats, "stream_kind": kind, "role": role}
    return frames, details
