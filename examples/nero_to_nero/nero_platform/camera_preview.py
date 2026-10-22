"""共用预览编码和 RGBD 预览；一个设备只持有一条管线，与录制互斥。"""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from . import azure_kinect, depth_images, femto, realsense, resolution
from .runtime import Runtime


def encode_jpeg(frame: np.ndarray) -> bytes:
    """RGB 缓存帧只在预览编码时转为 BGR，不修改采集/模型输入。"""
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("预览编码失败")
    return encoded.tobytes()


def _orbbec(info: dict) -> femto.FemtoCamera:
    # 取景使用独立的低分辨率输出，不修改用户保存的录制分辨率。
    return femto.FemtoCamera(serial=info["serial"], output_size=(640, 360))


# 检测支持和预览支持分开注册。
FACTORIES: dict[str, Callable[[dict], Any]] = {
    "orbbec": _orbbec,
    "azure_kinect": lambda info: azure_kinect.AzureKinectCamera(
        serial=info["serial"], output_size=(640, 360)
    ),
    "realsense": lambda info: realsense.preview_camera(info["serial"]),
}


def supported(info: dict) -> bool:
    return info.get("type") == "rgb" or info.get("backend") in FACTORIES


def open_depth(runtime: Runtime, cam_id: str, info: dict) -> None:
    """调用者先占用 device_busy；预热期间不阻塞状态接口。"""
    with runtime.preview_lock:
        if runtime.state["shutting_down"]:
            raise RuntimeError("程序正在退出")
        # 重试先清理上次打开/关闭失败的句柄，不能再创建第二条管线。
        previous = runtime.preview_cams.get(cam_id)
        if previous is not None:
            previous.disconnect()
            del runtime.preview_cams[cam_id]
        cam = FACTORIES[info["backend"]](info)
        runtime.preview_cams[cam_id] = cam
        try:
            cam.connect()
            cam.read_snapshot()  # 完整 RGBD 帧就绪后才报告预览成功。
        except Exception as exc:
            try:
                cam.disconnect()
            except Exception as cleanup:
                raise RuntimeError(f"{exc}；预览释放失败：{cleanup}，请检查设备后重试") from exc
            del runtime.preview_cams[cam_id]
            raise
        runtime.preview_states[cam_id] = {"phase": "ready", "message": "RGB / 深度并排预览"}


def paired_frame(cam: Any) -> np.ndarray:
    """一次读取同一个 RGBD 快照，左右拼接；伪彩色只供预览。"""
    sample = cam.read_snapshot()
    depth = depth_images.depth_preview(sample["depth"])
    rgb = resolution.resize(sample["rgb"], (depth.shape[1], depth.shape[0]))
    return np.concatenate((rgb, depth), axis=1)
