"""深度设备检测器注册表；仅枚举设备，不启动视频流或修改设备配置。"""

from collections.abc import Callable
from dataclasses import dataclass

from . import azure_kinect, femto
from .realsense import list_devices as list_realsense


def list_femto() -> list[dict]:
    if femto.ob is None:
        raise ImportError("未安装 pyorbbecsdk")
    return femto.list_devices()


@dataclass(frozen=True)
class Detector:
    label: str
    prefix: str
    backend: str
    enumerate_devices: Callable[[], list[dict]]
    recording_supported: bool = True


DETECTORS = (
    Detector("奥比 Femto", "femto", "orbbec", list_femto),
    Detector("Azure Kinect", "k4a", "azure_kinect", azure_kinect.list_devices),
    Detector("RealSense", "realsense", "realsense", list_realsense),
)


def detect() -> tuple[dict[str, dict], dict[str, dict]]:
    cameras, results = {}, {}
    for detector in DETECTORS:
        try:
            entries = detector.enumerate_devices()
            found = {}
            for entry in entries:
                serial = entry["serial"]
                if not serial:
                    raise RuntimeError("设备未返回有效序列号")
                found[f"{detector.prefix}:{serial}"] = {
                    **entry,
                    "type": "rgbd",
                    "backend": detector.backend,
                    "recording_supported": detector.recording_supported,
                }
            cameras.update(found)
            state, message = "ok", f"检测到 {len(entries)} 台" if entries else "未发现设备"
        except ImportError as exc:
            state, message = "missing_sdk", str(exc)
            entries = []
        except Exception as exc:
            state, message = "error", f"检测失败：{exc}"
            entries = []
        results[detector.backend] = {
            "label": detector.label,
            "state": state,
            "count": len(entries),
            "message": message,
        }
    return cameras, results
