"""相机发现和预览生命周期；录制会话使用单独打开的相机并共享采样帧。"""

import logging
from contextlib import suppress
from pathlib import Path

import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from . import camera_color, camera_preview, depth_discovery, resolution
from .camera_focus import configure_focus
from .camera_model import ROLES
from .runtime import Runtime


class RecordingCamera(OpenCVCamera):
    """缩放放在相机后台线程；录制循环继续只取缓存快照。"""

    output_size: tuple[int, int] | None = None

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = super()._postprocess_image(image)
        return resolution.resize(image, self.output_size) if self.output_size else image


def create_preview_camera(path: str | Path) -> OpenCVCamera:
    """两个平台共用设备预览参数，让 OpenCV 协商传输格式。"""
    return OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=Path(path), width=640, height=480, fps=30, color_mode=ColorMode.RGB,
        )
    )


def connect_rgb_camera(cam: OpenCVCamera, path: str | Path, focus_setting: dict | None = None) -> None:
    """设备预览、录制和推理使用同一套相机初始化。"""
    cam.connect()
    cam.metadata = {
        **getattr(cam, "metadata", {}),
        "focus": configure_focus(path, focus_setting),
        "white_balance": camera_color.configure_white_balance(path),
    }
    for result in (cam.metadata["focus"], cam.metadata["white_balance"]):
        if result["state"] == "warning":
            logging.getLogger(__name__).warning("%s：%s", path, result["message"])


def validate_selection(camera_map: dict[str, str], record_camera: str) -> None:
    if record_camera not in ("left", "right", "both", "head", "all"):
        raise ValueError("录制相机请选择左、右、左右、仅头部或全部已分配相机")
    if not isinstance(camera_map, dict) or any(
        role not in ("left", "right", "head", "unused") for role in camera_map.values()
    ):
        raise ValueError("相机角色只能为左、右、头部或未用")
    for role in ("left", "right", "head"):
        if list(camera_map.values()).count(role) > 1:
            raise ValueError(f"{role} 分配了多台相机，请为每个角色只保留一台")


def recording_sources(runtime: Runtime, detected: dict[str, dict]) -> dict[str, dict]:
    """按保存的角色解析每路设备与类型（rgb/rgbd）；缺失设备不忽略、不替换。"""
    camera_map = runtime.state["camera_map"]
    selection = runtime.state["record_camera"]
    validate_selection(camera_map, selection)
    roles = (
        [role for role in ROLES if role in camera_map.values()]
        if selection == "all"
        else (["left", "right"] if selection == "both" else [selection])
    )
    # 保留旧配置语义：左右选择附带已分配的头部；新增 head 可仅录头部。
    if selection in ("left", "right", "both") and "head" in camera_map.values():
        roles.append("head")
    if not roles:
        raise ValueError("请至少分配一台录制相机")
    sources = {}
    for role in roles:
        ids = [cam_id for cam_id, assigned in camera_map.items() if assigned == role]
        if not ids:
            raise ValueError(f"尚未分配 {role} 相机，请在对应预览下选择角色并保存")
        if ids[0] not in detected:
            raise ValueError(f"所选 {role} 相机未检测到：{ids[0]}，请检查连接和配置")
        sources[role] = {"id": ids[0], **detected[ids[0]]}
    return sources


def detect_cameras() -> dict[str, str]:
    cams = {}
    byid = Path("/dev/v4l/by-id")
    if not byid.exists():
        return cams
    for link in sorted(byid.glob("*video-index0")):
        if any(name in link.name.lower() for name in ("femto", "realsense", "azure_kinect")):
            continue  # 深度设备由对应 SDK 独占，避免 UVC 预览再次打开同一设备。
        cams[link.name.removesuffix("-video-index0")] = str(link)
    return cams


def detect_rgb_cameras() -> dict[str, dict]:
    return {
        cam_id: {"type": "rgb", "backend": "opencv", "path": path}
        for cam_id, path in detect_cameras().items()
    }


def detect_all_cameras() -> dict[str, dict]:
    """启动采集前重新核对设备；每个深度后端独立失败，不影响其他设备枚举。"""
    depth, results = depth_discovery.detect()
    for result in results.values():
        if result["state"] == "error":
            logging.getLogger(__name__).warning("%s：%s", result["label"], result["message"])
    return {**detect_rgb_cameras(), **depth}


def scan_inventory(runtime: Runtime, kind: str = "all") -> None:
    """更新指定类别；检测深度不会清空 RGB，反之亦然。"""
    if kind not in ("rgb", "depth", "all"):
        raise ValueError("检测类型必须为 rgb 或 depth")
    with runtime.camera_scan_lock:
        detected, results = {}, {}
        if kind in ("rgb", "all"):
            rgb = detect_rgb_cameras()
            detected.update(rgb)
            results["opencv"] = {
                "label": "RGB 相机",
                "state": "ok",
                "count": len(rgb),
                "message": f"检测到 {len(rgb)} 台" if rgb else "未发现设备",
            }
        if kind in ("depth", "all"):
            depth, depth_results = depth_discovery.detect()
            detected.update(depth)
            results.update(depth_results)
        replaced_types = {"rgb", "rgbd"} if kind == "all" else {"rgbd" if kind == "depth" else "rgb"}
        retained = {
            key: value
            for key, value in runtime.camera_inventory.items()
            if value["type"] not in replaced_types
        }
        runtime.camera_inventory = {**retained, **detected}
        runtime.camera_detection = {**runtime.camera_detection, **results}
        runtime.cameras_scanned = True


def inventory_payload(runtime: Runtime) -> dict:
    """页面编号只依赖实际设备 ID，不参与角色保存、配置或录制寻址。"""
    ordered = {
        cam_id: {**info, "preview_supported": camera_preview.supported(info)}
        for cam_id, info in sorted(runtime.camera_inventory.items())
    }
    return {
        "cameras": ordered,
        "camera_map": dict(runtime.state["camera_map"]),
        "detection": dict(runtime.camera_detection),
    }


def open_preview_cams(runtime: Runtime) -> None:
    for cam_id, path in detect_cameras().items():
        if cam_id in runtime.preview_cams:
            continue
        cam = None
        try:
            cam = create_preview_camera(path)
            connect_rgb_camera(cam, path, runtime.state["camera_focus_settings"].get(cam_id))
            runtime.camera_focus[cam_id] = cam.metadata["focus"]
            runtime.preview_cams[cam_id] = cam
        except Exception as e:
            if cam is not None:
                with suppress(Exception):
                    cam.disconnect()
            print(f"预览相机打开失败 {cam_id}: {e}")


def close_preview_cams(runtime: Runtime, kind: str | None = None, cam_id: str | None = None) -> None:
    """先释放预览才能转交录制；关闭失败保留句柄，禁止静默重复打开。"""
    errors = []
    with runtime.preview_lock:
        for device_id, cam in list(runtime.preview_cams.items()):
            device_kind = "depth" if device_id in runtime.preview_states else "rgb"
            if (kind and device_kind != kind) or (cam_id and device_id != cam_id):
                continue
            try:
                cam.disconnect()
            except Exception as exc:
                message = f"{device_id} 预览关闭失败：{exc}"
                errors.append(message)
                if device_kind == "depth":
                    runtime.preview_states[device_id] = {"phase": "error", "message": message}
                continue
            del runtime.preview_cams[device_id]
            if device_kind == "depth":
                runtime.preview_states[device_id] = {"phase": "closed", "message": "预览已释放"}
    if errors:
        raise RuntimeError("；".join(errors))
