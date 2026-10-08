"""Azure Kinect DK：单独后台接收 RGBD，标定 D2C，米单位深度与原子快照。"""

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import resolution, rgbd_timing
from .depth_images import DEPTH_MAX_METERS

try:
    import pyk4a as k4a
except (ImportError, OSError) as exc:
    k4a = None
    _SDK_ERROR = str(exc)
else:
    _SDK_ERROR = ""

# Sensor SDK 1.4 的固定模式表；不是将 UVC 深度接口当普通相机枚举。
# NFOV_UNBINNED 640×576 支持 5/15/30 FPS；3072p 彩色最高 15 FPS。
COLOR_MODES = {
    (1280, 720): "RES_720P",
    (1920, 1080): "RES_1080P",
    (2560, 1440): "RES_1440P",
    (2048, 1536): "RES_1536P",
    (3840, 2160): "RES_2160P",
    (4096, 3072): "RES_3072P",
}
USB_ROOT = Path("/sys/bus/usb/devices")


def require_sdk() -> Any:
    if k4a is None:
        raise ImportError("Azure Kinect 需要 libk4a 1.4 与 pyk4a；SDK 加载失败：" + _SDK_ERROR)
    return k4a


def list_devices() -> list[dict]:
    """Linux USB 序列号枚举，不占用已打开的 SDK 预览句柄，不启动流。"""
    entries = {}
    for path in sorted(USB_ROOT.glob("*")):
        try:
            if (path / "idVendor").read_text().strip() != "045e":
                continue
            if (path / "idProduct").read_text().strip() != "097c":
                continue
            serial = (path / "serial").read_text().strip()
            if not serial:
                raise RuntimeError("已发现 Azure Kinect，但 USB 未提供序列号")
            entries[serial] = {
                "name": "Azure Kinect DK",
                "serial": serial,
                "usb_speed_mbps": float((path / "speed").read_text().strip()),
            }
        except (FileNotFoundError, NotADirectoryError):
            continue  # 枚举期间拔出或遇到 USB interface 节点。
    if entries:
        require_sdk()
    return list(entries.values())


def list_profiles(serial: str = "") -> dict:
    require_sdk()
    matches = [item for item in list_devices() if not serial or item["serial"] == serial]
    if len(matches) != 1:
        raise RuntimeError("未找到唯一的 Azure Kinect，请重新检测并指定序列号")
    return {
        "profiles": [
            {
                "width": width,
                "height": height,
                "format": "BGRA32",
                "fps": [5, 15] if height == 3072 else [5, 15, 30],
                "reason": "",
            }
            for width, height in COLOR_MODES
        ],
        "ranges": [],
        "serial": matches[0]["serial"],
        "source": "Azure Kinect Sensor SDK 模式表；深度 NFOV_UNBINNED 640×576，标定 D2C",
    }


def _open_selected(sdk: Any, serial: str, config: Any) -> Any:
    failures = []
    for index in range(sdk.connected_device_count()):
        # pyk4a 1.5 的 thread_safe=True 会持有 GIL 等帧。False 释放 GIL；
        # 本驱动只让一个后台线程访问句柄，关闭前必须先等该线程退出。
        device = sdk.PyK4A(config, device_id=index, thread_safe=False)
        selected = False
        try:
            device.open()
            selected = device.serial == serial
            if selected:
                return device
        except Exception as exc:
            failures.append(str(exc) or type(exc).__name__)
        finally:
            if not selected and device.opened:
                device.close()
    detail = "；".join(failures)
    raise RuntimeError(f"无法打开所选 Azure Kinect {serial}；请检查 USB 3.0、设备权限和占用。{detail}")


class AzureKinectCamera:
    def __init__(
        self,
        color_size: tuple[int, int] = (1280, 720),
        fps: int = 30,
        serial: str = "",
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self.color_size, self.fps, self.serial = color_size, fps, serial
        self.output_size = output_size or color_size
        self.metadata: dict = {}
        self._device = None
        self._start_requested = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._error: str | None = None
        self._timestamps = rgbd_timing.TimestampTracker()

    def connect(self) -> "AzureKinectCamera":
        if self._device is not None:
            raise RuntimeError("Azure Kinect 已连接或尚未完成释放")
        sdk = require_sdk()
        if self.color_size not in COLOR_MODES or self.fps not in (5, 15, 30):
            raise ValueError("Azure Kinect 不支持所选分辨率/FPS，请读取分辨率后重新选择")
        if self.color_size == (4096, 3072) and self.fps > 15:
            raise ValueError("Azure Kinect 4096×3072 最高支持 15 FPS")
        entries = [item for item in list_devices() if not self.serial or item["serial"] == self.serial]
        if len(entries) != 1:
            raise RuntimeError("未找到唯一的所选 Azure Kinect，请重新检测并明确分配序列号")
        entry = entries[0]
        if entry["usb_speed_mbps"] < 5000:
            raise RuntimeError("Azure Kinect 当前为 USB 2.0，请使用 USB 3.0 数据线与接口（至少 5000 Mbps）")
        self.serial = entry["serial"]
        config = sdk.Config(
            color_resolution=getattr(sdk.ColorResolution, COLOR_MODES[self.color_size]),
            color_format=sdk.ImageFormat.COLOR_BGRA32,
            depth_mode=sdk.DepthMode.NFOV_UNBINNED,
            camera_fps=getattr(sdk.FPS, f"FPS_{self.fps}"),
            synchronized_images_only=True,
        )
        self._device = _open_selected(sdk, self.serial, config)
        self._stop.clear()
        self._ready.clear()
        self._latest, self._error = None, None
        self._timestamps = rgbd_timing.TimestampTracker()
        self.metadata = {
            **entry,
            "width": self.output_size[0],
            "height": self.output_size[1],
            "capture_width": self.color_size[0],
            "capture_height": self.color_size[1],
            "fps": self.fps,
            "color_format": "BGRA32",
            "depth_mode": "NFOV_UNBINNED",
            "depth_capture_size": [640, 576],
            "alignment": "calibrated_D2C",
            "frame_sync_requested": True,
            "frame_sync_error": None,
            "global_timestamp_supported": False,
            "global_timestamp_enabled": False,
            "timestamp_note": "设备时间戳与主机单调时钟分别记录；没有跨设备同步曝光验证",
            "unit": "m",
            "depth_encoding_range_m": [0.0, DEPTH_MAX_METERS],
        }
        try:
            self._start_requested = True
            self._device.start()
            self._thread = threading.Thread(
                target=self._read_loop, name=f"nero-k4a-{self.serial}", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(10):
                with self._lock:
                    detail = self._error
                raise TimeoutError(
                    "Azure Kinect 预热超时，未收到可用 RGBD；"
                    + (detail if detail else "请检查 USB、供电和 SDK 日志")
                )
            self.read_snapshot()
            return self
        except Exception as exc:
            try:
                self.disconnect()
            except Exception as cleanup:
                raise RuntimeError(f"{exc}；相机释放失败：{cleanup}") from exc
            raise

    @property
    def connected(self) -> bool:
        return self._device is not None and self._thread is not None and self._thread.is_alive()

    def _convert(self, capture: Any) -> dict:
        received_at = time.perf_counter()
        color, raw_depth = capture.color, capture.depth
        if color is None or raw_depth is None:
            raise ValueError("Azure Kinect capture 缺少 RGB 或深度")
        width, height = self.color_size
        if color.shape != (height, width, 4) or color.dtype != np.uint8:
            raise ValueError("Azure Kinect 彩色尺寸或 BGRA32 格式不符")
        aligned = capture.transformed_depth  # SDK 使用出厂标定将深度投影到彩色坐标系。
        if aligned is None or aligned.shape != (height, width) or aligned.dtype != np.uint16:
            raise ValueError("Azure Kinect D2C 输出尺寸或 DEPTH16 格式不符")
        rgb = resolution.resize(cv2.cvtColor(color, cv2.COLOR_BGRA2RGB), self.output_size)
        # 先按最近邻缩小 uint16，再转米；无效零值仍为零。
        depth = resolution.resize(aligned[..., None], self.output_size, depth=True).astype(np.float32) / 1000
        rgb.setflags(write=False)
        depth.setflags(write=False)
        return {
            "rgb": rgb,
            "depth": depth,
            "capture_timestamp": received_at,
            "rgb_timestamp_us": int(capture.color_timestamp_usec),
            "depth_timestamp_us": int(capture.depth_timestamp_usec),
            "rgb_system_timestamp_us": capture.color_system_timestamp_nsec / 1000,
            "depth_system_timestamp_us": capture.depth_system_timestamp_nsec / 1000,
            "rgb_global_timestamp_us": None,
            "depth_global_timestamp_us": None,
            "depth_scale_mm": 1.0,
            "invalid_fraction": float(np.count_nonzero(depth == 0) / depth.size),
            "above_range_fraction": float(np.count_nonzero(depth > DEPTH_MAX_METERS) / depth.size),
        }

    def _publish(self, sample: dict) -> None:
        identity = (sample["rgb_timestamp_us"], sample["depth_timestamp_us"])
        with self._lock:
            old = self._latest
            if (
                old is not None
                and all(rgbd_timing.positive(v) for v in identity)
                and identity == (old["rgb_timestamp_us"], old["depth_timestamp_us"])
            ):
                return  # SDK 重复交付同一帧不能刷新帧龄。
        skew = rgbd_timing.receive_skew_ms(sample)
        if skew is not None and skew > rgbd_timing.receive_skew_limit_ms(self.fps):
            raise TimeoutError(f"Azure Kinect RGB/深度接收时差过大：{skew:.1f} ms")
        sample["rgb_depth_receive_skew_ms"] = skew
        # pyk4a 的 system_timestamp_nsec 与 perf_counter 同一时基，可发现 SDK 队列积压。
        receive = [sample[f"{kind}_system_timestamp_us"] for kind in ("rgb", "depth")]
        if all(rgbd_timing.positive(v) for v in receive):
            age = (sample["capture_timestamp"] - min(receive) / 1e6) * 1000
            sample["sdk_delivery_age_ms"] = round(age, 3)
            if age > 500:
                raise TimeoutError(f"Azure Kinect SDK 帧已积压 {age:.1f} ms")
        sample.update(self._timestamps.assess(sample))
        with self._lock:
            self._latest, self._error = sample, None
        self._ready.set()

    def _read_loop(self) -> None:
        sdk = require_sdk()
        while not self._stop.is_set():
            try:
                capture = self._device.get_capture(timeout=100)
                self._publish(self._convert(capture))
            except sdk.K4ATimeoutException:
                continue  # read_snapshot 的帧龄上限负责拒绝长期不更新。
            except Exception as exc:
                with self._lock:
                    self._error = str(exc) or type(exc).__name__
                self._stop.wait(0.01)

    def read_snapshot(self, max_age_ms: float = 500) -> dict:
        with self._lock:
            sample, error = self._latest, self._error
        if error:
            raise RuntimeError("Azure Kinect RGBD 读取失败：" + error)
        if sample is None or not self.connected:
            raise RuntimeError("Azure Kinect 尚无完整 RGBD 帧或后台线程已退出")
        age = (time.perf_counter() - sample["capture_timestamp"]) * 1000
        age += max(0.0, sample.get("sdk_delivery_age_ms", 0.0))
        if age < 0 or age > max_age_ms:
            raise TimeoutError(f"Azure Kinect RGBD 帧已过期（{age:.1f} ms）")
        return {**sample, "age_ms": round(age, 3)}

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Azure Kinect 后台线程未退出，保留句柄；请检查设备后重试结束采集")
        if self._device is not None:
            if self._start_requested:
                self._device.stop()  # 也负责关闭设备；start 中途失败同样停止已启动的部分。
                self._start_requested = False
            elif self._device.opened:
                self._device.close()
        self._device = self._thread = None
        with self._lock:
            self._latest, self._error = None, None
