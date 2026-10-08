"""Femto Mega 完整 RGBD 后台采集；采样线程只取原子快照，不等待 USB。"""

import math
import threading
import time
from contextlib import suppress
from typing import Any

import cv2
import numpy as np

from . import resolution, rgbd_timing
from .depth_images import DEPTH_MAX_METERS

try:
    import pyorbbecsdk as ob
except ImportError:
    ob = None

DEFAULT_COLOR_SIZE = (1280, 720)


def frame_number(frame: Any, method: str) -> int | None:
    """元数据缺失不使正常 RGBD 图像丢帧；零值原样保留供有效性检查。"""
    try:
        value = getattr(frame, method)()
        return int(value) if not isinstance(value, bool) else None
    except Exception:
        return None


def require_sdk() -> Any:
    if ob is None:
        raise RuntimeError("当前 Python 环境未安装 pyorbbecsdk；可取消分配 RGBD 设备后继续使用 RGB")
    return ob


def device_entries(devices: Any) -> list[dict[str, str]]:
    entries = []
    for index in range(devices.get_count()):
        name = devices.get_device_name_by_index(index)
        if "femto mega" in name.lower():
            serial = devices.get_device_serial_number_by_index(index)
            if not serial:
                # 部分 Femto 固件的列表序列号为空，DeviceInfo 才有真实序列号。
                # 只读设备信息，不创建 Pipeline、不启动视频流。
                info = devices.get_device_by_index(index).get_device_info()
                serial = info.get_serial_number()
            if not serial:
                raise RuntimeError(f"已发现 {name}，但无法读取序列号，请检查设备连接与占用情况")
            entries.append({"name": name, "serial": serial})
    return entries


def device_by_serial(devices: Any, serial: str) -> Any:
    """列表没有序列号时，SDK 的按序列号查找同样失效；用设备信息逐项核对。"""
    for index in range(devices.get_count()):
        listed_serial = devices.get_device_serial_number_by_index(index)
        if listed_serial == serial:
            return devices.get_device_by_serial_number(serial)
        if not listed_serial and "femto mega" in devices.get_device_name_by_index(index).lower():
            device = devices.get_device_by_index(index)
            if device.get_device_info().get_serial_number() == serial:
                return device
    raise RuntimeError(f"未找到所选 Femto Mega：{serial}")


def list_devices() -> list[dict[str, str]]:
    """仅显式刷新时枚举，不启动流，不放进状态轮询或默认连接流程。"""
    context = require_sdk().Context()
    return device_entries(context.query_devices())


def detect_femto() -> bool:
    return bool(list_devices())


def color_formats(sdk: Any) -> list[Any]:
    return [
        getattr(sdk.OBFormat, name)
        for name in ("RGB", "MJPG", "BGR", "YUYV", "UYVY")
        if hasattr(sdk.OBFormat, name)
    ]


def matching_depths(pipeline: Any, color: Any, sdk: Any) -> list[Any]:
    profiles = pipeline.get_d2c_depth_profile_list(color, sdk.OBAlignMode.HW_MODE)
    return [
        profiles[i]
        for i in range(len(profiles))
        if profiles[i].get_fps() == color.get_fps() and profiles[i].get_format() == sdk.OBFormat.Y16
    ]


def list_profiles(serial: str = "") -> dict:
    """读取设备声明的彩色尺寸和硬件 D2C 组合，不 start 管线。"""
    sdk = require_sdk()
    context = sdk.Context()
    devices = context.query_devices()
    matches = [entry for entry in device_entries(devices) if not serial or entry["serial"] == serial]
    if len(matches) != 1:
        raise RuntimeError("未找到唯一的 Femto Mega；请在左侧刷新设备并选择序列号")
    pipeline = sdk.Pipeline(device_by_serial(devices, matches[0]["serial"]))
    colors = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    profiles = []
    for i in range(len(colors)):
        color = colors[i]
        reason = ""
        if color.get_format() not in color_formats(sdk):
            reason = "当前平台不支持此像素格式"
        else:
            try:
                if not matching_depths(pipeline, color, sdk):
                    reason = "无匹配的硬件 D2C 深度模式"
            except Exception as exc:
                reason = f"D2C 查询失败：{exc}"
        profiles.append(
            {
                "width": color.get_width(),
                "height": color.get_height(),
                "format": str(color.get_format()),
                "fps": [color.get_fps()],
                "reason": reason,
            }
        )
    return {"profiles": profiles, "ranges": [], "serial": matches[0]["serial"]}


class FemtoCamera:
    def __init__(
        self,
        color_size: tuple[int, int] = DEFAULT_COLOR_SIZE,
        depth_size: tuple[int, int] = (640, 576),
        fps: int = 30,
        serial: str = "",
        output_size: tuple[int, int] | None = None,
    ):
        self.color_size, self.depth_size, self.fps, self.serial = color_size, depth_size, fps, serial
        self.output_size = output_size or color_size
        self._color_format = None
        self._pipeline = None
        self._start_requested = False
        self._context = None
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._error: str | None = None
        self.metadata: dict = {}
        self._timestamps = rgbd_timing.TimestampTracker()

    def connect(self) -> "FemtoCamera":
        if self._pipeline is not None:
            raise RuntimeError("Femto RGBD 已连接")
        sdk = require_sdk()
        self._context = sdk.Context()
        devices = self._context.query_devices()
        entries = device_entries(devices)
        matches = [entry for entry in entries if not self.serial or entry["serial"] == self.serial]
        if len(matches) != 1:
            raise RuntimeError("未找到所选 Femto Mega，或有多台未指定序列号；请重新扫描相机并明确分配设备")
        self.serial = matches[0]["serial"]
        device = device_by_serial(devices, self.serial)
        pipeline = sdk.Pipeline(device)
        self._pipeline = pipeline
        self._stop.clear()
        self._ready.clear()
        self._latest = None
        self._error = None
        self._timestamps = rgbd_timing.TimestampTracker()
        try:
            global_supported, global_enabled, global_error = None, False, None
            try:
                global_supported = bool(device.is_global_timestamp_supported())
                if global_supported:
                    device.enable_global_timestamp(True)
                    global_enabled = True
            except Exception as exc:
                global_error = str(exc)
            profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            colors = [profiles[i] for i in range(len(profiles))]
            formats = color_formats(sdk)
            colors = [
                p
                for p in colors
                if (p.get_width(), p.get_height(), p.get_fps()) == (*self.color_size, self.fps)
                and p.get_format() in formats
            ]
            colors.sort(key=lambda p: formats.index(p.get_format()))
            color, depths = None, []
            for candidate in colors:
                try:
                    depths = matching_depths(pipeline, candidate, sdk)
                except Exception:
                    continue
                if depths:
                    color = candidate
                    break
            if color is None:
                raise RuntimeError(
                    f"Femto 相机不支持 RGBD {self.color_size} @ {self.fps} FPS 的硬件 D2C 组合；请重新选择"
                )
            self._color_format = color.get_format()
            depth = next((p for p in depths if (p.get_width(), p.get_height()) == self.depth_size), depths[0])
            cfg = sdk.Config()
            cfg.enable_stream(color)
            cfg.enable_stream(depth)
            cfg.set_align_mode(sdk.OBAlignMode.HW_MODE)
            cfg.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            sync_error = None
            try:
                pipeline.enable_frame_sync()
            except Exception as exc:
                sync_error = str(exc)
            self.metadata = {
                **matches[0],
                "width": self.output_size[0],
                "height": self.output_size[1],
                "capture_width": self.color_size[0],
                "capture_height": self.color_size[1],
                "color_format": str(self._color_format),
                "fps": self.fps,
                "alignment": "hardware_D2C",
                "frame_sync_requested": sync_error is None,
                "frame_sync_error": sync_error,
                "startup_queue_reset": False,
                "receive_skew_limit_ms": rgbd_timing.receive_skew_limit_ms(self.fps),
                "global_timestamp_supported": global_supported,
                "global_timestamp_enabled": global_enabled,
                "global_timestamp_error": global_error,
                "unit": "m",
                "depth_encoding_range_m": [0.0, DEPTH_MAX_METERS],
            }
            self._start_requested = True
            pipeline.start(cfg)
            self._thread = threading.Thread(target=self._read_loop, name="nero-head-rgbd", daemon=True)
            self._thread.start()
            if not self._ready.wait(4.0):
                raise RuntimeError("Femto RGBD 预热超时：" + (self._error or "未收到完整的对齐 RGB/深度帧"))
            self.read_snapshot()  # 首次完整帧的尺寸/比例/数据必须通过校验。
            return self
        except Exception:
            with suppress(Exception):
                self.disconnect()
            raise

    @property
    def connected(self) -> bool:
        thread = self._thread
        return self._pipeline is not None and thread is not None and thread.is_alive()

    def _convert(self, frames: Any) -> dict:
        received_at = time.perf_counter()
        rgb_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if rgb_frame is None or depth_frame is None:
            raise ValueError("FrameSet 缺少 RGB 或深度帧")
        sdk = require_sdk()
        color_format = self._color_format if self._color_format is not None else sdk.OBFormat.RGB
        if rgb_frame.get_format() != color_format or depth_frame.get_format() != sdk.OBFormat.Y16:
            raise ValueError("Femto RGBD 返回的像素格式与请求不一致")
        width, height = self.color_size
        if any(
            (frame.get_width(), frame.get_height()) != (width, height) for frame in (rgb_frame, depth_frame)
        ):
            raise ValueError("硬件 D2C 输出未对齐到声明的 RGB 分辨率，不能按该尺寸保存")
        scale = float(depth_frame.get_depth_scale())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("深度比例系数无效")
        data = np.frombuffer(rgb_frame.get_data(), dtype=np.uint8)
        if color_format == sdk.OBFormat.RGB:
            rgb = data.reshape(height, width, 3).copy()
        elif color_format == getattr(sdk.OBFormat, "MJPG", None):
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (height, width, 3):
                raise ValueError("Femto MJPG 解码失败或尺寸不一致")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif color_format == getattr(sdk.OBFormat, "BGR", None):
            rgb = cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        else:
            code = (
                cv2.COLOR_YUV2RGB_YUYV
                if color_format == getattr(sdk.OBFormat, "YUYV", None)
                else cv2.COLOR_YUV2RGB_UYVY
            )
            rgb = cv2.cvtColor(data.reshape(height, width, 2), code)
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth = (raw.astype(np.float32) * (scale / 1000.0))[..., None]
        rgb = resolution.resize(rgb, self.output_size)
        depth = resolution.resize(depth, self.output_size, depth=True)
        rgb.setflags(write=False)
        depth.setflags(write=False)
        return {
            "rgb": rgb,
            "depth": depth,
            "capture_timestamp": received_at,
            "rgb_timestamp_us": frame_number(rgb_frame, "get_timestamp_us"),
            "depth_timestamp_us": frame_number(depth_frame, "get_timestamp_us"),
            "rgb_global_timestamp_us": frame_number(rgb_frame, "get_global_timestamp_us"),
            "depth_global_timestamp_us": frame_number(depth_frame, "get_global_timestamp_us"),
            # 主机接收时间用于识别新帧和配对队列积压，不参与曝光时差校验。
            "rgb_system_timestamp_us": frame_number(rgb_frame, "get_system_timestamp_us"),
            "depth_system_timestamp_us": frame_number(depth_frame, "get_system_timestamp_us"),
            "rgb_frame_index": frame_number(rgb_frame, "get_index"),
            "depth_frame_index": frame_number(depth_frame, "get_index"),
            "depth_scale_mm": scale,
            "invalid_fraction": float(np.count_nonzero(depth == 0) / depth.size),
            "above_range_fraction": float(np.count_nonzero(depth > DEPTH_MAX_METERS) / depth.size),
        }

    def _publish(self, sample: dict) -> None:
        skew = rgbd_timing.receive_skew_ms(sample)
        limit = rgbd_timing.receive_skew_limit_ms(self.fps)
        if skew is not None and skew > limit:
            raise TimeoutError(
                f"RGB/深度主机接收时差 {skew:.1f} ms 超过 {limit:.0f} ms，疑似旧帧积压；"
                "本帧不用于录制，若持续出现请结束采集后重新开始采集"
            )
        sample["rgb_depth_receive_skew_ms"] = skew
        fields = (
            "rgb_timestamp_us",
            "depth_timestamp_us",
            "rgb_global_timestamp_us",
            "depth_global_timestamp_us",
            "rgb_system_timestamp_us",
            "depth_system_timestamp_us",
            "rgb_frame_index",
            "depth_frame_index",
        )
        identity = tuple(sample.get(key) for key in fields)
        with self._lock:
            previous = self._latest
            if (
                any(rgbd_timing.positive(value) for value in identity)
                and previous is not None
                and identity == tuple(previous.get(key) for key in fields)
            ):
                self._error = None
                return  # 同一 SDK 帧不刷新缓存帧龄，也不重复推进时钟检查。
        sample.update(self._timestamps.assess(sample))
        with self._lock:
            self._latest, self._error = sample, None
        self._ready.set()

    def _discard_startup_frames(self) -> None:
        """先等两路启动，再清配对队列和输出队列；只在后台启动时执行一次。"""
        deadline = time.monotonic() + 3.0
        while not self._stop.is_set() and time.monotonic() < deadline:
            frames = self._pipeline.wait_for_frames(100)
            if (
                frames is not None
                and frames.get_color_frame() is not None
                and frames.get_depth_frame() is not None
            ):
                break
        else:
            raise TimeoutError("未收到两路完整帧，无法清理启动缓存")

        # SDK 2.9.3 切换同步模式会清空配对队列。时间戳为 0 时，启动先到的
        # RGB 会一直与较晚到达的深度配对；只多等几秒不能消除固定积压。
        self._pipeline.disable_frame_sync()
        if self.metadata["frame_sync_requested"]:
            self._pipeline.enable_frame_sync()
        discarded = 1
        deadline = time.monotonic() + 1.0
        while not self._stop.is_set() and time.monotonic() < deadline:
            # 配对队列重置不会清 SDK 的 FrameSet 输出队列，旧输出也必须丢弃。
            if self._pipeline.wait_for_frames(1) is None:
                self.metadata.update(startup_queue_reset=True, startup_frames_discarded=discarded)
                return
            discarded += 1
        raise TimeoutError("启动输出队列未能排空，请检查Femto 相机后重试")

    def _read_loop(self) -> None:
        try:
            self._discard_startup_frames()
        except Exception as exc:
            with self._lock:
                self._error = "启动缓存清理失败：" + str(exc)
            self._ready.set()  # 唤醒 connect，由 read_snapshot 报错并关闭管线。
            return
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                self._publish(self._convert(frames))
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(0.01)

    def read_snapshot(self, max_age_ms: float = 500) -> dict:
        with self._lock:
            sample, error = self._latest, self._error
        if error:
            raise RuntimeError("Femto RGBD 读取失败：" + error)
        if sample is None or not self.connected:
            raise RuntimeError("Femto RGBD 尚无可用完整帧或后台线程已停止")
        age = (time.perf_counter() - sample["capture_timestamp"]) * 1000
        if age < 0 or age > max_age_ms:
            raise TimeoutError(f"Femto RGBD 帧已过期（{age:.1f} ms）")
        return {**sample, "age_ms": round(age, 3)}

    def status(self) -> dict:
        try:
            sample = self.read_snapshot()
            return {
                **self.metadata,
                "connected": True,
                "age_ms": sample["age_ms"],
                "invalid_fraction": sample["invalid_fraction"],
                "above_range_fraction": sample["above_range_fraction"],
                "rgb_depth_receive_skew_ms": sample.get("rgb_depth_receive_skew_ms"),
                **{key: sample.get(key) for key in rgbd_timing.CHECK_FIELDS},
                "timestamp_message": rgbd_timing.STATUS_TEXT.get(
                    sample.get("timestamp_status"), rgbd_timing.STATUS_TEXT["unavailable"]
                ),
                "error": None,
            }
        except Exception as exc:
            return {**self.metadata, "connected": self.connected, "error": str(exc)}

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        pipeline = self._pipeline
        if pipeline is not None and self._start_requested:
            pipeline.stop()
            self._start_requested = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("Femto 相机后台线程未退出，请结束平台进程后检查设备")
        self._pipeline = self._thread = self._context = None
        with self._lock:
            self._latest = None
