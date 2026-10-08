"""RealSense RGBD：按序列号选机，后台配对、标定 D2C，前台只取快照。"""

import math
import threading
import time
from typing import Any

import numpy as np

from . import resolution, rgbd_timing
from .depth_images import DEPTH_MAX_METERS

try:
    import pyrealsense2 as rs
except (ImportError, OSError) as exc:
    rs = None
    _SDK_ERROR = str(exc)
else:
    _SDK_ERROR = ""


def require_sdk() -> Any:
    if rs is None:
        raise ImportError("RealSense 需要 pyrealsense2；SDK 加载失败：" + _SDK_ERROR)
    return rs


def list_devices() -> list[dict]:
    sdk = require_sdk()
    context = sdk.context()
    entries = []
    for device in context.query_devices():
        if not any(sensor.is_depth_sensor() for sensor in device.query_sensors()):
            continue
        serial = device.get_info(sdk.camera_info.serial_number)
        if not serial:
            raise RuntimeError("已发现 RealSense 深度设备，但无法读取序列号")
        entries.append({"name": device.get_info(sdk.camera_info.name), "serial": serial})
    return entries


def _select(context: Any, serial: str) -> Any:
    sdk = require_sdk()
    matches = [d for d in context.query_devices() if d.get_info(sdk.camera_info.serial_number) == serial]
    if not serial or len(matches) != 1:
        raise RuntimeError(f"未找到唯一的 RealSense {serial}，请重新检测并指定序列号")
    return matches[0]


def _modes(device: Any) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    sdk = require_sdk()
    color, depth = set(), set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            mode = (video.width(), video.height(), video.fps())
            if profile.stream_type() == sdk.stream.color and profile.format() == sdk.format.rgb8:
                color.add(mode)
            elif profile.stream_type() == sdk.stream.depth and profile.format() == sdk.format.z16:
                depth.add(mode)
    return color, depth


def list_profiles(serial: str) -> dict:
    sdk = require_sdk()
    context = sdk.context()
    color, depth = _modes(_select(context, serial))
    depth_rates = {fps for _, _, fps in depth}
    sizes: dict[tuple[int, int], list[int]] = {}
    for width, height, fps in sorted(color):
        if fps in depth_rates:
            sizes.setdefault((width, height), []).append(fps)
    if not sizes:
        raise RuntimeError("RealSense 没有可配对的 RGB8 / Z16 模式")
    return {
        "profiles": [
            {"width": w, "height": h, "fps": rates, "format": "RGB8", "reason": ""}
            for (w, h), rates in sizes.items()
        ],
        "ranges": [],
        "serial": serial,
        "source": "RealSense SDK 实机模式；深度选择同 FPS 模式，按出厂标定对齐到 RGB",
    }


def _depth_size(modes: set[tuple[int, int, int]], color: tuple[int, int], fps: int) -> tuple[int, int]:
    matches = [(w, h) for w, h, rate in modes if rate == fps]
    if not matches:
        raise ValueError(f"RealSense 深度不支持 {fps} FPS")
    # 优先同尺寸，其次相近宽高比及像素数；实际原始深度尺寸写入报告。
    w, h = color
    return min(
        matches,
        key=lambda size: (round(abs(size[0] / size[1] - w / h), 1), abs(size[0] * size[1] - w * h), size),
    )


def preview_camera(serial: str) -> "RealSenseCamera":
    profiles = list_profiles(serial)["profiles"]
    rates = {fps for p in profiles for fps in p["fps"]}
    fps = 30 if 30 in rates else min(rates)
    choices = [p for p in profiles if fps in p["fps"]]
    profile = min(choices, key=lambda p: abs(p["width"] - 1280) + abs(p["height"] - 720))
    w, h = profile["width"], profile["height"]
    factor = max(w / 640, h / 480, 1)
    output = (max(2, int(w / factor) // 2 * 2), max(2, int(h / factor) // 2 * 2))
    return RealSenseCamera(serial=serial, fps=fps, color_size=(w, h), output_size=output)


def _frame_clock(frame: Any, prefix: str) -> dict:
    sdk = require_sdk()
    domain = frame.get_frame_timestamp_domain()
    stamp = float(frame.get_timestamp()) * 1000  # SDK timestamp 为毫秒。
    arrival = None
    key = sdk.frame_metadata_value.time_of_arrival
    if frame.supports_frame_metadata(key):
        arrival = float(frame.get_frame_metadata(key)) * 1000  # 主机 Unix 毫秒，不是 perf_counter。
    return {
        f"{prefix}_frame_index": int(frame.get_frame_number()),
        f"{prefix}_timestamp_domain": str(domain),
        f"{prefix}_sdk_timestamp_us": stamp,
        f"{prefix}_timestamp_us": stamp if domain == sdk.timestamp_domain.hardware_clock else None,
        f"{prefix}_global_timestamp_us": stamp if domain == sdk.timestamp_domain.global_time else None,
        f"{prefix}_system_timestamp_us": arrival,
    }


class RealSenseCamera:
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
        self._context = self._pipeline = self._align = None
        self._started = False
        self._depth_scale = 0.0
        self._thread: threading.Thread | None = None
        self._stop, self._ready = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._error: str | None = None
        self._timestamps = rgbd_timing.TimestampTracker()
        self._warm_frames = 0

    def connect(self) -> "RealSenseCamera":
        if self._pipeline is not None:
            raise RuntimeError("RealSense 已连接或尚未完成释放")
        sdk = require_sdk()
        context = sdk.context()
        device = _select(context, self.serial)
        color, depth = _modes(device)
        if (*self.color_size, self.fps) not in color:
            raise ValueError(f"RealSense 不支持 RGB {self.color_size} @ {self.fps} FPS，请读取分辨率后重选")
        depth_size = _depth_size(depth, self.color_size, self.fps)
        config = sdk.config()
        config.enable_device(self.serial)
        config.enable_stream(sdk.stream.color, *self.color_size, sdk.format.rgb8, self.fps)
        config.enable_stream(sdk.stream.depth, *depth_size, sdk.format.z16, self.fps)
        global_states, global_errors = [], []
        for sensor in device.query_sensors():
            # 只配置本次 RGB/深度传感器；不启动 IMU，也不改变曝光、激光功率等参数。
            if not any(
                p.stream_type() in (sdk.stream.color, sdk.stream.depth) for p in sensor.get_stream_profiles()
            ):
                continue
            supported = sensor.supports(sdk.option.global_time_enabled)
            enabled = False
            if supported:
                try:
                    sensor.set_option(sdk.option.global_time_enabled, 1)
                    enabled = bool(sensor.get_option(sdk.option.global_time_enabled))
                except RuntimeError as exc:
                    global_errors.append(str(exc))
            global_states.append((supported, enabled))
        self._context, self._pipeline = context, sdk.pipeline(context)
        self._stop.clear()
        self._ready.clear()
        self._latest, self._error = None, None
        self._timestamps = rgbd_timing.TimestampTracker()
        self._warm_frames = 0
        try:
            active = self._pipeline.start(config)
            self._started = True
            actual_device = active.get_device()
            if actual_device.get_info(sdk.camera_info.serial_number) != self.serial:
                raise RuntimeError("RealSense 打开设备与所选序列号不一致")
            for stream, size, fmt in (
                (sdk.stream.color, self.color_size, sdk.format.rgb8),
                (sdk.stream.depth, depth_size, sdk.format.z16),
            ):
                profile = active.get_stream(stream).as_video_stream_profile()
                if (profile.width(), profile.height(), profile.fps(), profile.format()) != (
                    *size,
                    self.fps,
                    fmt,
                ):
                    raise RuntimeError("RealSense 实际分辨率、格式或 FPS 与配置不一致")
            self._depth_scale = actual_device.first_depth_sensor().get_depth_scale()
            if not math.isfinite(self._depth_scale) or self._depth_scale <= 0:
                raise ValueError("RealSense 深度单位无效")
            self._align = sdk.align(sdk.stream.color)
            self.metadata = {
                "name": actual_device.get_info(sdk.camera_info.name),
                "serial": self.serial,
                "width": self.output_size[0],
                "height": self.output_size[1],
                "capture_width": self.color_size[0],
                "capture_height": self.color_size[1],
                "depth_capture_size": list(depth_size),
                "fps": self.fps,
                "color_format": "RGB8",
                "depth_format": "Z16",
                "depth_scale_m": self._depth_scale,
                "alignment": "calibrated_D2C",
                "frame_sync_requested": True,
                "frame_sync_error": None,
                "global_timestamp_supported": bool(global_states) and all(s for s, _ in global_states),
                "global_timestamp_enabled": bool(global_states) and all(e for _, e in global_states),
                "global_timestamp_error": "; ".join(global_errors) or None,
                "timestamp_note": "SDK 配对；按每帧时钟域记录时间戳，主机接收时钟为 Unix 时间；未验证跨设备同步曝光",
                "unit": "m",
                "depth_encoding_range_m": [0.0, DEPTH_MAX_METERS],
            }
            self._thread = threading.Thread(
                target=self._read_loop, name=f"nero-rs-{self.serial}", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(10):
                raise TimeoutError(
                    "RealSense 预热超时，未收到可用 RGBD；" + (self._error or "请检查 USB、权限和设备占用")
                )
            self.read_snapshot()
            return self
        except Exception as exc:
            try:
                self.disconnect()
            except Exception as cleanup:
                raise RuntimeError(f"{exc}；RealSense 释放失败：{cleanup}") from exc
            raise

    @property
    def connected(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    def _convert(self, frames: Any) -> dict:
        received_at, wall_us = time.perf_counter(), time.time() * 1e6
        color, raw_depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not raw_depth:
            raise ValueError("RealSense 帧组缺少 RGB 或深度")
        sample = {
            "capture_timestamp": received_at,
            **_frame_clock(color, "rgb"),
            **_frame_clock(raw_depth, "depth"),
        }
        skew = rgbd_timing.receive_skew_ms(sample)
        if skew is not None and skew > rgbd_timing.receive_skew_limit_ms(self.fps):
            raise TimeoutError(f"RealSense RGB/深度接收时差过大：{skew:.1f} ms")
        sample["rgb_depth_receive_skew_ms"] = skew
        arrivals = [sample[f"{kind}_system_timestamp_us"] for kind in ("rgb", "depth")]
        if all(rgbd_timing.positive(v) for v in arrivals):
            age = (wall_us - min(arrivals)) / 1000
            if age > 500:
                raise TimeoutError(f"RealSense SDK 帧已积压 {age:.1f} ms")
            # 系统时钟可能被校时；负值不混入单调时钟帧龄。
            sample["sdk_delivery_age_ms"] = max(0.0, age)
        aligned = self._align.process(frames)
        rgb = np.asanyarray(aligned.get_color_frame().get_data())
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        w, h = self.color_size
        if (
            rgb.shape != (h, w, 3)
            or rgb.dtype != np.uint8
            or depth.shape != (h, w)
            or depth.dtype != np.uint16
        ):
            raise ValueError("RealSense RGB / D2C 深度尺寸或像素格式不符")
        # 复制脱离 SDK 帧缓冲，下一帧到来或停流后快照仍有效。
        rgb = resolution.resize(rgb, self.output_size).copy()
        depth = (
            resolution.resize(depth[..., None], self.output_size, depth=True).astype(np.float32)
            * self._depth_scale
        )
        rgb.setflags(write=False)
        depth.setflags(write=False)
        return {
            **sample,
            "rgb": rgb,
            "depth": depth,
            "depth_scale_mm": self._depth_scale * 1000,
            "invalid_fraction": float(np.count_nonzero(depth == 0) / depth.size),
            "above_range_fraction": float(np.count_nonzero(depth > DEPTH_MAX_METERS) / depth.size),
        }

    def _publish(self, sample: dict) -> None:
        with self._lock:
            previous = self._latest
            if previous is not None and any(
                sample[f"{s}_frame_index"] == previous[f"{s}_frame_index"] for s in ("rgb", "depth")
            ):
                self._warm_frames = 0
                return  # 任一路重复不能用另一流的新帧刷新整个 RGBD 缓存帧龄。
        if not self._ready.is_set() and sample.get(
            "sdk_delivery_age_ms", 0
        ) > rgbd_timing.receive_skew_limit_ms(self.fps):
            self._warm_frames = 0
            return
        sample.update(self._timestamps.assess(sample))
        with self._lock:
            self._latest, self._error = sample, None
        self._warm_frames += 1
        if self._warm_frames >= max(3, math.ceil(self.fps * 0.5)):
            self._ready.set()  # 开流后先等至少半秒连续新鲜配对，避免短暂首帧后又读到积压。

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frames = self._pipeline.try_wait_for_frames(100)  # 官方绑定释放 GIL。
                if ok:
                    self._publish(self._convert(frames))
            except Exception as exc:
                self._warm_frames = 0
                with self._lock:
                    self._error = str(exc) or type(exc).__name__
                self._stop.wait(0.01)

    def read_snapshot(self, max_age_ms: float = 500) -> dict:
        with self._lock:
            sample, error = self._latest, self._error
        if error:
            raise RuntimeError("RealSense RGBD 读取失败：" + error)
        if sample is None or not self.connected:
            raise RuntimeError("RealSense 尚无完整 RGBD 帧或后台线程已退出")
        age = (time.perf_counter() - sample["capture_timestamp"]) * 1000 + sample.get(
            "sdk_delivery_age_ms", 0
        )
        if age < 0 or age > max_age_ms:
            raise TimeoutError(f"RealSense RGBD 帧已过期（{age:.1f} ms）")
        return {**sample, "age_ms": round(age, 3)}

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("RealSense 后台线程未退出，保留句柄；请检查设备后重试")
        if self._started:
            self._pipeline.stop()
            self._started = False
        self._pipeline = self._context = self._align = self._thread = None
        with self._lock:
            self._latest, self._error = None, None
