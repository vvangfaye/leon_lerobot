"""采集尺寸、保存缩放与 Linux V4L2 能力枚举；不启动视频流。"""

import errno
import fcntl
import math
import os
import struct
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np


def defaults() -> dict[str, dict[str, int]]:
    return {
        "left": {"width": 640, "height": 480, "scale": 1},
        "right": {"width": 640, "height": 480, "scale": 1},
        "head": {"width": 1280, "height": 720, "scale": 1},
    }


def output_size(setting: dict[str, int]) -> tuple[int, int]:
    # 视频编码要求偶数边长；只向下取整，不拉伸到另一种宽高比。
    return tuple(max(2, (setting[key] // setting["scale"]) // 2 * 2) for key in ("width", "height"))


def scales(width: int, height: int) -> list[int]:
    options = [1]
    while width / options[-1] > 640 and height / options[-1] > 480:
        options.append(options[-1] * 2)
    return options


def validate(settings: Any) -> None:
    if not isinstance(settings, dict) or set(settings) != {"left", "right", "head"}:
        raise ValueError("分辨率配置必须包含 left、right、head 三路相机")
    for role, item in settings.items():
        if not isinstance(item, dict) or set(item) != {"width", "height", "scale"}:
            raise ValueError(f"{role} 分辨率配置需要 width、height、scale")
        if any(type(value) is not int for value in item.values()):
            raise ValueError(f"{role} 分辨率与缩放必须为整数")
        if any(not 2 <= item[key] <= 16384 or item[key] % 2 for key in ("width", "height")):
            raise ValueError(f"{role} 分辨率必须为 2～16384 之间的偶数")
        if item["scale"] not in scales(item["width"], item["height"]):
            raise ValueError(f"{role} 缩放不合法；逐次减半，到宽 ≤640 或高 ≤480 为止")


def resize(image: np.ndarray, size: tuple[int, int], *, depth: bool = False) -> np.ndarray:
    if (image.shape[1], image.shape[0]) == size:
        return image
    scaled = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if depth else cv2.INTER_AREA)
    if image.ndim == 3 and scaled.ndim == 2:
        scaled = scaled[..., None]
    return scaled


def _enum(fd: int, number: int, length: int, *values: int) -> Iterator[bytes]:
    """Linux videodev2.h 的无指针 __u32 结构；只发 ENUM ioctl。"""
    index = 0
    while True:
        data = bytearray(length)
        struct.pack_into("=" + "I" * (len(values) + 1), data, 0, index, *values)
        try:
            fcntl.ioctl(fd, (3 << 30) | (length << 16) | (ord("V") << 8) | number, data)
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                return
            raise
        yield bytes(data)
        index += 1


def v4l2_profiles(path: str) -> dict[str, Any]:
    """枚举离散尺寸/帧率；连续尺寸给出范围及按步长对齐的常用档位。"""
    profiles, ranges = [], []
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        for fmt in _enum(fd, 2, 64, 1):  # VIDEO_CAPTURE
            pixel = struct.unpack_from("=I", fmt, 44)[0]
            fourcc = fmt[44:48].decode("ascii", errors="replace")
            sizes = set()
            for size in _enum(fd, 74, 44, pixel):
                kind = struct.unpack_from("=I", size, 8)[0]
                if kind == 1:
                    sizes.add(struct.unpack_from("=II", size, 12))
                elif kind in (2, 3):
                    low_w, high_w, step_w, low_h, high_h, step_h = struct.unpack_from("=6I", size, 12)
                    ranges.append(f"{fourcc}: {low_w}×{low_h}～{high_w}×{high_h}，步长 {step_w}×{step_h}")
                    for factor in scales(high_w, high_h):
                        w = low_w + max(0, high_w // factor - low_w) // max(1, step_w) * max(1, step_w)
                        h = low_h + max(0, high_h // factor - low_h) // max(1, step_h) * max(1, step_h)
                        sizes.add((w, h))
                    sizes.add((low_w, low_h))
                    break
            for width, height in sorted(sizes):
                rates, interval_range = [], None
                for interval in _enum(fd, 75, 52, pixel, width, height):
                    kind = struct.unpack_from("=I", interval, 16)[0]
                    if kind == 1:
                        num, den = struct.unpack_from("=II", interval, 20)
                        if num and den:
                            rates.append(den / num)
                    elif kind in (2, 3):
                        nums = struct.unpack_from("=6I", interval, 20)
                        interval_range = [nums[i] / nums[i + 1] if nums[i + 1] else 0 for i in (0, 2, 4)]
                        break
                profiles.append(
                    {
                        "width": width,
                        "height": height,
                        "format": fourcc,
                        "fps": sorted(set(rates)),
                        "interval_range": interval_range,
                    }
                )
    finally:
        os.close(fd)
    if not profiles:
        raise RuntimeError("设备未返回可枚举的分辨率；保留已保存尺寸，开始采集时严格校验")
    return {"profiles": profiles, "ranges": ranges}


def supports_fps(profile: dict, fps: int) -> bool:
    if profile.get("fps"):
        return any(math.isclose(rate, fps, rel_tol=1e-3) for rate in profile["fps"])
    limits = profile.get("interval_range")
    if limits:
        low, high, step = limits
        interval = 1 / fps
        return low <= interval <= high and (
            not step or math.isclose((interval - low) / step, round((interval - low) / step), abs_tol=1e-5)
        )
    return True  # 驱动未报告帧率时，在 OpenCV 启动阶段验证实际值。


def select_fourcc(profiles: list[dict], setting: dict[str, int], fps: int) -> str:
    matches = [
        p
        for p in profiles
        if (p["width"], p["height"]) == (setting["width"], setting["height"]) and supports_fps(p, fps)
    ]
    if not matches:
        raise ValueError(
            f"相机不支持 {setting['width']}×{setting['height']} @ {fps} FPS，请重新选择分辨率或帧率"
        )
    # 高分辨率双路优先压缩传输，避免 USB 带宽被未压缩 YUYV 占满。
    matches.sort(key=lambda p: (p["format"] not in ("MJPG", "JPEG"), p["format"]))
    return matches[0]["format"]
