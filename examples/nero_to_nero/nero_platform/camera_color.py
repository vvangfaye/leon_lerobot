"""采集与推理共用的 UVC 白平衡初始化；仅在连接时写入并读回。"""

import errno
import os
import struct
from fcntl import ioctl
from pathlib import Path

AUTO_WHITE_BALANCE = 0x0098090C
QUERY_CTRL = 0xC0445624
GET_CTRL = 0xC008561B
SET_CTRL = 0xC008561C


def configure_white_balance(path: str | Path) -> dict:
    """支持时启用自动白平衡，不套用其他相机的色温标尺。"""
    result = {"state": "warning", "automatic": None, "message": ""}
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
        try:
            try:
                raw = ioctl(fd, QUERY_CTRL, struct.pack(
                    "II32siiiiI2I", AUTO_WHITE_BALANCE, 0, b"", 0, 0, 0, 0, 0, 0, 0,
                ))
            except OSError as exc:
                if exc.errno not in (errno.EINVAL, errno.ENOTTY):
                    raise
                return {**result, "state": "unsupported", "message": "设备未提供自动白平衡控制"}
            flags = struct.unpack("II32siiiiI2I", raw)[7]
            if flags & 0x0001:  # V4L2_CTRL_FLAG_DISABLED
                return {**result, "state": "unsupported", "message": "设备未提供自动白平衡控制"}

            def read() -> int:
                return struct.unpack(
                    "Ii", ioctl(fd, GET_CTRL, struct.pack("Ii", AUTO_WHITE_BALANCE, 0)),
                )[1]

            result["automatic"] = read()
            if result["automatic"] != 1:
                if flags & 0x0004:  # V4L2_CTRL_FLAG_READ_ONLY
                    raise ValueError("自动白平衡控制为只读")
                ioctl(fd, SET_CTRL, struct.pack("Ii", AUTO_WHITE_BALANCE, 1))
                result["automatic"] = read()
            if result["automatic"] != 1:
                raise ValueError("启用后读回仍为关闭")
            result.update(state="verified", message="自动白平衡已开启")
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        result["message"] = f"白平衡未确认：{exc}"
    return result
