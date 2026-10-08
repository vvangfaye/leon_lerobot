#!/usr/bin/env python
"""Nero 从臂（follower/被控臂）配置。"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class NeroFollowerConfig:
    """Nero 从臂硬件配置。"""

    # CAN 通道，如 "can1"
    channel: str = "can1"

    # 固件版本字符串，映射到 pyAgxArm 的 NeroFW 枚举
    firmware: str = "1.21"

    # 夹爪最大行程（0.07 或 0.1，单位 m）
    max_range_config: float = 0.07

    # 首次跟随的缓启动时长（秒），避免从臂从静止直接跳到主臂位置导致抖动
    ramp_seconds: float = 3.0

    # 相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("nero_follower")
@dataclass
class NeroFollowerRobotConfig(RobotConfig, NeroFollowerConfig):
    pass
