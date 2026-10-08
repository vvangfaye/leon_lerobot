#!/usr/bin/env python
"""Nero 主臂（leader/示教器）配置。"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class NeroLeaderConfig:
    """Nero 主臂硬件配置。"""

    # CAN 通道，如 "can0"
    channel: str = "can0"

    # 固件版本字符串，映射到 pyAgxArm 的 NeroFW 枚举（1.21 -> V121）
    firmware: str = "1.21"

    # 示教摩擦（1~10，越大越硬，默认 8 避免夹爪抖动）
    teaching_friction: int = 8

    # 夹爪最大行程（0.07 或 0.1，单位 m）
    max_range_config: float = 0.07


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass
class NeroLeaderTeleopConfig(TeleoperatorConfig, NeroLeaderConfig):
    pass
