#!/usr/bin/env python
"""Nero 双臂从臂（被控臂）配置：left=can1（主左）+ right=can0（主右）。"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class NeroDualFollowerConfig:
    """Nero 双臂硬件配置（16 维：left 7 关节+夹爪 + right 7 关节+夹爪）。"""

    # 左臂 CAN 通道（主左），对应 pick01 数据里的 left 8 维
    left_channel: str = "can1"

    # 右臂 CAN 通道（主右），对应 pick01 数据里的 right 8 维
    right_channel: str = "can0"

    # 固件版本字符串，映射到 pyAgxArm 的 NeroFW 枚举
    firmware: str = "1.21"

    # 夹爪最大行程（0.07 或 0.1，单位 m）
    max_range_config: float = 0.07

    # 首次跟随的缓启动时长（秒），避免从静止直接跳目标导致抖动
    ramp_seconds: float = 3.0

    # 断开时是否失能关节（失能会让高位关节直接坠落）。
    # 默认 False：断开只关 CAN、保持当前姿态不掉臂；要掉电时用 nero_estop.py。
    disable_torque_on_disconnect: bool = False

    # 相机手动对焦（相机名 -> 对焦值）。DE475C08(2K) 自动对焦会周期性停顿，需固定。
    camera_focus: dict[str, int] = field(default_factory=lambda: {"head": 35})

    # 相机（head/left/right）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("nero_dual_follower")
@dataclass
class NeroDualFollowerRobotConfig(RobotConfig, NeroDualFollowerConfig):
    pass
