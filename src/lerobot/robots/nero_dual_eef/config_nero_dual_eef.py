#!/usr/bin/env python
"""Nero 双臂 EEF 从臂配置（20 维末端位姿）。"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class NeroDualEefFollowerConfig:
    """Nero 双臂 EEF 硬件配置（20 维：左右各 xyz3 + rot6d6 + 夹爪1）。"""

    # 左臂 CAN 通道（从左=从臂），对应数据里的 left 10 维
    # 4 臂角色（用户确认）：从左=can0，从右=can1，主左=can2，主右=can3
    # 部署时策略控制「从臂」：从左=can0（left）、从右=can1（right）。
    left_channel: str = "can0"

    # 右臂 CAN 通道（从右=从臂），对应数据里的 right 10 维
    right_channel: str = "can1"

    # 固件版本字符串，映射到 pyAgxArm 的 NeroFW 枚举
    firmware: str = "1.21"

    # 夹爪物理全开（米）。EEF 训练约定用 0.1m（preprocess_to_eef.py 的 GRIPPER_FULL_OPEN）
    gripper_full_open_m: float = 0.1

    # 首次跟随缓启动时长（秒），避免从静止直接跳目标导致抖动
    ramp_seconds: float = 3.0

    # IK 单步最大关节角变化（弧度），限制每 tick 的关节步长
    max_joint_step_rad: float = 0.20

    # IK 收敛容差（超过则 hold 不动，避免发危险命令）
    position_tolerance_mm: float = 50.0
    rotation_tolerance_deg: float = 10.0

    # 干跑模式：不连 CAN、用合成反馈（先别上真机时置 True）
    mock: bool = False

    # 断开时是否失能关节（失能会让高位关节直接坠落）。
    # 默认 False：断开只关 CAN、保持当前姿态不掉臂；要掉电时用 nero_estop.py。
    disable_torque_on_disconnect: bool = False

    # 相机（命名 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb，直接对齐 checkpoint）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 相机手动对焦（相机名 -> 对焦值）。DE475C08(2K) 自动对焦会周期性停顿，需固定。
    camera_focus: dict[str, int] = field(default_factory=lambda: {"base_0_rgb": 35})


@RobotConfig.register_subclass("nero_dual_eef")
@dataclass
class NeroDualEefFollowerRobotConfig(RobotConfig, NeroDualEefFollowerConfig):
    pass
