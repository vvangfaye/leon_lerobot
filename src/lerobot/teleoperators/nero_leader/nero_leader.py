#!/usr/bin/env python
"""Nero 主臂 Teleoperator：读主臂 7 关节角 + 夹爪示教输入作为 action。"""

import time
from typing import Any

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from lerobot.lerobot_types import RobotAction

from ..teleoperator import Teleoperator
from .config_nero_leader import NeroLeaderTeleopConfig

_FW_MAP = {
    "1.11": NeroFW.V111,
    "1.12": NeroFW.V112,
    "1.20": NeroFW.V120,
    "1.21": NeroFW.V121,
}

_JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]


class NeroLeader(Teleoperator):
    config_class = NeroLeaderTeleopConfig
    name = "nero_leader"

    def __init__(self, config: NeroLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper = None

    @property
    def action_features(self) -> dict:
        return {k: float for k in _JOINT_KEYS}

    @property
    def feedback_features(self) -> dict:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and self._arm.is_connected()

    @property
    def is_calibrated(self) -> bool:
        # Nero 固件内部自带零点标定，无需逐电机标定
        return True

    def connect(self, calibrate: bool = True) -> None:
        fw = _FW_MAP.get(self.config.firmware, NeroFW.V121)
        self._arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=fw, channel=self.config.channel))
        self._gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.connect()

        # 启动复位：退 leader + 关夹爪 + 失能（每步延时，避免固件卡死）
        self._reset()

        # 使能 -> 配示教参数 -> 进 leader 模式（延时让固件走完每一步）
        while not self._arm.enable():
            time.sleep(0.01)
        time.sleep(0.4)
        self._gripper.set_gripper_teaching_pendant_param(
            teaching_friction=self.config.teaching_friction,
            max_range_config=self.config.max_range_config,
        )
        time.sleep(0.2)
        self._arm.set_leader_mode()
        time.sleep(0.2)

    def _reset(self) -> None:
        try:
            self._arm.set_follower_mode()
            time.sleep(0.5)
        except Exception:
            pass
        if self._gripper is not None:
            try:
                self._gripper.disable_gripper()
                time.sleep(0.2)
            except Exception:
                pass
        try:
            for _ in range(60):
                if self._arm.disable():
                    time.sleep(0.3)
                    return
                time.sleep(0.02)
        except Exception:
            pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        ja = self._arm.get_leader_joint_angles()
        g = self._gripper.get_gripper_ctrl_states()
        angles = ja.msg if (ja is not None and ja.msg is not None) else [0.0] * 7
        action = {f"joint{i+1}.pos": float(angles[i]) for i in range(7)}
        action["gripper.pos"] = float(g.msg.value) if (g is not None and g.msg is not None) else 0.0
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._arm is None:
            return
        try:
            self._arm.set_follower_mode()
            time.sleep(0.5)
        except Exception:
            pass
        if self._gripper is not None:
            try:
                self._gripper.disable_gripper()
            except Exception:
                pass
        try:
            for _ in range(50):
                if self._arm.disable():
                    break
                time.sleep(0.02)
        except Exception:
            pass
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._arm = None
