#!/usr/bin/env python
"""Nero 从臂 Robot：读关节角 + 夹爪 + 相机作为 observation，move_js + 夹爪跟随作为 action。"""

import time
from functools import cached_property

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation

from ..robot import Robot
from .config_nero_follower import NeroFollowerRobotConfig

_FW_MAP = {
    "1.11": NeroFW.V111,
    "1.12": NeroFW.V112,
    "1.20": NeroFW.V120,
    "1.21": NeroFW.V121,
}

_JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]


class NeroFollower(Robot):
    config_class = NeroFollowerRobotConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper = None
        self.cameras = make_cameras_from_configs(config.cameras)
        # 缓启动状态
        self._ramp_from = None
        self._ramp_t0 = None

    @cached_property
    def observation_features(self) -> dict:
        ft = {k: float for k in _JOINT_KEYS}
        for name, cam in self.cameras.items():
            ft[name] = (cam.height, cam.width, 3)
        return ft

    @cached_property
    def action_features(self) -> dict:
        return {k: float for k in _JOINT_KEYS}

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and self._arm.is_connected() and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        # Nero 固件内部自带零点标定
        return True

    def connect(self, calibrate: bool = True) -> None:
        fw = _FW_MAP.get(self.config.firmware, NeroFW.V121)
        self._arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=fw, channel=self.config.channel))
        self._gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.connect()

        # 启动复位
        self._reset()

        # 使能 + 进 follower 模式
        while not self._arm.enable():
            time.sleep(0.01)
        time.sleep(0.4)
        self._arm.set_follower_mode()
        time.sleep(0.2)

        # 相机
        for cam in self.cameras.values():
            cam.connect()

        self._ramp_from = None
        self._ramp_t0 = None

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

    def get_observation(self) -> RobotObservation:
        ja = self._arm.get_joint_angles()
        g = self._gripper.get_gripper_status()
        angles = ja.msg if (ja is not None and ja.msg is not None) else [0.0] * 7
        obs = {f"joint{i+1}.pos": float(angles[i]) for i in range(7)}
        obs["gripper.pos"] = float(g.msg.value) if (g is not None and g.msg is not None) else 0.0
        for name, cam in self.cameras.items():
            obs[name] = cam.read_latest()
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        joints = [float(action[f"joint{i+1}.pos"]) for i in range(7)]
        # 首次跟随缓启动：从从臂当前位置线性渐变到目标，避免抖动
        if self._ramp_from is None:
            ja = self._arm.get_joint_angles()
            self._ramp_from = list(ja.msg) if (ja is not None and ja.msg is not None) else joints
            self._ramp_t0 = time.monotonic()
        alpha = min(1.0, (time.monotonic() - self._ramp_t0) / self.config.ramp_seconds)
        target = [self._ramp_from[i] + alpha * (joints[i] - self._ramp_from[i]) for i in range(7)]
        self._arm.move_js(target)
        w = max(0.0, min(self.config.max_range_config, float(action.get("gripper.pos", 0.0))))
        self._gripper.move_gripper_m(value=w, force=1.0)
        return action

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
        for cam in self.cameras.values():
            cam.disconnect()
        self._arm = None
