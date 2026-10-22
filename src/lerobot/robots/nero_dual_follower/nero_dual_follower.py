#!/usr/bin/env python
"""Nero 双臂从臂 Robot：读左右臂关节角 + 夹爪 + 相机作为 observation，
move_js + 夹爪跟随作为 action（16 维，与 pick01 训练数据对齐）。"""

import fcntl
import logging
import os
import struct
import time
from functools import cached_property

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation

from ..robot import Robot
from .config_nero_dual_follower import NeroDualFollowerRobotConfig

_FW_MAP = {
    "1.11": NeroFW.V111,
    "1.12": NeroFW.V112,
    "1.20": NeroFW.V120,
    "1.21": NeroFW.V121,
}

# 与 pick01 数据集的 observation.state / action names 严格对齐（先 left 后 right）
_LEFT_JOINT_KEYS = [f"left.joint{i}.pos" for i in range(1, 8)] + ["left.gripper.pos"]
_RIGHT_JOINT_KEYS = [f"right.joint{i}.pos" for i in range(1, 8)] + ["right.gripper.pos"]
_JOINT_KEYS = _LEFT_JOINT_KEYS + _RIGHT_JOINT_KEYS

logger = logging.getLogger(__name__)

# ---- V4L2 对焦控制（照搬 nero_dual_eef 的 ioctl 写法）----
_V4L2_S_CTRL = 0xC008561C
_V4L2_CTRL_FOCUS_AUTO = 0x009A090C
_V4L2_CTRL_FOCUS_ABSOLUTE = 0x009A090A
_VALUE_FMT = "Ii"


def set_manual_focus(device_path, value):
    """把相机切到手动对焦并设到指定值。DE475C08(2K) 自动对焦会周期性停顿。"""
    fd = os.open(str(device_path), os.O_RDWR | os.O_NONBLOCK)
    try:
        try:
            fcntl.ioctl(fd, _V4L2_S_CTRL, struct.pack(_VALUE_FMT, _V4L2_CTRL_FOCUS_AUTO, 0))
        except OSError:
            pass  # 不支持 focus_auto 控制的相机忽略
        fcntl.ioctl(fd, _V4L2_S_CTRL, struct.pack(_VALUE_FMT, _V4L2_CTRL_FOCUS_ABSOLUTE, int(value)))
    finally:
        os.close(fd)


class NeroDualFollower(Robot):
    config_class = NeroDualFollowerRobotConfig
    name = "nero_dual_follower"

    def __init__(self, config: NeroDualFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._left = None
        self._right = None
        self._left_gripper = None
        self._right_gripper = None
        self.cameras = make_cameras_from_configs(config.cameras)
        # 缓启动状态（左右各自独立，避免右臂以左臂位置为起点）
        self._ramp_from = {"left": None, "right": None}
        self._ramp_t0 = {"left": None, "right": None}

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
        return (
            self._left is not None
            and self._right is not None
            and self._left.is_connected()
            and self._right.is_connected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # Nero 固件内部自带零点标定
        return True

    def _make_arm(self, channel: str):
        fw = _FW_MAP.get(self.config.firmware, NeroFW.V121)
        arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=fw, channel=channel))
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        arm.connect()
        return arm, gripper

    @staticmethod
    def _reset_arm(arm, gripper) -> None:
        try:
            arm.set_follower_mode()
            time.sleep(0.5)
        except Exception:
            pass
        if gripper is not None:
            try:
                gripper.disable_gripper()
                time.sleep(0.2)
            except Exception:
                pass
        try:
            for _ in range(60):
                if arm.disable():
                    time.sleep(0.3)
                    return
                time.sleep(0.02)
        except Exception:
            pass

    @staticmethod
    def _enable_follower(arm) -> None:
        while not arm.enable():
            time.sleep(0.01)
        time.sleep(0.4)
        arm.set_follower_mode()
        time.sleep(0.2)

    def connect(self, calibrate: bool = True) -> None:
        # 左臂（can1 主左）
        self._left, self._left_gripper = self._make_arm(self.config.left_channel)
        self._reset_arm(self._left, self._left_gripper)
        self._enable_follower(self._left)

        # 右臂（can0 主右）
        self._right, self._right_gripper = self._make_arm(self.config.right_channel)
        self._reset_arm(self._right, self._right_gripper)
        self._enable_follower(self._right)

        for cam in self.cameras.values():
            cam.connect()
        self._apply_camera_focus()

        self._ramp_from = {"left": None, "right": None}
        self._ramp_t0 = {"left": None, "right": None}

    def _apply_camera_focus(self) -> None:
        """连接相机后固定对焦（V4L2 ioctl），避免自动对焦周期性停顿。"""
        for name, focus_value in self.config.camera_focus.items():
            cam_cfg = self.config.cameras.get(name)
            if cam_cfg is None:
                continue
            path = getattr(cam_cfg, "index_or_path", None)
            if not path or not os.path.exists(str(path)):
                logger.warning("[%s] 找不到相机设备 %s，跳过固定对焦", name, path)
                continue
            try:
                set_manual_focus(path, focus_value)
                logger.info("[%s] 已固定对焦 value=%d (%s)", name, focus_value, path)
            except Exception as exc:
                logger.warning("[%s] 固定对焦失败: %s", name, exc)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_arm(self, arm, gripper, prefix: str) -> dict:
        ja = arm.get_joint_angles()
        g = gripper.get_gripper_status()
        angles = ja.msg if (ja is not None and ja.msg is not None) else [0.0] * 7
        obs = {f"{prefix}.joint{i+1}.pos": float(angles[i]) for i in range(7)}
        obs[f"{prefix}.gripper.pos"] = float(g.msg.value) if (g is not None and g.msg is not None) else 0.0
        return obs

    def get_observation(self) -> RobotObservation:
        obs = self._read_arm(self._left, self._left_gripper, "left")
        obs.update(self._read_arm(self._right, self._right_gripper, "right"))
        for name, cam in self.cameras.items():
            # max_age 放宽到 1000ms：DE475C08 偶发 ~570ms 停顿，lerobot 默认 500ms 会误报超时
            obs[name] = cam.read_latest(max_age_ms=1000)
        return obs

    def _send_arm(self, arm, gripper, prefix: str, action: RobotAction) -> None:
        joints = [float(action[f"{prefix}.joint{i+1}.pos"]) for i in range(7)]
        # 首次跟随缓启动：每条臂从各自当前关节位置线性渐变到目标
        if self._ramp_from[prefix] is None:
            ja = arm.get_joint_angles()
            self._ramp_from[prefix] = list(ja.msg) if (ja is not None and ja.msg is not None) else joints
            self._ramp_t0[prefix] = time.monotonic()
        alpha = min(1.0, (time.monotonic() - self._ramp_t0[prefix]) / self.config.ramp_seconds)
        target = [self._ramp_from[prefix][i] + alpha * (joints[i] - self._ramp_from[prefix][i]) for i in range(7)]
        arm.move_js(target)
        w = max(0.0, min(self.config.max_range_config, float(action.get(f"{prefix}.gripper.pos", 0.0))))
        gripper.move_gripper_m(value=w, force=1.0)

    def send_action(self, action: RobotAction) -> RobotAction:
        self._send_arm(self._left, self._left_gripper, "left", action)
        self._send_arm(self._right, self._right_gripper, "right", action)
        return action

    def disconnect(self) -> None:
        for arm, gripper in ((self._left, self._left_gripper), (self._right, self._right_gripper)):
            if arm is None:
                continue
            try:
                arm.set_follower_mode()
                time.sleep(0.5)
            except Exception:
                pass
            if gripper is not None:
                try:
                    gripper.disable_gripper()
                except Exception:
                    pass
            if self.config.disable_torque_on_disconnect:
                try:
                    for _ in range(50):
                        if arm.disable():
                            break
                        time.sleep(0.02)
                except Exception:
                    pass
            try:
                arm.disconnect()
            except Exception:
                pass
        for cam in self.cameras.values():
            cam.disconnect()
        self._left = None
        self._right = None
        self._ramp_from = {"left": None, "right": None}
        self._ramp_t0 = {"left": None, "right": None}


# 约定式查找（make_device_from_device_class）要求 device 类名 = config 去 "Config"
NeroDualFollowerRobot = NeroDualFollower
