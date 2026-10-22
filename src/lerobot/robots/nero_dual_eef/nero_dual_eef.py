#!/usr/bin/env python
"""Nero 双臂 EEF 从臂 Robot：关节角+夹爪+相机 -> FK -> 20 维 EEF observation；
20 维 EEF action -> 带限位/限步长的数值 IK -> move_js + 夹爪跟随。

与 π0.5 EEF 训练约定严格对齐（Nero/preprocess_to_eef.py）：
  EEF20 = [左 xyz3, 左 rot6d6, 左 grip1, 右 xyz3, 右 rot6d6, 右 grip1]
  rot6d = 旋转矩阵前两列按列主序展平 [c0x,c0y,c0z,c1x,c1y,c1z]
  夹爪  = 物理开度(m) -> 20*grip-1 归一化到 [-1,1]（全开 0.1m -> +1）
  FK    = pyAgxArm 的 fk_from_mdh(get_mdh("nero"), q)（与训练/verify_fk 一致）

关键实现点：lerobot-rollout 会把状态/动作的标量特性按 `.pos`/`.vel` 后缀过滤
（rollout/context.py），所以 20 个 EEF 分量命名为 `left.eef_x.pos` 等（带 .pos 后缀、
按训练顺序排列），这样既通过过滤、又和 checkpoint 的 action_feature_names 顺序一致。
normalizer 是「按位置」逐元素变换，不看维度名，所以后缀不影响归一化正确性。
"""

import fcntl
import logging
import os
import struct
import time
from functools import cached_property
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
from pyAgxArm.utiles.tf import rpy_to_rot

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation

from ..robot import Robot
from .config_nero_dual_eef import NeroDualEefFollowerRobotConfig

logger = logging.getLogger(__name__)

_FW_MAP = {
    "1.11": NeroFW.V111,
    "1.12": NeroFW.V112,
    "1.20": NeroFW.V120,
    "1.21": NeroFW.V121,
}

_MDH = get_mdh("nero")

# SDK 预设关节限位（弧度），与 tests/fixtures/nero_kinematics.json 一致。
# 注意：J7=[-90°,90°] 是 SDK 软限位，机械实测约 ±99°；上真机前按控制器实际限位复核。
_JOINT_LIMITS = np.array(
    [
        [-2.705261, 2.705261],
        [-1.74533, 1.74533],
        [-2.757621, 2.757621],
        [-1.012291, 2.146755],
        [-2.757621, 2.757621],
        [-0.733039, 0.959932],
        [-1.570797, 1.570797],
    ]
)

# 20 维 EEF 分量名（与 π0.5 checkpoint 的 action_feature_names 严格一致）
_EEF_NAMES = (
    ["left.eef_x", "left.eef_y", "left.eef_z"]
    + [f"left.rot6d_col{c}_{ax}" for c in (0, 1) for ax in ("x", "y", "z")]
    + ["left.gripper"]
    + ["right.eef_x", "right.eef_y", "right.eef_z"]
    + [f"right.rot6d_col{c}_{ax}" for c in (0, 1) for ax in ("x", "y", "z")]
    + ["right.gripper"]
)
assert len(_EEF_NAMES) == 20

# rollout 过滤要求标量特性以 .pos/.vel 结尾，故加 .pos 后缀（顺序仍 = 训练顺序）
_EEF_POS_KEYS = [f"{name}.pos" for name in _EEF_NAMES]


def joints_to_eef10(q7, grip_m, full_open_m=0.1):
    """7 关节角(弧度) + 夹爪开度(m) -> 10 维 EEF [xyz3, rot6d6, grip_norm]."""
    x, y, z, roll, pitch, yaw = fk_from_mdh(_MDH, [float(v) for v in q7])
    rot = rpy_to_rot(roll, pitch, yaw)
    rot6d = [rot[0][0], rot[1][0], rot[2][0], rot[0][1], rot[1][1], rot[2][1]]
    grip = float(np.clip(2.0 * (grip_m / full_open_m) - 1.0, -1.0, 1.0))
    return [x, y, z, *rot6d, grip]


def rot6d_to_matrix(rot6d):
    """rot6d(前两列列主序展平) -> 3x3 旋转矩阵（Gram-Schmidt 正交化）。"""
    c0 = np.asarray(rot6d[:3], dtype=np.float64)
    c0 = c0 / np.linalg.norm(c0)
    c1 = np.asarray(rot6d[3:6], dtype=np.float64)
    c1 = c1 - c0 * np.dot(c0, c1)
    c1 = c1 / np.linalg.norm(c1)
    return np.column_stack((c0, c1, np.cross(c0, c1)))


def solve_ik(
    target10,
    seed7,
    limits=_JOINT_LIMITS,
    max_joint_step_rad=0.20,
    pos_tol_mm=50.0,
    rot_tol_deg=5.0,
):
    """10 维 EEF 目标 + 7 关节种子 -> 带限位/限步长的数值 IK。

    用 scipy.optimize.least_squares 反转 fk_from_mdh（与训练 FK 严格一致），
    结构对齐 nero_inference.kinematics.NeroKinematics.solve（位置 + 旋转*0.15 + 种子正则）。

    返回 (joints_rad, ok, pos_error_mm, rot_error_deg)。
    """
    target10 = np.asarray(target10, dtype=np.float64)
    seed7 = np.asarray(seed7, dtype=np.float64)
    if target10.shape != (10,) or seed7.shape != (7,) or not np.isfinite(target10).all():
        raise ValueError(f"IK 输入无效: target{target10.shape} seed{seed7.shape}")
    seed7 = np.clip(seed7, limits[:, 0], limits[:, 1])
    rot_target = rot6d_to_matrix(target10[3:9])

    lower = np.maximum(limits[:, 0], seed7 - max_joint_step_rad)
    upper = np.minimum(limits[:, 1], seed7 + max_joint_step_rad)

    def errors(q):
        x, y, z, roll, pitch, yaw = fk_from_mdh(_MDH, [float(v) for v in q])
        rot = rpy_to_rot(roll, pitch, yaw)
        pos = np.array([x, y, z]) - target10[:3]
        ori = Rotation.from_matrix(rot_target.T @ rot).as_rotvec()
        return pos, ori

    def objective(q):
        pos, ori = errors(q)
        return np.r_[pos, ori * 0.15, (q - seed7) * 0.0001]

    x0 = np.clip(seed7, lower + 1e-9, upper - 1e-9)
    fit = least_squares(
        objective, x0, bounds=(lower, upper), max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8
    )
    pos, ori = errors(fit.x)
    pos_mm = float(np.linalg.norm(pos) * 1000)
    rot_deg = float(np.degrees(np.linalg.norm(ori)))
    ok = bool(pos_mm <= pos_tol_mm and rot_deg <= rot_tol_deg)
    return fit.x, ok, pos_mm, rot_deg


# ---- V4L2 对焦控制（照搬采集平台 camera_focus.py 的 ioctl 写法）----
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


class _MockArm:
    """干跑用假臂：返回固定关节角，记录 move_js 命令，不碰 CAN。"""

    def __init__(self):
        self._q = [0.0] * 7
        self.commands = []

    def is_connected(self):
        return True

    def get_joint_angles(self):
        return SimpleNamespace(msg=list(self._q))

    def move_js(self, angles):
        self.commands.append(list(angles))
        self._q = list(angles)

    def set_follower_mode(self):
        pass

    def enable(self):
        return True

    def disable(self):
        return True

    def disconnect(self):
        pass


class _MockGripper:
    """干跑用假夹爪：返回固定开度，记录 move_gripper_m 命令。"""

    def __init__(self):
        self.width_m = 0.0
        self.commands = []

    def get_gripper_status(self):
        return SimpleNamespace(msg=SimpleNamespace(value=self.width_m))

    def move_gripper_m(self, value, force=1.0):
        self.commands.append(value)
        self.width_m = value

    def disable_gripper(self):
        pass


class _MockCamera:
    """干跑用假相机：返回合成灰帧，接口对齐 OpenCVCamera。"""

    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width
        self._connected = False
        self._frame = np.full((height, width, 3), 128, dtype=np.uint8)

    @property
    def is_connected(self):
        return self._connected

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def read_latest(self):
        return self._frame


class NeroDualEefFollower(Robot):
    config_class = NeroDualEefFollowerRobotConfig
    name = "nero_dual_eef"

    def __init__(self, config: NeroDualEefFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._left = None
        self._right = None
        self._left_gripper = None
        self._right_gripper = None
        if config.mock:
            self.cameras = {
                name: _MockCamera(name, cam.height, cam.width) for name, cam in config.cameras.items()
            }
        else:
            self.cameras = make_cameras_from_configs(config.cameras)
        self._ramp_from = {"left": None, "right": None}
        self._ramp_t0 = {"left": None, "right": None}

    @cached_property
    def observation_features(self) -> dict:
        ft = {k: float for k in _EEF_POS_KEYS}
        for name, cam in self.cameras.items():
            ft[name] = (cam.height, cam.width, 3)
        return ft

    @cached_property
    def action_features(self) -> dict:
        return {k: float for k in _EEF_POS_KEYS}

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
        if self.config.mock:
            return _MockArm(), _MockGripper()
        fw = _FW_MAP.get(self.config.firmware, NeroFW.V121)
        arm = AgxArmFactory.create_arm(
            create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=fw, channel=channel)
        )
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
        # 左臂（从左=从臂）
        self._left, self._left_gripper = self._make_arm(self.config.left_channel)
        self._reset_arm(self._left, self._left_gripper)
        self._enable_follower(self._left)

        # 右臂（从右=从臂）
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

    def _read_arm(self, arm, gripper) -> list:
        ja = arm.get_joint_angles()
        g = gripper.get_gripper_status()
        angles = ja.msg if (ja is not None and ja.msg is not None) else [0.0] * 7
        grip_m = float(g.msg.value) if (g is not None and g.msg is not None) else 0.0
        return joints_to_eef10(angles, grip_m, self.config.gripper_full_open_m)

    def get_observation(self) -> RobotObservation:
        eef20 = self._read_arm(self._left, self._left_gripper) + self._read_arm(
            self._right, self._right_gripper
        )
        obs = {key: float(eef20[i]) for i, key in enumerate(_EEF_POS_KEYS)}
        for name, cam in self.cameras.items():
            # max_age 放宽到 1000ms：DE475C08 偶发 ~570ms 停顿，lerobot 默认 500ms 会误报超时
            obs[name] = cam.read_latest(max_age_ms=1000)
        return obs

    def _send_arm(self, arm, gripper, prefix: str, target10) -> None:
        ja = arm.get_joint_angles()
        seed = np.array(ja.msg, dtype=np.float64) if (ja is not None and ja.msg is not None) else np.zeros(7)
        q, ok, pos_mm, rot_deg = solve_ik(
            target10,
            seed,
            _JOINT_LIMITS,
            self.config.max_joint_step_rad,
            self.config.position_tolerance_mm,
            self.config.rotation_tolerance_deg,
        )
        if not ok:
            logger.warning(
                "[%s] IK 未收敛 (pos %.1f mm / rot %.1f°) -> 本 tick 保持不动", prefix, pos_mm, rot_deg
            )
            return
        if self._ramp_from[prefix] is None:
            self._ramp_from[prefix] = seed.tolist()
            self._ramp_t0[prefix] = time.monotonic()
        alpha = min(1.0, (time.monotonic() - self._ramp_t0[prefix]) / self.config.ramp_seconds)
        cmd = [self._ramp_from[prefix][i] + alpha * (q[i] - self._ramp_from[prefix][i]) for i in range(7)]
        arm.move_js(cmd)
        width = float((np.clip(target10[9], -1.0, 1.0) + 1.0) * (self.config.gripper_full_open_m / 2.0))
        gripper.move_gripper_m(value=width, force=1.0)

    def send_action(self, action: RobotAction) -> RobotAction:
        eef20 = [float(action[k]) for k in _EEF_POS_KEYS]
        self._send_arm(self._left, self._left_gripper, "left", eef20[0:10])
        self._send_arm(self._right, self._right_gripper, "right", eef20[10:20])
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
                # 失能会让高位关节直接坠落；默认不关扭矩，保持当前姿态不掉臂
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
NeroDualEefFollowerRobot = NeroDualEefFollower
