"""Lazy NERO adapter. Importing this module never connects or enables a motor."""

import fcntl
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import JOINT_LIMITS, STREAM_TRACKING_LIMIT_RAD, TaskConfig, vector


@dataclass
class Snapshot:
    joints: np.ndarray
    width: float
    pose: np.ndarray  # 4x4 TCP in this arm's base frame
    images: dict[str, np.ndarray]  # RGB uint8 HWC


class StreamTrackingError(RuntimeError):
    """A streaming target cannot advance until the follower catches up."""


class NeroHardware:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self.resources = ExitStack()
        self.arm = self.gripper = None
        self.cameras: dict[str, Any] = {}
        self.seen: dict[str, tuple[float, float, bool]] = {}
        self.prepared = False
        self.abort_requested = lambda: False
        self.last_servo_target: np.ndarray | None = None
        self.last_servo_width: float | None = None
        self.last_frames: dict[str, np.ndarray] = {}

    def open(self) -> None:
        # Hardware-only imports are intentionally delayed until explicitly selected.
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
        from pyAgxArm.utiles.tf import rpy_to_rot

        from examples.nero_to_nero.nero_platform import hardware_status
        from examples.nero_to_nero.nero_platform.runtime import Runtime
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.status_api = hardware_status
        self.runtime = Runtime()
        # Share the existing controller's lock name, so these two tools exclude each other.
        lock = self.resources.enter_context(
            Path(f"/tmp/lerobot-nero-mcp-{self.cfg.channel}.lock").open("a")  # noqa: SIM115
        )
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.arm = AgxArmFactory.create_arm(
            create_agx_arm_config(
                robot=ArmModel.NERO,
                firmeware_version=NeroFW.V121,
                channel=self.cfg.channel,
            )
        )
        self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.arm.connect()
        self.runtime.channels[self.cfg.channel] = {
            "arm": self.arm,
            "gripper": self.gripper,
            "requested_role": None,
        }
        self.mdh, self.fk, self.rpy_to_rot = get_mdh("nero"), fk_from_mdh, rpy_to_rot
        for name, path in self.cfg.cameras.items():
            camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=path, fps=None, width=640, height=480))
            self.cameras[name] = camera
            camera.connect()

    def forward(self, q: np.ndarray) -> np.ndarray:
        x, y, z, roll, pitch, yaw = self.fk(self.mdh, q.tolist())
        pose = np.eye(4)
        pose[:3, :3] = self.rpy_to_rot(roll, pitch, yaw)
        offset = np.zeros(3) if self.cfg.tcp_offset_m is None else vector(self.cfg.tcp_offset_m, 3)
        pose[:3, 3] = np.array([x, y, z]) + pose[:3, :3] @ offset
        return pose

    def inverse(self, pose: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        lo = np.maximum(JOINT_LIMITS[:, 0], seed - self.cfg.max_joint_step_rad)
        hi = np.minimum(JOINT_LIMITS[:, 1], seed + self.cfg.max_joint_step_rad)
        if np.any(lo >= hi):
            return None

        def residual(q: np.ndarray) -> np.ndarray:
            actual = self.forward(q)
            rotation = Rotation.from_matrix(pose[:3, :3].T @ actual[:3, :3]).as_rotvec()
            return np.r_[actual[:3, 3] - pose[:3, 3], rotation * 0.15, (q - seed) * 0.0001]

        result = least_squares(residual, np.clip(seed, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), max_nfev=80)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.00025 or np.linalg.norm(error[3:6]) / 0.15 > np.deg2rad(0.25):
            return None
        return result.x

    def _fresh(self, key: str, packet: Any) -> bool:
        stamp = float(getattr(packet, "timestamp", 0))
        if packet is None or packet.msg is None or not np.isfinite(stamp) or stamp <= 0:
            return False
        now = time.monotonic()
        previous = self.seen.get(key)
        if previous is None or stamp < previous[0]:
            self.seen[key] = (stamp, now, False)
        elif stamp > previous[0]:
            self.seen[key] = (stamp, now, True)
        return self.seen[key][2] and now - self.seen[key][1] < 0.5

    def read(self, *, images: bool = True) -> Snapshot:
        deadline = time.monotonic() + 1.0
        while True:
            parser = self.arm._parser
            fresh = [self._fresh(k, getattr(parser, f"joint_{k}", None)) for k in ("12", "34", "56", "7")]
            grip = self.gripper.get_gripper_status()
            fresh.append(self._fresh("gripper", grip))
            if all(fresh):
                q = vector(self.arm.get_joint_angles().msg, 7)
                width = float(grip.msg.value)
                if not np.isfinite(width):
                    raise RuntimeError("Non-finite gripper feedback")
                frames = {k: c.read_latest(max_age_ms=500) for k, c in self.cameras.items()} if images else {}
                if images:
                    self.last_frames = frames
                return Snapshot(q, width, self.forward(q), frames)
            if time.monotonic() >= deadline:
                raise TimeoutError("Incomplete/stale joint or gripper feedback; no new action sent")
            time.sleep(0.01)

    def prepare(self) -> None:
        status = self.status_api.inspect_status(self.runtime, self.cfg.channel)
        if status["health"]["code"] not in (0, 6) or status["health"]["error_code"] != 0:
            raise RuntimeError(f"Robot is not ready: {status['health']}")
        self.arm.set_follower_mode()
        self.runtime.channels[self.cfg.channel]["requested_role"] = "follower"
        deadline = time.monotonic() + 3
        while not self.arm.enable():
            if time.monotonic() > deadline:
                raise TimeoutError("Enable timed out; no movement sent")
            time.sleep(0.02)
        self.status_api.require_ready_for_motion(self.runtime, self.cfg.channel)
        # Never fall back to unknown/default speed if this call fails.
        self.arm.set_speed_percent(100 if self.cfg.action_mode == "joint_target" else 1)
        self.prepared = True
        self.last_servo_target = None
        self.last_servo_width = None

    def servo_command(self, q: np.ndarray, width: float) -> None:
        """Bounded streaming command for joint-delta mode; never waits for a trajectory."""
        if not self.prepared:
            raise RuntimeError("Hardware not prepared")
        self.status_api.require_ready_for_motion(self.runtime, self.cfg.channel)
        before = self.read(images=False)
        if self.abort_requested():
            raise KeyboardInterrupt("Operator stop")
        q = vector(q, 7)
        if not np.isfinite(width) or not 0 <= width <= self.cfg.gripper_max_m:
            raise ValueError("Invalid gripper command")
        if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
            raise RuntimeError("Joint target exceeds limits")
        reference = before.joints if self.last_servo_target is None else self.last_servo_target
        reference_width = before.width if self.last_servo_width is None else self.last_servo_width
        if np.max(np.abs(q - reference)) > self.cfg.max_joint_step_rad + 1e-6:
            raise RuntimeError("Streamed joint target advanced too far in one cycle")
        if abs(width - reference_width) > self.cfg.gripper_step_m + 0.0002:
            raise RuntimeError("Streamed gripper target advanced too far in one cycle")
        if (
            self.last_servo_target is not None
            and np.max(np.abs(self.last_servo_target - before.joints)) > STREAM_TRACKING_LIMIT_RAD
        ):
            raise StreamTrackingError("Follower failed to track streamed joint command")
        if np.max(np.abs(q - before.joints)) > STREAM_TRACKING_LIMIT_RAD:
            raise StreamTrackingError("Proposed streamed target exceeds actual tracking allowance")
        self.arm.move_js(q.tolist())
        self.last_servo_target = q.copy()
        self.last_servo_width = float(width)
        if abs(width - before.width) > 0.00005:
            self.gripper.move_gripper_m(value=float(width), force=self.cfg.gripper_force_n)

    def native_executed_target(self, action: list[float]) -> np.ndarray:
        """Log the existing SDK's clamp/quantization, without changing its command."""
        q = vector(action[:7], 7)
        if self.arm._joint_limits_enabled:
            limits = np.array(list(self.arm._config["joint_limits"].values()), dtype=float)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        q = np.deg2rad(np.round(np.rad2deg(q) * 1000) / 1000)
        return np.r_[q, round(action[7] * 1e6) / 1e6]

    def hold(self) -> Snapshot:
        """Cancel the pending joint target at measured feedback, retaining torque."""
        if self.prepared:
            self.status_api.require_ready_for_motion(self.runtime, self.cfg.channel)
        snap = self.read()
        if self.prepared:
            self.arm.move_js(snap.joints.tolist())
        self.last_servo_target = snap.joints.copy()
        # Keep the existing gripper target/force while holding the arm, including
        # a closed target stopped by a grasped object's width.
        if self.last_servo_width is None:
            self.last_servo_width = snap.width
        return snap

    def command(self, q: np.ndarray, width: float) -> None:
        if not self.prepared:
            raise RuntimeError("Hardware not prepared")
        self.status_api.require_ready_for_motion(self.runtime, self.cfg.channel)
        before = self.read(images=False)
        if self.abort_requested():
            raise KeyboardInterrupt("Operator stop")
        if np.max(np.abs(q - before.joints)) > self.cfg.max_joint_step_rad + 1e-6:
            raise RuntimeError("Robot moved since planning; joint step rejected")
        if abs(width - before.width) > self.cfg.gripper_step_m + 0.0002:
            raise RuntimeError("Gripper moved since planning; width step rejected")
        self.arm.move_j(q.tolist())
        self.gripper.move_gripper_m(value=float(width), force=self.cfg.gripper_force_n)
        deadline, settled = time.monotonic() + 3, 0
        while True:
            if self.abort_requested():
                raise KeyboardInterrupt("Operator stop")
            now = self.read(images=False)
            status = self.arm.get_arm_status()
            if not self._fresh("controller", status):
                idle = False
            else:
                if status.msg.arm_status != 0 or status.msg.err_code != 0:
                    raise RuntimeError("Controller fault during action")
                idle = status.msg.motion_status == 0
            reached = np.max(np.abs(now.joints - q)) <= self.cfg.tracking_tolerance_rad
            settled = settled + 1 if reached and idle else 0
            if settled >= 2:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Action did not settle; episode aborted")
            time.sleep(0.02)

    def stop(self) -> None:
        if self.prepared:
            # Firmware damping stop; not a brake or a torque-disable command.
            self.arm.electronic_emergency_stop()
            self.prepared = False

    def close(self) -> None:
        # No automatic home or torque-disable: either could disturb the scene/drop an arm.
        try:
            for camera in self.cameras.values():
                camera.disconnect()
        finally:
            try:
                if self.arm is not None:
                    self.arm.disconnect()
            finally:
                self.resources.close()
