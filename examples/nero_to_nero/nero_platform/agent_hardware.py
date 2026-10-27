"""NERO MCP adapters; construction and imports do not connect to hardware."""

from __future__ import annotations

import fcntl
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots.nero_dual_eef.nero_dual_eef import joints_to_eef10, solve_ik

from . import hardware_status
from .agent_control import JOINT_LIMITS, ArmState, finite_vector
from .runtime import Runtime


@contextmanager
def claim_channel(channel: str) -> Iterator[Any]:
    """Hold one service-instance lock for the lifetime of a CAN connection."""
    with Path(f"/tmp/lerobot-nero-mcp-{channel}.lock").open("a") as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"Another NERO MCP process owns {channel}") from None
        yield lockfile


class NeroKinematics:
    """Reuse LeRobot's NERO kinematics with stricter task-space tolerances."""

    def forward(self, joints: list[float], width_m: float) -> list[float]:
        return joints_to_eef10(joints, width_m)

    def inverse(self, target: list[float], seed: list[float]) -> list[float]:
        joints, ok, pos_error, rot_error = solve_ik(
            target,
            seed,
            limits=np.asarray(JOINT_LIMITS),
            max_joint_step_rad=np.deg2rad(5),
            pos_tol_mm=1.0,
            rot_tol_deg=3.0,
        )
        if not ok:
            raise ValueError(f"IK rejected: position error {pos_error:.2f} mm, rotation {rot_error:.2f} deg")
        return [float(value) for value in joints]


class LiveBackend:
    """Connect on first read, without reset, mode changes or torque commands."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.runtime = Runtime()
        self.lock = threading.RLock()
        self.handles: dict[str, Any] = {}
        self.channel_locks: dict[str, Any] = {}
        self.resources = ExitStack()
        self.seen: dict[tuple[str, str], tuple[float, float, bool]] = {}

    def _connect(self, side: str) -> tuple[Any, Any]:
        ch = self.config["channels"][side]
        info = self.runtime.channels.setdefault(ch, {})
        if side in self.handles:
            return self.handles[side]
        # Serializes this MCP implementation's processes. Other SDK applications
        # must still be stopped explicitly; SocketCAN itself is multi-client.
        lockfile = self.resources.enter_context(claim_channel(ch))
        arm = None
        try:
            arm = AgxArmFactory.create_arm(
                create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121, channel=ch)
            )
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            arm.connect()
            info.update(arm=arm, gripper=gripper, requested_role=None)
            self.handles[side] = (arm, gripper)
            self.channel_locks[side] = lockfile
            return arm, gripper
        except Exception:
            if arm is not None:
                arm.disconnect()
            lockfile.close()
            raise

    def _fresh(self, side: str, name: str, packet: Any) -> bool:
        now = time.monotonic()
        if packet is None or getattr(packet, "msg", None) is None:
            return False
        stamp = getattr(packet, "timestamp", None)
        try:
            stamp = finite_vector([stamp], 1)[0]
        except ValueError:
            return False
        if stamp <= 0:
            return False
        key = side, name
        previous = self.seen.get(key)
        if previous is None or stamp < previous[0]:
            self.seen[key] = (stamp, now, False)
        elif stamp > previous[0]:
            self.seen[key] = (stamp, now, True)
        current = self.seen[key]
        return current[2] and now - current[1] <= 0.5

    def read(self, side: str) -> ArmState:
        with self.lock:
            arm, gripper = self._connect(side)
            ch = self.config["channels"][side]
            deadline = time.monotonic() + 1.0
            while True:
                # The aggregate timestamp only covers the last packet; check all
                # four packets so missing joints can never masquerade as zeros.
                parser = getattr(arm, "_parser", None)
                fresh = [
                    self._fresh(side, suffix, getattr(parser, "joint_" + suffix, None))
                    for suffix in ("12", "34", "56", "7")
                ]
                grip = gripper.get_gripper_status()
                grip_fresh = self._fresh(side, "gripper", grip)
                status = hardware_status.read_status(self.runtime, ch)
                if all(fresh) and grip_fresh:
                    joints = finite_vector(list(arm.get_joint_angles().msg), 7)
                    width = finite_vector([grip.msg.value], 1)[0]
                    mode = status["mode"]
                    controlled = mode["code"] == 1 or (
                        mode["code"] == 6 and mode["requested_role"] == "follower"
                    )
                    ready = (
                        controlled
                        and status["enabled"]["state"] == "enabled"
                        and status["health"]["state"] == "normal"
                    )
                    return ArmState(joints, width, ready)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{side}/{ch}: incomplete or stale joint/gripper feedback")
                time.sleep(0.02)

    def prepare(self, side: str) -> None:
        with self.lock:
            arm, _ = self._connect(side)
            ch = self.config["channels"][side]
            status = hardware_status.inspect_status(self.runtime, ch)
            if status["health"]["code"] not in (0, 6) or status["health"]["error_code"] != 0:
                raise RuntimeError(f"Cannot prepare {ch}: {status['health']['label']}")
            arm.set_follower_mode()
            self.runtime.channels[ch]["requested_role"] = "follower"
            deadline = time.monotonic() + 3.0
            while not arm.enable():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{ch}: enabling did not complete")
                time.sleep(0.02)
            hardware_status.require_ready_for_motion(self.runtime, ch)

    def write_joints(self, side: str, joints: list[float]) -> None:
        with self.lock:
            self.handles[side][0].move_js(joints)

    def write_gripper(self, side: str, width_m: float) -> None:
        with self.lock:
            gripper = self.handles[side][1]
            packet = gripper.get_gripper_status()
            if not self._fresh(side, "gripper", packet):
                raise RuntimeError("Gripper feedback is stale")
            if getattr(getattr(packet.msg, "foc_status", None), "driver_enable_status", None) is not True:
                raise RuntimeError("Gripper is not enabled; prepare it in the existing hardware console")
            gripper.move_gripper_m(value=width_m, force=3.0)

    def close(self) -> None:
        with self.lock:
            try:
                for arm, _ in self.handles.values():
                    arm.disconnect()  # No disable, reset or automatic return-to-home.
            finally:
                self.handles.clear()
                for lockfile in self.channel_locks.values():
                    lockfile.close()
                self.channel_locks.clear()
                self.resources.close()


class Cameras:
    """Lazily open configured RGB cameras; mock frames are visibly labelled."""

    def __init__(self, config: dict[str, Any], *, mock: bool = False) -> None:
        self.config = config
        self.mock = mock
        self.cameras: dict[str, OpenCVCamera] = {}
        self.lock = threading.Lock()

    def jpeg(self, name: str) -> bytes:
        with self.lock:
            cfg = self.config["cameras"].get(name)
            if cfg is None:
                raise ValueError(f"Unknown camera {name}; available: {list(self.config['cameras'])}")
            if self.mock:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"MOCK - {name} - NO REAL CAMERA",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            else:
                if cfg.get("backend") != "opencv" or cfg.get("type") != "rgb":
                    raise ValueError("This first version supports configured OpenCV RGB cameras only")
                if name not in self.cameras:
                    resolution = self.config["camera_resolution"].get(name, {})
                    camera = OpenCVCamera(
                        OpenCVCameraConfig(
                            index_or_path=cfg["path"],
                            # Snapshot tools do not require a recording rate;
                            # accept the camera's negotiated rate at this size.
                            fps=None,
                            width=resolution.get("width", 640),
                            height=resolution.get("height", 480),
                        )
                    )
                    try:
                        camera.connect()
                    except Exception:
                        if camera.is_connected:
                            camera.disconnect()
                        raise
                    self.cameras[name] = camera
                rgb = self.cameras[name].read_latest(max_age_ms=500)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return encoded.tobytes()

    def close(self) -> None:
        with self.lock:
            for camera in self.cameras.values():
                camera.disconnect()
            self.cameras.clear()
