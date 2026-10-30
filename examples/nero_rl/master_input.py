"""Right master input for the same SAC action path as the policy.

Cartesian mode uses relative TCP motion. Joint mode streams a bounded joint
target toward the master, with measured tracking checks in NeroEnv/hardware.
"""

import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from examples.nero_to_nero.nero_platform.joint_limits import AngleReader

from .config import TaskConfig, vector
from .hardware import NeroHardware


def request_state_feedback(arm: Any) -> None:
    """Restore V1.12+ ordinary CAN feedback without a motion target or enable.

    set_normal_mode is a no-op on these firmwares. Leader/follower switches
    disable CAN push; send the SDK's mode packet with MOVE=INVALID and push=ON.
    Preserve its cached fields so subsequent motion APIs retain their settings.
    """
    mode = arm._msg_mode
    previous = mode.enable_can_push, mode.move_mode
    try:
        mode.enable_can_push = mode.Enums.CanActiveMsgReporting.ENABLE
        mode.move_mode = 255
        arm._set_mode()
    finally:
        mode.enable_can_push, mode.move_mode = previous


@dataclass
class MasterSample:
    pose: np.ndarray
    width: float
    timestamp: float
    joints: np.ndarray | None = None


class RelativeMaster:
    """Convert consumed handle deltas to bounded actions without a target backlog."""

    def __init__(self, cfg: TaskConfig, translation_scale: float = 0.25):
        if not math.isfinite(translation_scale) or not 0 < translation_scale <= 1:
            raise ValueError("Master translation scale must be in (0, 1]")
        self.cfg, self.translation_scale = cfg, translation_scale
        self.previous: MasterSample | None = None

    def rebase(self, sample: MasterSample) -> None:
        pose = np.asarray(sample.pose, dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not math.isfinite(sample.width)
            or not 0 <= sample.width <= 0.105
            or not math.isfinite(sample.timestamp)
        ):
            raise ValueError("Invalid master feedback")
        self.previous = MasterSample(pose.copy(), sample.width, sample.timestamp)

    def action(self, sample: MasterSample) -> np.ndarray:
        previous = self.previous
        self.rebase(sample)
        if previous is None:
            return np.zeros(5, dtype=np.float32)
        dt = sample.timestamp - previous.timestamp
        if dt <= 0:
            raise RuntimeError("Master feedback did not advance")
        # A long pause, slow action or training update must not produce catch-up.
        if dt > 0.5:
            return np.zeros(5, dtype=np.float32)
        rotation = Rotation.from_matrix(sample.pose[:3, :3] @ previous.pose[:3, :3].T).as_rotvec()
        translation = (sample.pose[:3, 3] - previous.pose[:3, 3]) * self.translation_scale
        yaw, grip = rotation[2], sample.width - previous.width
        translation[np.abs(translation) < 0.00005] = 0
        yaw = 0.0 if abs(yaw) < math.radians(0.1) else yaw
        grip = 0.0 if abs(grip) < 0.0002 else grip
        # Consume the whole sample even if clipped/rejected downstream. No queue
        # can move the follower after the operator stops moving the handle.
        return np.clip(np.r_[translation, yaw, grip] / self.cfg.scales, -1, 1).astype(np.float32)


class WorkbenchGuard:
    """The workbench does not share our CAN locks: require its server to be down."""

    def __init__(self, url: str):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Workbench guard must use the local workbench URL")
        self.url = url.rstrip("/") + "/api/status"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check(self) -> None:
        try:
            with self.opener.open(self.url, timeout=0.3):
                pass
        except urllib.error.URLError as exc:
            # A timeout or malformed response does not prove the controller is off.
            if isinstance(exc.reason, ConnectionRefusedError):
                return
            raise RuntimeError("Cannot establish that the workbench is stopped") from exc
        raise RuntimeError("Stop the NERO workbench server before master RL control (CAN/camera conflict)")


class MasterInput:
    """Enable can3 teaching only on prepare(); can1 is owned by NeroEnv."""

    def __init__(self, cfg: TaskConfig, *, translation_scale: float = 0.25):
        self.backend = NeroHardware(replace(cfg, channel="can3", cameras={}))
        self.mapper = RelativeMaster(cfg, translation_scale)
        self.abort_requested: Callable[[], bool] = lambda: False
        self.changed_mode = False
        self.prepared = False
        self.active_sync_used = False
        self.control_faulted = False
        self.reader = None

    def open(self) -> None:
        self.backend.open()
        self.backend.runtime.channels["can3"]["label"] = "主右"
        self.reader = AngleReader(self.backend.arm)

    def prepare(self) -> None:
        if self.prepared:
            return
        from examples.nero_to_nero.nero_platform import devices

        backend = self.backend
        # Ordinary status is required before changing modes. Reuse the existing
        # workbench's verified enable / gripper-teaching preparation sequence.
        self.changed_mode = True
        devices.set_control_role(backend.runtime, "can3", "follower")
        status = backend.status_api.inspect_status(backend.runtime, "can3")
        if status["health"]["code"] not in (0, 6) or status["health"]["error_code"] != 0:
            raise RuntimeError(f"Right master is not ready: {status['health']}")
        deadline = time.monotonic() + 3
        while not backend.arm.enable():
            if self.abort_requested():
                raise KeyboardInterrupt("Operator stop")
            if time.monotonic() >= deadline:
                raise TimeoutError("Right master enable timed out")
            time.sleep(0.02)
        backend.status_api.require_ready_for_motion(backend.runtime, "can3")
        devices._prepare_leader_gripper(backend.runtime, "can3")
        devices.set_control_role(backend.runtime, "can3", "leader")
        # Drop pre-switch caches; require new teaching packets and gripper data.
        self.reader = AngleReader(backend.arm)
        backend.seen.clear()
        self.rebase()
        self.prepared = True

    def read(self) -> MasterSample:
        deadline = time.monotonic() + 1
        joints = None
        while time.monotonic() < deadline:
            if self.abort_requested():
                raise KeyboardInterrupt("Operator stop")
            now = time.monotonic()
            sample = self.reader.read(now)
            if sample is not None and sample[1] == "拖动反馈":
                joints = (vector(sample[0], 7), now)
            grip = self.backend.gripper.get_gripper_ctrl_states()
            if joints is not None and now - joints[1] <= 0.1 and self.backend._fresh("master_gripper", grip):
                width = float(grip.msg.value)
                result = MasterSample(self.backend.forward(joints[0]), width, now, joints[0].copy())
                # Validate without changing the engaged reference.
                RelativeMaster(self.backend.cfg).rebase(result)
                return result
            time.sleep(0.01)
        raise TimeoutError("Right master teaching joint/gripper feedback is stale; no new action sent")

    def rebase(self) -> None:
        self.mapper.rebase(self.read())

    def action(self) -> np.ndarray:
        return self.mapper.action(self.read())

    def joint_action(self, command_joints: np.ndarray, command_width: float) -> np.ndarray:
        sample = self.read()
        if sample.joints is None:
            raise RuntimeError("Missing master joint feedback")
        cfg = self.backend.cfg
        return np.clip(
            np.r_[sample.joints - command_joints, sample.width - command_width] / cfg.scales, -1, 1
        ).astype(np.float32)

    def close(self) -> None:
        try:
            if self.changed_mode and not self.control_faulted:
                from examples.nero_to_nero.nero_platform import devices

                devices.set_control_role(self.backend.runtime, "can3", "follower")
                if self.active_sync_used:
                    request_state_feedback(self.backend.arm)
                self.backend.status_api.require_ready_for_motion(self.backend.runtime, "can3")
                self.changed_mode = False
        finally:
            self.prepared = False
            self.backend.close()
