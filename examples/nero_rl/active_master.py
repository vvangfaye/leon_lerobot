"""Bounded motorized right-master mirroring of measured right-follower motion."""

import time
from collections.abc import Callable

import numpy as np

from examples.nero_to_nero.nero_platform import devices

from .config import JOINT_LIMITS, TaskConfig, vector
from .hardware import Snapshot
from .master_input import MasterInput, request_state_feedback


class MasterSyncError(RuntimeError):
    """Both arms must stop before retrying synchronization."""


class ActiveMasterSync:
    """One controller owns can3, in either teaching mode or bounded motor mode.

    Initial alignment is explicit. Policy execution accepts only a small initial
    mismatch. Commands always track measured follower joints, never actor targets.
    """

    INITIAL_GAP_RAD = np.deg2rad(2)
    MAX_PAIR_GAP_RAD = np.deg2rad(5)
    MAX_LEAD_RAD = np.deg2rad(1.5)
    HARD_TRACKING_RAD = np.deg2rad(3)
    MAX_INITIAL_RECOVERY_RAD = np.deg2rad(10)
    GRIPPER_SPEED_M_S = 0.02
    GRIPPER_FORCE_N = 1.0

    def __init__(self, master: MasterInput, cfg: TaskConfig):
        if master.backend.cfg.channel != "can3" or cfg.channel != "can1":
            raise ValueError("Active mirroring is restricted to right master can3 / follower can1")
        self.master, self.backend, self.cfg = master, master.backend, cfg
        self.cancel_requested: Callable[[], bool] = lambda: False
        self.running = False
        self.aligning = False
        self.drive_ready = False
        self.faulted = False
        self.motion_targets_sent = False
        self.phase = "not_prepared"
        self.reference: np.ndarray | None = None
        self.width = 0.0
        self.last_tick = 0.0
        self.alignment_deadline = 0.0
        self.fixed_target: Snapshot | None = None
        self.master_allowed_lo = JOINT_LIMITS[:, 0].copy()
        self.master_allowed_hi = JOINT_LIMITS[:, 1].copy()
        self.blocked_since: float | None = None
        self.progress_anchor: np.ndarray | None = None
        self.diagnostics: dict = {}

    @staticmethod
    def check_joints(joints: np.ndarray) -> np.ndarray:
        q = vector(joints, 7)
        if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
            raise MasterSyncError("主臂同步姿态超出关节范围；两臂保持，可使用相对接管。")
        return q

    def _cancel_check(self) -> None:
        if self.cancel_requested():
            raise MasterSyncError("主臂同步已取消。")

    def _current_master(self) -> Snapshot:
        if self.master.prepared and not self.drive_ready:
            sample = self.master.read()
            q = vector(sample.joints, 7)
            return Snapshot(q, sample.width, self.backend.forward(q), {})
        return self.backend.read(images=False)

    def begin(self, follower: Snapshot, *, align: bool) -> None:
        if self.faulted:
            raise MasterSyncError("主臂通信或控制异常尚未恢复，请先恢复连接；不会自动重试运动。")
        self._cancel_check()
        target = self.check_joints(follower.joints)
        if not np.isfinite(follower.width) or not 0 <= follower.width <= 0.09:
            raise MasterSyncError("从臂夹爪反馈超出主臂同步范围。")
        current = self._current_master()
        current_q = vector(current.joints, 7)
        outside = np.maximum(JOINT_LIMITS[:, 0] - current_q, current_q - JOINT_LIMITS[:, 1])
        if np.any(outside > 0) and (not align or np.max(outside) > self.MAX_INITIAL_RECOVERY_RAD):
            raise MasterSyncError("主臂当前姿态超出同步范围；只能先进行受限的向内同步恢复。")
        self.master_allowed_lo = np.minimum(JOINT_LIMITS[:, 0], current_q)
        self.master_allowed_hi = np.maximum(JOINT_LIMITS[:, 1], current_q)
        if not np.isfinite(current.width) or not 0 <= current.width <= 0.09:
            raise MasterSyncError("主臂夹爪反馈无效。")
        gap = float(np.max(np.abs(target - current.joints)))
        if not align and gap > self.INITIAL_GAP_RAD:
            raise MasterSyncError(
                f"主从最大相差 {np.rad2deg(gap):.1f}°；请先点“同步主臂到从臂”，从臂保持不动。"
            )
        if not self.drive_ready:
            # Switch to ordinary CAN control and immediately latch the measured
            # pose. No old master target or follower target is replayed here.
            self.master.prepared = False
            self.master.changed_mode = True
            self.master.active_sync_used = True
            self._cancel_check()
            self.drive_ready = True
            self.backend.prepared = True
            self.phase = "preparing"
            devices.set_control_role(self.backend.runtime, "can3", "follower")
            self.backend.arm.move_js(current.joints.tolist())
            request_state_feedback(self.backend.arm)
            deadline = time.monotonic() + 3
            while not self.backend.arm.enable():
                self._cancel_check()
                if time.monotonic() >= deadline:
                    raise MasterSyncError("主臂主动同步使能超时。")
                time.sleep(0.01)
            # Mark ownership before feedback verification, so failures still
            # stop this motorized arm through the common hold path.
            self.backend.seen.clear()
            self.backend.status_api.require_ready_for_motion(self.backend.runtime, "can3")
            self.backend.arm.set_speed_percent(100)
            current = self.backend.read(images=False)
            if np.any(current.joints < self.master_allowed_lo - 1e-5) or np.any(
                current.joints > self.master_allowed_hi + 1e-5
            ):
                raise MasterSyncError("模式切换期间主臂偏出初始位置范围，已停止同步。")
            self.backend.arm.move_js(current.joints.tolist())
            if not np.isfinite(current.width) or not 0 <= current.width <= 0.09:
                raise MasterSyncError("主臂夹爪反馈无效。")
            self.backend.gripper.move_gripper_m(value=current.width, force=self.GRIPPER_FORCE_N)
            self.backend.last_servo_target = current.joints.copy()
            self.backend.last_servo_width = current.width
        self.reference = current.joints.copy()
        self.width = float(current.width)
        self.fixed_target = (
            Snapshot(target.copy(), follower.width, follower.pose.copy(), {}) if align else None
        )
        self.running, self.aligning = True, align
        self.phase = "aligning" if align else "tracking"
        self.last_tick = time.monotonic()
        gap = float(np.max(np.abs(target - current.joints)))
        if not align and gap > self.INITIAL_GAP_RAD:
            raise MasterSyncError("切换模式时主臂位置发生变化，请先重新同步主臂。")
        self.alignment_deadline = self.last_tick + max(
            8,
            gap / np.deg2rad(min(5.0, self.cfg.policy_joint_speed_deg_s)) + 5,
            abs(current.width - follower.width) / self.GRIPPER_SPEED_M_S + 5,
        )
        self.blocked_since, self.progress_anchor = None, current.joints.copy()
        self.diagnostics = {"phase": self.phase, "pair_gap_deg": float(np.rad2deg(gap))}

    def update(self, follower: Snapshot) -> bool:
        """Advance one bounded master command; return initial-alignment completion."""
        if not self.running or self.reference is None:
            raise MasterSyncError("主臂同步尚未准备。")
        self._cancel_check()
        desired = self.fixed_target if self.aligning else follower
        if self.aligning and np.max(np.abs(follower.joints - desired.joints)) > np.deg2rad(0.5):
            raise MasterSyncError("同步期间从臂位置改变，已暂停。")
        target = self.check_joints(desired.joints)
        current = self.backend.read(images=False)
        self.backend.status_api.require_ready_for_motion(self.backend.runtime, "can3")
        current_q = vector(current.joints, 7)
        if np.any(current_q < self.master_allowed_lo - 1e-5) or np.any(
            current_q > self.master_allowed_hi + 1e-5
        ):
            raise MasterSyncError("主臂偏出允许返回的范围，已停止同步。")
        error = float(np.max(np.abs(self.reference - current.joints)))
        pair_gap = float(np.max(np.abs(target - current.joints)))
        if error > self.HARD_TRACKING_RAD:
            raise MasterSyncError("主臂未跟上同步指令，两臂暂停。")
        if not self.aligning and pair_gap > self.MAX_PAIR_GAP_RAD:
            raise MasterSyncError("主从同步误差超过 5°，两臂暂停；可切相对接管。")
        now = time.monotonic()
        if self.aligning and now > self.alignment_deadline:
            raise MasterSyncError("主臂同步超时，已暂停。")
        dt = max(0.0, min(now - self.last_tick, 1 / self.cfg.hz))
        advance = target - self.reference
        step = np.deg2rad(min(5.0, self.cfg.policy_joint_speed_deg_s)) * dt
        fraction = min(1.0, step / max(float(np.max(np.abs(advance))), 1e-12))
        advance = advance * fraction
        allowance = self.MAX_LEAD_RAD - np.sign(advance) * (self.reference - current.joints)
        ratios = np.divide(allowance, np.abs(advance), out=np.full(7, np.inf), where=np.abs(advance) > 1e-12)
        tracking_fraction = float(np.clip(ratios.min(), 0, 1))
        if tracking_fraction < 1:
            if self.blocked_since is None or np.max(
                np.abs(current.joints - self.progress_anchor)
            ) >= np.deg2rad(0.1):
                self.blocked_since, self.progress_anchor = now, current.joints.copy()
            elif now - self.blocked_since > 1:
                raise MasterSyncError("主臂持续未跟进，两臂暂停。")
        else:
            self.blocked_since = None
        command = self.reference + advance * tracking_fraction
        # An initially out-of-range master may only return toward the strict
        # in-range follower target; it can never be commanded farther outward.
        outside_low = self.reference < JOINT_LIMITS[:, 0]
        outside_high = self.reference > JOINT_LIMITS[:, 1]
        if np.any(command[outside_low] < self.reference[outside_low]) or np.any(
            command[outside_high] > self.reference[outside_high]
        ):
            raise MasterSyncError("主臂越界恢复只允许向关节范围内移动。")
        if not np.isfinite(desired.width) or not 0 <= desired.width <= 0.09:
            raise MasterSyncError("从臂夹爪同步值无效。")
        width = self.width + float(
            np.clip(desired.width - self.width, -self.GRIPPER_SPEED_M_S * dt, self.GRIPPER_SPEED_M_S * dt)
        )
        self._cancel_check()
        self.motion_targets_sent |= bool(np.max(np.abs(command - current.joints)) > 1e-6)
        self.backend.arm.move_js(command.tolist())
        self.backend.last_servo_target = command.copy()
        self.backend.gripper.move_gripper_m(value=width, force=self.GRIPPER_FORCE_N)
        self.backend.last_servo_width = width
        self.reference, self.width, self.last_tick = command, width, now
        complete = self.aligning and pair_gap < np.deg2rad(0.5) and abs(current.width - desired.width) < 0.002
        self.diagnostics = {
            "phase": self.phase,
            "pair_gap_deg": float(np.rad2deg(pair_gap)),
            "tracking_error_deg": float(np.rad2deg(error)),
            "master_joints_deg": np.rad2deg(current.joints).tolist(),
            "master_gripper_mm": current.width * 1000,
            "master_force_n": self.GRIPPER_FORCE_N,
        }
        return bool(complete)

    def prepare_hold(self, follower: Snapshot) -> None:
        """Enable at the measured master pose, without advancing a sync target."""
        self.begin(follower, align=True)
        self.stop()

    def stop(self) -> None:
        """Cancel tracking first, then replace its pending target with measured pose."""
        self.running = self.aligning = False
        if self.faulted:
            return
        if self.drive_ready:
            try:
                self.backend.hold()
            except Exception as exc:
                # A failed preparation that issued only a current-pose hold
                # must not introduce a damping/sag event. If motion had started,
                # request damping once; retain the error without cascading out
                # of recorder cleanup or shutting down the operator console.
                self.diagnostics["stop_error"] = str(exc)
                self.diagnostics["damping_requested"] = self.motion_targets_sent
                if self.motion_targets_sent:
                    try:
                        self.backend.stop()
                    except Exception as stop_error:
                        self.diagnostics["damping_error"] = str(stop_error)
                self.faulted = self.master.control_faulted = True
                self.drive_ready = self.backend.prepared = False
                self.phase = "faulted"
                self.diagnostics["phase"] = self.phase
                return
        self.motion_targets_sent = False
        self.phase = "holding" if self.drive_ready else "not_prepared"
        self.diagnostics["phase"] = self.phase

    def yield_to_human(self) -> None:
        if self.faulted:
            raise MasterSyncError("主臂反馈未恢复，无法确认可进入示教；请先恢复连接。")
        if not self.drive_ready:
            return
        self.stop()
        # MasterInput.prepare() now performs the verified ordinary->teaching
        # transition and releases the motorized gripper before reading the hand.
        self.drive_ready = False
        self.master.prepared = False
        self.backend.prepared = False
        self.phase = "teaching_pending"
        self.diagnostics["phase"] = self.phase
