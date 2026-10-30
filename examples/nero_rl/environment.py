"""Gym-compatible manual-reward task and a synthetic integration-test backend."""

import time
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .config import (
    JOINT_LIMITS,
    STREAM_PROGRESS_RAD,
    STREAM_STALL_SECONDS,
    STREAM_TARGET_LEAD_RAD,
    STREAM_TRACKING_LIMIT_RAD,
    STREAM_WORKSPACE_MARGIN_M,
    TaskConfig,
    vector,
)
from .hardware import Snapshot, StreamTrackingError


class MockHardware:
    """Synthetic point gripper. NOT a NERO physics simulator or a sim-to-real model."""

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self.commands: list[tuple[np.ndarray, float]] = []
        self.prepared = False
        self.reset()

    def reset(self) -> None:
        self.q = np.zeros(7)
        self.q[:3] = [0, 0, 0.03]
        self.width = 0.06
        self.object = np.array([0.01, 0, 0.0])
        self.goal = np.array([-0.015, 0, 0.0])
        self.grasped = False

    def forward(self, q: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_euler("z", q[3]).as_matrix()
        return pose

    def inverse(self, pose: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.copy()
        q[:3] = pose[:3, 3]
        q[3] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")[2]
        return q

    def open(self) -> None:
        pass

    def prepare(self) -> None:
        self.prepared = True

    def read(self, *, images: bool = True) -> Snapshot:
        frames = {}
        for name in self.cfg.cameras if images else ():
            frame = np.zeros((self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8)
            for xyz, color in (
                (self.goal, (0, 180, 0)),
                (self.object, (200, 140, 50)),
                (self.q[:3], (255, 255, 255)),
            ):
                pixel = np.clip(
                    (xyz[:2] / 0.12 + 0.5) * (self.cfg.image_size - 1), 0, self.cfg.image_size - 1
                ).astype(int)
                cv2.circle(frame, tuple(pixel), 3, color, -1)
            frames[name] = frame
        return Snapshot(self.q.copy(), self.width, self.forward(self.q), frames)

    def command(self, q: np.ndarray, width: float) -> None:
        if not self.prepared:
            raise RuntimeError("Mock not prepared")
        self.commands.append((q.copy(), width))
        self.q, self.width = q.copy(), width
        if not self.grasped and width < 0.015 and np.linalg.norm(q[:3] - self.object) < 0.008:
            self.grasped = True
        if self.grasped:
            self.object = q[:3].copy()
        if width > 0.04:
            self.grasped = False

    def success(self) -> bool:
        return not self.grasped and np.linalg.norm(self.object - self.goal) < 0.008

    def servo_command(self, q: np.ndarray, width: float) -> None:
        self.command(q, width)

    def hold(self) -> Snapshot:
        return self.read()

    def stop(self) -> None:
        self.prepared = False

    def close(self) -> None:
        pass


class NeroEnv(gym.Env):
    """Observation/action semantics shared by demos, online RL, and evaluation.

    Joint actions advance the previous command, rather than measured feedback.
    Cartesian actions are TCP increments [dx,dy,dz,base-yaw,dwidth].
    Terminations/reward come from a human; time limits are truncations.
    No automatic reset movement is performed.
    """

    def __init__(self, cfg: TaskConfig, backend: Any, *, mock: bool = False):
        cfg.validate(live=not mock)
        self.cfg, self.backend, self.mock = cfg, backend, mock
        self.observation_lo = vector(cfg.workspace_min_m, 3)
        self.observation_hi = vector(cfg.workspace_max_m, 3)
        self.lo = vector(
            cfg.workspace_min_m if cfg.policy_workspace_min_m is None else cfg.policy_workspace_min_m, 3
        )
        self.hi = vector(
            cfg.workspace_max_m if cfg.policy_workspace_max_m is None else cfg.policy_workspace_max_m, 3
        )
        self.action_space = gym.spaces.Box(-1, 1, (len(cfg.scales),), np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(-np.inf, np.inf, (cfg.state_dim,), np.float32),
                **{
                    f"observation.images.{name}": gym.spaces.Box(
                        0, 255, (3, cfg.image_size, cfg.image_size), np.uint8
                    )
                    for name in cfg.cameras
                },
            }
        )
        self.steps = 0
        self.finished = True
        self.command_joints: np.ndarray | None = None
        self.command_width: float | None = None
        self._tracking_since = np.full(7, np.nan)
        self._tracking_anchor = np.zeros(7)
        self._tracking_direction = np.zeros(7)

    def _inside(self, pose: np.ndarray) -> bool:
        if not np.isfinite(pose[:3, 3]).all():
            return False
        if not self.cfg.policy_workspace_enabled:
            return True
        return bool(np.all(pose[:3, 3] >= self.lo) and np.all(pose[:3, 3] <= self.hi))

    def _joint_path_inside(self, measured: np.ndarray, target: np.ndarray) -> bool:
        return all(
            self._inside(self.backend.forward(measured + t * (target - measured)))
            for t in np.linspace(0, 1, 5)
        )

    def _joint_workspace_target(
        self, measured: np.ndarray, reference: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Shorten an outward increment without disconnecting operator input.

        Every accepted candidate uses the same sampled path check as a normal
        command. If even the previous target has no valid path, require a hold.
        """
        if not self.cfg.policy_workspace_enabled:
            return target, 1.0
        reference_xyz = self.backend.forward(reference)[:3, 3]
        # Avoid aiming a quantized SDK joint command exactly at a floating point
        # TCP boundary. A reference already closer to the edge is not pushed out.
        target_lo = np.minimum(self.lo + STREAM_WORKSPACE_MARGIN_M, reference_xyz)
        target_hi = np.maximum(self.hi - STREAM_WORKSPACE_MARGIN_M, reference_xyz)

        def allowed(candidate: np.ndarray) -> bool:
            xyz = self.backend.forward(candidate)[:3, 3]
            return bool(np.all(xyz >= target_lo) and np.all(xyz <= target_hi)) and self._joint_path_inside(
                measured, candidate
            )

        if allowed(target):
            return target, 1.0
        if not self._joint_path_inside(measured, reference):
            return None, 0.0
        low, high = 0.0, 1.0
        safe = reference.copy()
        for _ in range(10):
            middle = (low + high) / 2
            candidate = reference + middle * (target - reference)
            if allowed(candidate):
                low, safe = middle, candidate
            else:
                high = middle
        return safe, low

    def observation(self, snap: Snapshot) -> dict[str, np.ndarray]:
        joints = 2 * (snap.joints - JOINT_LIMITS[:, 0]) / np.ptp(JOINT_LIMITS, axis=1) - 1
        xyz = 2 * (snap.pose[:3, 3] - self.observation_lo) / (self.observation_hi - self.observation_lo) - 1
        rot6d = snap.pose[:3, :2].T.reshape(-1)
        state = np.r_[joints, xyz, rot6d, 2 * snap.width / self.cfg.gripper_max_m - 1].astype(np.float32)
        if self.cfg.action_mode in ("joint_delta", "joint_target"):
            ref_q = snap.joints if self.command_joints is None else self.command_joints
            ref_width = snap.width if self.command_width is None else self.command_width
            if self.cfg.action_mode == "joint_target":
                reference = self.cfg.encode_joint_target(ref_q, ref_width)
            else:
                normalized_q = 2 * (ref_q - JOINT_LIMITS[:, 0]) / np.ptp(JOINT_LIMITS, axis=1) - 1
                reference = np.r_[normalized_q, 2 * ref_width / self.cfg.gripper_max_m - 1]
            state = np.r_[state, reference].astype(np.float32)
        obs = {"observation.state": state}
        for key, frame in snap.images.items():
            frame = cv2.resize(
                frame, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA
            )
            obs[f"observation.images.{key}"] = np.transpose(frame, (2, 0, 1)).copy()
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if self.mock:
            self.backend.reset()
        snap = self.backend.read()
        if not self._inside(snap.pose):
            raise ValueError("Current TCP is outside the configured task workspace")
        if not self.mock:
            start = vector(self.cfg.start_joints_rad, 7)
            if np.max(np.abs(snap.joints - start)) > self.cfg.start_tolerance_rad:
                raise ValueError("Manually restore the configured start pose before beginning this episode")
        # Validation precedes enable. Reset does not command a home or a gripper width.
        self.backend.prepare()
        self.command_joints, self.command_width = snap.joints.copy(), snap.width
        self._tracking_since.fill(np.nan)
        self.steps, self.finished = 0, False
        return self.observation(snap), {}

    def step(
        self,
        action: np.ndarray,
        *,
        outcome: str | None = None,
        autonomous: bool = False,
        cycle_started: float | None = None,
    ) -> tuple[dict, float, bool, bool, dict]:
        if self.finished:
            raise RuntimeError("Reset required before stepping")
        requested = vector(action, len(self.cfg.scales))
        if np.any(np.abs(requested) > 1.000001):
            raise ValueError("Action must be normalized to [-1, 1]")
        if outcome not in (None, "success", "failure"):
            raise ValueError("Unknown outcome")
        start = time.monotonic()
        before = self.backend.read()
        if not self._inside(before.pose):
            raise RuntimeError("Measured TCP left task workspace")
        applied = np.zeros(len(self.cfg.scales), dtype=np.float32)
        if self.cfg.action_mode == "joint_target":
            if not autonomous and outcome is None:
                raise ValueError("VLA human input uses the native sender, not NeroEnv.step")
            applied = self.cfg.encode_joint_target(*self.joint_reference())
        rejected = None
        tracking_limited = False
        workspace_limited = False
        tracking_details: dict = {}
        target, width = before.pose.copy(), before.width
        # An outcome button labels the present state and does not issue another move.
        if outcome is None and self.cfg.action_mode in ("joint_delta", "joint_target"):
            reference, reference_width = self.joint_reference()
            if self.cfg.action_mode == "joint_target":
                goal_q, goal_width = self.cfg.decode_joint_target(requested)
                delta = np.r_[goal_q - reference, np.clip(goal_width, 0, 0.07) - reference_width]
                delta[:7] = np.clip(delta[:7], -self.cfg.max_joint_step_rad, self.cfg.max_joint_step_rad)
                delta[7] = np.clip(delta[7], -self.cfg.gripper_step_m, self.cfg.gripper_step_m)
            else:
                delta = requested * self.cfg.scales
            if autonomous:
                limit = np.deg2rad(self.cfg.policy_joint_speed_deg_s) / self.cfg.hz
                delta[:7] = np.clip(delta[:7], -limit, limit)
            q = np.clip(reference + delta[:7], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            width = float(np.clip(reference_width + delta[7], 0, self.cfg.gripper_max_m))
            proposed = q.copy()
            advance = q - reference
            error = reference - before.joints
            # Reduce command velocity before reaching the hard tracking bound.
            # Scale the joint vector together, retaining its intended direction.
            allowance = STREAM_TARGET_LEAD_RAD - np.sign(advance) * error
            ratios = np.divide(
                allowance, np.abs(advance), out=np.full(7, np.inf), where=np.abs(advance) > 1e-12
            )
            fraction = float(np.clip(np.min(ratios), 0, 1))
            tracking_limited = fraction < 1 - 1e-6
            q = reference + fraction * advance
            blocked = ratios < 1 - 1e-6
            now = time.monotonic()
            stalled = []
            for joint in range(7):
                direction = np.sign(advance[joint])
                if not blocked[joint]:
                    self._tracking_since[joint] = np.nan
                elif (
                    not np.isfinite(self._tracking_since[joint])
                    or direction != self._tracking_direction[joint]
                    or direction * (before.joints[joint] - self._tracking_anchor[joint])
                    >= STREAM_PROGRESS_RAD
                ):
                    self._tracking_since[joint] = now
                    self._tracking_anchor[joint] = before.joints[joint]
                    self._tracking_direction[joint] = direction
                elif now - self._tracking_since[joint] >= STREAM_STALL_SECONDS:
                    stalled.append(joint + 1)
            tracking_details = {
                "reference_rad": reference.tolist(),
                "measured_rad": before.joints.tolist(),
                "proposed_rad": proposed.tolist(),
                "bounded_target_rad": q.tolist(),
                "velocity_fraction": fraction,
                "limiting_joints": (np.flatnonzero(blocked) + 1).tolist(),
                "stalled_joints": stalled,
            }
            desired_xyz = self.backend.forward(q)[:3, 3]
            bounded, workspace_fraction = self._joint_workspace_target(before.joints, reference, q)
            workspace_limited = workspace_fraction < 1
            tracking_details.update(
                workspace_requested_tcp_m=desired_xyz.tolist(),
                workspace_enforced=self.cfg.policy_workspace_enabled,
                workspace_min_m=self.lo.tolist() if self.cfg.policy_workspace_enabled else None,
                workspace_max_m=self.hi.tolist() if self.cfg.policy_workspace_enabled else None,
                workspace_fraction=workspace_fraction,
                workspace_bounded_target_rad=None if bounded is None else bounded.tolist(),
            )
            if bounded is not None:
                q = bounded
            if np.max(np.abs(error)) > STREAM_TRACKING_LIMIT_RAD:
                rejected = "tracking_error"
            elif stalled:
                rejected = "tracking_stall"
            elif bounded is None:
                rejected = "workspace"
            elif np.max(np.abs(q - before.joints)) > STREAM_TRACKING_LIMIT_RAD:
                rejected = "tracking_error"
            else:
                try:
                    self.backend.servo_command(q, width)
                except StreamTrackingError:
                    rejected = "tracking_error"
                else:
                    applied = (
                        self.cfg.encode_joint_target(q, width)
                        if self.cfg.action_mode == "joint_target"
                        else (np.r_[q - reference, width - reference_width] / self.cfg.scales).astype(
                            np.float32
                        )
                    )
                    self.command_joints, self.command_width = q.copy(), width
        elif outcome is None:
            delta = np.clip(requested, -1, 1) * self.cfg.scales
            target[:3, 3] = np.clip(target[:3, 3] + delta[:3], self.lo, self.hi)
            target[:3, :3] = Rotation.from_euler("z", delta[3]).as_matrix() @ target[:3, :3]
            width = float(np.clip(width + delta[4], 0, self.cfg.gripper_max_m))
            q = self.backend.inverse(target, before.joints)
            if q is None:
                rejected = "ik"
            else:
                q = vector(q, 7)
                feasible = self.backend.forward(q)
                rotation = Rotation.from_matrix(feasible[:3, :3] @ before.pose[:3, :3].T).as_rotvec()
                candidate = (
                    np.r_[feasible[:3, 3] - before.pose[:3, 3], rotation[2], width - before.width]
                    / self.cfg.scales
                )
                path_ok = all(
                    self._inside(self.backend.forward(before.joints + t * (q - before.joints)))
                    for t in np.linspace(0, 1, 11)
                )
                if (
                    np.any(q < JOINT_LIMITS[:, 0])
                    or np.any(q > JOINT_LIMITS[:, 1])
                    or np.max(np.abs(q - before.joints)) > self.cfg.max_joint_step_rad + 1e-6
                    or not path_ok
                    or np.any(np.abs(candidate) > 1.001)
                ):
                    rejected = "bounds_or_joint_step"
                else:
                    # Record the command that actually passed the constraints, not the raw policy output.
                    self.backend.command(q, width)
                    applied = candidate.astype(np.float32)
        if not self.mock:
            deadline = (start if cycle_started is None else cycle_started) + 1 / self.cfg.hz
            time.sleep(max(0, deadline - time.monotonic()))
        after = self.backend.read()
        if not self._inside(after.pose):
            raise RuntimeError("Measured TCP left task workspace after the command")
        self.steps += 1
        terminated = outcome is not None
        truncated = not terminated and self.steps >= round(self.cfg.hz * self.cfg.episode_seconds)
        self.finished = terminated or truncated
        info = {
            "requested_action": requested.astype(np.float32),
            "executed_action": applied,
            "safety_rejection": rejected,
            "tracking_limited": tracking_limited,
            "workspace_limited": workspace_limited,
            "tracking_diagnostics": tracking_details,
            "elapsed_s": time.monotonic() - (start if cycle_started is None else cycle_started),
            "tracking_error_deg": float(np.rad2deg(np.max(np.abs(self.command_joints - after.joints))))
            if self.command_joints is not None
            else 0.0,
        }
        return self.observation(after), float(outcome == "success"), terminated, truncated, info

    def close(self) -> None:
        self.backend.close()

    def joint_reference(self) -> tuple[np.ndarray, float]:
        if self.command_joints is None or self.command_width is None:
            raise RuntimeError("Joint command reference has not been initialized")
        return self.command_joints.copy(), self.command_width

    def hold(self) -> dict[str, np.ndarray]:
        snap = self.backend.hold()
        self.command_joints = snap.joints.copy()
        grip_target = getattr(self.backend, "last_servo_width", None)
        self.command_width = snap.width if grip_target is None else grip_target
        self._tracking_since.fill(np.nan)
        return self.observation(snap)

    def prepare_repositioning(self) -> dict[str, np.ndarray]:
        """Unrecorded operator control can restore the start even on a fresh run."""
        snap = self.backend.read()
        if not self._inside(snap.pose):
            raise ValueError("Current TCP is outside the configured task workspace")
        if not self.backend.prepared:
            self.backend.prepare()
        return self.hold()

    def jog(self, action: np.ndarray) -> dict:
        """Operator-only repositioning between episodes; excluded from replay."""
        if not self.backend.prepared:
            raise RuntimeError("Begin once at the validated start pose before jogging")
        finished, steps = self.finished, self.steps
        self.finished = False
        try:
            _, _, _, _, info = self.step(action)
            return info
        finally:
            self.finished, self.steps = finished, steps
