"""Explicit task geometry and limits for the NERO RL example."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

JOINT_LIMITS = np.array(
    [
        [-2.705261, 2.705261],
        [-1.74533, 1.74533],
        [-2.757621, 2.757621],
        [-1.012291, 2.146755],
        [-2.757621, 2.757621],
        [-0.733039, 0.872665],
        [-1.570797, 1.570797],
    ]
)
ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "dgripper")
STREAM_TRACKING_LIMIT_RAD = math.radians(3)
STREAM_TARGET_LEAD_RAD = math.radians(2.5)
STREAM_STALL_SECONDS = 1.0
STREAM_PROGRESS_RAD = math.radians(0.1)
STREAM_WORKSPACE_MARGIN_M = 0.0005


@dataclass
class TaskConfig:
    task: str = "Pick up the wooden ornament from the table and place it in its packaging."
    channel: str = "can1"
    # These are task measurements, not example values to copy onto hardware.
    workspace_min_m: list[float] | None = None
    workspace_max_m: list[float] | None = None
    # Optional native-policy execution bounds. Keep workspace_* unchanged for
    # the coordinate normalization learned by an existing checkpoint.
    policy_workspace_min_m: list[float] | None = None
    policy_workspace_max_m: list[float] | None = None
    policy_workspace_enabled: bool = True
    start_joints_rad: list[float] | None = None
    tcp_offset_m: list[float] | None = None
    geometry_verified: bool = False
    action_mode: str = "cartesian"
    cameras: dict[str, str] = field(default_factory=dict)
    image_size: int = 84
    hz: float = 5.0
    episode_seconds: float = 120.0
    translation_step_m: float = 0.001
    yaw_step_rad: float = math.radians(1)
    gripper_step_m: float = 0.002
    gripper_max_m: float = 0.07
    gripper_force_n: float = 3.0
    max_joint_step_rad: float = math.radians(0.5)
    policy_joint_speed_deg_s: float = 5.0
    start_tolerance_rad: float = math.radians(3)
    tracking_tolerance_rad: float = math.radians(0.1)
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 32
    replay_capacity: int = 5000
    updates_per_episode: int = 50
    learning_starts: int = 64
    # From scratch CNN is trainable; an optional local/HF encoder can be supplied.
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = False
    policy_reference_residual: bool = False
    # Opt-in continuation of a pretrained residual policy. Keep its visual
    # representation fixed while fitting Q, and anchor subsequent actor updates.
    protect_pretrained_policy: bool = False
    critic_warmup_updates: int = 1000
    bc_anchor_weight: float = 10.0
    rl_actor_learning_rate: float = 3e-5

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")

    def validate(self, *, live: bool = False) -> None:
        for name in (
            "hz",
            "episode_seconds",
            "translation_step_m",
            "yaw_step_rad",
            "gripper_step_m",
            "gripper_max_m",
            "gripper_force_n",
            "max_joint_step_rad",
            "start_tolerance_rad",
            "tracking_tolerance_rad",
            "policy_joint_speed_deg_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("image_size", "batch_size", "replay_capacity", "updates_per_episode", "learning_starts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 64:
            raise ValueError("The CNN requires image_size >= 64")
        if self.replay_capacity < max(self.batch_size, self.learning_starts):
            raise ValueError("Replay capacity is smaller than batch/learning threshold")
        if self.freeze_vision_encoder and self.vision_encoder_name is None:
            raise ValueError("Do not freeze an untrained random image encoder")
        if self.action_mode not in ("cartesian", "joint_delta", "joint_target"):
            raise ValueError("Unknown action_mode")
        if type(self.policy_workspace_enabled) is not bool:
            raise ValueError("policy_workspace_enabled must be boolean")
        if not self.policy_workspace_enabled and self.action_mode != "joint_target":
            raise ValueError("Disabling the task workspace requires native joint_target mode")
        if type(self.policy_reference_residual) is not bool:
            raise ValueError("policy_reference_residual must be boolean")
        if self.policy_reference_residual and self.action_mode != "joint_target":
            raise ValueError("Reference residual policy requires absolute joint targets")
        if type(self.protect_pretrained_policy) is not bool:
            raise ValueError("protect_pretrained_policy must be boolean")
        if self.protect_pretrained_policy and not self.policy_reference_residual:
            raise ValueError("Pretrained policy protection requires a reference residual policy")
        if type(self.critic_warmup_updates) is not int or self.critic_warmup_updates < 0:
            raise ValueError("critic_warmup_updates must be a nonnegative integer")
        for name in ("bc_anchor_weight", "rl_actor_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.workspace_min_m is not None or self.workspace_max_m is not None:
            lo, hi = vector(self.workspace_min_m, 3), vector(self.workspace_max_m, 3)
            if np.any(lo >= hi):
                raise ValueError("Workspace min must be below max on every axis")
        if self.policy_workspace_min_m is not None or self.policy_workspace_max_m is not None:
            lo = vector(self.policy_workspace_min_m, 3)
            hi = vector(self.policy_workspace_max_m, 3)
            if self.action_mode != "joint_target":
                raise ValueError("Separate policy workspace requires native joint_target mode")
            if np.any(lo >= hi):
                raise ValueError("Policy workspace min must be below max on every axis")
        if live:
            if self.channel != "can1":
                raise ValueError("This task is explicitly restricted to right follower can1")
            if not self.geometry_verified or not self.cameras:
                raise ValueError("Verify TCP, workspace, start pose and camera mapping before live use")
            vector(self.workspace_min_m, 3)
            vector(self.workspace_max_m, 3)
            vector(self.tcp_offset_m, 3)
            q = vector(self.start_joints_rad, 7)
            if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
                raise ValueError("Start pose exceeds joint limits")
            if (
                self.translation_step_m > 0.002
                or self.max_joint_step_rad > math.radians(1)
                or self.yaw_step_rad > math.radians(2)
                or self.hz > (100 if self.action_mode in ("joint_delta", "joint_target") else 10)
                or self.policy_joint_speed_deg_s > 5
                or self.gripper_step_m > 0.003
                or self.gripper_max_m > (0.09 if self.action_mode == "joint_target" else 0.07)
                or self.gripper_force_n > 5
            ):
                raise ValueError("Live configuration exceeds initial commissioning limits")

    @property
    def scales(self) -> np.ndarray:
        if self.action_mode == "joint_target":
            return np.array([np.pi] * 7 + [self.gripper_max_m / 2])
        if self.action_mode == "joint_delta":
            return np.array([self.max_joint_step_rad] * 7 + [self.gripper_step_m])
        return np.array([self.translation_step_m] * 3 + [self.yaw_step_rad, self.gripper_step_m])

    @property
    def state_dim(self) -> int:
        # A streamed target can lead measured feedback. Expose that reference so
        # the next joint increment has the same meaning in replay and on hardware.
        return 25 if self.action_mode in ("joint_delta", "joint_target") else 17

    def encode_joint_target(self, q: np.ndarray, width: float) -> np.ndarray:
        return np.r_[vector(q, 7) / np.pi, 2 * width / self.gripper_max_m - 1].astype(np.float32)

    def decode_joint_target(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = vector(action, 8)
        return action[:7] * np.pi, float((action[7] + 1) * self.gripper_max_m / 2)


def vector(values: object, size: int) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected {size} finite numbers")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Boolean values are not coordinates")
    a = np.asarray(values, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"Expected {size} finite numbers")
    return a
