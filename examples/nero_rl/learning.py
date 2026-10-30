"""SAC training, mixed demonstration replay, and portable episode files."""

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig

from .anchored_sac import AnchoredSACAlgorithm
from .config import (
    STREAM_PROGRESS_RAD,
    STREAM_STALL_SECONDS,
    STREAM_TARGET_LEAD_RAD,
    STREAM_TRACKING_LIMIT_RAD,
    STREAM_WORKSPACE_MARGIN_M,
    TaskConfig,
)
from .residual_actor import ReferenceResidualActor


def contract(cfg: TaskConfig, mock: bool) -> str:
    """Prevent silently mixing different robots, geometry, image/action conventions."""
    values = asdict(cfg)
    for key in (
        "device",
        "seed",
        "batch_size",
        "replay_capacity",
        "updates_per_episode",
        "learning_starts",
        "vision_encoder_name",
        "freeze_vision_encoder",
        "policy_reference_residual",
        # Execution bounds do not alter recorded absolute actions or normalized
        # observations. Each episode/checkpoint still stores its full config.
        "policy_workspace_min_m",
        "policy_workspace_max_m",
        "policy_workspace_enabled",
        "protect_pretrained_policy",
        "critic_warmup_updates",
        "bc_anchor_weight",
        "rl_actor_learning_rate",
    ):
        values.pop(key)
    values.update(
        schema=5 if cfg.action_mode == "joint_delta" else 2,
        mock=mock,
        state="q7,tcp_xyz3,rotation_columns6,gripper1,command_q7,command_width1"
        if cfg.action_mode == "joint_delta"
        else "q7,tcp_xyz3,rotation_columns6,gripper1",
        action="delta_previous_command:dq1,dq2,dq3,dq4,dq5,dq6,dq7,dwidth"
        if cfg.action_mode == "joint_delta"
        else "dx,dy,dz,base_yaw,dwidth",
    )
    if cfg.action_mode == "joint_delta":
        values["stream_tracking"] = {
            "hard_limit_rad": STREAM_TRACKING_LIMIT_RAD,
            "target_lead_rad": STREAM_TARGET_LEAD_RAD,
            "stall_seconds": STREAM_STALL_SECONDS,
            "progress_rad": STREAM_PROGRESS_RAD,
            "workspace": "shorten_joint_target_with_valid_sampled_path",
            "workspace_margin_m": STREAM_WORKSPACE_MARGIN_M,
        }
    elif cfg.action_mode == "joint_target":
        values.update(
            schema=6,
            state="q7,tcp_xyz3,rotation_columns6,gripper1,previous_absolute_action8",
            action="absolute_q7_div_pi,width_2w_over_max_minus_1",
            human_control="vla_native_js_unfiltered",
            policy_control="bounded_joint_target",
            policy_guard={
                "hard_limit_rad": STREAM_TRACKING_LIMIT_RAD,
                "target_lead_rad": STREAM_TARGET_LEAD_RAD,
                "stall_seconds": STREAM_STALL_SECONDS,
                "progress_rad": STREAM_PROGRESS_RAD,
                "workspace_margin_m": STREAM_WORKSPACE_MARGIN_M,
                "gripper_max_m": 0.07,
            },
        )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def tensors(
    observations: dict[str, np.ndarray], device: str, *, batch: bool = False
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in observations.items():
        tensor = torch.as_tensor(value, device=device).float()
        if key.startswith("observation.images."):
            tensor = tensor / 255.0 - 0.5
        result[key] = tensor if batch else tensor.unsqueeze(0)
    return result


class MixedReplay:
    """Bounded CPU replay retaining uint8 images; interventions also enter demo replay."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.online: list[dict[str, Any]] = []
        self.demos: list[dict[str, Any]] = []
        self.positions = [0, 0]
        self.rng = np.random.default_rng(seed)

    def _append(self, buffer: list, row: dict, index: int) -> None:
        if len(buffer) < self.capacity:
            buffer.append(row)
        else:
            buffer[self.positions[index]] = row
        self.positions[index] = (self.positions[index] + 1) % self.capacity

    def add(self, row: dict[str, Any]) -> None:
        self._append(self.online, row, 0)
        if row["intervention"]:
            self._append(self.demos, row, 1)

    def sample(self, size: int, device: str) -> dict[str, Any]:
        if not self.online:
            raise ValueError("Replay is empty")
        n_demo = size // 2 if self.demos else 0
        rows = [self.online[i] for i in self.rng.integers(len(self.online), size=size - n_demo)]
        rows += [self.demos[i] for i in self.rng.integers(len(self.demos), size=n_demo)]
        self.rng.shuffle(rows)
        result = {}
        for group in ("state", "next_state"):
            observations = {k: np.stack([r[group][k] for r in rows]) for k in rows[0][group]}
            result[group] = tensors(observations, device, batch=True)
        for key in ("action", "reward", "done", "truncated"):
            result[key] = torch.as_tensor(np.array([r[key] for r in rows]), device=device).float()
        result["complementary_info"] = {}
        return result


def save_episode(root: Path, rows: list[dict], cfg: TaskConfig, *, mock: bool, kind: str) -> Path:
    if not rows:
        raise ValueError("Cannot save an empty episode")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{kind}_{uuid.uuid4().hex}.npz"
    arrays: dict[str, Any] = {}
    for group in ("state", "next_state"):
        for key in rows[0][group]:
            arrays[f"{group}/{key}"] = np.stack([r[group][key] for r in rows])
    for key in (
        "action",
        "requested_action",
        "reward",
        "done",
        "truncated",
        "intervention",
        "elapsed_s",
        "safety_rejection",
    ):
        arrays[key] = np.array([row[key] for row in rows])
    arrays["metadata"] = np.array(
        json.dumps(
            {
                "schema": 1,
                "contract": contract(cfg, mock),
                "kind": kind,
                "config": asdict(cfg),
                "mock": mock,
                "action_sources": sorted({row.get("action_source", "unknown") for row in rows}),
            }
        )
    )
    arrays["action_source"] = np.array([row.get("action_source", "unknown") for row in rows])
    for key in ("control_timestamp", "camera_timestamps"):
        if all(key in row for row in rows):
            arrays[key] = np.array([row[key] for row in rows])
    np.savez_compressed(path, **arrays)
    return path


def load_episodes(root: Path, replay: MixedReplay, cfg: TaskConfig, *, mock: bool) -> int:
    count = 0
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["kind"] == "eval":
                continue
            if meta["contract"] != contract(cfg, mock):
                raise ValueError(f"Incompatible episode geometry/action schema: {path}")
            states = [key.removeprefix("state/") for key in data.files if key.startswith("state/")]
            # NPZ members are decompressed on every __getitem__. Materialize
            # each array once, then copy individual transitions into replay.
            observations = {
                group: {k: data[f"{group}/{k}"] for k in states} for group in ("state", "next_state")
            }
            arrays = {k: data[k] for k in ("action", "reward", "done", "truncated", "intervention")}
            for i in range(len(arrays["action"])):
                row = {
                    group: {k: values[i].copy() for k, values in observations[group].items()}
                    for group in ("state", "next_state")
                }
                row.update({k: values[i].copy() for k, values in arrays.items()})
                replay.add(row)
                count += 1
    return count


class Trainer:
    def __init__(self, cfg: TaskConfig, *, mock: bool):
        self.cfg, self.mock = cfg, mock
        torch.manual_seed(cfg.seed)
        features = {"observation.state": PolicyFeature(FeatureType.STATE, (cfg.state_dim,))}
        features.update(
            {
                f"observation.images.{name}": PolicyFeature(
                    FeatureType.VISUAL, (3, cfg.image_size, cfg.image_size)
                )
                for name in cfg.cameras
            }
        )
        policy_cfg = GaussianActorConfig(
            device=cfg.device,
            input_features=features,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (len(cfg.scales),))},
            vision_encoder_name=cfg.vision_encoder_name,
            freeze_vision_encoder=cfg.freeze_vision_encoder,
            image_encoder_hidden_dim=16,
            latent_dim=64,
            state_encoder_hidden_dim=64,
            dataset_stats=None,
        )
        self.policy = GaussianActorPolicy(policy_cfg).to(cfg.device)
        if cfg.policy_reference_residual:
            joint_scale = cfg.policy_joint_speed_deg_s / cfg.hz / 180
            self.policy.actor = ReferenceResidualActor(self.policy.actor, joint_scale)
        algorithm_cfg = SACAlgorithmConfig.from_policy_config(policy_cfg)
        if cfg.action_mode in ("joint_delta", "joint_target"):
            # Keep approximately the 10 Hz discount horizon when control runs faster.
            algorithm_cfg.discount = 0.99 ** (10 / cfg.hz)
        if cfg.policy_reference_residual:
            # Differential entropy shifts under the affine action transform.
            base_entropy = (
                algorithm_cfg.target_entropy
                if algorithm_cfg.target_entropy is not None
                else -len(cfg.scales) / 2
            )
            algorithm_cfg.target_entropy = base_entropy + 7 * np.log(joint_scale)
        algorithm_cfg.temperature_init = 0.1
        if cfg.protect_pretrained_policy:
            algorithm_cfg.actor_lr = cfg.rl_actor_learning_rate
            # The residual transform adds a large constant to log-probability
            # at 100 Hz. Do not let that unit-dependent offset dominate sparse
            # task rewards in the value targets; actor entropy remains enabled.
            algorithm_cfg.use_backup_entropy = False
            self.algorithm = AnchoredSACAlgorithm(
                self.policy,
                algorithm_cfg,
                warmup_updates=cfg.critic_warmup_updates,
                anchor_weight=cfg.bc_anchor_weight,
            )
        else:
            self.algorithm = SACAlgorithm(self.policy, algorithm_cfg)
        self.algorithm.make_optimizers_and_scheduler()
        self.replay = MixedReplay(cfg.replay_capacity, cfg.seed)
        self.updates = 0
        self.requires_pretrained_protection = False

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], *, deterministic: bool = False) -> np.ndarray:
        self.policy.eval()
        output = self.policy(tensors(obs, self.cfg.device))
        # The upstream actor returns pre-tanh means, while sampled actions are squashed.
        action = output["action_mean"].tanh() if deterministic else output["action"]
        return action[0].cpu().numpy()

    def update(self, count: int) -> dict[str, float]:
        if self.requires_pretrained_protection and not self.cfg.protect_pretrained_policy:
            raise ValueError("Visual BC continuation requires protect_pretrained_policy=true")
        if len(self.replay.online) < self.cfg.learning_starts:
            return {}
        self.policy.train()

        def batches():
            while True:
                yield self.replay.sample(self.cfg.batch_size, self.cfg.device)

        iterator = batches()
        stats = {}
        for _ in range(count):
            stats = self.algorithm.update(iterator).to_log_dict()
            if not all(np.isfinite(value) for value in stats.values()):
                raise FloatingPointError(f"Non-finite training statistics: {stats}")
            self.updates += 1
        self.policy.eval()
        return stats

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema": 1,
            "contract": contract(self.cfg, self.mock),
            "policy": self.policy.state_dict(),
            "algorithm": self.algorithm.state_dict(),
            "updates": self.updates,
            "optimization_step": self.algorithm._optimization_step,
            "optimizers": {key: opt.state_dict() for key, opt in self.algorithm.optimizers.items()},
            "rng": torch.get_rng_state(),
            "numpy_rng": json.dumps(self.replay.rng.bit_generator.state),
            "config": json.dumps(asdict(self.cfg)),
            "requires_pretrained_protection": self.requires_pretrained_protection,
        }
        if isinstance(self.algorithm, AnchoredSACAlgorithm):
            if self.algorithm.reference is None:
                raise RuntimeError("Protected SAC checkpoint has no pretrained reference")
            bundle["bc_reference"] = self.algorithm.reference.state_dict()
        temporary = path.with_suffix(".tmp")
        torch.save(bundle, temporary)
        # Keep earlier deployed revisions instead of replacing the only copy.
        if path.exists():
            revisions = path.parent / "checkpoints"
            revisions.mkdir(exist_ok=True)
            previous = torch.load(path, map_location="cpu", weights_only=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            revision = revisions / f"{path.stem}_updates_{previous['updates']:08d}_{digest}{path.suffix}"
            if not revision.exists():
                shutil.copy2(path, revision)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        bundle = torch.load(path, map_location=self.cfg.device, weights_only=True)
        if bundle["contract"] != contract(self.cfg, self.mock):
            raise ValueError("Checkpoint uses different geometry/actions or mock/live mode")
        old_cfg = json.loads(bundle["config"])
        if old_cfg.get("protect_pretrained_policy", False) and not self.cfg.protect_pretrained_policy:
            raise ValueError("Protected checkpoint requires protect_pretrained_policy=true")
        if self.cfg.protect_pretrained_policy and "bc_reference" not in bundle and bundle["updates"]:
            raise ValueError(
                "Start protected SAC from the original BC checkpoint, not an unanchored RL revision"
            )
        for key in ("vision_encoder_name", "freeze_vision_encoder"):
            if old_cfg[key] != getattr(self.cfg, key):
                raise ValueError(f"Checkpoint encoder setting differs: {key}")
        if old_cfg.get("policy_reference_residual", False) != self.cfg.policy_reference_residual:
            raise ValueError("Checkpoint residual-policy architecture differs")
        self.policy.load_state_dict(bundle["policy"])
        self.algorithm.load_state_dict(bundle["algorithm"], device=self.cfg.device)
        for key, opt in self.algorithm.optimizers.items():
            opt.load_state_dict(bundle["optimizers"][key])
        self.updates = bundle["updates"]
        self.requires_pretrained_protection = bool(bundle.get("requires_pretrained_protection", False))
        self.requires_pretrained_protection |= self.cfg.protect_pretrained_policy
        self.algorithm._optimization_step = bundle["optimization_step"]
        if isinstance(self.algorithm, AnchoredSACAlgorithm):
            self.algorithm.set_reference(bundle.get("bc_reference"))
            # Legacy BC exports have an empty Adam state at the upstream LR.
            for group in self.algorithm.optimizers["actor"].param_groups:
                group["lr"] = self.cfg.rl_actor_learning_rate
        torch.set_rng_state(bundle["rng"].cpu())
        self.replay.rng.bit_generator.state = json.loads(bundle["numpy_rng"])
