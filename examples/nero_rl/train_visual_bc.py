"""Offline visual BC with episode-held-out validation and a trained state-only control.

No hardware is connected. Existing coordinate/action conventions remain unchanged.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from .config import TaskConfig
from .learning import Trainer, contract


@dataclass
class BCDataset:
    observations: dict[str, Tensor]
    target: Tensor
    reference: Tensor
    future: Tensor

    def __len__(self) -> int:
        return len(self.target)


def residual_targets(reference: np.ndarray, future: np.ndarray, scale: float) -> np.ndarray:
    target = future.copy()
    delta = future[:, :7] - reference[:, :7]
    target[:, :7] = np.clip(delta / scale, -1, 1)
    target[:, :7][np.abs(delta) * 180 < 0.02] = 0
    return target


def load_split(root: Path, names: list[str], cfg: TaskConfig) -> BCDataset:
    parts: dict[str, list[np.ndarray]] = {}
    targets, references, futures = [], [], []
    scale = cfg.policy_joint_speed_deg_s / cfg.hz / 180
    for name in names:
        if Path(name).name != name:
            raise ValueError("Episode names must be plain filenames")
        with np.load(root / name, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["mock"] or meta["kind"] == "eval" or data["reward"][-1] != 1:
                raise ValueError(f"Not a successful real demonstration: {name}")
            if meta["contract"] != contract(cfg, False):
                raise ValueError(f"Incompatible demonstration: {name}")
            stamps = data["control_timestamp"]
            indices = np.arange(0, len(stamps), 5)
            future = np.searchsorted(stamps, stamps[indices] + 0.1)
            valid = future < len(stamps)
            indices, future = indices[valid], future[valid]
            valid = stamps[future] - stamps[indices] <= 0.2
            indices, future = indices[valid], future[valid]
            actions = data["action"]
            observations = {
                key.removeprefix("state/"): data[key][indices]
                for key in data.files
                if key.startswith("state/")
            }
            reference = observations["observation.state"][:, -8:]
            for key, values in observations.items():
                parts.setdefault(key, []).append(values)
            references.append(reference)
            futures.append(actions[future])
            targets.append(residual_targets(reference, actions[future], scale))
    if not targets:
        raise ValueError("Empty demonstration split")
    tensor = lambda arrays: torch.from_numpy(np.concatenate(arrays)).to(cfg.device)  # noqa: E731
    return BCDataset(
        {key: tensor(values) for key, values in parts.items()},
        tensor(targets),
        tensor(references),
        tensor(futures),
    )


def batch_inputs(data: BCDataset, index: Any, *, blank: bool = False) -> dict[str, Tensor]:
    result = {}
    for key, values in data.observations.items():
        batch = values[index].float()
        if key.startswith("observation.images."):
            batch = torch.zeros_like(batch) if blank else batch / 255 - 0.5
        result[key] = batch
    return result


def supervised_mean(trainer: Trainer, observations: dict[str, Tensor], *, state_dropout: float = 0) -> Tensor:
    # SAC detaches the shared encoder on the actor path. BC explicitly trains it.
    actor = trainer.policy.actor
    features = actor.encoder(observations, detach=False)
    if state_dropout:
        state_dim = actor.encoder.config.latent_dim
        keep = torch.rand(len(features), 1, device=features.device) >= state_dropout
        features = torch.cat((features[:, :-state_dim], features[:, -state_dim:] * keep), dim=-1)
    return actor.mean_layer(actor.network(features)).tanh()


def supervised_loss(prediction: Tensor, target: Tensor) -> Tensor:
    error = functional.smooth_l1_loss(prediction, target, reduction="none")
    joint_weight = torch.where(target[:, :7].abs() > 0.9, 2.0, 1.0)
    return (error[:, :7] * joint_weight).mean() + 2 * error[:, 7].mean()


@torch.no_grad()
def evaluate(trainer: Trainer, data: BCDataset, mode: str = "full") -> dict[str, float]:
    trainer.policy.eval()
    predictions = []
    for start in range(0, len(data), 256):
        stop = min(start + 256, len(data))
        obs = batch_inputs(data, slice(start, stop), blank=mode == "blank")
        if mode == "shuffle":
            # Keep proprioception fixed, swap synchronized pairs of real camera images.
            other = torch.arange(start, stop, device=data.target.device)
            other = (other + len(data) // 2) % len(data)
            for key in obs:
                if key.startswith("observation.images."):
                    obs[key] = data.observations[key][other].float() / 255 - 0.5
        predictions.append(supervised_mean(trainer, obs))
    predicted = torch.cat(predictions)
    active = (data.future[:, :7] - data.reference[:, :7]).abs() * 180 > 0.2
    correct = (predicted[:, :7].sign() == data.target[:, :7].sign()) & active
    return {
        "loss": supervised_loss(predicted, data.target).item(),
        "residual_mse": functional.mse_loss(predicted, data.target).item(),
        "moving_joint_direction_accuracy": correct.sum().item() / max(1, active.sum().item()),
        "moving_joint_cases": active.sum().item(),
        "gripper_mae_mm": ((predicted[:, 7] - data.target[:, 7]).abs() * trainer.cfg.gripper_max_m * 500)
        .mean()
        .item(),
    }


def export_bc(trainer: Trainer, output: Path) -> None:
    # Encoder changes invalidate old critic features/optimizer moments. Export a
    # fresh SAC bundle with trained policy weights, rather than stale critics.
    exported = Trainer(replace(trainer.cfg, protect_pretrained_policy=False), mock=False)
    exported.policy.load_state_dict(trainer.policy.state_dict())
    exported.requires_pretrained_protection = True
    exported.save(output)


def train_candidate(
    cfg: TaskConfig, args: argparse.Namespace, train: BCDataset, val: BCDataset, *, blank: bool
) -> dict:
    trainer = Trainer(cfg, mock=False)
    trainer.load(args.resume)
    policy = trainer.policy
    encoder = policy.actor.encoder
    # Visual BC is the explicit representation-learning stage; protected SAC
    # freezes this encoder, so a BC refit must deliberately enable its gradients.
    encoder.requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in encoder.image_encoder.state_dict().items()}
    parameters = [
        *encoder.parameters(),
        *policy.actor.network.parameters(),
        *policy.actor.mean_layer.parameters(),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    moving = (
        torch.nonzero((train.future[:, :7] - train.reference[:, :7]).abs().amax(dim=1) * 180 > 0.2)
        .flatten()
        .cpu()
        .numpy()
    )
    validation_mode = "blank" if blank else "full"
    before = evaluate(trainer, val, validation_mode)
    best_loss, best_step = float("inf"), 0
    best = None
    history = []
    first_gradient_norm = 0.0
    started = time.monotonic()
    name = "state_only" if blank else "visual"
    for step in range(1, args.steps + 1):
        policy.train()
        half = args.batch_size // 2
        chosen = np.r_[rng.integers(len(train), size=args.batch_size - half), rng.choice(moving, size=half)]
        index = torch.as_tensor(chosen, device=cfg.device)
        obs = batch_inputs(train, index, blank=blank)
        if not blank:
            # Mild brightness augmentation; do not mirror camera geometry.
            for key in obs:
                if key.startswith("observation.images."):
                    factor = 0.95 + 0.1 * torch.rand(len(index), 1, 1, 1, device=cfg.device)
                    obs[key] = ((obs[key] + 0.5) * factor).clamp(0, 1) - 0.5
        prediction = supervised_mean(trainer, obs, state_dropout=0 if blank else 0.1)
        loss = supervised_loss(prediction, train.target[index])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised loss")
        optimizer.zero_grad()
        loss.backward()
        if step == 1:
            first_gradient_norm = (
                sum(
                    float(p.grad.detach().square().sum())
                    for p in encoder.image_encoder.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            if not blank and first_gradient_norm <= 0:
                raise RuntimeError("Vision encoder received no BC gradient")
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if step == args.steps * 3 // 4:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            result = evaluate(trainer, val, validation_mode)
            entry = {
                "candidate": name,
                "step": step,
                "train_loss": loss.item(),
                "validation": result,
                "elapsed_s": time.monotonic() - started,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            if result["loss"] < best_loss:
                best_loss, best_step = result["loss"], step
                best = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best)
    policy.eval()
    with torch.no_grad():
        policy.actor.std_layer.weight.zero_()
        policy.actor.std_layer.bias[:7].fill_(math.log(0.2))
        policy.actor.std_layer.bias[7:].fill_(math.log(0.05))
    change = (
        sum(float((v - initial[k]).square().sum()) for k, v in encoder.image_encoder.state_dict().items())
        ** 0.5
    )
    path = args.output / f"{name}_checkpoint.pt"
    export_bc(trainer, path)
    restored = Trainer(cfg, mock=False)
    restored.load(path)
    sample = {k: v[0].detach().cpu().numpy() for k, v in val.observations.items()}
    np.testing.assert_allclose(
        trainer.act(sample, deterministic=True), restored.act(sample, deterministic=True), atol=1e-7
    )
    return {
        "checkpoint": str(path),
        "best_step": best_step,
        "gradient_steps_run": args.steps,
        "before": before,
        "after": evaluate(trainer, val, validation_mode),
        "blank_images": evaluate(trainer, val, "blank"),
        "shuffled_images": evaluate(trainer, val, "shuffle"),
        "first_cnn_gradient_norm": first_gradient_norm,
        "cnn_weight_change_l2": change,
        "checkpoint_reload_verified": True,
        "sac_critics_and_optimizers_reinitialized": True,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size < 2 or args.eval_every <= 0 or args.learning_rate <= 0:
        parser.error("Invalid training schedule")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing training artifacts are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = replace(TaskConfig.load(args.config), device=args.device)
    if not cfg.policy_reference_residual or cfg.freeze_vision_encoder or not cfg.cameras:
        parser.error("Requires an unfrozen visual reference-residual policy")
    cfg.validate(live=True)
    cfg.save(args.output / "config.json")
    split = json.loads(args.split_file.read_text())
    train_names, val_names = split["train_episodes"], split["validation_episodes"]
    if set(train_names) & set(val_names) or len(set(train_names + val_names)) != len(train_names + val_names):
        parser.error("Episode split contains overlap or duplicate names")
    train, val = load_split(args.data, train_names, cfg), load_split(args.data, val_names, cfg)
    report = {
        "source_checkpoint": str(args.resume),
        "method": "End-to-end visual behavior cloning; actual images, CNN and state encoder receive gradients.",
        "train_episodes": train_names,
        "validation_episodes": val_names,
        "train_samples": len(train),
        "validation_samples": len(val),
        "live_deployed": False,
        "hardware_connected": False,
        "limitation": "Held out from this BC run; source SAC previously saw all demos. No fresh task-success evaluation.",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("episodes")}), flush=True)
    for blank in (False, True):
        name = "state_only" if blank else "visual"
        report[name] = train_candidate(cfg, args, train, val, blank=blank)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    visual, baseline = report["visual"], report["state_only"]
    gain = 1 - visual["after"]["loss"] / baseline["after"]["loss"]
    shuffle_cost = visual["shuffled_images"]["loss"] / visual["after"]["loss"] - 1
    report["comparison"] = {
        "validation_loss_improvement_over_state_only": gain,
        "validation_loss_increase_when_images_shuffled": shuffle_cost,
        "passes_offline_visual_gate": gain > 0.05 and shuffle_cost > 0.05,
        "interpretation": "Image dependence and held-out imitation quality only, not object localization or physical task success.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL " + json.dumps(report["comparison"]), flush=True)


if __name__ == "__main__":
    main()
