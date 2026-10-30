"""SAC continuation that preserves a pretrained residual policy's representation.

This is an opt-in local experiment, not the full HIL-SERL implementation. The
reference is immutable and checkpointed: resuming never makes a drifting actor
its own new teacher. Images remain inputs, but Q gradients cannot rewrite the
pretrained shared encoder. Improving that encoder requires separate visual BC.
"""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy, Policy
from lerobot.rl.algorithms.configs import TrainingStats
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig


class AnchoredSACAlgorithm(SACAlgorithm):
    def __init__(
        self,
        policy: GaussianActorPolicy,
        config: SACAlgorithmConfig,
        *,
        warmup_updates: int,
        anchor_weight: float,
    ) -> None:
        super().__init__(policy, config)
        self.warmup_updates = warmup_updates
        self.anchor_weight = anchor_weight
        self.reference: Policy | None = None
        self.last_anchor_loss = 0.0
        # The encoder is shared by actor and critics. Merely detaching it on the
        # actor path does not stop critic gradients from changing visual behavior.
        self.policy.actor.encoder.requires_grad_(False)

    def set_reference(self, state: dict[str, Tensor] | None = None) -> None:
        self.reference = deepcopy(self.policy.actor).eval().requires_grad_(False)
        if state is not None:
            self.reference.load_state_dict(state)

    @staticmethod
    def residual_mean(actor: Policy, observations: dict[str, Tensor]) -> Tensor:
        features = actor.encoder(observations, detach=True)
        return actor.mean_layer(actor.network(features)).tanh()

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        if self.reference is None:
            raise RuntimeError("Load a pretrained checkpoint before protected SAC updates")
        sac_loss = super()._compute_loss_actor(batch)
        observations = batch["state"]
        with torch.no_grad():
            target = self.residual_mean(self.reference, observations)
        predicted = self.residual_mean(self.policy.actor, observations)
        # Regularize in residual units, not tiny absolute joint increments; the
        # latter would make joint preservation effectively zero at 100 Hz.
        anchor_loss = functional.smooth_l1_loss(predicted[:, :7], target[:, :7])
        anchor_loss = anchor_loss + 2 * functional.smooth_l1_loss(predicted[:, 7:], target[:, 7:])
        self.last_anchor_loss = float(anchor_loss.detach())
        return sac_loss + self.anchor_weight * anchor_loss

    @torch.no_grad()
    def _update_target_networks(self) -> None:
        for target, source in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            # Upstream target/online critics share the policy encoder. Averaging
            # a tensor with itself can still accumulate floating-point drift.
            if target is not source:
                target.lerp_(source, self.config.critic_target_update_weight)

    def update(self, batch_iterator: Iterator[dict[str, Any]]) -> TrainingStats:
        if self.reference is None:
            raise RuntimeError("Load a pretrained checkpoint before protected SAC updates")
        if self._optimization_step >= self.warmup_updates:
            stats = super().update(batch_iterator)
            stats.losses["loss_bc_anchor"] = self.last_anchor_loss
            stats.extra["critic_warmup_remaining"] = 0
            return stats

        # Fit fresh Q heads without an actor, encoder, or temperature step. This
        # counter persists in the normal SAC checkpoint across saved episodes.
        batch = self._prepare_forward_batch(next(batch_iterator), include_complementary_info=False)
        loss = self._compute_loss_critic(batch)
        self.optimizers["critic"].zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(), self.config.grad_clip_norm
        ).item()
        self.optimizers["critic"].step()
        self._update_target_networks()
        self._optimization_step += 1
        return TrainingStats(
            losses={"loss_critic": loss.item()},
            grad_norms={"critic": grad},
            extra={"critic_warmup_remaining": max(0, self.warmup_updates - self._optimization_step)},
        )
