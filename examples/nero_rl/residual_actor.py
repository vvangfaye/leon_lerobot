"""Small stochastic joint corrections with the existing absolute-action interface."""

import math

import torch
from torch import nn

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import Policy, TanhMultivariateNormalDiag


class ReferenceResidualActor(Policy):
    """Keep native absolute replay actions while bounding learned joint changes.

    The gripper retains its absolute [-1,1] distribution. Joint samples are an
    affine transform of a tanh distribution around the last commanded target;
    log probabilities include the transform's Jacobian.
    """

    def __init__(self, original: Policy, joint_scale: float):
        nn.Module.__init__(self)
        if not math.isfinite(joint_scale) or joint_scale <= 0:
            raise ValueError("Residual joint scale must be positive")
        self.encoder = original.encoder
        self.network = original.network
        self.mean_layer = original.mean_layer
        self.fixed_std = original.fixed_std
        if self.fixed_std is None:
            self.std_layer = original.std_layer
        self.std_min, self.std_max = original.std_min, original.std_max
        self.encoder_is_shared = original.encoder_is_shared
        self.action_dim = original.action_dim
        self.joint_scale = joint_scale

    def forward(self, observations, observation_features=None):
        encoded = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        if self.encoder_is_shared:
            encoded = encoded.detach()
        hidden = self.network(encoded)
        raw_mean = self.mean_layer(hidden)
        std = (
            self.std_layer(hidden).exp().clamp(self.std_min, self.std_max)
            if self.fixed_std is None
            else self.fixed_std.expand_as(raw_mean)
        )
        distribution = TanhMultivariateNormalDiag(loc=raw_mean, scale_diag=std)
        residual = distribution.rsample()
        reference = observations["observation.state"][..., -8:-1]
        action = torch.cat((reference + self.joint_scale * residual[..., :7], residual[..., 7:]), dim=-1)
        log_prob = distribution.log_prob(residual) - 7 * math.log(self.joint_scale)
        mean = raw_mean.tanh()
        absolute_mean = torch.cat((reference + self.joint_scale * mean[..., :7], mean[..., 7:]), dim=-1)
        # GaussianActorPolicy/Trainer expose pre-tanh absolute means.
        return action, log_prob, torch.atanh(absolute_mean.clamp(-1 + 1e-6, 1 - 1e-6))
