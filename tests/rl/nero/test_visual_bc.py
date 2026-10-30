from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.learning import Trainer, tensors
from examples.nero_rl.run import mock_config
from examples.nero_rl.train_visual_bc import export_bc, residual_targets, supervised_loss, supervised_mean


def test_visual_bc_trains_cnn_despite_sac_actor_encoder_detach_and_exports_fresh_critics(tmp_path):
    cfg = replace(mock_config(), action_mode="joint_target", hz=100, policy_reference_residual=True)
    trainer = Trainer(cfg, mock=False)
    # Synthetic observations only; no hardware connection or movement.
    env = NeroEnv(cfg, MockHardware(cfg), mock=True)
    obs, _ = env.reset()
    batch = tensors({key: np.stack([value, value]) for key, value in obs.items()}, "cpu", batch=True)
    trainer.policy.train()
    target = torch.ones(2, 8) * 0.5
    optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
    initial = trainer.policy.actor.encoder.image_encoder.image_enc_layers[0].weight.detach().clone()
    optimizer.zero_grad()
    prediction = supervised_mean(trainer, batch)
    supervised_loss(prediction, target).backward()
    conv = trainer.policy.actor.encoder.image_encoder.image_enc_layers[0]
    assert conv.weight.grad is not None
    assert conv.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(initial, conv.weight)
    # Simulate old Q heads and counters that must not survive an encoder refit.
    with torch.no_grad():
        for parameter in trainer.algorithm.critic_ensemble.critics.parameters():
            parameter.fill_(7)
    trainer.updates = 1050
    trainer.algorithm._optimization_step = 1050
    trainer.policy.eval()
    expected = trainer.act(obs, deterministic=True)
    export_bc(trainer, tmp_path / "trained.pt")
    restored = Trainer(cfg, mock=False)
    restored.load(tmp_path / "trained.pt")
    np.testing.assert_allclose(expected, restored.act(obs, deterministic=True), atol=1e-7)
    assert restored.updates == restored.algorithm._optimization_step == 0
    assert all(not opt.state for opt in restored.algorithm.optimizers.values())
    assert any(
        not torch.all(parameter == 7) for parameter in restored.algorithm.critic_ensemble.critics.parameters()
    )
    with pytest.raises(ValueError, match="protect_pretrained_policy"):
        restored.update(1)


def test_residual_supervision_limits_joint_directions_and_keeps_gripper_absolute():
    reference = np.zeros((2, 8), dtype=np.float32)
    future = reference.copy()
    future[0, 0] = 0.01
    future[0, 1] = -0.01
    future[0, 2] = 0.00001  # Below the 0.02 degree deadband.
    future[:, 7] = [-0.7, 0.2]
    target = residual_targets(reference, future, 0.05 / 180)
    assert target[0, 0] == 1 and target[0, 1] == -1
    assert target[0, 2] == 0
    np.testing.assert_array_equal(target[:, 7], future[:, 7])
    assert future[0, 0] == pytest.approx(0.01)  # Input is unchanged.
