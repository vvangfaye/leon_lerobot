from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.learning import Trainer, contract, tensors
from examples.nero_rl.run import mock_config


def test_residual_actor_bounds_samples_and_deterministic_targets_without_changing_replay_contract(tmp_path):
    cfg = replace(mock_config(), action_mode="joint_target", hz=100, policy_reference_residual=True)
    assert contract(cfg, True) == contract(replace(cfg, policy_reference_residual=False), True)
    trainer = Trainer(cfg, mock=True)
    env = NeroEnv(cfg, MockHardware(cfg), mock=True)
    obs, _ = env.reset()
    reference = obs["observation.state"][-8:-1]
    for _ in range(5):
        action = trainer.act(obs)
        assert np.max(np.abs(action[:7] - reference)) <= cfg.policy_joint_speed_deg_s / cfg.hz / 180 + 1e-7
    action = trainer.act(obs, deterministic=True)
    assert np.max(np.abs(action[:7] - reference)) <= cfg.policy_joint_speed_deg_s / cfg.hz / 180 + 1e-7
    trainer.save(tmp_path / "checkpoint.pt")
    restored = Trainer(cfg, mock=True)
    restored.load(tmp_path / "checkpoint.pt")
    np.testing.assert_allclose(action, restored.act(obs, deterministic=True))
    with pytest.raises(ValueError, match="architecture"):
        Trainer(replace(cfg, policy_reference_residual=False), mock=True).load(tmp_path / "checkpoint.pt")
    output = trainer.policy(tensors(obs, "cpu"))
    assert torch.isfinite(output["log_prob"]).all()
    trainer.policy.zero_grad()
    output["action"].sum().backward()
    assert trainer.policy.actor.mean_layer.weight.grad is not None
    assert all(parameter.grad is None for parameter in trainer.policy.encoder_actor.parameters())
    expected = -4 + 7 * np.log(cfg.policy_joint_speed_deg_s / cfg.hz / 180)
    assert trainer.algorithm.target_entropy == pytest.approx(expected)


def test_residual_actor_runs_a_real_sac_update():
    cfg = replace(mock_config(), action_mode="joint_target", hz=100, policy_reference_residual=True)
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    trainer = Trainer(cfg, mock=True)
    obs, _ = env.reset()
    for i in range(20):
        action = trainer.act(obs)
        after, reward, done, truncated, info = env.step(
            action, autonomous=True, outcome="success" if i == 19 else None
        )
        trainer.replay.add(
            {
                "state": obs,
                "next_state": after,
                "action": info["executed_action"],
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "intervention": True,
            }
        )
        obs = after
    assert trainer.update(2)
