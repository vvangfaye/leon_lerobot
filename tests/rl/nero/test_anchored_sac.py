from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.nero_rl.config import TaskConfig
from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.learning import Trainer, contract
from examples.nero_rl.run import mock_config


def setup_protected(tmp_path):
    torch.set_num_threads(2)
    original_cfg = replace(mock_config(), action_mode="joint_target", hz=100, policy_reference_residual=True)
    baseline = Trainer(original_cfg, mock=True)
    baseline.save(tmp_path / "bc.pt")
    cfg = replace(original_cfg, protect_pretrained_policy=True, critic_warmup_updates=2)
    trainer = Trainer(cfg, mock=True)
    trainer.load(tmp_path / "bc.pt")
    env = NeroEnv(cfg, MockHardware(cfg), mock=True)
    obs, _ = env.reset()
    for _ in range(20):
        action = trainer.act(obs)
        after, reward, done, truncated, info = env.step(action, autonomous=True)
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
    return trainer, obs


def test_critic_warmup_keeps_pretrained_actor_exact_and_resumes(tmp_path):
    trainer, obs = setup_protected(tmp_path)
    actor_before = {k: v.clone() for k, v in trainer.policy.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in trainer.algorithm.critic_ensemble.critics.state_dict().items()}
    expected = trainer.act(obs, deterministic=True)
    stats = trainer.update(2)
    assert stats["critic_warmup_remaining"] == 0
    assert all(torch.equal(v, actor_before[k]) for k, v in trainer.policy.actor.state_dict().items())
    assert any(
        not torch.equal(v, critic_before[k])
        for k, v in trainer.algorithm.critic_ensemble.critics.state_dict().items()
    )
    assert not trainer.algorithm.optimizers["actor"].state
    assert not trainer.algorithm.optimizers["temperature"].state
    trainer.save(tmp_path / "protected.pt")
    restored = Trainer(trainer.cfg, mock=True)
    restored.load(tmp_path / "protected.pt")
    assert restored.algorithm._optimization_step == restored.updates == 2
    np.testing.assert_array_equal(restored.act(obs, deterministic=True), expected)
    assert all(torch.equal(v, actor_before[k]) for k, v in restored.algorithm.reference.state_dict().items())


def test_actor_updates_after_warmup_but_encoder_and_teacher_stay_fixed(tmp_path):
    trainer, obs = setup_protected(tmp_path)
    trainer.update(2)
    actor_before = {k: v.clone() for k, v in trainer.policy.actor.state_dict().items()}
    teacher_before = {k: v.clone() for k, v in trainer.algorithm.reference.state_dict().items()}
    trainer.save(tmp_path / "protected.pt")
    stats = trainer.update(3)
    assert "loss_actor" in stats and stats["loss_bc_anchor"] >= 0
    state = trainer.policy.actor.state_dict()
    assert all(torch.equal(v, actor_before[k]) for k, v in state.items() if k.startswith("encoder."))
    assert any(not torch.equal(v, actor_before[k]) for k, v in state.items() if k.startswith("network."))
    assert all(torch.equal(v, teacher_before[k]) for k, v in trainer.algorithm.reference.state_dict().items())
    trainer.save(tmp_path / "protected.pt")
    assert len(list((tmp_path / "checkpoints").glob("protected_updates_00000002_*.pt"))) == 1
    restored = Trainer(trainer.cfg, mock=True)
    restored.load(tmp_path / "protected.pt")
    np.testing.assert_array_equal(restored.act(obs, deterministic=True), trainer.act(obs, deterministic=True))
    assert all(
        torch.equal(v, teacher_before[k]) for k, v in restored.algorithm.reference.state_dict().items()
    )


def test_protected_configuration_preserves_data_contract_and_rejects_bad_resume(tmp_path):
    trainer, _ = setup_protected(tmp_path)
    plain_cfg = replace(trainer.cfg, protect_pretrained_policy=False)
    assert contract(trainer.cfg, True) == contract(plain_cfg, True)
    trainer.save(tmp_path / "protected.pt")
    with pytest.raises(ValueError, match="requires protect_pretrained"):
        Trainer(plain_cfg, mock=True).load(tmp_path / "protected.pt")
    legacy = Trainer(plain_cfg, mock=True)
    legacy.updates = 200
    legacy.save(tmp_path / "unanchored.pt")
    with pytest.raises(ValueError, match="original BC"):
        Trainer(trainer.cfg, mock=True).load(tmp_path / "unanchored.pt")


@pytest.mark.parametrize(
    "changes",
    [
        {"protect_pretrained_policy": "true"},
        {"protect_pretrained_policy": True},
        {"critic_warmup_updates": -1},
        {"bc_anchor_weight": float("nan")},
        {"rl_actor_learning_rate": 0},
    ],
)
def test_invalid_protection_config(changes):
    with pytest.raises(ValueError):
        replace(TaskConfig(), **changes).validate()
