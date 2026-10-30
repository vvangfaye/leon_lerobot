"""Offline checks for the NERO RL contract, limits and real SAC updates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.nero_rl.config import vector
from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.hardware import NeroHardware
from examples.nero_rl.learning import MixedReplay, Trainer, load_episodes
from examples.nero_rl.run import mock_config, smoke


@pytest.fixture
def setup_env():
    cfg = mock_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    return cfg, backend, env


def test_constructor_does_not_enable(setup_env):
    _, backend, _ = setup_env
    assert not backend.prepared
    assert backend.commands == []


def test_bounds_store_projected_command_not_policy_request(setup_env):
    cfg, backend, env = setup_env
    env.reset()
    backend.q[0] = cfg.workspace_max_m[0] - cfg.translation_step_m / 4
    _, _, _, _, info = env.step(np.array([1, 0, 0, 0, 0]))
    assert info["executed_action"][0] == pytest.approx(0.25)
    assert info["requested_action"][0] == 1
    assert backend.commands[-1][0][0] == pytest.approx(cfg.workspace_max_m[0])


def test_ik_rejection_sends_neither_arm_nor_gripper(setup_env, monkeypatch):
    _, backend, env = setup_env
    env.reset()
    monkeypatch.setattr(backend, "inverse", lambda pose, seed: None)
    _, _, _, _, info = env.step(np.ones(5))
    assert info["safety_rejection"] == "ik"
    assert np.all(info["executed_action"] == 0)
    assert backend.commands == []


def test_joint_jump_rejected(setup_env, monkeypatch):
    _, backend, env = setup_env
    env.reset()
    monkeypatch.setattr(backend, "inverse", lambda pose, seed: seed + 0.3)
    _, _, _, _, info = env.step(np.zeros(5))
    assert info["safety_rejection"]
    assert backend.commands == []


@pytest.mark.parametrize("action", [np.array([np.nan, 0, 0, 0, 0]), np.ones(5) * 2, np.zeros(4)])
def test_invalid_action_cannot_move(setup_env, action):
    _, backend, env = setup_env
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert backend.commands == []


def test_only_human_success_rewards_and_time_limit_bootstraps(setup_env):
    cfg, backend, _ = setup_env
    cfg.episode_seconds = 0.1  # one step at 10 Hz
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    _, reward, done, truncated, _ = env.step(np.zeros(5))
    assert reward == 0 and not done and truncated
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(5))
    env.reset()
    commands = len(backend.commands)
    _, reward, done, truncated, info = env.step(np.ones(5), outcome="success")
    assert reward == 1 and done and not truncated
    assert len(backend.commands) == commands  # outcome button cannot issue another move
    assert np.all(info["executed_action"] == 0)


def test_live_requires_geometry_and_correct_channel():
    cfg = mock_config()
    with pytest.raises(ValueError, match="Verify TCP"):
        cfg.validate(live=True)
    cfg.channel = "can0"
    with pytest.raises(ValueError, match="right follower"):
        cfg.validate(live=True)


def test_live_reset_refuses_wrong_start_before_enable():
    cfg = replace(
        mock_config(),
        geometry_verified=True,
        tcp_offset_m=[0, 0, 0],
        start_joints_rad=[0.2, 0, 0, 0, 0, 0, 0],
        translation_step_m=0.001,
        max_joint_step_rad=np.deg2rad(0.5),
        hz=5,
    )
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend)
    with pytest.raises(ValueError, match="start pose"):
        env.reset()
    assert not backend.prepared
    assert backend.commands == []


def test_jog_between_episodes_does_not_change_episode_accounting(setup_env):
    _, backend, env = setup_env
    env.reset()
    env.finished, env.steps = True, 42
    env.jog(np.array([1, 0, 0, 0, 0]))
    assert env.finished and env.steps == 42
    assert len(backend.commands) == 1


def test_freshness_requires_advancing_timestamp_and_expires(monkeypatch):
    backend = NeroHardware(mock_config())
    now = [10.0]
    monkeypatch.setattr("examples.nero_rl.hardware.time.monotonic", lambda: now[0])
    packet = SimpleNamespace(timestamp=1, msg=object())
    assert not backend._fresh("j7", packet)
    assert not backend._fresh("j7", packet)
    packet.timestamp = 2
    assert backend._fresh("j7", packet)
    now[0] += 0.6
    assert not backend._fresh("j7", packet)


def test_replay_mixes_human_data_and_preserves_terminal_masks():
    replay = MixedReplay(8, 42)
    for i in range(8):
        replay.add(
            {
                "state": {"observation.state": np.array([i], np.float32)},
                "next_state": {"observation.state": np.array([i + 1], np.float32)},
                "action": np.array([float(i == 0)]),
                "reward": float(i == 0),
                "done": False,
                "truncated": True,
                "intervention": i == 0,
            }
        )
    batch = replay.sample(10, "cpu")
    assert (batch["action"] == 1).sum() >= 5
    assert batch["done"].sum() == 0
    assert batch["truncated"].sum() == 10


def test_real_sac_update_persistence_and_contract(tmp_path: Path):
    torch.set_num_threads(2)
    cfg = mock_config()
    result = smoke(cfg, tmp_path)
    assert result["updates"] == cfg.updates_per_episode
    assert result["checkpoint_restored"]
    assert all(np.isfinite(v) for v in result["losses"].values())
    trainer = Trainer(cfg, mock=True)
    trainer.load(tmp_path / "checkpoint.pt")
    count = load_episodes(tmp_path / "episodes", trainer.replay, cfg, mock=True)
    assert count > 1
    old_step = trainer.algorithm._optimization_step
    trainer.update(1)
    assert trainer.algorithm._optimization_step == old_step + 1
    assert all(opt.state for opt in trainer.algorithm.optimizers.values())
    changed = replace(cfg, translation_step_m=cfg.translation_step_m / 2)
    with pytest.raises(ValueError, match="Incompatible"):
        load_episodes(tmp_path / "episodes", MixedReplay(512, 0), changed, mock=True)
    with pytest.raises(ValueError, match="different geometry"):
        Trainer(cfg, mock=False).load(tmp_path / "checkpoint.pt")


def test_boolean_coordinates_rejected():
    with pytest.raises(ValueError):
        vector([True, 0, 0], 3)


def test_deterministic_action_is_squashed(monkeypatch):
    trainer = Trainer(mock_config(), mock=True)
    monkeypatch.setattr(trainer.policy, "forward", lambda obs: {"action_mean": torch.full((1, 5), 10.0)})
    action = trainer.act({"observation.state": np.zeros(17)}, deterministic=True)
    assert np.all(np.abs(action) <= 1)


def test_sdk_kinematics_uses_configured_tcp_without_connecting():
    pytest.importorskip("pyAgxArm")
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from pyAgxArm.utiles.tf import rpy_to_rot

    cfg = replace(mock_config(), tcp_offset_m=[0.175, 0, -0.0235])
    backend = NeroHardware(cfg)
    backend.mdh, backend.fk, backend.rpy_to_rot = get_mdh("nero"), fk_from_mdh, rpy_to_rot
    q = np.array([0, -0.8, 0, 1, 0, 0, 0], dtype=float)
    pose = backend.forward(q)
    flange_xyz = np.array(fk_from_mdh(backend.mdh, q.tolist())[:3])
    np.testing.assert_allclose(pose[:3, 3] - flange_xyz, pose[:3, :3] @ cfg.tcp_offset_m)
    target = pose.copy()
    target[0, 3] += 0.0001
    solved = backend.inverse(target, q)
    assert solved is not None
    assert np.max(np.abs(solved - q)) <= cfg.max_joint_step_rad
    assert np.linalg.norm(backend.forward(solved)[:3, 3] - target[:3, 3]) < 0.00025
    assert backend.arm is None
