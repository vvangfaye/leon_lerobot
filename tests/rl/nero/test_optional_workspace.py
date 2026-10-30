from dataclasses import replace

import numpy as np
import pytest

from examples.nero_rl.config import JOINT_LIMITS
from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.learning import Trainer, contract
from examples.nero_rl.native_run import policy_start_rejection
from examples.nero_rl.run import mock_config


def test_disabling_xyz_gate_preserves_model_inputs_and_checkpoint(tmp_path):
    original = replace(mock_config(), action_mode="joint_target", hz=100)
    cfg = replace(original, policy_workspace_enabled=False)
    backend = MockHardware(cfg)
    backend.q[:3] = [-0.08, 0.09, -0.003]
    snapshot = backend.read()
    old_env, env = NeroEnv(original, backend, mock=True), NeroEnv(cfg, backend, mock=True)
    assert policy_start_rejection(snapshot, old_env) is not None
    assert policy_start_rejection(snapshot, env) is None
    old_obs, obs = old_env.observation(snapshot), env.observation(snapshot)
    np.testing.assert_array_equal(old_obs["observation.state"], obs["observation.state"])
    assert contract(original, True) == contract(cfg, True)
    trainer = Trainer(original, mock=True)
    trainer.save(tmp_path / "original.pt")
    restored = Trainer(cfg, mock=True)
    restored.load(tmp_path / "original.pt")
    np.testing.assert_allclose(
        trainer.act(old_obs, deterministic=True), restored.act(obs, deterministic=True), atol=1e-7
    )


def test_xyz_bounds_no_longer_clip_start_or_execution_but_speed_is_retained():
    cfg = replace(mock_config(), action_mode="joint_target", hz=100, policy_workspace_enabled=False)
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    backend.q[:3] = [-0.08, 0.09, -0.003]
    env.command_joints = backend.q.copy()
    goal = backend.q.copy()
    goal[:3] += [-0.02, 0.02, -0.02]
    before = backend.q.copy()
    _, _, _, _, info = env.step(cfg.encode_joint_target(goal, backend.width), autonomous=True)
    assert info["safety_rejection"] is None
    assert not info["workspace_limited"]
    assert info["tracking_diagnostics"]["workspace_enforced"] is False
    assert info["tracking_diagnostics"]["workspace_min_m"] is None
    assert backend.q[0] < before[0] and backend.q[1] > before[1] and backend.q[2] < before[2]
    assert np.max(abs(backend.q - before)) == pytest.approx(np.deg2rad(5) / 100)


def test_joint_and_invalid_feedback_checks_remain_when_xyz_gate_disabled():
    cfg = replace(mock_config(), action_mode="joint_target", policy_workspace_enabled=False)
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    snapshot = backend.read()
    snapshot.joints[6] = JOINT_LIMITS[6, 1] + 0.01
    assert "关节 7" in policy_start_rejection(snapshot, env)
    snapshot.joints[6] = 0
    snapshot.pose[2, 3] = np.nan
    assert "无效数值" in policy_start_rejection(snapshot, env)
    assert not env._inside(snapshot.pose)


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_workspace_switch_requires_boolean(value):
    with pytest.raises(ValueError, match="policy_workspace_enabled"):
        replace(mock_config(), policy_workspace_enabled=value).validate()
