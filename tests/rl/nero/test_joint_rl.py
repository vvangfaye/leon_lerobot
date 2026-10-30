import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nero_rl.config import STREAM_TARGET_LEAD_RAD
from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.hardware import NeroHardware, Snapshot, StreamTrackingError
from examples.nero_rl.learning import Trainer, load_episodes, save_episode
from examples.nero_rl.master_input import MasterInput, MasterSample
from examples.nero_rl.run import mock_config, transition
from examples.nero_rl.web_control import WebControls


def joint_config():
    return replace(mock_config(), action_mode="joint_delta", hz=100, max_joint_step_rad=0.005)


def test_joint_demos_and_policy_use_same_eight_action_contract(tmp_path):
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    obs, _ = env.reset()
    rows = []
    for i in range(20):
        after, reward, done, trunc, info = env.step(np.full(8, 0.1), outcome="success" if i == 19 else None)
        assert info["executed_action"].shape == (8,)
        rows.append(transition(obs, after, info, reward, done, trunc, True, "right_master"))
        obs = after
    save_episode(tmp_path / "episodes", rows, cfg, mock=True, kind="record")
    trainer = Trainer(cfg, mock=True)
    assert load_episodes(tmp_path / "episodes", trainer.replay, cfg, mock=True) == 20
    assert trainer.act(obs).shape == (8,)
    assert trainer.update(1)
    trainer.save(tmp_path / "checkpoint.pt")
    restored = Trainer(cfg, mock=True)
    restored.load(tmp_path / "checkpoint.pt")
    np.testing.assert_allclose(trainer.act(obs, deterministic=True), restored.act(obs, deterministic=True))


def test_policy_speed_limited_without_slowing_human_action():
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    before = backend.q.copy()
    env.step(np.ones(8), autonomous=True)
    assert np.max(np.abs(backend.q - before)) == pytest.approx(np.deg2rad(5) / 100)
    before = backend.q.copy()
    env.step(np.ones(8), autonomous=False)
    assert np.max(np.abs(backend.q - before)) == pytest.approx(cfg.max_joint_step_rad)


def test_joint_workspace_limits_outward_target_without_pausing_and_allows_return():
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    backend.q[0] = cfg.workspace_max_m[0] - 0.0001
    env.hold()
    _, _, _, _, info = env.step(np.ones(8))
    assert info["safety_rejection"] is None
    assert info["workspace_limited"]
    assert len(backend.commands) == 1
    assert backend.q[0] <= cfg.workspace_max_m[0]
    assert 0 <= info["executed_action"][0] < 0.03
    before = backend.q[0]
    inward = np.zeros(8)
    inward[0] = -1
    _, _, _, _, info = env.step(inward)
    assert info["safety_rejection"] is None
    assert not info["workspace_limited"]
    assert backend.q[0] == pytest.approx(before - cfg.max_joint_step_rad)


def test_invalid_path_to_previous_joint_target_still_requires_a_hold(monkeypatch):
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    forward = backend.forward

    def obstructed_path(q):
        pose = forward(q)
        if 0.004 < q[3] < 0.016:
            pose[1, 3] = cfg.workspace_max_m[1] + 0.01
        return pose

    monkeypatch.setattr(backend, "forward", obstructed_path)
    env.command_joints[3] = 0.02
    _, _, _, _, info = env.step(np.zeros(8))
    assert info["safety_rejection"] == "workspace"
    assert backend.commands == []


def test_captured_submillimeter_y_boundary_event_projects_to_a_valid_command():
    pytest.importorskip("pyAgxArm")
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from pyAgxArm.utiles.tf import rpy_to_rot

    # Captured stop: the measured TCP was inside, but the next target was
    # 0.388 mm beyond the lower Y bound. This test never opens a CAN device.
    cfg = replace(
        joint_config(),
        workspace_min_m=[-0.46602579628733154, -0.32159941320654156, 0.01930175590619496],
        workspace_max_m=[-0.2012554326043779, 0.009008922564574848, 0.28316276019179876],
        tcp_offset_m=[0.17339324906847453, 0.007790625784292654, -0.017977765583954267],
    )
    backend = NeroHardware(cfg)
    backend.mdh, backend.fk, backend.rpy_to_rot = get_mdh("nero"), fk_from_mdh, rpy_to_rot
    env = NeroEnv(cfg, backend, mock=True)
    measured = np.array(
        [
            -0.007173303225696695,
            0.6293482749766353,
            1.1439236983421235,
            1.32042884559631,
            -0.6689148891193467,
            0.18374826364996302,
            0.8999615754983561,
        ]
    )
    reference = np.array(
        [
            0.001454776850551473,
            0.6258052565950868,
            1.1481648484244695,
            1.323814784345179,
            -0.6295539077302101,
            0.147286018528134,
            0.8594001236820079,
        ]
    )
    desired = np.array(
        [
            0.0027304710683067114,
            0.6258052565950868,
            1.1481648484244695,
            1.323814784345179,
            -0.6264821282467001,
            0.14421423904462394,
            0.8563283441984979,
        ]
    )
    target, fraction = env._joint_workspace_target(measured, reference, desired)
    assert target is not None
    assert 0 < fraction < 1
    assert env._joint_path_inside(measured, target)
    assert backend.forward(target)[1, 3] >= cfg.workspace_min_m[1] + 0.00049


def test_master_joint_input_preserves_all_seven_joints_and_gripper(monkeypatch):
    cfg = joint_config()
    master = MasterInput(cfg)
    q = np.arange(7) * 0.0001
    sample = MasterSample(np.eye(4), 0.061, 1, q)
    monkeypatch.setattr(master, "read", lambda: sample)
    action = master.joint_action(np.zeros(7), 0.06)
    np.testing.assert_allclose(action, np.r_[q, 0.001] / cfg.scales, atol=1e-6)


def test_streaming_backend_checks_real_tracking_and_never_waits_for_settle(monkeypatch):
    cfg = joint_config()
    backend = NeroHardware(cfg)
    backend.prepared = True
    commands = []
    backend.runtime = object()
    backend.status_api = SimpleNamespace(require_ready_for_motion=lambda *_: None)
    backend.arm = SimpleNamespace(move_js=lambda q: commands.append(q))
    backend.gripper = SimpleNamespace(move_gripper_m=lambda **_: None)
    monkeypatch.setattr(backend, "read", lambda **_: Snapshot(np.zeros(7), 0.06, np.eye(4), {}))
    backend.servo_command(np.full(7, 0.001), 0.06)
    assert len(commands) == 1
    # Slow feedback must not pin every command to the first small increment.
    backend.servo_command(np.full(7, 0.004), 0.06)
    assert len(commands) == 2
    backend.last_servo_target = np.full(7, 0.1)
    with pytest.raises(RuntimeError, match="failed to track"):
        backend.servo_command(backend.last_servo_target.copy(), 0.06)
    assert len(commands) == 2


def test_joint_targets_wait_for_stationary_feedback_then_pause_on_persistent_stall(monkeypatch):
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    now = [0.0]
    monkeypatch.setattr("examples.nero_rl.environment.time.monotonic", lambda: now[0])
    env.reset()
    monkeypatch.setattr(backend, "servo_command", lambda q, w: backend.commands.append((q.copy(), w)))
    action = np.zeros(8)
    action[3] = 1
    for _ in range(10):
        obs, _, _, _, info = env.step(action)
        assert info["safety_rejection"] is None
        assert 0 <= info["executed_action"][3] <= 1
    assert backend.q[3] == 0
    assert backend.commands[-1][0][3] == pytest.approx(STREAM_TARGET_LEAD_RAD)
    assert info["tracking_limited"]
    assert obs["observation.state"].shape == (25,)
    now[0] = 1.1
    _, _, _, _, info = env.step(action)
    assert info["safety_rejection"] == "tracking_stall"
    assert info["tracking_diagnostics"]["stalled_joints"] == [4]
    assert len(backend.commands) == 10
    env.hold()
    np.testing.assert_array_equal(env.joint_reference()[0], backend.q)


def test_slow_but_progressing_follower_reduces_velocity_without_repeated_pauses(monkeypatch):
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    now = [0.0]
    monkeypatch.setattr("examples.nero_rl.environment.time.monotonic", lambda: now[0])
    env.reset()

    def slowly_follow(q, width):
        backend.commands.append((q.copy(), width))
        backend.q += np.clip(q - backend.q, -0.0008, 0.0008)

    monkeypatch.setattr(backend, "servo_command", slowly_follow)
    action = np.zeros(8)
    action[3] = 1
    limited = 0
    for _ in range(300):
        now[0] += 0.01
        _, _, _, _, info = env.step(action)
        assert info["safety_rejection"] is None
        assert np.max(np.abs(env.command_joints - backend.q)) <= STREAM_TARGET_LEAD_RAD + 1e-9
        limited += int(info["tracking_limited"])
    assert limited > 100
    assert np.rad2deg(backend.q[3]) > 10


def test_actual_tracking_error_above_hard_limit_still_pauses():
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    env.command_joints[3] += np.deg2rad(3.1)
    _, _, _, _, info = env.step(np.zeros(8))
    assert info["safety_rejection"] == "tracking_error"
    assert backend.commands == []


def test_hardware_checks_next_target_tracking_and_command_step(monkeypatch):
    cfg = joint_config()
    backend = NeroHardware(cfg)
    backend.prepared = True
    sent = []
    backend.runtime = object()
    backend.status_api = SimpleNamespace(require_ready_for_motion=lambda *_: None)
    backend.arm = SimpleNamespace(move_js=lambda q: sent.append(q))
    backend.gripper = SimpleNamespace(move_gripper_m=lambda **_: None)
    monkeypatch.setattr(backend, "read", lambda **_: Snapshot(np.zeros(7), 0.06, np.eye(4), {}))
    backend.last_servo_target = np.full(7, np.deg2rad(2.9))
    with pytest.raises(RuntimeError, match="tracking allowance"):
        backend.servo_command(backend.last_servo_target + 0.004, 0.06)
    with pytest.raises(RuntimeError, match="one cycle"):
        backend.servo_command(backend.last_servo_target + 0.006, 0.06)
    assert sent == []
    backend.hold()
    np.testing.assert_array_equal(backend.last_servo_target, np.zeros(7))
    assert len(sent) == 1


def test_late_tracking_error_rejects_without_advancing_the_saved_reference(monkeypatch):
    cfg = joint_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    reference = env.joint_reference()

    def late_rejection(*_):
        raise StreamTrackingError("Feedback changed after planning")

    monkeypatch.setattr(backend, "servo_command", late_rejection)
    _, _, _, _, info = env.step(np.full(8, 0.1))
    assert info["safety_rejection"] == "tracking_error"
    np.testing.assert_array_equal(env.joint_reference()[0], reference[0])
    np.testing.assert_array_equal(info["executed_action"], np.zeros(8))
    assert backend.commands == []


def test_web_console_authentication_and_immediate_stop(tmp_path):
    controls = WebControls("127.0.0.1", 0, tmp_path / "url")
    client = controls.app.test_client()
    assert client.get("/status").status_code == 403
    assert client.get("/?token=" + controls.token).status_code == 200
    controls.publish(phase="waiting")
    assert client.post("/key", json={"key": "n"}).status_code == 200
    assert controls.poll() == ["n"]
    assert client.post("/key", json={"key": "a"}, headers={"Origin": "http://evil.test"}).status_code == 403
    assert client.post("/key", json={"key": "x"}).status_code == 200
    with pytest.raises(KeyboardInterrupt):
        controls.poll()


def test_private_control_link_survives_restart_of_same_experiment(tmp_path):
    path = tmp_path / "control_url.txt"
    original = WebControls("0.0.0.0", 5001, path, "192.168.112.232")
    path.write_text(f"http://192.168.112.232:5001/?token={original.token}\n")
    path.chmod(0o600)
    restored = WebControls("0.0.0.0", 5001, path, "192.168.112.232")
    assert restored.token == original.token
    other_host = WebControls("0.0.0.0", 5001, path, "127.0.0.1")
    assert other_host.token != original.token


def test_web_console_records_close_and_save_as_distinct_commands(tmp_path):
    controls = WebControls("127.0.0.1", 0, tmp_path / "url")
    client = controls.app.test_client()
    page = client.get("/?token=" + controls.token).data.decode()
    assert "关闭采集服务（网页将断开）" in page
    assert "保存并退出" not in page
    controls.publish(phase="human", mode="record", episode=1, steps=526)
    assert client.post("/key", json={"key": "0"}).status_code == 200
    assert client.post("/key", json={"key": "q"}).status_code == 200
    records = [json.loads(line) for line in (tmp_path / "control_events.jsonl").read_text().splitlines()]
    assert [record["key"] for record in records] == ["0", "q"]
    assert all(record["steps"] == 526 and record["mode"] == "record" for record in records)
    assert controls.token not in (tmp_path / "control_events.jsonl").read_text()
    assert controls.poll() == ["0", "q"]


def test_web_console_logging_failure_does_not_prevent_stop(tmp_path):
    controls = WebControls("127.0.0.1", 0, tmp_path / "url")
    controls.event_log = tmp_path  # Opening a directory for append must fail.
    client = controls.app.test_client()
    client.get("/?token=" + controls.token)
    controls.publish(phase="human")
    assert client.post("/key", json={"key": "p"}).status_code == 200
    assert controls.poll() == ["p"]
    assert client.post("/key", json={"key": "x"}).status_code == 200
    assert controls.abort.is_set()


def test_interactive_joint_recording_keeps_eight_dimensional_terminal_action(tmp_path, monkeypatch):
    from examples.nero_rl import run

    class Keys:
        def __enter__(self):
            self.abort = SimpleNamespace(is_set=lambda: False)
            # A fresh process must allow unrecorded master control to return to
            # the required start, without first beginning an episode there.
            self.events = iter([["i"], ["p"], ["n"], [], ["1"]])
            return self

        def poll(self):
            return next(self.events, ["q"])

        def __exit__(self, *_):
            pass

    class Master:
        prepared = False

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            pass

        def prepare(self):
            self.prepared = True

        def rebase(self):
            pass

        def joint_action(self, *_):
            return np.full(8, 0.1)

        def close(self):
            pass

    monkeypatch.setattr(run, "Keyboard", Keys)
    monkeypatch.setattr(run, "MasterInput", Master)
    monkeypatch.setattr(run.WorkbenchGuard, "check", lambda _: None)
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        mode="record",
        mock=True,
        human_input="master",
        master_translation_scale=0.25,
        workbench_url="http://127.0.0.1:5000",
        output=tmp_path,
        data=[],
        resume=None,
        episodes=1,
        preview=False,
        web=False,
    )
    run.run_interactive(args, joint_config())
    with np.load(next((tmp_path / "episodes").glob("*.npz"))) as data:
        assert data["action"].shape == (2, 8)
        assert data["reward"][-1] == 1
        np.testing.assert_array_equal(data["action"][-1], np.zeros(8))
