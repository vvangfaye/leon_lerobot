import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nero_rl.config import JOINT_LIMITS
from examples.nero_rl.environment import MockHardware, NeroEnv
from examples.nero_rl.hardware import NeroHardware
from examples.nero_rl.learning import Trainer, contract, load_episodes, save_episode
from examples.nero_rl.native_recording import NativeRecorder, NativeTick
from examples.nero_rl.native_run import (
    policy_observation,
    policy_start_rejection,
    prepare_current_pose_probe,
    send_native,
)
from examples.nero_rl.run import mock_config


def native_config():
    return replace(
        mock_config(),
        action_mode="joint_target",
        hz=100,
        max_joint_step_rad=0.005,
        gripper_max_m=0.09,
    )


def test_lowering_tcp_does_not_hide_remaining_x_start_violation():
    cfg = replace(
        native_config(),
        workspace_min_m=[-0.466, -0.322, 0.019],
        workspace_max_m=[-0.201, 0.009, 0.283],
    )
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    snap = backend.read()
    snap.pose[:3, 3] = [-0.492, -0.063, 0.242]
    message = policy_start_rejection(snap, env)
    assert "X=-49.2 cm" in message
    assert "需增大至少 2.6 cm" in message
    assert "Z=" not in message
    snap.pose[0, 3] = -0.46
    assert policy_start_rejection(snap, env) is None
    snap.joints[6] = JOINT_LIMITS[6, 1] + np.deg2rad(1)
    assert "关节 7=" in policy_start_rejection(snap, env)


@pytest.mark.parametrize("invalid_field", ["pose", "joints"])
def test_invalid_start_feedback_cannot_enter_policy(invalid_field):
    cfg = native_config()
    backend = MockHardware(cfg)
    snap = backend.read()
    getattr(snap, invalid_field).flat[-1 if invalid_field == "joints" else 3] = np.nan
    assert "无效数值" in policy_start_rejection(snap, NeroEnv(cfg, backend, mock=True))


def test_current_pose_probe_checks_before_enable_and_has_no_master_dependency():
    cfg = native_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    cameras = SimpleNamespace(ready=lambda _: True)
    backend.q[0] = 0.2
    with pytest.raises(ValueError, match="未启动策略"):
        prepare_current_pose_probe(backend, env, cameras)
    assert not backend.prepared and not backend.commands
    backend.q[0] = 0
    with pytest.raises(ValueError, match="相机"):
        prepare_current_pose_probe(backend, env, SimpleNamespace(ready=lambda _: False))
    assert not backend.prepared
    snap = prepare_current_pose_probe(backend, env, cameras)
    assert backend.prepared and not backend.commands
    np.testing.assert_array_equal(snap.joints, backend.q)


@pytest.mark.parametrize("stamps", [{}, {"head": 9.0}, {"head": float("nan")}])
def test_policy_cannot_use_missing_or_stale_camera_capture(stamps):
    cfg = native_config()
    backend = MockHardware(cfg)
    with pytest.raises(ValueError, match="相机"):
        policy_observation(
            NeroEnv(cfg, backend, mock=True),
            backend.read(),
            (10.0, {"head": np.zeros((84, 84, 3), dtype=np.uint8)}, stamps),
        )


def test_separate_policy_workspace_keeps_checkpoint_observations_and_enforces_new_boundary(tmp_path):
    original = native_config()
    cfg = replace(
        original,
        policy_workspace_min_m=[-0.12, -0.12, -0.002],
        policy_workspace_max_m=[0.12, 0.12, 0.1],
    )
    backend = MockHardware(cfg)
    old_env = NeroEnv(original, backend, mock=True)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    backend.q[0] = 0.08  # Outside the old workspace but inside the new execution bounds.
    env.command_joints = backend.q.copy()
    snap = backend.read()
    old_env.command_joints, old_env.command_width = env.command_joints.copy(), env.command_width
    assert not old_env._inside(snap.pose)
    assert policy_start_rejection(snap, env) is None
    np.testing.assert_array_equal(
        old_env.observation(snap)["observation.state"], env.observation(snap)["observation.state"]
    )
    assert contract(original, True) == contract(cfg, True)
    trainer = Trainer(original, mock=True)
    trainer.save(tmp_path / "original.pt")
    restored = Trainer(cfg, mock=True)
    restored.load(tmp_path / "original.pt")
    obs = env.observation(snap)
    np.testing.assert_array_equal(trainer.act(obs, deterministic=True), restored.act(obs, deterministic=True))
    goal = backend.q.copy()
    goal[0] += 0.01
    env.step(cfg.encode_joint_target(goal, backend.width), autonomous=True)
    assert 0.08 < backend.q[0] < 0.081
    backend.q[0] = env.hi[0] - 0.0001
    env.command_joints = backend.q.copy()
    env.step(cfg.encode_joint_target(goal + 0.1, backend.width), autonomous=True)
    assert backend.q[0] <= env.hi[0]
    backend.q[0] = env.hi[0] + 0.001
    sent = len(backend.commands)
    with pytest.raises(RuntimeError, match="left task workspace"):
        env.step(cfg.encode_joint_target(goal, backend.width), autonomous=True)
    assert len(backend.commands) == sent


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_workspace_min_m": [-0.1, -0.1, 0]},
        {"policy_workspace_max_m": [0.1, 0.1, 0.1]},
        {"policy_workspace_min_m": [0.1, 0, 0], "policy_workspace_max_m": [-0.1, 0.1, 0.1]},
        {"policy_workspace_min_m": [np.nan, 0, 0], "policy_workspace_max_m": [0.1, 0.1, 0.1]},
    ],
)
def test_separate_policy_workspace_rejects_incomplete_or_invalid_bounds(changes):
    with pytest.raises(ValueError):
        replace(native_config(), **changes).validate()


def native_backend():
    backend = NeroHardware(native_config())
    sent, grips = [], []
    backend.arm = SimpleNamespace(
        _joint_limits_enabled=True,
        _config={"joint_limits": {f"joint{i + 1}": pair.tolist() for i, pair in enumerate(JOINT_LIMITS)}},
        move_js=lambda q: sent.append(q),
    )
    backend.gripper = SimpleNamespace(move_gripper_m=lambda **kw: grips.append(kw))
    backend.runtime = SimpleNamespace(channels={"can1": {"arm": backend.arm, "gripper": backend.gripper}})
    return backend, sent, grips


def make_tick(index, previous, command):
    timestamp = 10 + index / 100
    return NativeTick(
        timestamp=timestamp,
        joints=previous[:7].copy(),
        width=float(previous[7]),
        previous_command=previous.copy(),
        command=command.copy(),
        requested=command.copy(),
        source="right_master_vla",
        images={"head": np.full((84, 84, 3), index, dtype=np.uint8)},
        camera_stamps={"head": timestamp},
    )


def test_native_sender_uses_vla_absolute_command_without_rl_filter(monkeypatch):
    backend, sent, grips = native_backend()
    monkeypatch.setattr(backend, "forward", lambda *_: pytest.fail("FK entered native control path"))
    monkeypatch.setattr(backend, "servo_command", lambda *_: pytest.fail("RL filter entered native path"))
    q = [0.2, -0.3, 0.4, 1.0, -0.2, 0.3, 2.0]
    readings = {"主臂关节": q, "主臂夹爪": [0.1]}
    executed, requested = send_native(backend, readings)
    assert sent == [q]  # SDK receives the unfiltered master target, like VLA capture.
    assert grips == [{"value": 0.09, "force": 3.0}]
    assert requested[6] == 2.0
    assert executed[6] == pytest.approx(np.pi / 2, abs=2e-5)  # SDK's own clamp is logged.
    assert backend.last_servo_width == 0.09


def test_native_absolute_transitions_train_and_preserve_command_feedback_pairing(tmp_path, monkeypatch):
    cfg = native_config()
    recorder = NativeRecorder(cfg, MockHardware(cfg), mock=True)
    previous = np.array([0, 0, 0.03, 0, 0, 0, 0, 0.06], dtype=float)
    try:
        recorder.start()
        commands = []
        for i in range(20):
            command = previous.copy()
            command[3] += 0.04  # 2.3 degrees/tick: must never be clipped to the old 0.5 degree limit.
            commands.append(command.copy())
            assert recorder.submit(make_tick(i, previous, command))
            previous = command
        rows = recorder.finish(make_tick(20, previous, previous), "success", truncated=False)
        assert recorder.error is None
        assert len(rows) == 20
        assert rows[-1]["reward"] == 1 and rows[-1]["done"]
        assert not rows[-1]["truncated"]
        np.testing.assert_allclose(
            rows[0]["next_state"]["observation.state"][-8:], cfg.encode_joint_target(commands[0][:7], 0.06)
        )
        for row, command in zip(rows, commands, strict=True):
            q, width = cfg.decode_joint_target(row["action"])
            np.testing.assert_allclose(q, command[:7], atol=1e-7)
            assert width == pytest.approx(command[7])
            assert row["elapsed_s"] == pytest.approx(0.01)
            assert row["action_source"] == "right_master_vla"
        path = save_episode(tmp_path, rows, cfg, mock=True, kind="record")
        with np.load(path) as data:
            assert data["state/observation.state"].shape == (20, 25)
            assert data["control_timestamp"].shape == (20,)
            assert data["camera_timestamps"].shape == (20, 1)
        trainer = Trainer(cfg, mock=True)
        # Resuming must not decompress complete image arrays for every row.
        original_load = np.load
        reads = {}

        class CountedArchive:
            def __init__(self, *args, **kwargs):
                self.archive = original_load(*args, **kwargs)
                self.files = self.archive.files

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.archive.close()

            def __getitem__(self, key):
                reads[key] = reads.get(key, 0) + 1
                return self.archive[key]

        with monkeypatch.context() as patch:
            patch.setattr(np, "load", CountedArchive)
            assert load_episodes(tmp_path, trainer.replay, cfg, mock=True) == 20
        assert reads and all(count == 1 for count in reads.values())
        assert trainer.update(1)
        trainer.save(tmp_path / "checkpoint.pt")
        restored = Trainer(cfg, mock=True)
        restored.load(tmp_path / "checkpoint.pt")
        np.testing.assert_allclose(
            trainer.act(rows[-1]["next_state"], deterministic=True),
            restored.act(rows[-1]["next_state"], deterministic=True),
        )
        with pytest.raises(ValueError, match="Incompatible"):
            load_episodes(tmp_path, trainer.replay, replace(cfg, action_mode="joint_delta"), mock=True)
    finally:
        recorder.close()


def test_stale_image_cannot_turn_an_incomplete_recording_into_success():
    cfg = native_config()
    recorder = NativeRecorder(cfg, MockHardware(cfg), mock=True)
    command = np.array([0, 0, 0.03, 0, 0, 0, 0, 0.06], dtype=float)
    try:
        recorder.start()
        recorder.submit(make_tick(0, command, command))
        recorder.submit(make_tick(1, command, command))
        stale = make_tick(2, command, command)
        stale.camera_stamps["head"] -= 1
        recorder.submit(stale)
        rows = recorder.finish(make_tick(3, command, command), "success", truncated=False)
        assert "Stale image" in recorder.error
        assert len(rows) == 1
        assert rows[-1]["reward"] == 0
        assert rows[-1]["truncated"]
    finally:
        recorder.close()


def test_relative_takeover_is_recorded_as_demonstration_with_absolute_actions():
    cfg = native_config()
    recorder = NativeRecorder(cfg, MockHardware(cfg), mock=True)
    command = np.array([0, 0, 0.03, 0, 0, 0, 0, 0.06], dtype=float)
    first = make_tick(0, command, command)
    first.source = "right_master_relative"
    try:
        recorder.start()
        recorder.submit(first)
        rows = recorder.finish(make_tick(1, command, command), "success", truncated=False)
        assert rows[0]["intervention"]
        assert rows[0]["action_source"] == "right_master_relative"
        np.testing.assert_allclose(rows[0]["action"], cfg.encode_joint_target(command[:7], command[7]))
    finally:
        recorder.close()


def test_recording_backpressure_does_not_filter_native_follow(monkeypatch):
    cfg = native_config()
    recorder = NativeRecorder(cfg, MockHardware(cfg), mock=True, capacity=1)
    entered, release = threading.Event(), threading.Event()
    consume = recorder._consume

    def slow_consumer(tick):
        entered.set()
        assert release.wait(2)
        consume(tick)

    monkeypatch.setattr(recorder, "_consume", slow_consumer)
    command = np.array([0, 0, 0.03, 0, 0, 0, 0, 0.06], dtype=float)
    try:
        recorder.start()
        assert recorder.submit(make_tick(0, command, command))
        assert entered.wait(1)
        assert recorder.submit(make_tick(1, command, command))
        assert not recorder.submit(make_tick(2, command, command))
        backend, sent, _ = native_backend()
        readings = {"主臂关节": [0.1] * 7, "主臂夹爪": [0.06]}
        send_native(backend, readings)
        assert sent == [[0.1] * 7]
    finally:
        release.set()
        recorder.close()


def test_policy_absolute_targets_keep_autonomous_speed_and_gripper_limits():
    cfg = native_config()
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    env.reset()
    goal = backend.q.copy()
    goal[3] = 1
    request = cfg.encode_joint_target(goal, 0.09)
    with pytest.raises(ValueError, match="native sender"):
        env.step(request)
    before = backend.q.copy()
    _, _, _, _, info = env.step(request, autonomous=True)
    assert backend.q[3] - before[3] == pytest.approx(np.deg2rad(5) / 100)
    target, width = cfg.decode_joint_target(info["executed_action"])
    np.testing.assert_allclose(target, backend.q, atol=1e-7)
    for _ in range(20):
        env.step(request, autonomous=True)
    assert backend.width <= 0.07


@pytest.mark.parametrize("web", [False, True])
@pytest.mark.parametrize("mode", ["record", "eval"])
@pytest.mark.parametrize("current_pose", [False, True])
@pytest.mark.parametrize("handoff", [False, True])
@pytest.mark.parametrize("active_sync", [False, True])
def test_native_interactive_recording_keeps_following_outside_policy_box(
    tmp_path, monkeypatch, web, mode, current_pose, handoff, active_sync
):
    if current_pose and mode == "record":
        pytest.skip("Current-pose probes are evaluation only")
    if handoff and mode != "eval":
        pytest.skip("Policy takeover is evaluation only in this fixture")
    if active_sync and mode != "eval":
        pytest.skip("Active sync is only used for train/eval")
    import time

    from examples.nero_rl import native_run
    from examples.nero_rl.run import finish_episode

    instances = []
    consoles = []
    takeovers = []
    sync_instances = []

    class Sync:
        def __init__(self, *_):
            self.aligning = self.drive_ready = self.running = False
            self.diagnostics = {}
            self.updates = 0
            self.stops = 0
            sync_instances.append(self)

        def begin(self, snap, *, align):
            assert not align
            self.running = self.drive_ready = True

        def update(self, snap):
            assert self.running
            self.updates += 1

        def stop(self):
            self.running = self.aligning = False
            self.stops += 1

        def yield_to_human(self):
            assert not self.running  # Hold must occur before switching to teaching.
            self.drive_ready = False

    class Backend(MockHardware):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.last_servo_target = self.last_servo_width = None
            self.last_frames = {}
            self.arm = SimpleNamespace(
                _joint_limits_enabled=False,
                move_js=lambda q: setattr(self, "q", np.array(q)),
            )
            self.gripper = SimpleNamespace(move_gripper_m=lambda **kw: setattr(self, "width", kw["value"]))
            self.runtime = SimpleNamespace(channels={"can1": {"arm": self.arm, "gripper": self.gripper}})
            self.status_api = SimpleNamespace(inspect_status=lambda *_: {"mock": True})
            self.cameras = {"head": object()}
            instances.append(self)

        def native_executed_target(self, action):
            return NeroHardware.native_executed_target(self, action)

        def hold(self):
            snap = self.read()
            self.last_servo_target = snap.joints.copy()
            if self.last_servo_width is None:
                self.last_servo_width = snap.width
            return snap

    class Master:
        def __init__(self, _cfg):
            self.prepared = False

        def open(self):
            pass

        def close(self):
            pass

    class EvalTrainer:
        def __init__(self, *_args, **_kwargs):
            self.replay = SimpleNamespace(demos=[])
            self.updates = 17

        def load(self, _path):
            pass

        def act(self, observation, **_kwargs):
            assert np.all(observation["observation.images.head"] == 123)
            if active_sync:
                assert sync_instances[0].running and sync_instances[0].updates > 0
            action = observation["observation.state"][-8:].copy()
            action[3] += 0.0001
            return action

    class Cache:
        def __init__(self, backend):
            self.backend = backend
            self.frame = np.full((84, 84, 3), 123, dtype=np.uint8)

        def start(self):
            pass

        @property
        def latest(self):
            return {"head": self.frame}, {"head": time.perf_counter()}

        def ready(self, _names):
            return True

        def close(self):
            pass

    class Feedback:
        def __init__(self, backend):
            self.backend = backend
            self.count = 0

        def read(self, _now):
            self.count += 1
            target = self.backend.q.copy()
            target[0] = 0.2 if mode == "record" else 0.0
            target[3] += 0.06  # Also beyond the previous 3 degree tracking threshold.
            return {
                "主臂关节": target.tolist(),
                "主臂夹爪": [0.08],
                "从臂关节": self.backend.q.tolist(),
                "从臂夹爪": [self.backend.width],
            }

    def engage(backend, _master, _keys):
        assert not current_pose, "Current-pose evaluation must not engage or align the master"
        backend.prepare()
        _master.prepared = True
        backend.hold()
        return Feedback(backend), np.r_[backend.q, backend.width]

    def relative_engage(backend, _master, _keys):
        from examples.nero_rl.native_takeover import NativeTakeover

        if active_sync:
            assert not sync_instances[0].running
        held = backend.hold()
        feedback = Feedback(backend)
        anchor = np.r_[held.joints, backend.last_servo_width]
        takeovers.append(anchor)
        return feedback, anchor, NativeTakeover(feedback.read(0), anchor)

    class Keys:
        def __init__(self, *_args, **_kwargs):
            self.published = {}
            self.messages = []
            consoles.append(self)

        def __enter__(self):
            self.abort, self.cancel = threading.Event(), threading.Event()
            self.count = 0
            return self

        def poll(self):
            self.count += 1
            events = {1: ["i"], 3: ["n"], 35: ["1"], 37: ["q"]}
            if current_pose:
                events = {1: ["b"], 35: ["1"], 37: ["q"]}
            elif mode == "eval":
                events[5] = ["a"]
            if handoff:
                events.pop(35, None)
                events.pop(37, None)
                events.update({28: ["i"], 60: ["1"], 62: ["q"]})
            return events.get(self.count, [])

        def publish(self, **state):
            self.published.update(state)
            self.messages.append(state.get("message", ""))

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(native_run, "NeroHardware", Backend)
    monkeypatch.setattr(native_run, "MasterInput", Master)
    monkeypatch.setattr(native_run, "CameraCache", Cache)
    monkeypatch.setattr(native_run, "engage_native", engage)
    monkeypatch.setattr(native_run, "engage_takeover", relative_engage)
    monkeypatch.setattr(native_run, "WebControls", Keys)
    monkeypatch.setattr(native_run, "Trainer", EvalTrainer)
    monkeypatch.setattr(native_run, "ActiveMasterSync", Sync)
    monkeypatch.setattr(native_run.WorkbenchGuard, "check", lambda _: None)
    if web:
        (tmp_path / "metrics.jsonl").write_text('{"success":true}\n{"success":false}\n')
    args = SimpleNamespace(
        mode=mode,
        mock=True,
        human_input="master",
        workbench_url="http://127.0.0.1:5000",
        output=tmp_path,
        data=[],
        resume=tmp_path / "eval.pt" if mode == "eval" else None,
        policy_burst_seconds=0.02 if mode == "eval" else None,
        active_master_sync=active_sync,
        episodes=1,
        web=web,
        web_host="127.0.0.1",
        web_port=0,
        web_public_host="127.0.0.1",
    )
    native_run.run_native(args, native_config(), keyboard_factory=Keys, episode_writer=finish_episode)
    path = next((tmp_path / "episodes").glob("*.npz"))
    with np.load(path) as data:
        assert len(data["action"]) > 1
        assert data["reward"][-1] == 1
        assert (
            ("policy" in data["action_source"])
            if mode == "eval"
            else set(data["action_source"]) == {"right_master_vla"}
        )
        q = data["action"][:, :7] * np.pi
        sdk_x = np.deg2rad(round(np.rad2deg(0.2) * 1000) / 1000)
        assert np.allclose(q[:, 0], sdk_x if mode == "record" else 0.0, atol=1e-7)
        if mode == "record":
            assert np.max(np.diff(q[:, 3])) > np.deg2rad(3)
        else:
            assert any("自主试动已结束" in message for message in consoles[0].messages)
        if handoff:
            assert len(takeovers) == 1
            assert "right_master_relative" in data["action_source"]
            mask = data["action_source"] == "right_master_relative"
            assert data["intervention"][mask].all()
    assert instances[0].q[0] == (0.2 if mode == "record" else 0.0)
    if active_sync:
        assert sync_instances[0].updates > 0 and sync_instances[0].stops > 0
        assert not sync_instances[0].running
    if web:
        assert consoles[0].count == (62 if handoff else 37)
        assert consoles[0].published["episode"] == 4
        assert consoles[0].published["saved_episodes"] == 3
        assert consoles[0].published["successful_episodes"] == 2
        assert consoles[0].published["failed_episodes"] == 1
