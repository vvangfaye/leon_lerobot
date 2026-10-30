"""Offline master-input checks: no CAN interfaces or cameras are opened."""

import json
import urllib.error
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.nero_rl import run
from examples.nero_rl.learning import MixedReplay, load_episodes
from examples.nero_rl.master_input import (
    MasterInput,
    MasterSample,
    RelativeMaster,
    WorkbenchGuard,
    request_state_feedback,
)


def sample(t=1.0, xyz=(0, 0, 0), width=0.04, rotation=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    return MasterSample(pose, width, t)


def test_relative_delta_has_no_absolute_alignment_and_same_action_units():
    cfg = replace(run.mock_config(), translation_step_m=0.001)
    control = RelativeMaster(cfg)
    control.rebase(sample(xyz=(1, 2, 3)))
    action = control.action(sample(1.2, (1.002, 2, 3), 0.041, (0, 0, cfg.yaw_step_rad / 2)))
    np.testing.assert_allclose(action, [0.5, 0, 0, 0.5, 0.5], atol=1e-6)
    control.rebase(sample(2, (-9, -9, -9)))  # move handle during pause
    np.testing.assert_array_equal(control.action(sample(2.2, (-9, -9, -9))), np.zeros(5))


def test_clipping_does_not_queue_motion_after_handle_stops():
    control = RelativeMaster(run.mock_config())
    control.rebase(sample())
    np.testing.assert_array_equal(control.action(sample(1.2, (1, 1, 1), 0.1)), [1, 1, 1, 0, 1])
    np.testing.assert_array_equal(control.action(sample(1.4, (1, 1, 1), 0.1)), np.zeros(5))


def test_roll_pitch_are_ignored_and_yaw_wrap_is_small():
    cfg = run.mock_config()
    control = RelativeMaster(cfg)
    control.rebase(sample(rotation=(0, 0, np.pi - 0.005)))
    action = control.action(sample(1.2, rotation=(0, 0, -np.pi + 0.005)))
    assert action[3] == pytest.approx(0.01 / cfg.yaw_step_rad)
    control.rebase(sample(2))
    np.testing.assert_array_equal(control.action(sample(2.2, rotation=(0.1, 0, 0))), np.zeros(5))


def test_slow_cycle_discards_motion_and_rebases():
    control = RelativeMaster(run.mock_config())
    control.rebase(sample())
    np.testing.assert_array_equal(control.action(sample(3, (1, 0, 0))), np.zeros(5))
    np.testing.assert_array_equal(control.action(sample(3.2, (1, 0, 0))), np.zeros(5))
    with pytest.raises(RuntimeError, match="advance"):
        control.action(sample(3.2))


@pytest.mark.parametrize("scale", [0, -1, 1.1, np.inf, np.nan])
def test_bad_scale_refused(scale):
    with pytest.raises(ValueError):
        RelativeMaster(run.mock_config(), scale)


def test_invalid_feedback_refused():
    control = RelativeMaster(run.mock_config())
    for bad in [sample(xyz=(np.nan, 0, 0)), sample(width=0.2), sample(t=np.inf)]:
        with pytest.raises(ValueError):
            control.rebase(bad)


def test_master_lifecycle_only_prepares_can3_and_never_sends_position(monkeypatch):
    from examples.nero_to_nero.nero_platform import devices

    events = []
    control = MasterInput(run.mock_config())
    assert control.backend.cfg.channel == "can3"
    assert control.backend.cfg.cameras == {}
    assert control.backend.arm is None  # construction is inert
    control.backend.arm = SimpleNamespace(enable=lambda: events.append("enable") or True)
    control.backend.runtime = object()
    control.backend.status_api = SimpleNamespace(
        inspect_status=lambda *_: {"health": {"code": 0, "error_code": 0}},
        require_ready_for_motion=lambda _, ch: events.append(("ready", ch)),
    )
    control.reader = lambda _: None
    monkeypatch.setattr(control.backend, "close", lambda: events.append("close"))
    monkeypatch.setattr(devices, "set_control_role", lambda _, ch, role: events.append((ch, role)))
    monkeypatch.setattr(devices, "_prepare_leader_gripper", lambda _, ch: events.append((ch, "grip")))
    monkeypatch.setattr(control, "rebase", lambda: events.append("rebase"))
    control.prepare()
    control.prepare()
    control.close()
    assert events == [
        ("can3", "follower"),
        "enable",
        ("ready", "can3"),
        ("can3", "grip"),
        ("can3", "leader"),
        "rebase",
        ("can3", "follower"),
        ("ready", "can3"),
        "close",
    ]


def test_failed_preparation_still_restores_master_mode(monkeypatch):
    from examples.nero_to_nero.nero_platform import devices

    control = MasterInput(run.mock_config())
    events = []
    control.backend.runtime = object()
    control.backend.status_api = SimpleNamespace(
        inspect_status=lambda *_: {"health": {"code": 1, "error_code": 1}},
        require_ready_for_motion=lambda *_: None,
    )
    monkeypatch.setattr(devices, "set_control_role", lambda _, ch, role: events.append((ch, role)))
    monkeypatch.setattr(control.backend, "close", lambda: events.append("closed"))
    with pytest.raises(RuntimeError, match="not ready"):
        control.prepare()
    control.close()
    assert events == [("can3", "follower"), ("can3", "follower"), "closed"]


def test_guard_requires_server_down_not_merely_idle():
    class Opener:
        error = None

        def open(self, *_args, **_kwargs):
            if self.error:
                raise self.error
            return nullcontext()

    guard = WorkbenchGuard("http://127.0.0.1:5000")
    opener = Opener()
    guard.opener = opener
    opener.error = urllib.error.URLError(ConnectionRefusedError())
    guard.check()
    opener.error = urllib.error.URLError(TimeoutError())
    with pytest.raises(RuntimeError, match="Cannot establish"):
        guard.check()
    opener.error = urllib.error.HTTPError(guard.url, 500, "Error", {}, None)
    with pytest.raises(RuntimeError):
        guard.check()
    opener.error = None
    with pytest.raises(RuntimeError, match="Stop the NERO workbench"):
        guard.check()


@pytest.mark.parametrize("source", ["普通反馈", "拖动反馈"])
def test_read_refuses_wrong_mode_or_stale_gripper(source, monkeypatch):
    from examples.nero_rl import master_input

    control = MasterInput(run.mock_config())
    clock = iter(np.arange(0, 10, 0.01))
    monkeypatch.setattr(master_input.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(master_input.time, "sleep", lambda _: None)
    control.reader = SimpleNamespace(read=lambda _: (np.zeros(7), source))
    control.backend.gripper = SimpleNamespace(get_gripper_ctrl_states=lambda: object())
    monkeypatch.setattr(control.backend, "_fresh", lambda *_: source == "普通反馈")
    with pytest.raises(TimeoutError, match="stale"):
        control.read()


def test_master_demos_and_policy_interventions_roundtrip_into_sac(tmp_path, monkeypatch):
    class Keys:
        events = []

        def __enter__(self):
            self.abort = SimpleNamespace(is_set=lambda: False)
            self.iterator = iter(self.events)
            return self

        def poll(self):
            return next(self.iterator, ["q"])

        def __exit__(self, *_):
            pass

    class FakeMaster:
        def __init__(self, cfg, **_):
            self.mapper = RelativeMaster(cfg)
            self.time = 0
            self.x = 0
            self.prepared = False

        def open(self):
            pass

        def prepare(self):
            self.prepared = True

        def rebase(self):
            self.time += 0.2
            self.mapper.rebase(sample(self.time, (self.x, 0, 0)))

        def action(self):
            self.time += 0.2
            self.x += 0.001
            return self.mapper.action(sample(self.time, (self.x, 0, 0)))

        def close(self):
            pass

    monkeypatch.setattr(run, "Keyboard", Keys)
    monkeypatch.setattr(run, "MasterInput", FakeMaster)
    monkeypatch.setattr(run.WorkbenchGuard, "check", lambda _: None)
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    cfg = replace(run.mock_config(), learning_starts=1, batch_size=2, updates_per_episode=1)
    args = SimpleNamespace(
        mock=True,
        human_input="master",
        master_translation_scale=0.25,
        workbench_url="http://127.0.0.1:5000",
        mode="record",
        resume=None,
        output=tmp_path / "demos",
        data=[],
        episodes=1,
        preview=False,
    )
    Keys.events = [["n"], [], [], ["p"], [], ["i"], [], ["1"]]
    run.run_interactive(args, cfg)
    path = next((args.output / "episodes").glob("*.npz"))
    with np.load(path) as data:
        assert set(data["action_source"]) == {"right_master", "outcome"}
        assert data["intervention"].all()
        assert data["reward"][-1] == 1
        assert json.loads(str(data["metadata"]))["mock"]
        assert (data["action"][:-1, 0] > 0).all()
    replay = MixedReplay(512, 0)
    assert load_episodes(args.output / "episodes", replay, cfg, mock=True) == 5
    args.data = [args.output / "episodes"]
    args.output, args.mode = tmp_path / "train", "train"
    Keys.events = [["n"], ["a"], [], ["i"], [], ["1"]]
    run.run_interactive(args, cfg)
    assert (args.output / "checkpoint.pt").exists()
    with np.load(next((args.output / "episodes").glob("*.npz"))) as data:
        assert set(data["action_source"]) == {"policy", "right_master", "outcome"}
        assert data["intervention"].tolist() == [False, False, True, True, True]


def test_feedback_request_sends_only_non_motion_reporting_fields_and_restores_cache():
    mode = SimpleNamespace(
        enable_can_push=0,
        move_mode=1,
        Enums=SimpleNamespace(CanActiveMsgReporting=SimpleNamespace(ENABLE=1)),
    )
    messages = []
    arm = SimpleNamespace(
        _msg_mode=mode,
        _set_mode=lambda: messages.append((mode.enable_can_push, mode.move_mode)),
    )
    request_state_feedback(arm)
    assert messages == [(1, 255)]
    assert (mode.enable_can_push, mode.move_mode) == (0, 1)


def test_faulted_master_cleanup_does_not_change_modes_or_mask_original_error(monkeypatch):
    from examples.nero_to_nero.nero_platform import devices

    control = MasterInput(run.mock_config())
    control.changed_mode = control.control_faulted = True
    closed = []
    monkeypatch.setattr(control.backend, "close", lambda: closed.append(True))
    monkeypatch.setattr(devices, "set_control_role", lambda *_: pytest.fail("Fault cleanup changed mode"))
    control.close()
    assert closed == [True]
