import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nero_rl.active_master import ActiveMasterSync, MasterSyncError
from examples.nero_rl.hardware import Snapshot
from examples.nero_rl.run import mock_config


class SimulatedMaster:
    def __init__(self, q=None, width=0.04):
        self.q = np.zeros(7) if q is None else np.array(q, dtype=float)
        self.width = width
        self.sent = []
        self.calls = []
        self.prepared = False
        self.cfg = SimpleNamespace(channel="can3")
        self.seen = {}
        self.runtime = SimpleNamespace(
            channels={"can3": {}}, feedback_seen={}, lifecycle_lock=threading.Lock()
        )
        self.arm = SimpleNamespace(
            set_normal_mode=lambda: pytest.fail("NERO >=1.12 normal mode is a no-op"),
            set_follower_mode=lambda: self.calls.append("follower"),
            move_js=self.move,
            enable=lambda: True,
            set_speed_percent=lambda x: self.calls.append(("speed", x)),
        )
        self.arm._msg_mode = SimpleNamespace(
            enable_can_push=0,
            move_mode=1,
            Enums=SimpleNamespace(CanActiveMsgReporting=SimpleNamespace(ENABLE=1)),
        )
        self.arm._set_mode = lambda: self.calls.append(
            ("feedback", self.arm._msg_mode.enable_can_push, self.arm._msg_mode.move_mode)
        )
        self.runtime.channels["can3"]["arm"] = self.arm
        self.gripper = SimpleNamespace(move_gripper_m=self.grip)
        self.status_api = SimpleNamespace(require_ready_for_motion=lambda *_: None)
        self.last_servo_target = self.last_servo_width = None

    def move(self, q):
        self.sent.append(np.array(q))
        self.q = np.array(q)

    def grip(self, *, value, force):
        assert force == 1
        self.width = value

    def read(self, **_):
        return Snapshot(self.q.copy(), self.width, np.eye(4), {})

    def hold(self):
        self.calls.append("hold")
        self.move(self.q)
        return self.read()

    def stop(self):
        self.calls.append("damping")


def build(monkeypatch, backend=None):
    now = [10.0]
    monkeypatch.setattr("examples.nero_rl.active_master.time.monotonic", lambda: now[0])
    backend = backend or SimulatedMaster()
    master = SimpleNamespace(backend=backend, prepared=False, changed_mode=False)
    cfg = replace(mock_config(), channel="can1", hz=100, policy_joint_speed_deg_s=5)
    return ActiveMasterSync(master, cfg), backend, now


def test_large_initial_gap_requires_explicit_alignment_before_any_master_command(monkeypatch):
    sync, backend, _ = build(monkeypatch)
    follower = Snapshot(np.full(7, np.deg2rad(10)), 0.06, np.eye(4), {})
    with pytest.raises(MasterSyncError, match="先点"):
        sync.begin(follower, align=False)
    assert not backend.sent and not backend.calls


def test_alignment_is_bounded_and_only_moves_master_toward_measured_follower(monkeypatch):
    sync, backend, now = build(monkeypatch)
    target = np.full(7, np.deg2rad(4))
    follower = Snapshot(target, 0.04, np.eye(4), {})
    sync.begin(follower, align=True)
    assert "follower" in backend.calls
    assert ("feedback", 1, 255) in backend.calls
    assert np.max(abs(backend.q)) == 0  # Mode preparation latches current pose.
    completed = False
    for _ in range(100):
        now[0] += 0.01
        if sync.update(follower):
            completed = True
            break
    assert completed
    assert np.max(abs(np.diff(backend.sent, axis=0))) <= np.deg2rad(0.05) + 1e-10
    np.testing.assert_array_equal(follower.joints, target)
    sync.stop()
    assert not sync.running and not sync.aligning


def test_policy_sync_uses_actual_feedback_and_fails_on_pair_divergence(monkeypatch):
    sync, backend, now = build(monkeypatch)
    follower = Snapshot(np.zeros(7), 0.04, np.eye(4), {})
    sync.begin(follower, align=False)
    follower.joints[2] = np.deg2rad(1)
    now[0] += 10  # A delayed tick must not release a burst of accumulated targets.
    sync.update(follower)
    assert backend.q[2] == pytest.approx(np.deg2rad(0.05))
    count = len(backend.sent)
    follower.joints[2] = np.deg2rad(8)
    with pytest.raises(MasterSyncError, match="5°"):
        sync.update(follower)
    assert len(backend.sent) == count


def test_handover_holds_master_then_requires_teaching_repreparation(monkeypatch):
    sync, backend, now = build(monkeypatch)
    sync.begin(backend.read(), align=False)
    now[0] += 0.01
    sync.update(backend.read())
    sync.yield_to_human()
    assert backend.calls[-1] == "hold"
    assert not sync.drive_ready and not sync.running
    assert not sync.master.prepared and not backend.prepared
    assert sync.phase == "teaching_pending"
    count = len(backend.sent)
    with pytest.raises(MasterSyncError, match="尚未准备"):
        sync.update(backend.read())
    assert len(backend.sent) == count


def test_sync_rejects_invalid_feedback_and_cancellation_before_moving(monkeypatch):
    sync, backend, _ = build(monkeypatch)
    bad = backend.read()
    bad.joints[0] = np.nan
    with pytest.raises(ValueError):
        sync.begin(bad, align=True)
    assert not backend.sent
    sync.cancel_requested = lambda: True
    with pytest.raises(MasterSyncError, match="取消"):
        sync.begin(backend.read(), align=True)
    assert not backend.sent


def test_master_stall_and_follower_change_during_alignment_stop_without_new_target(monkeypatch):
    sync, backend, now = build(monkeypatch)
    follower = Snapshot(np.full(7, 0.5), 0.04, np.eye(4), {})
    sync.begin(follower, align=True)
    now[0] += 0.01
    sync.update(follower)
    count = len(backend.sent)
    follower.joints[0] += np.deg2rad(1)
    with pytest.raises(MasterSyncError, match="从臂位置改变"):
        sync.update(follower)
    assert len(backend.sent) == count


def test_stale_master_feedback_stops_with_damping_fallback(monkeypatch):
    sync, backend, now = build(monkeypatch)
    sync.begin(backend.read(), align=False)
    target = backend.read()
    target.joints[0] += 0.01
    now[0] += 0.01
    sync.update(target)

    def failed_hold():
        raise TimeoutError("stale feedback")

    backend.hold = failed_hold
    sync.stop()
    assert not sync.running
    assert backend.calls[-1] == "damping"
    assert sync.phase == "faulted"
    sync.stop()
    assert backend.calls.count("damping") == 1


def test_failed_preparation_retains_console_without_damping_or_repeated_cleanup(monkeypatch):
    sync, backend, _ = build(monkeypatch)
    sync.begin(backend.read(), align=False)
    backend.hold = lambda: (_ for _ in ()).throw(TimeoutError("missing status"))
    sync.stop()
    assert sync.faulted and sync.master.control_faulted
    assert "missing status" in sync.diagnostics["stop_error"]
    assert "damping" not in backend.calls
    sync.stop()
    with pytest.raises(MasterSyncError, match="尚未恢复"):
        sync.begin(backend.read(), align=True)


def test_stalled_master_never_accumulates_unbounded_targets(monkeypatch):
    sync, backend, now = build(monkeypatch)
    follower = Snapshot(np.full(7, 0.5), 0.04, np.eye(4), {})
    sync.begin(follower, align=True)
    backend.arm.move_js = lambda q: backend.sent.append(np.array(q))
    with pytest.raises(MasterSyncError, match="持续未跟进"):
        for _ in range(250):
            now[0] += 0.01
            sync.update(follower)
    assert np.max(np.abs(backend.sent)) <= sync.MAX_LEAD_RAD + 1e-10


def test_cancel_between_updates_sends_no_new_master_target(monkeypatch):
    sync, backend, now = build(monkeypatch)
    follower = backend.read()
    sync.begin(follower, align=False)
    count = len(backend.sent)
    sync.cancel_requested = lambda: True
    now[0] += 0.01
    with pytest.raises(MasterSyncError, match="取消"):
        sync.update(follower)
    assert len(backend.sent) == count


def test_outside_master_may_only_recover_inward_during_explicit_alignment(monkeypatch):
    q = np.zeros(7)
    q[6] = np.deg2rad(97)
    sync, backend, now = build(monkeypatch, SimulatedMaster(q=q))
    target = q.copy()
    target[6] = np.deg2rad(80)
    follower = Snapshot(target, 0.04, np.eye(4), {})
    with pytest.raises(MasterSyncError, match="当前姿态"):
        sync.begin(follower, align=False)
    assert not backend.sent
    sync.begin(follower, align=True)
    for _ in range(160):
        now[0] += 0.01
        sync.update(follower)
    requested = np.array(backend.sent)[:, 6]
    assert np.all(np.diff(requested) <= 1e-10)
    assert requested.max() <= np.deg2rad(97)
    assert backend.q[6] < np.deg2rad(90)


def test_prepare_hold_enables_only_at_current_master_pose_without_aligning(monkeypatch):
    sync, backend, _ = build(monkeypatch)
    start = backend.q.copy()
    follower = Snapshot(np.full(7, 0.8), 0.06, np.eye(4), {})
    sync.prepare_hold(follower)
    assert sync.drive_ready and sync.phase == "holding"
    assert not sync.running and not sync.aligning
    assert backend.sent
    for command in backend.sent:
        np.testing.assert_array_equal(command, start)
    assert backend.width == 0.04  # No gripper synchronization either.
