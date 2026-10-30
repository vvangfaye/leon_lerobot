"""Delayed teaching broadcasts must not trigger a premature follower move."""

from types import SimpleNamespace

import pytest
from nero_platform import devices, pairing, recorder
from nero_platform.runtime import Runtime
from test_platform import Clock, attach_fake_pair


@pytest.fixture
def setup(monkeypatch):
    runtime = Runtime()
    events = []
    attach_fake_pair(runtime, events)
    clock = Clock(events)
    monkeypatch.setattr(devices, "time", clock)
    leader = runtime.channels["can0"]["arm"]
    return runtime, events, clock, leader


def test_waits_for_delayed_teaching_stream_without_motion(setup, monkeypatch):
    runtime, events, clock, leader = setup
    read = leader.get_leader_joint_angles
    monkeypatch.setattr(leader, "get_leader_joint_angles", lambda: read() if clock.now >= 0.8 else None)
    devices._wait_for_leader_feedback(runtime, "can0")
    assert clock.now >= 0.8
    assert not any(event[1] in ("enable", "move_js", "move_j", "disable") for event in events)


@pytest.mark.parametrize("kind", ["missing", "stale", "partial"])
def test_no_fresh_teaching_stream_stops_before_enabling_follower(setup, monkeypatch, kind):
    runtime, events, _, leader = setup
    monkeypatch.setattr(devices, "_exit_teaching_before_enable", lambda *args: None)
    monkeypatch.setattr(devices, "_prepare_leader_gripper", lambda *args: None)
    monkeypatch.setattr(devices.hardware_status, "require_ready_for_motion", lambda *args: None)
    if kind == "missing":
        monkeypatch.setattr(leader, "get_leader_joint_angles", lambda: None)
    elif kind == "stale":
        monkeypatch.setattr(
            leader, "get_leader_joint_angles", lambda: SimpleNamespace(msg=[0.0] * 7, timestamp=1.0)
        )
    else:
        leader._parser = SimpleNamespace(
            **{"leader_joint_" + suffix: SimpleNamespace(timestamp=1.0) for suffix in ("12", "34", "56", "7")}
        )
    with pytest.raises(devices.AlignmentError, match="持续更新"):
        devices._transition_one_pair(runtime, pairing.resolve(runtime)[0])
    assert len([event for event in events if event == ("主左", "leader")]) == 2
    assert ("从左", "enable") not in events
    assert not any(event[1] in ("move_js", "move_j") for event in events)


def test_mode_retry_recovers_only_after_new_teaching_feedback(setup, monkeypatch):
    runtime, events, _, leader = setup
    read = leader.get_leader_joint_angles
    monkeypatch.setattr(devices, "_exit_teaching_before_enable", lambda *args: None)
    monkeypatch.setattr(devices, "_prepare_leader_gripper", lambda *args: None)
    monkeypatch.setattr(devices.hardware_status, "require_ready_for_motion", lambda *args: None)
    monkeypatch.setattr(
        leader,
        "get_leader_joint_angles",
        lambda: read() if events.count(("主左", "leader")) == 2 else None,
    )
    devices._transition_one_pair(runtime, pairing.resolve(runtime)[0])
    assert events.count(("主左", "leader")) == 2
    assert ("从左", "enable") in events
    assert not any(event[1] in ("move_js", "move_j") for event in events)


def test_teleop_start_failure_restores_recording_status(setup, monkeypatch):
    runtime, _, _, _ = setup
    runtime.state.update(record_phase="idle", record_message="等待开始录制", teleop_test_active=True)

    def fail_alignment(*args):
        runtime.state.update(record_phase="aligning", record_message="正在读取主臂当前位置")
        raise devices.AlignmentError("主右示教反馈缺失")

    monkeypatch.setattr(devices, "transition_to_collection", lambda *args: None)
    monkeypatch.setattr(devices, "align_follower_before_recording", fail_alignment)
    monkeypatch.setattr(recorder, "_leave_leader_mode", lambda *args: None)
    monkeypatch.setattr(recorder.traceback, "print_exc", lambda: None)
    recorder._teleop_test_loop(runtime)
    assert runtime.state["record_phase"] == "idle"
    assert runtime.state["record_message"] == "等待开始录制"
    assert runtime.state["teleop_test_error"] == "主右示教反馈缺失"
    assert not runtime.state["teleop_test_active"]
