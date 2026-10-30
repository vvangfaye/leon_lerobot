import math
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nero_rl.config import JOINT_LIMITS
from examples.nero_rl.workspace import survey_active
from examples.nero_to_nero.nero_platform.survey_control import SurveyAlignmentError, SurveyControl


def reading(leader=1.0, follower=0.0, grip=0.04):
    return {
        "主臂关节": [leader] * 7,
        "从臂关节": [follower] * 7,
        "主臂夹爪": [grip],
        "从臂夹爪": [0.06],
    }


def test_survey_starts_at_follower_not_master_and_never_catches_up():
    control = SurveyControl(reading(), 0)
    np.testing.assert_allclose(control.action(reading(), 0.01), [0] * 7 + [0.06])
    moved = control.action(reading(leader=2, grip=0.1), 0.02)
    np.testing.assert_allclose(moved[:7], [math.radians(5) * 0.01] * 7)
    assert moved[7] == pytest.approx(0.0601)
    np.testing.assert_allclose(control.action(reading(leader=2, grip=0.1), 0.03), moved)


def test_survey_limits_and_feedback_delay():
    value = reading()
    value["从臂关节"][6] = float(JOINT_LIMITS[6, 1])
    control = SurveyControl(value, 0)
    changed = reading(leader=2)
    changed["从臂关节"][6] = float(JOINT_LIMITS[6, 1])
    assert control.action(changed, 0.01)[6] == JOINT_LIMITS[6, 1]
    with pytest.raises(RuntimeError, match="延迟"):
        control.action(changed, 1)
    with pytest.raises(RuntimeError, match="未跟上"):
        SurveyControl(reading(), 0).action(reading(follower=0.1), 0.01)


def test_aligned_survey_requires_alignment_and_tracks_one_to_one():
    with pytest.raises(SurveyAlignmentError, match="尚未对齐"):
        SurveyControl(reading(leader=0.1), 0, absolute=True)
    initial = reading(leader=0, grip=0.06)
    control = SurveyControl(initial, 0, absolute=True)
    moved = reading(leader=0.001, grip=0.06)
    first = control.action(moved, 0.01)
    assert first[0] == pytest.approx(math.radians(5) * 0.01)
    second = control.action(moved, 0.02)
    np.testing.assert_allclose(second[:7], [0.001] * 7)
    next_action = control.action(reading(leader=0.1, grip=0.06), 0.03)
    assert next_action[0] - second[0] <= math.radians(5) * 0.01 + 1e-10
    assert control.lag_deg > 3


def test_faster_operator_only_creates_lag_not_another_pause():
    value = reading(leader=0, grip=0.06)
    control = SurveyControl(value, 0, absolute=True)
    previous = np.zeros(7)
    for i in range(1, 401):
        desired = math.radians(12) * min(i / 100, 1)
        value["主臂关节"] = [desired] * 7
        value["从臂关节"] = previous.tolist()
        command = np.array(control.action(value, i / 100)[:7])
        assert np.max(np.abs(command - previous)) <= math.radians(5) / 100 + 1e-10
        previous = command
        if i == 100:
            assert control.lag_deg > 3  # formerly triggered a false pause
    np.testing.assert_allclose(previous, [math.radians(12)] * 7)


def test_explicit_catchup_still_starts_at_actual_and_limits_every_increment():
    initial = reading(leader=0.3, grip=0.065)
    control = SurveyControl(initial, 0, absolute=True, allow_catchup=True)
    command = control.action(initial, 0.01)
    np.testing.assert_allclose(command[:7], [math.radians(5) * 0.01] * 7)
    assert command[7] == pytest.approx(0.0601)


def test_aligned_reader_requires_correct_mode():
    status = {"teleop_test_active": True, "teleop_test_kind": "workspace_aligned", "teleop_test_ready": True}
    assert survey_active(status, "workspace_aligned")
    with pytest.raises(RuntimeError):
        survey_active(status)


def test_survey_reader_rejects_other_controllers_and_faults():
    assert not survey_active({})
    assert not survey_active({"teleop_test_active": True, "teleop_test_kind": "workspace_survey"})
    assert survey_active(
        {
            "teleop_test_active": True,
            "teleop_test_kind": "workspace_survey",
            "teleop_test_ready": True,
        }
    )
    for status in [
        {"teleop_test_active": True},
        {"hand_guiding_channel": "can1"},
        {"teleop_test_error": "fault"},
        {"recording": True},
    ]:
        with pytest.raises(RuntimeError):
            survey_active(status)


def test_workbench_survey_does_not_align_or_send_master_pose(monkeypatch):
    from examples.nero_to_nero.nero_platform import recorder
    from examples.nero_to_nero.nero_platform.runtime import Runtime

    runtime = Runtime()
    runtime.state.update(teleop_test_kind="workspace_survey", teleop_test_stop=False)
    commands, forces = [], []

    def move(q):
        commands.append(q)
        runtime.state["teleop_test_stop"] = True

    runtime.channels["can1"].update(
        arm=SimpleNamespace(move_js=move),
        gripper=SimpleNamespace(move_gripper_m=lambda **kwargs: forces.append(kwargs["force"])),
    )

    class Feedback:
        ready = True

        def __init__(self, *_):
            pass

        def read(self, _):
            return reading()

    pair = SimpleNamespace(leader="can3", follower="can1")
    monkeypatch.setattr(recorder.pairing, "resolve", lambda _: [pair])
    monkeypatch.setattr(recorder.devices, "transition_to_collection", lambda _: None)
    monkeypatch.setattr(
        recorder.devices, "align_follower_before_recording", lambda _: pytest.fail("Alignment forbidden")
    )
    monkeypatch.setattr(recorder.sampling, "RobotFeedback", Feedback)
    monkeypatch.setattr(recorder, "_leave_leader_mode", lambda _: None)
    recorder._teleop_test_loop(runtime)
    assert runtime.state.get("teleop_test_error") is None
    assert not runtime.state["teleop_test_active"]
    assert forces == [3.0]
    assert len(commands) == 2  # initial hold and stop hold
    np.testing.assert_array_equal(commands, np.zeros((2, 7)))


def test_start_alignment_keeps_teaching_until_explicit_catchup(monkeypatch):
    from examples.nero_to_nero.nero_platform import recorder
    from examples.nero_to_nero.nero_platform.runtime import Runtime

    runtime = Runtime()
    runtime.state.update(
        teleop_test_kind="workspace_aligned", teleop_test_stop=False, teleop_test_active=True
    )
    commands, exited, pause_seen = [], [], []

    def move(q):
        commands.append(q)
        if len(commands) == 1:
            runtime.state["teleop_test_stop"] = True

    runtime.channels["can1"].update(
        arm=SimpleNamespace(move_js=move), gripper=SimpleNamespace(move_gripper_m=lambda **_: None)
    )

    class Feedback:
        ready = True
        count = 0

        def __init__(self, *_):
            pass

        def read(self, _):
            self.count += 1
            if runtime.state.get("teleop_test_paused"):
                assert not exited
                assert not runtime.state["teleop_test_ready"]
                assert not commands  # no position command while paused
                pause_seen.append(True)
                assert recorder.resume_teleop_test(runtime, catchup=True)[0]
            return reading(leader=0.1, grip=0.06)

    pair = SimpleNamespace(leader="can3", follower="can1")
    monkeypatch.setattr(recorder.pairing, "resolve", lambda _: [pair])
    monkeypatch.setattr(recorder.devices, "transition_to_collection", lambda _: None)
    monkeypatch.setattr(recorder.hardware_status, "require_ready_for_motion", lambda *_: None)
    monkeypatch.setattr(recorder.sampling, "RobotFeedback", Feedback)
    monkeypatch.setattr(recorder, "_leave_leader_mode", lambda _: exited.append(True))
    recorder._teleop_test_loop(runtime)
    assert pause_seen and exited == [True]
    assert runtime.state.get("teleop_test_error") is None
    assert not runtime.state["teleop_test_paused"]
    assert len(commands) == 2  # bounded first command, then hold measured feedback
    assert 0 <= max(commands[0]) <= math.radians(1.001)  # 5 deg/s * maximum accepted 0.2 s
    np.testing.assert_array_equal(commands[-1], np.zeros(7))


def test_start_unaligned_keeps_master_teaching_until_operator_stops(monkeypatch):
    from examples.nero_to_nero.nero_platform import recorder
    from examples.nero_to_nero.nero_platform.runtime import Runtime

    runtime = Runtime()
    runtime.state.update(
        teleop_test_kind="workspace_aligned", teleop_test_stop=False, teleop_test_active=True
    )
    commands, paused, exited = [], [], []
    runtime.channels["can1"].update(
        arm=SimpleNamespace(move_js=lambda q: commands.append(q)),
        gripper=SimpleNamespace(move_gripper_m=lambda **_: pytest.fail("No gripper action while paused")),
    )

    class Feedback:
        ready = True

        def __init__(self, *_):
            pass

        def read(self, _):
            if runtime.state.get("teleop_test_paused"):
                assert not commands and not exited
                paused.append(True)
                runtime.state["teleop_test_stop"] = True
            return reading(leader=0.1, grip=0.06)

    monkeypatch.setattr(
        recorder.pairing, "resolve", lambda _: [SimpleNamespace(leader="can3", follower="can1")]
    )
    monkeypatch.setattr(recorder.devices, "transition_to_collection", lambda _: None)
    monkeypatch.setattr(recorder.sampling, "RobotFeedback", Feedback)
    monkeypatch.setattr(recorder, "_leave_leader_mode", lambda _: exited.append(True))
    recorder._teleop_test_loop(runtime)
    assert paused and exited == [True]
    assert runtime.state.get("teleop_test_error") is None
    np.testing.assert_array_equal(commands, np.zeros((1, 7)))
