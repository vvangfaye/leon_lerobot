import threading
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nero_rl.native_takeover import NativeTakeover, engage_takeover


def sample(q, width, follower=None):
    return {
        "主臂关节": list(q),
        "主臂夹爪": [width],
        "从臂关节": list(np.zeros(7) if follower is None else follower),
        "从臂夹爪": [0.04],
    }


def test_arbitrary_master_pose_cannot_jump_follower_or_release_grasp():
    master = np.linspace(-0.8, 0.9, 7)
    follower = np.r_[np.linspace(0.7, -0.6, 7), 0.012]
    readings = sample(master, 0.085)
    takeover = NativeTakeover(readings, follower)
    np.testing.assert_allclose(takeover.action(readings), follower, atol=1e-15)
    # Actual object width is 40mm; preserve the 12mm closing target at handover.
    assert readings["从臂夹爪"][0] == 0.04
    assert takeover.action(readings)[7] == pytest.approx(0.012)


def test_takeover_tracks_absolute_master_displacement_without_integrator_drift():
    master = np.linspace(-0.2, 0.4, 7)
    follower = np.r_[np.linspace(0.3, -0.4, 7), 0.05]
    takeover = NativeTakeover(sample(master, 0.04), follower)
    delta = np.linspace(-0.03, 0.03, 7)
    moved = sample(master + delta, 0.035)
    expected = follower + np.r_[delta, -0.005]
    for _ in range(100):
        np.testing.assert_allclose(takeover.action(moved), expected, atol=1e-15)
    np.testing.assert_allclose(takeover.action(sample(master, 0.04)), follower, atol=1e-15)
    assert takeover.action(sample(master, 0.2))[7] == 0.09
    assert takeover.action(sample(master, -0.2))[7] == 0


def test_takeover_rejects_invalid_feedback():
    takeover = NativeTakeover(sample(np.zeros(7), 0.05), np.r_[np.zeros(7), 0.04])
    with pytest.raises(ValueError):
        takeover.action(sample([np.nan] * 7, 0.05))


def test_engage_uses_fresh_follower_anchor_and_never_calls_alignment(monkeypatch):
    from examples.nero_rl import native_takeover

    calls = []
    q = np.linspace(-0.2, 0.2, 7)
    backend = SimpleNamespace(
        prepared=False,
        last_servo_width=0.015,
        runtime=SimpleNamespace(channels={"can1": object()}),
    )
    backend.prepare = lambda: calls.append("prepare_follower")
    backend.hold = lambda: calls.append("hold_current_pose")
    master = SimpleNamespace(
        prepare=lambda: calls.append("prepare_master"),
        backend=SimpleNamespace(runtime=SimpleNamespace(channels={"can3": object()})),
    )

    class Feedback:
        def __init__(self, *_):
            self.ready = False
            self.count = 0

        def read(self, _):
            self.count += 1
            self.ready = self.count == 2
            return sample(np.ones(7), 0.08, follower=q)

    monkeypatch.setattr(native_takeover.sampling, "RobotFeedback", Feedback)
    keys = SimpleNamespace(cancel=threading.Event(), abort=threading.Event())
    result = engage_takeover(backend, master, keys)
    assert result is not None
    feedback, command, takeover = result
    assert feedback.count == 2
    assert calls == ["prepare_follower", "hold_current_pose", "prepare_master"]
    np.testing.assert_allclose(command, np.r_[q, 0.015])
    np.testing.assert_allclose(takeover.action(sample(np.ones(7), 0.08)), command)


def test_cancel_preparation_never_starts_following():
    keys = SimpleNamespace(cancel=threading.Event(), abort=threading.Event())
    calls = []
    backend = SimpleNamespace(prepared=True, hold=lambda: calls.append("hold"))
    master = SimpleNamespace(prepare=keys.cancel.set)
    assert engage_takeover(backend, master, keys) is None
    assert calls == ["hold"]
