"""A new collection session must not inherit the previous session's stop request."""

import pytest
from nero_platform import camera_io, cameras, devices, recorder
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair


@pytest.fixture
def collection(tmp_path, monkeypatch):
    runtime = Runtime(config_path=tmp_path / "config.json", dataset_root=tmp_path / "datasets")
    events = []
    attach_fake_pair(runtime, events)
    runtime.state.update(record_camera="right", camera_map={"fake-camera": "right"})
    clock = Clock(events)
    monkeypatch.setattr(devices, "time", clock)
    monkeypatch.setattr(recorder, "time", clock)
    monkeypatch.setattr(
        cameras,
        "detect_all_cameras",
        lambda: {"fake-camera": {"type": "rgb", "backend": "opencv", "path": "/fake/camera"}},
    )
    monkeypatch.setattr(cameras, "close_preview_cams", lambda *args, **kwargs: None)
    monkeypatch.setattr(cameras, "open_preview_cams", lambda *args, **kwargs: None)
    monkeypatch.setattr(camera_io, "RecordingCamera", lambda *args, **kwargs: FakeCamera(events=events))
    monkeypatch.setattr(camera_io.OpenCVBackend, "profiles", lambda info: {"profiles": []})
    monkeypatch.setattr(camera_io, "connect_rgb_camera", lambda camera, *args: camera.connect())
    monkeypatch.setattr(recorder.LeRobotDataset, "create", lambda **kwargs: MemoryDataset(events))
    monkeypatch.setattr(devices.AgxArmFactory, "create_arm", lambda *args, **kwargs: pytest.fail("Real CAN"))
    monkeypatch.setattr(recorder.traceback, "print_exc", lambda: None)
    return runtime, events


def test_end_then_start_collection_does_not_inherit_stop_request(collection):
    runtime, events = collection
    assert recorder.do_connect(runtime)[0]
    recorder.do_disconnect(runtime, reopen_preview=False)
    assert runtime.state["stop_requested"]
    runtime.state["teleop_test_stop"] = True
    assert recorder.do_connect(runtime)[0]
    assert runtime.state["connected"]
    assert not runtime.state["stop_requested"]
    assert not runtime.state["teleop_test_stop"]
    assert not any(event[1] in ("move_js", "move_j", "move") for event in events)


def test_new_stop_during_camera_start_still_cancels_before_follower_motion(collection, monkeypatch):
    runtime, events = collection
    runtime.state["stop_requested"] = True
    original = FakeCamera.connect

    def connect_and_cancel(camera):
        original(camera)
        runtime.state["stop_requested"] = True

    monkeypatch.setattr(FakeCamera, "connect", connect_and_cancel)
    ok, message = recorder.do_connect(runtime)
    assert not ok
    assert "已取消" in message
    assert not runtime.state["connected"]
    assert not runtime.collection_cams
    assert ("从左", "enable") not in events
    assert not any(event[1] in ("move_js", "move_j", "move") for event in events)


def test_rejected_session_does_not_clear_existing_stop_request(collection):
    runtime, events = collection
    runtime.state.update(stop_requested=True, teleop_test_stop=True, close_error="save pending")
    assert not recorder.do_connect(runtime)[0]
    assert runtime.state["stop_requested"]
    assert runtime.state["teleop_test_stop"]
    assert not events
