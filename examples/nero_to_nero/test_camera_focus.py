"""模拟 V4L2 读写和真实录制调度，验证失败提示与采样循环零新增控制读取。"""

import copy
import errno
import json
import struct
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from nero_platform import camera_focus, camera_io, cameras, config, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair


@pytest.fixture
def controls(monkeypatch):
    values = {0x009A090C: 1, 0x009A090A: 90, 0x009A090D: 100, 0x009A0901: 1, 0x009A0902: 2500}
    writes = []
    failures = {}

    def ioctl(fd, request, raw):
        cid = struct.unpack_from("I", raw)[0]
        if (request, cid) in failures:
            raise failures[request, cid]
        if cid not in values:
            raise OSError(errno.EINVAL, "no such control")
        if request == 0xC0445624:
            return struct.pack(
                "II32siiiiI2I", cid, 1, b"mock", 0, 1 if cid == 0x009A090C else 10000, 1, 0, 0, 0, 0
            )
        if request == 0xC008561B:
            return struct.pack("Ii", cid, values[cid])
        assert request == 0xC008561C
        value = struct.unpack("Ii", raw)[1]
        if cid == 0x009A090A and 0x009A090C in values:
            assert values[0x009A090C] == 0, "AF 关闭并读回后才能设置手动对焦"
        writes.append((cid, value))
        values[cid] = value
        return raw

    monkeypatch.setattr(camera_focus, "ioctl", ioctl)
    return values, writes, failures


def test_focus_readback_and_readonly_audit_do_not_change_zoom_or_exposure(controls):
    values, writes, _ = controls
    result = camera_focus.fix_focus("/fake/camera")
    assert result["state"] == "verified"
    assert writes == [(0x009A090C, 0), (0x009A090A, 128)]
    before = copy.deepcopy(values)
    check = camera_focus.inspect_focus("/fake/camera", stage="episode_start", setting=result["setting"])
    assert check["values"] == {
        "focus_auto": 0,
        "focus_absolute": 128,
        "zoom_absolute": 100,
        "exposure_auto": 1,
        "exposure_absolute": 2500,
    }
    assert check["stage"] == "episode_start"
    assert values == before and len(writes) == 2


def test_unsupported_is_not_confused_with_read_or_permission_failure(controls, monkeypatch):
    values, writes, failures = controls
    del values[0x009A090C]
    del values[0x009A090A]
    assert camera_focus.fix_focus("/fake/fixed")["state"] == "unsupported"
    assert not writes
    values[0x009A090C] = 0
    failures[0xC008561B, 0x009A090C] = OSError(errno.EINVAL, "read failed despite supported control")
    check = camera_focus.inspect_focus("/fake/af", stage="episode_start")
    assert check["state"] == "warning" and "focus_auto" in check["errors"]

    @contextmanager
    def denied(*args, **kwargs):
        raise PermissionError("permission denied")
        yield  # pragma: no cover

    monkeypatch.setattr(camera_focus, "_device", denied)
    assert camera_focus.fix_focus("/fake/denied")["state"] == "warning"


def test_af_write_failure_is_visible_and_never_writes_manual_focus(controls):
    _, writes, failures = controls
    failures[0xC008561C, 0x009A090C] = OSError(errno.EBUSY, "device busy")
    result = camera_focus.fix_focus("/fake/camera")
    assert result["state"] == "warning"
    assert "busy" in result["message"]
    assert not writes


def test_af_query_failure_does_not_write_manual_focus_while_mode_is_unknown(controls):
    _, writes, failures = controls
    failures[0xC0445624, 0x009A090C] = OSError(errno.EIO, "USB error")
    result = camera_focus.configure_focus("/fake/camera", {"mode": "manual", "value": 40})
    assert result["state"] == "warning"
    assert "自动对焦状态未确认" in result["message"]
    assert not writes


def test_silent_write_ignored_by_driver_is_detected_without_retrying_writes(controls, monkeypatch):
    values, _, _ = controls
    original = camera_focus.ioctl
    writes = []

    def ignored(fd, request, raw):
        if request == 0xC008561C:
            writes.append(struct.unpack("Ii", raw))
            return raw
        return original(fd, request, raw)

    ticks = iter([0, 0.3, 0.3])
    monkeypatch.setattr(camera_focus, "ioctl", ignored)
    monkeypatch.setattr(camera_focus, "time", Mock(time=lambda: 123.0, monotonic=lambda: next(ticks)))
    result = camera_focus.fix_focus("/fake/camera")
    assert result["state"] == "warning"
    assert "读回仍为 1" in result["message"]
    assert writes == [(0x009A090C, 0), (0x009A090C, 1)] and values[0x009A090A] == 90


def test_manual_focus_without_af_control_is_supported(controls):
    values, writes, _ = controls
    del values[0x009A090C]
    assert camera_focus.fix_focus("/fake/manual")["state"] == "verified"
    check = camera_focus.inspect_focus("/fake/manual", stage="episode_start")
    assert check["state"] == "verified" and writes == [(0x009A090A, 128)]


def test_unsupported_manual_value_is_not_silently_clamped(controls, monkeypatch):
    _, writes, _ = controls
    original = camera_focus._query

    def limits(fd, name):
        result = original(fd, name)
        if name == "focus_absolute":
            result["max"] = 100
        return result

    monkeypatch.setattr(camera_focus, "_query", limits)
    result = camera_focus.fix_focus("/fake/camera")
    assert result["state"] == "warning" and "不支持 128" in result["message"]
    assert not writes, "先检查全部范围，不能先关 AF 再发现手动值越界"


def test_preview_retains_image_and_exposes_failed_focus_check(controls, monkeypatch):
    _, _, failures = controls
    failures[0xC008561C, 0x009A090C] = OSError(errno.EIO, "USB error")
    runtime = Runtime()
    monkeypatch.setattr(cameras, "detect_cameras", lambda: {"usb": "/fake/usb"})
    monkeypatch.setattr(cameras, "OpenCVCamera", FakeCamera)
    cameras.open_preview_cams(runtime)
    client = create_app(runtime).test_client()
    assert client.get("/api/status").json["camera_focus"]["usb"]["state"] == "warning"
    assert client.get("/api/camera/usb/frame").status_code == 200


def test_episode_boundaries_are_readonly_excluded_from_cadence_and_persisted(tmp_path, controls, monkeypatch):
    values, writes, _ = controls
    runtime = Runtime(dataset_root=tmp_path)
    events = []
    attach_fake_pair(runtime, events)
    info = {"id": "usb", "type": "rgb", "backend": "opencv", "path": "/fake/usb"}
    runtime.state.update(connected=True, fps=10, episode_sec=0.25, camera_sources={"right": info})
    cam = FakeCamera(events=events)
    camera_io.OpenCVBackend.connect(cam, info, {"mode": "manual", "value": 128})
    runtime.collection_cams["right"] = cam
    runtime.dataset = MemoryDataset(events)
    runtime.dataset.root = tmp_path
    clock = Clock(events)
    monkeypatch.setattr(recorder, "time", clock)
    inspect = camera_focus.inspect_focus
    boundary_calls = []

    def readback(path, *, stage, setting=None):
        boundary_calls.append((stage, len(runtime.dataset.frames)))
        # 模拟检查耗时一秒，不应改变 0.25 秒本集的 3 帧或控制节拍。
        clock.now += 1.0
        if stage == "episode_end":
            values[0x009A090C] = 1
            values[0x009A090D] = 125
        return inspect(path, stage=stage, setting=setting)

    monkeypatch.setattr(camera_focus, "inspect_focus", readback)
    recorder._record_episode(runtime, ["right"])
    assert boundary_calls == [("episode_start", 0), ("episode_end", 3)]
    assert writes == [(0x009A090C, 0), (0x009A090A, 128)], "集首尾禁止修正设备参数"
    last = runtime.state["last_episode"]
    assert last["frames"] == 3 and last["needs_review"]
    assert last["elapsed_seconds"] == 0.2  # 显示按一位小数四舍五入。
    report = json.loads((tmp_path / "meta/nero_timing/episode_000000.json").read_text())
    audit = report["camera_focus"]
    assert audit["continuous_monitoring"] is False
    assert audit["cameras"]["right"]["before"]["values"]["focus_auto"] == 0
    assert audit["cameras"]["right"]["after"]["values"]["focus_auto"] == 1
    assert any("zoom_absolute" in message for message in report["summary"]["warnings"])
    before_calls = boundary_calls[:]
    client = create_app(runtime).test_client()
    for _ in range(3):
        assert client.get("/api/status").json["camera_focus"]["usb"]["state"] == "warning"
    assert boundary_calls == before_calls, "状态轮询不得访问 V4L2 控制"


def test_auto_mode_does_not_write_manual_position_or_flag_normal_refocusing(controls):
    values, writes, _ = controls
    result = camera_focus.configure_focus("/fake/new")
    assert result["setting"]["mode"] == "auto"
    assert writes == [(0x009A090C, 1)]
    before = camera_focus.inspect_focus("/fake/new", stage="episode_start", setting=result["setting"])
    values[0x009A090A] = 24
    after = camera_focus.inspect_focus("/fake/new", stage="episode_end", setting=result["setting"])
    assert not camera_focus.audit_warnings({"head": {"before": before, "after": after}})
    values[0x009A090C] = 0
    after = camera_focus.inspect_focus("/fake/new", stage="episode_end", setting=result["setting"])
    assert camera_focus.audit_warnings({"head": {"before": before, "after": after}})


def test_manual_failure_restores_previous_auto_mode(controls):
    values, writes, failures = controls
    failures[0xC008561C, 0x009A090A] = OSError(errno.EIO, "focus motor unavailable")
    result = camera_focus.configure_focus("/fake/camera", {"mode": "manual", "value": 40})
    assert result["state"] == "warning" and values[0x009A090C] == 1
    assert writes == [(0x009A090C, 0), (0x009A090C, 1)]


def test_device_focus_api_persists_separately_and_reapplies_on_record_connect(tmp_path, controls):
    runtime = Runtime(config_path=tmp_path / "config.json")
    info = {"type": "rgb", "backend": "opencv", "path": "/fake/usb"}
    runtime.camera_inventory = {"usb": info}
    runtime.preview_cams["usb"] = FakeCamera()
    runtime.state["camera_focus_settings"] = {"another": {"mode": "auto"}}
    client = create_app(runtime).test_client()
    response = client.post("/api/camera/usb/focus", json={"mode": "manual", "value": 40})
    assert response.status_code == 200
    restarted = Runtime(config_path=runtime.config_path)
    config.load_saved(restarted)
    assert restarted.state["camera_focus_settings"] == {
        "another": {"mode": "auto"},
        "usb": {"mode": "manual", "value": 40},
    }
    cam = FakeCamera()
    camera_io.OpenCVBackend.connect(cam, info, restarted.state["camera_focus_settings"]["usb"])
    assert cam.metadata["focus"]["values"]["focus_absolute"] == 40


def test_focus_api_rejects_busy_unknown_and_invalid_values_without_writes(tmp_path, controls, monkeypatch):
    values, writes, _ = controls
    runtime = Runtime(config_path=tmp_path / "config.json")
    runtime.camera_inventory = {
        "usb": {"type": "rgb", "backend": "opencv", "path": "/fake/usb"},
        "femto": {"type": "rgbd", "backend": "orbbec"},
    }
    runtime.preview_cams["usb"] = FakeCamera()
    client = create_app(runtime).test_client()
    setting = {"mode": "manual", "value": 40}
    runtime.state["connected"] = True
    assert client.post("/api/camera/usb/focus", json=setting).status_code == 409
    runtime.state["connected"] = False
    runtime.device_busy.add("camera_detection")
    assert client.post("/api/camera/usb/focus", json=setting).status_code == 409
    runtime.device_busy.clear()
    for device in ("unknown", "femto"):
        assert client.post(f"/api/camera/{device}/focus", json=setting).status_code == 400
    for data in ({"mode": None}, {"mode": "manual", "value": True}, {"mode": "manual", "value": -1}):
        assert client.post("/api/camera/usb/focus", json=data).status_code == 400
    original = camera_focus._query
    monkeypatch.setattr(
        camera_focus,
        "_query",
        lambda fd, name: (
            {**original(fd, name), "max": 100} if name == "focus_absolute" else original(fd, name)
        ),
    )
    response = client.post("/api/camera/usb/focus", json={"mode": "manual", "value": 128})
    assert response.status_code == 400
    assert not writes and values[0x009A090C] == 1
    assert not runtime.state["camera_focus_settings"] and not runtime.config_path.exists()


def test_only_known_legacy_camera_inherits_128(tmp_path):
    path = tmp_path / "config.json"
    known = "usb-SN0002_1080P_USB_Camera_46435000_P030C00_SN0002"
    path.write_text(json.dumps({"camera_map": {known: "left", "usb-DE475": "head"}}))
    runtime = Runtime(config_path=path)
    config.load_saved(runtime)
    assert not runtime.state["config_error"]
    assert runtime.state["camera_focus_settings"] == {known: {"mode": "manual", "value": 128}}
