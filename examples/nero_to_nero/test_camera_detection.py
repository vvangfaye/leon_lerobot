"""设备身份与分类型检测：不启动视频流，不访问真实 SDK。"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from nero_platform import camera_io, cameras, depth_discovery, femto, realsense
from nero_platform.app import create_app
from nero_platform.runtime import Runtime


def empty_serial_device_list():
    devices = []
    for serial in ("A", "B"):
        info = SimpleNamespace(get_serial_number=lambda value=serial: value)
        devices.append(SimpleNamespace(get_device_info=lambda value=info: value))
    return SimpleNamespace(
        get_count=lambda: 2,
        get_device_name_by_index=lambda index: "Orbbec Femto Mega",
        get_device_serial_number_by_index=lambda index: "",
        get_device_by_index=lambda index: devices[index],
        get_device_by_serial_number=Mock(side_effect=AssertionError("列表为空时不可依赖 SDK 序列号查找")),
    ), devices


def test_femto_missing_list_serial_uses_device_info_and_selects_correct_camera():
    listing, devices = empty_serial_device_list()
    assert femto.device_entries(listing) == [
        {"name": "Orbbec Femto Mega", "serial": "A"},
        {"name": "Orbbec Femto Mega", "serial": "B"},
    ]
    assert femto.device_by_serial(listing, "B") is devices[1]
    assert femto.device_by_serial(listing, "A") is devices[0]
    with pytest.raises(RuntimeError, match="未找到"):
        femto.device_by_serial(listing, "MISSING")
    listing.get_device_by_serial_number.assert_not_called()


def test_femto_profile_query_uses_resolved_device_without_starting_stream(monkeypatch):
    listing, devices = empty_serial_device_list()
    listing.get_count = lambda: 1
    pipeline = Mock()
    pipeline.get_stream_profile_list.return_value = []
    sdk = SimpleNamespace(
        Context=lambda: SimpleNamespace(query_devices=lambda: listing),
        Pipeline=Mock(return_value=pipeline),
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color"),
    )
    monkeypatch.setattr(femto, "ob", sdk)
    assert femto.list_profiles("A")["serial"] == "A"
    sdk.Pipeline.assert_called_once_with(devices[0])
    pipeline.start.assert_not_called()


def test_depth_backends_report_sdk_errors_separately(monkeypatch):
    detectors = (
        depth_discovery.Detector("奥比 Femto", "femto", "orbbec", lambda: [{"serial": "F", "name": "Femto"}]),
        depth_discovery.Detector(
            "RealSense", "realsense", "realsense", Mock(side_effect=ImportError("未安装 pyrealsense2"))
        ),
    )
    monkeypatch.setattr(depth_discovery, "DETECTORS", detectors)
    inventory, status = depth_discovery.detect()
    assert list(inventory) == ["femto:F"]
    assert status["orbbec"]["count"] == 1
    assert status["realsense"]["state"] == "missing_sdk"
    assert "pyrealsense2" in status["realsense"]["message"]
    detectors[1].enumerate_devices.side_effect = RuntimeError("USB 权限错误")
    _, status = depth_discovery.detect()
    assert status["realsense"]["state"] == "error"
    assert "USB 权限错误" in status["realsense"]["message"]


def test_realsense_detection_ignores_tracking_only_and_never_starts_pipeline(monkeypatch):
    def device(serial, depth):
        return SimpleNamespace(
            query_sensors=lambda: [SimpleNamespace(is_depth_sensor=lambda: depth)],
            get_info=lambda key: {"serial": serial, "name": "RealSense D435"}[key],
        )

    context = SimpleNamespace(query_devices=lambda: [device("D435", True), device("T265", False)])
    sdk = SimpleNamespace(
        context=Mock(return_value=context),
        camera_info=SimpleNamespace(name="name", serial_number="serial"),
        pipeline=Mock(side_effect=AssertionError("检测禁止开启视频流")),
    )
    monkeypatch.setattr(realsense, "rs", sdk)
    assert depth_discovery.list_realsense() == [{"name": "RealSense D435", "serial": "D435"}]
    sdk.pipeline.assert_not_called()


def test_rgb_detector_does_not_double_count_depth_uvc_nodes(tmp_path):
    for name in (
        "usb-webcam-video-index0",
        "usb-Orbbec_Femto_Mega-video-index0",
        "usb-Intel_RealSense_D435-video-index0",
        "usb-Microsoft_Azure_Kinect_4K_Camera_123-video-index0",
        "usb-Microsoft_Azure_Kinect_Depth_Camera_123-video-index0",
    ):
        (tmp_path / name).touch()
    with patch.object(cameras, "Path", return_value=tmp_path):
        assert list(cameras.detect_cameras()) == ["usb-webcam"]


def test_separate_detection_preserves_other_type_and_role_assignments(monkeypatch):
    runtime = Runtime()
    runtime.state["camera_map"] = {"usb:A": "head"}
    client = create_app(runtime).test_client()
    monkeypatch.setattr(cameras, "detect_cameras", lambda: {"usb:A": "/fake/A", "usb:B": "/fake/B"})
    depth = {"femto:F": {"type": "rgbd", "backend": "orbbec", "serial": "F"}}
    detection = {"orbbec": {"label": "奥比 Femto", "state": "ok", "count": 1, "message": "检测到 1 台"}}
    enumerate_depth = Mock(return_value=(depth, detection))
    monkeypatch.setattr(depth_discovery, "detect", enumerate_depth)
    preview = Mock()
    monkeypatch.setattr(cameras, "open_preview_cams", preview)
    result = client.post("/api/cameras/detect/rgb")
    assert result.status_code == 200
    assert set(result.json["cameras"]) == {"usb:A", "usb:B"}
    enumerate_depth.assert_not_called()
    result = client.post("/api/cameras/detect/depth")
    assert set(result.json["cameras"]) == {"usb:A", "usb:B", "femto:F"}
    assert preview.call_count == 1, "检测深度不重开 RGB 预览"
    assert result.json["camera_map"] == {"usb:A": "head"}
    for _ in range(3):
        assert len(client.get("/api/cameras").json["cameras"]) == 3
    enumerate_depth.assert_called_once()
    monkeypatch.setattr(cameras, "detect_cameras", lambda: {"usb:B": "/fake/B"})
    result = client.post("/api/cameras/detect/rgb")
    assert set(result.json["cameras"]) == {"usb:B", "femto:F"}
    assert result.json["camera_map"] == {"usb:A": "head"}, "失联设备用途仍留给用户处理"


def test_camera_detection_is_locked_during_session_or_another_scan(monkeypatch):
    runtime = Runtime()
    client = create_app(runtime).test_client()
    scan = Mock(side_effect=AssertionError("操作期间不得重新枚举"))
    monkeypatch.setattr(cameras, "scan_inventory", scan)
    for connected, busy in ((True, False), (False, True)):
        runtime.state["connected"] = connected
        runtime.device_busy = {"camera_detection"} if busy else set()
        for kind in ("rgb", "depth"):
            assert client.post("/api/cameras/detect/" + kind).status_code == 409
        assert client.get("/api/cameras").status_code == 200
    scan.assert_not_called()
    assert client.post("/api/cameras/detect/unknown").status_code == 400


def test_detection_only_descriptor_is_rejected_and_realsense_has_own_backend():
    with pytest.raises(ValueError, match="录制驱动尚未接入"):
        camera_io.backend({"type": "rgbd", "backend": "realsense", "recording_supported": False})
    assert camera_io.backend({"type": "rgbd", "backend": "realsense"}) is camera_io.RealSenseBackend
