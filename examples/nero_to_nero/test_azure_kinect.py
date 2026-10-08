"""Azure Kinect 契约测试；只使用模拟 SDK/USB，覆盖设备身份、RGBD 与释放。"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from nero_platform import azure_kinect as azure, camera_io, camera_preview, depth_discovery


def usb_entry(root, port, product="097c", serial="K", speed="5000"):
    path = root / port
    path.mkdir()
    for key, value in {"idVendor": "045e", "idProduct": product, "serial": serial, "speed": speed}.items():
        (path / key).write_text(value)


def capture(size=(1280, 720), stamp=100_000):
    width, height = size
    color = np.zeros((height, width, 4), dtype=np.uint8)
    color[:] = [10, 20, 200, 255]
    depth = np.full((height, width), 1234, dtype=np.uint16)
    depth[0, 0] = 0
    now = time.perf_counter_ns()
    return SimpleNamespace(
        color=color,
        depth=np.full((576, 640), 1234, dtype=np.uint16),
        transformed_depth=depth,
        color_timestamp_usec=stamp,
        depth_timestamp_usec=stamp + 100,
        color_system_timestamp_nsec=now,
        depth_system_timestamp_nsec=now,
    )


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    usb_entry(tmp_path, "10-1.2")
    monkeypatch.setattr(azure, "USB_ROOT", tmp_path)
    handles = []

    class Device:
        def __init__(self, config, device_id, thread_safe):
            assert thread_safe is False, "SDK 等帧必须释放 GIL，不能阻塞机器人控制"
            self.config, self.serial = config, "K"
            self.opened = self.is_running = False
            self.index = 0
            self.stops = 0
            handles.append(self)

        def open(self):
            self.opened = True

        def close(self):
            self.opened = False

        def start(self):
            self.is_running = True

        def stop(self):
            self.stops += 1
            self.is_running = self.opened = False

        def get_capture(self, timeout):
            assert timeout == 100
            time.sleep(0.005)
            self.index += 1
            return capture(stamp=self.index * 33333)

    module = SimpleNamespace(
        connected_device_count=lambda: 1,
        PyK4A=Device,
        Config=lambda **kwargs: SimpleNamespace(**kwargs),
        ColorResolution=SimpleNamespace(**{name: name for name in azure.COLOR_MODES.values()}),
        ImageFormat=SimpleNamespace(COLOR_BGRA32="BGRA32"),
        DepthMode=SimpleNamespace(NFOV_UNBINNED="NFOV_UNBINNED"),
        FPS=SimpleNamespace(FPS_5=5, FPS_15=15, FPS_30=30),
        K4ATimeoutException=TimeoutError,
    )
    monkeypatch.setattr(azure, "k4a", module)
    return module, handles


def test_detects_once_by_physical_serial_without_opening_sdk(sdk, tmp_path):
    module, handles = sdk
    usb_entry(tmp_path, "10-1.1", product="097d")
    usb_entry(tmp_path, "9-1.3", product="097e", speed="480")
    assert azure.list_devices() == [{"name": "Azure Kinect DK", "serial": "K", "usb_speed_mbps": 5000.0}]
    assert not handles
    detector = next(d for d in depth_discovery.DETECTORS if d.backend == "azure_kinect")
    assert detector.recording_supported
    assert camera_preview.supported({"type": "rgbd", "backend": "azure_kinect"})


def test_sdk_missing_is_reported_without_hiding_other_detectors(sdk, monkeypatch):
    monkeypatch.setattr(azure, "k4a", None)
    with pytest.raises(ImportError, match="libk4a"):
        azure.list_devices()


def test_profiles_and_unsupported_modes_do_not_open_device(sdk):
    _, handles = sdk
    profiles = azure.list_profiles("K")["profiles"]
    assert len(profiles) == 6
    high = next(p for p in profiles if p["height"] == 3072)
    assert high["fps"] == [5, 15]
    for kwargs in ({"color_size": (640, 480)}, {"fps": 25}, {"color_size": (4096, 3072), "fps": 30}):
        with pytest.raises(ValueError):
            azure.AzureKinectCamera(serial="K", **kwargs).connect()
    assert not handles


def test_usb2_is_visible_but_cannot_start_rgbd(sdk, tmp_path):
    (tmp_path / "10-1.2" / "speed").write_text("480")
    assert azure.list_devices()[0]["usb_speed_mbps"] == 480
    with pytest.raises(RuntimeError, match="USB 2.0"):
        azure.AzureKinectCamera(serial="K").connect()
    assert not sdk[1]


def test_background_snapshot_alignment_color_order_scaling_and_close(sdk):
    cam = azure.AzureKinectCamera(serial="K", output_size=(640, 360))
    try:
        cam.connect()
        snapshot = cam.read_snapshot()
        assert snapshot["rgb"].shape == (360, 640, 3)
        assert snapshot["rgb"][0, 0].tolist() == [200, 20, 10]
        assert snapshot["depth"].shape == (360, 640, 1)
        assert snapshot["depth"].dtype == np.float32
        assert snapshot["depth"][0, 0, 0] == 0
        assert snapshot["depth"][1, 1, 0] == pytest.approx(1.234)
        assert not snapshot["depth"].flags.writeable
        assert cam.metadata["alignment"] == "calibrated_D2C"
        assert cam.metadata["global_timestamp_enabled"] is False
        assert sdk[1][0].config.synchronized_images_only is True
        images, stats = camera_io.sample("left", {"type": "rgbd", "backend": "azure_kinect"}, cam)
        assert set(images) == {"left", "left_depth"}
        assert stats["left"]["capture_timestamp"] == stats["left_depth"]["capture_timestamp"]
    finally:
        cam.disconnect()
    assert not cam.connected and not sdk[1][0].opened
    assert sdk[1][0].stops == 1
    cam.disconnect()


def test_incomplete_or_misaligned_capture_rejected():
    cam = azure.AzureKinectCamera()
    item = capture()
    item.depth = None
    with pytest.raises(ValueError, match="缺少"):
        cam._convert(item)
    item = capture()
    item.transformed_depth = np.zeros((576, 640), dtype=np.uint16)
    with pytest.raises(ValueError, match="D2C"):
        cam._convert(item)


def test_sdk_repeated_frame_does_not_refresh_age_and_stale_queue_is_rejected():
    cam = azure.AzureKinectCamera()
    item = capture()
    first = cam._convert(item)
    cam._publish(first)
    duplicate = cam._convert(item)
    cam._publish(duplicate)
    assert cam._latest is first
    old = capture(stamp=200_000)
    old.color_system_timestamp_nsec -= 1_000_000_000
    old.depth_system_timestamp_nsec -= 1_000_000_000
    with pytest.raises(TimeoutError, match="积压"):
        cam._publish(cam._convert(old))


def test_partial_start_failure_releases_selected_handle(sdk, monkeypatch):
    module, handles = sdk

    def fail(device):
        raise RuntimeError("模拟深度启动失败")

    monkeypatch.setattr(module.PyK4A, "start", fail)
    cam = azure.AzureKinectCamera(serial="K")
    with pytest.raises(RuntimeError, match="模拟深度"):
        cam.connect()
    assert handles[0].stops == 1 and not handles[0].opened
    assert cam._device is None


def test_selection_uses_serial_not_usb_enumeration_order(sdk, monkeypatch):
    module, handles = sdk
    constructor = module.PyK4A

    def create(*args, **kwargs):
        device = constructor(*args, **kwargs)
        device.serial = ["OTHER", "K"][kwargs["device_id"]]
        return device

    monkeypatch.setattr(module, "PyK4A", create)
    monkeypatch.setattr(module, "connected_device_count", lambda: 2)
    cam = azure.AzureKinectCamera(serial="K")
    try:
        cam.connect()
        assert not handles[0].opened and cam._device is handles[1]
    finally:
        cam.disconnect()


def test_shutdown_does_not_close_sdk_while_reader_still_runs():
    cam = azure.AzureKinectCamera()
    cam._device = Mock()
    cam._thread = Mock(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="保留句柄"):
        cam.disconnect()
    cam._device.stop.assert_not_called()
    cam._device.close.assert_not_called()
