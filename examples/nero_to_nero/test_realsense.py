"""RealSense 契约：模式、时钟域、数据所有权和管线生命周期。真实设备由冒烟验证。"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from nero_platform import camera_io, camera_preview, depth_discovery, realsense as driver


def profile(stream, size=(1280, 720), fps=30):
    value = SimpleNamespace(
        width=lambda: size[0],
        height=lambda: size[1],
        fps=lambda: fps,
        stream_type=lambda: stream,
        format=lambda: "rgb8" if stream == "color" else "z16",
        is_video_stream_profile=lambda: True,
    )
    value.as_video_stream_profile = lambda: value
    return value


def frames(index=1, domain="global", age_ms=0):
    rgb = np.full((720, 1280, 3), [220, 20, 10], np.uint8)
    depth = np.full((720, 1280), 1234, np.uint16)
    depth[0, 0] = 0
    now_ms = time.time() * 1000 - age_ms

    def frame(data):
        return SimpleNamespace(
            get_data=lambda: data,
            get_frame_number=lambda: index,
            get_timestamp=lambda: now_ms,
            get_frame_timestamp_domain=lambda: domain,
            supports_frame_metadata=lambda _: True,
            get_frame_metadata=lambda _: now_ms,
        )

    return SimpleNamespace(get_color_frame=lambda: frame(rgb), get_depth_frame=lambda: frame(depth))


@pytest.fixture
def sdk(monkeypatch):
    modes = [profile(s, size) for s in ("color", "depth") for size in ((1280, 720), (640, 480))]
    modes += [profile("color", (1280, 800)), profile("color", fps=60)]
    sensor = SimpleNamespace(
        is_depth_sensor=lambda: True,
        get_stream_profiles=lambda: modes,
        supports=lambda _: True,
        set_option=Mock(),
        get_option=lambda _: 1,
    )
    device = SimpleNamespace(
        get_info=lambda key: {"serial": "R", "name": "D455"}[key],
        query_sensors=lambda: [sensor],
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001),
    )
    context = SimpleNamespace(query_devices=lambda: [device])
    config = SimpleNamespace(enable_device=Mock(), enable_stream=Mock())
    active = SimpleNamespace(get_device=lambda: device, get_stream=lambda s: profile(s))
    pipeline = Mock(start=Mock(return_value=active))
    sequence = 0

    def wait(timeout):
        nonlocal sequence
        assert timeout == 100
        time.sleep(0.005)
        sequence += 1
        return True, frames(sequence)

    pipeline.try_wait_for_frames.side_effect = wait
    sdk = SimpleNamespace(
        context=lambda: context,
        config=lambda: config,
        pipeline=Mock(return_value=pipeline),
        align=lambda _: SimpleNamespace(process=lambda f: f),
        camera_info=SimpleNamespace(serial_number="serial", name="name"),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
        option=SimpleNamespace(global_time_enabled="global"),
        frame_metadata_value=SimpleNamespace(time_of_arrival="arrival"),
        timestamp_domain=SimpleNamespace(
            hardware_clock="hardware", global_time="global", system_time="system"
        ),
    )
    monkeypatch.setattr(driver, "rs", sdk)
    return SimpleNamespace(
        module=sdk, device=device, modes=modes, pipeline=pipeline, config=config, active=active
    )


def test_modes_are_device_derived_and_intersect_depth_rates_without_start(sdk):
    profiles = driver.list_profiles("R")["profiles"]
    assert {(p["width"], p["height"]) for p in profiles} == {(1280, 720), (1280, 800), (640, 480)}
    assert all(p["fps"] == [30] for p in profiles), "深度无 60 FPS 时不能向用户宣称可用 RGBD 60 FPS"
    sdk.module.pipeline.assert_not_called()
    assert driver._depth_size({(1280, 720, 30), (640, 480, 30)}, (1280, 800), 30) == (1280, 720)


def test_wrong_serial_or_unsupported_mode_does_not_start(sdk):
    for args, error in [({"serial": "other"}, RuntimeError), ({"serial": "R", "fps": 25}, ValueError)]:
        with pytest.raises(error):
            driver.RealSenseCamera(**args).connect()
    sdk.pipeline.start.assert_not_called()


def test_backend_uses_role_settings_and_own_driver(sdk, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(driver, "RealSenseCamera", constructor)
    camera_io.backend({"type": "rgbd", "backend": "realsense"}).create(
        {"serial": "R"}, {"width": 1280, "height": 800, "scale": 2}, 30
    )
    constructor.assert_called_once_with(serial="R", fps=30, color_size=(1280, 800), output_size=(640, 400))
    assert camera_preview.supported({"type": "rgbd", "backend": "realsense"})
    assert next(d for d in depth_discovery.DETECTORS if d.backend == "realsense").recording_supported


def test_snapshot_keeps_rgb_order_meters_zeros_and_shared_pair(sdk):
    cam = driver.RealSenseCamera(serial="R", output_size=(640, 360))
    try:
        cam.connect()
        sample = cam.read_snapshot()
        assert sample["rgb"].shape == (360, 640, 3)
        assert sample["rgb"][0, 0].tolist() == [220, 20, 10]
        assert sample["depth"].shape == (360, 640, 1)
        assert sample["depth"][0, 0, 0] == 0
        assert sample["depth"][1, 1, 0] == pytest.approx(1.234)
        assert sample["depth"].dtype == np.float32 and not sample["rgb"].flags.writeable
        images, stats = camera_io.sample("head", {"type": "rgbd", "backend": "realsense"}, cam)
        assert set(images) == {"head", "head_depth"}
        assert stats["head"]["capture_timestamp"] == stats["head_depth"]["capture_timestamp"]
        sdk.config.enable_device.assert_called_once_with("R")
    finally:
        cam.disconnect()
    sdk.pipeline.stop.assert_called_once()
    cam.disconnect()
    sdk.pipeline.stop.assert_called_once()


@pytest.mark.parametrize(
    "domain,field", [("global", "global_timestamp_us"), ("hardware", "timestamp_us"), ("system", None)]
)
def test_timestamp_domains_are_not_mixed_or_fabricated(sdk, domain, field):
    sample = driver._frame_clock(frames(domain=domain).get_color_frame(), "rgb")
    for name in ("global_timestamp_us", "timestamp_us"):
        assert (sample["rgb_" + name] is not None) == (name == field)
    assert sample["rgb_timestamp_domain"] == domain


def test_cache_owns_pixels_and_repeated_stream_does_not_refresh_age(sdk):
    cam = driver.RealSenseCamera(serial="R")
    cam._align = SimpleNamespace(process=lambda f: f)
    cam._depth_scale = 0.001
    item = frames()
    sample = cam._convert(item)
    item.get_color_frame().get_data()[:] = 0
    assert sample["rgb"][0, 0, 0] == 220, "SDK 缓冲复用不得改写已交给录制的帧"
    cam._publish(sample)
    cam._publish({**sample, "capture_timestamp": sample["capture_timestamp"] + 1, "depth_frame_index": 2})
    assert cam._latest is sample
    cam._thread = Mock(is_alive=lambda: True)
    cam._started = True
    sample["capture_timestamp"] -= 1
    with pytest.raises(TimeoutError, match="过期"):
        cam.read_snapshot()


def test_old_sdk_frames_and_incomplete_frames_are_rejected_before_alignment(sdk):
    cam = driver.RealSenseCamera(serial="R")
    cam._align = Mock()
    with pytest.raises(TimeoutError, match="积压"):
        cam._convert(frames(age_ms=1000))
    item = frames()
    item.get_depth_frame = lambda: None
    with pytest.raises(ValueError, match="缺少"):
        cam._convert(item)
    cam._align.process.assert_not_called()


def test_warmup_requires_continuous_fresh_pairs(sdk):
    cam = driver.RealSenseCamera(serial="R")
    cam._align = SimpleNamespace(process=lambda f: f)
    cam._depth_scale = 0.001
    for index in range(1, 15):
        cam._publish(cam._convert(frames(index)))
    assert not cam._ready.is_set()
    cam._publish(cam._convert(frames(15, age_ms=250)))
    assert cam._warm_frames == 0
    for index in range(16, 31):
        cam._publish(cam._convert(frames(index)))
    assert cam._ready.is_set()


def test_actual_profile_mismatch_stops_pipeline_before_ready(sdk):
    sdk.active.get_stream = lambda s: profile(s, fps=15)
    cam = driver.RealSenseCamera(serial="R")
    with pytest.raises(RuntimeError, match="实际分辨率"):
        cam.connect()
    sdk.pipeline.stop.assert_called_once()
    assert cam._pipeline is None


def test_start_failure_clears_owner_and_shutdown_waits_for_reader(sdk):
    sdk.pipeline.start.side_effect = RuntimeError("设备占用")
    cam = driver.RealSenseCamera(serial="R")
    with pytest.raises(RuntimeError, match="设备占用"):
        cam.connect()
    assert cam._pipeline is None and not cam.connected
    sdk.pipeline.stop.assert_not_called()  # SDK start 失败自行回滚；未启动时 stop 会抛异常。
    cam._pipeline, cam._started = sdk.pipeline, True
    cam._thread = Mock(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="保留句柄"):
        cam.disconnect()
    sdk.pipeline.stop.assert_not_called()
