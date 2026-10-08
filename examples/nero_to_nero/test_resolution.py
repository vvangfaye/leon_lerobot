"""分辨率枚举、启动前校验和缩放后的真实视频读回；不访问真实设备。"""

import errno
import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from nero_platform import camera_io, cameras, config, devices, femto, recorder, resolution
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_femto import Frame
from test_platform import Clock, FakeCamera, attach_fake_pair

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset


@pytest.mark.parametrize(
    "size,expected",
    [
        ((640, 480), [1]),
        ((1280, 720), [1, 2]),
        ((1920, 1080), [1, 2, 4]),
        ((3840, 2160), [1, 2, 4, 8]),
        ((1280, 480), [1]),
        ((800, 1200), [1, 2]),
    ],
)
def test_scaling_stops_when_either_edge_reaches_threshold(size, expected):
    assert resolution.scales(*size) == expected


def test_configuration_roundtrip_validation_and_legacy_defaults(tmp_path):
    runtime = Runtime(config_path=tmp_path / "config.json")
    settings = resolution.defaults()
    settings["left"] = {"width": 1920, "height": 1080, "scale": 4}
    settings["head"]["scale"] = 2
    config.save_config(runtime, {}, {"camera_resolution": settings})
    restored = Runtime(config_path=runtime.config_path)
    config.load_saved(restored)
    assert restored.state["camera_resolution"] == settings
    saved = runtime.config_path.read_bytes()
    for bad in (
        {},
        {**settings, "head": {"width": 1280, "height": 720, "scale": 4}},
        {**settings, "left": {"width": True, "height": 480, "scale": 1}},
    ):
        with pytest.raises(ValueError):
            config.save_config(runtime, {}, {"camera_resolution": bad})
        assert runtime.config_path.read_bytes() == saved
    runtime.config_path.write_text('{"fps":30}')
    legacy = Runtime(config_path=runtime.config_path)
    config.load_saved(legacy)
    assert legacy.state["camera_resolution"] == resolution.defaults()


def test_v4l2_enumerates_formats_sizes_and_rates_without_streaming():
    formats = [int.from_bytes(b"MJPG", "little"), int.from_bytes(b"YUYV", "little")]

    def ioctl(fd, request, data):
        assert fd == 7
        number = request & 255
        index = struct.unpack_from("=I", data)[0]
        assert number in (2, 74, 75), "能力枚举不能设置模式或启动流"
        if number == 2 and index < 2:
            struct.pack_into("=I", data, 44, formats[index])
        elif number == 74 and index < 2:
            struct.pack_into("=III", data, 8, 1, *[(1920, 1080), (640, 480)][index])
        elif number == 75 and index == 0:
            pixel, width = struct.unpack_from("=II", data, 4)
            fps = 10 if pixel == formats[1] and width == 1920 else 30
            struct.pack_into("=III", data, 16, 1, 1, fps)
        else:
            raise OSError(errno.EINVAL, "end")

    with (
        patch.object(resolution.os, "open", return_value=7),
        patch.object(resolution.os, "close") as close,
        patch.object(resolution.fcntl, "ioctl", side_effect=ioctl),
    ):
        result = resolution.v4l2_profiles("/fake/camera")
    close.assert_called_once_with(7)
    assert len(result["profiles"]) == 4
    setting = {"width": 1920, "height": 1080, "scale": 4}
    assert resolution.select_fourcc(result["profiles"], setting, 30) == "MJPG"
    with pytest.raises(ValueError, match="FPS"):
        resolution.select_fourcc(result["profiles"], setting, 60)


def test_background_rgb_and_depth_resize_preserve_color_alignment_and_zero():
    rgb_camera = cameras.RecordingCamera(
        OpenCVCameraConfig(0, width=1280, height=720, color_mode=ColorMode.RGB)
    )
    rgb_camera.output_size = (640, 360)
    bgr = np.full((720, 1280, 3), [0, 0, 255], dtype=np.uint8)
    rgb = rgb_camera._postprocess_image(bgr)
    assert rgb.shape == (360, 640, 3)
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
    assert bgr[0, 0, 2] == 255
    raw = np.tile(np.array([0, 2000, 8000, 35280], dtype=np.uint16), (720, 320))
    frames = SimpleNamespace(
        get_color_frame=lambda: Frame(bgr[..., ::-1].copy(), "RGB"), get_depth_frame=lambda: Frame(raw, "Y16")
    )
    with patch.object(femto, "ob", SimpleNamespace(OBFormat=SimpleNamespace(RGB="RGB", Y16="Y16"))):
        head = femto.FemtoCamera(output_size=(640, 360))._convert(frames)
    assert head["rgb"].shape == (360, 640, 3)
    assert head["depth"].shape == (360, 640, 1)
    np.testing.assert_allclose(head["depth"][0, :2, 0], [0, 2])
    assert head["invalid_fraction"] == 0.5
    assert not head["depth"].flags.writeable


def test_capability_endpoint_uses_detected_id_and_guards_active_session(tmp_path):
    runtime = Runtime(config_path=tmp_path / "config.json")
    client = create_app(runtime).test_client()
    with (
        patch.object(cameras, "detect_cameras", return_value={"known": "/fake/camera"}),
        patch.object(resolution, "v4l2_profiles", return_value={"profiles": [], "ranges": []}) as probe,
    ):
        assert client.get("/api/camera/resolutions/left?camera_id=/etc/passwd").status_code == 503
        probe.assert_not_called()
        assert client.get("/api/camera/resolutions/left?camera_id=known").status_code == 200
        probe.assert_called_once_with("/fake/camera")
        runtime.state["connected"] = True
        assert client.get("/api/camera/resolutions/head").status_code == 409
        assert not runtime.device_busy


def test_unsupported_camera_mode_never_changes_robot_mode(tmp_path):
    runtime = Runtime(config_path=tmp_path / "config.json", dataset_root=tmp_path / "data")
    runtime.state.update(record_camera="right", camera_map={"known": "right"})
    attach_fake_pair(runtime, [])
    with (
        patch.object(cameras, "detect_cameras", return_value={"known": "/fake/camera"}),
        patch.object(cameras, "open_preview_cams"),
        patch.object(
            resolution,
            "v4l2_profiles",
            return_value={"profiles": [{"width": 1920, "height": 1080, "format": "MJPG", "fps": [30]}]},
        ),
        patch.object(devices, "transition_to_collection") as transition,
    ):
        ok, message = recorder.do_connect(runtime)
    assert not ok and "不支持" in message
    transition.assert_not_called()
    assert not runtime.collection_cams


def test_actual_scaled_recording_feature_video_and_resume_agree(tmp_path):
    runtime = Runtime(config_path=tmp_path / "config.json", dataset_root=tmp_path / "data")
    runtime.state.update(fps=10, episode_sec=0.2, camera_map={"a": "left", "b": "right"})
    runtime.state["camera_resolution"]["left"] = {"width": 1920, "height": 1080, "scale": 4}
    runtime.state["camera_resolution"]["right"] = {"width": 1280, "height": 720, "scale": 2}
    events, opened = [], []
    attach_fake_pair(runtime, events)

    class Camera(FakeCamera):
        def __init__(self, cfg):
            super().__init__(cfg)
            opened.append((cfg.width, cfg.height))
            self.frame = np.full((cfg.height, cfg.width, 3), [240, 5, 5], dtype=np.uint8)

        def read_latest(self):
            return resolution.resize(self.frame, self.output_size)

    clock = Clock(events)
    with (
        patch.object(cameras, "detect_cameras", return_value={"a": "/fake/left", "b": "/fake/right"}),
        patch.object(cameras, "open_preview_cams"),
        patch.object(camera_io, "RecordingCamera", Camera),
        patch.object(recorder, "time", clock),
        patch.object(devices, "time", clock),
    ):
        ok, msg = recorder.start_recording(runtime)
        assert ok, msg
        runtime.record_thread.join(15)
        assert not runtime.record_thread.is_alive()
        assert not runtime.state["last_error"]
        recorder.do_disconnect(runtime, reopen_preview=False)
    assert opened == [(1920, 1080), (1280, 720)]
    root = recorder.dataset_path(runtime)
    dataset = LeRobotDataset(runtime.state["repo_id"], root=root, video_backend="pyav")
    assert dataset.num_frames == 2
    for role, size in (("left", (3, 270, 480)), ("right", (3, 360, 640))):
        frame = dataset[0][f"observation.images.{role}"]
        assert tuple(frame.shape) == size
        assert float(frame[0].mean()) > 0.85 and float(frame[2].mean()) < 0.05
    report = json.loads((root / "meta/nero_timing/episode_000000.json").read_text())
    assert report["camera_resolution"] == runtime.state["camera_resolution"]
    with pytest.raises(ValueError, match="shape"):
        recorder.validate_resume(
            root,
            10,
            recorder.make_features(
                {"left": {"type": "rgb", "path": "/l"}, "right": {"type": "rgb", "path": "/r"}}
            ),
        )
