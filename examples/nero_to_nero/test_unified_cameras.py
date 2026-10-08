"""统一相机池契约：模拟设备，真实视频；禁止接触 CAN 和真实相机。"""

import copy
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from nero_platform import (
    azure_kinect,
    camera_io,
    cameras,
    config,
    devices,
    femto,
    realsense,
    recorder,
    resolution,
    timing,
)
from nero_platform.app import create_app
from nero_platform.review import LocalReview
from nero_platform.runtime import Runtime
from test_femto import FakeHead
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import LeRobotDataset


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    blocked = Mock(side_effect=AssertionError("禁止访问真实设备"))
    monkeypatch.setattr(camera_io, "RecordingCamera", blocked)
    monkeypatch.setattr(cameras, "OpenCVCamera", blocked)
    monkeypatch.setattr(devices.AgxArmFactory, "create_arm", blocked)
    monkeypatch.setattr(cameras, "detect_cameras", lambda: {})
    monkeypatch.setattr(cameras, "open_preview_cams", Mock())
    monkeypatch.setattr(femto, "list_devices", lambda: [])


@pytest.mark.parametrize(
    "selection,expected", [("head", ["head"]), ("all", ["left", "head"]), ("left", ["left", "head"])]
)
def test_role_selection_does_not_depend_on_camera_type(selection, expected):
    runtime = Runtime()
    runtime.state.update(record_camera=selection, camera_map={"usb": "head", "rgbd": "left"})
    detected = {"usb": {"type": "rgb", "path": "/fake"}, "rgbd": {"type": "rgbd", "serial": "A"}}
    assert list(cameras.recording_sources(runtime, detected)) == expected
    runtime.state.update(record_camera="head", camera_map={})
    with pytest.raises(ValueError, match="尚未分配"):
        cameras.recording_sources(runtime, detected)
    runtime.state["record_camera"] = "all"
    with pytest.raises(ValueError, match="至少分配"):
        cameras.recording_sources(runtime, detected)


def test_inventory_and_rescan_return_same_descriptors(monkeypatch):
    monkeypatch.setattr(cameras, "detect_cameras", lambda: {"usb": "/fake"})
    monkeypatch.setattr(femto, "ob", object())
    monkeypatch.setattr(femto, "list_devices", lambda: [{"serial": "A"}, {"serial": "B"}])
    client = create_app(Runtime()).test_client()
    before = client.get("/api/cameras").json["cameras"]
    after = client.post("/api/cameras/detect/depth").json["cameras"]
    assert before == after
    assert before["usb"] == {"backend": "opencv", "type": "rgb", "path": "/fake", "preview_supported": True}
    assert before["femto:A"]["backend"] == before["femto:B"]["backend"] == "orbbec"


def test_rgbd_source_resolution_is_passed_to_driver():
    with patch.object(femto, "FemtoCamera") as constructor:
        camera_io.backend({"type": "rgbd"}).create(
            {"serial": "A"}, {"width": 1920, "height": 1080, "scale": 2}, 30
        )
    constructor.assert_called_once_with(serial="A", fps=30, color_size=(1920, 1080), output_size=(960, 540))


def test_partial_open_failure_closes_every_handle_before_restoring_preview(tmp_path, monkeypatch):
    runtime = Runtime(dataset_root=tmp_path)
    runtime.state.update(record_camera="all", camera_map={"A": "left", "B": "head"})
    detected = {key: {"type": "rgbd", "serial": key} for key in ("A", "B")}
    monkeypatch.setattr(cameras, "detect_all_cameras", lambda: detected)
    events = []
    attach_fake_pair(runtime, [])
    preview = Mock()
    preview.disconnect.side_effect = lambda: events.append("preview closed")
    runtime.preview_cams["old"] = preview
    created = []

    def create(**kwargs):
        serial = kwargs["serial"]
        cam = Mock(metadata={})

        def connect():
            events.append("connect " + serial)
            if serial == "B":
                raise RuntimeError("模拟第二台 RGBD 打开失败")

        cam.connect.side_effect = connect
        cam.disconnect.side_effect = lambda: events.append("close " + serial)
        created.append(cam)
        return cam

    monkeypatch.setattr(femto, "FemtoCamera", create)
    monkeypatch.setattr(cameras, "open_preview_cams", lambda _: events.append("restore preview"))
    with patch.object(devices, "transition_to_collection") as transition:
        ok, error = recorder.do_connect(runtime)
    assert not ok and "第二台" in error
    transition.assert_not_called()
    assert events == ["preview closed", "connect A", "connect B", "close A", "close B", "restore preview"]
    assert not runtime.collection_cams and not runtime.state["camera_sources"]
    assert all(cam.disconnect.call_count == 1 for cam in created)


@pytest.mark.parametrize("second_backend", ["azure_kinect", "realsense"])
def test_two_rgbd_devices_and_rgb_head_record_independently(tmp_path, monkeypatch, second_backend):
    runtime = Runtime(dataset_root=tmp_path)
    runtime.state.update(
        fps=10, episode_sec=0.3, record_camera="all", camera_map={"A": "left", "B": "right", "C": "head"}
    )
    detected = {
        "A": {"type": "rgbd", "serial": "A"},
        "B": {"type": "rgbd", "backend": second_backend, "serial": "B"},
        "C": {"type": "rgb", "path": "/fake"},
    }
    monkeypatch.setattr(cameras, "detect_all_cameras", lambda: detected)
    opened = {}

    def create(**kwargs):
        cam = FakeHead(**kwargs)
        cam.depth[:] = 1 if kwargs["serial"] == "A" else 2
        opened[kwargs["serial"]] = cam
        return cam

    monkeypatch.setattr(femto, "FemtoCamera", create)
    monkeypatch.setattr(azure_kinect, "AzureKinectCamera", create)
    monkeypatch.setattr(realsense, "RealSenseCamera", create)
    monkeypatch.setattr(camera_io, "RecordingCamera", FakeCamera)
    events = []
    attach_fake_pair(runtime, events)
    dataset = MemoryDataset(events)
    dataset.root = tmp_path / "report"
    monkeypatch.setattr(recorder.LeRobotDataset, "create", lambda **_: dataset)
    with patch.object(devices, "transition_to_collection"):
        ok, error = recorder.do_connect(runtime)
    assert ok, error
    assert set(runtime.collection_cams) == {"left", "right", "head"}
    clock = Clock(events)
    with patch.object(recorder, "time", clock), patch.object(devices, "time", clock):
        recorder.record_worker(runtime, ["left", "right", "head"])
    assert runtime.state["record_phase"] == "saved"
    assert opened["A"].reads == opened["B"].reads == 3
    for frame in dataset.frames:
        assert np.all(frame["observation.images.left_depth"] == 1)
        assert np.all(frame["observation.images.right_depth"] == 2)
        assert "observation.images.head_depth" not in frame
    stats = runtime.state["record_timing"]["cameras"]
    assert stats["left_depth"]["frames"] == stats["right_depth"]["frames"] == 3
    report = json.loads(timing.report_path(dataset.root, 0).read_text())
    assert report["camera_devices"]["left"]["serial"] == "A"
    assert report["camera_devices"]["right"]["serial"] == "B"
    runtime.state["record_phase"] = "recording"
    with patch("nero_platform.app.time.monotonic", return_value=0.21):
        client = create_app(runtime).test_client()
        for name in ("left", "left_depth", "right", "right_depth"):
            assert client.get(f"/api/record/camera/{name}/frame").status_code == 200
        assert client.get("/api/record/camera/head_depth/frame").status_code == 404
    assert opened["A"].reads == opened["B"].reads == 3, "预览不额外读取 SDK"
    recorder.do_disconnect(runtime, reopen_preview=False)
    assert not opened["A"].connected and not opened["B"].connected


@pytest.mark.parametrize(
    "info", [{"type": "rgbd", "backend": "unknown"}, {"type": "rgbd", "backend": "opencv"}]
)
def test_unknown_backend_rejected_before_device_operations(info):
    with pytest.raises(ValueError, match="后端/类型"):
        recorder.make_features({"head": info})


@pytest.mark.parametrize("serial", ["A", ""])
def test_legacy_head_setting_cannot_silently_drop_camera(tmp_path, serial):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"head_camera_mode": "rgbd", "head_camera_serial": serial}))
    original = path.read_bytes()
    runtime = Runtime(config_path=path)
    config.load_saved(runtime)
    assert not runtime.state["config_error"]
    assert "head" in runtime.state["camera_map"].values()
    assert runtime.state["camera_config_notice"]
    assert path.read_bytes() == original
    if serial:
        assert runtime.state["camera_map"] == {"femto:A": "head"}
    else:
        runtime.state["record_camera"] = "head"
        with pytest.raises(ValueError, match="未检测到"):
            cameras.recording_sources(runtime, {})


def test_multi_depth_real_video_roundtrip_and_resume_validation(tmp_path):
    sources = {role: {"type": "rgbd", "serial": role} for role in ("left", "head")}
    sizes = resolution.defaults()
    for role in sources:
        sizes[role] = {"width": 128, "height": 96, "scale": 1}
    features = recorder.make_features(sources, camera_resolution=sizes)
    root = tmp_path / "nero/multi"
    dataset = LeRobotDataset.create(
        "nero/multi",
        root=root,
        fps=30,
        robot_type="nero",
        features=features,
        image_writer_threads=1,
        concatenate_videos=False,
        depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=10.0),
    )
    try:
        for _ in range(2):
            frame = {
                "action": np.zeros(8, np.float32),
                "observation.state": np.zeros(8, np.float32),
                "task": "测试",
            }
            for role, value in (("left", 1.0), ("head", 2.0)):
                frame[f"observation.images.{role}"] = np.full((96, 128, 3), 150, np.uint8)
                frame[f"observation.images.{role}_depth"] = np.full((96, 128, 1), value, np.float32)
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        dataset.finalize()
    loaded = LeRobotDataset("nero/multi", root=root, video_backend="pyav", depth_output_unit="m")
    for role, value in (("left", 1.0), ("head", 2.0)):
        np.testing.assert_allclose(loaded[0][f"observation.images.{role}_depth"], value, atol=0.003)
        frame = LocalReview(tmp_path).depth_frame("nero/multi", 0, f"{role}_depth", 0)
        np.testing.assert_allclose(frame, value, atol=0.003)
    recorder.validate_resume(root, 30, features)
    incompatible = copy.deepcopy(features)
    incompatible["observation.images.depth"] = incompatible.pop("observation.images.left_depth")
    with pytest.raises(ValueError, match="相机字段不一致"):
        recorder.validate_resume(root, 30, incompatible)
    # 每一路深度都检查编码范围，不能只检查第一台 RGBD。
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.head_depth"]["info"]["video.depth_min"] = 0.01
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="深度编码范围"):
        recorder.validate_resume(root, 30, features)


@pytest.mark.parametrize("legacy_key", ["femto", "head_camera"])
def test_legacy_report_device_metadata_remains_visible(tmp_path, legacy_key):
    cadence = timing.EpisodeTiming(30, 0)
    for index in range(2):
        cadence.finish(index, index / 30, index / 30 + 0.001, index, None)
    report = {
        "schema_version": 2,
        "episode_index": 0,
        "summary": cadence.summary(2 / 30),
        "samples": cadence.samples,
        legacy_key: {"serial": "OLD", "alignment": "hardware_D2C"},
    }
    path = timing.save_report(tmp_path, 0, report)
    before = path.read_bytes()
    quality, _ = LocalReview(tmp_path).report(tmp_path, {"episode_index": 0, "length": 2}, 30)
    assert quality["camera_devices"]["legacy_rgbd"]["serial"] == "OLD"
    assert path.read_bytes() == before


def test_review_checks_each_rgbd_clock_independently(tmp_path):
    cadence = timing.EpisodeTiming(30, 0)
    for index in range(2):
        details = {}
        for role, clock_base in (("left", 1000), ("head", 90000000)):
            details[role + "_depth"] = {
                "capture_timestamp": index + 1,
                "age_ms": 1.0,
                "invalid_fraction": 0.0,
                "above_range_fraction": 0.0,
                "rgb_timestamp_us": clock_base + index * 33333,
                "depth_timestamp_us": clock_base + index * 33333,
            }
        cadence.finish(index, index / 30, index / 30 + 0.001, index, None, details)
    report = {
        "schema_version": 3,
        "episode_index": 0,
        "summary": cadence.summary(2 / 30),
        "samples": cadence.samples,
    }
    # 旧检查结果缺失时，回放按每台设备分别复核，不能把另一台设备的时间轴当作自己的上一帧。
    for sample in report["samples"]:
        for details in sample["cameras"].values():
            for key in (
                "timestamp_check_version",
                "timestamp_status",
                "timestamp_source",
                "rgb_depth_skew_ms",
            ):
                details.pop(key, None)
    timing.save_report(tmp_path, 0, report)
    quality, normalized = LocalReview(tmp_path).report(tmp_path, {"episode_index": 0, "length": 2}, 30)
    assert normalized is not None
    for name in ("left_depth", "head_depth"):
        stats = quality["summary"]["cameras"][name]
        assert stats["timestamp_checked_frames"] == 1
        assert stats["max_rgb_depth_skew_ms"] == 0
    events = LocalReview.events(normalized, [0, 1 / 30], 30)
    assert len(events) == 1
    text = json.dumps(events, ensure_ascii=False)
    assert "左深度" in text and "头部深度" in text


def test_old_depth_video_remains_readable(tmp_path):
    sizes = resolution.defaults()
    sizes["head"] = {"width": 128, "height": 96, "scale": 1}
    features = recorder.make_features({"head": {"type": "rgbd"}}, camera_resolution=sizes)
    features["observation.images.depth"] = features.pop("observation.images.head_depth")
    root = tmp_path / "nero/legacy"
    dataset = LeRobotDataset.create(
        "nero/legacy",
        root=root,
        fps=30,
        robot_type="nero",
        features=features,
        image_writer_threads=1,
        concatenate_videos=False,
        depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=10.0),
    )
    try:
        dataset.add_frame(
            {
                "action": np.zeros(8, np.float32),
                "observation.state": np.zeros(8, np.float32),
                "task": "旧数据",
                "observation.images.head": np.full((96, 128, 3), 150, np.uint8),
                "observation.images.depth": np.full((96, 128, 1), 1.0, np.float32),
            }
        )
        dataset.save_episode()
    finally:
        dataset.finalize()
    np.testing.assert_allclose(
        LocalReview(tmp_path).depth_frame("nero/legacy", 0, "depth", 0), 1.0, atol=0.003
    )
    response = (
        create_app(Runtime(dataset_root=tmp_path))
        .test_client()
        .get(
            "/api/review/depth", query_string={"repo_id": "nero/legacy", "episode_index": 0, "frame_index": 0}
        )
    )
    assert response.status_code == 200
