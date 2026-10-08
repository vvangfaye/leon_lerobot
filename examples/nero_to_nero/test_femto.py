"""RGBD 设备协议、缓存、配置与真实深度编码读回；全部设备为模拟对象。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from nero_platform import (
    camera_io,
    cameras,
    config,
    devices,
    femto,
    recorder,
    resolution,
    rgbd_timing,
    timing,
)
from nero_platform.app import create_app
from nero_platform.review import LocalReview
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, MemoryDataset, attach_fake_pair

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import LeRobotDataset


class Profile:
    def __init__(self, width: int, height: int, format: str, fps: int = 30):
        self.width, self.height, self.format, self.fps = width, height, format, fps

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_format(self):
        return self.format

    def get_fps(self):
        return self.fps


class Frame(Profile):
    def __init__(self, values: np.ndarray, format: str, scale: float = 0.25):
        super().__init__(values.shape[1], values.shape[0], format)
        self.values, self.scale = values, scale

    def get_data(self):
        return self.values.tobytes()

    def get_depth_scale(self):
        return self.scale

    def get_timestamp_us(self):
        return 123000

    def get_system_timestamp_us(self):
        return 1_000_000

    def get_index(self):
        return 4


class Pipeline:
    def __init__(self):
        self.closed = threading.Event()
        self.started = False
        self.color = Frame(np.zeros((6, 8, 3), dtype=np.uint8), "RGB")
        self.depth = Frame(np.full((6, 8), 2000, dtype=np.uint16), "Y16")
        self.frames = SimpleNamespace(get_color_frame=lambda: self.color, get_depth_frame=lambda: self.depth)
        self.reads = 0
        self.sync_calls = []

    def get_stream_profile_list(self, sensor):
        return [Profile(8, 6, "RGB")]

    def get_d2c_depth_profile_list(self, color, align):
        return [Profile(8, 6, "Y16")]

    def enable_frame_sync(self):
        self.sync_calls.append("enable")

    def disable_frame_sync(self):
        self.sync_calls.append("disable")

    def start(self, cfg):
        self.started = True

    def wait_for_frames(self, timeout):
        self.reads += 1
        # 第一帧用于启动清理；第二次读排空队列；第三次才供正式缓存使用。
        if self.reads in (1, 3):
            return self.frames
        self.closed.wait(timeout / 1000)
        return None

    def stop(self):
        self.closed.set()


class FakeHead:
    def __init__(self, **kwargs):
        self.serial = kwargs.get("serial") or "FAKE-HEAD"
        self.metadata = {"serial": self.serial, "fps": 30}
        self.connected = True
        self.reads = 0
        self.rgb = np.full((720, 1280, 3), [255, 0, 0], dtype=np.uint8)
        self.depth = np.full((720, 1280, 1), 0.5, dtype=np.float32)
        self.timestamps = rgbd_timing.TimestampTracker()
        self.timestamps.assess({"capture_timestamp": 0, "rgb_timestamp_us": 1, "depth_timestamp_us": 1})

    def connect(self):
        return self

    def disconnect(self):
        self.connected = False

    def read_rgbd(self, **kwargs):
        raise AssertionError("录制线程不得调用阻塞式 RGBD 读取")

    def read_snapshot(self):
        self.reads += 1
        snapshot = {
            "rgb": self.rgb,
            "depth": self.depth,
            "capture_timestamp": self.reads / 30,
            "age_ms": 1,
            "invalid_fraction": 0.0,
            "above_range_fraction": 0.0,
            "rgb_timestamp_us": self.reads * 33333,
            "depth_timestamp_us": self.reads * 33333,
            "depth_scale_mm": 0.25,
            "rgb_frame_index": self.reads,
            "depth_frame_index": self.reads,
        }
        snapshot.update(self.timestamps.assess(snapshot))
        return snapshot

    def status(self):
        return {"connected": self.connected, **self.metadata}


class FemtoTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="nero-rgbd-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pipeline = Pipeline()
        self.device_list = SimpleNamespace(
            get_count=lambda: 1,
            get_device_name_by_index=lambda i: "Femto Mega",
            get_device_serial_number_by_index=lambda i: "FAKE-HEAD",
            get_device_by_serial_number=lambda serial: object(),
        )
        sdk = SimpleNamespace(
            Context=lambda: SimpleNamespace(query_devices=lambda: self.device_list),
            Pipeline=lambda device: self.pipeline,
            Config=Mock,
            OBFormat=SimpleNamespace(RGB="RGB", Y16="Y16"),
            OBSensorType=SimpleNamespace(COLOR_SENSOR="color"),
            OBAlignMode=SimpleNamespace(HW_MODE="hw"),
            OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        )
        guard = patch.object(femto, "ob", sdk)
        guard.start()
        self.addCleanup(guard.stop)

    def test_sdk_scale_alignment_and_zero_depth(self):
        self.pipeline.depth.values[0, 0] = 0
        camera = femto.FemtoCamera(color_size=(8, 6)).connect()
        try:
            snapshot = camera.read_snapshot()
            self.assertEqual(snapshot["depth"].shape, (6, 8, 1))
            self.assertAlmostEqual(float(snapshot["depth"][2, 2, 0]), 0.5)
            self.assertEqual(snapshot["depth"][0, 0, 0], 0)
            self.assertFalse(snapshot["depth"].flags.writeable)
            self.assertEqual(snapshot["rgb_frame_index"], snapshot["depth_frame_index"])
            self.assertEqual(snapshot["depth_scale_mm"], 0.25)
        finally:
            camera.disconnect()
        self.assertTrue(self.pipeline.closed.is_set())
        self.assertFalse(camera.connected)

    def test_global_timestamp_enable_signature_and_nonfatal_fallback(self):
        for supported, error in ((True, None), (False, None), (True, RuntimeError("模拟不支持启用"))):
            with self.subTest(supported=supported, error=error):
                self.pipeline = Pipeline()
                device = SimpleNamespace(
                    is_global_timestamp_supported=Mock(return_value=supported),
                    enable_global_timestamp=Mock(side_effect=error),
                )
                self.device_list.get_device_by_serial_number = Mock(return_value=device)
                camera = femto.FemtoCamera(color_size=(8, 6)).connect()
                try:
                    self.assertEqual(camera.metadata["global_timestamp_enabled"], supported and error is None)
                    if supported:
                        device.enable_global_timestamp.assert_called_once_with(True)
                    else:
                        device.enable_global_timestamp.assert_not_called()
                    self.assertEqual(camera.read_snapshot()["rgb"].shape, (6, 8, 3))
                finally:
                    camera.disconnect()

    def test_startup_drops_pairing_backlog_and_queued_output_before_ready(self):
        old_rgb = Frame(np.full((6, 8, 3), 99, np.uint8), "RGB")
        old_rgb.get_system_timestamp_us = lambda: 600_000
        old = SimpleNamespace(get_color_frame=lambda: old_rgb, get_depth_frame=lambda: self.pipeline.depth)
        partial = SimpleNamespace(get_color_frame=lambda: old_rgb, get_depth_frame=lambda: None)
        queue = [partial, old, old, None, self.pipeline.frames]

        def read(timeout):
            if queue:
                return queue.pop(0)
            self.pipeline.closed.wait(timeout / 1000)
            return None

        self.pipeline.wait_for_frames = read
        camera = femto.FemtoCamera(color_size=(8, 6)).connect()
        try:
            self.assertEqual(self.pipeline.sync_calls, ["enable", "disable", "enable"])
            self.assertTrue(camera.metadata["startup_queue_reset"])
            self.assertEqual(camera.metadata["startup_frames_discarded"], 2)
            self.assertTrue(np.all(camera.read_snapshot()["rgb"] == 0), "旧 RGB 不得成为首个可录制快照")
        finally:
            camera.disconnect()

    def test_startup_reset_failure_closes_pipeline_without_publishing(self):
        self.pipeline.disable_frame_sync = Mock(side_effect=RuntimeError("模拟 SDK 清队列失败"))
        camera = femto.FemtoCamera(color_size=(8, 6))
        with self.assertRaisesRegex(RuntimeError, "启动缓存清理失败"):
            camera.connect()
        self.assertTrue(self.pipeline.closed.is_set())
        self.assertIsNone(camera._latest)

    def test_startup_wait_and_drain_have_deadlines(self):
        camera = femto.FemtoCamera(color_size=(8, 6))
        camera._pipeline = self.pipeline
        camera.metadata["frame_sync_requested"] = True
        self.pipeline.wait_for_frames = Mock(return_value=None)
        with (
            patch.object(femto.time, "monotonic", side_effect=[0, 4]),
            self.assertRaisesRegex(TimeoutError, "未收到两路完整帧"),
        ):
            camera._discard_startup_frames()
        self.pipeline.wait_for_frames = Mock(return_value=self.pipeline.frames)
        with (
            patch.object(femto.time, "monotonic", side_effect=[0, 0, 0, 2]),
            self.assertRaisesRegex(TimeoutError, "输出队列未能排空"),
        ):
            camera._discard_startup_frames()
        self.assertFalse(camera._ready.is_set())

    def test_receive_gap_rejects_old_pair_even_with_zero_device_timestamps(self):
        camera = femto.FemtoCamera(color_size=(8, 6))
        fresh = camera._convert(self.pipeline.frames)
        fresh.update(rgb_timestamp_us=0, depth_timestamp_us=0)
        camera._publish(fresh)
        for rgb, depth in ((1_000_000, 1_400_000), (1_400_000, 1_000_000)):
            delayed = {**fresh, "rgb_system_timestamp_us": rgb, "depth_system_timestamp_us": depth}
            with self.assertRaisesRegex(TimeoutError, "主机接收时差 400.0 ms"):
                camera._publish(delayed)
            self.assertIs(camera._latest, fresh)
        jitter = {**fresh, "rgb_system_timestamp_us": 1_100_000, "depth_system_timestamp_us": 1_140_000}
        camera._publish(jitter)
        self.assertEqual(camera._latest["rgb_depth_receive_skew_ms"], 40)
        self.assertEqual(camera._latest["timestamp_status"], "unavailable")

    def test_global_getter_is_used_when_device_clock_is_zero(self):
        self.pipeline.color.get_timestamp_us = lambda: 0
        self.pipeline.depth.get_timestamp_us = lambda: 0
        values = [1_000_000, 1_000_100]
        self.pipeline.color.get_global_timestamp_us = lambda: values[0]
        self.pipeline.depth.get_global_timestamp_us = lambda: values[1]
        camera = femto.FemtoCamera(color_size=(8, 6)).connect()
        try:
            first = camera.read_snapshot()
            self.assertIsNone(first["rgb_depth_skew_ms"])
            values[:] = [1_033_333, 1_033_433]
            camera._publish(camera._convert(self.pipeline.frames))
            second = camera.read_snapshot()
            self.assertEqual(second["rgb_timestamp_us"], 0)
            self.assertEqual(second["timestamp_source"], "global")
            self.assertEqual(second["rgb_depth_skew_ms"], 0.1)
            self.assertEqual(camera.status()["timestamp_status"], "valid")
        finally:
            camera.disconnect()

    def test_resolution_enumeration_lists_all_sizes_without_starting_stream(self):
        self.pipeline.get_stream_profile_list = lambda sensor: [
            Profile(1920, 1080, "RGB"),
            Profile(1280, 720, "RGB"),
            Profile(640, 480, "RLE"),
        ]
        result = femto.list_profiles("FAKE-HEAD")
        self.assertEqual([p["width"] for p in result["profiles"]], [1920, 1280, 640])
        self.assertFalse(result["profiles"][0]["reason"])
        self.assertIn("不支持", result["profiles"][2]["reason"])
        self.assertFalse(self.pipeline.started)
        self.assertEqual(self.pipeline.reads, 0)

    def test_mjpg_profile_decodes_to_rgb_before_shared_scaling(self):
        femto.ob.OBFormat.MJPG = "MJPG"
        camera = femto.FemtoCamera(color_size=(8, 6), output_size=(4, 2))
        camera._color_format = "MJPG"
        ok, jpeg = cv2.imencode(".jpg", np.full((6, 8, 3), [0, 0, 255], np.uint8))
        self.assertTrue(ok)
        self.pipeline.color.format = "MJPG"
        self.pipeline.color.get_data = lambda: jpeg.tobytes()
        sample = camera._convert(self.pipeline.frames)
        self.assertEqual(sample["rgb"].shape, (2, 4, 3))
        self.assertEqual(sample["depth"].shape, (2, 4, 1))
        self.assertGreater(int(sample["rgb"][0, 0, 0]), 250)
        self.assertLess(int(sample["rgb"][0, 0, 2]), 5)
        self.assertAlmostEqual(float(sample["depth"][0, 0, 0]), 0.5)

    def test_conversion_rejects_unaligned_shape_wrong_format_and_invalid_scale(self):
        camera = femto.FemtoCamera(color_size=(8, 6))
        for error in ("shape", "format", "scale"):
            self.pipeline.depth = Frame(np.ones((6, 8), np.uint16), "Y16")
            if error == "shape":
                self.pipeline.depth.width = 4
            elif error == "format":
                self.pipeline.depth.format = "MJPG"
            else:
                self.pipeline.depth.scale = float("nan")
            with self.assertRaises(ValueError):
                camera._convert(self.pipeline.frames)

    def test_partial_open_or_warmup_failure_stops_pipeline(self):
        camera = femto.FemtoCamera(color_size=(8, 6))
        with patch.object(camera._ready, "wait", return_value=False), self.assertRaises(RuntimeError):
            camera.connect()
        self.assertTrue(self.pipeline.closed.is_set())
        self.assertIsNone(camera._pipeline)
        self.pipeline = Pipeline()
        with self.assertRaisesRegex(RuntimeError, "不支持 RGB"):
            femto.FemtoCamera(color_size=(8, 6), fps=100).connect()
        self.assertFalse(self.pipeline.started)

    def test_snapshot_never_waits_and_stale_or_failed_frames_are_rejected(self):
        camera = femto.FemtoCamera(color_size=(8, 6))
        camera._pipeline = Mock(wait_for_frames=Mock(side_effect=AssertionError("不应等待 USB")))
        camera._thread = Mock(is_alive=lambda: True)
        camera._latest = camera._convert(self.pipeline.frames)
        with patch.object(femto.time, "perf_counter", return_value=camera._latest["capture_timestamp"] + 0.1):
            self.assertAlmostEqual(camera.read_snapshot()["age_ms"], 100)
        with (
            patch.object(femto.time, "perf_counter", return_value=camera._latest["capture_timestamp"] + 0.6),
            self.assertRaises(TimeoutError),
        ):
            camera.read_snapshot()
        camera._error = "格式错误"
        with self.assertRaisesRegex(RuntimeError, "格式错误"):
            camera.read_snapshot()
        camera._pipeline.wait_for_frames.assert_not_called()

    def test_explicit_selection_never_silently_falls_back(self):
        self.assertEqual(femto.list_devices()[0]["serial"], "FAKE-HEAD")
        with self.assertRaisesRegex(RuntimeError, "未找到所选"):
            femto.FemtoCamera(serial="missing").connect()
        self.device_list.get_count = lambda: 2
        with self.assertRaisesRegex(RuntimeError, "多台"):
            femto.FemtoCamera().connect()
        self.device_list.get_device_name_by_index = lambda i: "Gemini"
        self.assertEqual(femto.list_devices(), [])
        with patch.object(femto, "ob", None), self.assertRaisesRegex(RuntimeError, "未安装"):
            femto.list_devices()

    def test_config_roundtrip_locking_and_legacy_off(self):
        runtime = Runtime(config_path=self.root / "config.json")
        self.assertEqual(runtime.state["camera_map"], {})
        config.save_config(runtime, {}, {"camera_map": {"femto:FAKE-HEAD": "head"}})
        restored = Runtime(config_path=runtime.config_path)
        config.load_saved(restored)
        self.assertEqual(restored.state["camera_map"], {"femto:FAKE-HEAD": "head"})
        for invalid in ({"camera_map": {"a": "left", "b": "left"}}, {"camera_map": {"a": "side"}}):
            with self.assertRaises(ValueError):
                config.validate_settings(runtime, invalid)
        runtime.state["connected"] = True
        client = create_app(runtime).test_client()
        self.assertEqual(
            client.post("/api/save", json={"camera_map": {"femto:FAKE-HEAD": "right"}}).status_code, 409
        )
        with patch.object(femto, "list_devices", side_effect=AssertionError("会话内不得枚举")):
            self.assertEqual(client.get("/api/camera/resolutions/head").status_code, 409)
            self.assertEqual(client.get("/api/status").status_code, 200)

    def test_head_failure_before_arm_transition(self):
        runtime = Runtime(dataset_root=self.root)
        attach_fake_pair(runtime, [])
        runtime.state.update(camera_map={"fake": "right", "femto:FAKE-HEAD": "head"}, record_camera="right")
        detected = {
            "fake": {"type": "rgb", "path": "/fake/right"},
            "femto:FAKE-HEAD": {"type": "rgbd", "serial": "FAKE-HEAD"},
        }
        with (
            patch.object(cameras, "detect_all_cameras", return_value=detected),
            patch.object(cameras, "open_preview_cams"),
            patch.object(camera_io, "RecordingCamera", FakeCamera),
            patch.object(femto.FemtoCamera, "connect", side_effect=RuntimeError("模拟头部失败")),
            patch.object(devices, "transition_to_collection") as transition,
            patch.object(recorder.traceback, "print_exc"),
        ):
            ok, error = recorder.do_connect(runtime)
            self.assertFalse(ok)
            self.assertIn("模拟头部失败", error)
            transition.assert_not_called()

    def test_recording_uses_one_snapshot_for_rgb_depth_and_reports_both(self):
        runtime = Runtime(dataset_root=self.root)
        events = []
        attach_fake_pair(runtime, events)
        runtime.state.update(
            connected=True, fps=10, episode_sec=0.3, camera_sources={"head": {"type": "rgbd"}}
        )
        runtime.collection_cams["head"] = FakeHead()
        runtime.collection_cams["right"] = FakeCamera(None, events)
        runtime.dataset = MemoryDataset(events)
        clock = Clock(events)
        with patch.object(recorder, "time", clock), patch.object(devices, "time", clock):
            recorder.record_worker(runtime, ["right", "head"])
        self.assertEqual(runtime.state["record_frames"], 3)
        self.assertEqual(runtime.collection_cams["head"].reads, 3)
        self.assertEqual(runtime.state["record_phase"], "saved")
        for frame in runtime.dataset.frames:
            np.testing.assert_array_equal(
                frame["observation.images.head_depth"], runtime.collection_cams["head"].depth
            )
            np.testing.assert_array_equal(
                frame["observation.images.head"], runtime.collection_cams["head"].rgb
            )
        stats = runtime.state["record_timing"]["cameras"]
        self.assertEqual(stats["head_depth"]["frames"], 3)
        self.assertEqual(stats["head"]["frames"], 3)
        self.assertEqual(stats["head_depth"]["max_rgb_depth_skew_ms"], 0)
        # 录制预览读取已提交的缓存，不能额外读 SDK。
        runtime.state["record_phase"] = "recording"
        runtime.record_preview = {
            key: (value[0], value[1], 0) for key, value in runtime.record_preview.items()
        }
        with patch("nero_platform.app.time.monotonic", return_value=0.1):
            response = create_app(runtime).test_client().get("/api/record/camera/head_depth/frame")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.collection_cams["head"].reads, 3)

    def test_depth_quality_flags_do_not_silently_hide_saturation(self):
        cadence = timing.EpisodeTiming(30, 0)
        cadence.finish(
            0,
            0,
            0.001,
            0,
            None,
            {
                "depth": {
                    "capture_timestamp": 1,
                    "age_ms": 1,
                    "invalid_fraction": 1,
                    "above_range_fraction": 0.1,
                    "rgb_timestamp_us": 100000,
                    "depth_timestamp_us": 150000,
                }
            },
        )
        summary = cadence.summary(0.033)
        self.assertEqual(len(summary["warnings"]), 3)
        self.assertEqual(summary["cameras"]["depth"]["above_range_frames"], 1)

    def test_actual_depth_encoding_units_zero_and_review_endpoint(self):
        root = self.root / "nero/rgbd"
        features = recorder.make_features({"head": {"type": "rgbd", "serial": "FAKE-HEAD"}})
        self.assertEqual(features["observation.images.head_depth"]["shape"], (720, 1280, 1))
        dataset = LeRobotDataset.create(
            "nero/rgbd",
            root=root,
            fps=30,
            robot_type="nero",
            features=features,
            image_writer_threads=1,
            concatenate_videos=False,
            depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=10.0),
        )
        depth = np.zeros((720, 1280, 1), np.float32)
        depth[:, 320:640] = 0.5
        depth[:, 640:960] = 2.0
        depth[:, 960:] = 8.82
        depth.setflags(write=False)
        try:
            for _ in range(2):
                dataset.add_frame(
                    {
                        "action": np.zeros(8, np.float32),
                        "observation.state": np.zeros(8, np.float32),
                        "observation.images.head": np.full((720, 1280, 3), [255, 0, 0], np.uint8),
                        "observation.images.head_depth": depth,
                        "task": "RGBD 单位验证",
                    }
                )
            dataset.save_episode()
        finally:
            dataset.finalize()
        reader = LocalReview(self.root)
        actual = reader.depth_frame("nero/rgbd", 0, "head_depth", 1)
        self.assertEqual(float(actual[100, 100, 0]), 0.0, "无效零深度必须在读回后仍为零")
        np.testing.assert_allclose(actual[100, [100, 400, 800, 1100], 0], [0, 0.5, 2, 8.82], atol=0.003)
        loaded = LeRobotDataset("nero/rgbd", root=root, video_backend="pyav", depth_output_unit="m")
        np.testing.assert_allclose(
            loaded[0]["observation.images.head_depth"][0, 100, [100, 400, 800, 1100]],
            [0, 0.5, 2, 8.82],
            atol=0.003,
        )
        client = create_app(Runtime(dataset_root=self.root)).test_client()
        response = client.get(
            "/api/review/depth",
            query_string={
                "repo_id": "nero/rgbd",
                "episode_index": 0,
                "frame_index": 1,
                "camera": "head_depth",
            },
        )
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.mimetype, "image/jpeg")
        result = reader.episode("nero/rgbd", 0)
        self.assertTrue(next(v for v in result["videos"] if v["camera"] == "head_depth")["is_depth"])
        info = json.loads((root / "meta/info.json").read_text())
        self.assertEqual(info["features"]["observation.images.head_depth"]["info"]["video.depth_min"], 0)
        recorder.validate_resume(root, 30, features)
        info["features"]["observation.images.head_depth"]["info"]["video.depth_min"] = 0.01
        (root / "meta/info.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "无效深度"):
            recorder.validate_resume(root, 30, features)

    def test_actual_scaled_depth_encoding_preserves_dimensions_and_units(self):
        settings = resolution.defaults()
        settings["head"]["scale"] = 2
        features = recorder.make_features(
            {"head": {"type": "rgbd", "serial": "FAKE-HEAD"}}, camera_resolution=settings
        )
        root = self.root / "nero/scaled"
        dataset = LeRobotDataset.create(
            "nero/scaled",
            fps=30,
            features=features,
            root=root,
            depth_encoder=DepthEncoderConfig(depth_min=0, depth_max=10),
            concatenate_videos=False,
        )
        depth = np.full((720, 1280, 1), 2.0, np.float32)
        depth[:, :640] = 0
        try:
            for _ in range(2):
                dataset.add_frame(
                    {
                        "action": np.zeros(8, np.float32),
                        "observation.state": np.zeros(8, np.float32),
                        "task": "scaled",
                        "observation.images.head": np.zeros((360, 640, 3), np.uint8),
                        "observation.images.head_depth": resolution.resize(depth, (640, 360), depth=True),
                    }
                )
            dataset.save_episode()
        finally:
            dataset.finalize()
        loaded = LeRobotDataset("nero/scaled", root=root, video_backend="pyav", depth_output_unit="m")
        frame = loaded[0]
        self.assertEqual(tuple(frame["observation.images.head_depth"].shape), (1, 360, 640))
        self.assertEqual(tuple(frame["observation.images.head"].shape), (3, 360, 640))
        self.assertEqual(float(frame["observation.images.head_depth"][0, 100, 100]), 0)
        self.assertAlmostEqual(float(frame["observation.images.head_depth"][0, 100, 500]), 2.0, delta=0.003)


if __name__ == "__main__":
    unittest.main()
