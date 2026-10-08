"""模拟不同颜色的两台相机，验证选择、持久化以及真实视频中的设备对应。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from nero_platform import camera_io, cameras, config, devices, recorder
from nero_platform.app import create_app
from nero_platform.runtime import Runtime
from test_platform import Clock, FakeCamera, attach_fake_pair

from lerobot.datasets import LeRobotDataset


class CameraSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-camera-selection-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.runtime = Runtime(config_path=root / "config.json", dataset_root=root / "datasets")
        self.runtime.state.update(
            fps=10,
            episode_sec=0.25,
            record_camera="right",
            camera_map={"fake-left": "left", "fake-right": "right"},
        )
        self.events = []
        attach_fake_pair(self.runtime, self.events)
        self.detected = {
            "fake-left": {"type": "rgb", "path": "/fake/left"},
            "fake-right": {"type": "rgb", "path": "/fake/right"},
        }
        self.client = create_app(self.runtime).test_client()
        for guard in (
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("真实 CAN 禁用")),
            patch.object(cameras, "OpenCVCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("真实相机禁用")),
            patch.object(cameras, "open_preview_cams"),
            patch.object(cameras, "detect_all_cameras", return_value=self.detected),
            patch.object(recorder.traceback, "print_exc"),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def test_rgb_preview_jpeg_preserves_colors_without_mutating_camera_frame(self) -> None:
        camera = FakeCamera()
        camera.frame[:, :210] = [240, 5, 5]
        camera.frame[:, 210:420] = [5, 240, 5]
        camera.frame[:, 420:] = [5, 5, 240]
        original = camera.frame.copy()
        self.runtime.preview_cams["fake-right"] = camera
        response = self.client.get("/api/camera/fake-right/frame")
        try:
            self.assertEqual(response.status_code, 200)
            jpeg = response.data
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            for x in (100, 315, 525):
                np.testing.assert_allclose(rgb[240, x].astype(float), original[240, x], atol=6)
            np.testing.assert_array_equal(camera.frame, original)
        finally:
            response.close()

    def test_selection_survives_save_and_restart_independently_of_arm_roles(self) -> None:
        self.runtime.state.update(leader_label="主右", follower_label="从右")
        response = self.client.post("/api/save", json={"record_camera": "left"})
        self.assertTrue(response.json["ok"])
        restarted = Runtime(config_path=self.runtime.config_path)
        config.load_saved(restarted)
        self.assertEqual(restarted.state["record_camera"], "left")
        self.assertEqual(restarted.state["leader_label"], "主右")
        self.assertEqual(
            cameras.recording_sources(restarted, self.detected),
            {"left": {"id": "fake-left", "type": "rgb", "path": "/fake/left"}},
        )
        self.assertEqual(self.client.get("/api/config").json["record_camera"], "left")

    def test_record_preview_uses_submitted_rgb_frame_without_extra_camera_read(self) -> None:
        camera = FakeCamera(events=self.events)
        sampled = np.full((480, 640, 3), [230, 5, 10], dtype=np.uint8)
        camera.frame[:] = [5, 10, 230]
        self.runtime.collection_cams["right"] = camera
        self.runtime.state.update(connected=True, record_phase="recording")
        self.runtime.record_preview = {"right": (camera, sampled, 100.0)}
        with patch("nero_platform.app.time.monotonic", return_value=100.1):
            response = self.client.get("/api/record/camera/right/frame")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(response.data, np.uint8), 1), cv2.COLOR_BGR2RGB)
        np.testing.assert_allclose(rgb[100, 100], [230, 5, 10], atol=6)
        self.assertEqual(self.events, [], "录制预览不重复读相机或发送设备命令")
        self.assertEqual(self.client.get("/api/record/camera/left/frame").status_code, 404)
        self.runtime.state["record_phase"] = "saved"
        response = self.client.get("/api/record/camera/right/frame")
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(response.data, np.uint8), 1), cv2.COLOR_BGR2RGB)
        np.testing.assert_allclose(rgb[100, 100], [5, 10, 230], atol=6)
        self.assertEqual(self.events, [("camera", "read")], "集间取景复用已打开的相机")

    def test_missing_stale_or_replaced_record_preview_never_returns_old_image(self) -> None:
        camera = FakeCamera(events=self.events)
        self.runtime.collection_cams["right"] = camera
        self.runtime.state.update(connected=True, record_phase="recording")
        with patch("nero_platform.app.time.monotonic", return_value=102.0):
            for cache in (
                {},
                {"right": (camera, camera.frame, 100.0)},
                {"right": (object(), camera.frame, 102.0)},
            ):
                with self.subTest(cache=bool(cache)):
                    self.runtime.record_preview = cache
                    self.assertEqual(self.client.get("/api/record/camera/right/frame").status_code, 503)
        self.assertEqual(self.events, [])
        self.runtime.state["connected"] = False
        self.assertEqual(self.client.get("/api/record/camera/right/frame").status_code, 404)

    def test_preview_snapshot_never_opens_a_camera_or_falls_back_to_another_device(self) -> None:
        self.runtime.preview_cams["fake-left"] = FakeCamera(events=self.events)
        self.assertEqual(self.client.get("/api/camera/fake-right/frame").status_code, 404)
        self.assertEqual(self.events, [])
        response = self.client.get("/api/camera/fake-left/frame")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(self.events, [("camera", "read")])

    def test_duplicate_role_is_rejected_before_any_config_changes(self) -> None:
        original = self.runtime.state["camera_map"].copy()
        response = self.client.post(
            "/api/save", json={"camera_map": {"fake-left": "right", "fake-right": "right"}, "fps": 15}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("多台相机", response.json["msg"])
        self.assertEqual(self.runtime.state["camera_map"], original)
        self.assertEqual(self.runtime.state["fps"], 10)
        self.assertFalse(self.runtime.config_path.exists())

    def test_missing_selected_camera_fails_before_motor_or_camera_commands(self) -> None:
        with patch.object(
            cameras, "detect_all_cameras", return_value={"fake-left": {"type": "rgb", "path": "/fake/left"}}
        ):
            ok, message = recorder.do_connect(self.runtime)
        self.assertFalse(ok)
        self.assertIn("right 相机未检测到", message)
        self.assertEqual(self.events, [])
        self.assertEqual(self.runtime.collection_cams, {})

    def test_unselected_camera_can_be_absent(self) -> None:
        self.assertEqual(
            cameras.recording_sources(self.runtime, {"fake-right": {"type": "rgb", "path": "/fake/right"}}),
            {"right": {"id": "fake-right", "type": "rgb", "path": "/fake/right"}},
        )

    def test_mapping_swap_changes_source_without_using_device_order(self) -> None:
        self.runtime.state["camera_map"] = {"fake-left": "right", "fake-right": "left"}
        self.assertEqual(
            cameras.recording_sources(self.runtime, self.detected),
            {"right": {"id": "fake-left", "type": "rgb", "path": "/fake/left"}},
        )

    def test_cannot_change_selection_during_an_open_session(self) -> None:
        self.runtime.state["connected"] = True
        self.assertEqual(self.client.post("/api/save", json={"record_camera": "both"}).status_code, 409)
        self.assertEqual(self.runtime.state["record_camera"], "right")

    def record_and_load(self, selection: str) -> tuple[LeRobotDataset, list[str]]:
        self.runtime.state["record_camera"] = selection
        config.save_config(self.runtime, self.detected)
        restarted = Runtime(config_path=self.runtime.config_path, dataset_root=self.runtime.dataset_root)
        config.load_saved(restarted)
        attach_fake_pair(restarted, self.events)
        self.runtime = restarted
        opened = []

        class ColoredCamera(FakeCamera):
            def __init__(self, cfg):
                super().__init__(cfg)
                source = str(cfg.index_or_path)
                opened.append(source)
                self.frame[:] = 0
                self.frame[:, :, 0 if source == "/fake/left" else 2] = 230

        clock = Clock(self.events)
        with (
            patch.object(camera_io, "RecordingCamera", ColoredCamera),
            patch.object(recorder, "time", clock),
            patch.object(devices, "time", clock),
        ):
            ok, message = recorder.start_recording(self.runtime)
            self.assertTrue(ok, message)
            self.runtime.record_thread.join(15)
            self.assertFalse(self.runtime.record_thread.is_alive())
            self.assertIsNone(self.runtime.state["last_error"])
            status = create_app(self.runtime).test_client().get("/api/status").json
            self.assertEqual(
                status["active_cameras"],
                {
                    role: info["path"]
                    for role, info in cameras.recording_sources(self.runtime, self.detected).items()
                },
            )
            self.assertEqual(status["record"]["phase"], "saved")
            self.assertEqual(status["record"]["last_episode"]["frames"], 3)
            self.assertEqual(set(self.runtime.record_preview), set(status["active_cameras"]))
            recorder.do_disconnect(self.runtime, reopen_preview=False)
            self.assertEqual(self.runtime.record_preview, {})
        dataset = LeRobotDataset(
            self.runtime.state["repo_id"], root=recorder.dataset_path(self.runtime), video_backend="pyav"
        )
        return dataset, opened

    def test_right_only_opens_right_device_and_writes_right_image_content(self) -> None:
        dataset, opened = self.record_and_load("right")
        self.assertEqual(opened, ["/fake/right"])
        self.assertEqual(dataset.meta.video_keys, ["observation.images.right"])
        for index in range(dataset.num_frames):
            frame = dataset[index]["observation.images.right"]
            self.assertGreater(float(frame[2].mean()), 0.8)
            self.assertLess(float(frame[0].mean()), 0.05)
        self.assertFalse((recorder.dataset_path(self.runtime) / "videos/observation.images.left").exists())

    def test_two_cameras_write_distinct_correct_content_and_reject_schema_change(self) -> None:
        dataset, opened = self.record_and_load("both")
        self.assertEqual(opened, ["/fake/left", "/fake/right"])
        frame = dataset[0]
        self.assertGreater(float(frame["observation.images.left"][0].mean()), 0.8)
        self.assertLess(float(frame["observation.images.left"][2].mean()), 0.05)
        self.assertGreater(float(frame["observation.images.right"][2].mean()), 0.8)
        before = self.events[:]
        self.runtime.state["record_camera"] = "right"
        ok, message = recorder.do_connect(self.runtime)
        self.assertFalse(ok)
        self.assertIn("新的数据集名", message)
        self.assertEqual(self.events, before)
        self.assertEqual(dataset.num_episodes, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
