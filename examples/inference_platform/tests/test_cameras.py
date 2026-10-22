"""推理适配必须复用采集预览，带上按设备保存的设置。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inference_platform.hardware import CameraReader, camera_module


class SharedCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "collection.json"
        self.runtime = camera_module("runtime")
        self.cameras = camera_module("cameras")
        self.color = camera_module("camera_color")
        self.preview = camera_module("camera_preview")
        self.settings = self.runtime.Runtime(config_path=self.path)
        self.info = {"id": "usb-test", "backend": "opencv", "type": "rgb", "path": "/fake/camera"}

    def test_saved_focus_and_shared_encoding_are_used_without_changing_rgb(self) -> None:
        focus = {"mode": "manual", "value": 130}
        self.path.write_text(json.dumps({"camera_focus_settings": {"usb-test": focus}}))
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32, 0] = 255  # RGB 红色
        frame[:, 32:, 2] = 255  # RGB 蓝色
        original = frame.copy()
        cam = SimpleNamespace(
            connect=Mock(), disconnect=Mock(),
            read_latest_with_timestamp=lambda: (frame, time.monotonic(), 1.0),
        )
        with (
            patch.object(self.runtime, "Runtime", return_value=self.settings),
            patch.object(self.cameras, "create_preview_camera", return_value=cam) as create,
            patch.object(self.cameras, "configure_focus", return_value={
                "state": "verified", "message": "手动对焦 130",
            }) as configure_focus,
            patch.object(self.color, "configure_white_balance", return_value={
                "state": "verified", "message": "自动白平衡已开启", "automatic": 1,
            }) as configure_color,
        ):
            reader = CameraReader(self.info, "nero")
            reader.connect()
            sample, jpeg = reader.sample()
            self.assertEqual(jpeg, self.preview.encode_jpeg(frame))
            self.assertTrue(np.array_equal(frame, original))
            create.assert_called_once_with("/fake/camera")
            configure_focus.assert_called_once_with("/fake/camera", focus)
            configure_color.assert_called_once_with("/fake/camera")
            self.assertIn("手动对焦 130", sample["message"])
            reader.close()
            cam.disconnect.assert_called_once()

    def test_invalid_saved_settings_do_not_silently_open_with_other_parameters(self) -> None:
        self.path.write_text("{broken")
        with (
            patch.object(self.runtime, "Runtime", return_value=self.settings),
            patch.object(self.cameras, "create_preview_camera") as create,
        ):
            with self.assertRaisesRegex(ValueError, "配置读取失败"):
                CameraReader(self.info, "nero")
            create.assert_not_called()
