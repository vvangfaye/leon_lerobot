"""配置损坏、请求校验和原子保存；仅使用临时目录。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nero_platform import config
from nero_platform.app import create_app
from nero_platform.runtime import Runtime


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-config-")
        self.addCleanup(directory.cleanup)
        self.runtime = Runtime(config_path=Path(directory.name) / "config.json")

    def test_corrupt_config_stays_intact_and_page_explains_blocked_operations(self) -> None:
        for content in ('{"fps":', "[]", '{"fps": "30"}', '{"camera_map": null}'):
            with self.subTest(content=content):
                self.runtime.config_path.write_text(content)
                config.load_saved(self.runtime)
                self.assertIn("配置读取失败", self.runtime.state["config_error"])
                client = create_app(self.runtime).test_client()
                self.assertEqual(client.get("/").status_code, 200)
                self.assertIn("请修复", client.get("/api/status").json["config_error"])
                for url in ("/api/connect", "/api/save", "/api/arm/move/can0"):
                    self.assertEqual(client.post(url, json={}).status_code, 409)
                self.assertEqual(self.runtime.config_path.read_text(), content)

    def test_invalid_settings_never_partially_apply(self) -> None:
        before = copy.deepcopy(self.runtime.state)
        for bad in (
            {"fps": 0},
            {"episode_sec": float("nan")},
            {"repo_id": "../escape"},
            {"labels": []},
            {"task": ""},
            {"control_frequency": "100"},
            {"control_frequency": None},
            {"control_frequency": True},
            {"control_frequency": 100.0},
            {"control_frequency": 0},
            {"control_frequency": 75},
            {"control_frequency": 201},
            {"follow_motion_mode": "mit"},
            {"follow_motion_mode": None},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    config.apply_settings(self.runtime, {"camera_map": {"fake": "right"}, **bad})
                self.assertEqual(self.runtime.state, before)

    def test_replace_failure_preserves_previous_config_and_memory(self) -> None:
        config.save_config(self.runtime, {})
        before = self.runtime.config_path.read_bytes()
        with (
            patch.object(config.os, "replace", side_effect=OSError("模拟写盘失败")),
            self.assertRaises(OSError),
        ):
            config.save_config(self.runtime, {}, {"fps": 15})
        self.assertEqual(self.runtime.config_path.read_bytes(), before)
        self.assertEqual(self.runtime.state["fps"], 30)
        self.assertEqual(list(self.runtime.config_path.parent.iterdir()), [self.runtime.config_path])
        config.save_config(self.runtime, {}, {"fps": 15})
        restarted = Runtime(config_path=self.runtime.config_path)
        config.load_saved(restarted)
        self.assertEqual(restarted.state["fps"], 15)

    def test_invalid_api_payload_is_a_clear_error(self) -> None:
        client = create_app(self.runtime).test_client()
        for data in ({"fps": -1}, {"labels": []}, []):
            response = client.post("/api/save", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.json["ok"])
        self.assertFalse(self.runtime.config_path.exists())

    def test_old_smoothing_setting_is_removed_and_frequency_defaults_to_100(self) -> None:
        self.runtime.config_path.write_text(
            '{"fps": 30, "follow_smoothing": "light", "follow_motion_mode": "j"}'
        )
        config.load_saved(self.runtime)
        self.assertIsNone(self.runtime.state["config_error"])
        self.assertEqual(self.runtime.state["control_frequency"], 100)
        self.assertEqual(self.runtime.state["follow_motion_mode"], "j")
        self.assertNotIn("follow_smoothing", self.runtime.state)
        saved = config.save_config(self.runtime, {})
        self.assertNotIn("follow_smoothing", saved)
        self.assertNotIn("follow_smoothing", self.runtime.config_path.read_text())

    def test_control_frequency_choices_persist_and_lock_during_session(self) -> None:
        client = create_app(self.runtime).test_client()
        for frequency in ("capture", 50, 100, 150, 200):
            with self.subTest(frequency=frequency):
                self.runtime.state["connected"] = False
                self.assertEqual(
                    client.post("/api/save", json={"control_frequency": frequency}).status_code, 200
                )
                restarted = Runtime(config_path=self.runtime.config_path)
                config.load_saved(restarted)
                self.assertEqual(restarted.state["control_frequency"], frequency)
                self.assertEqual(client.get("/api/config").json["control_frequency"], frequency)
                self.runtime.state["connected"] = True
                response = client.post(
                    "/api/save", json={"control_frequency": 50 if frequency != 50 else 100}
                )
                self.assertEqual(response.status_code, 409)
                self.assertEqual(self.runtime.state["control_frequency"], frequency)

    def test_follow_capture_frequency_tracks_fps_without_changing_fixed_frequency(self) -> None:
        client = create_app(self.runtime).test_client()
        config.apply_settings(self.runtime, {"control_frequency": "capture", "fps": 30})
        self.assertEqual(client.get("/api/status").json["record"]["control_hz"], 30)
        config.apply_settings(self.runtime, {"fps": 15})
        self.assertEqual(client.get("/api/status").json["record"]["control_hz"], 15)
        config.apply_settings(self.runtime, {"control_frequency": 150})
        self.assertEqual(client.get("/api/status").json["record"]["control_hz"], 150)
        self.assertEqual(self.runtime.state["fps"], 15)

    def test_follow_motion_mode_defaults_to_js_and_locks_during_session(self) -> None:
        self.runtime.config_path.write_text('{"fps": 30}')
        config.load_saved(self.runtime)
        self.assertEqual(self.runtime.state["follow_motion_mode"], "js")
        config.save_config(self.runtime, {}, {"follow_motion_mode": "j"})
        restarted = Runtime(config_path=self.runtime.config_path)
        config.load_saved(restarted)
        self.assertEqual(restarted.state["follow_motion_mode"], "j")
        self.runtime.state["connected"] = True
        client = create_app(self.runtime).test_client()
        response = client.post("/api/save", json={"follow_motion_mode": "js"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.runtime.state["follow_motion_mode"], "j")


if __name__ == "__main__":
    unittest.main()
