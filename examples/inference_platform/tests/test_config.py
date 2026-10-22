"""Metadata and model/motion boundaries for the platform."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from inference_platform.app import checkpoint_candidates, create_platform_app, inspect_checkpoint
from inference_platform.checkpoint_picker import DirectoryPicker


class PlatformConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)

    def checkpoint(self, *, names: list[str | None] | None = None, dimension: int = 20) -> dict:
        config = {
            "type": "pi05",
            "output_features": {"action": {"shape": [dimension]}},
            "input_features": {"observation.images.head": {"shape": [3, 480, 640]}},
        }
        if names is not None:
            config["action_feature_names"] = names
        (self.path / "config.json").write_text(json.dumps(config))
        return inspect_checkpoint(self.path)

    def test_dimension_alone_does_not_imply_space_or_robot(self) -> None:
        for dimension in (7, 8, 14, 16, 20):
            item = self.checkpoint(dimension=dimension)
            self.assertIsNone(item["space"])
            self.assertIsNone(item["robot"])
            self.assertIsNone(item["arms"])

    def test_joint_names_identify_single_arm_and_dof_without_claiming_robot(self) -> None:
        names = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
        item = self.checkpoint(names=names, dimension=8)
        self.assertEqual((item["space"], item["arms"], item["joint_count"]), ("joint", "single", 7))
        self.assertIsNone(item["robot"])
        self.assertFalse(item["weights_present"])

    def test_eef_names_identify_dual_arm_but_not_hardware_compatibility(self) -> None:
        per_arm = ["eef_x", "eef_y", "eef_z",
                   *[f"rot6d_col{col}_{axis}" for col in (0, 1) for axis in "xyz"], "gripper"]
        item = self.checkpoint(names=[f"{role}.{name}" for role in ("left", "right") for name in per_arm])
        self.assertEqual((item["space"], item["arms"], item["action_dim"]), ("eef", "dual", 20))
        self.assertIsNone(item["robot"])

    def test_mixed_or_incomplete_name_count_remains_unknown(self) -> None:
        for names in (["joint1.pos", "eef_x"], ["joint1.pos"], [None, "gripper.pos"]):
            item = self.checkpoint(names=names, dimension=2)
            self.assertIsNone(item["space"])

    def test_openwam_requires_explicit_deployment_contract(self) -> None:
        cfg = {"dataloader": {"type": "nero_dual_eef", "action_mode": "eef20", "camera_layout": []}}
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg))
        item = inspect_checkpoint(self.path)
        self.assertEqual((item["model"], item["robot"], item["space"], item["arms"]),
                         ("openwam", "nero", "eef", "dual"))
        cfg["dataloader"].pop("action_mode")
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg))
        self.assertIsNotNone(inspect_checkpoint(self.path)["error"])

    def test_bad_metadata_is_an_entry_error_not_a_catalog_crash(self) -> None:
        (self.path / "config.json").write_text("{broken")
        app = create_platform_app(candidates=[self.path])
        reply = app.test_client().get("/api/platform/checkpoints")
        self.assertEqual(reply.status_code, 200)
        self.assertTrue(reply.json["checkpoints"][0]["error"])
        (self.path / "config.json").write_text(" " * (256 * 1024 + 1))
        self.assertIn("256 KB", inspect_checkpoint(self.path)["error"])

    def test_duplicate_resolved_paths_are_listed_once(self) -> None:
        self.checkpoint()
        app = create_platform_app(candidates=[self.path, self.path / "."])
        reply = app.test_client().get("/api/platform/checkpoints")
        self.assertEqual(len(reply.json["checkpoints"]), 1)

    def test_scan_only_outputs_and_ignore_external_directory_links(self) -> None:
        outputs = self.path / "outputs"
        first = outputs / "train/run/checkpoints/020000/pretrained_model"
        second = outputs / "openwam-deploy"
        external = self.path / "Downloads/pretrained_model"
        for directory, filename in ((first, "config.json"), (second, "config.yaml"), (external, "config.json")):
            directory.mkdir(parents=True)
            (directory / filename).write_text("{}")
        (outputs / "external-link").symlink_to(external, target_is_directory=True)
        (outputs / "latest").symlink_to(first, target_is_directory=True)
        with patch("inference_platform.app.OUTPUTS_DIR", outputs):
            self.assertEqual(set(checkpoint_candidates()), {first, second})
            response = create_platform_app().test_client().get("/api/platform/checkpoints")
            self.assertEqual(response.json["selection_version"], 2)
            self.assertEqual(response.json["directory"], str(outputs))
            self.assertEqual(len(response.json["checkpoints"]), 2)

    def test_manual_picker_reads_selected_folder_and_cancel_does_not_return_a_model(self) -> None:
        self.checkpoint()
        picker = Mock(return_value=self.path)
        client = create_platform_app(candidates=[], directory_picker=picker).test_client()
        headers = {"X-Platform-Request": "1"}
        self.assertEqual(client.post("/api/platform/checkpoints/pick", json={}).status_code, 403)
        picker.assert_not_called()
        response = client.post("/api/platform/checkpoints/pick", json={}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["checkpoint"]["path"], str(self.path))
        self.assertFalse(response.json["cancelled"])
        self.assertEqual(client.get("/api/platform/checkpoints").json["checkpoints"], [])
        picker.return_value = None
        response = client.post("/api/platform/checkpoints/pick", json={}, headers=headers)
        self.assertEqual(response.json, {"cancelled": True})

    def test_manual_selection_accepts_step_folder_and_rechecks_saved_paths(self) -> None:
        pretrained = self.path / "015000/pretrained_model"
        pretrained.mkdir(parents=True)
        (pretrained / "config.json").write_text(json.dumps({"type": "pi05"}))
        client = create_platform_app(directory_picker=lambda _: pretrained.parent).test_client()
        headers = {"X-Platform-Request": "1"}
        response = client.post("/api/platform/checkpoints/pick", json={}, headers=headers)
        self.assertEqual(response.json["checkpoint"]["path"], str(pretrained))
        response = client.post("/api/platform/checkpoints/inspect", json={"path": str(pretrained)}, headers=headers)
        self.assertEqual(response.status_code, 200)
        (pretrained / "config.json").write_text("{broken")
        response = client.post("/api/platform/checkpoints/inspect", json={"path": str(pretrained)}, headers=headers)
        self.assertEqual(response.status_code, 409)
        self.assertIn("不是可识别", response.json["error"])
        for data in ({}, {"path": []}, {"path": ""}):
            self.assertEqual(client.post(
                "/api/platform/checkpoints/inspect", json=data, headers=headers,
            ).status_code, 400)

    def test_manual_invalid_folder_and_window_failure_are_visible_errors(self) -> None:
        picker = Mock(return_value=self.path)
        client = create_platform_app(directory_picker=picker).test_client()
        headers = {"X-Platform-Request": "1"}
        self.assertEqual(client.post("/api/platform/checkpoints/pick", json={}, headers=headers).status_code, 409)
        picker.side_effect = ValueError("无法打开文件夹窗口")
        response = client.post("/api/platform/checkpoints/pick", json={}, headers=headers)
        self.assertEqual(response.status_code, 409)
        self.assertIn("无法打开", response.json["error"])

    def test_page_has_no_motion_endpoints_and_does_not_import_runtime(self) -> None:
        before = set(sys.modules)
        app = create_platform_app(candidates=[])
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        with client.get("/static/platform.js") as response:
            self.assertEqual(response.status_code, 200)
        for path in ("/api/predict", "/api/model/start", "/api/live/open", "/api/execution/home",
                     "/api/execution/stop"):
            self.assertEqual(client.post(path, json={}, headers={"X-Platform-Request": "1"}).status_code, 404)
        self.assertEqual(client.get("/", headers={"Host": "external.invalid"}).status_code, 400)
        imported = set(sys.modules) - before
        self.assertFalse(any(name.startswith(("torch", "openwam", "nero_inference")) for name in imported))


class DirectoryPickerTests(unittest.TestCase):
    def test_native_directory_dialog_and_cancellation(self) -> None:
        picker = DirectoryPicker()
        with (
            patch("inference_platform.checkpoint_picker.shutil.which", return_value="/usr/bin/zenity"),
            patch.dict("os.environ", {"DISPLAY": ":test"}),
            patch("inference_platform.checkpoint_picker.subprocess.run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="/tmp/model with spaces\n")
            self.assertEqual(picker(Path("/tmp")), Path("/tmp/model with spaces"))
            args = run.call_args.args[0]
            self.assertIn("--file-selection", args)
            self.assertIn("--directory", args)
            self.assertNotIn("shell", run.call_args.kwargs)
            run.return_value = SimpleNamespace(returncode=1, stdout="")
            self.assertIsNone(picker(Path("/tmp")))
            run.side_effect = subprocess.TimeoutExpired("zenity", 120)
            with self.assertRaisesRegex(ValueError, "超时"):
                picker(Path("/tmp"))
            self.assertFalse(picker.lock.locked())

    def test_duplicate_windows_and_missing_desktop_are_rejected(self) -> None:
        picker = DirectoryPicker()
        picker.lock.acquire()
        with self.assertRaisesRegex(ValueError, "已打开"):
            picker(Path("/tmp"))
        picker.lock.release()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "本机桌面"):
                picker(Path("/tmp"))
        self.assertFalse(picker.lock.locked())


if __name__ == "__main__":
    unittest.main()
