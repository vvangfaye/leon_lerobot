"""FastWAM's channel contract and runtime gate, without opening real devices."""

import json
import unittest
from pathlib import Path

import test_inference as baseline
from inference_platform.app import inspect_checkpoint
from inference_platform.contracts import Contract


class FastWAMRuntimeTests(unittest.TestCase):
    setUp = baseline.RuntimeTests.setUp
    checkpoint = baseline.RuntimeTests.checkpoint
    wait = baseline.RuntimeTests.wait
    load = baseline.RuntimeTests.load
    run_once = baseline.RuntimeTests.run_once

    def prepare_fastwam(self):
        path = Path(self.settings["path"])
        raw = json.loads((path / "config.json").read_text())
        names = raw.pop("action_feature_names")
        raw.update(type="fastwam", action_dim=8, proprio_dim=8, action_horizon=32)
        (path / "config.json").write_text(json.dumps(raw))
        dataset = self.root / "dataset"
        (dataset / "meta").mkdir(parents=True)
        features = {key: {"names": names, "shape": [8]} for key in ("action", "observation.state")}
        (dataset / "meta/info.json").write_text(json.dumps({"features": features}))
        (path / "train_config.json").write_text(json.dumps({"dataset": {"root": str(dataset)}}))
        self.settings.update(model="fastwam", rtcEnabled=False)

    def test_catalog_and_contract_use_saved_dataset_names(self):
        self.prepare_fastwam()
        item = inspect_checkpoint(Path(self.settings["path"]))
        self.assertIsNone(item["error"])
        self.assertEqual((item["model"], item["space"], item["joint_count"]), ("fastwam", "joint", 7))
        contract = Contract(self.settings["path"], self.settings)
        self.assertEqual(contract.names[-1], "gripper.pos")
        self.assertEqual(contract.steps, 3)

    def test_rejects_missing_or_inconsistent_dataset_names(self):
        self.prepare_fastwam()
        info = self.root / "dataset/meta/info.json"
        raw = json.loads(info.read_text())
        raw["features"]["observation.state"]["names"] = list(reversed(raw["features"]["action"]["names"]))
        info.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "顺序或维数"):
            Contract(self.settings["path"], self.settings)
        info.unlink()
        self.assertIsNotNone(inspect_checkpoint(Path(self.settings["path"]))["error"])

    def test_warmup_does_not_control_hardware_then_sync_execution_stops(self):
        self.prepare_fastwam()
        self.load()
        self.assertEqual(self.runtime.policy.calls, ["predict", "reset"])
        self.assertEqual(self.readers["arm"].commands, [])
        result = self.run_once()
        self.assertEqual(result["model"], "fastwam")
        self.assertEqual(result["job"]["phase"], "done", result)
        self.assertIn("send", self.readers["arm"].commands)
        self.assertEqual(self.readers["arm"].commands[-1], "hold")

    def test_rtc_rejected_before_any_hardware_commands(self):
        self.prepare_fastwam()
        self.settings["rtcEnabled"] = True
        with self.assertRaisesRegex(ValueError, "FastWAM.*RTC"):
            self.runtime.load(self.settings, self.runtime.revision)
        self.settings["rtcEnabled"] = False
        self.load()
        self.settings["rtcEnabled"] = True
        with self.assertRaisesRegex(ValueError, "FastWAM.*RTC"):
            self.runtime.start("single", self.settings, self.runtime.revision)
        self.assertEqual(self.readers["arm"].commands, [])


if __name__ == "__main__":
    unittest.main()
