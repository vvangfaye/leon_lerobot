"""Focused checks for the two boundaries where mistakes could affect a training run."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training_platform.jobs import LogReader
from training_platform.worker import SavedStopError, install_stop_hooks


class RuntimeBoundaries(unittest.TestCase):
    def test_stop_only_after_native_save_and_accumulation_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "stop.request"
            request.touch()
            events = []
            trainer = SimpleNamespace(
                should_save_checkpoint=lambda step, frequency, total: step % frequency == 0,
                update_last_checkpoint=lambda path: events.append("last_link_updated"),
            )
            install_stop_hooks(trainer, request, accumulation_steps=2)
            self.assertFalse(trainer.should_save_checkpoint(3, 100, 100))
            self.assertTrue(trainer.should_save_checkpoint(4, 100, 100))
            # Native trainer calls update_last only after optimizer/RNG/metadata/EMA writes.
            events.extend(["optimizer", "rng", "metadata", "ema"])
            with self.assertRaises(SavedStopError):
                trainer.update_last_checkpoint(Path(directory))
            self.assertEqual(events, ["optimizer", "rng", "metadata", "ema", "last_link_updated"])

    def test_resume_progress_uses_exact_tqdm_not_rounded_log(self):
        reader = LogReader()
        reader.parse(
            "Training: 50%|██| 10600/22000 [1:20:03<1:28:00, 2.16step/s]"
            "INFO 17:33:32 train.py:971 step:31K loss:0.010 latent_loss:0.2 lr:2.5e-06",
            {"config": {"steps": 42000}, "start_step": 20000},
        )
        self.assertEqual(reader.step, 30600)
        self.assertEqual(reader.history[0]["step"], 30600)
        self.assertEqual(set(reader.metrics), {"loss", "latent_loss", "lr"})


if __name__ == "__main__":
    unittest.main()
