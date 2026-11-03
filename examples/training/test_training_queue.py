import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "training_queue", Path(__file__).with_name("run_training_queue.py")
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def tensor_file(path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))


@pytest.fixture
def job(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    for name in (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/rng_state.safetensors",
    ):
        tensor_file(checkpoint / name)
    for name in (
        "pretrained_model/config.json",
        "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_postprocessor.json",
        "training_state/training_step.json",
        "training_state/optimizer_param_groups.json",
        "training_state/scheduler_state.json",
    ):
        (checkpoint / name).write_text(json.dumps({"step": 30000, "steps": 30000}))
    log = tmp_path / "pi05.log"
    log.write_text("End of training\n")
    return {
        "prerequisite": {
            "processes": [],
            "checkpoint": str(checkpoint),
            "log": str(log),
            "steps": 30000,
            "started_at_epoch": 0,
        },
        "state_path": str(tmp_path / "state.json"),
        "output_dir": str(tmp_path / "model"),
        "train_log": str(tmp_path / "fastwam.log"),
        "cwd": str(tmp_path),
        "env": {},
        "command": [sys.executable, "-c", "print('trained')"],
    }


def test_does_not_start_while_original_process_is_alive(job):
    job["prerequisite"]["processes"] = [
        {"pid": os.getpid(), "start_ticks": queue.process_identity(os.getpid())}
    ]
    assert queue.prerequisite_finished(job) == (False, "waiting_for_pi05_exit")


def test_reused_pid_does_not_block_completed_training(job):
    job["prerequisite"]["processes"] = [{"pid": os.getpid(), "start_ticks": "different-start-time"}]
    assert queue.prerequisite_finished(job) == (True, "pi05_completed")


@pytest.mark.parametrize("failure", ["missing_checkpoint", "wrong_step", "truncated_weights", "traceback"])
def test_rejects_incomplete_or_failed_training(job, failure):
    checkpoint = Path(job["prerequisite"]["checkpoint"])
    if failure == "missing_checkpoint":
        (checkpoint / "pretrained_model/config.json").unlink()
    elif failure == "wrong_step":
        (checkpoint / "training_state/training_step.json").write_text('{"step": 20000}')
    elif failure == "truncated_weights":
        path = checkpoint / "pretrained_model/model.safetensors"
        path.write_bytes(path.read_bytes()[:-2])
    else:
        Path(job["prerequisite"]["log"]).write_text("End of training\nTraceback (most recent call last):\n")
    assert not queue.prerequisite_finished(job)[0]


def test_runs_once_only_after_artifacts_are_complete(job, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert queue.run(path) == 0
    assert json.loads(Path(job["state_path"]).read_text())["status"] == "completed"
    assert Path(job["train_log"]).read_text() == "trained\n"
    with pytest.raises(RuntimeError, match="duplicate"):
        queue.run(path)


def test_does_not_launch_after_prerequisite_failure(job, tmp_path):
    Path(job["prerequisite"]["log"]).write_text("Interrupted\n")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert queue.run(path) == 1
    assert not Path(job["train_log"]).exists()
    assert json.loads(Path(job["state_path"]).read_text())["status"] == "prerequisite_failed"
