"""Run one local training job after an existing job saves its final checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


def process_identity(pid: int) -> str | None:
    """Use Linux start ticks as well as PID to avoid following a reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] == "Z":
        return None
    return fields[19]


def safetensors_complete(path: Path) -> bool:
    """Check that every declared tensor's bytes exist without loading the tensors."""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > min(size - 8, 64 * 1024**2):
                return False
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            return False
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        return bool(tensors) and all(
            0 <= tensor["data_offsets"][0] <= tensor["data_offsets"][1] <= size - 8 - length
            for tensor in tensors
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def prerequisite_finished(job: dict[str, Any]) -> tuple[bool, str]:
    prerequisite = job["prerequisite"]
    for process in prerequisite["processes"]:
        if process_identity(process["pid"]) == process["start_ticks"]:
            return False, "waiting_for_pi05_exit"
    checkpoint = Path(prerequisite["checkpoint"])
    required = (
        "pretrained_model/config.json",
        "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_postprocessor.json",
        "training_state/training_step.json",
        "training_state/optimizer_param_groups.json",
        "training_state/scheduler_state.json",
    )
    try:
        if any(not (checkpoint / name).is_file() for name in required):
            return False, "pi05_exited_without_final_checkpoint"
        metadata = json.loads((checkpoint / "training_state/training_step.json").read_text())
        if metadata.get("step") != prerequisite["steps"]:
            return False, "pi05_final_step_mismatch"
        trained = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
        if trained.get("steps") != prerequisite["steps"]:
            return False, "pi05_training_config_mismatch"
        for name in (
            "pretrained_model/model.safetensors",
            "training_state/optimizer_state.safetensors",
            "training_state/rng_state.safetensors",
        ):
            path = checkpoint / name
            if path.stat().st_mtime < prerequisite["started_at_epoch"] or not safetensors_complete(path):
                return False, "pi05_final_checkpoint_incomplete"
        with Path(prerequisite["log"]).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 256 * 1024))
            tail = stream.read().decode(errors="replace")
        if "End of training" not in tail or "Traceback (most recent call last)" in tail:
            return False, "pi05_did_not_finish_cleanly"
    except (OSError, ValueError, TypeError):
        return False, "pi05_completion_artifacts_unreadable"
    return True, "pi05_completed"


def write_state(path: Path, status: str, **values: Any) -> None:
    value = {"status": status, "updated_at_epoch": time.time(), "queue_pid": os.getpid(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(value), flush=True)


def run(job_path: Path) -> int:
    job = json.loads(job_path.read_text())
    state_path = Path(job["state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous["status"] in {"starting", "running", "completed", "failed"}:
                raise RuntimeError("This job was already launched; refusing a duplicate run")
        write_state(state_path, "waiting", reason="waiting_for_pi05_exit")
        while True:
            ready, reason = prerequisite_finished(job)
            if ready:
                break
            if reason != "waiting_for_pi05_exit":
                write_state(state_path, "prerequisite_failed", reason=reason)
                return 1
            time.sleep(job.get("poll_seconds", 15))

        output_dir = Path(job["output_dir"])
        if output_dir.exists():
            write_state(state_path, "failed", reason="fastwam_output_already_exists")
            return 1
        for filename, expected in job.get("sha256", {}).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                write_state(state_path, "failed", reason="queued_config_or_launcher_changed", path=filename)
                return 1
        write_state(state_path, "starting", reason="pi05_completed", command=job["command"])
        env = os.environ.copy()
        env.update(job["env"])
        child = None
        try:
            with Path(job["train_log"]).open("ab", buffering=0) as log:
                child = subprocess.Popen(job["command"], cwd=job["cwd"], env=env, stdout=log, stderr=log)
                write_state(state_path, "running", train_pid=child.pid, log=job["train_log"])
                returncode = child.wait()
            status = "completed" if returncode == 0 else "failed"
            write_state(state_path, status, exit_code=returncode, log=job["train_log"])
            return returncode
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
            write_state(state_path, "cancelled")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    raise SystemExit(run(args.job))


if __name__ == "__main__":
    main()
