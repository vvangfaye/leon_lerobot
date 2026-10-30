"""Record a hand-guided workspace sweep through the existing NERO workbench.

This process only reads joint feedback; the operator controls the workbench's
hand-guiding switch. It never enables, moves, or disables actuators.
"""

import argparse
import json
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

from examples.nero_to_nero.nero_platform.joint_limits import AngleReader

from .calibrate import atomic_json
from .config import JOINT_LIMITS, TaskConfig
from .hardware import NeroHardware


def summarize_sweep(joints: np.ndarray, xyz: np.ndarray, start_xyz: np.ndarray) -> dict:
    joints = np.asarray(joints, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 7 or xyz.shape != (len(joints), 3):
        raise ValueError("Invalid workspace samples")
    if len(joints) < 20 or not np.isfinite(joints).all() or not np.isfinite(xyz).all():
        raise ValueError("Need at least 20 finite samples")
    violations = (
        np.flatnonzero(
            np.any((joints < JOINT_LIMITS[:, 0] - 1e-3) | (joints > JOINT_LIMITS[:, 1] + 1e-3), axis=0)
        )
        + 1
    )
    points = np.vstack([xyz, np.asarray(start_xyz).reshape(1, 3)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    if np.any(hi - lo < 0.005):
        raise ValueError("Sweep must span at least 5 mm on each Cartesian axis")
    return {
        "samples": len(joints),
        "workspace_min_m": lo.tolist(),
        "workspace_max_m": hi.tolist(),
        "span_mm": ((hi - lo) * 1000).tolist(),
        "outside_joint_limits": violations.tolist(),
        "geometry_verified": False,
        "note": "An observed path does not certify the interior of its bounding box or full-arm clearance.",
    }


def survey_active(control: dict, kind: str = "workspace_survey") -> bool:
    if any(control.get(k) for k in ("recording", "measuring", "homing", "hand_guiding_channel")):
        raise RuntimeError("Conflicting workbench operation")
    if control.get("teleop_test_error"):
        raise RuntimeError(f"Workbench survey error: {control['teleop_test_error']}")
    if control.get("teleop_test_active"):
        if control.get("teleop_test_kind") != kind:
            raise RuntimeError("Use the slow relative workspace survey, not ordinary absolute teleoperation")
        return bool(control.get("teleop_test_ready"))
    return False


def record(cfg: TaskConfig, output: Path, url: str, timeout_s: float, source: str = "hand_guiding") -> None:
    cfg.validate()
    if cfg.channel != "can1" or cfg.tcp_offset_m is None or cfg.start_joints_rad is None:
        raise ValueError("Need right-arm TCP and recorded start pose")
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def status() -> dict:
        with opener.open(url.rstrip("/") + "/api/status", timeout=3) as response:
            return json.load(response)

    initial = status()
    if any(initial.get(k) for k in ("recording", "teleop_test_active", "measuring", "homing")):
        raise ValueError("Stop other workbench operations before starting the workspace reader")
    backend = NeroHardware(replace(cfg, cameras={}))
    rows, xyz, timestamps = [], [], []
    ended_by_operator = False
    try:
        backend.open()
        start = backend.read(images=False)
        if np.max(np.abs(start.joints - np.array(cfg.start_joints_rad))) > cfg.start_tolerance_rad:
            raise ValueError("Begin this sweep at the recorded task start pose")
        reader = AngleReader(backend.arm)
        began = time.monotonic()
        active = False
        last_status = 0.0
        control = initial
        waiting = (
            "waiting_for_workspace_survey" if source in ("survey", "aligned") else "waiting_for_hand_guiding"
        )
        atomic_json(output / "status.json", {"phase": waiting, "samples": 0, "source": source})
        print(f"READY: waiting for {source}; reader sends no actuator commands", flush=True)
        while time.monotonic() - began < timeout_s:
            now = time.monotonic()
            if now - last_status >= 0.2:
                control = status()
                last_status = now
                if source in ("survey", "aligned"):
                    kind = "workspace_aligned" if source == "aligned" else "workspace_survey"
                    is_active = survey_active(control, kind)
                    stopped = not control.get("teleop_test_active")
                else:
                    if any(
                        control.get(k) for k in ("recording", "teleop_test_active", "measuring", "homing")
                    ):
                        raise RuntimeError("Conflicting workbench operation")
                    channel = control.get("hand_guiding_channel")
                    if channel not in (None, "can1"):
                        raise RuntimeError("Unexpected guiding arm")
                    if control.get("hand_guiding_phase") == "error":
                        raise RuntimeError("Workbench hand-guiding error")
                    is_active = channel == "can1" and control.get("hand_guiding_phase") == "active"
                    stopped = channel is None
                if active and stopped:
                    ended_by_operator = True
                    break
                active = active or is_active
                atomic_json(
                    output / "status.json",
                    {"phase": "recording" if active else waiting, "samples": len(rows), "source": source},
                )
            if active:
                sample = reader.read(now)
                if sample is not None:
                    q = np.array(sample[0])
                    rows.append(q)
                    xyz.append(backend.forward(q)[:3, 3])
                    timestamps.append(now - began)
            time.sleep(0.02)
        if not ended_by_operator:
            raise TimeoutError("Workspace reader timed out; no live configuration enabled")
        report = summarize_sweep(np.array(rows), np.array(xyz), start.pose[:3, 3])
        report["returned_near_start"] = bool(
            np.max(np.abs(np.asarray(rows[-1]) - np.asarray(cfg.start_joints_rad))) <= cfg.start_tolerance_rad
        )
        atomic_json(output / "report.json", report)
        if report["outside_joint_limits"]:
            raise ValueError(f"Sweep exceeded conservative joint limits: {report['outside_joint_limits']}")
        draft = replace(
            cfg,
            workspace_min_m=report["workspace_min_m"],
            workspace_max_m=report["workspace_max_m"],
            geometry_verified=False,
        )
        draft.validate()
        draft.save(output / "live_config_workspace_draft.json")
        atomic_json(output / "status.json", {"phase": "awaiting_workspace_review", **report})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        atomic_json(
            output / "status.json", {"phase": "not_accepted", "error": str(exc), "samples": len(rows)}
        )
        raise
    finally:
        np.savez_compressed(
            output / "samples.npz", joints=np.array(rows), xyz=np.array(xyz), time_s=np.array(timestamps)
        )
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workbench", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--source", choices=["hand_guiding", "survey", "aligned"], default="hand_guiding")
    args = parser.parse_args()
    record(TaskConfig.load(args.config), args.output, args.workbench, args.timeout, args.source)


if __name__ == "__main__":
    main()
