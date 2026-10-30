"""Read-only NERO calibration: stationary samples, pivot fitting, draft task bounds.

No subcommand enables, changes control modes, homes, or moves a robot.
"""

import argparse
import copy
import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .config import JOINT_LIMITS, TaskConfig, vector
from .hardware import NeroHardware

FRAME = "nero_sdk_mdh_link7_in_right_base_v1"
KINDS = ("tcp", "tcp_check", "workspace", "start")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def session_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_pose(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("A sample must contain a finite 4x4 transform")
    rotation = pose[:3, :3]
    if not (
        np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("Invalid rigid transform")
    return pose


def fit_pivot(poses: list[Any], checks: list[Any], *, max_error_m: float = 0.0015) -> dict:
    """Solve R_i t + p_i = c for tool offset t and fixed base-frame point c."""
    if len(poses) < 4:
        raise ValueError("Need at least four calibration poses at one fixed reference point")
    if not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("Error threshold must be finite and positive")
    matrices = [validate_pose(pose) for pose in poses]
    held_out = [validate_pose(pose) for pose in checks]
    a = np.concatenate([np.c_[pose[:3, :3], -np.eye(3)] for pose in matrices])
    b = np.concatenate([-pose[:3, 3] for pose in matrices])
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6 or singular[-1] <= 0 or singular[0] / singular[-1] > 100:
        raise ValueError("Poses do not constrain all TCP coordinates; vary orientation about multiple axes")
    offset, point = solution[:3], solution[3:]

    def errors(items: list[np.ndarray]) -> np.ndarray:
        return np.array([np.linalg.norm(pose[:3, :3] @ offset + pose[:3, 3] - point) for pose in items])

    residuals, check_errors = errors(matrices), errors(held_out)
    # A reread of a fitting pose is not an independent orientation check.
    distinct_checks = [
        all(
            np.arccos(np.clip((np.trace(pose[:3, :3].T @ old[:3, :3]) - 1) / 2, -1, 1)) > np.deg2rad(2)
            for old in matrices
        )
        for pose in held_out
    ]
    rms = float(np.sqrt(np.mean(residuals**2)))
    fit_ok = bool(np.max(residuals) <= max_error_m)
    check_ok = bool(len(check_errors) and all(distinct_checks) and np.max(check_errors) <= max_error_m)
    return {
        "frame": FRAME,
        "calibration_point_offset_m": offset.tolist(),
        "reference_point_in_base_m": point.tolist(),
        "fit_samples": len(matrices),
        "holdout_samples": len(held_out),
        "condition_number": float(singular[0] / singular[-1]),
        "fit_rms_mm": rms * 1000,
        "fit_errors_mm": (residuals * 1000).tolist(),
        "holdout_errors_mm": (check_errors * 1000).tolist(),
        "holdout_distinct_orientations": distinct_checks,
        "max_allowed_error_mm": max_error_m * 1000,
        "fit_passed": fit_ok,
        "holdout_passed": check_ok,
        "accepted": fit_ok and check_ok,
        "note": "Residuals test repeatability/model consistency, not absolute accuracy or collision safety.",
    }


def stationary_sample(
    backend: Any, *, samples: int = 20, period_s: float = 0.03, max_motion_rad: float = np.deg2rad(0.1)
) -> dict:
    if samples < 5 or period_s <= 0:
        raise ValueError("Need at least five samples and a positive sampling interval")
    joints, widths = [], []
    for _ in range(samples):
        snapshot = backend.read(images=False)
        joints.append(vector(snapshot.joints, 7))
        widths.append(snapshot.width)
        time.sleep(period_s)
    positions = np.stack(joints)
    spread = np.ptp(positions, axis=0)
    if np.max(spread) > max_motion_rad:
        raise ValueError("Arm moved during sampling; pause teleoperation and recapture")
    if not np.isfinite(widths).all() or np.ptp(widths) > 0.0005:
        raise ValueError("Gripper moved during sampling; keep the reference point fixed")
    average = positions.mean(axis=0)
    pose = validate_pose(backend.forward(average))
    return {
        "joints_rad": average.tolist(),
        "gripper_m": float(np.mean(widths)),
        "pose": pose.tolist(),
        "pose_frame": FRAME,
        "samples": samples,
        "joint_peak_to_peak_deg": np.rad2deg(spread).tolist(),
        "outside_conservative_limits": (
            np.flatnonzero((average < JOINT_LIMITS[:, 0]) | (average > JOINT_LIMITS[:, 1])) + 1
        ).tolist(),
    }


def load_session(root: Path) -> dict:
    session = json.loads((root / "session.json").read_text())
    if session.get("invalidated"):
        raise ValueError("This calibration session was invalidated; start a new session")
    if session.get("schema") != 1 or session.get("frame") != FRAME or session.get("channel") != "can1":
        raise ValueError("Unknown calibration session schema/frame/arm")
    return session


def capture(args: argparse.Namespace) -> dict:
    cfg = TaskConfig.load(args.config)
    cfg.validate()
    if cfg.channel != "can1":
        raise ValueError("Calibration is restricted to right follower can1")
    if not args.name or not all(c.isalnum() or c in "_-" for c in args.name):
        raise ValueError("Sample name must use letters, digits, underscores or hyphens")
    # Always capture the raw SDK link7 pose even if the supplied config has an old TCP.
    raw_cfg = replace(cfg, tcp_offset_m=None, cameras={} if args.no_images else cfg.cameras)
    with session_lock(args.session):
        session_file = args.session / "session.json"
        if session_file.exists():
            session = load_session(args.session)
            if args.reference is not None and args.reference != session["reference"]:
                raise ValueError("Reference type differs from this session; use a new session")
        else:
            if args.reference is None:
                raise ValueError("First capture requires --reference grasp_center or auxiliary_tip")
            session = {
                "schema": 1,
                "frame": FRAME,
                "channel": "can1",
                "reference": args.reference,
                "created_at": datetime.now(UTC).isoformat(),
                "captures": [],
            }
        if any(item["name"] == args.name for item in session["captures"]):
            raise ValueError("Sample name already exists; use a new name (nothing overwritten)")
        backend = NeroHardware(raw_cfg)
        try:
            backend.open()
            record = stationary_sample(backend)
            images = backend.read().images if not args.no_images else {}
        finally:
            backend.close()
        record.update(name=args.name, kind=args.kind, captured_at=datetime.now(UTC).isoformat())
        folder = args.session / args.name
        folder.mkdir(exist_ok=False)
        for name, rgb in images.items():
            if not cv2.imwrite(str(folder / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not save {name} image")
        atomic_json(folder / "sample.json", record)
        session["captures"].append(record)
        atomic_json(session_file, session)
    return {"read_only": True, "sample": str(folder / "sample.json"), **record}


def fit_session(session: dict, max_error_mm: float = 1.5) -> dict:
    if session.get("invalidated"):
        raise ValueError("This calibration session was invalidated; start a new session")
    for item in session["captures"]:
        if item["pose_frame"] != FRAME:
            raise ValueError("Mixed frame conventions in session")
    widths = [item["gripper_m"] for item in session["captures"] if item["kind"] in ("tcp", "tcp_check")]
    if widths and (not np.isfinite(widths).all() or np.ptp(widths) > 0.001):
        raise ValueError("Keep the same gripper opening during TCP calibration")
    fit = [item["pose"] for item in session["captures"] if item["kind"] == "tcp"]
    checks = [item["pose"] for item in session["captures"] if item["kind"] == "tcp_check"]
    result = fit_pivot(fit, checks, max_error_m=max_error_mm / 1000)
    result["reference"] = session["reference"]
    return result


def build_draft(
    cfg: TaskConfig,
    session: dict,
    fit: dict,
    *,
    grasp_offset_mm: list[float] | None = None,
    inset_mm: float = 0,
    accept_coarse: bool = False,
) -> tuple[TaskConfig, dict]:
    if session.get("invalidated"):
        raise ValueError("This calibration session was invalidated; start a new session")
    if not fit["accepted"] and not accept_coarse:
        raise ValueError("TCP fit and an independent holdout sample must pass before building a draft")
    if accept_coarse and (
        not fit.get("holdout_samples")
        or not all(fit.get("holdout_distinct_orientations", []))
        or not fit.get("holdout_errors_mm")
        or not np.isfinite(fit["holdout_errors_mm"]).all()
    ):
        raise ValueError("Coarse acceptance still requires a finite independent holdout measurement")
    if session["reference"] == "auxiliary_tip" and grasp_offset_mm is None:
        raise ValueError(
            "Auxiliary tip is not the grasp center; provide measured --grasp-offset-mm in link7 axes"
        )
    correction = np.zeros(3) if grasp_offset_mm is None else vector(grasp_offset_mm, 3) / 1000
    offset = vector(fit["calibration_point_offset_m"], 3) + correction
    workspace = [item for item in session["captures"] if item["kind"] == "workspace"]
    starts = [item for item in session["captures"] if item["kind"] == "start"]
    if len(workspace) < 4 or len(starts) != 1:
        raise ValueError("Capture at least four workspace poses and exactly one task start pose")
    for item in workspace + starts:
        q = vector(item["joints_rad"], 7)
        if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
            raise ValueError(f"Sample {item['name']} exceeds RL's conservative joint limits")
    if not np.isfinite(inset_mm) or inset_mm < 0:
        raise ValueError("Workspace inset must be finite and nonnegative")

    def point(item: dict) -> np.ndarray:
        pose = validate_pose(item["pose"])
        return pose[:3, 3] + pose[:3, :3] @ offset

    xyz = np.stack([point(item) for item in workspace])
    lo, hi = xyz.min(axis=0) + inset_mm / 1000, xyz.max(axis=0) - inset_mm / 1000
    if np.any(hi - lo < 0.005):
        raise ValueError("Workspace must span at least 5 mm on every axis after any inset")
    start_xyz = point(starts[0])
    if np.any(start_xyz < lo) or np.any(start_xyz > hi):
        raise ValueError("Start TCP lies outside the proposed workspace; capture the required boundary poses")
    draft = copy.deepcopy(cfg)
    draft.tcp_offset_m = offset.tolist()
    draft.workspace_min_m, draft.workspace_max_m = lo.tolist(), hi.tolist()
    draft.start_joints_rad = starts[0]["joints_rad"]
    # A box inferred from samples does NOT certify its interior or full-arm swept geometry.
    draft.geometry_verified = False
    draft.validate()
    report = {
        "config_complete": True,
        "geometry_verified": False,
        "tcp_quality": "passes_threshold" if fit["accepted"] else "coarse_accepted",
        "tcp_threshold_passed": fit["accepted"],
        "tcp_fit_rms_mm": fit["fit_rms_mm"],
        "tcp_fit_max_mm": max(fit["fit_errors_mm"]),
        "tcp_holdout_errors_mm": fit["holdout_errors_mm"],
        "tcp_offset_m": offset.tolist(),
        "start_tcp_m": start_xyz.tolist(),
        "workspace_min_m": lo.tolist(),
        "workspace_max_m": hi.tolist(),
        "note": "Draft only. Check full-arm clearance and small manual moves before setting geometry_verified.",
    }
    return draft, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    read = subparsers.add_parser("capture", help="Read one stationary pose; never enable or move hardware")
    read.add_argument("--config", type=Path, required=True)
    read.add_argument("--session", type=Path, required=True)
    read.add_argument("--kind", choices=KINDS, required=True)
    read.add_argument("--name", required=True)
    read.add_argument("--reference", choices=["grasp_center", "auxiliary_tip"])
    read.add_argument("--no-images", action="store_true")
    fit = subparsers.add_parser("fit", help="Fit four or more poses, validate using tcp_check samples")
    fit.add_argument("--session", type=Path, required=True)
    fit.add_argument("--max-error-mm", type=float, default=1.5)
    build = subparsers.add_parser("build", help="Generate a task configuration draft, not a motion approval")
    build.add_argument("--session", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--max-error-mm", type=float, default=1.5)
    build.add_argument("--grasp-offset-mm", nargs=3, type=float)
    build.add_argument("--inset-mm", type=float, default=0)
    build.add_argument(
        "--accept-coarse",
        action="store_true",
        help="Explicitly accept measured coarse TCP accuracy; does not mark geometry verified",
    )
    args = parser.parse_args()
    if args.command == "capture":
        result = capture(args)
    else:
        with session_lock(args.session):
            session = load_session(args.session)
            result = fit_session(session, args.max_error_mm)
            atomic_json(args.session / "tcp_fit.json", result)
            if args.command == "build":
                if args.output.exists():
                    raise FileExistsError(
                        "Output already exists; choose a new path to preserve the old configuration"
                    )
                cfg = TaskConfig.load(args.config)
                if cfg.channel != session["channel"]:
                    raise ValueError("Task config and calibration session use different arms")
                cfg, result = build_draft(
                    cfg,
                    session,
                    result,
                    grasp_offset_mm=args.grasp_offset_mm,
                    inset_mm=args.inset_mm,
                    accept_coarse=args.accept_coarse,
                )
                atomic_json(args.output, vars(cfg))
                atomic_json(args.output.with_suffix(".quality.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
