"""Calibration algebra and capture quality checks; no physical hardware."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.nero_rl.calibrate import (
    FRAME,
    atomic_json,
    build_draft,
    fit_pivot,
    fit_session,
    load_session,
    stationary_sample,
)
from examples.nero_rl.config import TaskConfig


def pivot_data(noise: float = 0) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    offset = np.array([0.151, 0.003, -0.021])
    reference = np.array([0.35, 0.02, 0.11])
    angles = [[0, 0, 0], [25, 0, 0], [0, -30, 0], [15, 20, 25], [-20, 12, 5], [8, -15, -20]]
    poses = []
    rng = np.random.default_rng(11)
    for values in angles:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", values, degrees=True).as_matrix()
        pose[:3, 3] = reference - pose[:3, :3] @ offset + rng.normal(0, noise, 3)
        poses.append(pose)
    return poses, offset, reference


def test_recovers_known_offset_and_independent_validation():
    poses, offset, point = pivot_data()
    result = fit_pivot(poses[:4], poses[4:])
    np.testing.assert_allclose(result["calibration_point_offset_m"], offset, atol=1e-12)
    np.testing.assert_allclose(result["reference_point_in_base_m"], point, atol=1e-12)
    assert result["accepted"]


def test_repeatability_with_small_measurement_noise():
    poses, offset, _ = pivot_data(0.0001)
    result = fit_pivot(poses[:4], poses[4:])
    assert result["accepted"]
    assert np.linalg.norm(np.array(result["calibration_point_offset_m"]) - offset) < 0.001


def test_same_axis_or_identical_poses_do_not_constrain_tcp():
    poses, _, _ = pivot_data()
    with pytest.raises(ValueError, match="multiple axes"):
        fit_pivot([poses[0]] * 4, [])
    single_axis = []
    for angle in (0, 20, 40, 60):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        single_axis.append(pose)
    with pytest.raises(ValueError, match="multiple axes"):
        fit_pivot(single_axis, [])


def test_bad_fit_or_bad_holdout_never_accepted():
    poses, _, _ = pivot_data()
    corrupted = deepcopy(poses)
    corrupted[0][:3, 3] += [0.02, 0, 0]
    assert not fit_pivot(corrupted[:4], corrupted[4:])["accepted"]
    corrupted = deepcopy(poses)
    corrupted[4][:3, 3] += [0.02, 0, 0]
    result = fit_pivot(corrupted[:4], corrupted[4:])
    assert result["fit_passed"] and not result["holdout_passed"]
    assert not result["accepted"]
    assert not fit_pivot(poses[:4], [])["accepted"]
    assert not fit_pivot(poses[:4], [poses[0]])["accepted"]


def test_invalid_transform_rejected():
    poses, _, _ = pivot_data()
    poses[0][0, 0] = 2
    with pytest.raises(ValueError, match="rigid transform"):
        fit_pivot(poses[:4], poses[4:])


def test_capture_detects_motion(monkeypatch):
    monkeypatch.setattr("examples.nero_rl.calibrate.time.sleep", lambda seconds: None)
    counter = [0]

    def read(**kwargs):
        counter[0] += 1
        return SimpleNamespace(joints=np.ones(7) * counter[0] * 0.002, width=0.02)

    backend = SimpleNamespace(read=read, forward=lambda q: np.eye(4))
    with pytest.raises(ValueError, match="Arm moved"):
        stationary_sample(backend)


def test_capture_uses_stationary_average_without_control_calls(monkeypatch):
    monkeypatch.setattr("examples.nero_rl.calibrate.time.sleep", lambda seconds: None)
    backend = SimpleNamespace(
        read=lambda **kwargs: SimpleNamespace(joints=np.zeros(7), width=0.02), forward=lambda q: np.eye(4)
    )
    result = stationary_sample(backend)
    assert result["samples"] == 20
    assert result["pose_frame"] == FRAME


def draft_inputs() -> tuple[dict, dict]:
    poses, _, _ = pivot_data()
    fit = fit_pivot(poses[:4], poses[4:])
    session = {"reference": "grasp_center", "captures": []}
    for i, point in enumerate([[0.15, -0.1, 0.05], [0.45, 0.1, 0.3], [0.15, 0.1, 0.3], [0.45, -0.1, 0.05]]):
        pose = np.eye(4)
        pose[:3, 3] = point
        session["captures"].append(
            {
                "name": f"bound{i}",
                "kind": "workspace",
                "pose": pose.tolist(),
                "joints_rad": [0] * 7,
                "pose_frame": FRAME,
            }
        )
    pose = np.eye(4)
    pose[:3, 3] = [0.3, 0, 0.15]
    session["captures"].append(
        {"name": "start", "kind": "start", "pose": pose.tolist(), "joints_rad": [0] * 7, "pose_frame": FRAME}
    )
    return session, fit


def test_draft_never_arms_motion_and_preserves_source_config():
    session, fit = draft_inputs()
    cfg = TaskConfig()
    draft, report = build_draft(cfg, session, fit)
    assert draft.tcp_offset_m == fit["calibration_point_offset_m"]
    assert not draft.geometry_verified and not report["geometry_verified"]
    assert cfg.tcp_offset_m is None and draft.start_joints_rad == [0] * 7
    with pytest.raises(ValueError, match="Verify TCP"):
        draft.validate(live=True)


def test_auxiliary_tip_requires_explicit_measured_grasp_correction():
    session, fit = draft_inputs()
    session["reference"] = "auxiliary_tip"
    with pytest.raises(ValueError, match="Auxiliary"):
        build_draft(TaskConfig(), session, fit)
    draft, _ = build_draft(TaskConfig(), session, fit, grasp_offset_mm=[0, 0, 10])
    assert draft.tcp_offset_m[2] == pytest.approx(fit["calibration_point_offset_m"][2] + 0.01)


def test_start_must_fit_within_bounds_and_joint_limits():
    session, fit = draft_inputs()
    session["captures"][-1]["joints_rad"][1] = -2
    with pytest.raises(ValueError, match="joint limits"):
        build_draft(TaskConfig(), session, fit)
    session, fit = draft_inputs()
    session["captures"][-1]["pose"][0][3] = 2
    with pytest.raises(ValueError, match="outside"):
        build_draft(TaskConfig(), session, fit)


def test_fit_session_rejects_mixed_frames():
    with pytest.raises(ValueError, match="Mixed frame"):
        fit_session({"captures": [{"pose_frame": "camera"}]})


def test_invalidated_session_cannot_be_loaded_fitted_or_applied(tmp_path):
    session, fit = draft_inputs()
    session.update(schema=1, frame=FRAME, channel="can1", invalidated=True)
    atomic_json(tmp_path / "session.json", session)
    with pytest.raises(ValueError, match="invalidated"):
        load_session(tmp_path)
    with pytest.raises(ValueError, match="invalidated"):
        fit_session(session)
    with pytest.raises(ValueError, match="invalidated"):
        build_draft(TaskConfig(), session, fit)


def test_explicit_coarse_acceptance_keeps_failed_threshold_visible():
    session, fit = draft_inputs()
    fit["accepted"] = False
    fit["fit_rms_mm"] = 3.7
    fit["fit_errors_mm"] = [5.1, 2.2, 4.3, 1.0]
    with pytest.raises(ValueError, match="must pass"):
        build_draft(TaskConfig(), session, fit)
    draft, report = build_draft(TaskConfig(), session, fit, accept_coarse=True)
    assert not draft.geometry_verified
    assert report["tcp_quality"] == "coarse_accepted"
    assert report["tcp_threshold_passed"] is False
    assert report["tcp_fit_rms_mm"] == 3.7
    assert fit["accepted"] is False


def test_coarse_acceptance_cannot_skip_independent_check():
    session, fit = draft_inputs()
    fit["accepted"] = False
    fit["holdout_samples"] = 0
    with pytest.raises(ValueError, match="independent holdout"):
        build_draft(TaskConfig(), session, fit, accept_coarse=True)
