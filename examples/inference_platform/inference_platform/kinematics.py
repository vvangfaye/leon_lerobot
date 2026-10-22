"""Use the existing NERO FK/IK; standard Piper uses its SDK MDH chain."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def eef(robot: str, joints: list, gripper: float) -> list:
    if robot == "nero":
        from lerobot.robots.nero_dual_eef.nero_dual_eef import joints_to_eef10

        return joints_to_eef10(joints, gripper)
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from pyAgxArm.utiles.tf import rpy_to_rot

    x, y, z, roll, pitch, yaw = fk_from_mdh(get_mdh("piper"), list(joints))
    rot = np.asarray(rpy_to_rot(roll, pitch, yaw))
    return [x, y, z, *rot[:, 0], *rot[:, 1], float(np.clip(gripper * 20 - 1, -1, 1))]


def solve(robot: str, target: np.ndarray, seed: np.ndarray, limits: np.ndarray) -> np.ndarray:
    a, b = target[3:6], target[6:9]
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(np.cross(a, b)) < 1e-6:
        raise ValueError("EEF 旋转向量退化")
    if np.any(seed < limits[:, 0]) or np.any(seed > limits[:, 1]):
        raise ValueError("EEF 起点超出限位，请先回到初始位置")
    if robot == "nero":
        from lerobot.robots.nero_dual_eef.nero_dual_eef import solve_ik

        q, ok, pos, rot = solve_ik(target, seed, limits, 0.20, pos_tol_mm=5, rot_tol_deg=3)
    else:
        a = a / np.linalg.norm(a)
        b = b - a * np.dot(a, b)
        b /= np.linalg.norm(b)
        rotation = np.column_stack((a, b, np.cross(a, b)))
        lower, upper = np.maximum(limits[:, 0], seed - 0.20), np.minimum(limits[:, 1], seed + 0.20)

        def errors(q: np.ndarray) -> tuple:
            pose = np.asarray(eef(robot, q, 0))
            measured = np.column_stack((pose[3:6], pose[6:9], np.cross(pose[3:6], pose[6:9])))
            return pose[:3] - target[:3], Rotation.from_matrix(rotation.T @ measured).as_rotvec()

        def residual(q: np.ndarray) -> np.ndarray:
            p, r = errors(q)
            return np.r_[p, 0.15 * r, 0.0001 * (q - seed)]

        fit = least_squares(
            residual, np.clip(seed, lower + 1e-9, upper - 1e-9), bounds=(lower, upper), max_nfev=100
        )
        p, r = errors(fit.x)
        q, pos, rot = fit.x, np.linalg.norm(p) * 1000, np.degrees(np.linalg.norm(r))
        ok = pos <= 5 and rot <= 3
    if not ok:
        raise ValueError(f"EEF 无法到达：位置误差 {pos:.1f} mm / 姿态误差 {rot:.1f}°")
    return np.asarray(q)
