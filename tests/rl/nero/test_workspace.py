import numpy as np
import pytest

from examples.nero_rl.workspace import summarize_sweep


def test_sweep_preserves_measured_extents_and_flags_unverified():
    q = np.zeros((20, 7))
    xyz = np.linspace([0.1, -0.1, 0.1], [0.3, 0.1, 0.2], 20)
    result = summarize_sweep(q, xyz, np.array([0.2, 0, 0.25]))
    assert result["workspace_min_m"] == pytest.approx([0.1, -0.1, 0.1])
    assert result["workspace_max_m"] == pytest.approx([0.3, 0.1, 0.25])
    assert result["outside_joint_limits"] == []
    assert not result["geometry_verified"]


def test_sweep_flags_out_of_limit_joint():
    q = np.zeros((20, 7))
    q[3, 6] = np.deg2rad(97)
    xyz = np.linspace([0, 0, 0], [0.1, 0.1, 0.1], 20)
    assert summarize_sweep(q, xyz, np.zeros(3))["outside_joint_limits"] == [7]


@pytest.mark.parametrize("xyz", [np.zeros((20, 3)), np.full((20, 3), np.nan)])
def test_sweep_rejects_inadequate_data(xyz):
    with pytest.raises(ValueError):
        summarize_sweep(np.zeros((20, 7)), xyz, np.zeros(3))
