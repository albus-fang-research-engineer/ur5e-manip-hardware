"""outputs/runs/grasp_reach.py -- M3 step 4's pure parts: the tool0 goal for
a grasp at a standoff, the start state, and the attempt order. Offline (no
ROS, no sidecar); the sidecar is a stub that records what it was asked.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gr():
    spec = importlib.util.spec_from_file_location(
        "grasp_reach", REPO_ROOT / "outputs" / "runs" / "grasp_reach.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _grasp(p, rotvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = p
    return T


def test_tool0_goal_puts_the_pads_at_the_standoff_behind_the_grasp(gr):
    T_e = _grasp([-0.88, 0.10, 0.05], [0.3, -0.8, 0.2])
    for so in (0.10, 0.05, 0.0):
        T0 = gr.tool0_goal(T_e, so)
        pad = T0 @ gr.T_tool0_pad()
        approach = T_e[:3, 2]
        np.testing.assert_allclose(pad[:3, 3], T_e[:3, 3] - so * approach, atol=1e-12)
        np.testing.assert_allclose(T0[:3, :3], T_e[:3, :3], atol=1e-12)   # same orientation
        # the flange sits TOOL0_PAD_Z_M further back along the approach
        np.testing.assert_allclose(T0[:3, 3], pad[:3, 3] - gr.TOOL0_PAD_Z_M * approach, atol=1e-12)
    assert gr.TOOL0_PAD_Z_M == pytest.approx(0.150)


def test_start_state_carries_names_with_q(gr, tmp_path):
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    (tmp_path / "summary.json").write_text(json.dumps(
        {"packet": {"joints": {"status": "from_source", "names": names, "q": [0.1] * 6}}}))
    q, n = gr.start_state(tmp_path)
    assert n == names and q == [0.1] * 6
    (tmp_path / "summary.json").write_text(json.dumps(
        {"packet": {"joints": {"status": "absent_in_source"}}}))
    with pytest.raises(gr.ReachRefusal, match="no recorded start state"):
        gr.start_state(tmp_path)


class Sidecar:
    """Succeeds only for the listed (grasp position x, standoff) pairs."""

    def __init__(self, ok):
        self.ok, self.calls = ok, []

    def __call__(self, payload):
        T = np.asarray(payload["T_base_goal"], float)
        self.calls.append(payload)
        hit = any(np.allclose(T, g, atol=1e-6) for g in self.ok)
        return ({"ok": True, "success": True, "position_error": 0.001, "rotation_error": 0.002,
                 "solve_time": 0.5, "joint_names": payload["joint_names"],
                 "positions": np.zeros((3, 6), np.float32), "velocities": np.zeros((3, 6), np.float32),
                 "dt": 0.02, "ee_path": np.zeros((3, 3), np.float32)} if hit
                else {"ok": True, "success": False, "error": "planning failed"})


def test_order_every_grasp_at_5cm_before_any_at_10cm(gr):
    assert gr.STANDOFFS_M == (0.05, 0.10)                      # the blind approach: short first
    g = [(_grasp([-0.88, 0.1 + 0.01 * k, 0.05], [0, 0, 0]), 0.03 - 0.005 * k) for k in range(3)]
    q, names = [0.0] * 6, [f"j{k}" for k in range(6)]
    # nothing reaches at 5 cm; grasp 2 reaches at 10 cm
    side = Sidecar([gr.tool0_goal(g[2][0], 0.10)])
    rows, best = gr.attempts(g, q, names, side)
    assert [(r["grasp"], r["standoff_m"]) for r in rows] == \
        [(0, 0.05), (1, 0.05), (2, 0.05), (0, 0.10), (1, 0.10), (2, 0.10)]
    assert best["grasp"] == 2 and best["standoff_m"] == 0.10 and best["success"]
    assert best["dt"] == 0.02 and np.asarray(best["positions"]).shape == (3, 6)
    assert all(c["joint_names"] == names and c["q"] == q for c in side.calls)
    # the first success stops the search
    side = Sidecar([gr.tool0_goal(g[0][0], 0.05)])
    rows, best = gr.attempts(g, q, names, side)
    assert len(rows) == 1 and best["grasp"] == 0
    rows, best = gr.attempts(g, q, names, Sidecar([]))
    assert best is None and len(rows) == 6
    json.dumps(rows)                                            # the record is JSON-safe
