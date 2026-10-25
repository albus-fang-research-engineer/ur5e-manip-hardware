"""manip_bridge/ur_motion.py -- straight-line IK, retiming, start check.
Offline; needs manip_cbirrt (PYTHONPATH=../manip-cbirrt from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "manip_bridge"))
pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
from manip_bridge import ur_motion as um  # noqa: E402

Q0 = np.array([2.8166, -1.2194, 1.0823, -0.1403, 2.4661, 1.0577])   # 10/23 recorded joints


@pytest.fixture(scope="module")
def ch():
    return um.chain()


def test_chain_is_base_link_and_tool0(ch):
    # ur_description UR5e at q = 0: tool0 at (a2 + a3, d4 + d6, d1 - d5) rotated
    # pi about z into base_link
    np.testing.assert_allclose(ch.fk(np.zeros(6))[:3, 3], [0.8172, 0.2329, 0.0628], atol=1e-4)


def test_line_is_straight_holds_orientation_and_ends_on_goal(ch):
    T0 = ch.fk(Q0)
    goal = T0[:3, 3] + 0.05 * T0[:3, 2]
    Q, info = um.line(ch, Q0, goal)
    np.testing.assert_array_equal(Q[0], Q0)
    P = um.tool_path(ch, Q)
    d, ax = P - P[0], T0[:3, 2]
    assert np.linalg.norm(d - np.outer(d @ ax, ax), axis=1).max() < 1e-4      # on the line
    assert np.linalg.norm(P[-1] - goal) < 1e-4
    assert np.all(np.diff(d @ ax) > 0) and np.diff(d @ ax).max() <= 0.005 + 1e-6
    for q in Q:
        assert np.linalg.norm(um._rot_err(T0[:3, :3], ch.fk(q)[:3, :3])) < 1e-3
    assert info["n_steps"] == 10 and info["joint_step"] < 0.05


def test_line_refuses_an_unreachable_goal(ch):
    with pytest.raises(um.MotionRefusal):
        um.line(ch, Q0, [2.0, 0.0, 0.5])


def test_time_optimal_respects_limits_ends_exactly_and_stretches():
    pytest.importorskip("toppra")
    # a bent joint path (two legs, densely sampled), the kind a planner returns
    knee = Q0 + [0.4, -0.2, 0.1, 0.0, 0.0, 0.3]
    s = np.linspace(0, 1, 60)[:, None]
    leg1 = Q0 + s * (knee - Q0)
    leg2 = knee + s[1:] * ([0.1, 0.3, -0.2, 0.2, 0.0, 0.0])
    Q = np.vstack([leg1, leg2])
    v, a = um.UR5E_VMAX * 0.1, um.UR5E_AMAX * 0.1
    t, Qs, V = um.time_optimal(Q, v, a)
    assert t[0] == 0 and np.all(np.diff(t) > 0) and np.diff(t).max() <= um.SAMPLE_DT + 1e-9
    np.testing.assert_array_equal(Qs[0], Q[0])
    np.testing.assert_array_equal(Qs[-1], Q[-1])
    np.testing.assert_array_equal(V[0], 0.0)
    np.testing.assert_array_equal(V[-1], 0.0)
    assert np.all(np.abs(V) <= v * 1.02)
    A = np.gradient(V, t, axis=0)
    assert np.abs(A).max() <= a.max() * 1.15                 # finite-difference slack
    # time-optimal: no slower than a velocity-limited lower bound allows, and
    # faster than one uniform profile at the worst segment's pace would be
    T_v = np.sum(np.max(np.abs(np.diff(Q, axis=0)) / v, axis=1))
    assert T_v <= t[-1] < T_v + 2 * v.max() / a.min() + 0.1
    t2, _, V2 = um.time_optimal(Q, v, a, min_duration=30.0)
    assert t2[-1] == pytest.approx(30.0) and np.abs(V2).max() < np.abs(V).max()
    assert np.diff(t2).max() <= um.SAMPLE_DT + 1e-9               # still dt apart after the stretch
    with pytest.raises(um.MotionRefusal, match="zero-length"):
        um.time_optimal(np.stack([Q0, Q0]))


def test_time_optimal_refuses_a_spline_that_leaves_sparse_waypoints():
    pytest.importorskip("toppra")
    corner = np.stack([Q0, Q0 + [0.6, 0, 0, 0, 0, 0], Q0 + [0.6, 0.6, 0, 0, 0, 0], Q0 + [0, 0.6, 0, 0, 0, 0]])
    with pytest.raises(um.MotionRefusal, match="leaves the joint path"):
        um.time_optimal(corner)


def test_names_and_start_check():
    names = list(reversed(um.DH_ORDER))
    np.testing.assert_array_equal(um.to_dh(Q0[::-1], names), Q0)
    np.testing.assert_array_equal(um.to_dh_path(np.stack([Q0[::-1]] * 3), names)[1], Q0)
    with pytest.raises(um.MotionRefusal, match="lacks"):
        um.to_dh(Q0[:5], um.DH_ORDER[:5])
    assert um.check_start(Q0 + 0.005, Q0) < 0.01
    with pytest.raises(um.MotionRefusal, match="stale plan"):
        um.check_start(Q0 + 0.02, Q0)
