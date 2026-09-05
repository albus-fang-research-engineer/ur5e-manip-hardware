"""Offline tests for manip_bridge.grasp_filter -- pure numpy, no ROS, no
sidecars. Imports the bridge module straight from the colcon source tree
(manip_bridge/__init__.py reads env vars only, so no rclpy is needed).

Fixture pattern follows ur5e-manip-sim's propose_handle_grasps: a cloud of
proposals deliberately WIDER than the TSR on every axis plus junk poses
elsewhere, so classification is a real filter with a nontrivial rejection
rate, not a pass-through.

Run from the repo root:  python -m pytest test/test_grasp_filter.py -v
(needs `pip install -e ~/manip-tsr` in the interpreter you use).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "manip_bridge"))

from manip_tsr import TSR, bounds, displacement_to_pose, make_pose  # noqa: E402
from manip_bridge.grasp_filter import (                                # noqa: E402
    AG_TO_E, E_TO_AG, ROUTE_CONTAINED, ROUTE_DISTANCE, ROUTE_EMPTY,
    Grasp, anygrasp_to_e, e_to_anygrasp, filter_grasps, grasps_from_anygrasp,
)

RNG = np.random.default_rng(0)


def random_pose(rng) -> np.ndarray:
    return make_pose(rng.normal(size=3), R.random(rng=rng).as_matrix())


def grasp_tsr() -> TSR:
    """A stage-1-shaped region: tight lateral, a slide band along the bar,
    small roll/pitch, wide wrap. The nominal Tw_e is arbitrary here -- the
    filter must not care what it is."""
    return TSR(T0_w=make_pose([0.5, -0.2, 0.9], R.from_euler("z", 0.7).as_matrix()),
               Tw_e=make_pose([0.0, 0.0, 0.0], R.from_euler("x", -0.6).as_matrix()),
               Bw=bounds(x=(-0.005, 0.005), y=(-0.005, 0.005), z=(-0.02, 0.02),
                         roll=(-0.09, 0.09), pitch=(-0.09, 0.09),
                         yaw=(-np.pi / 4, np.pi / 4)),
               name="grasp/test")


def inside(tsr: TSR, rng, n: int) -> list[np.ndarray]:
    return [tsr.sample(rng) for _ in range(n)]


def outside_near(tsr: TSR, rng, n: int, excess: float) -> list[np.ndarray]:
    """Poses just past the z (slide) bound by `excess` metres -- rejected,
    with tsr_distance == excess up to sampling in the other axes."""
    out = []
    for _ in range(n):
        d = tsr.sample_displacement(rng)
        d[2] = tsr.Bw[2, 1] + excess
        out.append(tsr.T0_w @ displacement_to_pose(d) @ tsr.Tw_e)
    return out


def junk(rng, n: int) -> list[np.ndarray]:
    return [make_pose([2.0, 2.0, 0.0] + rng.normal(0, 0.3, 3),
                      R.random(rng=rng).as_matrix()) for _ in range(n)]


def as_grasps(poses, scores, pad_offset=0.0) -> list[Grasp]:
    """e-frame poses -> AnyGrasp arrays -> Grasp list, exercising the full
    conversion path rather than constructing Grasp directly."""
    Rs, ts = zip(*(e_to_anygrasp(T, pad_offset) for T in poses))
    n = len(poses)
    return grasps_from_anygrasp(np.stack(Rs), np.stack(ts), scores,
                                widths=np.full(n, 0.06), depths=np.full(n, 0.03),
                                pad_offset=pad_offset)


# ------------------------------------------------------------------- frames


def test_permutation_maps_axes_and_keeps_handedness():
    assert tuple(AG_TO_E[i] for i in E_TO_AG) == (0, 1, 2)
    T = anygrasp_to_e(np.eye(3), np.zeros(3))
    Re = T[:3, :3]
    np.testing.assert_allclose(Re[:, 2], [1, 0, 0])   # e +z (approach)  = ag +x
    np.testing.assert_allclose(Re[:, 0], [0, 1, 0])   # e +x (closing)   = ag +y
    np.testing.assert_allclose(Re[:, 1], [0, 0, 1])   # e +y             = ag +z
    for _ in range(20):
        Rag = R.random(rng=RNG).as_matrix()
        Re = anygrasp_to_e(Rag, np.zeros(3))[:3, :3]
        assert np.isclose(np.linalg.det(Re), 1.0)
        np.testing.assert_allclose(Re.T @ Re, np.eye(3), atol=1e-12)


def test_pad_offset_moves_origin_along_approach_only():
    Rag = R.random(rng=RNG).as_matrix()
    t = RNG.normal(size=3)
    T0 = anygrasp_to_e(Rag, t, 0.0)
    T1 = anygrasp_to_e(Rag, t, 0.012)
    np.testing.assert_allclose(T1[:3, :3], T0[:3, :3])
    np.testing.assert_allclose(T1[:3, 3] - T0[:3, 3], 0.012 * Rag[:, 0])
    np.testing.assert_allclose(T1[:3, 3] - T0[:3, 3], 0.012 * T0[:3, 2])  # = e's +z


def test_roundtrip_e_anygrasp():
    for pad in (0.0, 0.02):
        for _ in range(10):
            T = random_pose(RNG)
            Rag, tag = e_to_anygrasp(T, pad)
            np.testing.assert_allclose(anygrasp_to_e(Rag, tag, pad), T, atol=1e-12)


def test_reference_frame_transform():
    """Proposals given in a camera frame + T_ref_cam classify identically to
    the same proposals given directly in the reference frame."""
    tsr = grasp_tsr()
    T_ref_cam = random_pose(RNG)
    poses_ref = inside(tsr, RNG, 5) + junk(RNG, 5)
    poses_cam = [np.linalg.inv(T_ref_cam) @ T for T in poses_ref]
    Rs, ts = zip(*(e_to_anygrasp(T) for T in poses_cam))
    scores = np.linspace(1.0, 0.1, 10)
    direct = filter_grasps(tsr, as_grasps(poses_ref, scores))
    via_cam = filter_grasps(tsr, grasps_from_anygrasp(
        np.stack(Rs), np.stack(ts), scores, np.full(10, 0.06), np.full(10, 0.03),
        T_ref_cam=T_ref_cam))
    assert [g.index for g in direct.survivors] == [g.index for g in via_cam.survivors]
    np.testing.assert_allclose([g.tsr_distance for g in direct.rejected],
                               [g.tsr_distance for g in via_cam.rejected], atol=1e-9)


def test_array_length_mismatch_raises():
    with pytest.raises(ValueError):
        grasps_from_anygrasp(np.zeros((3, 3, 3)), np.zeros((3, 3)), [1, 1],
                             [0.06] * 3, [0.03] * 3)


# ------------------------------------------------------------------- funnel


def test_contained_route_funnel_and_order():
    tsr = grasp_tsr()
    n_in, n_near, n_junk = 8, 6, 10
    poses = inside(tsr, RNG, n_in) + outside_near(tsr, RNG, n_near, 0.01) + junk(RNG, n_junk)
    # scores NOT monotone in construction order, so the sort has work to do
    scores = RNG.uniform(0.1, 1.0, size=len(poses))
    res = filter_grasps(tsr, as_grasps(poses, scores))

    assert res.route == ROUTE_CONTAINED
    assert res.n_in == len(poses)
    assert res.n_contained == n_in
    assert len(res.survivors) == n_in
    assert {g.index for g in res.survivors} == set(range(n_in))
    # every survivor is inside -> distance exactly 0; ordering is AnyGrasp score desc
    assert all(g.tsr_distance == 0.0 for g in res.survivors)
    sv = [g.score for g in res.survivors]
    assert sv == sorted(sv, reverse=True)
    # rejected = the rest, distance ascending, near ones before junk
    assert len(res.rejected) == n_near + n_junk
    rd = [g.tsr_distance for g in res.rejected]
    assert rd == sorted(rd) and rd[0] > res.tol
    assert all(g.index >= n_in for g in res.rejected)
    assert set(g.index for g in res.rejected[:n_near]) == set(range(n_in, n_in + n_near))


def test_score_ties_keep_anygrasp_order():
    tsr = grasp_tsr()
    poses = inside(tsr, RNG, 6)
    res = filter_grasps(tsr, as_grasps(poses, np.full(6, 0.5)))
    assert [g.index for g in res.survivors] == list(range(6))


def test_distance_route_takes_nearest_k_under_cap():
    tsr = grasp_tsr()
    poses = (outside_near(tsr, RNG, 4, 0.005)      # d ~ 0.005: within cap
             + outside_near(tsr, RNG, 3, 0.02)     # d ~ 0.02:  within cap
             + outside_near(tsr, RNG, 2, 0.10)     # d ~ 0.10:  beyond cap
             + junk(RNG, 5))
    scores = RNG.uniform(0.1, 1.0, size=len(poses))
    res = filter_grasps(tsr, as_grasps(poses, scores), max_distance=0.03, fallback_k=5)

    assert res.route == ROUTE_DISTANCE
    assert res.n_contained == 0
    assert len(res.survivors) == 5                              # k caps the 7 within reach
    sd = [g.tsr_distance for g in res.survivors]
    assert sd == sorted(sd) and all(0 < d <= 0.03 for d in sd)  # distance order, NOT score order
    assert {g.index for g in res.survivors[:4]} == {0, 1, 2, 3}  # the 0.005 band first
    assert len(res.rejected) == len(poses) - 5


def test_fallback_k_zero_means_no_fallback():
    tsr = grasp_tsr()
    res = filter_grasps(tsr, as_grasps(outside_near(tsr, RNG, 3, 0.005), [1, 1, 1]),
                        fallback_k=0)
    assert res.route == ROUTE_EMPTY and res.survivors == [] and len(res.rejected) == 3


def test_empty_route():
    tsr = grasp_tsr()
    res = filter_grasps(tsr, as_grasps(junk(RNG, 7), np.ones(7)), max_distance=0.03)
    assert res.route == ROUTE_EMPTY
    assert res.survivors == [] and len(res.rejected) == 7
    assert all(g.tsr_distance > 0.03 for g in res.rejected)

    empty = filter_grasps(tsr, [])
    assert empty.route == ROUTE_EMPTY and empty.n_in == 0 and empty.rejected == []


def test_report_is_json_safe_and_complete():
    import json
    tsr = grasp_tsr()
    poses = inside(tsr, RNG, 3) + junk(RNG, 4)
    res = filter_grasps(tsr, as_grasps(poses, np.linspace(1, 0.4, 7)))
    d = json.loads(json.dumps(res.to_dict()))
    assert d["route"] == ROUTE_CONTAINED and d["tsr"] == "grasp/test"
    assert d["n_in"] == 7 and d["n_contained"] == 3 and d["n_kept"] == 3
    assert [p["index"] for p in d["proposals"]] == list(range(7))    # original order
    assert [p["kept"] for p in d["proposals"]] == [True] * 3 + [False] * 4
    assert d["survivor_indices"] == [0, 1, 2]
    assert "3 contained -> 3 kept [tsr_contained]" in res.summary()
