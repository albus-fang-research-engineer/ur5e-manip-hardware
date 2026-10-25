"""manip_bridge/scene_ops.py -- poses, table plane, clear box, carried spheres.

Offline (no ROS, no sidecar), on the 10/23 registration run's recorded
numbers (/data/runs/20261023_032004: hand-eye, table plane, both
pose_on_final poses, the tracked meshes' extents). The meshes themselves are
not in the repo, so the objects stand in as their bounding boxes -- exact
for the clear box's extent along the body axes (final_mesh is bbox-centred),
a superset of the real shape everywhere else.

Two real-mesh checks run where the run is (the ROS2 container mounts
/data/runs and /data/any6d; the second also needs the cuRobo sidecar):

    SCENE_OPS_RUN=/data/runs/20261023_032004 python3 -m pytest test/test_scene_ops.py -v -s
        -k real_meshes        the teapot's clear box vs the mug
        -k real_mesh_spheres  MORPHIT spheres cover the surface at the 6 cm mask
                              threshold (SCENE_OPS_KEY=m2 by default)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

trimesh = pytest.importorskip("trimesh")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"))
from manip_bridge import scene_ops as so  # noqa: E402

# ---- 10/23 (20261023_032004), from summary.json (rotations re-orthonormalised)
_BC = np.array([[0.008266, -0.366931, 0.930212, -1.193813],
                [-0.9979, -0.062791, -0.015901, 0.272376],
                [0.064244, -0.928127, -0.366679, 0.229418],
                [0, 0, 0, 1]])
PLANE = {"n": [0.041035, -0.920205, -0.389279], "d": 0.257863}


def _so3(M):
    """The transcription rounds to 5-6 decimals (det 1 +- 7e-6); project back
    onto SO(3) so rigidity checks test the code, not the rounding."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float)[:3, :3])
    return U @ Vt


def _T(rows, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _so3(rows), t
    return T


CAM_T = {
    "m1": _T([[-0.92640, 0.07068, -0.36985], [0.10754, -0.89165, -0.43978],
              [-0.36086, -0.44718, 0.81842]], [-0.07514, 0.08032, 0.36386]),   # mug
    "m2": _T([[-0.58980, -0.23911, -0.77134], [0.51734, -0.84529, -0.13355],
              [-0.62007, -0.47781, 0.62226]], [0.09247, 0.07968, 0.39215]),    # teapot
}
T_BASE_CAM = _T(_BC[:3, :3], _BC[:3, 3]).tolist()
EXTENTS = {"m1": (0.1174, 0.0927, 0.0912), "m2": (0.0906, 0.1021, 0.1764)}


def summary(cam_T=CAM_T):
    return {"packet": {"t_base_cam": {"source": "tf", "T": T_BASE_CAM}, "plane": PLANE},
            "objects": {k: {"pose_on_final": {"cam_T_obj": T.tolist(),
                                              "mesh": f"/data/any6d/final_mesh_{k}.obj"}}
                        for k, T in cam_T.items()}}


def box_V(k):
    return np.asarray(trimesh.creation.box(EXTENTS[k]).vertices)


# ------------------------------------------------------------------ frames

def test_table_plane_in_base_is_nearly_horizontal_and_up():
    n, d = so.table_plane_base(summary())
    assert np.linalg.norm(n) == pytest.approx(1.0)
    assert n[2] > 0
    tilt = np.degrees(np.arccos(n[2]))
    assert tilt == pytest.approx(1.9, abs=0.1)         # 10/23: table 1.9 deg off base z
    # a point on the table in the camera frame is on the base plane
    n_c, d_c = np.asarray(PLANE["n"]), PLANE["d"]
    p_c = -d_c * n_c / (n_c @ n_c) + 0.1 * np.cross(n_c, [1, 0, 0])
    p_b = np.asarray(T_BASE_CAM)[:3, :3] @ p_c + np.asarray(T_BASE_CAM)[:3, 3]
    assert n @ p_b + d == pytest.approx(0.0, abs=1e-9)


def test_object_poses_compose_hand_eye_and_registration():
    s = summary()
    for k in CAM_T:
        np.testing.assert_allclose(so.T_base_obj(s, k), np.asarray(T_BASE_CAM) @ CAM_T[k])
    assert so.mesh_path(s, "m2") == "/data/any6d/final_mesh_m2.obj"
    with pytest.raises(ValueError, match="t_base_cam"):
        so.T_base_obj({"objects": s["objects"]}, "m2")
    with pytest.raises(ValueError, match="pose_on_final"):
        so.T_base_obj(s, "m9")


# --------------------------------------------------------------- clear box

@pytest.mark.parametrize("k", ["m1", "m2"])
def test_clear_box_contains_the_object_and_keeps_the_table(k):
    s = summary()
    plane, T = so.table_plane_base(s), so.T_base_obj(s, k)
    V = box_V(k)
    lo, hi = so.clear_box(V, T, plane, margin=0.02)
    W = V @ T[:3, :3].T + T[:3, 3]
    # every vertex above the bottom is inside with the margin to spare
    above = W[:, 2] >= lo[2]
    assert np.all(W[above] >= lo - 1e-12) and np.all(W <= hi - 0.02 + 1e-12)
    assert np.all(W[:, :2] >= lo[:2] + 0.02 - 1e-12)
    # the bottom: above the table everywhere under the box, by half a voxel
    corners_z = so.table_z(plane, [lo[0], hi[0], lo[0], hi[0]], [lo[1], lo[1], hi[1], hi[1]])
    assert lo[2] >= corners_z.max() + 0.5 * so.VOXEL_M - 1e-12
    # ... and not above the object's lowest point plus that half voxel: the
    # base of the object is cleared, the table under it is not
    assert lo[2] <= max(W[:, 2].min() - 0.02, corners_z.max() + 0.5 * so.VOXEL_M) + 1e-12


def test_clear_box_bottom_follows_a_tilted_table():
    V = box_V("m1")
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.05 + 0.0912 / 2]
    flat = (np.array([0, 0, 1.0]), -0.05)                  # z = 0.05
    n = R.from_euler("x", 5, degrees=True).apply([0, 0, 1.0])
    tilted = (n, -float(n @ [0.5, 0, 0.05]))               # through (0.5, 0, 0.05), 5 deg
    lo_f, _ = so.clear_box(V, T, flat)
    lo_t, _ = so.clear_box(V, T, tilted)
    assert lo_f[2] == pytest.approx(0.05 + 0.005)
    # the highest table point under the footprint, not the centre's height
    lo, hi = so.clear_box(V, T, flat)
    y_hi = hi[1] if n[1] < 0 else lo[1]
    assert lo_t[2] == pytest.approx(float(so.table_z(tilted, 0.5, y_hi)) + 0.005, abs=1e-9)
    assert lo_t[2] > lo_f[2]


def test_clear_box_refuses_an_object_below_the_table():
    V = box_V("m1")
    T = np.eye(4)
    T[2, 3] = -0.5
    with pytest.raises(ValueError, match="below the table"):
        so.clear_box(V, T, (np.array([0, 0, 1.0]), 0.0))


def test_10_23_teapot_clear_box_vs_mug():
    """The reviewer's check: does clearing the teapot delete mug voxels?
    On 10/23, with the meshes' bounding boxes standing in for the meshes:
    YES -- the teapot's base-aligned box (2 cm margin) reaches 34 mm into
    the mug's along base y (14 mm with no margin at all), because both
    objects sit yawed in base, which inflates their base-aligned boxes, and
    their centres are 17 cm apart. So conflicts() is a refusal grasp_plan
    must honour, and the real meshes (tighter than their boxes) decide it on
    the workstation: SCENE_OPS_RUN, below."""
    s = summary()
    plane = so.table_plane_base(s)
    teapot = so.clear_box(box_V("m2"), so.T_base_obj(s, "m2"), plane)
    mug = (box_V("m1"), so.T_base_obj(s, "m1"))
    c = so.conflicts(teapot, {"m1": mug})
    assert set(c) == {"m1"}
    assert c["m1"][1] == pytest.approx(0.034, abs=0.002)
    assert so.conflicts(so.clear_box(box_V("m2"), so.T_base_obj(s, "m2"), plane, margin=0.0),
                        {"m1": mug})["m1"][1] == pytest.approx(0.014, abs=0.002)
    # move the mug 5 cm further along base y: no conflict
    Tm = so.T_base_obj(s, "m1").copy()
    Tm[1, 3] += 0.05
    assert so.conflicts(teapot, {"m1": (box_V("m1"), Tm)}) == {}
    # the mug's box never reaches the teapot it is not grasping, either way round
    assert so.overlap(teapot, so.aabb(*mug)) and not so.overlap(teapot, so.aabb(box_V("m1"), Tm))


@pytest.mark.skipif(not os.environ.get("SCENE_OPS_RUN"), reason="set SCENE_OPS_RUN=<run dir>")
def test_real_meshes_teapot_clear_box_vs_mug():
    run = Path(os.environ["SCENE_OPS_RUN"])
    s = json.loads((run / "summary.json").read_text())
    plane = so.table_plane_base(s)
    V = {k: np.asarray(trimesh.load(so.mesh_path(s, k), force="mesh", process=False).vertices)
         for k in ("m1", "m2")}
    for margin in (0.0, 0.01, so.CLEAR_MARGIN_M):
        box = so.clear_box(V["m2"], so.T_base_obj(s, "m2"), plane, margin=margin)
        c = so.conflicts(box, {"m1": (V["m1"], so.T_base_obj(s, "m1"))})
        print(f"margin {margin*100:.0f} cm: teapot box {np.round(box[0], 3)}..{np.round(box[1], 3)}"
              f"  overlap with mug: {np.round(c['m1'] * 1000, 1).tolist() if c else 'none'} mm")
    assert not c, "the teapot's clear box reaches into the mug: move them apart"


# ------------------------------------------------------- carried spheres

def test_grasp_transform_round_trip_and_spheres_ride_with_tool0():
    s = summary()
    T_obj = so.T_base_obj(s, "m2")
    T_tool0 = np.eye(4)
    T_tool0[:3, :3] = R.from_rotvec([0.4, -1.2, 0.3]).as_matrix()
    T_tool0[:3, 3] = T_obj[:3, 3] + [0.05, -0.02, 0.18]
    T_eb = so.T_ee_body(T_tool0, T_obj)
    np.testing.assert_allclose(T_tool0 @ T_eb, T_obj, atol=1e-12)
    S_body = np.array([[0.0, 0.0, 0.0, 0.03], [0.02, -0.01, 0.07, 0.015], [-0.03, 0.02, -0.06, 0.02]])
    S_t = so.spheres_tool0(S_body, T_eb)
    np.testing.assert_array_equal(S_t[:, 3], S_body[:, 3])
    # the tool0-frame spheres, carried by tool0, land on the body's spheres
    world_via_tool0 = S_t[:, :3] @ T_tool0[:3, :3].T + T_tool0[:3, 3]
    world_via_body = S_body[:, :3] @ T_obj[:3, :3].T + T_obj[:3, 3]
    np.testing.assert_allclose(world_via_tool0, world_via_body, atol=1e-12)
    # rigid: move tool0 anywhere, the pairwise distances are the body's
    T2 = np.eye(4)
    T2[:3, :3] = R.from_rotvec([-0.7, 0.2, 1.1]).as_matrix()
    T2[:3, 3] = [-0.4, 0.3, 0.5]
    W2 = S_t[:, :3] @ T2[:3, :3].T + T2[:3, 3]
    D = lambda P: np.linalg.norm(P[:, None] - P[None], axis=-1)  # noqa: E731
    np.testing.assert_allclose(D(W2), D(S_body[:, :3]), atol=1e-12)


class FitStub:
    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        assert payload["cmd"] == "fit_spheres"
        self.calls += 1
        V = np.asarray(payload["vertices"])
        c = V.mean(axis=0)
        return {"ok": True, "fit_type": "voxel",
                "spheres": np.array([[*c, 0.01 * self.calls]] * payload["num_spheres"], np.float32)}


def test_object_spheres_cached_per_mesh_sha(tmp_path):
    mesh = tmp_path / "final_mesh_m2.obj"
    trimesh.creation.box(EXTENTS["m2"]).export(mesh)
    cache = tmp_path / "objects" / "m2" / "spheres_body.json"
    fit = FitStub()
    S1 = so.object_spheres(mesh, cache, fit)
    assert S1.shape == (16, 4) and fit.calls == 1
    rec = json.loads(cache.read_text())
    assert rec["fit_type"] == "voxel" and rec["num_spheres"] == 16 and len(rec["mesh_sha256"]) == 64
    np.testing.assert_allclose(so.object_spheres(mesh, cache, fit), S1)
    assert fit.calls == 1                                   # cached
    trimesh.creation.box((0.1, 0.1, 0.2)).export(mesh)      # mesh changed -> refit
    S2 = so.object_spheres(mesh, cache, fit)
    assert fit.calls == 2 and S2[0, 3] == pytest.approx(0.02)
    so.object_spheres(mesh, cache, fit, num_spheres=8)      # a different count -> refit
    assert fit.calls == 3


@pytest.mark.skipif(not os.environ.get("SCENE_OPS_RUN"), reason="set SCENE_OPS_RUN=<run dir>")
def test_real_mesh_spheres_cover_the_surface_at_the_mask_threshold():
    """The sidecar's held mask drops pixels within 6 cm of a held sphere's
    surface, so a mesh surface point farther than that from every sphere
    would leak into the map while the object is held. MORPHIT, 16 spheres,
    on the tracked mesh (its convex hull when not watertight); the fit must
    also model the object (cuRobo interior coverage >= 0.9). Reports the
    worst point (the spout tip is the suspect). Needs the cuRobo sidecar ($CUROBO_ADDR, default localhost)."""
    from manip_bridge.zmq_client import SidecarClient
    thr = 0.06
    run = Path(os.environ["SCENE_OPS_RUN"])
    key = os.environ.get("SCENE_OPS_KEY", "m2")
    s = json.loads((run / "summary.json").read_text())
    mesh = trimesh.load(so.mesh_path(s, key), force="mesh", process=False)
    cli = SidecarClient(os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671"), timeout_ms=600_000)
    rep = cli.call({"cmd": "fit_spheres", "vertices": np.asarray(mesh.vertices, np.float32),
                    "faces": np.asarray(mesh.faces, np.int32), "num_spheres": 16})
    assert rep.get("ok"), rep.get("error")
    S = np.asarray(rep["spheres"], float)
    pts = mesh.sample(20000)
    gap = (np.linalg.norm(pts[:, None] - S[None, :, :3], axis=-1) - S[None, :, 3]).min(axis=1)
    k = int(np.argmax(gap))
    frac = float((gap < thr).mean())
    print(f"\n[spheres] {key} ({rep.get('fit_type')}): {len(S)} spheres, cuRobo metrics "
          f"{ {m: round(v, 4) for m, v in (rep.get('metrics') or {}).items()} }")
    print(f"[spheres] {key}: surface within {thr*100:.0f} cm of a sphere: {frac:.4f}; worst gap "
          f"{gap[k]*100:.1f} cm at body {np.round(pts[k], 3).tolist()} "
          f"(mesh extents {np.round(mesh.extents, 3).tolist()})")
    print(f"[spheres] {key}: fitted to {rep.get('fitted_to')}")
    assert len(S) <= 16
    # a degenerate fit (a non-watertight mesh fitted raw) can pass the 6 cm
    # check by luck while modelling almost none of the object
    assert (rep.get("metrics") or {}).get("coverage", 0.0) >= 0.9, \
        "cuRobo's interior coverage < 0.9: the spheres do not model this object"
    assert frac >= 0.99, "the held mask would leak this object's surface into the map"
