"""cuRobo sidecar `hold` / `release` (held-object depth mask) and
`fit_spheres`, over the ZMQ wire. Same packet + sidecar as test_curobo_esdf;
the small-sphere case needs manip_cbirrt for an FK independent of the
sidecar's:

    docker compose up -d curobo
    PYTHONPATH=../manip-cbirrt python -m pytest test/test_curobo_scene_ops.py -v -s

hold, big sphere (wiring): nothing of the scene integrates; an integrate
  without q while holding is refused; release restores.
hold, small sphere (FK, joint order, frames): tool0 at the packet's q from
  manip_cbirrt's DH chain (validated against cuRobo FK by
  test_cbirrt_frames); a 3 cm sphere put, through that pose, on a table
  point. Against the same frame integrated without a hold (same q, so the
  robot mask is identical in both): that table probe goes unobserved while
  a probe 20-25 cm away reads the same; the held mask covers the pixel and
  not the far one; the sidecar's T_base_tool0 equals the DH chain's.
Every test releases on teardown, so a long-running sidecar is never left
masking a phantom object.

The real-mesh check (MORPHIT spheres on the tracked teapot leave no surface
point beyond the 6 cm mask threshold) needs the run's meshes, so it lives in
test_scene_ops.py (-k real_mesh_spheres) and runs in the ROS2 container.
"""
import numpy as np
import pytest

from test_curobo_esdf import (CALL_TIMEOUT, FIRST_CALL_TIMEOUT, _esdf_lookup,  # noqa: F401
                              _T_map_cam, mapper)

trimesh = pytest.importorskip("trimesh")

DH_ORDER = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
THR = 0.06                       # the robot mask's default distance_threshold


@pytest.fixture
def released(mapper):
    mapper.ok({"cmd": "release"}, timeout_ms=CALL_TIMEOUT)
    yield mapper
    mapper.ok({"cmd": "release"}, timeout_ms=CALL_TIMEOUT)


# ------------------------------------------------------------------ frames

def _q(packet):
    return (np.asarray(packet["qpos_arm"], np.float32),
            [str(n) for n in packet.meta["joint_names_arm"]])


def _T_base_cam(packet):
    return (np.linalg.inv(packet["T_world_base"].astype(np.float64))
            @ packet["T_world_cam"].astype(np.float64))


def _T_map_world(packet):
    T = np.eye(4)
    T[:3, 3] = -np.asarray(packet.meta["table_offset"], np.float64)
    return T


def _integrate_q(m, packet, **extra):
    q, names = _q(packet)
    return m.ok({"cmd": "integrate", "depth": packet["depth"].astype(np.float32),
                 "intrinsics": packet["K"].astype(np.float32), "pose": _T_map_cam(packet),
                 "T_base_cam": _T_base_cam(packet).astype(np.float32),
                 "q": q, "joint_names": names, **extra}, timeout_ms=FIRST_CALL_TIMEOUT)


def _pixel_to_map(packet, v, u):
    z = float(packet["depth"][v, u])
    K = packet["K"]
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    T = _T_map_cam(packet).astype(np.float64)
    return T[:3, :3] @ p_cam + T[:3, 3], T[:3, :3] @ (p_cam / np.linalg.norm(p_cam))


def _table_pixels(packet):
    """(near, far): two observed table pixels 20-25 cm apart, both away from
    the robot's GT mask, the near one closest to the table-mask centroid."""
    depth = packet["depth"]
    ok = packet["mask_table"].astype(bool) & (depth > 0)
    if "mask_robot" in packet.arrays:
        from scipy.ndimage import binary_dilation
        ok &= ~binary_dilation(packet["mask_robot"].astype(bool), iterations=25)
    vs, us = np.nonzero(ok)
    assert len(vs) > 100, "too few table pixels away from the robot"
    i = int(np.argmin((vs - vs.mean()) ** 2 + (us - us.mean()) ** 2))
    p0, _ = _pixel_to_map(packet, vs[i], us[i])
    sel = np.linspace(0, len(vs) - 1, min(len(vs), 4000)).astype(int)
    P = np.stack([_pixel_to_map(packet, vs[k], us[k])[0] for k in sel])
    d = np.linalg.norm(P - p0, axis=1)
    cand = np.nonzero((d > 0.20) & (d < 0.25))[0]
    assert len(cand), "no table pixel 20-25 cm from the near probe"
    j = sel[cand[0]]
    return (int(vs[i]), int(us[i])), (int(vs[j]), int(us[j]))


def _dh_tool0(packet):
    """tool0 in base at the packet's q, by manip_cbirrt's DH chain (base_link
    = DH base rotated pi about z, as cbirrt_backend.dh_chain)."""
    mc = pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
    Rz = np.diag([-1.0, -1.0, 1.0, 1.0])
    q, names = _q(packet)
    q_dh = np.array([q[names.index(n)] for n in DH_ORDER], float)
    return mc.ur5e_chain(T_base=Rz, T_tool=np.eye(4)).fk(q_dh)


# ------------------------------------------------------------------- hold

def test_hold_big_sphere_masks_everything_and_release_restores(released, packet):
    m = released
    (vn, un), _ = _table_pixels(packet)
    p_surf, ray = _pixel_to_map(packet, vn, un)
    p_free, p_occ = p_surf - 0.12 * ray, p_surf + 0.04 * ray

    m.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    m.ok({"cmd": "hold", "spheres": np.array([[0.0, 0.0, 0.0, 5.0]], np.float32)},
         timeout_ms=CALL_TIMEOUT)
    rep = _integrate_q(m, packet)
    n_valid = int((packet["depth"] > 0).sum())
    print(f"[hold] big sphere: {rep['n_masked_held']} of {n_valid} valid px held-masked")
    assert rep["n_masked_held"] >= 0.95 * (n_valid - rep["n_masked"])
    d_free, d_occ = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [p_free, p_occ])
    assert abs(d_free - d_occ) < 0.08, "the table was integrated through a full held mask"

    err = m.err({"cmd": "integrate", "depth": packet["depth"].astype(np.float32),
                 "intrinsics": packet["K"].astype(np.float32), "pose": _T_map_cam(packet)},
                timeout_ms=CALL_TIMEOUT)
    assert "held" in err["error"]

    assert m.ok({"cmd": "release"}, timeout_ms=CALL_TIMEOUT)["was_held"] is True
    rep = _integrate_q(m, packet)
    assert "n_masked_held" not in rep
    d_occ = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [p_occ])[0]
    assert d_occ < 0.03, f"table not integrated after release ({d_occ:.3f} m)"


def test_hold_small_sphere_lands_where_fk_says(released, packet):
    m = released
    T_bt = _dh_tool0(packet)
    (vn, un), (vf, uf) = _table_pixels(packet)
    probes = []
    for v, u in ((vn, un), (vf, uf)):
        p_surf, ray = _pixel_to_map(packet, v, u)
        probes.append((p_surf, p_surf + 0.04 * ray))       # (surface, 4 cm behind)
    # the sphere: on the near table point, given in tool0 through the DH pose
    T_bw = np.linalg.inv(packet["T_world_base"].astype(np.float64))
    p_base = (T_bw @ np.linalg.inv(_T_map_world(packet)) @ np.r_[probes[0][0], 1])[:3]
    c_tool0 = (np.linalg.inv(T_bt) @ np.r_[p_base, 1])[:3]
    sphere = np.array([[*c_tool0, 0.03]], np.float32)

    # the same frame, same q, no hold: the reference (robot mask identical)
    m.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate_q(m, packet)
    ref = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [probes[0][1], probes[1][1]])
    assert ref[0] < 0.03 and ref[1] < 0.03, f"probes not on an integrated table: {ref}"

    m.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    m.ok({"cmd": "hold", "spheres": sphere, "threshold": THR}, timeout_ms=CALL_TIMEOUT)
    rep = _integrate_q(m, packet, return_held_mask=True)
    T_side = np.asarray(rep["T_base_tool0"], float)
    print(f"[hold] sidecar vs DH tool0: {np.linalg.norm(T_side[:3, 3] - T_bt[:3, 3])*1e3:.3f} mm; "
          f"{rep['n_masked_held']} px held-masked")
    np.testing.assert_allclose(T_side, T_bt, atol=1e-4)     # FK + joint order
    hm = np.asarray(rep["held_mask"]).astype(bool).reshape(packet["depth"].shape)
    assert hm[vn, un] and not hm[vf, uf]                    # camera / base conventions
    held = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [probes[0][1], probes[1][1]])
    print(f"[hold] probe 4 cm behind the table: near {ref[0]*100:+.1f} -> {held[0]*100:+.1f} cm, "
          f"far {ref[1]*100:+.1f} -> {held[1]*100:+.1f} cm")
    assert held[0] > ref[0] + 0.03, "the held sphere's table point was integrated anyway"
    assert abs(held[1] - ref[1]) < 0.01, "the held mask reached a table point 20+ cm away"


# ------------------------------------------------------------ fit_spheres

@pytest.mark.parametrize("extents", [(0.0906, 0.1021, 0.1764),     # 10/23 teapot bbox
                                     (0.1174, 0.0927, 0.0912)])    # 10/23 mug bbox
def test_fit_spheres_box_sanity(curobo, extents):
    """A box is the worst shape for spheres (flat faces, corners): sanity
    only, and the numbers printed. The coverage bar is on real meshes."""
    mesh = trimesh.creation.box(extents)
    rep = curobo.ok({"cmd": "fit_spheres", "vertices": np.asarray(mesh.vertices, np.float32),
                     "faces": np.asarray(mesh.faces, np.int32), "num_spheres": 16},
                    timeout_ms=FIRST_CALL_TIMEOUT)
    S = np.asarray(rep["spheres"], float)
    assert rep["fit_type"] == "morphit" and S.ndim == 2 and S.shape[1] == 4
    assert rep["fitted_to"] == "mesh"                  # a box is watertight: no hull fallback
    assert 1 <= len(S) <= 16                           # fits the 16 attached_object slots
    assert np.all(S[:, 3] > 0)
    lo, hi = mesh.bounds
    assert np.all(S[:, :3] >= lo - S[:, 3:]) and np.all(S[:, :3] <= hi + S[:, 3:])
    mt = rep["metrics"]
    print(f"[fit_spheres] box {extents}: {len(S)} spheres, r {S[:, 3].min()*100:.1f}-"
          f"{S[:, 3].max()*100:.1f} cm, coverage {mt.get('coverage', float('nan')):.2f}, "
          f"protrusion {mt.get('protrusion', float('nan')):.2f}, "
          f"gap p95 {mt.get('surface_gap_p95', float('nan'))*100:.1f} cm, "
          f"max gap {mt.get('max_uncovered_gap', float('nan'))*100:.1f} cm")
