"""cuRobo sidecar `freeze` and `fit_spheres`, over the ZMQ wire (the commands
scene_ops / grasp_plan use to take the grasped object out of the map and
carry it as spheres). Same packet + sidecar as test_curobo_esdf:

    docker compose up -d curobo
    python -m pytest test/test_curobo_scene_ops.py -v -s

freeze: clear a region, freeze, integrate the same depth again -> the ESDF
is bit-identical to the cleared one (integrate changed nothing); unfreeze,
integrate -> the surface is back. The sidecar is unfrozen on teardown even
if a check fails, so a long-running sidecar is never left blind.
"""
import numpy as np
import pytest

from test_curobo_esdf import (CALL_TIMEOUT, FIRST_CALL_TIMEOUT, _esdf_lookup,  # noqa: F401
                              _integrate, _surface_probe_points, _T_map_cam, mapper)

trimesh = pytest.importorskip("trimesh")


@pytest.fixture
def unfreeze_after(mapper):
    yield mapper
    mapper.ok({"cmd": "freeze", "on": False}, timeout_ms=CALL_TIMEOUT)


def test_freeze_keeps_a_cleared_region_clear(unfreeze_after, packet):
    m = unfreeze_after
    m.ok({"cmd": "freeze", "on": False}, timeout_ms=CALL_TIMEOUT)
    m.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(m, packet, FIRST_CALL_TIMEOUT)
    p_free, p_occ = _surface_probe_points(packet)
    d_occ_live = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [p_occ])[0]
    assert d_occ_live < 0.03                          # the surface is there

    lo, hi = p_occ - 0.06, p_occ + 0.06                # a 12 cm box around the probe
    assert m.ok({"cmd": "freeze", "on": True}, timeout_ms=CALL_TIMEOUT)["frozen"] is True
    assert m.ok({"cmd": "clear_region", "min": lo.tolist(), "max": hi.tolist()},
                timeout_ms=CALL_TIMEOUT)["n_cleared"] > 0
    cleared = m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    d_occ_cleared = _esdf_lookup(cleared, [p_occ])[0]
    print(f"[freeze] probe 4 cm behind the table: live {d_occ_live*100:+.1f} cm, "
          f"cleared {d_occ_cleared*100:+.1f} cm")
    assert d_occ_cleared > d_occ_live + 0.02          # the clear removed it

    # the bridge keeps integrating at 2 Hz: while frozen that changes nothing
    rep = m.ok({"cmd": "integrate", "depth": packet["depth"].astype(np.float32),
                "intrinsics": packet["K"].astype(np.float32),
                "pose": _T_map_cam(packet)},
               timeout_ms=CALL_TIMEOUT)
    assert rep.get("frozen") is True
    after = m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    np.testing.assert_array_equal(np.asarray(after["esdf"]), np.asarray(cleared["esdf"]))

    # unfrozen, the next integrate brings the surface back
    assert m.ok({"cmd": "freeze", "on": False}, timeout_ms=CALL_TIMEOUT)["frozen"] is False
    _integrate(m, packet, CALL_TIMEOUT)
    d_back = _esdf_lookup(m.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT), [p_occ])[0]
    assert d_back < 0.03, f"surface not re-integrated after unfreeze ({d_back:.3f} m)"


@pytest.mark.parametrize("extents", [(0.0906, 0.1021, 0.1764),     # 10/23 teapot bbox
                                     (0.1174, 0.0927, 0.0912)])    # 10/23 mug bbox
def test_fit_spheres_wraps_an_object_sized_box(curobo, extents):
    mesh = trimesh.creation.box(extents)
    rep = curobo.ok({"cmd": "fit_spheres", "vertices": np.asarray(mesh.vertices, np.float32),
                     "faces": np.asarray(mesh.faces, np.int32), "num_spheres": 16},
                    timeout_ms=FIRST_CALL_TIMEOUT)
    S = np.asarray(rep["spheres"], float)
    assert rep["fit_type"] == "voxel" and S.ndim == 2 and S.shape[1] == 4
    assert 1 <= len(S) <= 16                           # fits the 16 attached_object slots
    assert np.all(S[:, 3] > 0)
    lo, hi = mesh.bounds
    assert np.all(S[:, :3] >= lo - S[:, 3:]) and np.all(S[:, :3] <= hi + S[:, 3:])
    pts = mesh.sample(2000)
    d = (np.linalg.norm(pts[:, None] - S[None, :, :3], axis=-1) - S[None, :, 3]).min(axis=1)
    frac = float((d < 0.01).mean())
    print(f"[fit_spheres] box {extents}: {len(S)} spheres, r {S[:, 3].min()*100:.1f}-"
          f"{S[:, 3].max()*100:.1f} cm, coverage@10mm {frac:.2f}")
    assert frac > 0.85
