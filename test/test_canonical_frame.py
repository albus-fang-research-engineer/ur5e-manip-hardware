"""canonical_frame (+ refine_compat, its sim import site).

Offline: a synthetic tapered mug with a handle, built upright, then tilted
into an arbitrary body frame the way TRELLIS leaves real meshes, and posed
under a random cam_T_obj. The sim checkout is found at $SIM_DIR or the
sibling ../ur5e-manip-sim (refine/refine_frame are numpy/scipy only); the
tests skip without it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "manip_sim" / "refine_frame.py").is_file()

sys.path.insert(0, str(BRIDGE))


@pytest.fixture(scope="module")
def cf():
    if not HAVE_SIM:
        pytest.skip(f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
    sys.path.insert(0, str(SIM_DIR))
    try:
        import manip_bridge.canonical_frame as mod
        yield mod
    finally:
        sys.path.remove(str(SIM_DIR))


# ------------------------------------------------------------- fixtures

def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _grid(nu, nv, closed_u=True):
    F = []
    for i in range(nv - 1):
        for j in range(nu if closed_u else nu - 1):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            c, d = (i + 1) * nu + j, (i + 1) * nu + (j + 1) % nu
            F += [[a, b, d], [a, d, c]]
    return np.asarray(F)


def upright_mug():
    """Canonical frame by construction: z up, base at z=0, handle on -x
    (Orient Anything's mug front is away from the handle -> front = +x).
    Tapered open-top body, base disc, tube handle."""
    nu, nv = 64, 24
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    h = np.linspace(0.0, 0.095, nv)
    r = 0.040 + 0.08 * h                                   # 40 -> ~47.6 mm
    body = np.stack([np.outer(r, np.cos(th)), np.outer(r, np.sin(th)),
                     np.repeat(h[:, None], nu, 1)], -1).reshape(-1, 3)
    Fb = _grid(nu, nv)
    # base disc: fan from a centre vertex to the bottom ring (indices 0..nu-1)
    c0 = len(body)
    base = np.array([[0.0, 0.0, 0.0]])
    Fd = np.array([[c0, (j + 1) % nu, j] for j in range(nu)])
    # handle: tube of radius 6 mm swept along a half-circle in the xz plane at -x
    ns, nr = 32, 12
    s = np.linspace(-np.pi / 2, np.pi / 2, ns)
    centre = np.stack([-0.042 - 0.022 * np.cos(s), np.zeros(ns), 0.048 + 0.028 * np.sin(s)], -1)
    tang = np.gradient(centre, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    ey = np.array([0.0, 1.0, 0.0])
    nrm = np.cross(tang, ey)
    ring = np.linspace(0, 2 * np.pi, nr, endpoint=False)
    tube = (centre[:, None, :] + 0.006 * (np.cos(ring)[None, :, None] * nrm[:, None, :]
                                          + np.sin(ring)[None, :, None] * ey)).reshape(-1, 3)
    Ft = _grid(nr, ns) + c0 + 1
    V = np.vstack([body, base, tube])
    V = V - 0.5 * (V.min(0) + V.max(0))                    # bbox-centred, like final_mesh
    return V, np.vstack([Fb, Fd, Ft])


TILT = _rot([0.3, -1.0, 0.2], 67.0)       # canon -> body: the TRELLIS-style 67 deg tip
CAM = _rot([1.0, 0.4, -0.3], 128.0)       # body -> camera rotation of the registration


def scene(tilt=TILT, table_err_deg=2.0, oa_yaw_err_deg=15.0, oa_up=None, alpha=2):
    Vc, F = upright_mug()
    V = Vc @ tilt.T
    up_true, front_true = tilt[:, 2], tilt[:, 0]
    T = np.eye(4)
    T[:3, :3] = CAM
    T[:3, 3] = [0.02, -0.05, 0.48]
    # table normal measured with a small tilt error, expressed in the camera frame
    n_body = _rot(np.cross(up_true, front_true), table_err_deg) @ up_true
    oa_front_body = _rot(up_true, oa_yaw_err_deg) @ front_true
    oa = {"front_cam": (CAM @ oa_front_body).tolist(),
          "up_cam": (CAM @ (up_true if oa_up is None else oa_up)).tolist(),
          "alpha": alpha}
    return V, F, T, CAM @ n_body, oa, up_true, front_true


def ang(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


# ------------------------------------------------------------------ tests

def test_compat_contract(cf):
    import manip_bridge.refine_compat as rc
    assert set(rc.__all__) == {"MAX_SNAP_DEG", "SEMANTIC_SIGMA_DEG", "RefineResult", "AzimuthResult",
                               "FrameResult", "refine_axis", "azimuth_from_semantic",
                               "assemble_frame", "provenance", "SOURCES"}
    assert rc.SOURCES[0].endswith(os.path.join("manip_sim", "refine.py"))
    assert rc.SOURCES[1].endswith(os.path.join("manip_sim", "refine_frame.py"))


def test_recovers_tilted_mug(cf):
    """Refined up beats the 2 deg-off table normal; front carries OA's 15 deg
    yaw error (semantic route: projection removes tilt, not yaw); the
    canonical copy of the mesh is upright; the record is JSON."""
    V, F, T, n_cam, oa, up_true, front_true = scene()
    R, rec = cf.canonical_frame(V, F, T, n_cam, oa)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    assert rec["up"]["source"] == "refined"
    assert ang(R[:, 2], up_true) < 1.0, rec["up"]
    assert 1.0 < rec["up"]["angle_to_table_deg"] < 3.0
    assert abs(ang(R[:, 0], front_true) - 15.0) < 2.0
    assert rec["azimuth"] == {**rec["azimuth"], "route": "semantic", "accepted": True,
                              "alpha_reported": 2}
    assert rec["oa_up_angle_deg"] < 1.5
    # canonical copy: its own up is +z, so the sim's UP_BODY holds by construction
    Vc = cf.to_canonical(V, R)
    assert ang(cf.to_canonical(up_true[None], R)[0], [0, 0, 1]) < 1.0
    # handle sits on -x in the canonical copy (front = away from the handle)
    n_body, n_tube = 64 * 24, 32 * 12                      # upright_mug's vertex blocks
    axis_xy = Vc[:n_body, :2].mean(0)
    assert (Vc[-n_tube:, 0] - axis_xy[0]).mean() < -0.03
    json.dumps(rec)


def test_map_back_round_trips(cf):
    V, F, T, n_cam, oa, *_ = scene()
    R, _ = cf.canonical_frame(V, F, T, n_cam, oa)
    X = np.random.default_rng(1).normal(size=(50, 3))
    assert np.allclose(cf.to_body(cf.to_canonical(X, R), R), X, atol=1e-12)
    # column convention: canonical +x maps to R[:, 0] in the body frame
    assert np.allclose(cf.to_body(np.eye(3), R), R.T)
    assert np.allclose(cf.to_body([[1.0, 0, 0]], R)[0], R[:, 0])


def test_refine_beyond_snap_keeps_table_normal(cf):
    """Table normal 8 deg off the true axis: the fit lands > 5 deg away, is
    rejected, and up is the table normal itself -- the frame never drifts
    further from the table than check_symbols tolerates."""
    V, F, T, n_cam, oa, up_true, _ = scene(table_err_deg=8.0)
    R, rec = cf.canonical_frame(V, F, T, n_cam, oa)
    assert rec["up"]["source"] == "table_normal"
    assert rec["up"]["angle_to_table_deg"] < 1e-6
    assert rec["up"]["fit_note"]
    assert abs(ang(R[:, 2], up_true) - 8.0) < 0.5


def test_no_front_still_upright(cf):
    V, F, T, n_cam, oa, up_true, _ = scene()
    for reading, why in ((None, "no Orient Anything"), ({**oa, "alpha": 0}, "alpha 0")):
        R, rec = cf.canonical_frame(V, F, T, n_cam, reading)
        assert rec["azimuth"]["accepted"] is False and rec["azimuth"]["route"] == "none"
        assert why in rec["azimuth"]["note"]
        assert "arbitrary" in rec["note"]
        assert ang(R[:, 2], up_true) < 1.0
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9


def test_front_along_up_is_rejected_not_trusted(cf):
    """An OA front within 20 deg of up has no usable azimuth: the semantic
    route rejects on conditioning and x falls back, flagged."""
    V, F, T, n_cam, oa, up_true, front_true = scene()
    near_up = _rot(np.cross(up_true, front_true), 10.0) @ up_true
    R, rec = cf.canonical_frame(V, F, T, n_cam, {**oa, "front_cam": (CAM @ near_up).tolist()})
    assert rec["azimuth"]["route"] == "semantic" and rec["azimuth"]["accepted"] is False
    assert "conditioning" in rec["azimuth"]["note"]


def test_oa_up_is_diagnostic_only(cf):
    """OA calling a tumbled object upright (its up 90 deg off) is recorded,
    and changes nothing: up comes from the table."""
    V, F, T, n_cam, oa, up_true, front_true = scene(oa_up=TILT[:, 0])
    R, rec = cf.canonical_frame(V, F, T, n_cam, oa)
    ref, _ = cf.canonical_frame(V, F, T, n_cam, scene()[4])
    assert abs(rec["oa_up_angle_deg"] - 90.0) < 2.0
    assert np.allclose(R, ref, atol=1e-12)


def test_algebra_imports_without_sim():
    """render_asset writes the canonical copy with to_canonical; that must not
    need the sim mount (render_asset is sim-free)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); "
            "from manip_bridge.canonical_frame import to_canonical, to_body; "
            "import numpy as np; R = np.eye(3)[[1, 2, 0]].T; "
            "assert np.allclose(to_body(to_canonical(np.ones((2, 3)), R), R), 1); "
            "assert not any(m.startswith('manip_sim') for m in sys.modules)") % str(BRIDGE)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd="/")


def test_deterministic(cf):
    V, F, T, n_cam, oa, *_ = scene()
    R1, r1 = cf.canonical_frame(V, F, T, n_cam, oa)
    R2, r2 = cf.canonical_frame(V, F, T, n_cam, oa)
    assert np.array_equal(R1, R2) and r1 == r2


def test_rejects_improper_pose(cf):
    V, F, T, n_cam, oa, *_ = scene()
    T[:3, 0] *= -1
    with pytest.raises(ValueError, match="not proper"):
        cf.canonical_frame(V, F, T, n_cam, oa)
