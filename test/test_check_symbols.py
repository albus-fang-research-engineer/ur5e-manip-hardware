"""Offline tests for outputs/runs/check_symbols.py -- no GPU, no sidecar.

Fixture: a subdivided cylinder (r=0.04, h=0.10) with a box "handle"
(x in [0.05, 0.07], thickness 0.02 along the viewing ray) attached toward
the camera, posed 0.5 m ahead with body +x along -z_cam (handle faces the
camera) and body +z along -y_cam (up in the optical frame). Mask and depth
are splatted from dense surface samples at the true pose, so the scene is
single-layer and non-occluded by construction -- matching the tool's stated
precondition.

Residual semantics under test (the third-draft bug): a correct CENTERLINE
symbol reads ~ +t/2 on BOTH residuals with (vs_measured - vs_mesh) ~ 0;
vs_mesh ~ 0 is what an ON-SURFACE symbol reads. A free-space symbol marked
depth_check=skip is not depth-gated at all.

Run from the repo root:  python -m pytest test/test_check_symbols.py -v
Host deps: pip install numpy scipy pillow trimesh
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
PIL_Image = pytest.importorskip("PIL.Image")

_MOD = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "check_symbols.py"
_spec = importlib.util.spec_from_file_location("check_symbols", _MOD)
cs = importlib.util.module_from_spec(_spec)
sys.modules["check_symbols"] = cs
_spec.loader.exec_module(cs)

W, H = 640, 480
K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
# body +x -> -z_cam (toward camera), +y -> +x_cam, +z (up) -> -y_cam
R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
T = np.eye(4)
T[:3, :3], T[:3, 3] = R, [0.0, 0.0, 0.5]
HANDLE_T = 0.02                       # handle thickness along the viewing ray


def _mesh():
    cyl = trimesh.creation.cylinder(radius=0.04, height=0.10, sections=64)
    box = trimesh.creation.box(extents=[HANDLE_T, 0.015, 0.05],
                               transform=trimesh.transformations.translation_matrix([0.06, 0, 0]))
    m = trimesh.util.concatenate([cyl, box])
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=0.002)
    return trimesh.Trimesh(v, f, process=False)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    rd = tmp_path_factory.mktemp("run")
    mesh = _mesh()
    mesh_path = rd / "final_mesh_test.obj"
    mesh.export(mesh_path)

    S = trimesh.sample.sample_surface(mesh, 400000, seed=0)[0]
    S_cam = S @ R.T + T[:3, 3]
    zbuf = cs.vertex_zbuffer(S_cam, K, H, W)          # splat: nearest sample wins
    mask = np.isfinite(zbuf)
    depth = np.where(mask, zbuf, 0.0)

    PIL_Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(rd / "rgb.png")
    PIL_Image.fromarray((mask * 255).astype(np.uint8)).save(rd / "mask_m2.png")
    PIL_Image.fromarray(np.round(depth * 1e3).astype(np.uint16)).save(rd / "depth_mm.png")

    # table through the cylinder base (body z=-0.05): n=[0,-1,0] (up), d=0.05>0
    summary = {"frame": {"K": K.tolist()},
               "packet": {"plane": {"n": [0.0, -1.0, 0.0], "d": 0.05}},
               "objects": {"m2": {"pose_on_final": {"cam_T_obj": T.tolist(),
                                                    "mesh": str(mesh_path)}}}}
    (rd / "summary.json").write_text(json.dumps(summary))
    return {"rd": rd, "mesh": mesh, "mask": mask, "depth": depth}


def _frames(points=None, axes=None, quantities=None):
    return {"object": "test", "points": points or {}, "axes": axes or {},
            "quantities": quantities or {}}


def _eval(scene, frames, plane={"n": [0.0, -1.0, 0.0]}, **kw):
    rec, _ = cs.evaluate(frames, T, np.asarray(scene["mesh"].vertices),
                         scene["mask"], scene["depth"], K, plane, **kw)
    return rec


def test_surface_point(scene):
    # on the cylinder wall, offset up so the handle is not in its ray
    rec = _eval(scene, _frames({"front_wall": {"xyz": [0.04, 0, 0.035]}}))
    r = rec["points"]["front_wall"]
    assert r["status"] == "pass" and r["depth_check"] == "surface"
    assert abs(r["vs_mesh_mm"]) < 3 and abs(r["vs_measured_mm"]) < 3
    # hand-computed pixel, independent of the module's projection (the depth
    # image is splatted through vertex_zbuffer, so this breaks the loop):
    # X_cam = 0.04*[0,0,-1] + 0.035*[0,-1,0] + [0,0,0.5] = [0, -0.035, 0.46]
    # u = 600*0/0.46 + 320 = 320;  v = 240 - 600*0.035/0.46 = 194.34783
    assert np.allclose(r["px"], [320.0, 240.0 - 600 * 0.035 / 0.46], atol=1e-3)


def test_interior_centerline(scene):
    # handle centerline: +t/2 on BOTH residuals, difference ~ 0 (the
    # corrected semantics -- vs_mesh ~ 0 here was the inverted assertion)
    rec = _eval(scene, _frames({"handle_center":
                                {"xyz": [0.06, 0, 0], "depth_check": "interior"}}))
    r = rec["points"]["handle_center"]
    assert r["status"] == "pass"
    assert abs(r["vs_mesh_mm"] - 1e3 * HANDLE_T / 2) < 3
    assert abs(r["vs_measured_mm"] - 1e3 * HANDLE_T / 2) < 3
    assert abs(r["diff_mm"]) < 3
    assert 0 <= r["vs_mesh_mm"] <= 1e3 * cs.values()["interior_hi_m"]


def test_punched_through_fails(scene):
    # body-axis center: 70 mm behind the handle face along its ray -- outside
    # the interior band: a grounding bug, not a fidelity question
    rec = _eval(scene, _frames({"body_center":
                                {"xyz": [0.0, 0, 0], "depth_check": "interior"}}))
    r = rec["points"]["body_center"]
    assert r["status"] == "fail" and "vs_mesh_band" in r["fail_reasons"]
    assert r["vs_mesh_mm"] > 1e3 * cs.values()["interior_hi_m"]


def test_skip_free_space(scene):
    # floating 15 mm in front of the handle face: meaningless residuals,
    # gated on in-mask placement only
    rec = _eval(scene, _frames({"opening_like":
                                {"xyz": [0.085, 0, 0], "depth_check": "skip"}}))
    r = rec["points"]["opening_like"]
    assert r["status"] == "pass(skip)" and r["in_mask"]
    assert r["vs_measured_mm"] < -8            # recorded, shows why skip exists
    # the same symbol under the default gate must fail
    rec2 = _eval(scene, _frames({"opening_like": {"xyz": [0.085, 0, 0]}}))
    assert rec2["points"]["opening_like"]["status"] == "fail"


def test_stray_point_not_in_mask(scene):
    rec = _eval(scene, _frames({"stray": {"xyz": [0.0, 0.15, 0.0]}}))
    assert rec["points"]["stray"]["status"] == "fail"
    assert "not_in_mask" in rec["points"]["stray"]["fail_reasons"]


def test_up_axis_referent(scene):
    fr = _frames({"opening_center": {"xyz": [0, 0, 0.05], "depth_check": "skip"}},
                 axes={"up_axis": {"xyz": [0, 0, 1]}})
    rec = _eval(scene, fr)
    a = rec["axes"]["up_axis"]
    assert a["status"] == "pass" and a["angle_deg"] < 0.5
    # tilt the fitted plane 10 deg: outside the 5 deg gate
    th = np.radians(10)
    rec2 = _eval(scene, fr, plane={"n": [np.sin(th), -np.cos(th), 0.0]})
    assert rec2["axes"]["up_axis"]["status"] == "fail"
    # no plane in the packet: report-only, never a silent pass/fail
    rec3 = _eval(scene, fr, plane=None)
    assert rec3["axes"]["up_axis"]["status"] == "report_only"


def test_front_class_axis_report_only(scene):
    rec = _eval(scene, _frames(axes={"handle_axis": {"xyz": [1, 0, 0]}}))
    a = rec["axes"]["handle_axis"]
    assert a["status"] == "report_only" and a["referent"] is None
    assert rec["verdict"]["pass"]              # report-only never fails a run


def test_depth_hole_not_a_failure(scene):
    # zero the depth around the surface point: vs_measured ungated, vs_mesh
    # still checks placement -- a sensor hole is not a grounding failure
    depth = scene["depth"].copy()
    px, _ = cs.project(K, (R @ [0.04, 0, 0.035]) + T[:3, 3])
    u, v = int(round(px[0, 0])), int(round(px[0, 1]))
    depth[v - 4:v + 5, u - 4:u + 5] = 0.0
    rec, _ = cs.evaluate(_frames({"front_wall": {"xyz": [0.04, 0, 0.035]}}),
                         T, np.asarray(scene["mesh"].vertices),
                         scene["mask"], depth, K, {"n": [0, -1.0, 0]})
    r = rec["points"]["front_wall"]
    assert r["status"] == "pass" and r["vs_measured_mm"] is None
    assert r["n_depth_px"] == 0 and "note" in r


def test_main_verdicts_and_outputs(scene):
    rd = scene["rd"]
    good = rd / "good_frames.json"
    good.write_text(json.dumps(_frames(
        {"front_wall": {"xyz": [0.04, 0, 0.035]},
         "handle_center": {"xyz": [0.06, 0, 0], "depth_check": "interior"},
         "opening_center": {"xyz": [0, 0, 0.05], "depth_check": "skip"}},
        axes={"up_axis": {"xyz": [0, 0, 1]}},
        quantities={"rim_radius": {"value": 0.04}})))
    bad = rd / "bad_frames.json"
    bad.write_text(json.dumps(_frames(
        {"body_center": {"xyz": [0.0, 0, 0], "depth_check": "interior"}})))

    assert cs.main([str(rd), "--frames", str(good), "--key", "m2",
                    "--out-prefix", str(rd / "symbols_good")]) == 0
    assert cs.main([str(rd), "--frames", str(bad), "--key", "m2",
                    "--out-prefix", str(rd / "symbols_bad")]) == 1
    rec = json.loads((rd / "symbols_good.json").read_text())
    assert rec["verdict"]["pass"] and (rd / "symbols_good.png").exists()
    assert rec["params"]["surface_tol_m"]["basis"]      # thresholds travel with the record
    assert json.loads((rd / "symbols_bad.json").read_text())["verdict"]["pass"] is False


def test_depth_check_flag_for_fieldless_output(scene):
    # grounding's symbols_from_parts emits no depth_check field: a correct
    # free-space symbol fails under the surface default and passes once the
    # flag supplies the mode; a schema field wins over the flag.
    rd = scene["rd"]
    fieldless = rd / "fieldless_frames.json"
    fieldless.write_text(json.dumps(_frames(
        {"opening_like": {"xyz": [0.085, 0, 0]}})))          # no depth_check
    base = [str(rd), "--frames", str(fieldless), "--key", "m2"]
    assert cs.main(base + ["--out-prefix", str(rd / "s_nofield")]) == 1
    assert cs.main(base + ["--depth-check", "opening_like=skip",
                           "--out-prefix", str(rd / "s_flag")]) == 0
    rec = json.loads((rd / "s_flag.json").read_text())
    assert rec["points"]["opening_like"]["depth_check"] == "skip"
    carried = rd / "carried_frames.json"                     # schema field present
    carried.write_text(json.dumps(_frames(
        {"opening_like": {"xyz": [0.085, 0, 0], "depth_check": "skip"}})))
    assert cs.main([str(rd), "--frames", str(carried), "--key", "m2",
                    "--depth-check", "opening_like=interior",
                    "--out-prefix", str(rd / "s_schema")]) == 0
    rec = json.loads((rd / "s_schema.json").read_text())
    assert rec["points"]["opening_like"]["depth_check"] == "skip"
    with pytest.raises(SystemExit):                          # typo'd mode refused
        cs.main(base + ["--depth-check", "opening_like=off"])
