"""outputs/runs/canonicalize_asset.py -- summary -> R -> canonical asset.

Synthetic run dir: a tilted textured mug exported as the TRELLIS GLB, the
final mesh made from it the way Any6D does (bbox-centre + scale; isotropic
here so the mug stays a surface of revolution), and a summary.json with
pose_on_final / packet.plane / decision.oa_real. End-to-end tests need the
sim checkout (canonical_frame's refinement) and trimesh; the input
resolution tests need neither.
"""
import hashlib
import importlib.util
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
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "canonicalize_asset", REPO_ROOT / "outputs" / "runs" / "canonicalize_asset.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


drv = _load_driver()
STUB_PROV = {"root": "/opt/manip-sim", "commit": "stub", "dirty": False}


# ------------------------------------------------------------ input resolution (no sim)

def _summary(objects, plane=True):
    doc = {"objects": objects}
    if plane:
        doc["packet"] = {"plane": {"n": [0.0, -0.8, -0.6]}}
    return doc


def _reg(mesh="/nonexistent/final_mesh_mug.obj", **extra):
    return {"pose_on_final": {"cam_T_obj": np.eye(4).tolist(), "mesh": mesh}, **extra}


def test_key_is_a_typed_refusal_never_a_prompt():
    objs = {"m1": _reg(), "m2": _reg(), "m3": {"oriany": {}}}
    with pytest.raises(drv.CanonicalizeError, match=r"several registered objects \['m1', 'm2'\]"):
        drv.resolve_key(objs, None)
    assert drv.resolve_key(objs, "m2") == "m2"
    with pytest.raises(drv.CanonicalizeError, match="not in summary"):
        drv.resolve_key(objs, "m9")
    with pytest.raises(drv.CanonicalizeError, match="no pose_on_final"):
        drv.resolve_key(objs, "m3")
    with pytest.raises(drv.CanonicalizeError, match="nothing is registered"):
        drv.resolve_key({"m3": {"oriany": {}}}, None)
    assert drv.resolve_key({"m2": _reg(), "m3": {}}, None) == "m2"


def test_input_refusals_name_their_fix(tmp_path):
    final = tmp_path / "final_mesh_mug.obj"
    final.write_text("v 0 0 0\n")
    glb = tmp_path / "mug.glb"
    glb.write_bytes(b"x")
    with pytest.raises(drv.CanonicalizeError, match="packet.plane.n"):
        drv.resolve_inputs(_summary({"m2": _reg(str(final))}, plane=False), None, None, str(glb))
    # a --final-mesh run: no trellis2 block -> --canonical required
    with pytest.raises(drv.CanonicalizeError, match="--canonical"):
        drv.resolve_inputs(_summary({"m2": _reg(str(final))}), None, None, None)
    inp = drv.resolve_inputs(_summary({"m2": _reg(str(final))}), None, None, str(glb))
    assert inp["final"] == str(final) and inp["canonical"] == str(glb) and inp["oa_real"] is None
    # trellis2.glb is the default when present
    inp = drv.resolve_inputs(_summary({"m2": _reg(str(final), trellis2={"glb": str(glb)})}),
                             None, None, None)
    assert inp["canonical"] == str(glb)
    with pytest.raises(drv.CanonicalizeError, match="no such file"):
        drv.resolve_inputs(_summary({"m2": _reg()}), None, None, str(glb))
    bad = _reg(str(final))
    bad["pose_on_final"]["cam_T_obj"] = [[1, 0, 0]]
    with pytest.raises(drv.CanonicalizeError, match="4x4"):
        drv.resolve_inputs(_summary({"m2": bad}), None, None, str(glb))


def test_cli_refusal_exits_2(tmp_path, capsys):
    (tmp_path / "summary.json").write_text(json.dumps(_summary({"m1": _reg(), "m2": _reg()})))
    assert drv.main(["--run", str(tmp_path)]) == 2
    assert "pass --key" in capsys.readouterr().err


# ------------------------------------------------------------ end to end (sim + trimesh)

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    if not HAVE_SIM:
        pytest.skip(f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image
    import test_canonical_frame as tcf

    sys.path.insert(0, str(SIM_DIR))
    d = tmp_path_factory.mktemp("run")
    Vc, F = tcf.upright_mug()
    V = Vc @ tcf.TILT.T + np.array([0.21, -0.13, 0.08])         # TRELLIS-style tip, off-centre
    uv = np.stack([0.5 + 0.5 * np.tanh(5 * V[:, 0]), 0.5 + 0.5 * np.tanh(5 * V[:, 1])], 1)
    tex = Image.fromarray(np.tile(np.array([[200, 30, 30], [30, 30, 200]], np.uint8)[:, None],
                                  (16, 32, 1)).reshape(32, 32, 3))
    m = trimesh.Trimesh(V, F, process=False,
                        visual=trimesh.visual.TextureVisuals(uv=uv, image=tex))
    glb = d / "mug_offline.glb"
    m.export(glb)
    # Any6D's chain: load as any6d_server does, bbox-centre, scale, export OBJ untextured
    g = trimesh.load(str(glb), force="mesh")
    Vg = np.asarray(g.vertices, float)
    Vg = (Vg - 0.5 * (Vg.min(0) + Vg.max(0))) * 0.93
    fm = g.copy()
    fm.vertices = Vg
    fm.visual = trimesh.visual.ColorVisuals(fm)
    final = d / "final_mesh_mug.obj"
    fm.export(final)

    up_true, front_true = tcf.TILT[:, 2], tcf.TILT[:, 0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = tcf.CAM, [0.02, -0.05, 0.48]
    n_body = tcf._rot(np.cross(up_true, front_true), 2.0) @ up_true
    oa = {"front_cam": (tcf.CAM @ tcf._rot(up_true, 15.0) @ front_true).tolist(),
          "up_cam": (tcf.CAM @ up_true).tolist(), "alpha": 2}
    summary = {"packet": {"plane": {"n": (tcf.CAM @ n_body).tolist()}},
               "objects": {"m2": {"pose_on_final": {"cam_T_obj": T.tolist(), "mesh": str(final),
                                                    "decide": True},
                                  "decision": {"oa_real": oa}},
                           "m1": {"oriany": {}}}}
    (d / "summary.json").write_text(json.dumps(summary))
    yield d, glb, up_true, summary
    sys.path.remove(str(SIM_DIR))


def test_end_to_end(run_dir):
    d, glb, up_true, _ = run_dir
    doc = drv.canonicalize(d, canonical=glb, provenance=STUB_PROV)
    out = d / "assets_canon" / "mug"
    ra = json.loads((out / "render_asset.json").read_text())
    cj = json.loads((out / "canonical.json").read_text())
    R = np.asarray(cj["R_body_from_canon"])
    assert np.array_equal(R, np.asarray(ra["rotation_body_from_canon"]))   # single source of R
    assert cj == json.loads(json.dumps(doc))
    assert cj["up"]["source"] == "refined"
    ang = np.degrees(np.arccos(np.clip(R[:, 2] @ up_true, -1, 1)))
    assert ang < 1.0
    assert cj["inputs"]["key"] == "m2" and cj["inputs"]["canonical_glb"] == str(glb)
    assert cj["inputs"]["oa_real_present"] is True and cj["sim"] == STUB_PROV
    assert cj["asset"]["obj_sha256"] == ra["obj_sha256"] == \
        hashlib.sha256((out / "meshes" / "mug_visual.obj").read_bytes()).hexdigest()
    assert any("alpha 2" in w for w in cj["warnings"])
    assert not list((out / "meshes").glob("*_fp.obj"))


def test_rerun_reproduces_bytes(run_dir):
    """Deterministic end to end: the acceptance on real data compares this
    hash against the hand-built asset's."""
    d, glb, _, _ = run_dir
    a = drv.canonicalize(d, canonical=glb, out=d / "x", provenance=STUB_PROV)
    b = drv.canonicalize(d, canonical=glb, out=d / "y", provenance=STUB_PROV)
    assert a["asset"]["obj_sha256"] == b["asset"]["obj_sha256"]


def test_no_oa_real_warns_and_stays_upright(run_dir, tmp_path):
    d, glb, up_true, summary = run_dir
    s = json.loads(json.dumps(summary))
    del s["objects"]["m2"]["decision"]
    (tmp_path / "summary.json").write_text(json.dumps(s))
    doc = drv.canonicalize(tmp_path, canonical=glb, provenance=STUB_PROV)
    assert doc["azimuth"]["route"] == "none" and doc["azimuth"]["accepted"] is False
    assert any("no decision.oa_real" in w for w in doc["warnings"])
    R = np.asarray(doc["R_body_from_canon"])
    assert np.degrees(np.arccos(np.clip(R[:, 2] @ up_true, -1, 1))) < 1.0


def test_refuses_a_tracking_asset_dir(run_dir, capsys):
    """--out pointing at the tracked assets root: render_asset's kind guard
    refuses, and the CLI reports it as a refusal (exit 2)."""
    d, glb, _, _ = run_dir
    from manip_bridge.render_asset import build_render_asset
    final = d / "final_mesh_mug.obj"
    build_render_asset(final, glb, d / "assets" / "mug", "mug")
    rc = drv.main(["--run", str(d), "--canonical", str(glb), "--out", str(d / "assets")])
    assert rc == 2 and "tracking" in capsys.readouterr().err
    assert json.loads((d / "assets" / "mug" / "render_asset.json").read_text())[
        "rotation_body_from_canon"] is None


def test_cli_success(run_dir, capsys, monkeypatch):
    d, glb, _, _ = run_dir
    import manip_bridge.refine_compat as rc
    monkeypatch.setattr(rc, "provenance", lambda: STUB_PROV)
    assert drv.main(["--run", str(d), "--canonical", str(glb), "--out", str(d / "cli")]) == 0
    out = capsys.readouterr().out
    assert "up: refined" in out and "canonical.json written" in out and "WARNING" in out
