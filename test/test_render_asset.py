"""Offline tests for manip_bridge.render_asset -- no ROS, no sidecars.

Fixture: a synthetic textured mesh (icosphere, spherical UVs, checkerboard
texture) written as the "canonical" GLB, DELIBERATELY OFF-CENTRE, plus a
"final_mesh" produced by replaying Any6D's chain on it: load the way
any6d_server does (trimesh.load force="mesh"), bbox-centre, scale each
axis by its own factor about the origin (register_any6d: coarse isotropic,
refinement per-axis, 252-sample rescale -- all diagonal, nothing rotates),
export as OBJ through trimesh like `est.mesh.export(...)`. The builder must
reproduce final_mesh's vertices verbatim, carry the texture over by index,
recover the per-axis scale and the centre, and refuse anything that breaks
the correspondence or the centred-frame condition.

The render test needs a MuJoCo GL backend: MUJOCO_GL=egl on the
workstation, osmesa on a GPU-less box (`apt install libosmesa6`). It
skips, not fails, when neither initialises.

Run from the repo root:  python -m pytest test/test_render_asset.py -v
Host deps: pip install trimesh scipy pillow mujoco
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "manip_bridge"))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
PIL_Image = pytest.importorskip("PIL.Image")

from manip_bridge.render_asset import (                       # noqa: E402
    BODY_NAME, RenderAssetError, build_render_asset, from_summary, read_obj,
    render_check,
)

SCALE = np.array([0.0922, 0.0664, 0.1571])       # per-axis, mug-run ratios x 0.1
OFFSET = np.array([0.31, -0.22, 0.14])            # canonical-frame offset


def _checker(px=256, tile=32):
    yy, xx = np.mgrid[0:px, 0:px]
    on = ((yy // tile + xx // tile) % 2).astype(bool)
    img = np.zeros((px, px, 3), np.uint8)
    img[on] = (230, 40, 40)
    img[~on] = (40, 60, 230)
    return PIL_Image.fromarray(img)


def _textured_sphere() -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    m.apply_translation(OFFSET)
    p = m.vertices - OFFSET
    u = 0.5 + np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi)
    v = 0.5 + np.arcsin(np.clip(p[:, 2] / 0.45, -1, 1)) / np.pi
    m.visual = trimesh.visual.TextureVisuals(uv=np.stack([u, v], 1), image=_checker())
    return m


def _any6d_chain(canonical_glb: Path, out: Path, scale=SCALE, shift=None,
                 textured_export=False) -> Path:
    """What register_any6d does to the vertices, then `est.mesh.export`."""
    m = trimesh.load(str(canonical_glb), force="mesh")        # any6d_server's load
    V = np.asarray(m.vertices, np.float64)
    V = V - 0.5 * (V.min(0) + V.max(0))                       # reset_object: bbox-centre
    V = V * scale                                             # per-axis, about origin
    if shift is not None:
        V = V + shift
    mesh = m.copy()
    mesh.vertices = V
    if not textured_export:
        mesh.visual = trimesh.visual.ColorVisuals(mesh)       # the mug run: "visual vertex"
    mesh.export(out)
    return out


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("perception")
    canonical = d / "mug.glb"
    _textured_sphere().export(canonical)
    final = _any6d_chain(canonical, d / "final_mesh_mug.obj")
    return final, canonical


@pytest.fixture(scope="module")
def asset(pair, tmp_path_factory):
    final, canonical = pair
    out = tmp_path_factory.mktemp("assets") / "mug"
    rec = build_render_asset(final, canonical, out, "mug", require_texture=True)
    return out, rec


# ------------------------------------------------------------------ identity

def test_vertices_are_final_mesh_verbatim(asset, pair):
    out, rec = asset
    V, F = read_obj(Path(rec.obj))
    Vf, Ff = read_obj(pair[0])
    assert np.array_equal(V, Vf), "asset vertices must be final_mesh's, bit for bit"
    assert np.array_equal(F, Ff)
    assert rec.n_vertices == len(Vf) and rec.n_faces == len(Ff)


def test_fit_recovers_scale_and_centre(asset):
    out, rec = asset
    np.testing.assert_allclose(rec.scale_xyz, SCALE, rtol=1e-6)
    np.testing.assert_allclose(rec.centre_xyz, OFFSET, atol=1e-6)     # sphere: bbox centre = OFFSET
    assert rec.fit_residual_m <= 1e-6
    assert np.abs(rec.bbox_centre_m).max() <= 1e-6


def test_textured_any6d_export_also_accepted(pair, tmp_path):
    """A final_mesh whose OBJ carries vt/mtllib (textured input survived
    Any6D) parses to the same vertices; the texture still comes from the
    canonical GLB, not from Any6D's shared material.png."""
    final, canonical = pair
    final_tex = _any6d_chain(canonical, tmp_path / "final_mesh_mug.obj", textured_export=True)
    assert "mtllib" in (tmp_path / "final_mesh_mug.obj").read_text()[:200]
    rec = build_render_asset(final_tex, canonical, tmp_path / "mug", "mug", require_texture=True)
    Va, _ = read_obj(Path(rec.obj))
    Vf, _ = read_obj(final_tex)
    assert np.array_equal(Va, Vf)
    assert Path(rec.texture).name == "mug_texture.png"


# ------------------------------------------------------------------ refusals

def test_wrong_reconstruction_refused(pair, tmp_path):
    """A different TRELLIS run (other vertex count) cannot pair with this final_mesh."""
    final, _ = pair
    other = trimesh.creation.icosphere(subdivisions=2)
    other.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(other.vertices), 2),
                                                 image=_checker())
    other.export(tmp_path / "other.glb")
    with pytest.raises(RenderAssetError, match="vertices"):
        build_render_asset(final, tmp_path / "other.glb", tmp_path / "x", "x")


def test_scrambled_order_refused(pair, tmp_path):
    """Same counts and faces, vertices permuted: the affine fit must fail."""
    final, canonical = pair
    V, F = read_obj(final)
    perm = np.random.default_rng(0).permutation(len(V))
    from manip_bridge.render_asset import write_obj
    write_obj(tmp_path / "scrambled.obj", V[perm], F, None)
    with pytest.raises(RenderAssetError, match="residual"):
        build_render_asset(tmp_path / "scrambled.obj", canonical, tmp_path / "x", "x")


def test_faces_mismatch_refused(pair, tmp_path):
    final, canonical = pair
    V, F = read_obj(final)
    from manip_bridge.render_asset import write_obj
    write_obj(tmp_path / "refaced.obj", V, F[::-1], None)
    with pytest.raises(RenderAssetError, match="[Ff]ace"):
        build_render_asset(tmp_path / "refaced.obj", canonical, tmp_path / "x", "x")


def test_uncentred_final_refused(pair, tmp_path):
    """If final_mesh were not bbox-centred, Any6D's pose compensation would
    not be the identity and the pose would not be in this file's frame."""
    _, canonical = pair
    shifted = _any6d_chain(canonical, tmp_path / "shifted.obj", shift=np.array([0.005, 0, 0]))
    with pytest.raises(RenderAssetError, match="centre"):
        build_render_asset(shifted, canonical, tmp_path / "x", "x")


def test_flipped_axis_refused(pair, tmp_path):
    _, canonical = pair
    flipped = _any6d_chain(canonical, tmp_path / "flipped.obj", scale=SCALE * [1, -1, 1])
    with pytest.raises(RenderAssetError, match="positive"):
        build_render_asset(flipped, canonical, tmp_path / "x", "x")


# ------------------------------------------------------------------ texture

def test_texture_written_and_wired(asset):
    out, rec = asset
    assert rec.texture and Path(rec.texture).is_file() and Path(rec.texture).is_absolute()
    obj_text = Path(rec.obj).read_text()
    n_vt = sum(1 for l in obj_text.splitlines() if l.startswith("vt "))
    assert n_vt == rec.n_vertices, "one vt per vertex"
    assert "/" in next(l for l in obj_text.splitlines() if l.startswith("f "))
    assert "mtllib" not in obj_text
    root = ET.parse(rec.xml).getroot()
    tex = root.find("asset/texture")
    mat = root.find("asset/material")
    geom = root.find(f"worldbody/body/body[@name='{BODY_NAME}']/geom")
    assert tex is not None and tex.get("file") == rec.texture and tex.get("type") == "2d"
    assert mat is not None and mat.get("texture") == tex.get("name")
    assert geom is not None and geom.get("material") == mat.get("name")
    assert geom.get("group") == "1" and geom.get("contype") == "0"
    assert root.find("asset/mesh").get("file") == "meshes/mug_visual.obj"
    assert not [m for m in root.findall("asset/mesh") if "_col_" in m.get("name", "")]


def test_untextured_canonical_is_flagged(tmp_path):
    plain = trimesh.creation.icosphere(subdivisions=2)
    plain.apply_translation([0.2, 0, 0])
    plain.export(tmp_path / "plain.glb")
    final = _any6d_chain(tmp_path / "plain.glb", tmp_path / "final_mesh_p.obj")
    with pytest.raises(RenderAssetError, match="texture"):
        build_render_asset(final, tmp_path / "plain.glb", tmp_path / "p", "p", require_texture=True)
    rec = build_render_asset(final, tmp_path / "plain.glb", tmp_path / "p", "p")
    assert rec.texture is None
    assert ET.parse(rec.xml).getroot().find("asset/texture") is None


def test_fp_obj_is_textured_identity_copy(asset):
    """trimesh (as pose_server loads it) must see a TextureVisuals with an
    image, and the vertices must be final_mesh's -- FoundationPose registering
    on <name>_fp.obj is registering on the tracked geometry."""
    out, rec = asset
    assert rec.fp_obj and Path(rec.fp_obj).is_file() and Path(rec.fp_obj).with_suffix(".mtl").is_file()
    m = trimesh.load(rec.fp_obj, force="mesh")
    assert m.visual.kind == "texture"
    img = getattr(m.visual.material, "image", None) or getattr(m.visual.material, "baseColorTexture", None)
    assert img is not None
    Va, _ = read_obj(Path(rec.obj))
    assert np.array_equal(np.asarray(m.vertices, np.float64).round(9), Va.round(9)) or \
        np.allclose(np.sort(np.asarray(m.vertices), axis=0), np.sort(Va, axis=0), atol=1e-9)
    Vf, _ = read_obj(Path(rec.fp_obj))
    assert np.array_equal(Vf, Va)


# ------------------------------------------------------------------ summary

def test_from_summary_reads_run_scene_record(pair, tmp_path):
    final, canonical = pair
    doc = {"objects": {"mug": {"trellis2": {"glb": str(canonical)},
                               "any6d": {"mesh": str(final), "source": "trellis"}}}}
    (tmp_path / "summary.json").write_text(json.dumps(doc))
    f, c = from_summary(tmp_path / "summary.json", "mug")
    assert f == final and c == canonical
    doc["objects"]["mug"]["any6d"]["source"] = "img_to_3d"
    (tmp_path / "summary.json").write_text(json.dumps(doc))
    with pytest.raises(RenderAssetError, match="img_to_3d"):
        from_summary(tmp_path / "summary.json", "mug")


# ------------------------------------------------------------------ render

def _gl_ok():
    mujoco = pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(m, 16, 16).close()
        return True
    except Exception:
        if os.environ["MUJOCO_GL"] != "osmesa":
            os.environ["MUJOCO_GL"] = "osmesa"
            return _gl_ok()
        return False


def test_render_shows_texture(asset):
    """Loaded the way render_candidates.build_model loads it (from_xml_string
    + injected meshdir); the object's pixels must carry the checkerboard,
    not one flat shade. This is the assertion that catches MuJoCo silently
    ignoring the texture (no material on the geom, MTL-only, bad path)."""
    if not _gl_ok():
        pytest.skip("no MuJoCo GL backend (set MUJOCO_GL=egl|osmesa; osmesa needs libosmesa6)")
    out, rec = asset
    img = render_check(out, "mug", px=320).astype(np.int32)
    bg = img[0, 0]
    obj_px = np.abs(img - bg).sum(-1) > 30
    assert obj_px.sum() > 0.05 * img.shape[0] * img.shape[1], "object not in view"
    hue = img[obj_px]
    red_dom = (hue[:, 0] > hue[:, 2] + 60).mean()
    blue_dom = (hue[:, 2] > hue[:, 0] + 60).mean()
    assert red_dom > 0.15 and blue_dom > 0.15, \
        f"expected both checker colours on the object, got red {red_dom:.2f} blue {blue_dom:.2f}"


# ------------------------------------------------------------------ canonical copy (rotate=R)

import hashlib                                                # noqa: E402

from manip_bridge.render_asset import (                       # noqa: E402
    main as render_asset_main, rotation_from_json, validate_rotation,
)

SIM_DIR = Path(os.environ.get("SIM_DIR", Path(__file__).resolve().parents[2] / "ur5e-manip-sim")).resolve()


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(np.deg2rad(deg) * a / np.linalg.norm(a)).as_matrix()


R_CANON = _rot([0.3, -1.0, 0.2], 67.0)       # a TRELLIS-sized tip, arbitrary axis


@pytest.fixture(scope="module")
def canon(pair, tmp_path_factory):
    """Identity (tracking) asset and canonical copy built from the same pair,
    in separate roots -- the intended layout (<run>/assets, <run>/assets_canon)."""
    final, canonical = pair
    root = tmp_path_factory.mktemp("run")
    ident = build_render_asset(final, canonical, root / "assets" / "mug", "mug", require_texture=True)
    rot = build_render_asset(final, canonical, root / "assets_canon" / "mug", "mug",
                             require_texture=True, rotate=R_CANON)
    return final, ident, rot


def _lines(path, prefix):
    return [l for l in Path(path).read_text().splitlines() if l.startswith(prefix)]


def test_canonical_copy_is_final_mesh_at_R(canon):
    final, ident, rot = canon
    Vf, Ff = read_obj(final)
    Vr, Fr = read_obj(Path(rot.obj))
    assert np.abs(Vr - Vf @ R_CANON).max() < 1e-8                       # the vertices ARE V @ R
    assert np.array_equal(Fr, Ff)
    assert _lines(rot.obj, "vt ") == _lines(ident.obj, "vt ")            # uv unchanged
    assert _lines(rot.obj, "f ") == _lines(ident.obj, "f ")
    assert not _lines(rot.obj, "vn"), "a normal line must never be written unrotated"
    assert Path(rot.texture).read_bytes() == Path(ident.texture).read_bytes()
    # proofs ran on the unrotated final_mesh: same fit as the identity build
    assert rot.scale_xyz == ident.scale_xyz and rot.centre_xyz == ident.centre_xyz
    assert np.allclose(rot.bounds_min, Vr.min(0)) and np.allclose(rot.bounds_max, Vr.max(0))


def test_canonical_copy_is_never_a_tracking_asset(canon):
    _, ident, rot = canon
    mesh_dir = Path(rot.obj).parent
    assert rot.fp_obj is None
    assert not list(mesh_dir.glob("*_fp.obj")) and not list(mesh_dir.glob("*.mtl"))
    assert "NOT a tracking asset" in rot.transform
    assert "CANONICAL COPY" in Path(rot.obj).read_text().splitlines()[0]
    assert ident.fp_obj is not None                                     # identity build unchanged


def test_record_carries_applied_rotation_and_hash(canon):
    _, ident, rot = canon
    doc = json.loads((Path(rot.out_dir) / "render_asset.json").read_text())
    assert np.allclose(doc["rotation_body_from_canon"], R_CANON, atol=0)
    assert doc["obj_sha256"] == hashlib.sha256(Path(rot.obj).read_bytes()).hexdigest()
    idoc = json.loads((Path(ident.out_dir) / "render_asset.json").read_text())
    assert idoc["rotation_body_from_canon"] is None
    assert idoc["obj_sha256"] == hashlib.sha256(Path(ident.obj).read_bytes()).hexdigest()
    assert idoc["obj_sha256"] != doc["obj_sha256"]


@pytest.mark.parametrize("bad, match", [
    (R_CANON * 1.01, "orthonormal"),
    (np.diag([1.0, 1.0, -1.0]), "reflection"),
    (np.eye(3)[:2], "3x3"),
    (np.full((3, 3), np.nan), "3x3"),
])
def test_bad_rotation_refused(pair, tmp_path, bad, match):
    final, canonical = pair
    with pytest.raises(RenderAssetError, match=match):
        build_render_asset(final, canonical, tmp_path / "mug", "mug", rotate=bad)
    assert not (tmp_path / "mug").exists(), "refused before writing anything"


def test_dir_kinds_do_not_mix(pair, tmp_path):
    final, canonical = pair
    ident_dir, canon_dir = tmp_path / "assets" / "mug", tmp_path / "assets_canon" / "mug"
    build_render_asset(final, canonical, ident_dir, "mug", require_texture=True)
    with pytest.raises(RenderAssetError, match="tracking"):
        build_render_asset(final, canonical, ident_dir, "mug", rotate=R_CANON)
    assert json.loads((ident_dir / "render_asset.json").read_text())["rotation_body_from_canon"] is None
    build_render_asset(final, canonical, canon_dir, "mug", rotate=R_CANON)
    with pytest.raises(RenderAssetError, match="rotated canonical copy"):
        build_render_asset(final, canonical, canon_dir, "mug")
    # an untextured identity build has no _fp.obj: its json alone marks the kind
    plain_dir = tmp_path / "plain" / "mug"
    build_render_asset(final, canonical, plain_dir, "mug")
    (plain_dir / "meshes" / "mug_fp.obj").unlink(missing_ok=True)
    with pytest.raises(RenderAssetError, match="tracking"):
        build_render_asset(final, canonical, plain_dir, "mug", rotate=R_CANON)


def test_rotated_rebuild_over_rotated_is_allowed(pair, tmp_path):
    """Iterating R (e.g. an OA front shows up on a later run) rebuilds in
    place; the record, hash and geometry follow the new R, and a stale check
    render does not survive."""
    final, canonical = pair
    d = tmp_path / "assets_canon" / "mug"
    first = build_render_asset(final, canonical, d, "mug", rotate=R_CANON)
    (d / "mug_check.png").write_bytes(b"stale")
    R2 = _rot([0, 0, 1], 30.0) @ R_CANON
    second = build_render_asset(final, canonical, d, "mug", rotate=R2)
    assert not (d / "mug_check.png").exists()
    assert np.allclose(json.loads((d / "render_asset.json").read_text())["rotation_body_from_canon"], R2)
    assert second.obj_sha256 != first.obj_sha256
    Vf, _ = read_obj(final)
    assert np.abs(read_obj(Path(second.obj))[0] - Vf @ R2).max() < 1e-8


def test_writer_bug_is_caught_by_read_back(pair, tmp_path, monkeypatch):
    """The read-back is a real check, not a tautology: a writer that drifts
    by 10 um is refused."""
    import manip_bridge.render_asset as ra
    final, canonical = pair
    real = ra.write_obj

    def drifting(path, V, F, uv, header=""):
        real(path, V + np.array([1e-5, 0, 0]), F, uv, header)
    monkeypatch.setattr(ra, "write_obj", drifting)
    with pytest.raises(RenderAssetError, match="read back"):
        build_render_asset(final, canonical, tmp_path / "mug", "mug", rotate=R_CANON)


def test_rotate_from_json(pair, tmp_path):
    final, canonical = pair
    good = tmp_path / "canonical.json"
    good.write_text(json.dumps({"R_body_from_canon": R_CANON.tolist(), "up": {}}))
    assert np.allclose(rotation_from_json(good), R_CANON)
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"R": R_CANON.tolist()}))
    with pytest.raises(RenderAssetError, match="R_body_from_canon"):
        rotation_from_json(bad)
    refl = tmp_path / "refl.json"
    refl.write_text(json.dumps({"R_body_from_canon": np.diag([1.0, -1.0, 1.0]).tolist()}))
    with pytest.raises(RenderAssetError, match="reflection"):
        rotation_from_json(refl)
    # end to end through the CLI
    assert render_asset_main(["--final", str(final), "--canonical", str(canonical), "--name", "mug",
                              "--out", str(tmp_path / "assets_canon"),
                              "--rotate-from", str(good)]) == 0
    doc = json.loads((tmp_path / "assets_canon" / "mug" / "render_asset.json").read_text())
    assert np.allclose(doc["rotation_body_from_canon"], R_CANON)
    assert doc["fp_obj"] is None


def test_validate_rotation_accepts_json_round_trip():
    """R as canonical_frame writes it (json floats) passes the 1e-6 gate."""
    R = np.asarray(json.loads(json.dumps(R_CANON.tolist())))
    assert np.array_equal(validate_rotation(R), R)


@pytest.mark.skipif(not (SIM_DIR / "scripts" / "ground_parts.py").is_file(),
                    reason="no sim checkout (set SIM_DIR)")
def test_sim_load_obj_reads_the_canonical_copy(canon, tmp_path):
    """Parity with what grounding actually reads: the sim's own parser
    (through render_compat, the one sanctioned import path) on the written
    canonical OBJ, against final_mesh @ R -- not against read_obj, which
    would be circular. In a subprocess: test_make_part_masks installs a stub
    render_compat in sys.modules, and this must see the real one."""
    import subprocess
    final, _, rot = canon
    Vf, Ff = read_obj(final)
    np.save(tmp_path / "want_V.npy", Vf @ R_CANON)
    np.save(tmp_path / "want_F.npy", Ff)
    bridge = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "manip_bridge"
    code = (
        "import sys, numpy as np; from pathlib import Path\n"
        "from manip_bridge.render_compat import load_obj\n"
        f"V, F = load_obj(Path({rot.obj!r}))\n"
        f"d = Path({str(tmp_path)!r})\n"
        "err = float(np.abs(np.asarray(V, float) - np.load(d / 'want_V.npy')).max())\n"
        "assert err < 1e-8, err\n"
        "assert np.array_equal(np.asarray(F), np.load(d / 'want_F.npy'))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(bridge)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_canonical_copy_renders_textured(canon):
    if not _gl_ok():
        pytest.skip("no MuJoCo GL backend (set MUJOCO_GL=egl|osmesa; osmesa needs libosmesa6)")
    _, _, rot = canon
    img = render_check(Path(rot.out_dir), "mug", px=320).astype(np.int32)
    obj_px = np.abs(img - img[0, 0]).sum(-1) > 30
    assert obj_px.sum() > 0.05 * img.shape[0] * img.shape[1], "object not in view"
    hue = img[obj_px]
    assert (hue[:, 0] > hue[:, 2] + 60).mean() > 0.15 and (hue[:, 2] > hue[:, 0] + 60).mean() > 0.15
