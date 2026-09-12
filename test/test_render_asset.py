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
