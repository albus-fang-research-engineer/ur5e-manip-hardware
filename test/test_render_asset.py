"""Offline tests for manip_bridge.render_asset -- no ROS, no sidecars.

Fixture: a synthetic textured mesh (icosphere, spherical UVs, checkerboard
texture) written as a "canonical" GLB, DELIBERATELY OFF-CENTRE so a
recentring bug is detectable, plus a "metric" GLB produced by the exact
recipe trellis2_server/server.py uses (trimesh.load -> concatenate ->
apply_scale -> geometry-only export). The builder must reproduce the
metric GLB's vertex set from (canonical x scale), keep the texture, and
refuse a wrong scale.

The render test needs a MuJoCo GL backend: MUJOCO_GL=egl on the
workstation, osmesa on a GPU-less box (`apt install libosmesa6`). It
skips, not fails, when neither initialises.

Run from the repo root:  python -m pytest test/test_render_asset.py -v
Host deps: pip install trimesh scipy pillow mujoco
"""

import io
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
    BODY_NAME, RenderAssetError, build_render_asset, load_metric_vertices,
    render_check, vertex_set_deviation,
)

SCALE = 0.1372                       # canonical -> metres, teapot-ish
OFFSET = np.array([0.31, -0.22, 0.14])   # canonical-frame offset; recentring would remove it


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


def _server_metric_recipe(glb_bytes: bytes, scale: float, out: Path) -> Path:
    """Verbatim what trellis2_server does for the metric copy."""
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    tm = (trimesh.util.concatenate(list(scene.geometry.values()))
          if isinstance(scene, trimesh.Scene) else scene)
    tm_metric = tm.copy()
    tm_metric.apply_scale(scale)
    tm_metric.export(out)
    return out


@pytest.fixture(scope="module")
def glbs(tmp_path_factory):
    d = tmp_path_factory.mktemp("trellis")
    canonical = d / "obj.glb"
    _textured_sphere().export(canonical)
    metric = _server_metric_recipe(canonical.read_bytes(), SCALE, d / "obj_metric.glb")
    return canonical, metric


@pytest.fixture(scope="module")
def asset(glbs, tmp_path_factory):
    canonical, metric = glbs
    out = tmp_path_factory.mktemp("assets") / "obj"
    rec = build_render_asset(canonical, SCALE, out, "obj", metric_glb=metric,
                             require_texture=True)
    return out, rec


def _read_obj(path: Path):
    """The sim's load_obj rule: `v` lines, first index of each face token."""
    V, F, VT = [], [], 0
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT += 1
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(V), np.asarray(F), VT


# ------------------------------------------------------------------ identity

def test_vertex_set_identity_against_metric_glb(asset, glbs):
    out, rec = asset
    V, F, _ = _read_obj(Path(rec.obj))
    Vm = load_metric_vertices(glbs[1])
    assert rec.max_vertex_deviation_m is not None and rec.max_vertex_deviation_m <= 1e-6
    assert vertex_set_deviation(V, Vm) <= 1e-6         # what was WRITTEN, not just computed
    assert len(F) > 0 and F.max() < len(V)


def test_no_recentre(asset, glbs):
    """The metric GLB is off-centre by design; the asset must be too
    (convert_asset.py would have moved it to the origin)."""
    out, rec = asset
    V, _, _ = _read_obj(Path(rec.obj))
    Vm = load_metric_vertices(glbs[1])
    np.testing.assert_allclose(V.min(0), Vm.min(0), atol=1e-6)
    np.testing.assert_allclose(V.max(0), Vm.max(0), atol=1e-6)
    center = 0.5 * (V.min(0) + V.max(0))
    assert np.linalg.norm(center - OFFSET * SCALE) < 1e-6
    assert np.linalg.norm(center) > 0.01                # i.e. NOT at the origin


def test_wrong_scale_refused(glbs, tmp_path):
    canonical, metric = glbs
    with pytest.raises(RenderAssetError, match="deviates"):
        build_render_asset(canonical, SCALE * 1.02, tmp_path / "bad", "bad", metric_glb=metric)


def test_recentred_copy_refused(glbs, tmp_path):
    """A metric GLB that someone recentred (the convert_asset failure mode)
    must not verify against the canonical x scale."""
    canonical, _ = glbs
    m = trimesh.load(str(_server_metric_recipe(canonical.read_bytes(), SCALE,
                                               tmp_path / "m.glb")), force="mesh")
    m.apply_translation(-m.bounds.mean(axis=0))
    m.export(tmp_path / "m_recentred.glb")
    with pytest.raises(RenderAssetError, match="deviates"):
        build_render_asset(canonical, SCALE, tmp_path / "bad", "bad",
                           metric_glb=tmp_path / "m_recentred.glb")


def test_multi_geometry_refused(tmp_path):
    a, b = _textured_sphere(), _textured_sphere()
    b.apply_translation([2.0, 0, 0])
    trimesh.Scene([a, b]).export(tmp_path / "two.glb")
    with pytest.raises(RenderAssetError, match="geometries"):
        build_render_asset(tmp_path / "two.glb", 1.0, tmp_path / "two", "two")


# ------------------------------------------------------------------ texture

def test_texture_written_and_wired(asset):
    out, rec = asset
    assert rec.texture and Path(rec.texture).is_file() and Path(rec.texture).is_absolute()
    V, F, n_vt = _read_obj(Path(rec.obj))
    assert n_vt == len(V), "one vt per vertex"
    assert "/" in next(l for l in Path(rec.obj).read_text().splitlines() if l.startswith("f "))
    root = ET.parse(rec.xml).getroot()
    tex = root.find("asset/texture")
    mat = root.find("asset/material")
    geom = root.find(f"worldbody/body/body[@name='{BODY_NAME}']/geom")
    assert tex is not None and tex.get("file") == rec.texture and tex.get("type") == "2d"
    assert mat is not None and mat.get("texture") == tex.get("name")
    assert geom is not None and geom.get("material") == mat.get("name")
    assert geom.get("group") == "1" and geom.get("contype") == "0"
    assert root.find("asset/mesh").get("file") == "meshes/obj_visual.obj"
    assert not [m for m in root.findall("asset/mesh") if "_col_" in m.get("name", "")]


def test_untextured_mesh_is_flagged(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2)
    m.export(tmp_path / "plain.glb")
    with pytest.raises(RenderAssetError, match="texture"):
        build_render_asset(tmp_path / "plain.glb", 1.0, tmp_path / "p", "p", require_texture=True)
    rec = build_render_asset(tmp_path / "plain.glb", 1.0, tmp_path / "p", "p")
    assert rec.texture is None
    assert ET.parse(rec.xml).getroot().find("asset/texture") is None


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
    img = render_check(out, "obj", px=320).astype(np.int32)
    bg = img[0, 0]
    obj_px = np.abs(img - bg).sum(-1) > 30
    assert obj_px.sum() > 0.05 * img.shape[0] * img.shape[1], "object not in view"
    hue = img[obj_px]
    red_dom = (hue[:, 0] > hue[:, 2] + 60).mean()
    blue_dom = (hue[:, 2] > hue[:, 0] + 60).mean()
    assert red_dom > 0.15 and blue_dom > 0.15, \
        f"expected both checker colours on the object, got red {red_dom:.2f} blue {blue_dom:.2f}"
