"""outputs/runs/show_candidates.py and candidates_compat.render_pool -- the
whole written candidate pool drawn on the canonical views (inspection).

Offline with a stand-in renderer module; the real sim renderer only in the
GL-gated test, in a SUBPROCESS (MuJoCo binds its GL backend at first
import). Needs the sim checkout at $SIM_DIR or ../ur5e-manip-sim.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
RUNS = REPO_ROOT / "outputs" / "runs"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
pytestmark = pytest.mark.skipif(not (SIM_DIR / "scripts" / "render_candidates.py").is_file(),
                                reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")

POOL = [{"id": 2, "source": "constructed", "symbol": "rim_center", "part": "rim",
         "xyz": [0.0, 0.0, 0.08]},
        {"id": 0, "source": "part", "part": "rim", "xyz": [0.04, 0.0, 0.08]},
        {"id": 1, "source": "fps", "xyz": [0.0, 0.04, 0.02]}]


@pytest.fixture
def mods():
    for p in (str(BRIDGE), str(SIM_DIR)):
        sys.path.insert(0, p)
    import manip_bridge.candidates_compat as cc
    spec = importlib.util.spec_from_file_location("show_candidates", RUNS / "show_candidates.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    yield cc, sc
    for p in (str(BRIDGE), str(SIM_DIR)):
        sys.path.remove(p)


class FakeRC:
    """scripts.render_candidates' inspection surface, recording its inputs."""
    __file__ = "fake_render_candidates.py"
    VIEW_PX = 64

    def __init__(self):
        self.seen = {}

    def load_pool(self, obj_dir):
        doc = json.loads((Path(obj_dir) / "candidates.json").read_text())
        return {int(c["id"]): {**c, "xyz": np.asarray(c["xyz"], float)} for c in doc["candidates"]}

    def load_visual_mesh(self, name, obj_dir):
        return np.zeros((4, 3)), np.zeros((1, 3), int)

    def render_views(self, name, obj_dir, V, candidates, size, font_px, mark_r, axes=None):
        self.seen.update(ids=[c["id"] for c in candidates], axes=axes, size=size)
        vis = {"iso": np.array([True, False, False]), "top": np.array([True, True, False])}
        return {"iso": None, "top": None}, vis, []

    def montage(self, views, name):
        from PIL import Image
        return Image.new("RGB", (8, 8))


def _run(tmp_path, names=("m1",)):
    run = tmp_path / "run"
    for n in names:
        d = run / "grounding_vlm" / "canon" / n
        d.mkdir(parents=True)
        (d / "candidates.json").write_text(json.dumps({"candidates": POOL}))
        (d / "frames.json").write_text(json.dumps({"axes": {"up_axis": {"xyz": [0, 0, 1]}}}))
    return run


def test_whole_written_pool_in_id_order_with_axes(mods, tmp_path, monkeypatch):
    cc, sc = mods
    fake = FakeRC()
    monkeypatch.setattr(cc, "_load", lambda: fake)
    run = _run(tmp_path, ("m1", "m2"))
    out = sc.show(run)
    assert sorted(out) == ["m1", "m2"]
    png, rows = out["m1"]
    assert png == run / "grounding_vlm" / "candidates_m1.png" and png.is_file()
    assert fake.seen["ids"] == [0, 1, 2]                       # all of it, id order, no subset
    assert fake.seen["axes"] == {"up_axis": [0, 0, 1]} and fake.seen["size"] == FakeRC.VIEW_PX
    assert rows == [(0, "part", "rim", "", 2), (1, "fps", "", "", 1),
                    (2, "constructed", "rim", "rim_center", 0)]
    assert sc.main(["--run", str(run), "--names", "m2"]) == 0


def test_refusals(mods, tmp_path, monkeypatch):
    cc, sc = mods
    with pytest.raises(SystemExit, match="run ground_asset.py first"):
        sc.show(tmp_path)
    run = _run(tmp_path)
    with pytest.raises(SystemExit, match=r"no written pool for \['m9'\]"):
        sc.show(run, names=["m9"])

    class Renamed:
        __file__ = "renamed.py"
    monkeypatch.setattr(cc, "_load", Renamed)
    with pytest.raises(cc.CandidatesCompatError, match="has no"):
        sc.show(run)


def _gl_ok() -> bool:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    r = subprocess.run([sys.executable, "-c", "import mujoco; m=mujoco.MjModel.from_xml_string("
                        "'<mujoco/>'); mujoco.Renderer(m, 8, 8).close()"],
                       env=env, capture_output=True, timeout=120)
    return r.returncode == 0


def test_real_renderer_draws_the_pool(tmp_path):
    """Against the ACTUAL sim render_candidates: the names render_pool needs
    exist and the montage is written."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    if not _gl_ok():
        pytest.skip("no MuJoCo GL backend (MUJOCO_GL=egl|osmesa)")
    run = tmp_path / "run"
    asset = run / "grounding_vlm" / "canon" / "box"
    (asset / "meshes").mkdir(parents=True)
    trimesh.creation.box((0.06, 0.04, 0.08)).export(asset / "meshes" / "box_visual.obj")
    (asset / "box.xml").write_text(
        '<mujoco model="box"><asset><mesh name="box_visual" file="meshes/box_visual.obj"/>'
        '</asset><worldbody><body name="box"><geom type="mesh" mesh="box_visual" '
        'group="1" rgba="0.8 0.6 0.4 1"/></body></worldbody></mujoco>')
    (asset / "candidates.json").write_text(json.dumps({"candidates": [
        {"id": 0, "source": "part", "part": "side", "xyz": [0.03, 0.0, 0.0]},
        {"id": 1, "source": "fps", "xyz": [0.0, 0.0, 0.04]}]}))
    (asset / "frames.json").write_text(json.dumps({"axes": {"up_axis": {"xyz": [0, 0, 1]}}}))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    env.setdefault("MUJOCO_GL", "egl")
    r = subprocess.run([sys.executable, str(RUNS / "show_candidates.py"), "--run", str(run)],
                       env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "box: 2 candidates" in r.stdout
    png = run / "grounding_vlm" / "candidates_box.png"
    from PIL import Image
    w, h = Image.open(png).size
    assert w > 1000 and h > 500                                # 4 x 2 montage of 640 px views
