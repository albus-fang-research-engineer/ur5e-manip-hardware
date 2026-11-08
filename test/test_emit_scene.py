"""outputs/runs/emit_scene.py -- VLM call #3 on a hardware run -- and
emit_compat.

Offline and GL-free: the pieces the driver adds around the sim's loop
(asset aliasing, the hardware scene stand-in, the geometry report, the
call-#2/plan consistency gate, the dry-run request dump). The loop itself
is sim emit_constraints' and runs in the container (--dry-run / --replay).
Needs the sim checkout at $SIM_DIR or ../ur5e-manip-sim; skips without it.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
RUNS = REPO_ROOT / "outputs" / "runs"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "scripts" / "emit_constraints.py").is_file()
pytestmark = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
pytest.importorskip("trimesh")                       # check_symbols, via select_scene
pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module", autouse=True)
def _paths():
    paths = [str(BRIDGE), str(SIM_DIR), str(RUNS)]
    for p in paths:
        sys.path.insert(0, p)
    yield
    for p in paths:
        sys.path.remove(p)


@pytest.fixture(scope="module")
def es(_paths):
    spec = importlib.util.spec_from_file_location("emit_scene", RUNS / "emit_scene.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_emit_compat_surface():
    from manip_bridge import emit_compat as ec
    for name in ec.__all__:
        assert hasattr(ec, name), name
    assert ec.COMPILE_RETRIES >= 0
    from manip_bridge import selection_compat, vlm_compat
    assert vlm_compat.MAX_PARSE_RETRIES >= 0 and callable(vlm_compat.entry_line)
    assert callable(selection_compat.selected_points)
    prov = ec.provenance()
    assert any(f.endswith("compile_tsr.py") for f in prov["files"])


def test_alias_assets_renames_to_contract_names(es, tmp_path):
    canon = tmp_path / "canon" / "lib_teapot"
    (canon / "meshes").mkdir(parents=True)
    for f in ("frames.json", "candidates.json", "lib_teapot.xml"):
        (canon / f).write_text("{}")
    for f in ("lib_teapot_visual.obj", "lib_teapot_texture.png"):
        (canon / "meshes" / f).write_text("x")
    dirs = es.alias_assets({"teapot": {"canon_dir": canon}}, tmp_path / "out")
    d = dirs["teapot"]
    assert (d / "teapot.xml").resolve() == (canon / "lib_teapot.xml").resolve()
    assert (d / "meshes" / "teapot_visual.obj").resolve() == \
        (canon / "meshes" / "lib_teapot_visual.obj").resolve()
    # the xml's own relative mesh paths still resolve
    assert (d / "meshes" / "lib_teapot_texture.png").is_file()
    assert (d / "frames.json").is_file() and (d / "candidates.json").is_file()
    # re-running replaces the alias dir, never the canonical copy
    es.alias_assets({"teapot": {"canon_dir": canon}}, tmp_path / "out")
    assert (canon / "lib_teapot.xml").is_file() and not (canon / "teapot.xml").exists()


def test_hardware_scene_has_no_spawn_poses(es, tmp_path):
    sc = es.HardwareScene({"mug": tmp_path}, 0.01, (1, 1, 0.02), tmp_path / "plan.json")
    assert sc.objects["mug"].xml == tmp_path / "mug.xml"
    assert sc.asset_dirs == {"mug": tmp_path}
    with pytest.raises(RuntimeError):
        sc.fixed_poses()


def test_geometry_reports_points_relative_to_w(es):
    T_sub = np.eye(4)
    T_sub[:3, 3] = [0.5, 0.0, 0.2]                  # mover's canonical frame at the subgoal
    cs = SimpleNamespace(subgoal=SimpleNamespace(nominal=lambda: T_sub))
    sym = {"teapot": SimpleNamespace(points={"handle_center": np.array([0.0, 0.0, 0.0]),
                                             "spout_center": np.array([0.1, 0.0, 0.05])}),
           "mug": SimpleNamespace(points={"rim_center": np.array([0.0, 0.0, 0.1])})}
    T_mug = np.eye(4)
    T_mug[:3, 3] = [0.5, 0.0, 0.0]
    stage = SimpleNamespace(active="teapot", passive="mug")
    g = es.geometry(stage, cs, {"mug": T_mug}, np.array([0.0, 0.0, 0.1]),
                    np.array([0.0, 0.0, 0.0]), sym)
    assert g["pinned"] == "teapot.handle_center"
    assert g["w_origin_base"] == [0.5, 0.0, 0.1]
    assert g["points"]["handle_center"] == {"horizontal_m": 0.0, "height_m": 0.1}
    assert g["points"]["spout_center"] == {"horizontal_m": 0.1, "height_m": 0.15}


def test_strip_images_keeps_text(es):
    p = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 40}}]}]}
    s = es._strip_images(p)
    assert s["messages"][0]["content"][0]["text"] == "hi"
    assert s["messages"][0]["content"][1]["source"]["data"] == "<40 base64 chars>"
    assert p["messages"][0]["content"][1]["source"]["data"] == "A" * 40     # input untouched


def _fake_inputs(es, monkeypatch, plan_sha):
    doc = {"hardware": {"plan_sha": plan_sha}}
    monkeypatch.setattr(es.ss, "load_inputs", lambda run, plan: (SimpleNamespace(roles={}), doc, {}))


def test_refuses_selections_made_for_another_plan(es, tmp_path, monkeypatch):
    _fake_inputs(es, monkeypatch, "aaaa")
    (tmp_path / "call2").mkdir()
    (tmp_path / "call2" / "selections.json").write_text("{}")
    (tmp_path / "call2" / "selections.log.json").write_text(json.dumps({"plan_sha": "bbbb"}))
    with pytest.raises(es.EmitRefusal, match="re-run select_scene"):
        es.emit_scene(tmp_path)


def test_refuses_selections_from_a_failed_select_run(es, tmp_path, monkeypatch):
    _fake_inputs(es, monkeypatch, "aaaa")
    (tmp_path / "call2").mkdir()
    (tmp_path / "call2" / "selections.json").write_text("{}")
    (tmp_path / "call2" / "selections.log.json").write_text(json.dumps(
        {"plan_sha": "aaaa", "failure": "'grasp': mark 3 does not resolve"}))
    with pytest.raises(es.EmitRefusal, match="failed select_scene run"):
        es.emit_scene(tmp_path)


def test_refuses_missing_selections_and_bad_arm(es, tmp_path, monkeypatch):
    _fake_inputs(es, monkeypatch, "aaaa")
    with pytest.raises(es.EmitRefusal, match="select_scene.py first"):
        es.emit_scene(tmp_path)
    with pytest.raises(es.EmitRefusal, match="--arm"):
        es.emit_scene(tmp_path, arm="framed")


def test_main_refusal_exits_2(es, tmp_path, monkeypatch, capsys):
    _fake_inputs(es, monkeypatch, "aaaa")
    assert es.main(["--run", str(tmp_path)]) == 2
    assert "REFUSED" in capsys.readouterr().err
