"""outputs/runs/select_scene.py -- VLM call #2 on a hardware run -- and its
compats (selection_compat, candidates_compat, vlm_record).

Offline: a run bound by the REAL plan_scene --grounding pass (canned call
#1), a grounded canonical asset dir, a stub renderer that writes the
manifest the way run_vlm does (from the same sim menu functions), and
CANNED call-#2 responses through the real sim client, parser and resolver.
The real renderer runs only in the GL-gated test at the end. Needs the sim
checkout at $SIM_DIR or ../ur5e-manip-sim; skips without it.
"""
import ast
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
HAVE_SIM = (SIM_DIR / "manip_sim" / "selection.py").is_file() and \
    (SIM_DIR / "scripts" / "render_candidates.py").is_file()
pytestmark = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("trimesh")                       # check_symbols, imported by the driver


def _load(name):
    spec = importlib.util.spec_from_file_location(name, RUNS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    for p in (str(BRIDGE), str(SIM_DIR), str(RUNS)):
        sys.path.insert(0, p)
    import test_plan_scene as tps
    yield _load("plan_scene"), _load("select_scene"), tps
    for p in (str(BRIDGE), str(SIM_DIR), str(RUNS)):
        sys.path.remove(p)


# canonical -> body: 90 deg about z (so a canonical -x handle sits on body -y)
R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T = np.eye(4)
T[:3, 3] = [0.02, -0.01, 0.50]
K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
HANDLE = [[-0.060, 0.0, 0.050], [-0.070, 0.0, 0.030], [-0.060, 0.0, 0.010]]
OTHER = [[0.0, 0.040, 0.050]]
PICK = 1                                             # the middle handle mark


def _px(xc):
    xb = R @ np.asarray(xc)
    x = T[:3, :3] @ xb + T[:3, 3]
    return [int(round(K[0, 0] * x[0] / x[2] + K[0, 2])), int(round(K[1, 1] * x[1] / x[2] + K[1, 2]))]


def _bound_run(mods, tmp):
    """run_scene-like run -> call #1 (canned) -> grounding_vlm -> plan_scene
    --grounding: the bound plan.json select_scene starts from."""
    ps, _, tps = mods
    run = tps._run_dir(tmp)
    summary = json.loads((run / "summary.json").read_text())
    summary["frame"] = {"K": K.tolist()}
    summary["objects"]["m2"]["pose_on_final"]["cam_T_obj"] = T.tolist()
    (run / "summary.json").write_text(json.dumps(summary))
    PIL_Image.fromarray(np.full((480, 640, 3), 90, np.uint8)).save(run / "rgb.png")
    mask = np.zeros((480, 640), np.uint8)
    for xc in HANDLE:
        u, v = _px(xc)
        mask[v - 6:v + 7, u - 6:u + 7] = 255
    PIL_Image.fromarray(mask).save(run / "mask_m2.png")
    art, log = tps._plan_and_log(ps, run, tmp, tps.HANDLE_BODY)
    g = tps._grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"],
                      ["handle"], dropped={"body": "no mask in any of 8 views"})
    canon = g / "canon" / "mug"
    cands = [{"id": i, "source": "part", "part": "handle", "xyz": x} for i, x in enumerate(HANDLE)]
    cands += [{"id": len(HANDLE), "source": "fps", "xyz": OTHER[0]}]
    (canon / "candidates.json").write_text(json.dumps({"candidates": cands}))
    (canon / "frames.json").write_text(json.dumps({
        "object": "mug", "points": {"handle_center": {"xyz": HANDLE[1]}},
        "axes": {"up_axis": {"xyz": [0, 0, 1]}, "front_axis": {"xyz": [1, 0, 0]}},
        "quantities": {}}))
    fp = g / "m2" / "frames.json"
    doc = json.loads(fp.read_text())
    doc["provenance"]["canonical"]["R_body_from_canon"] = R.tolist()
    fp.write_text(json.dumps(doc))
    code, _, _ = ps.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                               grounding=str(g))
    assert code == 0
    return run


class StubRender:
    """Writes what run_vlm writes, from the SAME sim menu functions."""

    def __init__(self, bad_menu=False):
        self.calls, self.bad_menu = [], bad_menu

    def __call__(self, name, obj_dir, parts, out_dir):
        from manip_bridge import selection_compat as sc
        self.calls.append((name, Path(obj_dir), list(parts), Path(out_dir)))
        sub = sc.vlm_subset(sc.load_pool(obj_dir), list(parts))
        menu = sc.menu_from_pool(sub)
        if self.bad_menu:
            menu = {k: t for k, t in list(menu.items())[:-1]}
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        views = {}
        for v in ("+x", "top"):
            p = d / f"{v.replace('+', 'p')}.png"
            PIL_Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
            views[v] = {"path": str(p), "visible_ids": sorted(sub)}
        man = {"object": name, "source": str(Path(obj_dir) / "candidates.json"),
               "parts_filter": list(parts), "ids": sorted(sub),
               "menu": {str(i): t for i, t in menu.items()}, "views": views}
        (d / "manifest.json").write_text(json.dumps(man))
        return man, d / "manifest.json"


def _resp(cid, why="the handle's middle is where a gripper closes"):
    return json.dumps({"candidate_id": cid, "rationale": why})


# ------------------------------------------------------------------ happy path

def test_grasp_selection_resolves_and_lands_on_the_real_handle(mods, tmp_path):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    t, rend = tps.Canned(_resp(PICK)), StubRender()
    code, sels, rep = ss.select_scene(run, transport=t, render=rend)
    assert code == 0
    # one render per object, the object's GROUNDED parts (body pruned), under <out>/vlm
    assert rend.calls == [("mug", run / "grounding_vlm" / "canon" / "mug", ["handle"],
                           run / "call2" / "vlm")]
    # the real sim prompt: stage text with the pruned parts, both views attached
    blocks = t.payloads[0]["messages"][0]["content"]
    assert "grasp mug by handle" in blocks[0]["text"] and "'mug': ['handle']" in blocks[0]["text"]
    assert sum(b.get("type") == "image" for b in blocks) == 2
    # the artifact is sim's format, proven by sim's reader
    from manip_bridge import selection_compat as sc
    path = run / "call2" / "selections.json"
    loaded = sc.load_selections(path)
    assert loaded["grasp"].candidate_id == PICK and sc.selection_objects(path) == {"grasp": "mug"}
    # resolved, mapped to the real frame exactly as computed by hand
    r = rep["resolved"]["grasp"]
    assert r["pixel"] == _px(HANDLE[PICK]) and r["in_image"] and r["in_outline"] is True
    assert np.allclose(r["xyz_body"], R @ np.asarray(HANDLE[PICK]))
    assert "point-only default" in r["axis_source"]
    assert Path(r["overlay"]).is_file()
    log = json.loads((run / "call2" / "selections.log.json").read_text())
    assert log["raw_responses"] == [_resp(PICK)] and log["failure"] is None
    assert log["menus"]["mug"][str(PICK)] == "part:handle" and log["sim"]["selection"]["files"]


def test_replay_is_offline_and_identical(mods, tmp_path, monkeypatch):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    _, live, _ = ss.select_scene(run, transport=tps.Canned(_resp(PICK)), render=StubRender())
    log = tmp_path / "sel.log.json"
    log.write_text((run / "call2" / "selections.log.json").read_text())
    monkeypatch.setattr(ss.vlm_compat, "live_transport",
                        lambda p: (_ for _ in ()).throw(AssertionError("network")))
    _, again, rep = ss.select_scene(run, replay=str(log), render=StubRender())
    assert again == live
    assert json.loads((run / "call2" / "selections.log.json").read_text())["replayed_from"] \
        == str(log.resolve())


def test_off_menu_id_is_rejected_retried_and_logged(mods, tmp_path):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    code, sels, _ = ss.select_scene(run, transport=tps.Canned(_resp(99), _resp(PICK)),
                                    render=StubRender())
    assert code == 0 and sels["grasp"]["candidate_id"] == PICK
    log = json.loads((run / "call2" / "selections.log.json").read_text())
    assert log["raw_responses"] == [_resp(99), _resp(PICK)]
    assert "not on the offered menu" in log["client_logs"][0]["rejections"][0]


def test_off_outline_pick_is_reported_not_refused(mods, tmp_path, capsys, monkeypatch):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    other = len(HANDLE)                                    # the fps mark, off the handle mask
    monkeypatch.setattr(ss.vlm_compat, "live_transport", tps.Canned(_resp(other)))
    monkeypatch.setattr(ss, "_backend", lambda: (StubRender(), lambda: {"render": "stub"}))
    assert ss.main(["--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "in_outline False" in out and "WARNING: grasp's point is off the object's outline" in out


# ------------------------------------------------------------------ refusals

def test_menu_differing_from_the_render_is_refused(mods, tmp_path):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    t = tps.Canned(_resp(PICK))
    with pytest.raises(ss.SelectRefusal, match="must be one artifact"):
        ss.select_scene(run, transport=t, render=StubRender(bad_menu=True))
    assert t.payloads == []                                 # refused before any call


def test_unbound_plan_is_refused_by_plan_stages_reader(mods, tmp_path):
    ps, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    ps.plan_scene(run, "pick up the mug by its handle", transport=tps.Canned(tps.HANDLE_BODY))
    with pytest.raises(ss.SelectRefusal, match=r"reader refused .*roles\[deferred\]"):
        ss.select_scene(run, transport=tps.Canned(_resp(PICK)), render=StubRender())


def test_grounding_changed_since_binding_is_refused(mods, tmp_path):
    _, ss, tps = mods
    run = _bound_run(mods, tmp_path)
    fp = run / "grounding_vlm" / "m2" / "frames.json"
    fp.write_text(fp.read_text() + " ")
    with pytest.raises(ss.SelectRefusal, match="changed since plan_scene --grounding"):
        ss.select_scene(run, transport=tps.Canned(_resp(PICK)), render=StubRender())


def test_missing_plan_and_missing_views_are_refused(mods, tmp_path):
    _, ss, tps = mods
    with pytest.raises(ss.SelectRefusal, match="plan.json not found"):
        ss.select_scene(tmp_path / "nope", render=StubRender())
    run = _bound_run(mods, tmp_path)

    class Lost(StubRender):
        def __call__(self, *a):
            man, mp = super().__call__(*a)
            for v in man["views"].values():
                Path(v["path"]).unlink()
            return man, mp
    with pytest.raises(ss.SelectRefusal, match="views missing on disk"):
        ss.select_scene(run, transport=tps.Canned(_resp(PICK)), render=Lost())


# ------------------------------------------------------------------ candidates_compat

@pytest.fixture
def cc():
    sys.path.insert(0, str(BRIDGE))
    sys.path.insert(0, str(SIM_DIR))
    import manip_bridge.candidates_compat as m
    yield m
    sys.path.remove(str(SIM_DIR))


class FakeRC:
    """Stands in for scripts.render_candidates: reads VLM_DIR at CALL time,
    as the real run_vlm does."""
    __file__ = "fake_render_candidates.py"

    def __init__(self, mode="ok"):
        self.VLM_DIR = Path("outputs/candidates/vlm")
        self.mode, self.seen = mode, None

    def run_vlm(self, name, obj_dir, parts):
        self.seen = self.VLM_DIR
        if self.mode == "raise":
            raise RuntimeError("GL died")
        if self.mode == "silent":
            return
        d = self.VLM_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"source": str(obj_dir / "candidates.json")}))


def test_render_menu_redirects_and_restores(cc, tmp_path, monkeypatch):
    fake = FakeRC()
    monkeypatch.setattr(cc, "_load", lambda: fake)
    stale = tmp_path / "out" / "mug" / "manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"source": "a previous run"}))
    man, mp = cc.render_menu("mug", tmp_path / "canon", ["handle"], tmp_path / "out")
    assert fake.seen == tmp_path / "out" and fake.VLM_DIR == Path("outputs/candidates/vlm")
    assert mp == stale and man["source"] == str(tmp_path / "canon" / "candidates.json")


def test_render_menu_guards(cc, tmp_path, monkeypatch):
    class NoDir:
        __file__ = "renamed.py"

        def run_vlm(self, *a):
            raise AssertionError("must not run")
    monkeypatch.setattr(cc, "_load", NoDir)
    with pytest.raises(cc.CandidatesCompatError, match=r"no \['VLM_DIR'\]"):
        cc.render_menu("mug", tmp_path, [], tmp_path / "out")
    fake = FakeRC("raise")
    monkeypatch.setattr(cc, "_load", lambda: fake)
    with pytest.raises(RuntimeError, match="GL died"):
        cc.render_menu("mug", tmp_path, [], tmp_path / "out")
    assert fake.VLM_DIR == Path("outputs/candidates/vlm"), "restored after an exception"
    # a previous run's manifest must not pass for this call's
    stale = tmp_path / "out" / "mug" / "manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"source": str(tmp_path / "candidates.json")}))
    monkeypatch.setattr(cc, "_load", lambda: FakeRC("silent"))
    with pytest.raises(cc.CandidatesCompatError, match="wrote no manifest"):
        cc.render_menu("mug", tmp_path, [], tmp_path / "out")


def test_real_run_vlm_reads_vlm_dir_at_call_time():
    """The redirect's precondition, pinned against the ACTUAL sim source
    without importing it (it imports mujoco): VLM_DIR is a module-level
    assignment, run_vlm reads it as a global in its body, and no parameter
    default binds it at definition time."""
    tree = ast.parse((SIM_DIR / "scripts" / "render_candidates.py").read_text())
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name)}
    assert "VLM_DIR" in assigned
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_vlm")
    assert [a.arg for a in fn.args.args] == ["name", "obj_dir", "parts"]
    defaults = [d for d in fn.args.defaults + fn.args.kw_defaults if d is not None]
    assert not any(isinstance(x, ast.Name) and x.id == "VLM_DIR"
                   for d in defaults for x in ast.walk(d))
    body_reads = {x.id for st in fn.body for x in ast.walk(st)
                  if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    assert "VLM_DIR" in body_reads
    src = ast.get_source_segment((SIM_DIR / "scripts" / "render_candidates.py").read_text(), fn)
    assert '"source": str(obj_dir / "candidates.json")' in src     # what render_menu checks


def test_compat_contracts_and_import_weight():
    code = ("import sys\n"
            "import manip_bridge.candidates_compat as c\n"
            "assert set(c.__all__) == {'CandidatesCompatError', 'render_menu', 'render_pool', "
            "'provenance', 'SOURCES'}\n"
            "assert c.SOURCES[0].endswith('scripts/render_candidates.py')\n"
            "import manip_bridge.selection_compat as s\n"
            "assert set(s.__all__) >= {'load_pool', 'vlm_subset', 'menu_from_pool', "
            "'resolve_selection', 'ResolutionError', 'load_selections', 'selection_objects', "
            "'selection_to_json', 'load_symbols'}\n"
            "assert s.SOURCES[0].endswith('manip_sim/selection.py')\n"
            "import manip_bridge.vlm_record\n"
            "assert 'mujoco' not in sys.modules, 'importing the compats must stay GL-free'\n"
            "assert 'scripts.render_candidates' not in sys.modules\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


# ------------------------------------------------------------------ the real renderer

def _gl_ok():
    """Probed in a SUBPROCESS: without a usable backend MuJoCo can abort the
    interpreter (a C-level abort, not an exception), which in-process would
    take the whole test session down instead of skipping."""
    code = ("import mujoco\n"
            "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')\n"
            "mujoco.Renderer(m, 16, 16).close()\n")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, env=env,
                              timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_real_render_lands_under_out(tmp_path):
    """The real run_vlm through render_menu: the eight views and manifest
    land under <out>/<name>/, nothing under the default outputs/, and the
    sim module's VLM_DIR is back to its default afterwards. Rendered in a
    SUBPROCESS: MuJoCo binds its GL backend at first import, so an
    in-process render after another test imported it can abort the session."""
    trimesh = pytest.importorskip("trimesh")
    if not _gl_ok():
        pytest.skip("no MuJoCo GL backend (MUJOCO_GL=egl|osmesa)")
    asset = tmp_path / "canon" / "box"
    (asset / "meshes").mkdir(parents=True)
    trimesh.creation.box((0.06, 0.04, 0.08)).export(asset / "meshes" / "box_visual.obj")
    (asset / "box.xml").write_text(
        '<mujoco model="box"><asset><mesh name="box_visual" file="meshes/box_visual.obj"/>'
        '</asset><worldbody><body name="box"><geom type="mesh" mesh="box_visual" '
        'group="1" rgba="0.8 0.6 0.4 1"/></body></worldbody></mujoco>')   # group 1: what
    # render_asset writes and what render_views shows (it hides group 0)
    cands = [{"id": 0, "source": "part", "part": "side", "xyz": [0.03, 0.0, 0.0]},
             {"id": 1, "source": "fps", "xyz": [0.0, 0.0, 0.04]}]
    (asset / "candidates.json").write_text(json.dumps({"candidates": cands}))
    code = ("import json, sys\n"
            "from pathlib import Path\n"
            "import manip_bridge.candidates_compat as cc\n"
            f"man, mp = cc.render_menu('box', Path({str(asset)!r}), ['side'], "
            f"Path({str(tmp_path / 'out')!r}))\n"
            "print(json.dumps({'mp': str(mp), 'man': man, "
            "'restored': str(cc._load().VLM_DIR)}))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    env.setdefault("MUJOCO_GL", "egl")
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    man = res["man"]
    assert res["mp"] == str(tmp_path / "out" / "box" / "manifest.json")
    assert res["restored"] == "outputs/candidates/vlm"
    assert len(man["views"]) == 8 and all(Path(v["path"]).is_file() for v in man["views"].values())
    assert all(str(tmp_path / "out" / "box") in v["path"] for v in man["views"].values())
    assert not (tmp_path / "outputs").exists()      # the default (cwd-relative) got nothing
    # the object is actually in the colour render, not just the marks on black
    img = np.asarray(PIL_Image.open(man["views"]["iso"]["path"]).convert("RGB"))
    body = (img[..., 0] > 120) & (img[..., 2] < img[..., 0])             # the tan box
    assert body.mean() > 0.01, "object missing from the colour render"
