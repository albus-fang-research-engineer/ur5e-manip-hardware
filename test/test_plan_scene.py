"""outputs/runs/plan_scene.py -- VLM call #1 on a hardware run -- and its compats.

Offline: a fixture run dir (summary.json, a two-mark set with marked.png, a
canonical asset for m2) and CANNED model responses fed through the real sim
client and parser (vlm_compat), so nothing reaches the network. Needs the
sim checkout at $SIM_DIR or the sibling ../ur5e-manip-sim; skips without it.
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
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "manip_sim" / "vlm.py").is_file() and \
    (SIM_DIR / "scripts" / "plan_stages.py").is_file()
pytestmark = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
PIL_Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def drv():
    sys.path.insert(0, str(BRIDGE))
    sys.path.insert(0, str(SIM_DIR))
    spec = importlib.util.spec_from_file_location(
        "plan_scene", REPO_ROOT / "outputs" / "runs" / "plan_scene.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    yield m
    sys.path.remove(str(SIM_DIR))


def _run_dir(tmp, registered=("m2",)):
    run = tmp / "run"
    marks = run / "marks"
    marks.mkdir(parents=True)
    img = np.zeros((48, 64, 3), np.uint8)
    PIL_Image.fromarray(img).save(marks / "marked.png")
    PIL_Image.fromarray(img).save(marks / "rgb.png")
    for i in (1, 2):
        PIL_Image.fromarray(np.zeros((48, 64), np.uint8)).save(marks / f"mask_{i}.png")
    (marks / "marks.json").write_text(json.dumps({
        "source": "sam", "image": "rgb.png", "marked": "marked.png",
        "marks": {"1": {"mask": "mask_1.png", "bbox": [2, 2, 20, 30], "centroid": [10, 15], "area": 400},
                  "2": {"mask": "mask_2.png", "bbox": [30, 5, 60, 40], "centroid": [45, 22], "area": 900}}}))
    objects = {k: {"pose_on_final": {"cam_T_obj": np.eye(4).tolist(),
                                     "mesh": "/data/any6d/final_mesh_mug.obj"}} for k in registered}
    objects.setdefault("m1", {"oriany": {}})
    (run / "summary.json").write_text(json.dumps({"objects": objects}))
    asset = run / "assets_canon" / "mug"
    asset.mkdir(parents=True)
    (asset / "canonical.json").write_text(json.dumps({"inputs": {"key": "m2"}}))
    return run


def _resp(objects, stages):
    return json.dumps({"objects": objects, "stages": stages})


GRASP = _resp({"2": "mug"}, [{"name": "grasp the mug by its handle", "active": 2,
                              "passive": None, "parts": {"2": ["handle"]}}])


class Canned:
    """A transport that answers from a list and records what it was sent."""

    def __init__(self, *responses):
        self.responses, self.payloads = list(responses), []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.responses[len(self.payloads) - 1]


# ------------------------------------------------------------------ happy path

def test_grasp_plan_feeds_grounding(drv, tmp_path):
    run = _run_dir(tmp_path)
    t = Canned(GRASP)
    code, art, rep = drv.plan_scene(run, "pick up the mug by its handle", transport=t)
    assert code == 0
    # the real sim client was driven: one call, the marked image attached
    assert len(t.payloads) == 1
    blocks = t.payloads[0]["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in blocks)
    assert art["object_parts"] == {"mug": ["handle"]}
    assert art["hardware"]["mark_key"] == {"mug": "m2"}
    assert art["roles"] == {} and art["gate"]["blocking"] == ["roles[deferred]"]
    by = {c["check"]: c for c in art["checks"]}
    assert by["registered[mug]"]["pass"] and by["parts[mug]"]["pass"]
    assert by["structure[grasp<interaction]"]["pass"]          # trivially: no interaction stage
    assert "trivially" in by["structure[grasp<interaction]"]["detail"]
    on_disk = json.loads((run / "stage_plan" / "plan.json").read_text())
    assert on_disk == json.loads(json.dumps(art))
    log = json.loads((run / "stage_plan" / "plan.log.json").read_text())
    assert log["raw_responses"] == [GRASP] and log["model"] == art["model"]
    # the model's parts replace the hand-typed --parts, with the asset already built
    nxt = "\n".join(rep["next"])
    assert "--parts handle" in nxt and "--name mug" in nxt and "--key m2" in nxt
    assert f"masks_vlm_{art['hardware']['plan_sha']}" in nxt and "grounding_vlm" in nxt
    assert "canonicalize_asset" not in nxt
    assert art["sim"]["files"]                                   # vlm.py provenance


def test_replay_is_offline_and_identical(drv, tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    _, live, _ = drv.plan_scene(run, "pick up the mug by its handle", transport=Canned(GRASP))
    log = tmp_path / "plan.log.json"
    log.write_text((run / "stage_plan" / "plan.log.json").read_text())
    # replay must never touch the live transport
    monkeypatch.setattr(drv.vlm_compat, "live_transport",
                        lambda p: (_ for _ in ()).throw(AssertionError("network")))
    _, rep_art, _ = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log))
    strip = lambda a: {k: v for k, v in a.items() if k != "hardware"}     # noqa: E731
    assert strip(rep_art) == strip(live)
    assert rep_art["hardware"]["replayed_from"] == str(log.resolve())


def test_parse_rejection_retry_is_recorded_and_replays(drv, tmp_path):
    """A response naming a mark not on the image is rejected by the sim
    parser and retried; both raw responses are logged and replay in order."""
    run = _run_dir(tmp_path)
    bad = _resp({"7": "mug"}, [{"name": "grasp", "active": 7, "passive": None, "parts": {"7": ["handle"]}}])
    code, art, _ = drv.plan_scene(run, "pick up the mug", transport=Canned(bad, GRASP))
    log = json.loads((run / "stage_plan" / "plan.log.json").read_text())
    assert code == 0 and log["raw_responses"] == [bad, GRASP]
    assert "mark 7 is not on the image" in log["client_logs"][0]["rejections"][0]
    lp = tmp_path / "l.json"
    lp.write_text(json.dumps(log))
    _, again, _ = drv.plan_scene(run, "pick up the mug", replay=str(lp))
    assert again["plan"] == art["plan"]


def test_replay_exhaustion_never_calls_the_api(drv, tmp_path):
    run = _run_dir(tmp_path)
    bad = _resp({"7": "mug"}, [{"name": "g", "active": 7, "passive": None, "parts": {"7": ["h"]}}])
    lp = tmp_path / "l.json"
    lp.write_text(json.dumps({"model": "m", "raw_responses": [bad]}))
    with pytest.raises(drv.PlanRefusal, match="replay exhausted"):
        drv.plan_scene(run, "pick up the mug", replay=str(lp))


# ------------------------------------------------------------------ refusals

def test_unregistered_mark_refuses_but_keeps_the_record(drv, tmp_path, capsys):
    run = _run_dir(tmp_path, registered=())
    code, art, rep = drv.plan_scene(run, "pick up the mug", transport=Canned(GRASP))
    assert code == 2 and rep["next"] == []
    assert "registered[mug]" in art["gate"]["blocking"]
    by = {c["check"]: c for c in art["checks"]}
    assert "--register 2" in by["registered[mug]"]["detail"]
    assert (run / "stage_plan" / "plan.json").is_file()        # inspectable anyway


def test_object_without_parts_refuses(drv, tmp_path):
    """The quiet path: a used object with no parts would ground nothing and
    pass every later gate. Refused here (and in make_part_masks/ground_asset)."""
    run = _run_dir(tmp_path, registered=("m1", "m2"))
    plan = _resp({"2": "mug", "1": "box"},
                 [{"name": "grasp mug", "active": 2, "passive": None, "parts": {"2": ["handle"]}},
                  {"name": "put mug by box", "active": 2, "passive": 1, "parts": {"2": ["base"]}}])
    code, art, _ = drv.plan_scene(run, "put the mug next to the box", transport=Canned(plan))
    assert code == 2 and "parts[box]" in art["gate"]["blocking"]
    assert art["object_parts"] == {"mug": ["handle", "base"], "box": []}


def test_structure_is_recorded_not_enforced(drv, tmp_path):
    run = _run_dir(tmp_path, registered=("m1", "m2"))
    plan = _resp({"2": "mug", "1": "box"},
                 [{"name": "push mug to box", "active": 2, "passive": 1,
                   "parts": {"2": ["body"], "1": ["side"]}}])
    code, art, _ = drv.plan_scene(run, "push the mug to the box", transport=Canned(plan))
    by = {c["check"]: c for c in art["checks"]}
    assert by["structure[grasp<interaction]"]["pass"] is False
    assert "structure[grasp<interaction]" not in art["gate"]["blocking"]
    assert code == 0
    # m1 has no canonical asset yet: its commands start with canonicalize_asset
    _, _, rep = drv.plan_scene(run, "push the mug to the box", transport=Canned(plan))
    assert any("canonicalize_asset.py" in l and "--key m1" in l for l in rep["next"])


def test_input_refusals(drv, tmp_path):
    run = _run_dir(tmp_path)
    with pytest.raises(drv.PlanRefusal, match="--task is empty"):
        drv.plan_scene(run, "  ", transport=Canned(GRASP))
    (run / "marks" / "marks.json").unlink()
    with pytest.raises(drv.PlanRefusal, match="marks.json missing"):
        drv.plan_scene(run, "pick up the mug", transport=Canned(GRASP))


def test_cli(drv, tmp_path, capsys, monkeypatch):
    run = _run_dir(tmp_path)
    monkeypatch.setattr(drv.vlm_compat, "live_transport", Canned(GRASP))
    assert drv.main(["--run", str(run), "--task", "pick up the mug by its handle"]) == 0
    out = capsys.readouterr().out
    assert "stage 0: 'grasp the mug by its handle'" in out and "make_part_masks.py" in out
    assert "FAIL roles[deferred]" in out


# ------------------------------------------------------------------ consumed, not shaped

def test_load_bindings_refuses_it_for_exactly_the_deferred_roles(drv, tmp_path):
    """The shared-layout claim, pinned by consumption: plan_stages' own
    reader (what calls #2/#3 will use, via stageplan_compat) loads the
    artifact's structure and refuses it naming roles[deferred] -- and only
    that. Without the deferred check it would ACCEPT empty roles and fail
    later as a KeyError inside emit_constraints."""
    run = _run_dir(tmp_path)
    drv.plan_scene(run, "pick up the mug by its handle", transport=Canned(GRASP))
    path = run / "stage_plan" / "plan.json"
    code = ("import sys\n"
            "from manip_bridge.stageplan_compat import load_bindings\n"
            "try:\n"
            f"    load_bindings({str(path)!r})\n"
            "    print('ACCEPTED')\n"
            "except SystemExit as e:\n"
            "    print('REFUSED', e)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.startswith("REFUSED") and "['roles[deferred]']" in r.stdout, r.stdout
    # the drift the check guards against: without it, empty roles load silently
    doc = json.loads(path.read_text())
    doc["gate"]["blocking"] = []
    path.write_text(json.dumps(doc))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.stdout.startswith("ACCEPTED"), r.stdout + r.stderr[-1000:]


# ------------------------------------------------------------------ compats

def test_compat_contracts_and_import_weight():
    code = ("import sys\n"
            "import manip_bridge.vlm_compat as v\n"
            "assert set(v.__all__) >= {'Client', 'Vocabulary', 'StagePlan', 'VLMError', "
            "'MODEL', 'live_transport', 'provenance'}\n"
            "assert v.SOURCES[0].endswith('manip_sim/vlm.py')\n"
            "assert 'mujoco' not in sys.modules\n"
            "import manip_bridge.stageplan_compat as s\n"
            "assert set(s.__all__) == {'Bindings', 'load_bindings', 'provenance', 'SOURCES'}\n"
            "assert s.SOURCES[0].endswith('scripts/plan_stages.py')\n"
            "assert 'mujoco' not in sys.modules\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
