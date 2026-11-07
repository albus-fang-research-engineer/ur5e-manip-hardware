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
    # phrases come before masks, and the masks command consumes them
    cmds = rep["next"]
    i_ph = next(i for i, l in enumerate(cmds) if "part_phrases.py" in l)
    i_mk = next(i for i, l in enumerate(cmds) if "make_part_masks.py" in l)
    phrases = f"masks_vlm_{art['hardware']['plan_sha']}/mug/phrases.json"
    assert i_ph < i_mk and phrases in cmds[i_ph] and f"--phrases" in cmds[i_mk]
    assert phrases in cmds[i_mk] and "--parts handle" in cmds[i_ph]


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
            "assert set(s.__all__) == {'Bindings', 'TaskSpec', 'DEFAULT_TASK_SPEC', 'bind_roles', "
            "'blocking', 'contract_changed', 'load_bindings', 'load_plan', 'load_pool', 'load_task_spec', "
            "'prune_parts', 'verify', 'provenance', 'SOURCES'}\n"
            "assert s.SOURCES[0].endswith('scripts/plan_stages.py')\n"
            "assert s.SOURCES[1].endswith('manip_sim/selection.py')\n"
            "assert s.DEFAULT_TASK_SPEC.name == 'pour_tea.json'\n"
            "assert s.load_task_spec(s.DEFAULT_TASK_SPEC).name == 'pour_tea'\n"
            "assert 'mujoco' not in sys.modules\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


# ------------------------------------------------------------------ after grounding

HANDLE_BODY = _resp({"2": "mug"}, [{"name": "grasp mug by handle", "active": 2, "passive": None,
                                    "parts": {"2": ["handle", "body"]}}])
MUG_GRASP = {"name": "mug_grasp", "required_tags": {"mug": ["handle"]},
             "roles": {"grasp": {"active": "mug", "passive": None, "tags": {"mug": ["handle"]}}}}
POUR_TEA = SIM_DIR / "tasks" / "pour_tea.json"


def mug_contract(tmp):
    """A grasp-only contract in sim's format (sim has none for this task)."""
    f = tmp / "mug_grasp.json"
    f.write_text(json.dumps(MUG_GRASP))
    return str(f)


def _plan_and_log(drv, run, tmp, response, task="pick up the mug by its handle"):
    """Live call #1 (canned) -> (artifact, replayable log path)."""
    _, art, _ = drv.plan_scene(run, task, transport=Canned(response))
    log = tmp / "plan.log.json"
    log.write_text((run / "stage_plan" / "plan.log.json").read_text())
    return art, log


def _grounded(run, key, name, plan_sha, requested, grounded, dropped=None, primitives=None,
              pool_parts=None, root=None):
    """What ground_asset leaves: <run>/grounding_vlm/<key>/frames.json (its
    provenance.canonical) and canon/<name>/candidates.json."""
    g = run / "grounding_vlm"
    canon = g / "canon" / name
    canon.mkdir(parents=True, exist_ok=True)
    cands = [{"id": i, "source": "part", "part": p, "xyz": [0.01 * i, 0.0, 0.0]}
             for i, p in enumerate(pool_parts if pool_parts is not None else grounded)]
    cands.append({"id": len(cands), "source": "fps", "xyz": [0.0, 0.05, 0.0]})
    (canon / "candidates.json").write_text(json.dumps({"candidates": cands}))
    (canon / "frames.json").write_text(json.dumps({"points": {}, "axes": {}}))
    (g / key).mkdir(parents=True, exist_ok=True)
    (g / key / "frames.json").write_text(json.dumps({"provenance": {"canonical": {
        "masks_root": str(run / (root or f"masks_vlm_{plan_sha}")),
        "parts_requested": list(requested), "parts_grounded": list(grounded),
        "parts_dropped": dropped or {},
        "primitives": primitives or {p: "line" for p in grounded},
        "canonical_frames": str(canon / "frames.json")}}}))
    return g


def _load_plan(path):
    code = ("from manip_bridge.stageplan_compat import load_plan\n"
            "try:\n"
            f"    b = load_plan({str(path)!r})\n"
            "    print('ACCEPTED', {r: (o, s.index, list(s.parts[o])) for r, (o, s) in b.roles.items()},"
            " sorted(b.entry_synthesized))\n"
            "except SystemExit as e:\n"
            "    print('REFUSED', e)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_body_dropped_binds_grasp_and_loads(drv, tmp_path):
    """M2's run: the model named handle,body; SAM3 found body nowhere. The
    pass prunes body, verifies against the grounded pool with the contract,
    binds grasp to the mug's stage, and plan_stages' own reader ACCEPTS it."""
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    sha = art["hardware"]["plan_sha"]
    g = _grounded(run, "m2", "mug", sha, ["handle", "body"], ["handle"],
                  dropped={"body": "no mask in any of 8 views of x/manifest.json"})
    spec = mug_contract(tmp_path)
    code, art2, rep = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                                     grounding=str(g), task_spec=spec)
    assert code == 0 and rep["refused"] is None, (art2["gate"], rep["refused"])
    assert json.loads((run / "stage_plan" / "plan.json").read_text()) == json.loads(json.dumps(art2))
    art2 = json.loads(json.dumps(art2))                          # compare as written (tuples -> lists)
    assert art2["roles"] == {"grasp": {"object": "mug", "stage": 0}}
    assert art2["plan_grounded"]["stages"][0]["parts"] == {"mug": ["handle"]}
    assert art2["plan"] == json.loads(json.dumps(art["plan"]))  # as emitted, untouched
    assert art2["parts_dropped"] == {"mug": ["body"]}
    assert art2["hardware"]["parts_dropped_reasons"]["mug"]["body"].startswith("no mask in any")
    assert art2["object_parts"] == {"mug": ["handle"]}
    assert art2["object_parts_requested"] == {"mug": ["handle", "body"]}
    by = {c["check"]: c for c in art2["checks"]}
    assert "roles[deferred]" not in by
    for c in ("identity[mug]", "grounds[stage0]", "menu[mug]", "coverage[mug]", "role[grasp]"):
        assert by[c]["pass"], by[c]
    assert art2["gate"] == {"defer_grounding": False, "blocking": []}
    assert art2["task_spec"] == "mug_grasp"
    hw = art2["hardware"]
    assert hw["task_spec_file"] == str(Path(spec).resolve()) and len(hw["task_spec_sha256"]) == 64
    assert hw["object_names"] == {}
    rec = hw["grounding"]["objects"]["mug"]
    assert rec["key"] == "m2" and len(rec["frames_sha256"]) == 64
    out = _load_plan(run / "stage_plan" / "plan.json")
    assert out.startswith("ACCEPTED") and "'grasp': ('mug', 0, ['handle'])" in out, out


def test_edited_contract_is_reported_not_refused(drv, tmp_path):
    """load_plan compares the contract file with the sha256 recorded at
    binding: an edit since then is a stderr WARNING naming both hashes, and
    the plan is still read (log, not gate) -- with the edited rules."""
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"], ["handle", "body"])
    spec = mug_contract(tmp_path)
    code, _, _ = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                                grounding=str(g), task_spec=spec)
    assert code == 0
    path = run / "stage_plan" / "plan.json"
    code = ("import sys\n"
            "from manip_bridge.stageplan_compat import load_plan\n"
            f"b = load_plan({str(path)!r})\n"
            "print('ACCEPTED', sorted(b.roles))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))

    def read():
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout, r.stderr

    out, err = read()
    assert out.startswith("ACCEPTED ['grasp']") and "WARNING" not in err
    Path(spec).write_text(Path(spec).read_text() + "\n")       # any byte changes the hash
    out, err = read()
    assert out.startswith("ACCEPTED ['grasp']")
    assert "task contract changed since binding" in err and "!= bound" in err


def test_missing_contract_file_is_a_reader_refusal(drv, tmp_path):
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"], ["handle", "body"])
    spec = mug_contract(tmp_path)
    assert drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                          grounding=str(g), task_spec=spec)[0] == 0
    Path(spec).unlink()
    out = _load_plan(run / "stage_plan" / "plan.json")
    assert out.startswith("REFUSED") and "missing here" in out


def test_handle_dropped_body_survives_now_blocks(drv, tmp_path):
    """The gap structural binding left open: the grasp stage keeps `body`,
    but the contract needs `handle` -- coverage[mug] and role[grasp] block."""
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"], ["body"],
                  dropped={"handle": "no mask"})
    code, art2, _ = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                                   grounding=str(g), task_spec=mug_contract(tmp_path))
    assert code == 2
    assert {"coverage[mug]", "role[grasp]"} <= set(art2["gate"]["blocking"])
    assert art2["roles"] == {"grasp": {"object": "mug", "stage": None}}
    out = _load_plan(run / "stage_plan" / "plan.json")
    assert out.startswith("REFUSED") and "role[grasp]" in out, out


def test_every_grasp_part_lost_is_fatal(drv, tmp_path):
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"], ["handle"],
                  dropped={"body": "no mask"}, primitives={"handle": "ungrounded"}, pool_parts=[])
    code, art2, _ = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                                   grounding=str(g), task_spec=mug_contract(tmp_path))
    assert code == 2 and "role[grasp]" in art2["gate"]["blocking"]
    assert art2["hardware"]["parts_dropped_reasons"]["mug"] == {
        "handle": "ungrounded by ground_object", "body": "no mask"}


def test_part_never_requested_counts_as_dropped(drv, tmp_path):
    """ground_asset run by hand with --parts handle: body was never asked for,
    so it is neither grounded nor in ground_asset's parts_dropped -- the pass
    still records it, it does not vanish."""
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle"], ["handle"])
    code, art2, _ = drv.plan_scene(run, "pick up the mug by its handle", replay=str(log),
                                   grounding=str(g), task_spec=mug_contract(tmp_path))
    assert code == 0
    assert art2["hardware"]["parts_dropped_reasons"] == {"mug": {"body": "not requested from ground_asset"}}


def test_grounding_from_another_plan_is_refused(drv, tmp_path):
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle"], ["handle"],
                  root="masks_canon")
    with pytest.raises(drv.PlanRefusal, match=r"'masks_canon', not 'masks_vlm_"):
        drv.plan_scene(run, "pick up the mug by its handle", replay=str(log), grounding=str(g),
                       task_spec=mug_contract(tmp_path))


def test_grounding_input_refusals(drv, tmp_path):
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = run / "grounding_vlm"
    spec = mug_contract(tmp_path)
    with pytest.raises(drv.PlanRefusal, match="--grounding needs --replay"):
        drv.plan_scene(run, "pick up the mug", transport=Canned(HANDLE_BODY), grounding=str(g))
    with pytest.raises(drv.PlanRefusal, match="no such file"):
        drv.plan_scene(run, "pick up the mug by its handle", replay=str(log), grounding=str(g),
                       task_spec=str(tmp_path / "nope.json"))
    with pytest.raises(drv.PlanRefusal, match="m2/frames.json not found -- run the ground_asset"):
        drv.plan_scene(run, "pick up the mug by its handle", replay=str(log), grounding=str(g),
                       task_spec=spec)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle"], ["handle"])
    fp = g / "m2" / "frames.json"
    doc = json.loads(fp.read_text())
    del doc["provenance"]["canonical"]["parts_grounded"]
    fp.write_text(json.dumps(doc))
    with pytest.raises(drv.PlanRefusal, match="predates ground_asset's parts_grounded"):
        drv.plan_scene(run, "pick up the mug by its handle", replay=str(log), grounding=str(g),
                       task_spec=spec)


# ---- pour_tea: sim's own contract file

def _pour_run(drv, tmp, stages, handles=("mug", "teapot")):
    """Two registered marks (1 = mug, 2 = teapot) under the model's handles,
    call #1 canned, both grounded with every part the plan named."""
    run = _run_dir(tmp, registered=("m1", "m2"))
    plan = _resp({"1": handles[0], "2": handles[1]}, stages)
    art, log = _plan_and_log(drv, run, tmp, plan, "pour tea into the mug")
    sha = art["hardware"]["plan_sha"]
    for mark, key in (("1", "m1"), ("2", "m2")):
        ps = art["object_parts"][handles[int(mark) - 1]]
        g = _grounded(run, key, key, sha, ps, ps)
    return run, log, g


GRASP_T = {"name": "grasp teapot", "active": 2, "passive": None, "parts": {"2": ["handle", "body"]}}
CARRY_T = {"name": "move spout over mug", "active": 2, "passive": 1,
           "parts": {"2": ["spout"], "1": ["rim"]}}
POUR_T = {"name": "pour tea into mug", "active": 2, "passive": 1,
          "parts": {"2": ["spout"], "1": ["rim", "opening"]}}


def test_pour_tea_binds_every_sim_role(drv, tmp_path):
    """grasp, carry, pour as sim's call #1 emits it: sim's tasks/pour_tea.json
    binds all four roles exactly as plan_stages would, and load_plan
    accepts them with the contract's stage restriction."""
    run, log, g = _pour_run(drv, tmp_path, [GRASP_T, CARRY_T, POUR_T])
    code, art, rep = drv.plan_scene(run, "pour tea into the mug", replay=str(log),
                                    grounding=str(g), task_spec=str(POUR_TEA))
    assert code == 0, (art["gate"], rep["refused"])
    assert art["task_spec"] == "pour_tea"
    assert art["roles"] == {"grasp": {"object": "teapot", "stage": 0},
                            "transport_active": {"object": "teapot", "stage": 1},
                            "pour": {"object": "teapot", "stage": 2},
                            "transport_passive": {"object": "mug", "stage": 1}}
    out = _load_plan(run / "stage_plan" / "plan.json")
    assert out.startswith("ACCEPTED"), out
    assert "'pour': ('teapot', 2, ['spout'])" in out and "'transport_passive': ('mug', 1, ['rim'])" in out


def test_model_handles_are_relabelled_to_contract_names(drv, tmp_path, capsys):
    """10/23's plan said teal_teapot / red_mug: unmapped, the contract's
    objects are missing and the gate blocks; --object teapot=2 --object
    mug=1 relabels them (sim's manifest relabel) and everything binds."""
    h = ("red_mug", "teal_teapot")
    run, log, g = _pour_run(drv, tmp_path, [GRASP_T, CARRY_T, POUR_T], handles=h)
    code, art, _ = drv.plan_scene(run, "pour tea into the mug", replay=str(log), grounding=str(g),
                                  task_spec=str(POUR_TEA))
    assert code == 2 and {"coverage[teapot]", "coverage[mug]", "role[grasp]"} <= set(
        art["gate"]["blocking"])
    assert drv.main(["--run", str(run), "--task", "pour tea into the mug", "--replay", str(log),
                     "--grounding", str(g), "--task-spec", str(POUR_TEA),
                     "--object", "teapot=2", "--object", "mug=1"]) == 0
    out = capsys.readouterr().out
    art = json.loads((run / "stage_plan" / "plan.json").read_text())
    assert art["hardware"]["object_names"] == {"red_mug": "mug", "teal_teapot": "teapot"}
    assert art["hardware"]["mark_key"] == {"red_mug": "m1", "teal_teapot": "m2"}  # emitted handles
    assert set(art["hardware"]["grounding"]["objects"]) == {"mug", "teapot"}
    assert art["hardware"]["grounding"]["objects"]["teapot"]["key"] == "m2"
    assert art["plan"]["objects"].keys() == {"red_mug", "teal_teapot"}           # as emitted
    assert art["plan_grounded"]["stages"][2]["passive"] == "mug"
    assert "role pour               -> teapot @ stage 2" in out and "load_plan" in out


def test_two_stage_pour_blocks_in_the_gate_itself(drv, tmp_path, capsys):
    """Hardware's 10/31 and 11/04 plans (Sonnet 4.6): grasp, then one "pour"
    stage, no transport. pour_tea needs grasp, transport, pour: transport_active
    takes the one stage and pour has none after it. That is bound[pour],
    blocking, in the artifact's own gate (plan_stages' reader refuses it too),
    exit 2."""
    run, log, g = _pour_run(drv, tmp_path, [GRASP_T, POUR_T])
    code, art, rep = drv.plan_scene(run, "pour tea into the mug", replay=str(log),
                                    grounding=str(g), task_spec=str(POUR_TEA))
    assert code == 2 and art["roles"]["pour"]["stage"] is None
    assert art["gate"]["blocking"] == ["bound[pour]"]
    by = {c["check"]: c for c in art["checks"]}
    assert "after transport_active" in by["bound[pour]"]["detail"] and by["bound[grasp]"]["pass"]
    out = _load_plan(run / "stage_plan" / "plan.json")
    assert out.startswith("REFUSED") and "bound[pour]" in out


def test_object_mapping_is_validated(drv, tmp_path):
    """--object typos and collisions refuse by name instead of surfacing as an
    unexplained coverage failure (or two objects merged onto one name)."""
    h = ("red_mug", "teal_teapot")
    run, log, g = _pour_run(drv, tmp_path, [GRASP_T, CARRY_T, POUR_T], handles=h)
    for objects, msg in (({"teapot": 3}, "no object on mark 3"),
                         ({"teapot": 2, "mug": 2}, "mark 2 given twice"),
                         ({"red_mug": 2}, "would both be named")):
        with pytest.raises(drv.PlanRefusal, match=msg):
            drv.plan_scene(run, "pour tea into the mug", replay=str(log), grounding=str(g),
                           task_spec=str(POUR_TEA), objects=objects)


def test_malformed_contract_is_a_refusal_not_a_traceback(drv, tmp_path):
    run, log, g = _pour_run(drv, tmp_path, [GRASP_T, CARRY_T, POUR_T])
    bad = json.loads(POUR_TEA.read_text())
    bad["roles"]["pour"]["after"] = "no_such_role"
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad))
    with pytest.raises(drv.PlanRefusal, match="no_such_role"):
        drv.plan_scene(run, "pour tea into the mug", replay=str(log), grounding=str(g),
                       task_spec=str(f))


def test_shipped_mug_grasp_contract_is_the_tested_one():
    f = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "tasks" / "mug_grasp.json"
    assert json.loads(f.read_text()) == MUG_GRASP


def test_grounding_cli(drv, tmp_path, capsys):
    run = _run_dir(tmp_path)
    art, log = _plan_and_log(drv, run, tmp_path, HANDLE_BODY)
    g = _grounded(run, "m2", "mug", art["hardware"]["plan_sha"], ["handle", "body"], ["handle"],
                  dropped={"body": "no mask in any of 8 views"})
    capsys.readouterr()
    assert drv.main(["--run", str(run), "--task", "pick up the mug by its handle",
                     "--replay", str(log), "--grounding", str(g),
                     "--task-spec", mug_contract(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "dropped mug.body: no mask in any of 8 views" in out
    assert "role grasp              -> mug @ stage 0" in out and "load_plan" in out
    assert "contract mug_grasp" in out and "make_part_masks" not in out
    with pytest.raises(SystemExit):
        drv.main(["--run", str(run), "--task", "x", "--object", "teapot"])
