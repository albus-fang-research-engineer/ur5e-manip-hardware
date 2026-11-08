"""Replay regression for the recorded 11/05 hardware run (20261105_035449).

The VLM responses recorded on that run are fed back through the CURRENT
drivers (outputs/runs/plan_scene.py, select_scene.py, emit_scene.py) and
what the pipeline derives from them is compared with goldens:

  call #1  plan parsing: plan sha, objects, stage list, per-object parts,
           and the --grounding pass (contract names, verify checks, role
           bindings, gate)
  call #2  per role: object, candidate id -> resolved point name, canonical
           and body coordinates, pixel on the real frame, axis source; the
           per-object menus; selections.json as written
  call #3  per stage: the parsed emissions, the compiled TSR pair at full
           precision (Bw, T0_w, Tw_e, subgoal nominal) captured from
           compile_stage, the artifact's compiled records (attempts,
           rejections, accounting), the w frames the framed arm draws, and
           the geometry report (the teapot's named points relative to w at
           the subgoal nominal)

The three calls run as one chain, like the run: call #2 reads the plan.json
the call-#1 replay just wrote, call #3 the selections the call-#2 replay
wrote. No API call is made anywhere: every client gets a ReplayTransport
over a recorded log (a replay that asks for more responses than were
recorded fails instead of calling out).

Free and host-only: no GL, no container, no meshes.
  - recorded inputs are the small JSON files under test/fixtures/run_20261105/
    (byte copies of the run's files; the run dir itself is gitignored), copied
    to a tmp run dir whose container paths (/data/runs/<stamp>) are rewritten
    to the tmp dir
  - images the drivers only READ into prompts (the marked image, the call-#2
    views, the call-#3 frames) are tiny placeholders: a replayed response does
    not depend on its prompt. rgb.png is a placeholder of the recorded size
    (640x480), so pixels and in_image are real; the object masks are not
    tracked, so in_outline is not covered (None here, True on the run)
  - call #2's renderer is the RECORDED manifest (marks, ids, parts filter) with
    placeholder views, so select_scene's menu gate still compares the recorded
    marks with the menu the current code rebuilds from candidates.json
  - call #3's renderer keeps the sim's real compute_stage_frames (pure
    geometry, reached through emit_compat) and stubs only scene_extent (it
    needs the visual meshes) and render_role (MuJoCo); the stub returns the
    record render_role builds, so the prompt's frames legend is built as on
    the run

Goldens live in test/golden/run_20261105/call{1,2,3}.json. Regenerate them
(after a change that is MEANT to alter the derivation) with

    UPDATE_REPLAY_GOLDENS=1 python -m pytest -q test/test_replay_run_20261105.py

test_goldens_agree_with_recording then checks the new goldens against what the
run itself wrote, so a regeneration cannot quietly move away from the record.

Needs the sim checkout at $SIM_DIR or ../ur5e-manip-sim and Python >= 3.10
(the drivers and the sim evaluate PEP 604 / 585 annotations at import); skips
without them. Call #3 also needs mujoco importable (no GL is used).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    pytest.skip("the drivers and the sim need Python >= 3.10 (PEP 604 annotations "
                "evaluated at import); run with the manip-tsr env", allow_module_level=True)
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
RUNS = REPO_ROOT / "outputs" / "runs"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_20261105"
GOLDEN = Path(__file__).resolve().parent / "golden" / "run_20261105"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = all((SIM_DIR / f).is_file() for f in (
    "scripts/plan_stages.py", "scripts/emit_constraints.py", "scripts/render_stage_frames.py",
    "manip_sim/selection.py", "tasks/pour_tea.json"))
pytestmark = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("trimesh")                       # check_symbols, via select_scene

STAMP = "20261105_035449"
CONTAINER_RUN = f"/data/runs/{STAMP}"                # the run's own paths, in the container
OBJECTS = {"teapot": 2, "mug": 1}                    # the run's --object NAME=MARK
RGB_SIZE = (640, 480)                                # the run's rgb.png (not tracked)
SCENE_RADIUS = 0.25                                  # stub scene_extent: drawn lengths only
TOL = 1e-6
UPDATE = os.environ.get("UPDATE_REPLAY_GOLDENS") == "1"


# ------------------------------------------------------------------ helpers

def _load(name):
    spec = importlib.util.spec_from_file_location(name, RUNS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _png(path: Path, size=(8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL_Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(path)


def _jsonable(x):
    """numpy -> plain JSON types (floats kept at full precision)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _diff(got, want, path="$", tol=TOL) -> list[str]:
    """Every place `got` departs from `want`: floats within `tol` (equal
    infinities match), everything else exactly."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f"{path}: missing key {k!r}" for k in want if k not in got]
        out += [f"{path}: unexpected key {k!r}" for k in got if k not in want]
        for k in want:
            if k in got:
                out += _diff(got[k], want[k], f"{path}.{k}", tol)
        return out
    if isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            return [f"{path}: length {len(got)} != golden {len(want)}"]
        return [d for i, (g, w) in enumerate(zip(got, want)) for d in _diff(g, w, f"{path}[{i}]", tol)]
    num = (int, float)
    if (isinstance(want, num) and isinstance(got, num)
            and not isinstance(want, bool) and not isinstance(got, bool)
            and (isinstance(want, float) or isinstance(got, float))):
        ok = math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol) or \
            (math.isnan(float(got)) and math.isnan(float(want)))
        return [] if ok else [f"{path}: {got!r} != golden {want!r} (tol {tol})"]
    if type(got) is not type(want) or got != want:
        return [f"{path}: {got!r} != golden {want!r}"]
    return []


def _check_golden(name: str, got: dict):
    path = GOLDEN / f"{name}.json"
    got = _jsonable(got)
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"UPDATE_REPLAY_GOLDENS=1: rewrote {path}")
    if not path.is_file():
        pytest.fail(f"{path} missing -- generate it with UPDATE_REPLAY_GOLDENS=1")
    want = json.loads(path.read_text())
    diffs = _diff(got, want)
    assert not diffs, (f"{name}: the replay of run {STAMP} derives something other than "
                       f"{path.relative_to(REPO_ROOT)} ({len(diffs)} difference(s)):\n  "
                       + "\n  ".join(diffs[:40]))


def _golden(name: str) -> dict:
    path = GOLDEN / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{path} not generated yet")
    return json.loads(path.read_text())


def _recorded(rel: str) -> dict:
    return json.loads((FIXTURE / rel).read_text())


def _no_tmp_paths(doc, tmp: Path):
    """Goldens must not depend on where the tmp run dir is."""
    text = json.dumps(_jsonable(doc))
    assert str(tmp) not in text, "a pinned value carries a tmp path"


# ------------------------------------------------------------------ the run

def _rewrite_paths(x, new_root: str):
    """Every string that IS the run's container path, or starts with it
    followed by '/', re-rooted at `new_root`; everything else unchanged."""
    if isinstance(x, dict):
        return {k: _rewrite_paths(v, new_root) for k, v in x.items()}
    if isinstance(x, list):
        return [_rewrite_paths(v, new_root) for v in x]
    if isinstance(x, str) and (x == CONTAINER_RUN or x.startswith(CONTAINER_RUN + "/")):
        return new_root + x[len(CONTAINER_RUN):]
    return x


def _tmp_run(root: Path) -> Path:
    """The fixture as a run dir, container paths rewritten to it, plus the
    placeholder images the drivers read into prompts."""
    run = root / STAMP
    shutil.copytree(FIXTURE, run)
    for f in run.rglob("*.json"):
        f.write_text(json.dumps(_rewrite_paths(json.loads(f.read_text()), str(run)), indent=2)
                     + "\n")
    _png(run / "marks" / "marked.png")
    _png(run / "rgb.png", RGB_SIZE)
    return run


def _call2_renderer(run: Path):
    """select_scene's render(name, obj_dir, parts, out_dir): the RECORDED
    manifest, its views replaced by placeholders under out_dir."""
    def render(name, obj_dir, parts, out_dir):
        man = json.loads((run / "call2" / "vlm" / name / "manifest.json").read_text())
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        for v in man["views"].values():
            p = d / Path(v["path"]).name
            _png(p)
            v["path"] = str(p)
        mpath = d / "manifest.json"
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        return man, mpath
    return render


def _call3_renderer(ec, records: dict):
    """(compute_stage_frames, render_role, scene_extent) for emit_scene:
    sim's real compute_stage_frames (no GL); scene_extent and render_role
    stubbed. The stub returns the record sim's render_role builds (minus the
    per-view end_on facts, which come from the images)."""
    compute_stage_frames = ec.stage_frames_renderer()[0]

    def scene_extent(scene, poses):
        return np.zeros(3), SCENE_RADIUS

    def render_role(scene, role, sf, poses, out_dir, entry_from):
        views = {}
        for v in ("iso", "iso-opp", "top"):
            p = Path(out_dir) / role / f"{v}.png"
            _png(p)
            views[v] = str(p)
        rec = {
            "stage": sf.stage, "name": sf.name, "active": sf.active, "passive": sf.passive,
            "w": {"object": sf.w_object, "point_body": sf.w_point_body,
                  "point": next(t["point"] for t in sf.triads if t["role_in_stage"] == "w_owner"),
                  "candidate_id": sf.w_candidate_id, "x_route": sf.w_x_route,
                  "fallback": sf.w_fallback, "merged": sf.merged},
            "triads": sf.triads, "end_on": {}, "views": views,
            "poses": {n: np.round(np.asarray(T, float), 6).tolist() for n, T in poses.items()},
            "entry_from": entry_from,
        }
        w = next(f for f in sf.frames if f.label == "w")
        records[role] = {**{k: v for k, v in rec.items() if k not in ("views", "end_on")},
                         "w_origin_world": w.origin, "w_axes_world": w.axes}
        return rec

    return compute_stage_frames, render_role, scene_extent


def _spy_gates(ec, mp, calls: list):
    """Wrap the compile gate emit_scene runs per attempt (compile_stage, then
    check_pair_consistency, then check_stage_seam) and record each attempt's
    outcome: `attempt` counts per stage in call order, `outcome` is
    "accepted" only once every gate the driver ran on that attempt passed,
    else the gate that rejected it ("compile_stage", ...) with its text. The
    pair and seam checks are matched to their attempt by the compiled stage
    object they are handed, not by stage name."""
    compile_stage, pair, seam = ec.compile_stage, ec.check_pair_consistency, ec.check_stage_seam
    by_cs = {}                                       # id(compiled stage) -> its record

    def _record(em, **fields):
        n = sum(1 for c in calls if c["stage"] == em.name)
        calls.append({"stage": em.name, "attempt": n, **fields})
        return calls[-1]

    def spy_compile(em, *a, **k):
        try:
            cs = compile_stage(em, *a, **k)
        except ec.CompileError as e:
            _record(em, outcome="compile_stage", error=e.text())
            raise
        by_cs[id(cs)] = _record(
            em, outcome="accepted", error=None, x_source=cs.x_source,
            w_point=cs.w_frame.point,
            **{f"{t}_{a}": getattr(getattr(cs, t), a)
               for t in ("path", "subgoal") for a in ("Bw", "T0_w", "Tw_e")},
            subgoal_nominal=cs.subgoal.nominal())
        return cs

    def _gated(fn, name, cs_arg):
        def spy(*a, **k):
            try:
                return fn(*a, **k)
            except ec.CompileError as e:
                rec = by_cs.get(id(a[cs_arg]))
                assert rec is not None, f"{name} ran on a stage compile_stage did not return"
                rec.update(outcome=name, error=e.text())
                raise
        return spy

    mp.setattr(ec, "compile_stage", spy_compile)
    mp.setattr(ec, "check_pair_consistency", _gated(pair, "check_pair_consistency", 0))
    mp.setattr(ec, "check_stage_seam", _gated(seam, "check_stage_seam", 1))


# Three chained module fixtures, one per call, so a refusal downstream (say,
# select_scene's menu gate) leaves the upstream comparisons reporting their
# own differences.

_DRIVERS = ("plan_scene", "select_scene", "emit_scene", "check_symbols")
_PACKAGES = ("manip_bridge", "manip_sim", "scripts")


def _forget_new_modules(before: set):
    """Drop from sys.modules what this module's chain imported: the drivers
    (emit_scene imports select_scene and check_symbols off outputs/runs) and
    anything under manip_bridge / manip_sim / scripts that was not loaded
    before. All manip_bridge submodules go, not only the compats: vlm_record
    holds the vlm_compat it imported, so keeping it would pair a later test's
    fresh vlm_compat with a stale VLMError. A surviving parent package also
    loses the attribute, since `from pkg import sub` reads that before
    sys.modules."""
    for name in sorted(set(sys.modules) - before, key=lambda n: -n.count(".")):
        if not (name in _DRIVERS or any(name == p or name.startswith(p + ".") for p in _PACKAGES)):
            continue
        mod = sys.modules.pop(name)
        parent, _, leaf = name.rpartition(".")
        if parent and getattr(sys.modules.get(parent), leaf, None) is mod:
            delattr(sys.modules[parent], leaf)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    paths = (str(BRIDGE), str(SIM_DIR), str(RUNS))
    before = set(sys.modules)
    for p in paths:
        sys.path.insert(0, p)
    try:
        from manip_bridge import emit_compat as ec
        from manip_bridge import selection_compat as sc
        es = _load("emit_scene")
        root = tmp_path_factory.mktemp("replay")
        yield {"tmp": root, "run": _tmp_run(root), "ec": ec, "sc": sc,
               "ps": _load("plan_scene"), "es": es,
               "ss": es.ss}                          # the select_scene emit_scene gates with
    finally:
        for p in paths:
            sys.path.remove(p)
        _forget_new_modules(before)


@pytest.fixture(scope="module")
def call1(env):
    """The --grounding pass over the recorded plan log, as on the run."""
    run = env["run"]
    log1 = run / "stage_plan" / "plan.log.json"
    task = json.loads(log1.read_text())["task"]
    code1, art1, rep1 = env["ps"].plan_scene(
        run, task, replay=str(log1), out=str(run / "stage_plan_replay"),
        grounding=str(run / "grounding_vlm"), task_spec=str(SIM_DIR / "tasks" / "pour_tea.json"),
        objects=OBJECTS)
    hw = art1["hardware"]
    return run / "stage_plan_replay" / "plan.json", {
        "exit": code1, "task": task, "model": art1["model"], "plan_sha": hw["plan_sha"],
        "plan": art1["plan"], "plan_grounded": art1["plan_grounded"],
        "parts_dropped": art1["parts_dropped"], "object_parts": art1["object_parts"],
        "object_parts_requested": art1["object_parts_requested"],
        "mark_key": hw["mark_key"], "registered": hw["registered"],
        "object_names": hw["object_names"], "task_spec": art1["task_spec"],
        "roles": art1["roles"], "checks": art1["checks"], "gate": art1["gate"],
        "blocking_policy": hw["blocking_policy"], "refused": rep1["refused"],
    }


@pytest.fixture(scope="module")
def call2(env, call1):
    """The recorded selection log over the plan call1 just wrote."""
    run, ec, sc = env["run"], env["ec"], env["sc"]
    plan_path, _ = call1
    out2 = run / "call2_replay"
    code2, _, rep2 = env["ss"].select_scene(run, plan=str(plan_path),
                                            replay=str(run / "call2" / "selections.log.json"),
                                            out=str(out2), render=_call2_renderer(run))
    log2 = json.loads((out2 / "selections.log.json").read_text())
    doc1 = json.loads(plan_path.read_text())
    canon = {n: Path(o["canon_dir"]) for n, o in doc1["hardware"]["grounding"]["objects"].items()}
    resolved = {}
    for role, r in rep2["resolved"].items():
        sym = {r["object"]: sc.load_symbols(canon[r["object"]])}
        tok = ec.point_token(r["object"], sym, r["xyz_canonical"])
        resolved[role] = {
            "object": r["object"], "key": r["key"], "candidate_id": r["candidate_id"],
            "menu_tag": r["menu_tag"], "point": tok.split(".", 1)[1] if tok else None,
            **{k: r[k] for k in ("xyz_canonical", "xyz_body", "pixel", "in_image",
                                 "in_outline", "axis_source", "frame_comment")}}
    return out2 / "selections.json", {
        "exit": code2, "plan_sha": log2["plan_sha"], "model": log2["model"],
        "parts": log2["parts"], "menus": log2["menus"], "stages": log2["stages"],
        "selections": json.loads((out2 / "selections.json").read_text()),
        "resolved": resolved, "failure": log2["failure"],
        "client_logs": [{k: e[k] for k in ("touchpoint", "attempts", "rejections")}
                        for e in log2["client_logs"]],
    }


@pytest.fixture(scope="module")
def call3(env, call1, call2):
    """The stand-in emission log over the selections call2 just wrote."""
    pytest.importorskip("mujoco", reason="call #3 reaches compute_stage_frames through sim's "
                        "render_stage_frames, which imports mujoco (no GL is used)")
    run, ec = env["run"], env["ec"]
    (plan_path, _), (sel_path, _) = call1, call2
    out3 = run / "call3_replay"
    frames, compiled = {}, []
    with pytest.MonkeyPatch.context() as mp:
        _spy_gates(ec, mp, compiled)
        code3, rep3 = env["es"].emit_scene(
            run, plan=str(plan_path), selections=str(sel_path), out=str(out3),
            arm="framed-chained", replay=str(run / "call3_simreplay" / "stand_in.log.json"),
            renderer=_call3_renderer(ec, frames))
    art3 = json.loads((out3 / "emissions.json").read_text())
    log3 = json.loads((out3 / "emissions.log.json").read_text())
    return {
        "exit": code3, "failed": rep3["failed"], "requests": rep3["requests"],
        "flags": rep3["flags"], "roles": art3["roles"], "arm": art3["arm"],
        "stages_reached": art3["stages_reached"], "stages_total": art3["stages_total"],
        "plan_sha": art3["hardware"]["plan_sha"],
        "measured_poses": art3["hardware"]["measured_poses"],
        "emissions": art3["emissions"], "compiled": art3["compiled"],
        "compiled_full": compiled, "frames": frames,
        "geometry": art3["hardware"]["geometry"],
        "client_logs": [{k: e[k] for k in ("touchpoint", "attempts", "rejections", "views",
                                           "legend", "entry")}
                        for e in log3["client_logs"]],
    }


# ------------------------------------------------------------------ tests

def test_call1_plan_and_bindings(env, call1):
    c1 = call1[1]
    _no_tmp_paths(c1, env["tmp"])
    assert c1["exit"] == 0 and c1["refused"] is None
    _check_golden("call1", c1)


def test_call2_resolution(env, call2):
    c2 = call2[1]
    _no_tmp_paths(c2, env["tmp"])
    assert c2["exit"] == 0 and c2["failure"] is None
    _check_golden("call2", c2)


def test_call3_compile_and_geometry(env, call3):
    _no_tmp_paths(call3, env["tmp"])
    _check_golden("call3", call3)


def test_goldens_agree_with_recording():
    """The goldens reproduce what the run itself wrote (where it wrote it),
    so a regenerated golden cannot drift from the record unnoticed. Floats
    the drivers round before writing are compared at that rounding."""
    g1, g2, g3 = _golden("call1"), _golden("call2"), _golden("call3")

    plan = _recorded("stage_plan/plan.json")
    assert g1["plan_sha"] == plan["hardware"]["plan_sha"] == "476f07ea"
    for k in ("plan", "plan_grounded", "object_parts", "roles", "gate"):
        assert not _diff(g1[k], plan[k]), k
    assert g1["object_names"] == plan["hardware"]["object_names"]
    assert [s["name"] for s in g1["plan"]["stages"]] == [s["name"] for s in plan["plan"]["stages"]]
    assert len(g1["plan"]["stages"]) == 4

    sel_log = _recorded("call2/selections.log.json")
    assert g2["selections"] == _recorded("call2/selections.json")
    assert g2["menus"] == sel_log["menus"] and g2["parts"] == sel_log["parts"]
    for role, want in sel_log["resolved"].items():
        got = g2["resolved"][role]
        for k in ("object", "key", "candidate_id", "menu_tag", "xyz_canonical", "xyz_body",
                  "pixel", "in_image", "axis_source", "frame_comment"):
            assert not _diff(got[k], want[k]), (role, k)
    picks = {r: (v["object"], v["candidate_id"], v["point"]) for r, v in g2["resolved"].items()}
    assert picks == {"grasp": ("teapot", 2, "handle_center"),
                     "transport_active": ("teapot", 2, "handle_center"),
                     "pour": ("teapot", 4, "spout_center"),
                     "transport_passive": ("mug", 1, "rim_center")}

    em = _recorded("call3_simreplay/emissions.json")
    assert g3["emissions"] == em["emissions"]
    assert not _diff(g3["compiled"], em["compiled"])
    assert not _diff(g3["geometry"], em["hardware"]["geometry"])
    assert not _diff(g3["measured_poses"], em["hardware"]["measured_poses"])
    # the per-attempt records against the recorded compiled list, by attempt index
    full: dict[str, list] = {}
    for c in g3["compiled_full"]:
        full.setdefault(c["stage"], []).append(c)
    recorded_stages = [r["stage"] for r in em["compiled"]]
    assert list(full) == recorded_stages, (
        f"compiled_full stages {list(full)} != recorded compiled stages {recorded_stages}")
    for rec in em["compiled"]:
        st, atts = rec["stage"], full[rec["stage"]]
        assert [c["attempt"] for c in atts] == list(range(len(atts))), (st, "attempt indices")
        assert len(atts) == rec["attempts"], (
            f"{st}: {len(atts)} compile attempt(s) captured, the record says {rec['attempts']}")
        rejected = [c for c in atts if c["outcome"] != "accepted"]
        assert [c["error"].split(":")[0] for c in rejected] == rec["rejection_slots"], (
            f"{st}: rejected attempts' slots differ from the record's rejection_slots")
        if not rec["grounded"]:
            assert rec["Bw"] is None, f"{st}: ungrounded in the record but carries Bw"
            assert not any(c["outcome"] == "accepted" for c in atts), (
                f"{st}: ungrounded in the record, but an attempt passed every gate here")
            assert [c["error"] for c in rejected] == (rec["error"] or {}).get("rejections", []), (
                f"{st}: rejection texts differ from the record's")
            continue
        assert rec["Bw"] is not None, f"{st}: grounded in the record but has no Bw"
        last = atts[-1]
        assert last["outcome"] == "accepted" and len(rejected) == len(atts) - 1, (
            f"{st}: grounded in the record, so only its last attempt may pass the gates; "
            f"outcomes here {[c['outcome'] for c in atts]}")
        for k in ("path", "subgoal"):                # full precision agrees with the rounding
            assert not _diff(last[f"{k}_Bw"], rec["Bw"][k], tol=5e-5 + 1e-12), (st, k)
    assert g3["stages_reached"] == em["stages_reached"]
    assert g3["roles"] == em["roles"]
