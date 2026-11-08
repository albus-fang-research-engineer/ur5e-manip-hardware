"""outputs/runs/plan_stages_hw.py --method vlm, offline: the recorded 11/05
run (test/fixtures/run_20261105/) with call #3's sim stand-in emissions,
a fake cuRobo sidecar and stubbed collision spheres. No GL, no cuRobo, no
network, no API call.

Covered: the hand default is unchanged (CLI and plan notes); the carry is
compiled at the ACTUAL entry (FK(q_lift) . T_ee_canon, with T_ee_canon from
the grounding canonical frame); the stand-in pour's compile rejection
becomes a typed failure record; sidecar funnels and reasons map onto
FAILURE_TYPES; a failed pour re-plans the carry with the next seed.

Needs Python >= 3.10, the sim checkout at $SIM_DIR or ../ur5e-manip-sim and
manip_cbirrt (PYTHONPATH=../manip-cbirrt).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    pytest.skip("the drivers and the sim need Python >= 3.10; run with the manip-tsr env",
                allow_module_level=True)
np = pytest.importorskip("numpy")
pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
pytest.importorskip("trimesh")                   # check_symbols, via emit_scene -> select_scene

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
RUNS = REPO_ROOT / "outputs" / "runs"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_20261105"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
pytestmark = pytest.mark.skipif(not (SIM_DIR / "manip_sim" / "compile_tsr.py").is_file(),
                                reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
STAMP = "20261105_035449"
CONTAINER_RUN = f"/data/runs/{STAMP}"
CONTAINER_SIM = "/opt/manip-sim"
EMISSIONS = "call3_simreplay/emissions.json"

_DRIVERS = ("plan_stages_hw", "emit_scene", "select_scene", "check_symbols", "grasp_exec")
_PACKAGES = ("manip_bridge", "manip_sim", "scripts")


def _rewrite(x, root: str):
    """The run's container path -> the tmp run; the container's sim mount
    (/opt/manip-sim, where the plan's task contract lives) -> SIM_DIR."""
    if isinstance(x, dict):
        return {k: _rewrite(v, root) for k, v in x.items()}
    if isinstance(x, list):
        return [_rewrite(v, root) for v in x]
    for old, new in ((CONTAINER_RUN, root), (CONTAINER_SIM, str(SIM_DIR))):
        if isinstance(x, str) and (x == old or x.startswith(old + "/")):
            return new + x[len(old):]
    return x


def _tmp_run(root: Path) -> Path:
    """The fixture as a run dir. Only files that hold the run's container
    path are rewritten: the others stay byte-identical, so the emissions'
    recorded selections sha256 still matches selections.json."""
    run = root / STAMP
    shutil.copytree(FIXTURE, run)
    for f in run.rglob("*.json"):
        doc = json.loads(f.read_text())
        new = _rewrite(doc, str(run))
        if new != doc:
            f.write_text(json.dumps(new, indent=2) + "\n")
    # a rewritten frames.json has new bytes: re-record its sha256 where the
    # plan's grounding binding pins it (the content is the run's)
    plan = run / "stage_plan" / "plan.json"
    doc = json.loads(plan.read_text())
    for rec in doc["hardware"]["grounding"]["objects"].values():
        rec["frames_sha256"] = hashlib.sha256(Path(rec["frames"]).read_bytes()).hexdigest()
    plan.write_text(json.dumps(doc, indent=2) + "\n")
    return run


def _forget_new_modules(before: set):
    for name in sorted(set(sys.modules) - before, key=lambda n: -n.count(".")):
        if not (name in _DRIVERS or any(name == p or name.startswith(p + ".") for p in _PACKAGES)):
            continue
        mod = sys.modules.pop(name)
        parent, _, leaf = name.rpartition(".")
        if parent and getattr(sys.modules.get(parent), leaf, None) is mod:
            delattr(sys.modules[parent], leaf)


@pytest.fixture(scope="module")
def hw():
    paths = (str(BRIDGE), str(SIM_DIR), str(RUNS))
    before = set(sys.modules)
    for p in paths:
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location("plan_stages_hw", RUNS / "plan_stages_hw.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        yield m
    finally:
        for p in paths:
            sys.path.remove(p)
        _forget_new_modules(before)


class FakeSidecar:
    """plan_constrained replies: request number n (0-based) gets `fail[n]`
    as a failure reply if scripted, otherwise success with a short straight
    joint path from q_start. Every request is kept."""

    STEP = np.array([0.05, 0, 0, 0, 0, 0.05])
    FUNNEL = {"n_sampled": 40, "n_ik": 30, "n_collision_free": 20, "n_contained": 12}

    def __init__(self, dh_order, fail: dict):
        self.dh, self.fail, self.requests = list(dh_order), fail, []

    def call(self, msg, timeout_ms=None):
        n = len(self.requests)
        self.requests.append(msg)
        if n in self.fail:
            return {"ok": True, "success": False, "funnel": dict(self.FUNNEL), **self.fail[n]}
        q0 = np.asarray(msg["q_start"], float)
        P = q0 + np.linspace(0, 1, 5)[:, None] * self.STEP
        return {"ok": True, "success": True, "reason": "", "funnel": dict(self.FUNNEL),
                "joint_names": self.dh, "positions": P.tolist(), "goal_index": 0,
                "max_excess": 0.0, "tree_sizes": [3, 3]}


@pytest.fixture
def vlm_env(hw, tmp_path, monkeypatch):
    """A tmp run, the fake sidecar module, stub spheres, and a spy on
    compile_stage. Returns (run dir, setup(script) -> (sidecar, calls))."""
    from manip_bridge import emit_compat as ec
    run = _tmp_run(tmp_path)
    monkeypatch.setattr(hw.so, "object_spheres",
                        lambda mesh, cache, call, **k: np.array([[0.0, 0.0, 0.05, 0.03]]))
    calls = []
    real = ec.compile_stage

    def spy(em, symbols, poses, **k):
        calls.append({"role_name": em.name, "poses": {n: np.array(T) for n, T in poses.items()}})
        return real(em, symbols, poses, **k)
    monkeypatch.setattr(ec, "compile_stage", spy)

    def setup(fail):
        side = FakeSidecar(hw.um.DH_ORDER, fail)
        fake = types.ModuleType("manip_bridge.zmq_client")
        fake.SidecarClient = lambda addr, timeout_ms=None: side
        monkeypatch.setitem(sys.modules, "manip_bridge.zmq_client", fake)
        return side, calls
    return run, setup


def _main(hw, run: Path, *extra) -> int:
    return hw.main(["--run", str(run), "--method", "vlm", "--emissions", str(run / EMISSIONS),
                    "--scene", "live", "--tries", "2", *extra])


# ------------------------------------------------------------------ hand default

def test_hand_is_the_default_and_rejects_vlm_inputs(hw):
    with pytest.raises(SystemExit):
        hw.main(["--run", "x", "--emissions", "e.json"])          # vlm inputs without --method vlm
    with pytest.raises(SystemExit):
        hw.main(["--run", "x", "--method", "vlm"])                # vlm without --emissions
    with pytest.raises(SystemExit):                               # a hand-only knob under vlm
        hw.main(["--run", "x", "--method", "vlm", "--emissions", "e.json", "--tilt-deg", "110"])


def test_exec_plan_notes_unchanged_for_hand(hw):
    seg = lambda n, **kw: types.SimpleNamespace(name=n, Q=np.zeros((2, 6)), info={"length_m": 0.05},
                                                **kw)
    segs = [seg("open", gap_m=0.08), seg("approach"), seg("close", effort_n=20.0), seg("lift"),
            seg("release", gap_m=0.08)]
    reach = {"joint_names": ["a"], "positions": [[0]], "velocities": [[0]], "dt": 0.1}
    Q = np.zeros((2, 6))
    notes = {s["name"]: s.get("note") for s in hw.exec_plan_doc(reach, segs, Q, Q, 0.05)["steps"]}
    assert notes["transport"] == notes["pour"] == "CBiRRT, hand arm"
    vlm = hw.exec_plan_doc(reach, segs, Q, Q, 0.05, source="call #3 regions")["steps"]
    assert {s["name"]: s.get("note") for s in vlm}["pour"] == "CBiRRT, call #3 regions"


# ------------------------------------------------------------------ the vlm method

def test_carry_compiles_at_the_actual_entry_and_pour_rejection_is_typed(hw, vlm_env):
    run, setup = vlm_env
    side, calls = setup({})
    assert _main(hw, run) == 2                      # the stand-in pour never compiles
    rep = json.loads((run / "m4_vlm" / "report.json").read_text())

    # T_ee_canon from the grounding canonical frame; the carry's entry is FK(q_lift) . T_ee_canon
    ch = hw.um.chain()
    poses = {k: np.array(T) for k, T in rep["poses"]["T_base_canon"].items()}
    q_grasp, q_lift = np.array(rep["grasp"]["q_grasp"]), np.array(rep["grasp"]["q_lift"])
    T_ee_canon = np.linalg.inv(ch.fk(q_grasp)) @ poses["teapot"]
    np.testing.assert_allclose(rep["grasp"]["T_ee_canon"], T_ee_canon, atol=1e-9)
    carry = calls[0]
    assert carry["role_name"] == "lift and move teapot above mug"
    np.testing.assert_allclose(carry["poses"]["teapot"], ch.fk(q_lift) @ T_ee_canon, atol=1e-9)
    np.testing.assert_allclose(carry["poses"]["mug"], poses["mug"], atol=1e-12)
    # the pour is compiled at the carry's actual end
    pour = calls[1]
    q_end = np.array(side.requests[0]["q_start"]) + FakeSidecar.STEP
    np.testing.assert_allclose(pour["poses"]["teapot"], ch.fk(q_end) @ T_ee_canon, atol=1e-9)

    # the carry's entry is the same on every seed: compiled once
    assert [c["role_name"] for c in calls].count("lift and move teapot above mug") == 1
    # the carry planned (its compiled pair recorded); the pour is a typed compile record per seed
    assert rep["inputs"]["emit_gate_failed"] == ["tilt teapot to pour tea into mug"]
    assert len(rep["attempts"]) == 2
    st = rep["attempts"][0]["stages"]
    assert st[0]["outcome"] == "planned" and set(st[0]["compiled"]) == {"path", "subgoal"}
    assert st[1]["outcome"] == "compile_rejected"
    f = rep["failures"][0]
    assert f["phase"] == "compile" and f["stage"] == "pour" and f["slots"]
    assert f["type"] is None and f["routes_to"] == "call3_compile_retry"   # not an empty box
    assert rep["status"] == "failed" and not (run / "m4_vlm" / "plan.json").exists()


def test_failed_pour_replans_the_carry_with_the_next_seed(hw, vlm_env, monkeypatch):
    run, setup = vlm_env
    from manip_bridge import emit_compat as ec
    spy, carry_cs = ec.compile_stage, {}

    def compile_pour_as_carry(em, symbols, poses, **k):
        """The stand-in pour never compiles; reuse the carry's regions for
        it so the retry structure can be exercised."""
        if em.name.startswith("lift"):
            carry_cs["cs"] = spy(em, symbols, poses, **k)
            return carry_cs["cs"]
        return carry_cs["cs"]
    monkeypatch.setattr(ec, "compile_stage", compile_pour_as_carry)
    # request 0 = seed 0's carry, 1 = seed 0's pour (fails), 2-3 = seed 1's carry and pour
    setup({1: {"reason": "CBiRRT failed on 1 goal(s): no path in 5.0s"}})
    assert _main(hw, run) == 0
    rep = json.loads((run / "m4_vlm" / "report.json").read_text())
    assert [a["seed"] for a in rep["attempts"]] == [0, 1]
    assert [f["type"] for f in rep["failures"]] == ["planner_timeout"]
    assert rep["failures"][0]["stage"] == "pour" and rep["failures"][0]["atoms"]
    assert rep["status"] == "planned"
    stages = json.loads((run / "m4_vlm" / "stages.json").read_text())
    assert [s["role"] for s in stages["stages"]] == ["transport_active", "pour"]
    steps = json.loads((run / "m4_vlm" / "plan.json").read_text())["steps"]
    assert {s["name"]: s.get("note") for s in steps}["transport"] == "CBiRRT, call #3 regions"


# ------------------------------------------------------------------ typed failures

@pytest.mark.parametrize("funnel,reason,kind", [
    ({"n_ik": 0, "n_collision_free": 0, "n_contained": 0}, "no goal survived the funnel", "no_feasible_ik"),
    ({"n_ik": 25, "n_collision_free": 0, "n_contained": 0}, "no goal survived the funnel", "no_feasible_ik"),
    ({"n_ik": 25, "n_collision_free": 9, "n_contained": 0}, "no goal survived the funnel", "no_feasible_ik"),
    ({"n_ik": 0, "n_contained": 0, "stopped": "thin_intersection"}, "subgoal INTERSECT path is (nearly) empty",
     "empty_goal_intersection"),
    ({"n_ik": 9, "n_contained": 4}, "CBiRRT failed on 2 goal(s): start/goal projection failed",
     "projection_divergence"),
    ({"n_ik": 9, "n_contained": 4}, "CBiRRT failed on 3 goal(s): no path in 6.0s", "planner_timeout"),
    ({"n_ik": 9, "n_contained": 4}, "CBiRRT failed on 1 goal(s): start projection moved the object "
     "0.040 m (budget 0.03 m): the path TSR excludes the start pose", None),
    ({"n_ik": 9, "n_contained": 4}, "CBiRRT failed on 1 goal(s): goal in collision after projection",
     None),
    ({"n_ik": 9, "n_contained": 4}, "CBiRRT failed on 0 goal(s): no attempt fit in the time budget",
     "planner_timeout"),
    ({"n_ik": 0, "n_contained": 0}, "q_start in collision (scene pen -0.0100 m)", None),
    ({}, "sidecar gave no funnel", None),
])
def test_plan_failures_are_typed(hw, funnel, reason, kind):
    pair = types.SimpleNamespace(path=types.SimpleNamespace(name="p"),
                                 subgoal=types.SimpleNamespace(name="g"))
    f = hw.plan_failure("pour", "tilt", 3, {"funnel": funnel, "reason": reason}, pair)
    assert f["type"] == kind
    if kind:
        from manip_bridge import vlm_compat
        assert kind in vlm_compat.FAILURE_TYPES
    assert f["funnel"] == funnel and f["atoms"] == {"path": "p", "subgoal": "g"} and f["seed"] == 3


def test_compile_failures_are_typed(hw):
    from manip_bridge import emit_compat as ec
    empty = ec.CompileError("s.path.trans[0]", "the path's z band [+0.1, +0.2] does not meet the "
                                               "subgoal's [+0.3, +0.4]")
    other = ec.CompileError("s.path.rot[0]", "tilt axis is not on a w basis vector")
    assert hw.compile_failure("pour", "tilt", 0, empty)["type"] == "empty_goal_intersection"
    pair = ec.CompileError("s.path", "no subgoal sample lies on the path (64 tries): the path and "
                                     "subgoal regions do not overlap")
    assert hw.compile_failure("pour", "tilt", 0, pair)["routes_to"] == "call5"
    row = ec.CompileError("s.path.trans[1]", "row 2 intersection empty: existing [0.1, 0.2] vs "
                                             "[0.3, 0.4]")                  # one TSR's own terms clash
    assert hw.compile_failure("pour", "tilt", 0, row)["routes_to"] == "call3_compile_retry"
    f = hw.compile_failure("pour", "tilt", 0, ec.CompileError("a", "x", others=(other,)))
    assert f["type"] is None and f["slots"] == ["a", "s.path.rot[0]"]
