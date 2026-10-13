#!/usr/bin/env python3
"""VLM call #1 on a hardware run: the task + the run's marked image -> a
stage plan whose part names feed the grounding pipeline (M2's first step).

    <run>/marks/marked.png + marks.json   (run_scene's set-of-mark)
            |  Client.plan_stages (sim, via vlm_compat), mark-addressed
            v
    <run>/stage_plan/plan.json            stage plan + per-object parts + checks
    <run>/stage_plan/plan.log.json        raw model responses (replayable)

then prints, per object the plan uses, the commands that ground the parts
the MODEL named (canonicalize if needed -> make_part_masks -> ground_asset
-> check_symbols): the hand-typed `--parts handle,rim` of milestone 1 is
replaced by call #1's output.

Checks (recorded in the artifact; the parser has already enforced the
schema and that every mark id is on the image):
  registered[<obj>]  BLOCKING  the object's mark was registered by run_scene
                               (objects.m<id>.pose_on_final) -- an
                               unregistered mark has no mesh to ground on;
                               the fix is run_scene --register <id>
  parts[<obj>]       BLOCKING  the plan names >= 1 part for every object it
                               uses (an empty list would ground nothing and
                               pass every later gate quietly)
  structure[...]     recorded  grasp (active only) before interaction
                               (active + passive) -- the one contract rule
                               that survives without a task file; printed,
                               not enforced
  roles[deferred]    BLOCKING  for calls #2/#3 only: roles are bound by
                               the --grounding pass below, so plan_stages'
                               reader (load_bindings) refuses this artifact
                               by name until then. It does not stop
                               grounding.

After grounding: --replay LOG --grounding DIR (no API call) re-reads the
same plan and REWRITES plan.json against what was grounded -- sim
plan_stages' --replay --grounding, on hardware:

  gate        every used object's DIR/<key>/frames.json must exist and come
              from THIS plan (provenance.canonical.masks_root is
              masks_vlm_<plan sha8>) -- a grounding from another plan is
              refused, exit 2
  prune       a part the plan names but that did not ground (dropped by
              ground_asset, ungrounded by ground_object, or never
              requested) is removed from plan_grounded and recorded in
              parts_dropped (+ hardware.parts_dropped_reasons)
  verify      sim plan_stages.verify with STRUCTURAL_SPEC (no task
              contract): identity[], grounds[], menu[] against the
              grounded pool (DIR/canon/<name>/candidates.json), all blocking
  roles       STRUCTURAL binding (hardware has no task contract): grasp =
              the first active-only stage, its object = that stage's
              active. role[grasp] BLOCKS when that stage has no surviving
              part on its object -- the drop is then fatal. Other stages
              are recorded as unbound[stage<i>], never blocking
              (active+passive stages bind as mover roles at milestone 4).
              No bound role at all -> roles[none], blocking.

The known gap, stated rather than hidden: structure cannot tell which part
a role NEEDS, so "handle dropped, body survives" binds the grasp to the
body's stage. That case is caught where OmniManip catches it -- by looking:
call #2 sees "grasp mug by handle" with no handle marks, its overlay on the
real frame, and later the call-#4 critic -- not here. A task-requirements
file would close it at the cost of one hand-written file per task; that was
considered and declined (2026-10-12).

The artifact uses plan_stages' layout (plan, plan_grounded, object_parts,
roles, checks, gate) so calls #2/#3 extend these same files. Hardware has
no manifest, so plan_grounded is the plan as emitted (model handles), and
`hardware.mark_key` maps each handle to its run key (m<id>).

Record/replay: every raw response goes into plan.log.json; --replay LOG
feeds them back in order (no API call, same plan, same artifact), with the
logged model recorded. A replay that runs out of responses means the
prompt/parser asked for more than was recorded -- it fails, it never calls
the API.

Exit: 0 plan written, every used object registered with parts (roles stay
deferred), or with --grounding: roles bound and nothing blocks; 2 refused
-- the artifact is still written when a plan exists (and with --grounding
once the inputs pass the gate), so what was decided is always inspectable.

In the ros2 container (ANTHROPIC_API_KEY set; MANIP_VLM_MODEL optional):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/plan_scene.py --run /data/runs/<stamp> \\
        --task "pick up the mug by its handle"
    python3 /data/runs/plan_scene.py --run /data/runs/<stamp> \\
        --task "pick up the mug by its handle" --replay /data/runs/<stamp>/stage_plan/plan.log.json
    # after the printed make_part_masks / ground_asset / check_symbols:
    python3 /data/runs/plan_scene.py --run /data/runs/<stamp> \\
        --task "pick up the mug by its handle" --replay /data/runs/<stamp>/stage_plan/plan.log.json \\
        --grounding /data/runs/<stamp>/grounding_vlm
"""
import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from manip_bridge import marks_compat, vlm_compat
except ImportError as e:
    sys.exit(f"plan_scene: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

DEFERRED = "roles[deferred]"


class PlanRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


# ------------------------------------------------------------------ transports

class RecordingTransport:
    """Wraps the live transport so every raw response survives into the log
    (Client.logs keeps the accepted text and the rejections, not each raw
    attempt)."""

    def __init__(self, inner=None):
        self.inner = inner
        self.raw: list[str] = []

    def __call__(self, payload: dict) -> str:
        text = (self.inner or vlm_compat.live_transport)(payload)
        self.raw.append(text)
        return text


class ReplayTransport:
    """Returns recorded responses in order; never reaches the network."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.raw: list[str] = []

    def __call__(self, payload: dict) -> str:
        if len(self.raw) >= len(self.responses):
            raise vlm_compat.VLMError(
                f"replay exhausted: the log holds {len(self.responses)} response(s) and the "
                "client asked for another -- the prompt or parser changed since recording")
        text = self.responses[len(self.raw)]
        self.raw.append(text)
        return text


# ------------------------------------------------------------------ run inputs

def load_run(run: Path) -> dict:
    try:
        summary = json.loads((run / "summary.json").read_text())
    except (OSError, ValueError) as e:
        raise PlanRefusal(f"{run}/summary.json unreadable ({e})") from e
    mark_dir = run / "marks"
    if not (mark_dir / "marks.json").is_file():
        raise PlanRefusal(f"{mark_dir}/marks.json missing -- run_scene writes the mark set; "
                          "re-run it (or --from-run --reuse-marks)")
    markset = marks_compat.load_marks(mark_dir)
    if not markset.marked or not (mark_dir / markset.marked).is_file():
        raise PlanRefusal(f"{mark_dir}: no marked image in marks.json -- call #1 needs it")
    objects = summary.get("objects") or {}
    registered = {k: v for k, v in objects.items() if isinstance(v, dict) and v.get("pose_on_final")}
    assets = {}
    for cj in sorted((run / "assets_canon").glob("*/canonical.json")):
        try:
            key = (json.loads(cj.read_text()).get("inputs") or {}).get("key")
        except (OSError, ValueError):
            continue
        if key:
            assets[key] = cj.parent
    return {"markset": markset, "registered": registered, "assets": assets}


def object_parts(plan) -> dict[str, list[str]]:
    """Per used object, the order-preserving union of the parts the plan
    names for it -- every used object is a key, even with no parts."""
    acc = {h: [] for h in plan.objects}
    for s in plan.stages:
        for h in s.objects():
            acc.setdefault(h, [])
        for h, ps in s.parts.items():
            for p in ps:
                if p not in acc[h]:
                    acc[h].append(p)
    return acc


def run_checks(plan, parts: dict, registered: dict) -> tuple[list[dict], dict]:
    checks, mark_key = [], {}
    for h, ref in sorted(plan.objects.items()):
        key = f"m{int(ref.mark)}"                 # scene_marks.obj_key: the run's object key
        mark_key[h] = key
        ok = key in registered
        checks.append({"check": f"registered[{h}]", "pass": ok, "blocking": True,
                       "detail": f"mark {ref.mark} ({ref.label!r}) -> {key}"
                                 + ("" if ok else f": not registered in this run (no "
                                    f"objects.{key}.pose_on_final) -- run_scene "
                                    f"--register {ref.mark}")})
    for h, ps in sorted(parts.items()):
        bad = [p for p in ps if "," in p]
        ok = bool(ps) and not bad
        checks.append({"check": f"parts[{h}]", "pass": ok, "blocking": True,
                       "detail": (f"{ps}" if ok else
                                  "no part named for this object -- nothing to ground"
                                  if not ps else f"part names with ',' {bad} cannot be "
                                  "passed as --parts")})
    grasp = [s.index for s in plan.stages if s.passive is None]
    inter = [s.index for s in plan.stages if s.passive is not None]
    ok = not inter or (bool(grasp) and min(grasp) < min(inter))
    checks.append({"check": "structure[grasp<interaction]", "pass": ok, "blocking": False,
                   "detail": f"active-only stages {grasp}, active+passive stages {inter}"
                             + ("" if inter else " (no interaction stage: trivially satisfied)")})
    checks.append({"check": DEFERRED, "pass": False, "blocking": True,
                   "detail": "role binding on hardware is deferred to the step that brings "
                             "calls #2/#3 here; this artifact feeds grounding only"})
    return checks, mark_key


def next_commands(run: Path, plan_sha: str, parts: dict, mark_key: dict, info: dict) -> list[str]:
    """The grounding commands for each usable object, as the container runs them."""
    lines, root = [], run / f"masks_vlm_{plan_sha}"
    out = run / "grounding_vlm"
    for h, key in sorted(mark_key.items()):
        if key not in info["registered"] or not parts.get(h):
            continue
        ps = shlex.quote(",".join(parts[h]))
        asset = info["assets"].get(key)
        if asset is None:
            mesh = (info["registered"][key].get("pose_on_final") or {}).get("mesh", "")
            name = Path(mesh).stem.replace("final_mesh_", "") or key
            lines.append(f"python3 /data/runs/canonicalize_asset.py --run {run} --key {key} --check"
                         "   # add --canonical <glb> on a --final-mesh run")
            asset = run / "assets_canon" / name
        name = asset.name
        lines += [
            f"python3 /data/runs/make_part_masks.py {asset} --name {name} --parts {ps} "
            f"--masks-root {root} --timeout-ms 180000",
            f"python3 /data/runs/ground_asset.py --run {run} --name {name} --masks-root {root} "
            f"--parts {ps} --out {out}",
            f"python3 /data/runs/check_symbols.py {run} --key {key} "
            f"--frames {out}/{key}/frames.json --out-prefix {out}/symbols_{key}",
        ]
    return lines


# ------------------------------------------------------------------ after grounding

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_grounding(grounding: Path, plan_sha: str, parts: dict, mark_key: dict) -> dict:
    """Per used object (plan handle): what was grounded, from
    <grounding>/<key>/frames.json. Refuses (PlanRefusal) anything that is
    not this plan's grounding."""
    want = f"masks_vlm_{plan_sha}"
    out = {}
    for h, ps in sorted(parts.items()):
        key = mark_key[h]
        fp = grounding / key / "frames.json"
        if not fp.is_file():
            raise PlanRefusal(f"{fp} not found -- run the ground_asset command printed for {h} "
                              f"({key}) first (a ground_asset refusal removes it on purpose)")
        try:
            prov = (json.loads(fp.read_text()).get("provenance") or {}).get("canonical") or {}
        except (OSError, ValueError) as e:
            raise PlanRefusal(f"{fp} unreadable ({e})") from e
        if "parts_grounded" not in prov:
            raise PlanRefusal(f"{fp} predates ground_asset's parts_grounded record -- re-run "
                              "the printed ground_asset command")
        root = Path(prov.get("masks_root") or "")
        if root.name != want:
            raise PlanRefusal(f"{fp} was grounded from masks root {root.name or '?'!r}, not "
                              f"{want!r} (this plan) -- a grounding belongs to the plan that "
                              "named its parts; run this plan's printed commands")
        prims = prov.get("primitives") or {}
        grounded = [p for p in prov["parts_grounded"] if prims.get(p, "ungrounded") != "ungrounded"]
        why = prov.get("parts_dropped") or {}
        dropped = {}
        for p in ps:
            if p in grounded:
                continue
            dropped[p] = (why.get(p) or ("ungrounded by ground_object" if p in prov["parts_grounded"]
                                         else "not requested from ground_asset"))
        canon = Path(prov.get("canonical_frames") or "").parent
        out[h] = {"key": key, "frames": str(fp.resolve()), "frames_sha256": _sha256(fp),
                  "canon_dir": str(canon), "dropped": dropped}
    return out


def bind_structural(gplan) -> tuple[dict, list[dict]]:
    """roles + role checks, by stage structure only (no task contract)."""
    checks, roles = [], {}
    grasp = [s for s in gplan.stages if s.passive is None]
    if grasp:
        s = grasp[0]
        left = list(s.parts.get(s.active, ()))
        roles["grasp"] = {"object": s.active, "stage": s.index}
        checks.append({"check": "role[grasp]", "pass": bool(left), "blocking": True,
                       "detail": f"stage {s.index} {s.name!r} (active-only) -> {s.active}; "
                                 + (f"grounded parts {left}" if left else
                                    "NO grounded part left on it -- every part the plan "
                                    "named for the grasp was dropped, nothing to grasp")})
    for s in gplan.stages:
        if roles.get("grasp", {}).get("stage") == s.index:
            continue
        kind = ("active+passive: mover roles bind at milestone 4" if s.passive is not None
                else "a further active-only stage: only the first binds grasp")
        checks.append({"check": f"unbound[stage{s.index}]", "pass": False, "blocking": False,
                       "detail": f"{s.name!r} ({kind})"})
    if not roles:
        checks.append({"check": "roles[none]", "pass": False, "blocking": True,
                       "detail": "no active-only stage: nothing binds structurally (mover-only "
                                 "plans wait for milestone 4) -- calls #2/#3 would get empty roles"})
    return roles, checks


def ground_pass(plan, plan_sha: str, parts: dict, mark_key: dict, grounding: Path):
    """(plan_grounded, dropped {h: {part: reason}}, checks, roles, record)."""
    from manip_bridge import stageplan_compat as sp      # loaded only on this pass
    got = read_grounding(grounding, plan_sha, parts, mark_key)
    dropped = {h: g["dropped"] for h, g in got.items() if g["dropped"]}
    gplan = sp.prune_parts(plan, {h: tuple(d) for h, d in dropped.items()})
    pools = {}
    for h, g in got.items():
        try:
            pools[h] = sp.load_pool(g["canon_dir"])
        except Exception as e:                            # selection.ResolutionError
            raise PlanRefusal(f"pool for {h}: {e}") from e
    checks = [{"check": c, "pass": ok, "blocking": True, "detail": d}
              for c, ok, d in sp.verify(gplan, pools, sp.STRUCTURAL_SPEC, False)]
    roles, rchecks = bind_structural(gplan)
    record = {"dir": str(grounding.resolve()), "spec": sp.STRUCTURAL_SPEC.name,
              "objects": {h: {k: v for k, v in g.items() if k != "dropped"}
                          for h, g in got.items()},
              "stageplan": sp.provenance()}
    return gplan, dropped, checks + rchecks, roles, record


# ------------------------------------------------------------------ the call

def plan_scene(run, task: str, replay: str | None = None, out: str | None = None,
               transport=None, grounding: str | None = None) -> tuple[int, dict, dict]:
    """Returns (exit_code, artifact, report). `transport` is injectable for
    tests (a callable payload -> raw text, recorded like the live one).
    `grounding` (requires `replay`): the pass after grounding, see the
    module docstring."""
    run = Path(run)
    if not task or not task.strip():
        raise PlanRefusal("--task is empty")
    if grounding and not replay:
        raise PlanRefusal("--grounding needs --replay: a grounding belongs to the plan that "
                          "named its parts, so the pass re-reads that plan, it never asks anew")
    info = load_run(run)
    vocab = vlm_compat.Vocabulary.from_marks(info["markset"])
    view = info["markset"].dir / info["markset"].marked
    out_dir = Path(out) if out else run / "stage_plan"

    replayed_from, model = None, vlm_compat.MODEL
    if replay:
        try:
            log = json.loads(Path(replay).read_text())
        except (OSError, ValueError) as e:
            raise PlanRefusal(f"--replay {replay}: unreadable ({e})") from e
        if not log.get("raw_responses"):
            raise PlanRefusal(f"--replay {replay}: no raw_responses recorded")
        rec = ReplayTransport(log["raw_responses"])
        model = log.get("model") or model
        replayed_from = str(Path(replay).resolve())
    else:
        rec = RecordingTransport(transport)
    client = vlm_compat.Client(transport=rec, model=model)
    try:
        plan = client.plan_stages(task, vocab, [view])
    except vlm_compat.VLMError as e:
        raise PlanRefusal(f"call #1 failed: {e}") from e

    parts = object_parts(plan)
    checks, mark_key = run_checks(plan, parts, info["registered"])
    blocking = [c["check"] for c in checks if c["blocking"] and not c["pass"]]
    plan_doc = asdict(plan)
    plan_sha = hashlib.sha256(json.dumps(plan_doc, sort_keys=True).encode()).hexdigest()[:8]
    artifact = {
        # plan_stages layout: calls #2/#3 extend these files through load_bindings
        "plan": plan_doc,
        "plan_grounded": plan_doc,                 # hardware: no manifest, handles as emitted
        "parts_dropped": {},
        "mode": "marks",
        "model": client.model,
        "views": [str(view)],
        "task_spec": None,
        "object_parts": parts,
        "object_parts_requested": parts,
        "roles": {},
        "checks": [{"check": c["check"], "pass": c["pass"], "detail": c["detail"]}
                   for c in checks],
        "gate": {"defer_grounding": True, "blocking": blocking},
        # hardware-only
        "hardware": {"run": str(run.resolve()), "task": task, "mark_key": mark_key,
                     "registered": sorted(info["registered"]), "plan_sha": plan_sha,
                     "blocking_policy": {c["check"]: c["blocking"] for c in checks},
                     "replayed_from": replayed_from},
        "sim": vlm_compat.provenance(),
    }
    grounding_blocked = [b for b in blocking if b != DEFERRED]
    roles = None
    if grounding:
        if grounding_blocked:
            raise PlanRefusal(f"--grounding: the plan itself is blocked for grounding "
                              f"({grounding_blocked}) -- nothing was grounded from it")
        gplan, dropped, gchecks, roles, grec = ground_pass(plan, plan_sha, parts, mark_key,
                                                           Path(grounding))
        checks = [c for c in checks if c["check"] != DEFERRED] + gchecks
        blocking = [c["check"] for c in checks if c["blocking"] and not c["pass"]]
        artifact.update({
            "plan_grounded": asdict(gplan),
            "parts_dropped": {h: list(d) for h, d in dropped.items()},
            "task_spec": grec["spec"],
            "object_parts": object_parts(gplan),
            "roles": roles,
            "checks": [{"check": c["check"], "pass": c["pass"], "detail": c["detail"]}
                       for c in checks],
            "gate": {"defer_grounding": False, "blocking": blocking},
        })
        artifact["hardware"].update({
            "blocking_policy": {c["check"]: c["blocking"] for c in checks},
            "parts_dropped_reasons": dropped, "grounding": grec})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (out_dir / "plan.log.json").write_text(json.dumps({
        "model": client.model, "task": task,
        "client_logs": [asdict(log) for log in client.logs],
        "raw_responses": rec.raw}, indent=2) + "\n")
    report = {"plan": plan, "parts": parts, "mark_key": mark_key, "checks": checks,
              "out": str(out_dir / "plan.json"), "grounding_blocked": grounding_blocked,
              "roles": roles, "blocking": blocking,
              "next": [] if grounding_blocked or grounding else
              next_commands(run, plan_sha, parts, mark_key, info)}
    if grounding:
        return (2 if blocking else 0), artifact, report
    return (2 if grounding_blocked else 0), artifact, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--task", required=True, help='e.g. "pick up the mug by its handle"')
    ap.add_argument("--replay", default=None, help="a plan.log.json to replay (no API call)")
    ap.add_argument("--out", default=None, help="default <run>/stage_plan")
    ap.add_argument("--grounding", default=None,
                    help="after grounding (needs --replay): the --out of the printed "
                         "ground_asset commands, e.g. <run>/grounding_vlm")
    a = ap.parse_args(argv)
    try:
        code, art, rep = plan_scene(a.run, a.task, a.replay, a.out, grounding=a.grounding)
    except PlanRefusal as e:
        print(f"[plan-scene] REFUSED: {e}", file=sys.stderr)
        return 2
    plan = rep["plan"]
    print(f"[plan-scene] model {art['model']}{' (replayed)' if a.replay else ''}; "
          f"task {a.task!r} -> {rep['out']}")
    for h, ref in sorted(plan.objects.items()):
        print(f"    object {h:14s} mark {ref.mark} ({ref.label!r}) -> {rep['mark_key'][h]}")
    for s in plan.stages:
        print(f"    stage {s.index}: {s.name!r}  active={s.active} passive={s.passive}  "
              f"parts={ {k: list(v) for k, v in s.parts.items()} }")
    for c in rep["checks"]:
        tag = "PASS" if c["pass"] else ("FAIL" if c["blocking"] else "note")
        print(f"    {tag:4s} {c['check']}: {c['detail']}")
    if a.grounding:
        for h, d in sorted(art["hardware"]["parts_dropped_reasons"].items()):
            for p, why in sorted(d.items()):
                print(f"    dropped {h}.{p}: {why}")
        for r, b in sorted((rep["roles"] or {}).items()):
            print(f"    role {r:16s} -> {b['object']} @ stage {b['stage']}")
        if rep["blocking"]:
            print(f"[plan-scene] BLOCKED for calls #2/#3: {rep['blocking']} "
                  "(artifact written for inspection)", file=sys.stderr)
        else:
            print("[plan-scene] roles bound; plan.json is ready for calls #2/#3 "
                  "(read it with stageplan_compat.load_structural)")
        return code
    if rep["grounding_blocked"]:
        print(f"[plan-scene] REFUSED for grounding: {rep['grounding_blocked']} "
              "(artifact written for inspection)", file=sys.stderr)
        return code
    print("[plan-scene] next (fresh masks root per plan; outputs beside M1's, not over them):")
    for line in rep["next"]:
        print(f"    {line}")
    return code


if __name__ == "__main__":
    sys.exit(main())
