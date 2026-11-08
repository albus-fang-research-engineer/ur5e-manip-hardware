#!/usr/bin/env python3
"""VLM call #3 on a hardware run: per bound stage, the model authors the
TSR pair (path + subgoal) in the symbolic vocabulary, and the sim's compiler
grounds it at the run's measured poses -- the hardware twin of the sim's
scripts/emit_constraints.py (framed-chained arm by default, as the sim's
run_pipeline). Nothing moves; the output is the emission artifact plus a
geometry report.

    <run>/stage_plan/plan.json         plan_scene --grounding (roles bound)
    <run>/call2/selections.json        select_scene (same plan sha): the anchor
                                       points -- w's origin on the passive and
                                       the active feature Tw_e pins
    <run>/grounding_vlm/canon/<name>/  ground_asset: frames.json (symbols, in
                                       canonical coordinates), mesh, xml
    <run>/summary.json                 measured poses: base <- body (hand-eye x
                                       pose_on_final), body <- canonical
                                       (R_body_from_canon)
            |
            v  per stage (sim emit_constraints' loop, unchanged):
            v    render the stage at its entry poses (framed-chained arm: w and
            v    the canonical triads drawn at the call-#2 points)
            v    Client.emit_constraints  -> compile_stage -> check_pair_consistency
            v    (-> check_stage_seam); a CompileError goes back to the model,
            v    COMPILE_RETRIES times; the mover's next entry = this subgoal's center
    <out>/emissions.json       sim format (plan_task / plan_pour_tea --emissions)
                               + "hardware": poses source, geometry report
    <out>/emissions.log.json   model, raw responses (replay), client logs, provenance
    <out>/frames/<role>/*.png  what the model saw (framed arm)

Entry poses are MEASURED (the run's registered poses), not the sim's
upright spawn poses: the compiler freezes each goal attitude on them.

Geometry report (log only, never a gate): for every stage with a passive
object, at the subgoal's center, each named point of the mover relative
to w's origin -- horizontal distance and height. This is where "the carry
puts the handle, not the spout, over the mug" shows up as a number.

Call count: one call per bound stage with an active object (pour tea: grasp,
transport_active, pour = 3). Each compile rejection re-asks (COMPILE_RETRIES)
and each parse rejection re-asks (MAX_PARSE_RETRIES), so the worst case is
stages x (1 + COMPILE_RETRIES) x (1 + MAX_PARSE_RETRIES). The script prints
the budget before the first call and the requests actually made at the end.

  --dry-run    renders the first stage and writes the exact request it would
               send to <out>/dry_request.json; no API call.
  --replay LOG feeds the logged responses back (no API call). Rendering and
               compiling still run.

Exit: 0 every stage grounded; 1 a stage failed the compile gate (artifacts
written: a failing emission is evidence); 2 refused.

In the ros2 container (GL for the renders; ANTHROPIC_API_KEY for a live call):

    scripts/vlm_run.sh emit_scene.py --run /data/runs/<stamp>          # live
    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/emit_scene.py --run /data/runs/<stamp> --dry-run
    python3 /data/runs/emit_scene.py --run /data/runs/<stamp> \\
        --replay /data/runs/<stamp>/call3/emissions.log.json
"""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from manip_bridge import emit_compat as ec
    from manip_bridge import scene_ops as so
    from manip_bridge import selection_compat as sc
    from manip_bridge import vlm_compat
    from manip_bridge.vlm_record import RecordingTransport, ReplayTransport
except ImportError as e:
    sys.exit(f"emit_scene: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_scene as ss  # noqa: E402  -- load_inputs: the plan + grounding-sha gate

ARMS = ("framed-chained", "schema-only")
TABLE_MARGIN_M = 0.3        # render slab reaches this far past the farthest object


class EmitRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


class DryRun(Exception):
    """Raised by the dry-run transport with the request it was handed."""

    def __init__(self, payload):
        super().__init__("dry run: request captured, not sent")
        self.payload = payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _block(R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _r(T) -> list:
    return np.round(np.asarray(T, float), 6).tolist()


# ------------------------------------------------------------------ inputs

def alias_assets(objs: dict, root: Path) -> dict:
    """The sim's renderer and loaders find an object's files by its NAME
    (<dir>/<name>.xml, meshes/<name>_visual.obj); the canonical copies are
    named after the library asset (lib_teapot), the plan after the contract
    (teapot). One symlinked dir per contract name, nothing copied."""
    out = {}
    for obj, o in objs.items():
        canon, d = Path(o["canon_dir"]), root / obj
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in canon.iterdir():
            if f.name == "meshes":
                (d / "meshes").mkdir()
                for m in f.iterdir():
                    (d / "meshes" / m.name).symlink_to(m.resolve())
                vis = f / f"{canon.name}_visual.obj"
                if vis.is_file() and obj != canon.name:
                    (d / "meshes" / f"{obj}_visual.obj").symlink_to(vis.resolve())
            else:
                (d / f.name).symlink_to(f.resolve())
        xml = canon / f"{canon.name}.xml"
        if obj != canon.name and xml.is_file():
            (d / f"{obj}.xml").symlink_to(xml.resolve())
        out[obj] = d
    return out


def measured_poses(summary: dict, objs: dict) -> dict:
    """contract name -> base <- canonical frame (4x4): the measured body pose
    composed with the grounding's R_body_from_canon. Symbols are canonical."""
    out = {}
    for obj, o in objs.items():
        try:
            out[obj] = so.T_base_obj(summary, o["key"]) @ _block(o["R"])
        except ValueError as e:
            raise EmitRefusal(f"summary.json: {e}") from e
    return out


@dataclass
class _SceneObject:
    name: str
    asset: Path

    @property
    def xml(self) -> Path:
        return self.asset / f"{self.name}.xml"


class HardwareScene:
    """What the sim's stage-frame renderer reads from its Scene: the objects'
    asset dirs and a table slab for the ground cue. Poses are always passed
    in (the run's measured / chained poses); there is no spawn pose."""

    def __init__(self, asset_dirs: dict, table_top_z: float, table_size, path: Path):
        self.objects = {n: _SceneObject(n, Path(d)) for n, d in asset_dirs.items()}
        self.asset_dirs = {n: Path(d) for n, d in asset_dirs.items()}
        self.table_top_z = float(table_top_z)
        self.table_size = tuple(float(v) for v in table_size)
        self.path = path

    def fixed_poses(self):
        raise RuntimeError("HardwareScene has no spawn poses: pass the measured poses")


def table_for_render(summary: dict, poses: dict) -> tuple:
    """(top z, slab size) in base: the plane's height under the objects, a
    slab (centered on the base origin, as the sim's renderer places it) wide
    enough to sit under all of them. A ground cue only; nothing is planned on it."""
    xy = np.array([T[:2, 3] for T in poses.values()])
    try:
        plane = so.table_plane_base(summary)
    except ValueError as e:
        raise EmitRefusal(f"summary.json: {e}") from e
    z = float(np.mean(so.table_z(plane, xy[:, 0], xy[:, 1])))
    half = np.abs(xy).max(0) + TABLE_MARGIN_M
    return z, (2 * half[0], 2 * half[1], 0.02)


# ------------------------------------------------------------------ geometry report

def geometry(stage, cs, poses_entry: dict, w_point, e_point, symbols) -> dict:
    """At the subgoal center: each named point of the mover relative to w's
    origin (the passive's selected point; the passive does not move)."""
    T_sub = cs.subgoal.nominal()
    w = (poses_entry[stage.passive] @ np.r_[np.asarray(w_point, float), 1.0])[:3]
    e_tok = ec.point_token(stage.active, symbols, e_point) if e_point is not None else None
    pts = {}
    for name, p in symbols[stage.active].points.items():
        x = (T_sub @ np.r_[np.asarray(p, float), 1.0])[:3]
        pts[name] = {"horizontal_m": round(float(np.linalg.norm(x[:2] - w[:2])), 4),
                     "height_m": round(float(x[2] - w[2]), 4)}
    return {"w_origin_base": np.round(w, 4).tolist(), "pinned": e_tok,
            "at": "subgoal center (TSR.nominal)", "points": pts}


# ------------------------------------------------------------------ the call

def emit_scene(run, plan=None, selections=None, out=None, arm="framed-chained",
               replay=None, dry_run=False, transport=None, renderer=None):
    """Returns (exit_code, report). `transport` and `renderer` are injectable
    for tests (renderer: (compute_stage_frames, render_role, scene_extent))."""
    run = Path(run)
    plan_path = Path(plan) if plan else run / "stage_plan" / "plan.json"
    sel_path = Path(selections) if selections else run / "call2" / "selections.json"
    out = Path(out) if out else run / "call3"
    if arm not in ARMS:
        raise EmitRefusal(f"--arm {arm!r}: one of {ARMS}")
    b, doc, objs = ss.load_inputs(run, plan_path)     # plan readable, grounding unchanged
    plan_sha = (doc.get("hardware") or {}).get("plan_sha")

    # call #2 must describe THIS plan
    if not sel_path.is_file():
        raise EmitRefusal(f"{sel_path} not found -- run select_scene.py first")
    sel_log_path = sel_path.with_name("selections.log.json")
    sel_log = json.loads(sel_log_path.read_text()) if sel_log_path.is_file() else {}
    if sel_log.get("plan_sha") != plan_sha:
        raise EmitRefusal(f"{sel_path} was selected for plan {sel_log.get('plan_sha')!r}, "
                          f"the plan is {plan_sha!r} -- re-run select_scene.py")
    if sel_log.get("failure"):
        raise EmitRefusal(f"{sel_log_path} records a failed select_scene run "
                          f"({sel_log['failure']}) -- re-run select_scene.py")
    sels = sc.load_selections(sel_path)
    sel_objects = sc.selection_objects(sel_path, sels)
    missing = sorted(set(b.roles) - set(sels))
    if missing:
        raise EmitRefusal(f"{sel_path} has no selection for bound role(s) {missing}")
    resolved = sel_log.get("resolved") or {}
    unlogged = sorted(r for r, s in sels.items()
                      if (resolved.get(r) or {}).get("candidate_id") != s.candidate_id)
    if unlogged:
        raise EmitRefusal(f"{sel_log_path} does not record the marks {sel_path} holds for "
                          f"{unlogged} -- the selections and their log come from different "
                          "select_scene runs; re-run select_scene.py")

    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise EmitRefusal(f"{summary_path} not found -- not a run dir")
    summary = json.loads(summary_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    asset_dirs = alias_assets(objs, out / "assets")
    symbols = {n: sc.load_symbols(d) for n, d in asset_dirs.items()}
    vocab = vlm_compat.Vocabulary.from_asset_dirs(asset_dirs)
    poses = measured_poses(summary, objs)
    poses0 = dict(poses)

    by_idx = {s.index: s for s in b.plan.stages}
    stages = ec.bound_stages(b, by_idx)
    synthesized = b.entry_synthesized
    role_index = {r: st.index for r, (_, st) in b.roles.items()}
    try:        # selected_points refuses by SystemExit (exit 1 would read as a gate FAIL)
        points = {role: sc.selected_points(role, stage, sels, sel_objects, role_index,
                                           asset_dirs, symbols)
                  for stage, role in stages}
    except (SystemExit, sc.ResolutionError) as e:
        raise EmitRefusal(f"{sel_path}: anchor points do not resolve ({e}) -- "
                          "re-run select_scene.py") from None
    budget = len(stages) * (1 + ec.COMPILE_RETRIES) * (1 + vlm_compat.MAX_PARSE_RETRIES)
    print(f"[emit-scene] {len(stages)} stage(s) {[r for _, r in stages]}, arm {arm}: "
          f"{len(stages)} request(s) if every emission parses and compiles, at most {budget}")

    scene = None
    if arm == "framed-chained":
        compute_stage_frames, render_role, scene_extent = (
            renderer or ec.stage_frames_renderer()[:3])
        z, size = table_for_render(summary, poses)
        scene = HardwareScene(asset_dirs, z, size, plan_path)

    replayed_from, model = None, vlm_compat.MODEL
    if dry_run:
        def rec(payload):
            raise DryRun(payload)
        rec.raw = []
    elif replay:
        try:
            log = json.loads(Path(replay).read_text())
        except (OSError, ValueError) as e:
            raise EmitRefusal(f"--replay {replay}: unreadable ({e})") from e
        if not log.get("raw_responses"):
            raise EmitRefusal(f"--replay {replay}: no raw_responses recorded")
        rec = ReplayTransport(log["raw_responses"])
        model = log.get("model") or model
        replayed_from = str(Path(replay).resolve())
    else:
        rec = RecordingTransport(transport)
    client = vlm_compat.Client(transport=rec, model=model)

    emissions, gate, framed, geo = [], [], {}, {}
    prev_cs, prev_active = None, None
    chained: set = set()
    entry_from = "measured"
    try:
        for stage, role in stages:
            sel = views = frec = None
            if scene is not None:
                _, radius = scene_extent(scene, poses)
                sf = compute_stage_frames(((stage, role),), sels, sel_objects, role_index,
                                          asset_dirs, symbols, poses, radius)[role]
                frec = render_role(scene, role, sf, poses, out / "frames", entry_from)
                views = [Path(frec["views"][v]) for v in frec["views"]]
                sel = sels[role]
                framed[role] = [str(v) for v in views]
            w_point, e_point = points[role]
            anchor = w_anchor = None
            if stage.passive and e_point is not None:
                e_tok = ec.point_token(stage.active, symbols, e_point) or f"{stage.active}.anchor"
                w_tok = ec.point_token(stage.passive, symbols, w_point) or f"{stage.passive}.anchor"
                anchor = (e_tok, e_point)
                w_anchor = (w_tok, w_point, poses[stage.passive])
            synth = role in synthesized
            entry_text = None if synth else vlm_compat.entry_line(
                stage.active, poses[stage.active], symbols[stage.active].axes,
                chained=stage.active in chained, tol_rad=ec.ALIGN_TOL_RAD,
                anchor=anchor, w_anchor=w_anchor)
            rejections, err, em, made = [], None, None, 0
            for attempt in range(1 + ec.COMPILE_RETRIES):
                try:
                    em = client.emit_constraints(stage, vocab, selection=sel, view_paths=views,
                                                 rejections=rejections, frames=frec,
                                                 entry=entry_text)
                except DryRun as d:
                    (out / "dry_request.json").write_text(json.dumps(
                        {"role": role, "stage": asdict(stage), "arm": arm, "model": model,
                         "temperature": vlm_compat.TEMPERATURE, "views": framed.get(role, []),
                         "request": _strip_images(d.payload)}, indent=2, default=str) + "\n")
                    print(f"[emit-scene] dry run: {role}'s request -> {out / 'dry_request.json'}"
                          f" (+ views under {out / 'frames' / role}); no API call made")
                    return 0, {"out": str(out), "dry_run": True, "budget": budget,
                               "stages": [r for _, r in stages]}
                except vlm_compat.VLMError as e:
                    err = {"slot": "vlm", "reason": str(e), "others": []}
                    break
                made += 1
                print(f"[emit-scene] stage {em.stage} ({em.name}) attempt {attempt}")
                try:
                    cs = ec.compile_stage(em, symbols, poses, w_point=w_point, e_point=e_point,
                                          entry_synthesized=synth)
                    ec.check_pair_consistency(cs)
                    if prev_cs is not None and em.passive and em.active == prev_active:
                        for n_ in ec.check_stage_seam(prev_cs, cs, em):
                            print(f"         {n_}")
                except ec.CompileError as e:
                    rejections.append((client.logs[-1].raw, e.text()))
                    print(f"         compile rejected: {e.text()}")
                    err = {"slot": e.slot, "reason": e.reason,
                           "others": [{"slot": o.slot, "reason": o.reason} for o in e.others]}
                    continue
                for n in cs.notes:
                    print(f"         {n}")
                rows = {k: np.round(getattr(cs, k).Bw, 4).tolist() for k in ("path", "subgoal")}
                acct = {**ec.stage_accounting(cs, em, symbols, poses),
                        "attempts": attempt + 1,
                        "rejection_slots": [r.split(":")[0] for _, r in rejections],
                        "notes": list(cs.notes),
                        **({"entry_from": "synthesized"} if synth else {})}
                gate.append((em.name, True, rows, None, {n: _r(T) for n, T in poses.items()}, acct))
                if em.passive:
                    geo[role] = geometry(stage, cs, poses, w_point, e_point, symbols)
                    poses = {**poses, em.active: cs.subgoal.nominal()}
                    chained.add(em.active)
                    entry_from = f"{em.name}.subgoal.nominal"
                    prev_cs, prev_active = cs, em.active
                err = None
                break
            if err is not None or em is None:
                gate.append((stage.name, False, None,
                             {**(err or {}), "rejections": [r for _, r in rejections]},
                             {n: _r(T) for n, T in poses.items()},
                             {"attempts": made,
                              "rejection_slots": [r.split(":")[0] for _, r in rejections],
                              **({"entry_from": "synthesized"} if synth else {})}))
            if em is not None:
                emissions.append(em)
            if not gate[-1][1]:
                remaining = [r for _, r in stages[len(gate):]]
                if remaining:
                    print(f"[emit-scene] {role} failed the gate; not emitting {remaining} "
                          "(no entry pose for them)")
                break
    except BaseException as exc:     # paid responses survive a crash mid-loop
        if rec.raw and not replay and not dry_run:
            (out / "emissions.log.json").write_text(json.dumps(
                {"model": client.model, "temperature": vlm_compat.TEMPERATURE, "plan": str(plan_path),
                 "plan_sha": plan_sha, "selections": str(sel_path), "arm": arm,
                 "aborted": repr(exc), "requests_made": len(rec.raw),
                 "client_logs": [asdict(entry) for entry in client.logs],
                 "raw_responses": rec.raw}, indent=2, default=str) + "\n")
            print(f"[emit-scene] aborted after {len(rec.raw)} request(s); responses kept in "
                  f"{out / 'emissions.log.json'} (--replay it)", file=sys.stderr)
        raise

    artifact = {
        "task": b.plan.task, "stage_plan": str(plan_path), "selections": str(sel_path),
        "frames": None, "roles": [r for _, r in stages], "arm": arm,
        "model": client.model, "temperature": vlm_compat.TEMPERATURE,
        "render_dir": str(out / "frames") if scene is not None else None,
        "prompt_deltas": list(vlm_compat.PROMPT_DELTAS), "views": framed,
        "emissions": [asdict(e) for e in emissions],
        "compiled": [{"stage": n, "grounded": ok, "Bw": rows, "error": e_, "entry_poses": ent,
                      **acct} for n, ok, rows, e_, ent, acct in gate],
        "stages_reached": len(gate), "stages_total": len(stages),
        "hardware": {"run": str(run), "plan_sha": plan_sha,
                     "selections_sha256": _sha256(sel_path),
                     "poses": "measured (summary.json pose_on_final x R_body_from_canon), base frame",
                     "measured_poses": {n: _r(T) for n, T in poses0.items()},
                     "geometry": geo},
    }
    log = {"model": client.model, "temperature": vlm_compat.TEMPERATURE, "plan": str(plan_path),
           "plan_sha": plan_sha, "selections": str(sel_path), "arm": arm,
           "requests_made": 0 if replay else len(rec.raw),
           "replayed_from": replayed_from,
           "client_logs": [asdict(entry) for entry in client.logs],
           # a replay keeps the whole recorded list: --replay over the default --out
           # rewrites this same file, and a diverging replay consumes fewer responses
           "raw_responses": rec.responses if replay else rec.raw,
           "sim": {"emit": ec.provenance(render=scene is not None and renderer is None),
                   "vlm": vlm_compat.provenance(), "selection": sc.provenance()}}
    # the log first: it is the paid record, and replay needs only it
    (out / "emissions.log.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    (out / "emissions.json").write_text(json.dumps(artifact, indent=2, default=float) + "\n")
    failed = [n for n, ok, *_ in gate if not ok]
    return (1 if failed else 0), {"out": str(out), "gate": gate, "geometry": geo,
                                  "failed": failed, "requests": log["requests_made"],
                                  "budget": budget,
                                  "flags": [f for entry in client.logs for f in entry.flags]}


def _strip_images(payload: dict) -> dict:
    """The request with base64 image data replaced by its size."""
    def strip(x):
        if isinstance(x, dict):
            if x.get("type") == "base64" and "data" in x:
                return {**x, "data": f"<{len(x['data'])} base64 chars>"}
            return {k: strip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(payload)


def main(argv=None) -> int:
    # no prefix abbreviations: vlm_run.sh matches --replay / --dry-run verbatim
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--plan", default=None, help="default <run>/stage_plan/plan.json")
    ap.add_argument("--selections", default=None, help="default <run>/call2/selections.json")
    ap.add_argument("--out", default=None, help="default <run>/call3")
    ap.add_argument("--arm", choices=ARMS, default="framed-chained",
                    help="framed-chained (sim default): the stage rendered at its entry with w "
                         "and the triads drawn; schema-only: text only (ablation arm)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--replay", default=None, help="an emissions.log.json to replay (no API call)")
    g.add_argument("--dry-run", action="store_true",
                   help="write the first stage's request to <out>/dry_request.json; no API call")
    a = ap.parse_args(argv)
    try:
        code, rep = emit_scene(a.run, a.plan, a.selections, a.out, a.arm, a.replay, a.dry_run)
    except (EmitRefusal, ss.SelectRefusal) as e:
        print(f"[emit-scene] REFUSED: {e}", file=sys.stderr)
        return 2
    if rep.get("dry_run"):
        return code
    print("\n[emit-scene] compile gate:")
    for name, ok, _, err, _, _ in rep["gate"]:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + ("" if ok else f": {err.get('slot')}: {err.get('reason')}"))
    for role, g_ in rep["geometry"].items():
        print(f"[emit-scene] {role} at its subgoal center (pinned {g_['pinned']}), "
              "relative to w's origin:")
        for name, p in sorted(g_["points"].items(), key=lambda kv: kv[1]["horizontal_m"]):
            print(f"    {name:<16} horizontal {100 * p['horizontal_m']:5.1f} cm   "
                  f"height {100 * p['height_m']:+5.1f} cm")
    if rep["flags"]:
        print(f"[emit-scene] {len(rep['flags'])} flagged numeric literal(s): {rep['flags']}")
    print(f"[emit-scene] API requests made: {rep['requests']} (budget {rep['budget']})")
    print(f"[emit-scene] wrote {rep['out']}/emissions.json (+ emissions.log.json)")
    return code


if __name__ == "__main__":
    sys.exit(main())
