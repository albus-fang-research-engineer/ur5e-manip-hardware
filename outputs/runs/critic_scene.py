#!/usr/bin/env python3
"""VLM call #4 (critic v2) on a hardware run: preview the specification
(call #2's points + call #3's regions) before planning, let the critic
accept it or amend it with typed edits, and hand the accepted specification
to the planner. Library for plan_stages_hw.py --method vlm --critic preview;
nothing here moves the robot or talks to the cuRobo sidecar.

    stage plan, selections, emissions (+ emissions.log.json: call #3's raw
    responses, replayed when a stage is re-emitted), summary.json,
    the grasp (FK at q_grasp) and the carry's actual entry (FK at q_lift)
            |
            v  per round (LoopLedger: 2 rounds of edits, shared with call #5):
            v    compile the chain: the grasp at the measured poses (context,
            v      not editable), the first mover at its actual entry, every
            v      later mover against the previous stage's subgoal.nominal()
            v    preview: build_preview_report + render_preview (state images,
            v      path-row images); one caption per image, built from the
            v      renderer's manifest (kind, stage, panels, views, collisions,
            v      dropped panels, colors)
            v    critique_preview_v2 (parse retries go back to the model)
            v    accept -> done; reject -> ledger.admit -> apply_edits ->
            v      ledger.record -> re-emit (call #3) any reemit_stage target
            v      with the critic's diagnosis -> next round re-previews
            v    a reject once the budget is spent -> preview_rejected
    <out>/critic/round_<n>/   images/ (+ manifest.json), preview_inputs.json,
                              report.json, prompt.json, prompt.txt,
                              verdict.json, rejections.json, edits.json,
                              reemit.json
    <out>/critic/critic.json  the summary (also report.json["critic"])
    <out>/critic/responses/   every raw response, as files: replay a paid run
                              with --critic-responses <that dir>

Gripper: the preview draws the 2F-85 from <run>/critic_assets/gripper_2f85.obj
when it exists (--gripper-mesh to name another), else from the URDF. The
ros2 container cannot see the URDF; export the mesh on the host first:

    python3 outputs/runs/critic_scene.py export-gripper --run outputs/runs/<stamp>

Summary label: "model" is the model only for a paid run (MANIP_VLM_MODEL);
replay and dry runs record null, with "mode" saying which ran.

Response files (--critic-responses DIR): critic_*.txt|json are the critic's
raw responses in order (parse retries included), emit_*.txt|json are call
#3's for re-emitted stages. A file is one raw response, used verbatim.

Re-emit (reemit_stage): after the verdict's other edits are applied and the
chain recompiled, call #3 runs again for each re-emitted stage, in stage
order, at that stage's entry in the new chain. The stage's CURRENT emission
(edited rows included; its recorded raw response only when it still parses
to exactly that emission) is replayed as the assistant turn, followed by
one user turn that states what happened (REEMIT_COMPILED / REEMIT_UNCOMPILED
below), then call #3's usual compile retries. Views follow the arm recorded
in emissions.json: framed-chained renders the stage frames at the stage's
entry as emit_scene does; schema-only sends none. "chained" in the entry
line is True when an earlier mover stage moved the same object, as in
emit_scene.

NEW PROMPT TEXT (flagged): the two follow-up texts below are new text sent
to call #3, only on this path (--critic preview, a reemit_stage edit); the
prompts of calls 1-3 on every other path are unchanged. They reach the
model verbatim through Client.emit_constraints(feedback=[(raw, text)]), an
optional argument the sim must add (default None: requests byte-identical);
a sim without it refuses the re-emit with a typed record (reemit.json)
instead of sending wrong wording through the compile-rejection template.

Call budget (printed before the first paid call): 1 critic call per round
if every response parses, so 1-3 for budget 2 (accept on round 1 is 1; two
rounds of edits and a final check is 3); each parse rejection adds one
(MAX_PARSE_RETRIES per call); each re-emitted stage adds 1 call #3 call
(more on compile or parse rejections).
"""

import argparse
import colorsys
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

import inspect

from manip_bridge import emit_compat as ec
from manip_bridge import selection_compat as sc
from manip_bridge import vlm_compat
from manip_bridge.vlm_record import RecordingTransport, ReplayTransport

try:
    from manip_bridge import critic_compat as cc
    _CC_ERROR = None
except ImportError as _e:                 # a sim checkout without the critic
    cc, _CC_ERROR = None, _e

CRITIC_BUDGET = 2                          # rounds of edits (critic.LoopLedger default)
SPOUT_GAP = (("spout_center",), ("rim_center", "opening_center"))   # logged, never gated
# NEW PROMPT TEXT (call #3 re-emit, --critic only; see the module docstring)
REEMIT_COMPILED = ("A preview critic reviewed this stage before planning and rejected it: "
                   "{diagnosis}. Re-emit the corrected JSON object only, changing as little "
                   "as needed.")
REEMIT_UNCOMPILED = ("The compiler rejected that emission: {error}. A preview critic also "
                     "reviewed the stage before planning: {diagnosis}. Re-emit the corrected "
                     "JSON object only, changing as little as needed.")


class CriticRefusal(Exception):
    """An input the critic loop will not run on; the message names the fix."""


class CriticUnavailable(CriticRefusal):
    """The sim checkout has no critic (manip_sim.critic / render_preview)."""


def require_critic() -> None:
    """Raise CriticUnavailable when critic_compat could not import."""
    if cc is None:
        raise CriticUnavailable(
            f"sim checkout has no manip_sim.critic ({_CC_ERROR.__class__.__name__}: "
            f"{str(_CC_ERROR).splitlines()[0]}): merge critic-repair or point SIM_DIR at it")


# ------------------------------------------------------------------ context

@dataclass
class CriticContext:
    """Everything the loop reads; built by plan_stages_hw.run_vlm from
    load_vlm_inputs. `poses` are the measured canonical poses; `T_entry`
    is the first mover's actual entry (FK(q_lift) . T_ee_canon);
    `T_tcp_grasp` the TCP (tool0) at the grasp."""
    run_dir: Path
    out: Path
    plan_path: Path
    stages: list                      # [(StageSpec, role)] in order (ec.bound_stages)
    objs: dict
    asset_dirs: dict
    symbols: dict
    summary: dict
    poses: dict
    selections: dict                  # role -> PointAxisSelection (call #2)
    sel_objects: dict                 # role -> object
    role_index: dict
    synthesized: frozenset
    emissions: dict                   # role -> StageEmission (call #3)
    raw_emissions: dict               # stage name -> call #3's raw response
    offered: dict | None
    T_entry: np.ndarray
    grasp_obj: str
    T_tcp_grasp: np.ndarray
    es: object = None                 # the emit_scene module (table, HardwareScene)
    arm: str = "framed-chained"       # call #3's arm, from emissions.json (re-emit views)
    gripper_mesh: Path | None = None   # an exported 2F-85 (export_gripper); wins over the URDF
    gripper_urdf: Path | None = None
    gripper_meshes: Path | None = None
    notes: list = field(default_factory=list)


def raw_by_stage_name(log_path: Path) -> dict:
    """stage name -> the last raw response in an emit_scene log that parses
    to that stage (the accepted one, for a stage with compile retries)."""
    if not Path(log_path).is_file():
        return {}
    out = {}
    for raw in json.loads(Path(log_path).read_text()).get("raw_responses") or []:
        try:
            name = vlm_compat.load_json(raw).get("name")
        except Exception:                 # noqa: BLE001 -- an unparsable attempt
            continue
        if isinstance(name, str):
            out[name] = raw
    return out


def offered_from_log(sel_log: Path, objects: dict) -> dict | None:
    """role -> candidate ids call #2 was shown (select_scene's per-object
    menu), so a re-pick stays inside what call #2 could pick."""
    if not Path(sel_log).is_file():
        return None
    menus = json.loads(Path(sel_log).read_text()).get("menus") or {}
    if not menus:
        return None
    return {r: frozenset(int(k) for k in menus.get(o, {})) for r, o in objects.items()}


def spec_state(ctx: CriticContext):
    pools = {o: sc.load_pool(ctx.asset_dirs[o]) for o in sorted(set(ctx.sel_objects.values()))}
    return cc.SpecState(emissions=dict(ctx.emissions), selections=dict(ctx.selections),
                        objects=dict(ctx.sel_objects), pools=pools, offered=ctx.offered)


# ------------------------------------------------------------------ the compiled chain

@dataclass
class ChainStage:
    role: str
    stage: object                      # StageSpec
    emission: object                   # StageEmission
    poses: dict                        # every object's pose at this stage's entry
    w_point: np.ndarray | None = None
    e_point: np.ndarray | None = None
    compiled: object | None = None     # CompiledStage
    error: str | None = None
    synthesized: bool = False
    compile_error: object | None = None  # the CompileError, when the compiler rejected it
    no_entry: bool = False             # an earlier mover did not compile: no entry pose


def compile_chain(ctx: CriticContext, state) -> list[ChainStage]:
    """Compile every bound stage the way the planner will: the grasp at the
    measured poses (context only), the first mover at its actual entry,
    each later mover against the previous mover's subgoal.nominal(). A
    stage that does not compile ends the chain for the stages after it
    (they have no entry), and each says why."""
    out, cur, first, broken = [], dict(ctx.poses), True, None
    for stage, role in ctx.stages:
        em = state.emissions.get(role)
        if em is None:
            continue                     # a role call #3 did not author
        synth = role in ctx.synthesized
        rec = ChainStage(role, stage, em, dict(cur), synthesized=synth)
        out.append(rec)
        if em.passive is not None:
            if first:
                cur = {**cur, em.active: np.asarray(ctx.T_entry, float)}
                rec.poses, first = dict(cur), False
            if broken is not None:
                rec.error = (f"not compiled: stage {broken!r} did not compile, so this stage "
                             "has no entry pose")
                rec.no_entry = True
                continue
        try:
            rec.w_point, rec.e_point = sc.selected_points(
                role, stage, state.selections, state.objects, ctx.role_index,
                ctx.asset_dirs, ctx.symbols)
        except (SystemExit, sc.ResolutionError) as e:
            rec.error = f"anchor points do not resolve: {e}"
        else:
            try:
                cs = ec.compile_stage(em, ctx.symbols, rec.poses, w_point=rec.w_point,
                                      e_point=rec.e_point, entry_synthesized=synth)
                ec.check_pair_consistency(cs)
                rec.compiled = cs
            except ec.CompileError as e:
                rec.error = f"compile rejected: {e.text()}"
                rec.compile_error = e
        if em.passive is not None:
            if rec.compiled is None:
                broken = em.name
            else:
                cur = {**cur, em.active: np.asarray(rec.compiled.subgoal.nominal(), float)}
    return out


def accepted_spec(ctx: CriticContext, state, roles) -> tuple[dict, list]:
    """What the critic accepted, for the planner: role -> (emission,
    w_point, e_point) from the accepted SpecState (edited rows, re-picked
    points, re-emitted stages), never from the inputs it replaced. A role
    whose emission is missing or whose anchor points do not resolve is a
    typed failure record instead (the planner then plans nothing); a
    compile error is the planner's to type, at the stage's actual entry."""
    by_stage = {role: stage for stage, role in ctx.stages}
    out, failures = {}, []
    for role in roles:
        stage, em = by_stage.get(role), state.emissions.get(role)
        if stage is None or em is None:
            failures.append({"type": None, "routes_to": None, "stage": role, "stage_name": None,
                             "phase": "accepted_spec", "reason": "the accepted specification "
                             "has no emission for this stage"})
            continue
        try:
            w, e = sc.selected_points(role, stage, state.selections, state.objects,
                                      ctx.role_index, ctx.asset_dirs, ctx.symbols)
        except (SystemExit, sc.ResolutionError) as exc:
            failures.append({"type": None, "routes_to": None, "stage": role,
                             "stage_name": em.name, "phase": "anchor",
                             "reason": f"the accepted anchor points do not resolve: {exc}"})
            continue
        out[role] = (em, w, e)
    return out, failures


def preview_inputs(ctx: CriticContext, state, chain: list[ChainStage]) -> list:
    """StagePreviewInput per chain stage (compiled None where it failed)."""
    context = state.context_roles()
    return [cc.StagePreviewInput(
        role=c.role, emission=c.emission, compiled=c.compiled, poses=c.poses,
        symbols=ctx.symbols,
        w_point=np.zeros(3) if c.w_point is None else c.w_point, e_point=c.e_point,
        editable=c.role not in context) for c in chain]


def _uncompiled_record(c: ChainStage, state, editable: bool) -> dict:
    em = c.emission
    rec = {"stage": em.name, "role": c.role, "active": em.active, "passive": em.passive,
           "editable": editable, "goal": em.goal, "compile_error": c.error}
    if state.objects.get(c.role) == em.active and c.role in state.selections:
        rec["point_choice"] = {"pointer": f"{em.name}.point.{em.active}",
                               "current": state.point_of(c.role)}
    if not editable:                      # a context stage: trimmed as stage_report trims it
        rec.pop("passive")
        return rec
    for ctx_name, spec in (("path", em.path_tsr), ("subgoal", em.subgoal_tsr)):
        rows = [{"pointer": f"{em.name}.{ctx_name}.rot[{i}]", "text": cc.row_text(r)}
                for i, r in enumerate(spec.rot)]
        rows += [{"pointer": f"{em.name}.{ctx_name}.trans[{i}]", "text": cc.row_text(t)}
                 for i, t in enumerate(spec.trans)]
        rec[ctx_name] = {"rows": rows}
    return rec


def preview_report(state, chain: list[ChainStage], previews: list) -> dict:
    """build_preview_report over the compiled stages, with a record for
    each stage that did not compile spliced in at its place (rows as text
    and the compile error: the critic must still list it and may re-emit
    or re-pick it). Collision and search facts are not in the report: the
    image captions carry them (caption_for)."""
    ok = [p for p, c in zip(previews, chain) if c.compiled is not None]
    rep = cc.build_preview_report(ok, state)   # collision and search facts: the captions
    by_role = {r["role"]: r for r in rep["stages"]}
    rep["stages"] = [by_role[c.role] if c.compiled is not None
                     else _uncompiled_record(c, state, p.editable)
                     for p, c in zip(previews, chain)]
    return rep


def point_gaps(ctx: CriticContext, chain: list[ChainStage], names=SPOUT_GAP) -> dict:
    """Per compiled mover stage, at its subgoal nominal: each named active
    point vs each named passive point -- horizontal distance (across the
    passive's up) and height (along it). Log only."""
    act_names, pas_names = names
    out = {}
    for c in chain:
        em = c.emission
        if em.passive is None or c.compiled is None:
            continue
        T_a = np.asarray(c.compiled.subgoal.nominal(), float)
        T_p = np.asarray(c.poses[em.passive], float)
        up = T_p[:3, :3] @ np.asarray(ctx.symbols[em.passive].axes["up_axis"], float)
        up = up / np.linalg.norm(up)
        act = {a: ctx.symbols[em.active].points[a] for a in act_names
               if a in ctx.symbols[em.active].points}
        rec = {}
        for p in pas_names:
            pp = ctx.symbols[em.passive].points.get(p)
            if pp is None:
                continue
            ref = (T_p @ np.r_[np.asarray(pp, float), 1.0])[:3]
            for a, (h, z) in cc.point_offsets(T_a, ref, up, act).items():
                rec[f"{em.active}.{a} -> {em.passive}.{p}"] = {
                    "horizontal_m": round(float(h), 4), "height_m": round(float(z), 4)}
        out[c.role] = rec
    return out


# ------------------------------------------------------------------ captions
#
# One text block per image, placed right before it in the critic's request.
# Built from the renderer's manifest (render_preview): what the image is,
# which stage, each panel with its view, the collision facts the renderer
# measured (a goal that intersects another object, a panel in collision, a
# panel left out because it collides while the center pose is free, a
# center pose in collision) and the color legend. Fields the renderer does
# not report are left out of the caption, never guessed.

_VIEW_NAMES = {"iso": "iso", "iso-opp": "iso (opposite side)", "top": "top",
               "+x": "side (+x)", "-x": "side (-x)", "+y": "side (+y)", "-y": "side (-y)"}
_TRANS_ROWS = {"x", "y", "z"}


def color_name(rgb) -> str:
    """A coarse color word for an RGB triple in [0, 1]."""
    r, g, b = (float(v) for v in rgb[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.15:
        return "black"
    if s < 0.2:
        return "white" if v > 0.85 else "gray"
    for top, name in ((0.05, "red"), (0.11, "orange"), (0.19, "yellow"), (0.5, "green"),
                      (0.7, "blue"), (0.85, "purple"), (0.94, "pink"), (1.01, "red")):
        if h < top:
            return name
    return "red"


def object_colors(asset_dirs: dict) -> dict:
    """object -> color word, from the median of its texture's non-black
    pixels (meshes/*_texture.png); 'textured' when there is no texture."""
    out = {}
    for obj, d in asset_dirs.items():
        tex = sorted(Path(d, "meshes").glob("*_texture.png")) if Path(d, "meshes").is_dir() else []
        try:
            from PIL import Image
            a = np.asarray(Image.open(tex[0]).convert("RGB"), float).reshape(-1, 3) / 255.0
            a = a[a.max(1) > 0.08]
            out[obj] = color_name(np.median(a, 0)) if len(a) else "textured"
        except (IndexError, OSError, ImportError):
            out[obj] = "textured"
    return out


def _legend(manifest: dict, img: dict, colors: dict) -> str:
    leg = img.get("legend") or manifest.get("legend") or {}
    merged = {**colors, **(leg if isinstance(leg, dict) else {})}
    if manifest.get("gripper") and "gripper" not in merged:
        merged["gripper"] = color_name(cc.GRIPPER_RGBA)
    return ", ".join(f"{k} = {v}" for k, v in merged.items())


def contact_text(contacts, verb: str = "collides with") -> str:
    """The renderer's own wording (render_preview.contact_text)."""
    return cc.contact_text(contacts or [], verb)


def _bodies(contacts) -> str:
    return cc.contact_bodies(contacts or [])


def _path_kind(img: dict) -> str:
    g = img.get("group")
    if g in ("translation", "rotation"):
        return f"path {g} rows"
    rows = {r.get("row") for r in img.get("rows") or []}
    if rows and rows <= _TRANS_ROWS:
        return "path translation rows"
    if rows and not rows & _TRANS_ROWS:
        return "path rotation rows"
    return "path region rows"


def _path_panels(img: dict) -> tuple[list[str], list[str]]:
    """(drawn panel lines, not-drawn lines) from the rows' per-panel
    records; a row without them lists every caption as drawn."""
    drawn, dropped = [], []
    for row in img.get("rows") or []:
        view = row.get("view")
        view = _VIEW_NAMES.get(view, view)
        caps = row.get("captions") or []
        panels = row.get("panels") if isinstance(row.get("panels"), list) else []
        for j, cap in enumerate(caps):
            p = panels[j] if j < len(panels) and isinstance(panels[j], dict) else {}
            if p.get("dropped"):
                why = contact_text(p.get("collisions")) or "in collision"
                dropped.append(f"{p.get('desc') or cap}: {why} (the planner avoids this)")
                continue
            note = ""
            if p.get("collides"):
                b = _bodies(p.get("collisions"))
                note = f" -- in collision with {b}" if b else " -- in collision"
            drawn.append(f"{cap}" + (f" ({view} view)" if view else "") + note)
    return drawn, dropped


def _stage_label(c) -> str:
    return f'stage "{c.emission.name}" (role {c.role})'


def caption_for(img: dict, manifest: dict, k: int, n: int, chain: list, colors: dict,
                extra: list[str] | None = None) -> str:
    """The text block placed before image k (1-based) of n."""
    by_role = {c.role: c for c in chain}
    lines = []
    if img.get("kind") == "state":
        st = str(img.get("state", ""))
        if st == "initial":
            head = "initial scene (measured poses, before any stage)"
        elif st == "grasp":
            g = next((c for c in chain if c.emission.passive is None), None)
            head = "grasp goal: the gripper at the taught grasp" + (
                f" -- {_stage_label(g)}, context, not editable" if g else "")
        else:
            role = img.get("role") or st.split(":", 1)[-1]
            c = by_role.get(role)
            shown = (img.get("search") or {}).get("shown")
            where = ("the nearest collision-free pose of its subgoal region (the region's "
                     "center collides)" if shown == "nearest_free" else
                     "the center of its subgoal region (no collision-free pose found)"
                     if shown == "none_free" else "the center of its subgoal region")
            head = (f"stage goal -- {_stage_label(c)}: the {c.emission.active} at {where}"
                    if c else f"stage goal (role {role})")
        lines.append(f"Image {k} of {n}: {head}.")
        if img.get("panels"):
            lines.append("Panels, left to right: " + "; ".join(
                f"{_VIEW_NAMES.get(p, p)} view" for p in img["panels"]) + ".")
        search = img.get("search")
        if "collisions" in img and img["collisions"] is not None:
            txt = contact_text(img["collisions"], "intersects")
            label = ("Collisions at the subgoal center" if search and search.get("nominal_collides")
                     else "Collisions")
            lines.append(f"{label}: {txt}." if txt else "Collisions: none.")
        if search and search.get("nominal_collides"):
            lines += [f"Renderer: {nt}." for nt in img.get("notes") or []]
    else:
        role = img.get("stage")
        c = by_role.get(role)
        label = _stage_label(c) if c else f'stage "{img.get("name", role)}" (role {role})'
        mover = c.emission.active if c else "object"
        lines.append(f"Image {k} of {n}: {_path_kind(img)} -- {label}: the {mover} with one "
                     "path row at a time at its bound, the other rows at the center pose "
                     "(midpoint of the stage's entry and goal, inside the path region).")
        col = colors.get(mover)
        lines.append(f"Moving object: {mover}" + (f" ({col})" if col and col != "textured" else "")
                     + ". No point labels in path images; every object keeps its color from "
                     "the state images.")
        drawn, dropped = _path_panels(img)
        if drawn:
            lines.append("Panels, left to right then top to bottom:")
            lines += [f"  {i + 1}. {d}" for i, d in enumerate(drawn)]
        if img.get("notes") is not None:
            # the renderer's own lines (center search, panels kept); the
            # dropped-panel lines are rebuilt below from the panel records
            own = [nt for nt in img["notes"] if not nt.endswith("(the planner avoids this)")]
            if own:
                lines.append("Renderer: " + "; ".join(own) + ".")
        else:
            center = (img.get("representative") or {}).get("collisions")
            if img.get("center_collides") or center:
                txt = contact_text(center)
                lines.append("The center pose itself is in collision"
                             + (f" ({txt})" if txt else "")
                             + ": no collision-free pose to show, so every panel is kept.")
        if dropped:
            lines.append("Not drawn: " + "; ".join(dropped) + ".")
        if img.get("pinned_rows"):
            lines.append("Pinned rows (fixed, not drawn): " + ", ".join(img["pinned_rows"]) + ".")
    lines += extra or []
    leg = _legend(manifest, img, colors)
    if leg:
        lines.append(f"Colors: {leg}.")
    return "\n".join(lines)


def _text_only_lines(rec: dict) -> str:
    """A path group with no panel left (every pose dropped): text only."""
    _, dropped = _path_panels(rec)
    return (f"{_path_kind(rec).capitalize()} of this stage: no image, every panel collides "
            "while the center pose is free -- " + "; ".join(dropped) + ".")


def views_from_manifest(manifest: dict, image_dir: Path, chain: list,
                        colors: dict) -> list[tuple[str, Path]]:
    imgs = manifest.get("images") or []
    extra: dict[str, list[str]] = {}            # stage goal role -> text-only path groups
    for rec in manifest.get("path_text_only") or []:
        extra.setdefault(rec.get("stage"), []).append(_text_only_lines(rec))
    out = []
    for k, img in enumerate(imgs):
        more = []
        if img.get("kind") == "state" and str(img.get("state", "")).startswith("goal:"):
            more = extra.get(img.get("role") or img["state"].split(":", 1)[1], [])
        if k == 0 and manifest.get("collision_check"):
            chk = manifest["collision_check"]
            more = more + [f"Collision checks in these images: {chk.get('bodies', 'scene')}"
                           + (f", tolerance {chk['tol_mm']} mm" if chk.get("tol_mm") is not None
                              else "") + "."]
        out.append((caption_for(img, manifest, k + 1, len(imgs), chain, colors, more),
                    Path(image_dir) / img["file"]))
    return out


# ------------------------------------------------------------------ rendering

def render_round(ctx: CriticContext, chain: list[ChainStage], previews: list,
                 image_dir: Path) -> dict:
    """Render the preview (GL at call time) and return the manifest. The
    renderer inputs are also written as preview_inputs.json beside the
    images, so a round can be re-rendered offline with the sim's CLI."""
    initial, stages, labels = cc.from_preview_inputs(previews)
    for c in chain:                       # the passive's anchor point, named
        em = c.emission
        if em.passive and c.w_point is not None:
            tok = ec.point_token(em.passive, ctx.symbols, c.w_point)
            if tok:
                pt = tok.split(".", 1)[1]
                labels.setdefault(em.passive, {})[pt] = np.asarray(
                    ctx.symbols[em.passive].points[pt], float)
    z, size = _table(ctx)
    gripper = None
    if ctx.gripper_mesh is not None:
        gripper = load_gripper_mesh(ctx.gripper_mesh)
    elif ctx.gripper_urdf is not None:
        gripper = cc.gripper_from_urdf(ctx.gripper_urdf, "tool0", ctx.gripper_meshes)
    _write_inputs_doc(ctx, image_dir.parent / "preview_inputs.json", initial, stages, labels,
                      z, size)
    scene = _scene(ctx, z, size)
    return cc.render_preview(scene, initial, stages, labels, image_dir,
                             grasp=cc.GraspInput(ctx.grasp_obj, np.asarray(ctx.T_tcp_grasp)),
                             gripper=gripper)


def _table(ctx):
    return ctx.es.table_for_render(ctx.summary, ctx.poses)


def _scene(ctx, z, size):
    return ctx.es.HardwareScene(ctx.asset_dirs, z, size, ctx.plan_path)


def _tsr_doc(t):
    return None if t is None else {"T0_w": np.asarray(t.T0_w).tolist(),
                                   "Tw_e": np.asarray(t.Tw_e).tolist(),
                                   "Bw": np.asarray(t.Bw).tolist(), "name": t.name}


def _write_inputs_doc(ctx, path: Path, initial, stages, labels, z, size):
    """render_preview.load_inputs' schema (the 11/05 scratchpad builder's
    output, now produced here)."""
    doc = {"scene": {"asset_dirs": {k: str(v) for k, v in ctx.asset_dirs.items()},
                     "table_top_z": z, "table_size": list(size)},
           "initial": {k: np.asarray(v).tolist() for k, v in initial.items()},
           "stages": [{"role": s.role, "name": s.name, "mover": s.mover, "passive": s.passive,
                       "path": _tsr_doc(s.path), "subgoal": _tsr_doc(s.subgoal),
                       "entry": None if s.entry is None else np.asarray(s.entry).tolist(),
                       "error": s.error} for s in stages],
           "grasp": {"obj": ctx.grasp_obj, "tcp_pose": np.asarray(ctx.T_tcp_grasp).tolist()},
           "labels": {o: {p: np.asarray(x).tolist() for p, x in pts.items()}
                      for o, pts in labels.items()},
           "gripper": ({"kind": "mesh", "obj": str(ctx.gripper_mesh), "frame": "tool0"}
                       if ctx.gripper_mesh else
                       {"kind": "urdf", "urdf": str(ctx.gripper_urdf), "root_link": "tool0",
                        "mesh_dir": str(ctx.gripper_meshes)} if ctx.gripper_urdf else None)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


# ------------------------------------------------------------------ the gripper as one mesh
#
# Ros2Bridge cannot see the 2F-85's URDF or meshes (curobo_runtime/ and
# test/ are not mounted there), but it sees the run folder (/data/runs).
# export_gripper writes the gripper the renderer would build from the URDF
# -- every link below tool0 along fixed joints, collision geometry, in
# tool0's frame (the TCP the hardware IK targets, GraspInput.tcp_pose) --
# as one OBJ, exactly as render_preview writes its gripper.obj. Loaded back
# it is the same triangle mesh (to the OBJ's 1e-6 m rounding), placed by
# the same TCP pose, so the renderer's collision check (which counts the
# gripper as part of the mover) sees the same geometry either way.

CRITIC_ASSETS = "critic_assets"
GRIPPER_MESH = "gripper_2f85.obj"


def export_gripper(urdf: Path, mesh_dir: Path | None, out_dir: Path) -> Path:
    """<out_dir>/gripper_2f85.obj (+ .json provenance) from the URDF."""
    g = cc.gripper_from_urdf(urdf, "tool0", mesh_dir)
    path = g.write_obj(Path(out_dir) / GRIPPER_MESH)
    _w(path.with_suffix(".json"), {
        "source": g.source, "urdf": str(urdf), "mesh_dir": str(mesh_dir), "root_link": "tool0",
        "frame": "tool0 (the TCP the hardware IK targets; GraspInput.tcp_pose)",
        "geometry": "collision", "n_vertices": int(len(g.V)), "n_faces": int(len(g.F)),
        "obj_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def load_gripper_mesh(path: Path):
    """An exported OBJ (v / f lines, as GripperModel.write_obj writes it)
    -> GripperModel in the TCP frame."""
    V, F = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    if not V or not F:
        raise CriticRefusal(f"{path}: no vertices or faces -- re-run export-gripper")
    meta = Path(path).with_suffix(".json")
    src = (json.loads(meta.read_text()).get("source") if meta.is_file() else None) or Path(path).name
    return cc.GripperModel(np.asarray(V, float), np.asarray(F, int), f"{src} (mesh {Path(path).name})")


def find_gripper_mesh(run_dir: Path) -> Path | None:
    p = Path(run_dir) / CRITIC_ASSETS / GRIPPER_MESH
    return p if p.is_file() else None


# ------------------------------------------------------------------ transports

def _response_files(d: Path, prefix: str) -> list[str]:
    files = sorted(p for p in Path(d).glob(f"{prefix}_*") if p.suffix in (".txt", ".json"))
    return [p.read_text() for p in files]


def make_transports(mode: str, responses: Path | None):
    """(critic transport, call #3 transport). 'replay': the files in
    `responses`; 'live': recording transports over the live API."""
    if mode == "replay":
        if not Path(responses).is_dir():
            raise CriticRefusal(f"--critic-responses {responses}: not a directory")
        crit = _response_files(responses, "critic")
        if not crit:
            raise CriticRefusal(f"--critic-responses {responses}: no critic_* files")
        return ReplayTransport(crit), ReplayTransport(_response_files(responses, "emit"))
    return RecordingTransport(), RecordingTransport()


# ------------------------------------------------------------------ logging

def _messages_for_log(system: str, messages: list, views) -> dict:
    """The request with each image block replaced by its file path."""
    paths = iter(str(p) for _, p in views)
    out = []
    for m in messages:
        content = []
        for b in m["content"]:
            if b.get("type") == "image":
                content.append({"type": "image", "path": next(paths, "?")})
            else:
                content.append(b)
        out.append({"role": m["role"], "content": content})
    return {"system": system, "messages": out}


def _prompt_text(req: dict) -> str:
    parts = ["=== system ===", req["system"]]
    for m in req["messages"]:
        parts.append(f"=== {m['role']} ===")
        for b in m["content"]:
            parts.append(f"[image: {b['path']}]" if b.get("type") == "image" else b.get("text", ""))
    return "\n".join(parts) + "\n"


def _edit_rec(e) -> dict:
    return {"target": e.target, "action": e.action, "token": e.token, "role": e.role,
            "kind": e.kind, "before": e.before, "after": e.after}


def _verdict_rec(v) -> dict:
    return {"source": v.source, "verdict": v.verdict,
            "stages": [asdict(s) for s in v.stages],
            "edits": [_edit_rec(e) for e in v.edits], "diagnosis": v.diagnosis}


def _w(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")


# ------------------------------------------------------------------ re-emit (call #3)

_AXIS_DIR = {"front_axis": "front", "lateral_axis": "left", "up_axis": "up"}


def _dir_token(axis: str, sign: str = "+") -> str:
    """'teapot.up_axis' -> 'teapot.+up' (the signed token the model writes);
    world.z stays world.z."""
    if axis.startswith("world."):
        return axis if sign == "+" else "world.-z"
    obj, ax = axis.split(".", 1)
    return f"{obj}.{sign}{_AXIS_DIR.get(ax, ax)}"


def emission_raw(em) -> str:
    """A StageEmission written in call #3's output schema, as the model
    writes it (signed direction tokens, rotation rows in the "relation"
    form, a free table as "free"). parse_emission reads it back to the
    same emission (tested on every 11/05 stage)."""
    def rot(rows):
        if all(r.relation == "free" for r in rows) and {r.row for r in rows} == {
                "roll", "pitch", "yaw"}:
            return "free"
        out = []
        for r in rows:
            if r.relation == "free":
                out.append({"relation": "free", "row": r.row})
            else:
                out.append({"axis": _dir_token(r.axis), "relation": r.relation,
                            "reference": _dir_token(r.reference), "tol": r.tol})
        return out

    def trans(terms):
        out = []
        for t in terms:
            if t.term == "along":
                out.append({"term": "along", "axis": _dir_token(t.axis, t.sign or "+")})
            elif t.term == "expr":
                out.append({"term": "expr", "row": t.row, "lo": t.expr_lo, "hi": t.expr_hi})
            else:
                out.append({k: v for k, v in (("term", t.term), ("anchor", t.anchor),
                                              ("clearance", t.clearance), ("slack", t.slack),
                                              ("tol", t.tol)) if v is not None})
        return out
    return json.dumps({"stage": em.stage, "name": em.name, "active": em.active,
                       "passive": em.passive, "goal": em.goal,
                       "path_tsr": {"rot": rot(em.path_tsr.rot), "trans": trans(em.path_tsr.trans)},
                       "subgoal_tsr": {"rot": rot(em.subgoal_tsr.rot),
                                       "trans": trans(em.subgoal_tsr.trans)},
                       "verify": em.verify})


def prior_emission_raw(ctx: CriticContext, em, vocab) -> tuple[str, str]:
    """(raw, source) for the re-emit's assistant turn: the stage's recorded
    raw response when it still parses to exactly `em`, else `em` itself
    written in the output schema (a row edit changed it)."""
    raw = ctx.raw_emissions.get(em.name)
    if raw is not None:
        try:
            if vlm_compat.parse_emission(raw, vocab) == em:
                return raw, "recorded"
        except vlm_compat.ParseRejection:
            pass
    return emission_raw(em), "current"


def _chained(ctx: CriticContext, state, role: str) -> bool:
    """emit_scene's rule: True when an earlier mover stage moved the same
    object (its subgoal nominal is this stage's entry)."""
    stage = next(st for st, r in ctx.stages if r == role)
    for st, r in ctx.stages:
        if r == role:
            return False
        em = state.emissions.get(r)
        if em is not None and em.passive is not None and em.active == stage.active:
            return True
    return False


def feedback_supported(client) -> bool:
    return "feedback" in inspect.signature(client.emit_constraints).parameters


def reemit(ctx: CriticContext, state, c: ChainStage, diagnosis: str, client, rd: Path,
           frames_renderer=None):
    """Call #3 again for one stage, at its entry in the CURRENT chain (`c`
    comes from the chain recompiled after the verdict's other edits): the
    stage's current emission replayed, then the follow-up that says what
    happened (REEMIT_COMPILED, or REEMIT_UNCOMPILED with the real compile
    error), then call #3's compile retries. Returns (emission or None,
    record). The emission is returned even when its compile failed (the
    next preview shows the error), as emit_scene keeps it."""
    stage, role = c.stage, c.role
    rec = {"role": role, "stage": stage.name, "diagnosis": diagnosis, "attempts": []}
    if c.no_entry:
        rec["error"] = f"no entry pose ({c.error}); not re-emitted"
        return None, rec
    if not feedback_supported(client):
        rec["error"] = ("sim Client.emit_constraints has no feedback= argument: the re-emit "
                        "follow-up cannot be sent truthfully; not re-emitted (apply the sim "
                        "change named in critic_scene.py)")
        return None, rec
    vocab = vlm_compat.Vocabulary.from_asset_dirs(ctx.asset_dirs)
    poses = c.poses
    sel = state.selections.get(role)
    try:
        w_point, e_point = sc.selected_points(role, stage, state.selections, state.objects,
                                              ctx.role_index, ctx.asset_dirs, ctx.symbols)
    except (SystemExit, sc.ResolutionError) as e:
        rec["error"] = f"anchor points do not resolve: {e}"
        return None, rec
    views = frec = None
    if ctx.arm == "framed-chained":
        compute_stage_frames, render_role, scene_extent = (
            frames_renderer or ec.stage_frames_renderer()[:3])
        z, size = _table(ctx)
        scene = _scene(ctx, z, size)
        _, radius = scene_extent(scene, poses)
        sf = compute_stage_frames(((stage, role),), state.selections, state.objects,
                                  ctx.role_index, ctx.asset_dirs, ctx.symbols, poses,
                                  radius)[role]
        frec = render_role(scene, role, sf, poses, rd / "reemit_frames", "preview chain")
        views = [Path(frec["views"][v]) for v in frec["views"]] or None
        if views is None:
            frec = None
    anchor = w_anchor = None
    if stage.passive and e_point is not None:
        e_tok = ec.point_token(stage.active, ctx.symbols, e_point) or f"{stage.active}.anchor"
        w_tok = ec.point_token(stage.passive, ctx.symbols, w_point) or f"{stage.passive}.anchor"
        anchor, w_anchor = (e_tok, e_point), (w_tok, w_point, poses[stage.passive])
    chained = _chained(ctx, state, role)
    entry_text = None if c.synthesized else vlm_compat.entry_line(
        stage.active, poses[stage.active], ctx.symbols[stage.active].axes, chained=chained,
        tol_rad=ec.ALIGN_TOL_RAD, anchor=anchor, w_anchor=w_anchor)
    prior, prior_src = prior_emission_raw(ctx, c.emission, vocab)
    text = (REEMIT_COMPILED.format(diagnosis=diagnosis) if c.compiled is not None else
            REEMIT_UNCOMPILED.format(error=(c.compile_error.text() if c.compile_error is not None
                                            else c.error), diagnosis=diagnosis))
    feedback = [(prior, text)]
    rec.update(arm=ctx.arm, chained=chained, prior=prior_src, prior_raw=prior, followup=text,
               views=[str(v) for v in views or []],
               entry_pose=np.round(np.asarray(poses[stage.active], float), 6).tolist())
    rejections, em = [], None
    for attempt in range(1 + ec.COMPILE_RETRIES):
        try:
            em = client.emit_constraints(stage, vocab, selection=sel, view_paths=views,
                                         rejections=rejections, frames=frec, entry=entry_text,
                                         feedback=feedback)
        except vlm_compat.VLMError as e:
            rec["error"] = f"call #3 failed: {e}"
            break
        raw = client.logs[-1].raw
        try:
            cs = ec.compile_stage(em, ctx.symbols, poses, w_point=w_point, e_point=e_point,
                                  entry_synthesized=c.synthesized)
            ec.check_pair_consistency(cs)
        except ec.CompileError as e:
            rec["attempts"].append({"raw": raw, "compile_error": e.text()})
            rejections.append((raw, e.text()))
            continue
        rec["attempts"].append({"raw": raw, "compiled": True})
        ctx.raw_emissions[em.name] = raw
        break
    rec["emission"] = asdict(em) if em is not None else None
    return em, rec


# ------------------------------------------------------------------ the loop

def run_critic(ctx: CriticContext, mode: str = "live", responses: Path | None = None,
               budget: int = CRITIC_BUDGET, frames_renderer=None, renderer=None,
               verdict_only: bool = False) -> dict:
    """mode: 'live' (paid), 'replay' (responses dir), 'dry_run' (round 1's
    prompt and images, no call). verdict_only: one critic call on the
    specification as given, its verdict recorded (status 'verdict'), no
    edit applied and no further round (critic_cases.py). Returns the summary dict; on 'accepted'
    it also carries 'chain' (the compiled stages of the accepted
    specification) and 'state'."""
    require_critic()
    crit_dir = ctx.out / "critic"
    state = spec_state(ctx)
    vocab = vlm_compat.Vocabulary.from_asset_dirs(ctx.asset_dirs)
    colors = object_colors(ctx.asset_dirs)
    ledger = cc.LoopLedger(budget=budget)
    summary = {"mode": mode, "budget": budget,
               "model": vlm_compat.MODEL if mode == "live" else None,   # replay/dry: no model ran
               "sim": cc.provenance(), "notes": list(ctx.notes), "rounds": [],
               "edits_applied": [], "status": None, "rounds_to_accept": None,
               "point_gaps": {}, "failure": None,
               "calls": {"critic": 0, "call3_reemit": 0}}
    if mode == "dry_run":
        crit_t = emit_t = None
    else:
        crit_t, emit_t = make_transports(mode, responses)
        if mode == "live":
            if verdict_only:
                print(f"[critic] LIVE: 1 critic call (verdict only), up to "
                      f"{1 + vlm_compat.MAX_PARSE_RETRIES} with parse retries")
            else:
                print(f"[critic] LIVE: 1 critic call per round if every response parses; this "
                      f"run makes 1 (accept) to {budget + 1} (two rounds of edits and a final "
                      f"check), up to {(budget + 1) * (1 + vlm_compat.MAX_PARSE_RETRIES)} with "
                      "parse retries, plus 1 call #3 call per re-emitted stage")
    crit_client = vlm_compat.Client(transport=crit_t) if crit_t else None
    emit_client = vlm_compat.Client(transport=emit_t) if emit_t else None
    render = renderer or render_round
    n, chain = 0, []

    def save():
        summary["calls"] = {"critic": len(crit_t.raw) if crit_t else 0,
                            "call3_reemit": len(emit_t.raw) if emit_t else 0}
        if mode == "live":                # the paid record, replayable as files
            for kind, t in (("critic", crit_t), ("emit", emit_t)):
                for i, raw in enumerate(t.raw if t else []):
                    p = crit_dir / "responses" / f"{kind}_{i + 1:02d}.txt"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(raw)
        _w(crit_dir / "critic.json", {k: v for k, v in summary.items()
                                      if k not in ("chain", "state")})

    try:
        while True:
            n += 1
            rd = crit_dir / f"round_{n}"
            rnd = {"round": n, "dir": str(rd)}
            summary["rounds"].append(rnd)
            chain = compile_chain(ctx, state)
            previews = preview_inputs(ctx, state, chain)
            manifest = render(ctx, chain, previews, rd / "images")
            report = preview_report(state, chain, previews)
            _w(rd / "report.json", report)
            rnd["compile_errors"] = {c.role: c.error for c in chain if c.error}
            gaps = point_gaps(ctx, chain)
            rnd["point_gaps"] = gaps
            if n == 1:
                summary["point_gaps"]["before"] = gaps
            views = views_from_manifest(manifest, rd / "images", chain, colors)
            system, messages = cc.build_critic_prompt_v2(report, views, vocab)
            req = _messages_for_log(system, messages, views)
            _w(rd / "prompt.json", req)
            (rd / "prompt.txt").write_text(_prompt_text(req))
            if mode == "dry_run":
                summary["status"] = "dry_run"
                print(f"[critic] dry run: round 1's prompt and {len(views)} image(s) -> {rd}; "
                      "no API call made")
                return summary
            try:
                verdict = crit_client.critique_preview_v2(report, views, vocab, state, ledger)
            except vlm_compat.VLMError as e:
                log = crit_client.logs[-1]
                _w(rd / "rejections.json", log.rejections)
                summary["status"] = "failed"
                summary["failure"] = {"type": None, "routes_to": None, "phase": "critic",
                                      "round": n, "reason": str(e)}
                print(f"[critic] round {n}: {e}")
                return summary
            log = crit_client.logs[-1]
            _w(rd / "rejections.json", log.rejections)
            _w(rd / "verdict.json", _verdict_rec(verdict))
            (rd / "raw_response.txt").write_text(log.raw or "")
            rnd.update(verdict=verdict.verdict, attempts=log.attempts,
                       rejections=list(log.rejections), diagnosis=verdict.diagnosis)
            print(f"[critic] round {n}: {verdict.verdict}"
                  + (f" ({len(verdict.edits)} edit(s): "
                     + ", ".join(f"{e.action} {e.target} {e.token or ''}".strip()
                                 for e in verdict.edits) + ")" if verdict.edits else "")
                  + (f"; {len(log.rejections)} parse rejection(s)" if log.rejections else ""))
            if verdict_only:
                summary["status"] = "verdict"
                summary["verdict"] = _verdict_rec(verdict)
                return summary
            try:
                outcome = ledger.admit(verdict)
            except cc.EditRefusal as e:
                outcome = {"type": "preview_rejected", "routes_to": None, "source": "critic",
                           "rounds_used": ledger.rounds_used, "budget": ledger.budget,
                           "refusal": {"code": e.code, "reason": e.reason}}
            if outcome == "accept":
                summary["status"] = "accepted"
                summary["rounds_to_accept"] = n
                summary["point_gaps"]["after"] = gaps
                summary["chain"], summary["state"] = chain, state
                return summary
            if isinstance(outcome, dict):
                summary["status"] = "preview_rejected"
                summary["failure"] = {**outcome, "phase": "preview", "round": n}
                print(f"[critic] preview_rejected after {ledger.rounds_used} round(s) of edits")
                return summary
            try:
                res = cc.apply_edits(state, verdict.edits, verdict.diagnosis)
            except cc.EditRefusal as e:
                _w(rd / "edits.json", {"refused": {"code": e.code, "reason": e.reason}})
                summary["status"] = "preview_rejected"
                summary["failure"] = {"type": "preview_rejected", "routes_to": None,
                                      "phase": "preview", "round": n,
                                      "refusal": {"code": e.code, "reason": e.reason}}
                return summary
            ledger.record(verdict)
            applied = [{"round": n, **_edit_rec(e)} for e in verdict.edits]
            summary["edits_applied"] += applied
            _w(rd / "edits.json", {"applied": applied, "changed_roles": list(res.changed_roles),
                                   "reemit": res.reemit})
            state = res.state
            reemits = []
            order = [r for _, r in ctx.stages if r in res.reemit]
            for role in order:            # each at its entry in the chain as it now stands
                c = next((x for x in compile_chain(ctx, state) if x.role == role), None)
                if c is None:
                    continue
                em, rec = reemit(ctx, state, c, res.reemit[role], emit_client, rd,
                                 frames_renderer)
                reemits.append(rec)
                if em is not None:
                    state = replace(state, emissions={**state.emissions, role: em})
            if reemits:
                _w(rd / "reemit.json", reemits)
    finally:
        save()


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    """export-gripper: write <run>/critic_assets/gripper_2f85.obj on the host
    (which has the URDF), so the ros2 container renders the gripper from the
    run folder."""
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export-gripper", help="the 2F-85 below tool0 as one OBJ in the run")
    ex.add_argument("--run", required=True)
    ex.add_argument("--gripper-urdf",
                    default=str(here.parents[1] / "curobo_runtime/robot/ur5e_robotiq.urdf"))
    ex.add_argument("--gripper-meshes",
                    default=str(here.parents[1] / "test/curobo_incontainer/robotiq/meshes/collision"))
    a = ap.parse_args(argv)
    require_critic()
    out = export_gripper(Path(a.gripper_urdf), Path(a.gripper_meshes),
                         Path(a.run) / CRITIC_ASSETS)
    print(f"[critic] gripper -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
