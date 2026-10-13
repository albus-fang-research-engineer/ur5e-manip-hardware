#!/usr/bin/env python3
"""VLM call #2 on a hardware run: for each bound role, the model picks the
interaction point from marked canonical views of the grounded object, and
the pick is resolved and drawn on the real frame (milestone 2).

    <run>/stage_plan/plan.json          plan_scene --grounding (roles bound)
    <run>/grounding_vlm/canon/<name>/   ground_asset: frames.json, candidates.json,
                                        <name>.xml + meshes/ links -- a full asset dir
            |
            v  per object: run_vlm (sim, via candidates_compat) on the canonical copy
    <out>/vlm/<name>/{<view>.png, manifest.json}   8 marked views, <= 8 marks
            |
            v  per role: Client.select_point_axis (sim, point-only scheme)
            v  resolve_selection (sim): the typed gate
    <out>/selections.json            role-keyed, EXACTLY sim's format (load_selections)
    <out>/selections.log.json        model, stage sent, menu, raw responses, resolved
                                     point (canonical / body / pixel), sim provenance
    <out>/selection_<role>.png       the pick on the real RGB frame

Gates, each a refusal naming its fix (exit 2):
  plan      read with stageplan_compat.load_structural -- a blocked or
            unbound plan is refused by plan_stages' own reader
  grounding each bound object's grounding_vlm/<key>/frames.json must still
            hash to what plan_scene --grounding recorded: a grounding redone
            after binding would otherwise reach call #2 silently
  menu      the menu rebuilt from candidates.json (vlm_subset over the
            object's grounded parts, menu_from_pool) must equal the render
            manifest's: marks drawn and IDs the parser accepts are one
            artifact (sim select_frames' rule)
  pick      an off-menu ID is rejected by the sim parser and retried (logged);
            a licit ID that cannot anchor a frame fails resolve_selection

The real-frame check (report only, never blocking): the selected point,
canonical -> body (R_body_from_canon from the grounding provenance) ->
camera (summary pose_on_final) -> pixel (summary K), drawn on rgb.png with
the other menu marks, plus in_outline: whether that pixel is inside the
object's mask with enclosed holes filled, dilated as check_symbols does
(its outline, its mask_dilate_frac). This image is where "the model picked
the wrong part" is caught -- including the known gap of structural role
binding ("handle dropped, body survives"; see the README).

Point-only scheme: the model picks a mark, no axis; resolve_selection
anchors the frame's axis on the canonical up BY EXPLICIT DEFAULT and says
so in axis_source. Direction semantics are call #3's; M3's grasp takes the
point from here and its approach from its own corridor TSR.

Record/replay as plan_scene: --replay LOG feeds the logged responses back
(no API call). Rendering still runs (GL), deterministically.

Exit: 0 every bound role selected and resolved; 2 refused.

In the ros2 container (GL for the renders; ANTHROPIC_API_KEY for a live call):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/select_scene.py --run /data/runs/<stamp>
    python3 /data/runs/select_scene.py --run /data/runs/<stamp> \\
        --replay /data/runs/<stamp>/call2/selections.log.json
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from manip_bridge import selection_compat as sc
    from manip_bridge import stageplan_compat as spc
    from manip_bridge import vlm_compat
    from manip_bridge.vlm_record import RecordingTransport, ReplayTransport
except ImportError as e:
    sys.exit(f"select_scene: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_symbols as cs  # noqa: E402  -- project() and the outline's dilation, one definition


class SelectRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _backend():
    from manip_bridge import candidates_compat
    return candidates_compat.render_menu, candidates_compat.provenance


# ------------------------------------------------------------------ inputs

def load_inputs(run: Path, plan_path: Path) -> tuple:
    """(bindings, plan doc, per-object grounding record)."""
    if not plan_path.is_file():
        raise SelectRefusal(f"{plan_path} not found -- run plan_scene.py, then the printed "
                            "commands, then plan_scene.py --replay ... --grounding")
    try:
        b = spc.load_structural(plan_path)
    except SystemExit as e:                       # plan_stages' reader refuses by exiting
        raise SelectRefusal(f"plan_stages' reader refused {plan_path}: {e}") from None
    doc = json.loads(plan_path.read_text())
    grec = (doc.get("hardware") or {}).get("grounding")
    if not grec:
        raise SelectRefusal(f"{plan_path} has no hardware.grounding record -- roles were not "
                            "bound by plan_scene --grounding")
    objs = {}
    for role, (obj, _) in b.roles.items():
        if obj in objs:
            continue
        rec = (grec.get("objects") or {}).get(obj)
        if rec is None:
            raise SelectRefusal(f"role {role!r} binds {obj!r}, which has no grounding record "
                                f"in {plan_path}")
        fp = Path(rec["frames"])
        if not fp.is_file():
            raise SelectRefusal(f"{fp} is gone since roles were bound -- re-run the printed "
                                "ground_asset command, then plan_scene --grounding")
        if _sha256(fp) != rec["frames_sha256"]:
            raise SelectRefusal(f"{fp} changed since plan_scene --grounding bound the roles "
                                "(sha256 differs) -- re-run plan_scene --replay ... "
                                "--grounding so the binding describes this grounding")
        prov = (json.loads(fp.read_text()).get("provenance") or {}).get("canonical") or {}
        R = np.asarray(prov.get("R_body_from_canon"), float)
        if R.shape != (3, 3):
            raise SelectRefusal(f"{fp}: no provenance.canonical.R_body_from_canon")
        objs[obj] = {"key": rec["key"], "canon_dir": Path(rec["canon_dir"]), "R": R,
                     "frames": str(fp)}
    return b, doc, objs


def real_frame(run: Path, key: str):
    """(K, cam_T_obj, rgb, mask or None) from the run, as check_symbols reads them."""
    summary = json.loads((run / "summary.json").read_text())
    K = np.asarray(summary["frame"]["K"], float)
    po = ((summary.get("objects") or {}).get(key) or {}).get("pose_on_final")
    if not po:
        raise SelectRefusal(f"summary has no objects[{key}].pose_on_final")
    T = np.asarray(po["cam_T_obj"], float).reshape(4, 4)
    rgb = Image.open(run / "rgb.png").convert("RGB")
    mp = run / f"mask_{key}.png"
    mask = (np.array(Image.open(mp).convert("L")) > 0) if mp.is_file() else None
    return K, T, rgb, mask


def to_pixels(X_canon, R, T, K):
    """canonical -> body (R x) -> camera (cam_T_obj) -> pixels."""
    Xb = np.atleast_2d(X_canon) @ R.T
    Xc = Xb @ T[:3, :3].T + T[:3, 3]
    px, z = cs.project(K, Xc)
    return Xb, px, z


def outline(mask):
    from scipy.ndimage import binary_dilation, binary_fill_holes
    ys, xs = np.nonzero(mask)
    diag = float(np.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 0.0
    it = max(1, int(round(cs.values()["mask_dilate_frac"] * diag)))
    return binary_dilation(binary_fill_holes(mask), iterations=it)


def draw_overlay(rgb, menu_px, ids, sel_id, tag, path: Path):
    img = rgb.copy()
    dr = ImageDraw.Draw(img)
    for (u, v), i in zip(menu_px, ids):
        if i == sel_id:
            continue
        dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 255, 255), width=2)
        dr.text((u + 6, v - 6), str(i), fill=(255, 255, 255))
    (u, v) = menu_px[ids.index(sel_id)]
    for r, col in ((11, (0, 0, 0)), (9, (255, 40, 40))):
        dr.ellipse([u - r, v - r, u + r, v + r], outline=col, width=3)
    dr.text((u + 13, v - 8), f"{sel_id} {tag}", fill=(255, 40, 40))
    img.save(path)


# ------------------------------------------------------------------ the call

def select_scene(run, plan=None, replay=None, out=None, transport=None, render=None):
    """Returns (exit_code, selections, report). `transport` and `render`
    are injectable for tests."""
    run = Path(run)
    plan_path = Path(plan) if plan else run / "stage_plan" / "plan.json"
    out = Path(out) if out else run / "call2"
    b, doc, objs = load_inputs(run, plan_path)
    if render is None:
        render, render_prov = _backend()
    else:
        render_prov = lambda: {"render": "injected"}            # noqa: E731

    replayed_from, model = None, vlm_compat.MODEL
    if replay:
        try:
            log = json.loads(Path(replay).read_text())
        except (OSError, ValueError) as e:
            raise SelectRefusal(f"--replay {replay}: unreadable ({e})") from e
        if not log.get("raw_responses"):
            raise SelectRefusal(f"--replay {replay}: no raw_responses recorded")
        rec = ReplayTransport(log["raw_responses"])
        model = log.get("model") or model
        replayed_from = str(Path(replay).resolve())
    else:
        rec = RecordingTransport(transport)
    client = vlm_compat.Client(transport=rec, model=model)

    # one menu + render per object (every role on an object shares it, as in sim)
    inputs = {}
    for obj, o in objs.items():
        parts = list(b.object_parts.get(obj, ()))
        canon = o["canon_dir"]
        try:
            pool = sc.load_pool(canon)
        except sc.ResolutionError as e:
            raise SelectRefusal(f"{obj}: {e}") from e
        menu = sc.menu_from_pool(sc.vlm_subset(pool, parts))
        man, mpath = render(canon.name, canon, parts, out / "vlm")
        if man.get("parts_filter") != parts or man.get("menu") != {str(i): t for i, t in menu.items()}:
            raise SelectRefusal(f"{mpath}: rendered marks {man.get('menu')} (parts "
                                f"{man.get('parts_filter')}) differ from the menu rebuilt from "
                                f"{canon / 'candidates.json'} ({menu}, parts {parts}) -- marks "
                                "and accepted IDs must be one artifact")
        views = [Path(v["path"]) for _, v in sorted(man["views"].items())]
        missing = [str(p) for p in views if not p.is_file()]
        if missing:
            raise SelectRefusal(f"{mpath}: views missing on disk {missing}")
        inputs[obj] = {"pool": pool, "menu": menu, "manifest": str(mpath), "views": views,
                       "parts": parts, "symbols": sc.load_symbols(canon),
                       "vocab": vlm_compat.Vocabulary.from_asset_dirs({obj: canon}, menu=menu)}

    out.mkdir(parents=True, exist_ok=True)
    selections, resolved, stages, failure = {}, {}, {}, None
    for role, (obj, stage) in b.roles.items():
        i = inputs[obj]
        stages[role] = asdict(stage)
        try:
            sel = client.select_point_axis(stage, i["vocab"], i["views"])
        except vlm_compat.VLMError as e:
            failure = f"call #2 for {role!r} failed: {e}"
            break
        selections[role] = {"object": obj, **sc.selection_to_json(sel)}
        try:
            rf = sc.resolve_selection(sel, i["pool"], i["symbols"])
        except sc.ResolutionError as e:
            failure = f"{role!r}: mark {sel.candidate_id} does not resolve: {e}"
            break
        o = objs[obj]
        K, T, rgb, mask = real_frame(run, o["key"])
        ids = sorted(i["menu"])
        Xc = np.stack([i["pool"][k]["xyz"] for k in ids])
        Xb, px, z = to_pixels(Xc, o["R"], T, K)
        j = ids.index(sel.candidate_id)
        u, v = (int(round(px[j, 0])), int(round(px[j, 1])))
        H, W = (mask.shape if mask is not None else (rgb.height, rgb.width))
        in_img = 0 <= u < W and 0 <= v < H and z[j] > 0
        in_out = None if mask is None or not in_img else bool(outline(mask)[v, u])
        png = out / f"selection_{role}.png"
        draw_overlay(rgb, px.tolist(), ids, sel.candidate_id, i["menu"][sel.candidate_id], png)
        resolved[role] = {
            "object": obj, "key": o["key"], "candidate_id": sel.candidate_id,
            "menu_tag": i["menu"][sel.candidate_id], "rationale": sel.rationale,
            "xyz_canonical": np.round(rf.frame.point, 6).tolist(),
            "xyz_body": np.round(Xb[j], 6).tolist(),
            "pixel": [u, v], "in_image": bool(in_img), "in_outline": in_out,
            "axis_source": rf.axis_source, "frame_comment": rf.frame.comment,
            "overlay": str(png)}

    (out / "selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    log = {"model": client.model, "plan": str(plan_path.resolve()),
           "plan_sha": (doc.get("hardware") or {}).get("plan_sha"),
           "stages": stages, "parts": {o: i["parts"] for o, i in inputs.items()},
           "menus": {o: {str(k): t for k, t in i["menu"].items()} for o, i in inputs.items()},
           "manifests": {o: i["manifest"] for o, i in inputs.items()},
           "resolved": resolved, "failure": failure, "replayed_from": replayed_from,
           "client_logs": [asdict(entry) for entry in client.logs],
           "raw_responses": rec.raw,
           "sim": {"vlm": vlm_compat.provenance(), "selection": sc.provenance(),
                   "render": render_prov()}}
    (out / "selections.log.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    report = {"out": str(out), "resolved": resolved, "menus": log["menus"],
              "failure": failure}
    if failure:
        raise SelectRefusal(failure + f" (log written: {out / 'selections.log.json'})")
    return 0, selections, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--plan", default=None, help="default <run>/stage_plan/plan.json")
    ap.add_argument("--replay", default=None, help="a selections.log.json to replay (no API call)")
    ap.add_argument("--out", default=None, help="default <run>/call2")
    a = ap.parse_args(argv)
    try:
        code, sels, rep = select_scene(a.run, a.plan, a.replay, a.out)
    except SelectRefusal as e:
        print(f"[select-scene] REFUSED: {e}", file=sys.stderr)
        return 2
    for obj, menu in rep["menus"].items():
        print(f"[select-scene] {obj} menu:")
        for k, t in menu.items():
            print(f"    [{k}] {t}")
    for role, r in rep["resolved"].items():
        print(f"[select-scene] {role}: {r['object']} mark {r['candidate_id']} [{r['menu_tag']}]"
              f"{' (replayed)' if a.replay else ''}")
        print(f"    rationale: {r['rationale']}")
        print(f"    axis: {r['axis_source']}")
        print(f"    real frame: pixel {r['pixel']}, in_image {r['in_image']}, "
              f"in_outline {r['in_outline']} -> {r['overlay']}")
        if r["in_outline"] is False or not r["in_image"]:
            print(f"[select-scene] WARNING: {role}'s point is off the object's outline in the "
                  "real frame -- look at the overlay before using it")
    print(f"[select-scene] wrote {rep['out']}/selections.json (+ selections.log.json)")
    return code


if __name__ == "__main__":
    sys.exit(main())
