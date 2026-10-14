#!/usr/bin/env python3
"""M3 step 2: the stage-1 grasp TSR on a hardware run, anchored at call #2's
pick, in the robot base frame.

    <run>/stage_plan/plan.json          plan_scene --grounding (roles bound)
    <run>/call2/selections.json         select_scene: the grasp role's mark
    <run>/call2/selections.log.json     select_scene: the pick's real-frame pixel
    <run>/summary.json                  packet.t_base_cam.T, objects[key].pose_on_final
            |
            v  resolve_selection (sim, the same call select_scene made)
            v  handle_grasp_tsr (sim, via grasping_compat), DEFAULT BOUNDS
    <out>/grasp_tsr.yaml       tsr_spec format, frame_id base_link: what
                               tsr_from_yaml publishes and grasp_filter reads
    <out>/grasp_tsr.log.json   inputs, numbers, checks, sim provenance

Frames. The grasp frame is resolved in the object's CANONICAL coordinates
(z = table up, the up the sim's 35 deg elevation is measured from). It is
placed in the base frame by

    T_base_canon = T_base_cam @ cam_T_obj @ [R_body_from_canon | 0]

(x_body = R x_canon, as select_scene and check_symbols map points). The
approach is the sim's: the horizontal direction from the handle point toward
the canonical origin (plan_pour_tea's a_body = -handle.point).

Bounds are the sim's handle_grasp_tsr defaults, unchanged -- the hardware
grasp region is the sim's, so the two stay comparable. With call #2's
point-only scheme the frame's axis (the TSR's "bar", w z) is the canonical
up by explicit default (resolve_selection's axis_source says so).

Gates, each a refusal naming its fix (exit 2):
  plan, grounding   select_scene.load_inputs, unchanged
  selection         the grasp role is bound, selections.json has it, and its
                    mark is the one selections.log.json resolved and drew
  t_base_cam        summary.json's packet must carry it (run_scene writes it;
                    older runs say "absent_in_source"): without it the TSR
                    cannot be placed in the base frame
  round trip        the written YAML, read back by tsr_spec, must give the
                    same T0_w / Tw_e / Bw
  pixel             the TSR origin, taken back through T_base_cam^-1 and K,
                    must land within PIXEL_TOL of the pixel select_scene drew.
                    It checks the composition order above (a reversed or
                    transposed factor moves the point); it cannot check
                    T_base_cam itself, which cancels.

Exit: 0 written; 2 refused.

In the ros2 container (no GL, no API key):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/grasp_tsr.py --run /data/runs/<stamp>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from manip_bridge import grasping_compat as gc
    from manip_bridge import selection_compat as sc
    from manip_bridge.tsr_spec import AXES, tsr_from_spec
except ImportError as e:
    sys.exit(f"grasp_tsr: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_symbols as cs  # noqa: E402  -- project(), one definition
import select_scene as ss  # noqa: E402  -- load_inputs / real_frame: the same gates and reads

ROLE = "grasp"
PIXEL_TOL = 2.0          # px; select_scene logs the pixel rounded to an int
ROUND_TRIP_TOL = 1e-9


class GraspTsrRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def t_base_cam(run: Path) -> np.ndarray:
    summary = json.loads((run / "summary.json").read_text())
    tbc = (summary.get("packet") or {}).get("t_base_cam") or {}
    if tbc.get("T") is None:
        why = tbc.get("source") or "no packet.t_base_cam"
        raise GraspTsrRefusal(f"{run / 'summary.json'}: no T_base_cam ({why}) -- the TSR cannot "
                              "be placed in the base frame. Re-capture with run_scene and the "
                              "hand-eye TF up")
    T = np.asarray(tbc["T"], float).reshape(4, 4)
    if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-5):
        raise GraspTsrRefusal(f"{run / 'summary.json'}: packet.t_base_cam.T is not a rigid transform")
    return T


def _bw_spec(Bw: np.ndarray) -> dict:
    """6x2 radians/metres -> tsr_spec's bw mapping (metres, degrees)."""
    out = {}
    for a, (lo, hi) in zip(AXES, Bw):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            out[a] = "free"
        elif a in ("roll", "pitch", "yaw"):
            out[a] = [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
        else:
            out[a] = [float(lo), float(hi)]
    return out


def tsr_spec_doc(tsr, obj: str) -> dict:
    return {"name": tsr.name, "frame_id": "base_link", "topic": f"/tsr/{obj}/grasp",
            "t0_w": {"matrix": np.asarray(tsr.T0_w, float).reshape(16).tolist()},
            "tw_e": {"matrix": np.asarray(tsr.Tw_e, float).reshape(16).tolist()},
            "bw": _bw_spec(np.asarray(tsr.Bw, float))}


def grasp_tsr(run, plan=None, selections=None, out=None, elevation_deg=35.0):
    """Returns (exit_code, report, tsr) -- tsr is the sim's TSR object, for tests."""
    run = Path(run)
    plan_path = Path(plan) if plan else run / "stage_plan" / "plan.json"
    sel_path = Path(selections) if selections else run / "call2" / "selections.json"
    log_path = sel_path.with_name("selections.log.json")
    out = Path(out) if out else run / "m3"

    try:
        b, _, objs = ss.load_inputs(run, plan_path)
    except ss.SelectRefusal as e:
        raise GraspTsrRefusal(str(e)) from None
    if ROLE not in b.roles:
        raise GraspTsrRefusal(f"{plan_path}: no {ROLE!r} role bound (roles {sorted(b.roles)})")
    obj = b.roles[ROLE][0]
    o = objs[obj]

    for p in (sel_path, log_path):
        if not p.is_file():
            raise GraspTsrRefusal(f"{p} not found -- run select_scene.py first")
    sels = sc.load_selections(sel_path)
    if ROLE not in sels:
        raise GraspTsrRefusal(f"{sel_path}: no {ROLE!r} selection (has {sorted(sels)})")
    sel = sels[ROLE]
    logged = (json.loads(log_path.read_text()).get("resolved") or {}).get(ROLE)
    if not logged or logged.get("candidate_id") != sel.candidate_id:
        raise GraspTsrRefusal(f"{log_path} does not record mark {sel.candidate_id} for {ROLE!r} "
                              f"(has {logged and logged.get('candidate_id')}) -- the selections "
                              "and their log come from different select_scene runs")

    canon = o["canon_dir"]
    try:
        rf = sc.resolve_selection(sel, sc.load_pool(canon), sc.load_symbols(canon))
    except sc.ResolutionError as e:
        raise GraspTsrRefusal(f"{ROLE!r}: mark {sel.candidate_id} does not resolve: {e}") from e
    handle = rf.frame

    T_bc = t_base_cam(run)
    K, cam_T_obj, _, _ = ss.real_frame(run, o["key"])
    R4 = np.eye(4)
    R4[:3, :3] = o["R"]
    T_base_canon = T_bc @ cam_T_obj @ R4

    # the sim's approach (plan_pour_tea): handle point -> canonical origin,
    # expressed in the handle frame; nominal_grip_in_world drops its bar part
    a_canon = -np.asarray(handle.point, float)
    a_h = handle.T()[:3, :3].T @ a_canon
    tsr = gc.handle_grasp_tsr(T_base_canon, handle, a_h, elevation=np.deg2rad(elevation_deg))

    doc = tsr_spec_doc(tsr, obj)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    back, frame_id, _ = tsr_from_spec(yaml.safe_load(text))
    rt = {"T0_w": float(np.abs(back.T0_w - tsr.T0_w).max()),
          "Tw_e": float(np.abs(back.Tw_e - tsr.Tw_e).max()),
          "Bw": float(np.abs(back.Bw - np.asarray(tsr.Bw)).max())}
    if frame_id != "base_link" or max(rt.values()) > ROUND_TRIP_TOL:
        raise GraspTsrRefusal(f"tsr_spec read back a different TSR: max |diff| {rt}, "
                              f"frame_id {frame_id!r}")

    origin = tsr.T0_w[:3, 3]
    p_cam = np.linalg.inv(T_bc) @ np.r_[origin, 1.0]
    px, z = cs.project(K, p_cam[:3])
    px = px[0]
    d_px = float(np.hypot(*(px - np.asarray(logged["pixel"], float))))
    if not (z[0] > 0 and d_px <= PIXEL_TOL):
        raise GraspTsrRefusal(f"TSR origin projects to pixel {np.round(px, 1).tolist()} (z "
                              f"{z[0]:.3f} m), {d_px:.1f} px from select_scene's "
                              f"{logged['pixel']} (tol {PIXEL_TOL}) -- the base-frame chain "
                              "T_base_cam . cam_T_obj . R_body_from_canon is composed wrong")

    T0_e = tsr.T0_w @ tsr.Tw_e
    approach = T0_e[:3, 2]
    report = {
        "role": ROLE, "object": obj, "key": o["key"], "candidate_id": sel.candidate_id,
        "menu_tag": logged.get("menu_tag"), "axis_source": rf.axis_source,
        "elevation_deg_requested": elevation_deg,
        # measured in the base frame: base z is up, canonical z is the table normal
        "elevation_deg_in_base": float(np.rad2deg(np.arcsin(np.clip(-approach[2], -1, 1)))),
        "origin_base_m": origin.tolist(), "bar_axis_base": tsr.T0_w[:3, 2].tolist(),
        "approach_base": approach.tolist(),
        "bounds": doc["bw"], "pixel": {"tsr_origin": np.round(px, 2).tolist(),
                                       "select_scene": logged["pixel"], "distance": d_px},
        "round_trip_max_diff": rt,
        "T0_w": tsr.T0_w.tolist(), "Tw_e": tsr.Tw_e.tolist(), "Bw": np.asarray(tsr.Bw).tolist(),
        "inputs": {str(p): _sha256(p) for p in (plan_path, sel_path, log_path,
                                                Path(o["frames"]), run / "summary.json")},
        "sim": {"grasping": gc.provenance(), "selection": sc.provenance()},
    }
    out.mkdir(parents=True, exist_ok=True)
    head = ("# Stage-1 grasp TSR -- written by outputs/runs/grasp_tsr.py; do not hand-edit.\n"
            f"# {obj} mark {sel.candidate_id} [{logged.get('menu_tag')}], sim handle_grasp_tsr "
            f"defaults, elevation {elevation_deg:g} deg.\n"
            f"#   ros2 run manip_bridge tsr_from_yaml -- {out / 'grasp_tsr.yaml'}\n")
    (out / "grasp_tsr.yaml").write_text(head + text)
    (out / "grasp_tsr.log.json").write_text(json.dumps(report, indent=2) + "\n")
    report["out"] = str(out)
    return 0, report, tsr


def _fmt(v, unit):
    if v == "free":
        return "free"
    lo, hi = v
    k = 1000.0 if unit == "mm" else 1.0
    return f"[{lo * k:+.1f}, {hi * k:+.1f}] {unit}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--plan", default=None, help="default <run>/stage_plan/plan.json")
    ap.add_argument("--selections", default=None, help="default <run>/call2/selections.json")
    ap.add_argument("--out", default=None, help="default <run>/m3")
    ap.add_argument("--elevation-deg", type=float, default=35.0,
                    help="approach pitch below horizontal (sim plan_pour_tea default 35)")
    a = ap.parse_args(argv)
    try:
        code, r, _ = grasp_tsr(a.run, a.plan, a.selections, a.out, a.elevation_deg)
    except GraspTsrRefusal as e:
        print(f"[grasp-tsr] REFUSED: {e}", file=sys.stderr)
        return 2
    o = np.asarray(r["origin_base_m"]) * 1000.0
    print(f"[grasp-tsr] {r['object']} mark {r['candidate_id']} [{r['menu_tag']}]")
    print(f"    axis: {r['axis_source']}")
    print(f"    origin (base, mm): [{o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}]")
    print(f"    bar axis (base): {np.round(r['bar_axis_base'], 3).tolist()}")
    print(f"    approach (base): {np.round(r['approach_base'], 3).tolist()}, "
          f"{r['elevation_deg_in_base']:.1f} deg below horizontal "
          f"(requested {r['elevation_deg_requested']:g})")
    print("    bounds in w (z = bar):")
    for ax in AXES:
        print(f"      {ax:5s} {_fmt(r['bounds'][ax], 'mm' if ax in ('x', 'y', 'z') else 'deg')}")
    p = r["pixel"]
    print(f"    pixel check: TSR origin -> {p['tsr_origin']}, select_scene drew {p['select_scene']}, "
          f"{p['distance']:.2f} px (tol {PIXEL_TOL})")
    print(f"[grasp-tsr] wrote {r['out']}/grasp_tsr.yaml (+ grasp_tsr.log.json)")
    return code


if __name__ == "__main__":
    sys.exit(main())
