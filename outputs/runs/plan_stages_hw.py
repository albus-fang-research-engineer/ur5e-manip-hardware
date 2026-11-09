#!/usr/bin/env python3
"""Milestone 4, hand-authored arm: plan the carry (transport) and the pour on
a hardware run, offline, with the sim's stage compilers -- plan_pour_tea's
stages 2-3 on the cuRobo sidecar's plan_constrained. Nothing moves.

    <run>/summary.json            hand-eye, table plane, both registrations
    <run>/m3/reach.json           grasp_reach's plan + the taught grasp T_base_e
    <library>/<obj>.symbols_hand.json   hand_symbols.py (sim mesh calibration)
            |
            v  grasp_exec.build: reach -> approach -> close -> lift (the
            v  segments the grasp run executes); q_grasp, q_lift from them
            v  T_ee_body = inv(FK(q_grasp)) . T_base_teapot  (planned; canonical)
            v  pour_stages.TaskFrames (sim, via pour_stages_compat):
            v    transport pair at the rest pose, plan from q_lift
            v    pour pair FROZEN at the transport's actual end, plan from there
            v  a pour failure re-plans the transport with the next seed: the
            v  sidecar walks its own ranked goals but has no pour lookahead,
            v  so a different transport end stands in for the sim's ranking
            v  logging (no gate): every stage's path is timed by TOPP-RA as
            v  exec_plan will time it, sampled every 20 ms, and its excess
            v  against the stage's path TSR reported (max / median / p95,
            v  translation mm, rotation deg), at the waypoints and densely;
            v  and each joint's total travel vs net change (wrist winding)
    <out>/report.json      inputs, poses, every attempt's funnel + reason, metrics
    <out>/stages.json      the transport / pour joint paths (UR joint names)
    <out>/plan.json        exec_plan format, the whole chain: open, reach,
                           approach, close, lift, transport, pour, then the
                           pour and the transport reversed, lower, release,
                           retreat -- every motion gated

Frames. The sim's pair compilers assume body +z = up. They get each object
in its CANONICAL frame ([x=front, y=up x front, z=up], from the fixture's own
axes): T_base_canon = T_base_body . [R | 0], symbols x_c = R^T x_body, and
the grasp transform T_ee_canon = inv(FK(q_grasp)) . T_base_canon. The
attached spheres stay in the body frame (they only need tool0).

Scene. "registered" (default): the table plane as a slab and the mug's
tracked mesh at its registration -- the teapot is the carried object, so it
is not an obstacle. The live ESDF cannot be used offline: the teapot still
stands on the table, its voxels would collide with its own attached spheres,
and clear_region is re-filled by the next integrate (2 Hz). "live" plans on
the live ESDF as it is (use after the teapot has been lifted).

In the ros2 container (cuRobo sidecar up):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/plan_stages_hw.py --run /data/runs/<stamp>
    python3 /data/runs/exec_plan.py /data/runs/<stamp>/m4/plan.json --dry   # rviz

--method vlm plans the same two stages on call #3's regions (emit_scene's
artifact) instead, compiled at each stage's actual entry; see "VLM method"
below. It writes to <run>/m4_vlm:

    python3 /data/runs/plan_stages_hw.py --run /data/runs/<stamp> --method vlm \\
        --emissions /data/runs/<stamp>/call3/emissions.json

--critic preview (vlm only; default off, which plans exactly as before) runs
VLM call #4 first (critic_scene.py): preview the specification, let the
critic accept it or amend it with typed edits (2 rounds), then plan the
accepted specification. No-spend modes: --critic-dry-run writes round 1's
prompt and images and stops; --critic-responses DIR replays canned
responses. A paid run goes through scripts/vlm_run.sh:

    scripts/vlm_run.sh plan_stages_hw.py --run /data/runs/<stamp> --method vlm \\
        --emissions /data/runs/<stamp>/call3/emissions.json --critic preview
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from manip_bridge import pour_stages_compat as psc
    from manip_bridge import scene_ops as so
    from manip_bridge import ur_motion as um
except ImportError as e:
    sys.exit(f"plan_stages_hw: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]          # outputs/runs -> the repo (host); "/" in the ros2 container
CUROBO_ADDR = os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671")
CALL_TIMEOUT_MS = 300_000

# plan_pour_tea's hand-arm values (sim): the transport band and corridor,
# the pour target and the pour's smaller CBiRRT step
TRANSPORT_HEIGHT = (0.04, 0.10)
TRANSPORT_Z_CORRIDOR = (-0.02, 0.45)
TRANSPORT_EPS = 0.12            # manip_cbirrt's default, what the sim's transport used
POUR_EPS = 0.05
TILT_DEG = 95.0
TABLE_THICK_M = 0.04
TABLE_HALF_M = 0.6


class PlanRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block(R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _pose7(T) -> list:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [float(v) for v in (*T[:3, 3], w, x, y, z)]


def _tsr_msg(t) -> dict:
    return {"t0_w": np.asarray(t.T0_w, float).ravel().tolist(),
            "tw_e": np.asarray(t.Tw_e, float).ravel().tolist(),
            "bw": np.asarray(t.Bw, float).ravel().tolist(), "name": t.name}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ inputs


def object_keys(summary: dict) -> tuple[str, str]:
    """(teapot key, mug key) from the tracked meshes' names."""
    found = {}
    for k, o in (summary.get("objects") or {}).items():
        mesh = str(((o or {}).get("pose_on_final") or {}).get("mesh", ""))
        for name in ("teapot", "mug"):
            if name in Path(mesh).name:
                found[name] = k
    if set(found) != {"teapot", "mug"}:
        raise PlanRefusal(f"cannot tell teapot and mug apart from the tracked meshes ({found}); "
                          "pass --teapot-key/--mug-key")
    return found["teapot"], found["mug"]


def load_fixture(library: Path, mesh: Path) -> dict:
    fp = library / f"{mesh.stem}.symbols_hand.json"
    if not fp.is_file():
        raise PlanRefusal(f"{fp} missing: run hand_symbols.py for {mesh.name}")
    doc = json.loads(fp.read_text())
    if doc["provenance"]["mesh_sha256"] != _sha(mesh):
        raise PlanRefusal(f"{fp} was calibrated on a different mesh than {mesh}: re-run hand_symbols.py")
    R = np.asarray(doc["canonical"]["R_body_from_canon"], float)
    U, _, Vt = np.linalg.svd(R)
    doc["_R"], doc["_path"] = U @ Vt, str(fp)
    return doc


def _c(doc, section, name) -> np.ndarray:
    """A fixture symbol in canonical coordinates."""
    return doc["_R"].T @ np.asarray(doc[section][name]["xyz"], float)


def task_frames(tp: dict, mug: dict):
    """The sim's four-symbol composition for the hand arm (plan_pour_tea's
    else-branch), in canonical coordinates. secondary None = body -z there."""
    down = np.array([0.0, 0.0, -1.0])
    spout_tip = psc.Frame("teapot.spout_tip", _c(tp, "points", "spout_tip"),
                          _c(tp, "axes", "pour_axis"), down)
    tilt = psc.Frame("teapot.spout_tip/tilt", _c(tp, "points", "spout_tip"),
                     _c(tp, "axes", "tilt_axis"), _c(tp, "axes", "pour_axis"))
    opening = psc.Frame("mug.opening_center", _c(mug, "points", "opening_center"),
                        _c(mug, "axes", "up_axis"), down)
    rim = float(mug["quantities"]["rim_radius"]["value"])
    return psc.TaskFrames(spout_tip=spout_tip, tilt_frame=tilt, opening=opening,
                          rim_margin=0.5 * rim, emissions=None, symbols=None)


def registered_scene(summary, T_mug_body, mug_mesh: Path, near_xy) -> dict:
    import trimesh
    n, d = so.table_plane_base(summary)
    x, y = near_xy
    p0 = np.array([x, y, float(so.table_z((n, d), x, y))])
    z = n / np.linalg.norm(n)
    xa = np.cross([0.0, 1.0, 0.0], z)
    xa /= np.linalg.norm(xa)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([xa, np.cross(z, xa), z])
    T[:3, 3] = p0 - 0.5 * TABLE_THICK_M * z
    m = trimesh.load(str(mug_mesh), force="mesh", process=False)
    return {"cuboid": {"table": {"pose": _pose7(T),
                                 "dims": [2 * TABLE_HALF_M, 2 * TABLE_HALF_M, TABLE_THICK_M]}},
            "mesh": {"mug": {"pose": _pose7(T_mug_body),
                             "vertices": np.asarray(m.vertices, float).tolist(),
                             "faces": np.asarray(m.faces, int).tolist()}}}


# ----------------------------------------------------------------- metrics


def _stats(v) -> dict:
    v = np.asarray(v, float)
    return {"max": round(float(v.max()), 3), "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3)}


def excess_metrics(ch, Q, T_ee_canon, path_tsr, speed: float) -> dict:
    """The stage's path-TSR excess, at the planner's waypoints and along the
    TOPP-RA spline exec_plan will send (sampled every SAMPLE_DT). Logging
    only. Translation in mm (norm of x y z), rotation in deg (norm of r p y)."""
    def ex(Qs):
        E = np.array([path_tsr.excess(ch.fk(q) @ T_ee_canon) for q in Qs])
        return np.linalg.norm(E[:, :3], axis=1) * 1e3, np.degrees(np.linalg.norm(E[:, 3:], axis=1))

    out = {"n_waypoints": int(len(Q))}
    tw, rw = ex(Q)
    out["waypoints"] = {"trans_mm": _stats(tw), "rot_deg": _stats(rw)}
    try:
        t, Qs, _ = um.time_optimal(um.densify(Q), um.UR5E_VMAX * speed, um.UR5E_AMAX * speed)
    except um.MotionRefusal as e:
        out["dense"] = {"refused": str(e)}
        return out
    td, rd = ex(Qs)

    def at(i):
        return {"t_s": round(float(t[i]), 2), "fraction": round(i / (len(Qs) - 1), 3),
                "body_xyz": np.round((ch.fk(Qs[i]) @ T_ee_canon)[:3, 3], 4).tolist()}

    out["dense"] = {"n_samples": int(len(Qs)), "duration_s": round(float(t[-1]), 2),
                    "trans_mm": _stats(td), "rot_deg": _stats(rd),
                    "worst_trans_at": at(int(np.argmax(td))), "worst_rot_at": at(int(np.argmax(rd)))}
    return out


def joint_travel(Q) -> dict:
    """Per joint (UR names, DH order), in degrees: total distance travelled
    along the path, net start->end change, and the swept range. Logging
    only: a total far above |net| is a joint that winds and unwinds (CBiRRT
    can spin wrist 3 the way the cuRobo reach did) -- watch it in rviz."""
    Q = np.degrees(np.asarray(Q, float))
    total = np.abs(np.diff(Q, axis=0)).sum(axis=0)
    net, rng = Q[-1] - Q[0], Q.max(axis=0) - Q.min(axis=0)
    return {n: {"total": round(float(t), 1), "net": round(float(d), 1), "range": round(float(r), 1)}
            for n, t, d, r in zip(um.DH_ORDER, total, net, rng)}


def _fmt_travel(jt) -> str:
    return ", ".join(f"{n.replace('_joint', '')} {v['total']:.0f} (net {v['net']:+.0f})"
                     for n, v in jt.items())


def _fmt_metrics(m) -> str:
    w, d = m["waypoints"], m["dense"]
    s = (f"waypoints ({m['n_waypoints']}): trans max {w['trans_mm']['max']} / med "
         f"{w['trans_mm']['median']} mm, rot max {w['rot_deg']['max']} / med {w['rot_deg']['median']} deg")
    if "refused" in d:
        return s + f"; dense: TOPP-RA refused ({d['refused']})"
    return (s + f"; dense ({d['n_samples']} @ {um.SAMPLE_DT * 1000:.0f} ms): trans max "
            f"{d['trans_mm']['max']} / med {d['trans_mm']['median']} / p95 {d['trans_mm']['p95']} mm, "
            f"rot max {d['rot_deg']['max']} / med {d['rot_deg']['median']} / p95 "
            f"{d['rot_deg']['p95']} deg (worst trans at {d['worst_trans_at']['fraction']:.0%}, "
            f"rot at {d['worst_rot_at']['fraction']:.0%})")


# ---------------------------------------------------------------- planning


def plan_stage(cli, label, q_start, T_ee_canon, pair, spheres, scene, a, seed, eps) -> dict:
    msg = {"cmd": "plan_constrained", "q_start": np.asarray(q_start, float).tolist(),
           "joint_names": list(um.DH_ORDER), "T_ee_body": np.asarray(T_ee_canon, float).tolist(),
           "subgoal": _tsr_msg(pair.subgoal), "path": [_tsr_msg(pair.path)],
           "n_goal_samples": int(a.n_goal_samples), "timeout": float(a.timeout),
           "eps": float(eps), "seed": int(seed), "attached_spheres": np.asarray(spheres, float).tolist()}
    if scene is not None:
        msg["scene"] = scene
    t0 = time.time()
    rep = cli.call(msg, timeout_ms=CALL_TIMEOUT_MS)
    if not rep.get("ok", False):
        raise PlanRefusal(f"sidecar error on {label}: {rep.get('error')}")
    out = {"stage": label, "seed": seed, "eps": eps, "success": bool(rep.get("success")),
           "reason": rep.get("reason", ""), "funnel": rep.get("funnel"),
           "goal_index": rep.get("goal_index"), "max_excess_planner": rep.get("max_excess"),
           "tree_sizes": rep.get("tree_sizes"), "wall_s": round(time.time() - t0, 2)}
    if out["success"]:
        names = list(rep["joint_names"])
        P = np.asarray(rep["positions"], float)
        out["Q"] = um.to_dh_path(P, names)
    f = out["funnel"] or {}
    print(f"[{label}] seed {seed}: {'OK' if out['success'] else 'FAIL'} in {out['wall_s']} s -- funnel "
          f"{f.get('n_sampled')} sampled -> {f.get('n_ik')} IK -> {f.get('n_collision_free')} free -> "
          f"{f.get('n_contained')} contained; {out['reason']}")
    return out


def _traj_step(name, Q, gate=True, note="", min_duration_s=0.0):
    traj = {"joint_names": list(um.DH_ORDER), "positions": np.asarray(Q, float).tolist()}
    if min_duration_s:
        traj["min_duration_s"] = round(float(min_duration_s), 3)
    return {"name": name, "gate": gate, "note": note, "traj": traj}


def exec_plan_doc(reach_plan, segs, Q_tr, Q_pour, line_speed, source="hand arm") -> dict:
    """The blind lines keep grasp_exec's line-speed floor (length / line_speed).
    `source` names where the transport and pour regions came from."""
    by = {s.name: s for s in segs}
    md = {k: by[k].info["length_m"] / line_speed for k in ("approach", "lift")}
    steps = [{"name": "open", "gate": False, "gripper": {"gap_m": by["open"].gap_m}},
             {"name": "reach", "traj": {"joint_names": list(reach_plan["joint_names"]),
                                        "positions": reach_plan["positions"],
                                        "velocities": reach_plan["velocities"],
                                        "dt": reach_plan["dt"]}},
             _traj_step("approach", by["approach"].Q, note="BLIND", min_duration_s=md["approach"]),
             {"name": "close", "gripper": {"gap_m": 0.0, "effort_n": by["close"].effort_n,
                                           "expect_object": True}},
             _traj_step("lift", by["lift"].Q, note="BLIND", min_duration_s=md["lift"]),
             _traj_step("transport", Q_tr, note=f"CBiRRT, {source}"),
             _traj_step("pour", Q_pour, note=f"CBiRRT, {source}"),
             _traj_step("unpour", Q_pour[::-1], note="the pour reversed"),
             _traj_step("return", Q_tr[::-1], note="the transport reversed"),
             _traj_step("lower", by["lift"].Q[::-1], note="the lift reversed",
                        min_duration_s=md["lift"]),
             {"name": "release", "gate": False, "gripper": {"gap_m": by["release"].gap_m}},
             _traj_step("retreat", by["approach"].Q[::-1], note="the approach reversed",
                        min_duration_s=md["approach"])]
    return {"steps": steps}


def run(a) -> int:
    from manip_bridge.zmq_client import SidecarClient
    run_dir = Path(a.run)
    out = Path(a.out) if a.out else run_dir / "m4"
    summary = json.loads((run_dir / "summary.json").read_text())
    tk, mk = (a.teapot_key, a.mug_key) if a.teapot_key else object_keys(summary)
    try:                                   # scene_ops raises ValueError on a missing field
        tp_mesh, mug_mesh = Path(so.mesh_path(summary, tk)), Path(so.mesh_path(summary, mk))
        T_tp, T_mug = so.T_base_obj(summary, tk), so.T_base_obj(summary, mk)
    except ValueError as e:
        raise PlanRefusal(f"{run_dir / 'summary.json'}: {e}") from e
    tp, mug = load_fixture(Path(a.library), tp_mesh), load_fixture(Path(a.library), mug_mesh)

    # ---- poses (base_link), canonical frames
    T_tpC, T_mugC = T_tp @ _block(tp["_R"]), T_mug @ _block(mug["_R"])
    up_dev = {k: round(float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))), 2)
              for k, T in (("teapot", T_tpC), ("mug", T_mugC))}
    print(f"[poses] teapot {np.round(T_tp[:3, 3], 3).tolist()}, mug {np.round(T_mug[:3, 3], 3).tolist()}; "
          f"canonical up vs base z: {up_dev} deg")

    # ---- the grasp run's own segments: q_grasp, q_lift
    ch = um.chain()
    ge = _load("grasp_exec")
    reach = json.loads(Path(a.reach or run_dir / "m3" / "reach.json").read_text())
    segs = ge.build(reach["plan"], ch, deeper=a.deeper)
    by = {s.name: s for s in segs}
    q_grasp, q_lift = by["approach"].Q[-1], by["lift"].Q[-1]
    T_tool0_grasp = ch.fk(q_grasp)
    T_ee_canon = np.linalg.inv(T_tool0_grasp) @ T_tpC
    T_ee_body = so.T_ee_body(T_tool0_grasp, T_tp)
    print(f"[grasp] {reach['plan'].get('grasp')}: T_ee_body translation "
          f"{np.round(T_ee_body[:3, 3], 3).tolist()} m; lift ends tool0 at "
          f"{np.round(ch.fk(q_lift)[:3, 3], 3).tolist()}")

    cli = SidecarClient(CUROBO_ADDR, timeout_ms=CALL_TIMEOUT_MS)
    S_body = so.object_spheres(tp_mesh, run_dir / "objects" / tk / "spheres_body.json", cli.call)
    spheres = so.spheres_tool0(S_body, T_ee_body)
    scene = (registered_scene(summary, T_mug, mug_mesh, 0.5 * (T_tp[:2, 3] + T_mug[:2, 3]))
             if a.scene == "registered" else None)

    frames = task_frames(tp, mug)
    tilt_target = np.deg2rad(a.tilt_deg)
    report = {"run": str(run_dir), "arm": "hand (pour_stages)", "scene": a.scene,
              "inputs": {"reach": reach["plan"].get("grasp"), "teapot_key": tk, "mug_key": mk,
                         "fixtures": [tp["_path"], mug["_path"]], "sim_module": psc.SIM_MODULE,
                         "sim": psc.provenance(),
                         "tilt_deg": a.tilt_deg, "deeper_m": a.deeper},
              "poses": {"T_base_teapot": T_tp.tolist(), "T_base_mug": T_mug.tolist(),
                        "canonical_up_vs_base_z_deg": up_dev},
              "grasp": {"q_grasp": q_grasp.tolist(), "q_lift": q_lift.tolist(),
                        "T_ee_body": T_ee_body.tolist(), "T_ee_canon": T_ee_canon.tolist(),
                        "n_spheres": int(len(spheres))},
              "attempts": [], "status": "failed"}
    out.mkdir(parents=True, exist_ok=True)

    pair_tr = frames.transport(T_tpC, T_mugC, height=TRANSPORT_HEIGHT, z_corridor=TRANSPORT_Z_CORRIDOR)
    print(f"[transport] start vs path excess: {np.round(pair_tr.path.excess(ch.fk(q_lift) @ T_ee_canon), 4).tolist()}")
    def save_report():
        (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")

    done = None
    try:
        for k in range(a.tries):
            tr = plan_stage(cli, "transport", q_lift, T_ee_canon, pair_tr, spheres, scene, a, k,
                            TRANSPORT_EPS)
            att = {"seed": k, "transport": {kk: v for kk, v in tr.items() if kk != "Q"}}
            report["attempts"].append(att)
            if not tr["success"]:
                continue
            Q_tr = tr["Q"]
            T_entry = ch.fk(Q_tr[-1]) @ T_ee_canon
            pair_p = frames.pour(T_entry, T_mugC, tilt_target)
            att["pour_entry_excess"] = np.round(pair_p.path.excess(T_entry), 5).tolist()
            po = plan_stage(cli, "pour", Q_tr[-1], T_ee_canon, pair_p, spheres, scene, a, k, POUR_EPS)
            att["pour"] = {kk: v for kk, v in po.items() if kk != "Q"}
            if po["success"]:
                done = (Q_tr, pair_tr, po["Q"], pair_p)
                break
    except PlanRefusal as e:                 # keep the attempts that did run
        report["status"], report["error"] = "refused", str(e)
        save_report()
        raise
    if done is None:
        save_report()
        print(f"[plan] FAILED after {a.tries} transport seeds -> {out / 'report.json'}")
        return 2

    # the planner's waypoints are written as they are; exec_plan densifies and
    # times them, and excess_metrics samples exactly that
    Q_tr, pair_tr, Q_pour, pair_p = done
    report["metrics"] = {"transport": excess_metrics(ch, Q_tr, T_ee_canon, pair_tr.path, a.speed),
                         "pour": excess_metrics(ch, Q_pour, T_ee_canon, pair_p.path, a.speed)}
    d_end = pair_p.subgoal.displacement(ch.fk(Q_pour[-1]) @ T_ee_canon)
    report["pour_tilt_achieved_deg"] = round(float(np.degrees(d_end[3])), 1)
    report["joint_travel_deg"] = {"transport": joint_travel(Q_tr), "pour": joint_travel(Q_pour)}
    for name, m in report["metrics"].items():
        print(f"[metrics] {name}: {_fmt_metrics(m)}")
        print(f"[travel] {name} (deg): {_fmt_travel(report['joint_travel_deg'][name])}")
    print(f"[pour] achieved tilt {report['pour_tilt_achieved_deg']} deg (target {a.tilt_deg:.0f})")

    (out / "stages.json").write_text(json.dumps({
        "joint_names": list(um.DH_ORDER), "q_lift": q_lift.tolist(), "T_ee_canon": T_ee_canon.tolist(),
        "stages": [{"name": "transport", "positions": Q_tr.tolist()},
                   {"name": "pour", "positions": Q_pour.tolist()}]}, indent=1) + "\n")
    (out / "plan.json").write_text(json.dumps(
        exec_plan_doc(reach["plan"], segs, Q_tr, Q_pour, ge.LINE_SPEED_M_S)) + "\n")
    report["status"] = "planned"
    report["outputs"] = {"stages": str(out / "stages.json"), "plan": str(out / "plan.json")}
    save_report()
    print(f"[plan] OK -> {out}/{{report,stages,plan}}.json\n"
          f"       rviz: python3 /data/runs/exec_plan.py {out / 'plan.json'} --dry")
    return 0


# ---------------------------------------------------------------- VLM method
#
# --method vlm plans the same carry and pour on call #3's regions instead of
# pour_stages' hand-written ones, the way the sim's plan_task does:
#
#   <run>/stage_plan/plan.json + <run>/call2/selections.json   (calls 1, 2)
#   <emissions>  emit_scene.py's artifact (call 3), read back per role
#        |
#   per mover stage, in bound order after the taught grasp (pour tea:
#   transport_active, then pour): compile_stage at the ACTUAL entry --
#   FK(q at the previous stage's end) . T_ee_canon -- then the sidecar's
#   plan_constrained (goal sampling, IK, collision, containment, CBiRRT).
#   A failed pour re-plans the carry with the next seed, as the hand method.
#
# --critic preview runs critic_scene.run_critic before any sidecar call: the
# accepted specification (edited emissions, re-picked points) replaces call
# #3's and call #2's for planning; a reject at the cap is the typed failure
# preview_rejected (exit 2). The summary goes into report.json["critic"].
#
# Frames: everything is in the GROUNDING canonical frame emit_scene compiled
# in (summary pose x R_body_from_canon, emit_scene.measured_poses), so
# T_ee_canon = inv(FK(q_grasp)) . T_base_canon. hand_symbols' fixtures are
# not used. Every failure is recorded as a typed record (FAILURE_TYPES, the
# names call #5 will consume) in report.json; nothing new is gated.

VLM_ROLES = ("transport_active", "pour")      # exec_plan's transport and pour segments
# compile_stage's and check_pair_consistency's texts for a path and subgoal
# that do not intersect (call #5's empty_goal_intersection). A row that two
# terms of ONE TSR narrow to nothing ("row k intersection empty") is an
# emission defect and stays with call #3's compile retry.
_EMPTY_MARKERS = ("does not meet the subgoal", "excludes every subgoal", "do not overlap",
                  "excludes the subgoal")


def _tsr_record(t) -> dict:
    return {"name": t.name, "T0_w": np.round(np.asarray(t.T0_w, float), 6).tolist(),
            "Tw_e": np.round(np.asarray(t.Tw_e, float), 6).tolist(),
            "Bw": np.asarray(t.Bw, float).tolist()}


def compile_failure(role: str, name: str, seed: int, e) -> dict:
    """A CompileError as a typed record. A box intersection the compiler
    proved empty is call #5's empty_goal_intersection; any other rejection
    is call #3's (its compile-retry path), so it carries no failure type."""
    errs = [{"slot": x.slot, "reason": x.reason} for x in e.all()]
    empty = any(m in x["reason"] for x in errs for m in _EMPTY_MARKERS)
    return {"type": "empty_goal_intersection" if empty else None,
            "routes_to": "call5" if empty else "call3_compile_retry",
            "stage": role, "stage_name": name, "seed": seed, "phase": "compile",
            "slots": [x["slot"] for x in errs], "errors": errs, "funnel": None}


def plan_failure(role: str, name: str, seed: int, out: dict, pair) -> dict:
    """A sidecar failure as a typed record: the goal funnel decides between
    an empty intersection and no feasible IK; after goals exist, the CBiRRT
    reason decides between projection divergence and timeout."""
    f, why = out.get("funnel") or {}, str(out.get("reason", ""))
    if why.startswith("q_start in collision") or not f:
        kind = None                      # the stage never reached its goal funnel
    elif f.get("stopped") == "thin_intersection":
        kind = "empty_goal_intersection"
    elif not f.get("n_contained"):
        kind = "no_feasible_ik"          # zero IK, or IK with nothing free / contained
    elif "projection failed" in why:   # manip_cbirrt: "start/goal projection failed"
        kind = "projection_divergence"
    elif "no path in" in why or "time budget" in why:
        kind = "planner_timeout"
    else:
        kind = None
    return {"type": kind, "routes_to": "call5" if kind else None, "stage": role,
            "stage_name": name, "seed": seed, "phase": "plan", "reason": why,
            "atoms": {"path": pair.path.name, "subgoal": pair.subgoal.name},
            "funnel": out.get("funnel"),
            "per_constraint_rejections": f.get("per_constraint_rejections")}


def compile_at_entry(ec, em, symbols, poses, T_entry, mover, w_pt, e_pt, synth):
    """compile_stage at the actual entry, then the emit gate's joint
    path-and-subgoal intersection check (check_pair_consistency), so an
    empty intersection is reported here, not as a starved goal funnel."""
    cs = ec.compile_stage(em, symbols, {**poses, mover: T_entry},
                          w_point=w_pt, e_point=e_pt, entry_synthesized=synth)
    ec.check_pair_consistency(cs)
    return cs


@dataclass
class VlmInputs:
    """What --method vlm reads, loaded once (load_vlm_inputs). `movers`:
    [(role, stage, emission, w_point, e_point, synthesized)] for the bound
    stages with a passive object, in order; the rest is what the critic
    also needs (every bound stage, every role's emission, call #2's
    selections and where the inputs live)."""
    b: object
    objs: dict
    symbols: dict
    summary: dict
    poses: dict
    movers: list
    em_doc: dict
    es: object                         # the emit_scene module
    plan_path: Path
    sel_path: Path
    em_path: Path
    asset_dirs: dict
    sels: dict
    sel_objects: dict
    role_index: dict
    ems: dict
    stages: list                       # [(StageSpec, role)], ec.bound_stages
    synthesized: frozenset


def write_report(out: Path, report: dict) -> None:
    """The one writer of <out>/report.json (every exit path of --method vlm)."""
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")


def load_vlm_inputs(a, run_dir: Path, out: Path) -> VlmInputs:
    """Load and cross-check the plan, call #2's selections, call #3's
    emissions and the run's measured poses (see VlmInputs)."""
    es = _load("emit_scene")
    plan_path = Path(a.plan) if a.plan else run_dir / "stage_plan" / "plan.json"
    sel_path = Path(a.selections) if a.selections else run_dir / "call2" / "selections.json"
    em_path = Path(a.emissions)
    for p in (plan_path, sel_path, em_path, run_dir / "summary.json"):
        if not p.is_file():
            raise PlanRefusal(f"{p} not found")
    try:
        b, doc, objs = es.ss.load_inputs(run_dir, plan_path)
    except es.ss.SelectRefusal as e:
        raise PlanRefusal(str(e)) from None
    em_doc = json.loads(em_path.read_text())
    hw_rec = em_doc.get("hardware") or {}
    plan_sha = (doc.get("hardware") or {}).get("plan_sha")
    if not plan_sha or hw_rec.get("plan_sha") != plan_sha:
        raise PlanRefusal(f"{em_path} was emitted for plan {hw_rec.get('plan_sha')!r}, the plan is "
                          f"{plan_sha!r} -- re-run emit_scene.py")
    if hw_rec.get("selections_sha256") != _sha(sel_path):
        raise PlanRefusal(f"{em_path} was emitted on other selections than {sel_path} -- "
                          "re-run emit_scene.py")
    roles = em_doc.get("roles") or []
    ems = {r: es.vlm_compat.emission_from_json(e) for r, e in zip(roles, em_doc.get("emissions", []))}

    asset_dirs = es.alias_assets(objs, out / "assets")
    symbols = {n: es.sc.load_symbols(d) for n, d in asset_dirs.items()}
    summary = json.loads((run_dir / "summary.json").read_text())
    try:
        poses = es.measured_poses(summary, objs)
    except es.EmitRefusal as e:
        raise PlanRefusal(str(e)) from e
    sels = es.sc.load_selections(sel_path)
    sel_objects = es.sc.selection_objects(sel_path, sels)
    by_idx = {s.index: s for s in b.plan.stages}
    role_index = {r: st.index for r, (_, st) in b.roles.items()}
    movers = []
    for stage, role in es.ec.bound_stages(b, by_idx):
        if stage.passive is None:
            continue                       # the grasp: taught, not planned from its emission
        if role not in ems:
            raise PlanRefusal(f"{em_path} has no emission for {role!r} (has {sorted(ems)}): "
                              "call #3 stopped before it -- see the artifact's 'compiled'")
        try:
            w_pt, e_pt = es.sc.selected_points(role, stage, sels, sel_objects, role_index,
                                               asset_dirs, symbols)
        except (SystemExit, es.sc.ResolutionError) as e:
            raise PlanRefusal(f"{sel_path}: anchor points do not resolve ({e})") from None
        movers.append((role, stage, ems[role], w_pt, e_pt, role in b.entry_synthesized))
    if tuple(m[0] for m in movers) != VLM_ROLES:
        raise PlanRefusal(f"--method vlm plans pour tea only (mover roles {list(VLM_ROLES)}); "
                          f"this plan binds {[m[0] for m in movers]}")
    if any(m[5] for m in movers):
        raise PlanRefusal(f"synthesized entries {[m[0] for m in movers if m[5]]} are not "
                          "supported on hardware yet (plan_task's synthesized start)")
    return VlmInputs(b=b, objs=objs, symbols=symbols, summary=summary, poses=poses,
                     movers=movers, em_doc=em_doc, es=es, plan_path=plan_path, sel_path=sel_path,
                     em_path=em_path, asset_dirs=asset_dirs, sels=sels, sel_objects=sel_objects,
                     role_index=role_index, ems=ems, stages=list(es.ec.bound_stages(b, by_idx)),
                     synthesized=frozenset(b.entry_synthesized))


def run_vlm(a) -> int:
    from manip_bridge import emit_compat as es_ec      # loaded only for this method
    from manip_bridge.zmq_client import SidecarClient
    run_dir = Path(a.run)
    out = Path(a.out) if a.out else run_dir / "m4_vlm"
    out.mkdir(parents=True, exist_ok=True)
    vin = load_vlm_inputs(a, run_dir, out)
    objs, symbols, summary, poses = vin.objs, vin.symbols, vin.summary, vin.poses
    movers, em_doc = vin.movers, vin.em_doc
    mover = movers[0][1].active
    passive = movers[0][1].passive
    if any(m[1].active != mover for m in movers):
        raise PlanRefusal(f"every mover stage must carry {mover!r}: "
                          f"{[(m[0], m[1].active) for m in movers]}")
    try:
        mesh_mv = Path(so.mesh_path(summary, objs[mover]["key"]))
        T_mv_body = so.T_base_obj(summary, objs[mover]["key"])
        T_ps_body = so.T_base_obj(summary, objs[passive]["key"])
        mesh_ps = Path(so.mesh_path(summary, objs[passive]["key"]))
    except ValueError as e:
        raise PlanRefusal(f"{run_dir / 'summary.json'}: {e}") from e
    up_dev = {k: round(float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))), 2)
              for k, T in poses.items()}

    ch = um.chain()
    ge = _load("grasp_exec")
    reach = json.loads(Path(a.reach or run_dir / "m3" / "reach.json").read_text())
    segs = ge.build(reach["plan"], ch, deeper=a.deeper)
    by = {s.name: s for s in segs}
    q_grasp, q_lift = by["approach"].Q[-1], by["lift"].Q[-1]
    T_tool0_grasp = ch.fk(q_grasp)
    T_ee_canon = np.linalg.inv(T_tool0_grasp) @ poses[mover]
    T_ee_body = so.T_ee_body(T_tool0_grasp, T_mv_body)

    gate_failed = [c["stage"] for c in em_doc.get("compiled", []) if not c.get("grounded")]
    report = {"run": str(run_dir), "method": "vlm (call #3 emissions)", "scene": a.scene,
              "inputs": {"reach": reach["plan"].get("grasp"), "emissions": str(a.emissions),
                         "plan_sha": (em_doc.get("hardware") or {}).get("plan_sha"),
                         "emit_gate_failed": gate_failed, "deeper_m": a.deeper,
                         "sim": es_ec.provenance()},
              "frames": "grounding canonical (summary pose x R_body_from_canon)",
              "poses": {"T_base_canon": {k: T.tolist() for k, T in poses.items()},
                        "canonical_up_vs_base_z_deg": up_dev},
              "grasp": {"q_grasp": q_grasp.tolist(), "q_lift": q_lift.tolist(),
                        "T_ee_body": T_ee_body.tolist(), "T_ee_canon": T_ee_canon.tolist()},
              "attempts": [], "failures": [], "status": "failed"}

    critic = None
    if a.critic == "preview":
        critic = run_critic_preview(a, run_dir, out, vin, ch.fk(q_lift) @ T_ee_canon, mover,
                                    T_tool0_grasp)
        if critic["status"] == "dry_run":
            return 0
        report["method"] = "vlm (call #3 emissions) + critic"
        report["critic"] = critic_summary(critic)
        if critic["status"] == "verdict":         # --critic-verdict-only: no edit, no planning
            report["status"] = "verdict_only"
            write_report(out, report)
            print(f"[plan] critic verdict: {critic['verdict']['verdict']}; not planned "
                  f"(--critic-verdict-only) -> {out / 'report.json'}")
            return 0
        if critic["status"] != "accepted":
            report["failures"].append(critic["failure"] or {"type": None, "phase": "critic"})
            write_report(out, report)
            print(f"[plan] critic: {critic['status']}; not planned -> {out / 'report.json'}")
            return 2
        movers, failures = accepted_movers(critic, movers)
        if failures:
            report["failures"] += failures
            write_report(out, report)
            print(f"[plan] the accepted specification cannot be planned as accepted: "
                  f"{[(f['stage'], f['phase']) for f in failures]} -> {out / 'report.json'}")
            return 2

    cli = SidecarClient(CUROBO_ADDR, timeout_ms=CALL_TIMEOUT_MS)
    S_body = so.object_spheres(mesh_mv, run_dir / "objects" / objs[mover]["key"] / "spheres_body.json",
                               cli.call)
    spheres = so.spheres_tool0(S_body, T_ee_body)
    scene = (registered_scene(summary, T_ps_body, mesh_ps, 0.5 * (T_mv_body[:2, 3] + T_ps_body[:2, 3]))
             if a.scene == "registered" else None)
    report["grasp"]["n_spheres"] = int(len(spheres))
    if gate_failed:
        print(f"[plan] note: call #3's gate failed {gate_failed}; compiling at the actual entry anyway")

    def save_report():
        write_report(out, report)

    def compile_rec(role, stage, em, w_pt, e_pt, synth, T_entry, seed, rec):
        """-> the compiled stage, or None with the typed failure recorded."""
        try:
            cs = compile_at_entry(es_ec, em, symbols, poses, T_entry, mover, w_pt, e_pt, synth)
        except es_ec.CompileError as e:
            fail = compile_failure(role, stage.name, seed, e)
            rec.update(outcome="compile_rejected", failure=fail)
            report["failures"].append(fail)
            print(f"[{role}] seed {seed}: compile rejected ({fail['type']}): {e.text()}")
            return None
        rec.update(compiled={"path": _tsr_record(cs.path), "subgoal": _tsr_record(cs.subgoal)},
                   notes=list(cs.notes))
        return cs

    # the first stage starts at q_lift on every seed: compile it once
    first = movers[0]
    T_first = ch.fk(q_lift) @ T_ee_canon
    first_rec = {"role": first[0], "name": first[1].name,
                 "entry_T_base_canon": np.round(T_first, 6).tolist()}
    cs_first = compile_rec(*first, T_first, 0, first_rec)
    report["first_stage_compile"] = first_rec
    done = None
    try:
        for k in range(a.tries if cs_first is not None else 0):
            att = {"seed": k, "stages": []}
            report["attempts"].append(att)
            q_cur, paths = q_lift, {}
            for i, (role, stage, em, w_pt, e_pt, synth) in enumerate(movers):
                T_entry = ch.fk(q_cur) @ T_ee_canon
                rec = {"role": role, "name": stage.name,
                       "entry_T_base_canon": np.round(T_entry, 6).tolist()}
                att["stages"].append(rec)
                if i == 0:
                    cs = cs_first
                    rec.update(compiled=first_rec["compiled"], notes=first_rec["notes"])
                else:
                    cs = compile_rec(role, stage, em, w_pt, e_pt, synth, T_entry, k, rec)
                    if cs is None:
                        break
                res = plan_stage(cli, role, q_cur, T_ee_canon, cs, spheres, scene, a, k,
                                 TRANSPORT_EPS if role == VLM_ROLES[0] else POUR_EPS)
                rec["plan"] = {kk: v for kk, v in res.items() if kk != "Q"}
                if not res["success"]:
                    fail = plan_failure(role, stage.name, k, res, cs)
                    rec.update(outcome="plan_failed", failure=fail)
                    report["failures"].append(fail)
                    break
                rec["outcome"] = "planned"
                paths[role] = (res["Q"], cs)
                q_cur = res["Q"][-1]
            if len(paths) == len(movers):
                done = paths
                break
    except PlanRefusal as e:                 # keep the attempts that did run
        report["status"], report["error"] = "refused", str(e)
        save_report()
        raise
    if done is None:
        save_report()
        print(f"[plan] FAILED after {len(report['attempts'])} attempt(s); typed failures "
              f"{[f['type'] for f in report['failures']]} -> {out / 'report.json'}")
        return 2

    (Q_tr, cs_tr), (Q_pour, cs_p) = done[VLM_ROLES[0]], done[VLM_ROLES[1]]
    report["metrics"] = {"transport": excess_metrics(ch, Q_tr, T_ee_canon, cs_tr.path, a.speed),
                         "pour": excess_metrics(ch, Q_pour, T_ee_canon, cs_p.path, a.speed)}
    report["joint_travel_deg"] = {"transport": joint_travel(Q_tr), "pour": joint_travel(Q_pour)}
    for name, m in report["metrics"].items():
        print(f"[metrics] {name}: {_fmt_metrics(m)}")
    (out / "stages.json").write_text(json.dumps({
        "joint_names": list(um.DH_ORDER), "q_lift": q_lift.tolist(), "T_ee_canon": T_ee_canon.tolist(),
        "stages": [{"name": "transport", "role": VLM_ROLES[0], "positions": Q_tr.tolist()},
                   {"name": "pour", "role": VLM_ROLES[1], "positions": Q_pour.tolist()}]},
        indent=1) + "\n")
    (out / "plan.json").write_text(json.dumps(
        exec_plan_doc(reach["plan"], segs, Q_tr, Q_pour, ge.LINE_SPEED_M_S,
                      source="call #3 regions")) + "\n")
    report["status"] = "planned"
    report["outputs"] = {"stages": str(out / "stages.json"), "plan": str(out / "plan.json")}
    save_report()
    print(f"[plan] OK -> {out}/{{report,stages,plan}}.json\n"
          f"       rviz: python3 /data/runs/exec_plan.py {out / 'plan.json'} --dry")
    return 0


def run_critic_preview(a, run_dir, out, vin: VlmInputs, T_entry, mover, T_tool0_grasp) -> dict:
    """critic_scene.run_critic on this run (see critic_scene.py). On
    'accepted' the result also carries 'accepted_spec' (critic_scene.
    accepted_spec over the mover roles). A sim without the critic, or an
    input the critic refuses, is a PlanRefusal."""
    cs_mod = _load("critic_scene")
    try:
        cs_mod.require_critic()
        gu = Path(a.gripper_urdf) if a.gripper_urdf else None
        gm = Path(a.gripper_meshes) if a.gripper_meshes else None
        mesh = Path(a.gripper_mesh) if a.gripper_mesh else cs_mod.find_gripper_mesh(run_dir)
        notes = []
        if mesh is not None:
            if not mesh.is_file():
                raise PlanRefusal(f"--gripper-mesh {mesh}: not found")
            gu = None                      # the exported mesh wins: the same 2F-85
            notes.append(f"gripper: mesh {mesh}")
        elif gu is not None and not gu.is_file():
            notes.append(f"gripper not drawn: {gu} not found (pass --gripper-urdf/"
                         f"--gripper-meshes, or run critic_scene.py export-gripper --run "
                         f"{run_dir} on the host)")
            print(f"[critic] note: {notes[-1]}")
            gu = None
        ctx = cs_mod.CriticContext(
            run_dir=run_dir, out=out, plan_path=vin.plan_path, stages=vin.stages, objs=vin.objs,
            asset_dirs=vin.asset_dirs, symbols=vin.symbols, summary=vin.summary,
            poses=vin.poses, selections=vin.sels, sel_objects=vin.sel_objects,
            role_index=vin.role_index, synthesized=vin.synthesized, emissions=vin.ems,
            raw_emissions=cs_mod.raw_by_stage_name(vin.em_path.with_name("emissions.log.json")),
            offered=cs_mod.offered_from_log(vin.sel_path.with_name("selections.log.json"),
                                            vin.sel_objects),
            T_entry=T_entry, grasp_obj=mover, T_tcp_grasp=T_tool0_grasp, es=vin.es,
            arm=vin.em_doc.get("arm") or "framed-chained",
            gripper_mesh=mesh, gripper_urdf=gu, gripper_meshes=gm, notes=notes)
        mode = ("dry_run" if a.critic_dry_run else "replay" if a.critic_responses else "live")
        critic = cs_mod.run_critic(ctx, mode=mode, responses=(Path(a.critic_responses)
                                                              if a.critic_responses else None),
                                   verdict_only=a.critic_verdict_only)
    except cs_mod.CriticRefusal as e:
        raise PlanRefusal(str(e)) from None
    if critic["status"] == "accepted":
        critic["accepted_spec"] = cs_mod.accepted_spec(ctx, critic["state"],
                                                       [m[0] for m in vin.movers])
    return critic


def critic_summary(critic: dict) -> dict:
    return {k: v for k, v in critic.items() if k not in ("chain", "state", "accepted_spec")}


def accepted_movers(critic: dict, movers: list) -> tuple[list, list]:
    """(the planner's movers, failures): every mover planned exactly as the
    critic accepted it -- its accepted emission and the anchor points
    resolved from the accepted selections. Never falls back to the inputs
    the critic saw first: a mover the accepted specification cannot supply
    is a typed failure record (critic_scene.accepted_spec), and the caller
    plans nothing."""
    spec, failures = critic["accepted_spec"]
    out = [(role, stage, *spec[role], synth)
           for role, stage, _em, _w, _e, synth in movers if role in spec]
    return out, list(failures)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--reach", default=None, help="default <run>/m3/reach.json")
    ap.add_argument("--library", default=str(HERE / "library"))
    ap.add_argument("--teapot-key", default=None)
    ap.add_argument("--mug-key", default=None)
    ap.add_argument("--scene", choices=("registered", "live"), default="registered")
    ap.add_argument("--tilt-deg", type=float, default=TILT_DEG)
    ap.add_argument("--tries", type=int, default=5, help="transport seeds to try for a pourable end")
    ap.add_argument("--timeout", type=float, default=20.0, help="CBiRRT budget per stage call (s)")
    ap.add_argument("--n-goal-samples", type=int, default=60)
    ap.add_argument("--deeper", type=float, default=0.0, help="as grasp_exec --deeper")
    ap.add_argument("--speed", type=float, default=0.1, help="timing used for the dense metrics")
    ap.add_argument("--out", default=None, help="default <run>/m4 (hand) or <run>/m4_vlm (vlm)")
    ap.add_argument("--method", choices=("hand", "vlm"), default="hand",
                    help="hand: pour_stages' hand-written regions; vlm: call #3's emissions")
    ap.add_argument("--plan", default=None, help="vlm: default <run>/stage_plan/plan.json")
    ap.add_argument("--selections", default=None, help="vlm: default <run>/call2/selections.json")
    ap.add_argument("--emissions", default=None,
                    help="vlm (required): emit_scene's artifact, e.g. <run>/call3/emissions.json")
    ap.add_argument("--critic", choices=("off", "preview"), default="off",
                    help="vlm: off (default) plans call #3's regions as they are; preview runs "
                         "VLM call #4 (critic_scene.py) first and plans what it accepts")
    ap.add_argument("--critic-responses", default=None,
                    help="--critic preview: replay critic_* / emit_* response files from this "
                         "dir instead of calling the API")
    ap.add_argument("--critic-dry-run", action="store_true",
                    help="--critic preview: write round 1's prompt and images, stop (no call)")
    ap.add_argument("--critic-verdict-only", action="store_true",
                    help="--critic preview: one critic call on the specification as given, "
                         "record its verdict, apply no edit and do not plan (critic_cases.py)")
    ap.add_argument("--gripper-urdf", default=str(REPO / "curobo_runtime/robot/ur5e_robotiq.urdf"),
                    help="--critic preview: the 2F-85 below tool0, drawn in the preview")
    ap.add_argument("--gripper-meshes",
                    default=str(REPO / "test/curobo_incontainer/robotiq/meshes/collision"))
    ap.add_argument("--gripper-mesh", default=None,
                    help="--critic preview: the 2F-85 as one OBJ in the TCP frame (critic_scene.py "
                         "export-gripper); default <run>/critic_assets/gripper_2f85.obj if it "
                         "exists, else the URDF")
    a = ap.parse_args(argv)
    if bool(a.teapot_key) != bool(a.mug_key):
        ap.error("--teapot-key and --mug-key go together")
    if a.method == "vlm" and not a.emissions:
        ap.error("--method vlm needs --emissions")
    if a.method == "hand" and (a.plan or a.selections or a.emissions):
        ap.error("--plan/--selections/--emissions are for --method vlm")
    if a.method == "hand" and a.critic != "off":
        ap.error("--critic is for --method vlm")
    if a.critic == "off" and (a.critic_responses or a.critic_dry_run or a.critic_verdict_only):
        ap.error("--critic-responses/--critic-dry-run/--critic-verdict-only need --critic preview")
    if a.critic_responses and a.critic_dry_run:
        ap.error("--critic-responses and --critic-dry-run are exclusive")
    if a.method == "vlm" and (a.teapot_key or a.library != str(HERE / "library")
                              or a.tilt_deg != TILT_DEG):
        ap.error("--teapot-key/--mug-key/--library/--tilt-deg are for --method hand: the vlm "
                 "method takes objects from the plan's bindings and the pour from call #3")
    try:
        return run_vlm(a) if a.method == "vlm" else run(a)
    except PlanRefusal as e:
        print(f"[plan] REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
