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
            v  translation mm, rotation deg), at the waypoints and densely
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
"""
import argparse
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
    i = int(np.argmax(td + rd))
    out["dense"] = {"n_samples": int(len(Qs)), "duration_s": round(float(t[-1]), 2),
                    "trans_mm": _stats(td), "rot_deg": _stats(rd),
                    "worst_at": {"t_s": round(float(t[i]), 2), "fraction": round(i / (len(Qs) - 1), 3),
                                 "body_xyz": np.round((ch.fk(Qs[i]) @ T_ee_canon)[:3, 3], 4).tolist()}}
    return out


def _fmt_metrics(m) -> str:
    w, d = m["waypoints"], m["dense"]
    s = (f"waypoints ({m['n_waypoints']}): trans max {w['trans_mm']['max']} / med "
         f"{w['trans_mm']['median']} mm, rot max {w['rot_deg']['max']} / med {w['rot_deg']['median']} deg")
    if "refused" in d:
        return s + f"; dense: TOPP-RA refused ({d['refused']})"
    return (s + f"; dense ({d['n_samples']} @ {um.SAMPLE_DT * 1000:.0f} ms): trans max "
            f"{d['trans_mm']['max']} / med {d['trans_mm']['median']} / p95 {d['trans_mm']['p95']} mm, "
            f"rot max {d['rot_deg']['max']} / med {d['rot_deg']['median']} / p95 "
            f"{d['rot_deg']['p95']} deg (worst at {d['worst_at']['fraction']:.0%})")


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


def exec_plan_doc(reach_plan, segs, Q_tr, Q_pour, line_speed) -> dict:
    """The blind lines keep grasp_exec's line-speed floor (length / line_speed)."""
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
             _traj_step("transport", Q_tr, note="CBiRRT, hand arm"),
             _traj_step("pour", Q_pour, note="CBiRRT, hand arm"),
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
    tp_mesh, mug_mesh = Path(so.mesh_path(summary, tk)), Path(so.mesh_path(summary, mk))
    tp, mug = load_fixture(Path(a.library), tp_mesh), load_fixture(Path(a.library), mug_mesh)

    # ---- poses (base_link), canonical frames
    T_tp, T_mug = so.T_base_obj(summary, tk), so.T_base_obj(summary, mk)
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
    done = None
    for k in range(a.tries):
        tr = plan_stage(cli, "transport", q_lift, T_ee_canon, pair_tr, spheres, scene, a, k, TRANSPORT_EPS)
        att = {"seed": k, "transport": {kk: v for kk, v in tr.items() if kk != "Q"}}
        report["attempts"].append(att)
        if not tr["success"]:
            continue
        Q_tr = tr["Q"]
        T_entry = ch.fk(Q_tr[-1]) @ T_ee_canon
        pair_p = frames.pour(T_entry, T_mugC, tilt_target)
        ex = pair_p.path.excess(T_entry)
        att["pour_entry_excess"] = np.round(ex, 5).tolist()
        po = plan_stage(cli, "pour", Q_tr[-1], T_ee_canon, pair_p, spheres, scene, a, k, POUR_EPS)
        att["pour"] = {kk: v for kk, v in po.items() if kk != "Q"}
        if po["success"]:
            done = (Q_tr, pair_tr, po["Q"], pair_p)
            break
    if done is None:
        (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(f"[plan] FAILED after {a.tries} transport seeds -> {out / 'report.json'}")
        return 2

    Q_tr, pair_tr, Q_pour, pair_p = done
    # the sim executor's densify: the planner's waypoints kept, straight joint
    # segments filled in; what exec_plan times and what the metrics sample
    report["metrics"] = {"transport": excess_metrics(ch, Q_tr, T_ee_canon, pair_tr.path, a.speed),
                         "pour": excess_metrics(ch, Q_pour, T_ee_canon, pair_p.path, a.speed)}
    Q_tr, Q_pour = um.densify(Q_tr), um.densify(Q_pour)
    d_end = pair_p.subgoal.displacement(ch.fk(Q_pour[-1]) @ T_ee_canon)
    report["pour_tilt_achieved_deg"] = round(float(np.degrees(d_end[3])), 1)
    for name, m in report["metrics"].items():
        print(f"[metrics] {name}: {_fmt_metrics(m)}")
    print(f"[pour] achieved tilt {report['pour_tilt_achieved_deg']} deg (target {a.tilt_deg:.0f})")

    (out / "stages.json").write_text(json.dumps({
        "joint_names": list(um.DH_ORDER), "q_lift": q_lift.tolist(), "T_ee_canon": T_ee_canon.tolist(),
        "stages": [{"name": "transport", "positions": Q_tr.tolist()},
                   {"name": "pour", "positions": Q_pour.tolist()}]}, indent=1) + "\n")
    (out / "plan.json").write_text(json.dumps(
        exec_plan_doc(reach["plan"], segs, Q_tr, Q_pour, ge.LINE_SPEED_M_S)) + "\n")
    report["status"] = "planned"
    report["outputs"] = {"stages": str(out / "stages.json"), "plan": str(out / "plan.json")}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"[plan] OK -> {out}/{{report,stages,plan}}.json\n"
          f"       rviz: python3 /data/runs/exec_plan.py {out / 'plan.json'} --dry")
    return 0


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
    ap.add_argument("--out", default=None, help="default <run>/m4")
    a = ap.parse_args(argv)
    if bool(a.teapot_key) != bool(a.mug_key):
        ap.error("--teapot-key and --mug-key go together")
    try:
        return run(a)
    except PlanRefusal as e:
        print(f"[plan] REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
