#!/usr/bin/env python3
"""Is the mug's tracked mesh a real obstacle to cuRobo? (offline, nothing moves)

plan_stages_hw.py's default "registered" scene hands the cuRobo sidecar the
table as a slab and the mug as its TRACKED mesh (the reconstruction pose
tracking registers -- on hardware it stands in for the sim's privileged
mesh). Nothing so far showed that cuRobo actually blocks motion against
that mesh: the sidecar's own tests use a box. A mesh with holes or flipped
faces can read as partly empty space, and the pour would then plan through
the mug and still report success.

The check asks the sidecar for teapot poses that sit INSIDE the mug: the
teapot upright as registered, its mesh centre moved onto the mug's, its
lowest point 1.5 cm above the table, a little xyz/yaw freedom so the goal
sampler has something to sample. Two requests, same everything else:

    with_mug     table + mug mesh  -> expect n_collision_free == 0
    table_only   table only        -> expect n_collision_free  > 0 (control:
                                      a zero above is the mug, not IK, the
                                      table or the grasp)

Both need n_ik > 0, or the check says nothing. Plus the mesh's own stats
(watertight, consistent winding, volume). Log only: the verdict is printed
and written to <run>/m4/mug_obstacle_check.json; the exit code is 0 unless
the check could not run.

In the ros2 container (cuRobo sidecar up):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/check_mug_obstacle.py --run /data/runs/<stamp>
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIFT_M = 0.015          # teapot's lowest point above the table
XYZ_FREE_M = 0.005
YAW_FREE_RAD = 0.3


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _world(T, V):
    return V @ T[:3, :3].T + T[:3, 3]


def mesh_stats(m) -> dict:
    return {"n_vertices": int(len(m.vertices)), "n_faces": int(len(m.faces)),
            "watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent),
            "volume_m3": float(m.volume) if m.is_watertight else None,
            "extents_m": np.round(m.extents, 4).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--reach", default=None, help="default <run>/m3/reach.json")
    ap.add_argument("--library", default=str(HERE / "library"))
    ap.add_argument("--n-goal-samples", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=3.0, help="CBiRRT budget; only the goal funnel matters")
    a = ap.parse_args(argv)

    import trimesh
    hw = _load("plan_stages_hw")
    so, um, psc = hw.so, hw.um, hw.psc
    from manip_bridge.zmq_client import SidecarClient

    run_dir = Path(a.run)
    summary = json.loads((run_dir / "summary.json").read_text())
    tk, mk = hw.object_keys(summary)
    tp_mesh, mug_mesh = Path(so.mesh_path(summary, tk)), Path(so.mesh_path(summary, mk))
    T_tp, T_mug = so.T_base_obj(summary, tk), so.T_base_obj(summary, mk)
    tp = hw.load_fixture(Path(a.library), tp_mesh)
    T_tpC = T_tp @ hw._block(tp["_R"])

    # grasp, exactly as plan_stages_hw builds it
    ch, ge = um.chain(), hw._load("grasp_exec")
    reach = json.loads(Path(a.reach or run_dir / "m3" / "reach.json").read_text())
    by = {s.name: s for s in ge.build(reach["plan"], ch)}
    q_grasp, q_lift = by["approach"].Q[-1], by["lift"].Q[-1]
    T_tool0_grasp = ch.fk(q_grasp)
    T_ee_canon = np.linalg.inv(T_tool0_grasp) @ T_tpC
    cli = SidecarClient(hw.CUROBO_ADDR, timeout_ms=hw.CALL_TIMEOUT_MS)
    S_body = so.object_spheres(tp_mesh, run_dir / "objects" / tk / "spheres_body.json", cli.call)
    spheres = so.spheres_tool0(S_body, so.T_ee_body(T_tool0_grasp, T_tp))

    # target: teapot (registered orientation) centred on the mug, just off the table
    m_tp = trimesh.load(str(tp_mesh), force="mesh", process=False)
    m_mug = trimesh.load(str(mug_mesh), force="mesh", process=False)
    Vt, Vm = _world(T_tp, np.asarray(m_tp.vertices)), _world(T_mug, np.asarray(m_mug.vertices))
    ct, cm = 0.5 * (Vt.min(0) + Vt.max(0)), 0.5 * (Vm.min(0) + Vm.max(0))
    plane = so.table_plane_base(summary)
    z_table = float(so.table_z(plane, cm[0], cm[1]))
    d = np.array([cm[0] - ct[0], cm[1] - ct[1], z_table + LIFT_M - Vt[:, 2].min()])
    T_target = T_tpC.copy()
    T_target[:3, 3] += d
    print(f"[target] teapot moved {np.round(d, 3).tolist()} m: centre on the mug at "
          f"{np.round(cm[:2], 3).tolist()}, lowest point {LIFT_M * 100:.1f} cm above the table "
          f"(mug top {100 * (Vm[:, 2].max() - z_table):.1f} cm above it)")

    bw_goal = np.array([[-XYZ_FREE_M, XYZ_FREE_M]] * 3 + [[0, 0], [0, 0], [-YAW_FREE_RAD, YAW_FREE_RAD]])
    bw_path = np.array([[-1.0, 1.0]] * 3 + [[-np.pi, np.pi]] * 3)
    pair = type("Pair", (), {})()
    pair.subgoal = psc.TSR(T0_w=T_target, Bw=bw_goal, name="teapot_in_mug")
    pair.path = psc.TSR(T0_w=T_target, Bw=bw_path, name="anywhere")

    full = hw.registered_scene(summary, T_mug, mug_mesh, 0.5 * (T_tp[:2, 3] + T_mug[:2, 3]))
    scenes = {"with_mug": full, "table_only": {"cuboid": full["cuboid"]}}
    args = argparse.Namespace(n_goal_samples=a.n_goal_samples, timeout=a.timeout)
    res = {}
    for name, scene in scenes.items():
        r = hw.plan_stage(cli, name, q_lift, T_ee_canon, pair, spheres, scene, args, 0, hw.TRANSPORT_EPS)
        res[name] = {k: r.get(k) for k in ("funnel", "reason", "wall_s")}

    f_mug, f_ctl = res["with_mug"]["funnel"] or {}, res["table_only"]["funnel"] or {}
    ik_ok = (f_mug.get("n_ik") or 0) > 0 and (f_ctl.get("n_ik") or 0) > 0
    blocked = (f_mug.get("n_collision_free") or 0) == 0
    control_free = (f_ctl.get("n_collision_free") or 0) > 0
    if not ik_ok:
        verdict = "inconclusive: no IK solutions at the target (move it / widen the freedom)"
    elif not control_free:
        verdict = "inconclusive: the table-only control is blocked too (table or grasp, not the mug)"
    elif blocked:
        verdict = "PASS: the mug mesh blocks every teapot-in-mug goal the table-only scene allows"
    else:
        verdict = (f"FAIL: {f_mug.get('n_collision_free')} of {f_mug.get('n_ik')} teapot-in-mug goals "
                   "are collision-free with the mug in the scene")
    report = {"run": str(run_dir), "teapot_key": tk, "mug_key": mk, "mug_mesh": str(mug_mesh),
              "mug_mesh_stats": mesh_stats(m_mug), "target_T_base_canon": T_target.tolist(),
              "shift_m": d.tolist(), "results": res, "verdict": verdict}
    out = run_dir / "m4" / "mug_obstacle_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"[mesh] {report['mug_mesh_stats']}")
    print(f"[verdict] {verdict}\n          -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
