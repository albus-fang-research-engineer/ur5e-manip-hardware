#!/usr/bin/env python3
"""M3 step 4: can the arm reach a standoff in front of each filtered grasp,
collision-free against the live ESDF? cuRobo's `plan` from the run's
recorded joint state to the tool0 pose that puts the pads STANDOFF behind
the grasp along its approach. No motion; output is a plan to look at.

    /grasp/<obj>/filtered   grasp_filter (base_link, e axes: +z approach,
                            +x closing, origin = pad centre), its ranking
    <run>/summary.json      packet.joints {q, names}: the start state
            |
            v  per grasp in that order, per standoff in STANDOFFS_M:
            v  T_base_tool0 = T_base_e . trans(0, 0, -standoff) . inv(T_TOOL0_PAD)
            v  curobo sidecar {"cmd": "plan", q, joint_names, T_base_goal}
    <run>/m3/reach.json            every attempt (grasp, standoff, success,
                                   errors, solve time) + the first success's
                                   trajectory and end-effector path
    /grasp/<obj>/reach_markers     rviz: the planned tool path (green), each
                                   tried tool0 goal as a triad, latched

Standoffs: 10 cm first; only if no grasp reaches at 10 cm, 5 cm (backing off
along an elevated approach moves the goal up and out, so a 10 cm failure is
not yet a reach verdict).

The teapot stays in the ESDF: the standoff is outside it, and the arm must
avoid it on the way. Carving the target out is step 5's business (the
fingers entering the handle).

T_TOOL0_PAD (PROVISIONAL -- replaced by the ruler measurement, flange face to
pad centre with the gripper closed on the handle, before any motion):
rotation identity (the 2F-85 mounts with coupler rotation 0: tool0 +z is the
approach, the knuckles sit at +-x so the jaws close along tool0 x), offset
along tool0 z = 0.150 m. Derived from robotiq_description's URDF chain and
the left fingertip collision STL (metres, no scale in the xacro): tool0 ->
+0.011 adapter -> knuckle (0.0306, 0, 0.0549) -> finger (0.0315, 0, -0.0038)
-> fingertip joint (0.0056, 0, 0.0472), the tip counter-rotating so the pads
stay parallel; the pad's flat inner face spans z 0.136-0.174 m in tool0 at a
10-25 mm jaw gap (knuckle q 0.6-0.7), centre 0.1506-0.1509 m. Fully open (85
mm) it is 0.137 m. So 0.150 m for a handle-sized gap, +-1 mm across 10-45 mm.

In the ros2 container (bridges up -- curobo_bridge integrating the bag into
the ESDF; grasp_filter has published /grasp/<obj>/filtered):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/grasp_reach.py --run /data/runs/<stamp> --obj m2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TOOL0_PAD_Z_M = 0.150          # PROVISIONAL, see the docstring
STANDOFFS_M = (0.10, 0.05)
CUROBO_ADDR = os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671")
PLAN_TIMEOUT_MS = 600_000      # the first plan builds the MotionPlanner (warmup + graph capture)


class ReachRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


def T_tool0_pad() -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = TOOL0_PAD_Z_M
    return T


def tool0_goal(T_base_e: np.ndarray, standoff: float) -> np.ndarray:
    """tool0 pose that puts the pad centre `standoff` behind the grasp along
    its approach (+z of e), same orientation."""
    back = np.eye(4)
    back[2, 3] = -float(standoff)
    return np.asarray(T_base_e, float) @ back @ np.linalg.inv(T_tool0_pad())


def pose_msg_to_T(p) -> np.ndarray:
    T = np.eye(4)
    q = p.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def start_state(run: Path) -> tuple[list, list]:
    """(q, joint_names) from summary.json packet.joints -- names travel WITH q:
    the sidecar reorders by name, so a bare q would plan from a wrong pose."""
    s = json.loads((run / "summary.json").read_text())
    j = (s.get("packet") or {}).get("joints") or {}
    q, names = j.get("q"), j.get("names")
    if q is None or names is None or len(q) != len(names):
        raise ReachRefusal(f"{run / 'summary.json'}: packet.joints has no q + names "
                           f"(status {j.get('status')!r}) -- no recorded start state")
    return [float(x) for x in q], [str(n) for n in names]


def attempts(grasps: list[tuple[np.ndarray, float]], q, names, call,
             standoffs=STANDOFFS_M) -> tuple[list[dict], dict | None]:
    """Try each grasp (in the given order) at each standoff; stop at the
    first success. A shorter standoff is tried only after every grasp failed
    at the longer one. `call(payload) -> reply` is the sidecar."""
    rows = []
    for so in standoffs:
        for k, (T_e, score) in enumerate(grasps):
            T_goal = tool0_goal(T_e, so)
            t0 = time.time()
            rep = call({"cmd": "plan", "q": list(q), "joint_names": list(names),
                        "T_base_goal": T_goal.astype(np.float32)})
            row = {"grasp": k, "score": float(score), "standoff_m": so,
                   "success": bool(rep.get("success")),
                   "position_error": rep.get("position_error"),
                   "rotation_error": rep.get("rotation_error"),
                   "solve_time": rep.get("solve_time"), "wall_s": round(time.time() - t0, 2),
                   "error": rep.get("error"),
                   "T_base_tool0_goal": T_goal.tolist(), "T_base_e": np.asarray(T_e).tolist()}
            rows.append(row)
            if row["success"]:
                plan = {k2: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                        for k2, v in rep.items() if k2 in ("joint_names", "positions",
                                                           "velocities", "dt", "ee_path")}
                return rows, {**row, **plan}
    return rows, None


# ------------------------------------------------------------------ ROS shell

def _ros_main(a) -> int:
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray

    from manip_bridge.zmq_client import SidecarClient
    from manip_interfaces.msg import GraspArray

    run = Path(a.run)
    q, names = start_state(run)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    rclpy.init()
    node = Node("grasp_reach")
    got = {}
    node.create_subscription(GraspArray, f"/grasp/{a.obj}/filtered",
                             lambda m: got.setdefault("msg", m), latched)
    t_end = time.time() + a.wait
    while "msg" not in got and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "msg" not in got:
        raise ReachRefusal(f"no message on /grasp/{a.obj}/filtered within {a.wait:g} s -- "
                           "is grasp_filter running with its TSR and grasps?")
    msg = got["msg"]
    if msg.header.frame_id != "base_link":
        raise ReachRefusal(f"/grasp/{a.obj}/filtered is in {msg.header.frame_id!r}, "
                           "the planner plans in base_link")
    grasps = [(pose_msg_to_T(g.pose), g.score) for g in msg.grasps]
    if not grasps:
        raise ReachRefusal(f"/grasp/{a.obj}/filtered is empty (route empty) -- nothing to reach")
    print(f"[reach] {len(grasps)} grasp(s) from /grasp/{a.obj}/filtered; start q from "
          f"summary ({len(q)} joints); tool0->pad {TOOL0_PAD_Z_M:.3f} m (provisional)")

    cli = SidecarClient(CUROBO_ADDR, timeout_ms=PLAN_TIMEOUT_MS)
    rows, best = attempts(grasps, q, names, cli.call)
    for r in rows:
        err = (f"pos {r['position_error']:.4f} m, rot {r['rotation_error']:.4f}"
               if r["success"] else (r["error"] or "failed"))
        print(f"    grasp {r['grasp']} (score {r['score']:.3f}) standoff {r['standoff_m']*100:.0f} cm: "
              f"{'OK' if r['success'] else 'FAIL'}  {err}  ({r['wall_s']} s)")

    out = run / "m3"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"obj": a.obj, "tool0_pad_z_m": TOOL0_PAD_Z_M, "tool0_pad_status": "provisional (URDF)",
           "start": {"q": q, "joint_names": names}, "standoffs_m": list(STANDOFFS_M),
           "attempts": rows, "plan": best, "curobo": CUROBO_ADDR}
    (out / "reach.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[reach] {'reached: grasp %d at %d cm' % (best['grasp'], round(best['standoff_m']*100)) if best else 'NO grasp reached at any standoff'}"
          f" -> {out / 'reach.json'}")

    # rviz: tried goals as triads, the planned tool path as a line
    pub = node.create_publisher(MarkerArray, f"/grasp/{a.obj}/reach_markers", latched)
    hdr = Header(frame_id="base_link")
    ma = MarkerArray()
    ma.markers.append(Marker(action=Marker.DELETEALL))

    def line(ns, mid, pts, rgba, w):
        m = Marker(header=hdr, ns=ns, id=mid, type=Marker.LINE_LIST, action=Marker.ADD)
        m.scale.x = w
        m.color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=1.0)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        return m
    for i, r in enumerate(rows):
        T = np.asarray(r["T_base_tool0_goal"])
        o = T[:3, 3]
        pts = sum(([o, o + 0.05 * T[:3, j]] for j in range(3)), [])
        ma.markers.append(line("tool0_goal", i, pts, (0.2, 0.8, 0.2) if r["success"] else (0.6, 0.6, 0.6),
                               0.004 if r["success"] else 0.002))
    if best:
        P = np.asarray(best["ee_path"])
        seg = sum(([P[k], P[k + 1]] for k in range(len(P) - 1)), [])
        ma.markers.append(line("tool_path", 0, seg, (0.1, 0.9, 0.1), 0.005))
    pub.publish(ma)
    print(f"[reach] markers latched on /grasp/{a.obj}/reach_markers (base_link); Ctrl-C to exit")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0 if best else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--obj", required=True, help="grasp_filter's obj tag (e.g. m2)")
    ap.add_argument("--wait", type=float, default=10.0, help="s to wait for the filtered grasps")
    a = ap.parse_args(argv)
    try:
        return _ros_main(a)
    except ReachRefusal as e:
        print(f"[reach] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
