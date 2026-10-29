#!/usr/bin/env python3
"""General executor: run a list of arm trajectories and gripper commands on
the UR5e, in order, each behind an operator gate. For any method that
produces joint paths (a baseline such as OmniManip with cuRobo underneath,
CBiRRT, a hand-made path); grasp_exec stays the grasp-only run.

    plan.json
        {"steps": [
          {"name": "reach",  "traj": {"joint_names": [6], "positions": [[6], ...],
                                      "velocities": [[6], ...], "dt": 0.02}},
          {"name": "close",  "gripper": {"gap_m": 0.0, "effort_n": 60, "expect_object": true}},
          {"name": "carry",  "traj": {"joint_names": [6], "positions": [[6], ...]}},
          {"name": "open",   "gripper": {"gap_m": 0.085}, "gate": false}
        ]}
            |
            v  every step is built before anything moves:
            v    traj with velocities + dt   the planner's timing, stretched
            v                                uniformly under --speed (cuRobo)
            v    traj, positions only        TOPP-RA under --speed (CBiRRT, IK paths)
            v    each traj must start where the previous traj ended (0.01 rad)
            |
    gates (Enter = go, anything else = stop); "gate": false skips one
    <plan dir>/exec.json      every step: what was sent, how long, the result,
                              the joint state after
    /exec/markers             rviz (base_link): each traj's tool0 path

Joint names travel with the positions, in any order. The first traj must
start where the arm is (else refused; --mock moves there first). A gripper
step with "expect_object": true stops the run if nothing was gripped; the
arm stays where it is. Ctrl-C during a motion cancels it; the e-stop stays
with the operator.

In the ROS2 container (laptop stack and gripper node up):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/exec_plan.py <plan.json> --dry     # build, print, markers
    python3 /data/runs/exec_plan.py <plan.json>
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _grasp_exec():
    spec = importlib.util.spec_from_file_location("grasp_exec", HERE / "grasp_exec.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(spec: dict, ch, ge, speed: float) -> list:
    """Segments for every step. Raises ExecRefusal / MotionRefusal before
    anything moves."""
    from manip_bridge import ur_motion as um
    steps = spec.get("steps") if isinstance(spec, dict) else None
    if not steps:
        raise ge.ExecRefusal("the plan has no 'steps'")
    segs, q_end = [], None
    for i, st in enumerate(steps):
        name = str(st.get("name") or f"step{i}")
        gate = bool(st.get("gate", True))
        if ("traj" in st) == ("gripper" in st):
            raise ge.ExecRefusal(f"step {i} ({name}): needs exactly one of 'traj' or 'gripper'")
        if "gripper" in st:
            g = st["gripper"]
            if g.get("gap_m") is None:
                raise ge.ExecRefusal(f"step {i} ({name}): gripper needs gap_m")
            seg = ge.Segment(name, "gripper", gate=gate, gap_m=float(g["gap_m"]),
                             effort_n=float(g.get("effort_n", ge.CLOSE_EFFORT_N)),
                             note="(must detect an object)" if g.get("expect_object") else "")
            seg.info["expect_object"] = bool(g.get("expect_object", False))
            segs.append(seg)
            continue
        t = st["traj"]
        if not t.get("joint_names") or not t.get("positions"):
            raise ge.ExecRefusal(f"step {i} ({name}): traj needs joint_names and positions")
        Q = um.to_dh_path(t["positions"], t["joint_names"])
        if Q.ndim != 2 or Q.shape[1] != 6 or len(Q) < 2:
            raise ge.ExecRefusal(f"step {i} ({name}): want (N>=2, 6) positions, got {Q.shape}")
        if q_end is not None:
            off = float(np.max(np.abs(Q[0] - q_end)))
            if off > ge.START_TOL:
                raise ge.ExecRefusal(f"step {i} ({name}) starts {off:.3f} rad from where the "
                                     f"previous trajectory ends (> {ge.START_TOL})")
        if t.get("velocities") is not None and t.get("dt") is not None:
            V = um.to_dh_path(t["velocities"], t["joint_names"])
            seg = ge._timed(name, ch, Q, V, float(t["dt"]), speed, note="(planner timing)")
        else:
            seg = ge._traj(name, ch, Q, speed, note="(TOPP-RA)")
        seg.gate = gate
        segs.append(seg)
        q_end = seg.Q[-1]
    if q_end is None:
        raise ge.ExecRefusal("the plan has no traj step")
    return segs


def run(segs, arm, gripper, gate, log, ge, mock=False, speed=0.1, ch=None) -> str:
    """Execute in order; returns the final status string."""
    from manip_bridge import ur_motion as um
    first = next(s for s in segs if s.kind == "traj")
    try:
        um.check_start(arm.q(), first.Q[0], ge.START_TOL)
    except um.MotionRefusal as e:
        if not mock:
            raise ge.ExecRefusal(str(e))
        go = ge._traj("to_start (mock only)", ch, np.linspace(arm.q(), first.Q[0], 50), speed,
                      note="(joint interpolation, NOT checked)")
        if not gate(go):
            return "stopped at to_start"
        ge._do_traj(go, arm, log)
    for s in segs:
        if s.gate and not gate(s):
            return f"stopped at {s.name}"
        if s.kind == "traj":
            ge._do_traj(s, arm, log)
            continue
        r = gripper.command(s.gap_m, s.effort_n)
        log.append({"segment": s.name, "kind": "gripper", "gap_m": s.gap_m, "result": r,
                    "q_after": arm.q().tolist(), "time": time.time()})
        if s.info["expect_object"] and not r.get("stalled"):
            print(f"[exec] NO OBJECT detected at {s.name}: stopping, the arm stays here")
            return f"no object at {s.name}"
    return "done"


def _ros_main(a, ge) -> int:
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray

    from manip_bridge import ur_motion as um

    path = Path(a.plan)
    ch = um.chain()
    segs = build(json.loads(path.read_text()), ch, ge, a.speed)
    print(f"[exec] {len(segs)} steps built (speed {a.speed:.0%}):")
    for s in segs:
        print("   ", s.summary())

    node = ge.start_node("exec_plan")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, "/exec/markers", latched)
    ma = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
    for i, s in enumerate(x for x in segs if x.kind == "traj"):
        m = Marker(header=Header(frame_id="base_link"), ns="exec", id=i, type=Marker.LINE_STRIP)
        m.scale.x = 0.004
        m.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0) if i % 2 == 0 else ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in s.path]
        ma.markers.append(m)
    pub.publish(ma)
    print("[exec] tool0 paths latched on /exec/markers (base_link)")
    if a.dry:
        time.sleep(1.0)
        rclpy.shutdown()
        return 0

    log, status = [], "error"
    try:
        arm = ge.RosArm(node, a.controller)
        needs_gripper = any(s.kind == "gripper" for s in segs)
        gripper = (ge.MockGripper() if a.mock else ge.RosGripper(node, a.gripper_action)) \
            if needs_gripper else None
        status = run(segs, arm, gripper, ge.ask, log, ge, mock=a.mock, speed=a.speed, ch=ch)
    finally:
        out = path.parent / "exec.json"
        out.write_text(json.dumps({"plan": str(path), "status": status, "speed": a.speed,
                                   "controller": a.controller, "mock": a.mock,
                                   "segments": log}, indent=1, default=float) + "\n")
        print(f"[exec] {status} -> {out}")
        rclpy.shutdown()
    return 0 if status == "done" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="plan.json with a 'steps' list")
    ap.add_argument("--speed", type=float, default=0.10, help="fraction of joint limits")
    ap.add_argument("--controller", default="scaled_joint_trajectory_controller")
    ap.add_argument("--gripper-action", default="/gripper/gripper_cmd")
    ap.add_argument("--mock", action="store_true",
                    help="in-process gripper, and allowed to move to the first start (mock hardware only)")
    ap.add_argument("--dry", action="store_true", help="build, print, publish markers; no motion")
    a = ap.parse_args(argv)
    if not 0 < a.speed <= 0.5:
        ap.error("--speed must be in (0, 0.5]")
    ge = _grasp_exec()
    from manip_bridge.ur_motion import MotionRefusal
    try:
        return _ros_main(a, ge)
    except (ge.ExecRefusal, MotionRefusal) as e:
        print(f"[exec] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
