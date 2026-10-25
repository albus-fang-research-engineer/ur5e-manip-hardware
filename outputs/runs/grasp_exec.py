#!/usr/bin/env python3
"""Grasp-only live run: execute the reach plan grasp_reach wrote, then a blind
straight approach, close, a blind lift, and put the object back. Every motion
stops at a gate for the operator first.

    <run>/m3/reach.json         grasp_reach's first success: the cuRobo plan
                                to the standoff (from the RECORDED joint state)
                                + the grasp pose T_base_e and the standoff
            |
            v  all trajectories are built before anything moves:
            v    reach     the cuRobo plan, re-timed at --speed
            v    approach  tool0 straight to the grasp (pads at T_base_e),
            v              IK every 5 mm on the DH chain, at --line-speed
            v    lift      tool0 straight up LIFT_M in base z
            v    lower / retreat   the lift / approach joint paths reversed
            v  any IK refusal stops here, with nothing moved
            |
    gates (Enter = go, anything else = stop):
        open gripper -> [gate] reach -> [gate] approach -> [gate] close
        -> no object detected: open, [gate] retreat, stop
        -> [gate] lift -> [gate] hold ("lower and release?") -> lower -> open
        -> [gate] retreat
    <run>/m3/exec.json          every segment: what was sent, how long, the
                                result, the joint state after
    /grasp/exec_markers         rviz (base_link): each segment's tool0 path

Checks the executor enforces:
  - the robot is at the reach plan's start within START_TOL (0.01 rad) --
    the plan starts at the joint state recorded with the run, so re-run
    grasp_reach if the arm moved since (or --mock: a slow joint move there);
  - the reach plan's last configuration puts tool0 within 5 mm of the planned
    standoff on the DH chain (a joint-order or frame mix-up shows up here);
  - after each trajectory the joints are within ARRIVE_TOL of its last point;
  - Ctrl-C during a motion cancels the trajectory goal (the e-stop stays
    with the operator).

Speeds: --speed is the fraction of the UR5e's joint limits (pi rad/s and,
here, pi rad/s^2) for every trajectory. The reach keeps cuRobo's own timing
(positions, velocities, dt), stretched by one uniform factor until it is
under those caps; the straight lines (untimed, from the line IK) get a
trapezoidal profile and take at least length / --line-speed. Defaults:
10 %, 2 cm/s.

The arm goes through the UR driver's trajectory action
(/<controller>/follow_joint_trajectory, joint names attached); the gripper
through a control_msgs/GripperCommand action (position = jaw gap in metres,
0 closes; result.stalled = an object was gripped), served by the gripper
node on the robot laptop (ros2-essentials robotiq_gripper).

Live (ROS2 container, laptop stack up):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/grasp_exec.py --run /data/runs/<stamp> --dry   # build, print, markers
    python3 /data/runs/grasp_exec.py --run /data/runs/<stamp>

Offline dry run on the driver's mock hardware (same container, second
terminal), --mock = in-process gripper + allowed to move to the plan start:

    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e \\
        robot_ip:=127.0.0.1 use_mock_hardware:=true launch_rviz:=false \\
        initial_joint_controller:=joint_trajectory_controller
    python3 /data/runs/grasp_exec.py --run /data/runs/<stamp> --mock \\
        --controller joint_trajectory_controller
"""
import argparse
import importlib.util
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIFT_M = 0.05
OPEN_GAP_M = 0.085                # Robotiq 2F-85 full stroke
CLOSE_EFFORT_N = 60.0             # gentle; the 2F-85 range is ~20-235 N
START_TOL = 0.01                  # rad
ARRIVE_TOL = 0.01                 # rad
GOAL_CHECK_M = 0.005              # reach end vs planned standoff (DH FK)
SPEED_FRAC = 0.10
LINE_SPEED_M_S = 0.02


class ExecRefusal(Exception):
    """Something the executor will not do; the message names the fix."""


def _grasp_reach():
    spec = importlib.util.spec_from_file_location("grasp_reach", HERE / "grasp_reach.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass
class Segment:
    name: str
    kind: str                                    # "traj" | "gripper" | "hold"
    gate: bool = True
    Q: np.ndarray | None = None                  # (K, 6) DH order, Q[0] = start
    t: np.ndarray | None = None
    V: np.ndarray | None = None
    gap_m: float | None = None
    effort_n: float | None = None
    path: np.ndarray | None = None               # tool0 positions (rviz, log)
    note: str = ""
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.kind == "traj":
            p0, p1 = self.path[0], self.path[-1]
            return (f"{self.name}: {len(self.Q)} pts, {self.t[-1]:.1f} s, max joint speed "
                    f"{np.abs(self.V).max():.3f} rad/s, tool0 {np.round(p0, 3).tolist()} -> "
                    f"{np.round(p1, 3).tolist()} ({np.linalg.norm(p1 - p0)*100:.1f} cm) {self.note}")
        if self.kind == "gripper":
            return f"{self.name}: gripper to {self.gap_m*1000:.0f} mm gap {self.note}"
        return f"{self.name}: {self.note}"


def _traj(name, ch, Q, speed, min_duration=0.0, note="", info=None):
    from manip_bridge import ur_motion as um
    t, V = um.retime(Q, um.UR5E_VMAX * speed, um.UR5E_AMAX * speed, min_duration=min_duration)
    return Segment(name, "traj", Q=np.asarray(Q), t=t, V=V, path=um.tool_path(ch, Q),
                   note=note, info=info or {})


def _fields(seg):
    return {k: getattr(seg, k) for k in ("Q", "t", "V", "path", "note", "info")}


def _timed(name, ch, Q, V, dt, speed, note=""):
    """cuRobo's own timing, uniformly slowed to the --speed caps."""
    from manip_bridge import ur_motion as um
    t, V, k = um.scale_timed(Q, V, dt, um.UR5E_VMAX * speed, um.UR5E_AMAX * speed)
    return Segment(name, "traj", Q=np.asarray(Q), t=t, V=V, path=um.tool_path(ch, Q),
                   note=note, info={"time_scale": k, "planner_dt": float(dt)})


def build(plan: dict, ch, speed=SPEED_FRAC, line_speed=LINE_SPEED_M_S) -> list[Segment]:
    """Every segment of the grasp-only run, from reach.json's plan. Raises
    ExecRefusal / MotionRefusal before anything moves."""
    from manip_bridge import ur_motion as um
    gr = _grasp_reach()
    for k in ("positions", "velocities", "dt", "joint_names", "T_base_e", "T_base_tool0_goal",
              "standoff_m"):
        if plan.get(k) is None:
            raise ExecRefusal(f"reach plan lacks {k!r} -- re-run grasp_reach")
    P = um.to_dh_path(plan["positions"], plan["joint_names"])
    Vp = um.to_dh_path(plan["velocities"], plan["joint_names"])
    T_goal = np.asarray(plan["T_base_tool0_goal"], float)
    err = float(np.linalg.norm(ch.fk(P[-1])[:3, 3] - T_goal[:3, 3]))
    if err > GOAL_CHECK_M:
        raise ExecRefusal(f"the reach plan ends {err*1000:.1f} mm from its own standoff goal on the "
                          f"DH chain (> {GOAL_CHECK_M*1000:.0f} mm): joint order or frame mix-up")
    T_grasp = gr.tool0_goal(np.asarray(plan["T_base_e"], float), 0.0)
    Q_app, i_app = um.line(ch, P[-1], T_grasp[:3, 3])
    p_lift = ch.fk(Q_app[-1])[:3, 3] + np.array([0.0, 0.0, LIFT_M])
    Q_lift, i_lift = um.line(ch, Q_app[-1], p_lift)
    so = float(plan["standoff_m"])
    return [
        Segment("open", "gripper", gate=False, gap_m=OPEN_GAP_M, effort_n=CLOSE_EFFORT_N),
        _timed("reach", ch, P, Vp, float(plan["dt"]), speed,
               note="(cuRobo plan + its own timing, collision-checked)"),
        _traj("approach", ch, Q_app, speed, min_duration=i_app["length_m"] / line_speed,
              note=f"(BLIND, {so*100:.0f} cm along the approach)", info=i_app),
        Segment("close", "gripper", gap_m=0.0, effort_n=CLOSE_EFFORT_N, note="(must detect an object)"),
        _traj("lift", ch, Q_lift, speed, min_duration=i_lift["length_m"] / line_speed,
              note=f"(BLIND, {LIFT_M*100:.0f} cm up)", info=i_lift),
        Segment("hold", "hold", note="holding -- next: lower and release"),
        Segment("lower", "traj", gate=False, **_fields(_traj(
            "lower", ch, Q_lift[::-1], speed, min_duration=i_lift["length_m"] / line_speed,
            note="(the lift reversed; gated by hold)"))),
        Segment("release", "gripper", gate=False, gap_m=OPEN_GAP_M, effort_n=CLOSE_EFFORT_N),
        _traj("retreat", ch, Q_app[::-1], speed, min_duration=i_app["length_m"] / line_speed,
              note="(the approach reversed)"),
    ]


def run(segments: list[Segment], arm, gripper, gate, log: list, mock=False, speed=SPEED_FRAC,
        ch=None) -> str:
    """Execute in order. arm: q() -> DH q, execute(seg) -> (ok, msg);
    gripper: command(gap_m, effort_n) -> {"stalled", "reached_goal", "position"};
    gate(seg) -> bool. Returns the final status string."""
    from manip_bridge import ur_motion as um
    by = {s.name: s for s in segments}
    reach = by["reach"]
    try:
        um.check_start(arm.q(), reach.Q[0], START_TOL)
    except um.MotionRefusal as e:
        if not mock:
            raise ExecRefusal(str(e))
        q_now = arm.q()
        Qs = np.linspace(q_now, reach.Q[0], 50)
        go = _traj("to_start (mock only)", ch, Qs, speed, note="(joint interpolation, NOT checked)")
        if not gate(go):
            return "stopped at to_start"
        _do_traj(go, arm, log)

    def do(seg):
        if seg.gate and not gate(seg):
            return False
        if seg.kind == "traj":
            _do_traj(seg, arm, log)
        elif seg.kind == "gripper":
            r = gripper.command(seg.gap_m, seg.effort_n)
            log.append({"segment": seg.name, "kind": "gripper", "gap_m": seg.gap_m, "result": r,
                        "q_after": arm.q().tolist(), "time": time.time()})
            seg.info["result"] = r
        else:
            log.append({"segment": seg.name, "kind": "hold", "time": time.time()})
        return True

    for name in ("open", "reach", "approach", "close"):
        if not do(by[name]):
            return f"stopped at {name}"
    if not by["close"].info["result"].get("stalled"):
        print("[exec] NO OBJECT detected after closing: opening and retreating")
        do(Segment("open (no object)", "gripper", gate=False, gap_m=OPEN_GAP_M,
                   effort_n=CLOSE_EFFORT_N))
        return "no object" if do(by["retreat"]) else "no object, stopped at retreat"
    for name in ("lift", "hold", "lower", "release", "retreat"):
        if not do(by[name]):
            return f"stopped at {name}"
    return "done"


def _do_traj(seg, arm, log):
    t0 = time.time()
    ok, msg = arm.execute(seg)
    q = arm.q()
    off = float(np.max(np.abs(q - seg.Q[-1])))
    log.append({"segment": seg.name, "kind": "traj", "n_points": len(seg.Q),
                "duration_s": float(seg.t[-1]), "wall_s": round(time.time() - t0, 2),
                "ok": bool(ok), "msg": msg, "q_after": q.tolist(), "arrival_err_rad": off,
                "info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in seg.info.items() if k != "result"},
                "time": time.time()})
    if not ok:
        raise ExecRefusal(f"{seg.name}: the trajectory failed ({msg})")
    if off > ARRIVE_TOL:
        raise ExecRefusal(f"{seg.name}: ended {off:.3f} rad from its last point (> {ARRIVE_TOL})")


def ask(seg) -> bool:
    print(f"\n[gate] {seg.summary()}")
    return input("[gate] Enter = go, anything else = stop: ").strip() == ""


class MockGripper:
    def command(self, gap_m, effort_n):
        print(f"[mock gripper] -> {gap_m*1000:.0f} mm")
        closing = gap_m < OPEN_GAP_M / 2
        return {"position": 0.02 if closing else gap_m, "effort": effort_n,
                "stalled": closing, "reached_goal": not closing}


# ------------------------------------------------------------------ ROS shell

def start_node(name):
    """rclpy init + a node spun by a background executor (the action waits
    below poll their futures from the main thread)."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    rclpy.init()
    node = Node(name)
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    threading.Thread(target=exe.spin, daemon=True).start()
    return node


def _wait(fut, on_interrupt=None):
    try:
        while not fut.done():
            time.sleep(0.01)
    except KeyboardInterrupt:
        if on_interrupt:
            on_interrupt()
        raise
    return fut.result()


class RosArm:
    """The UR driver: /joint_states in, FollowJointTrajectory out (DH-order
    names attached; the controller matches by name)."""

    def __init__(self, node, controller):
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState

        from manip_bridge import ur_motion as um
        self._um, self._js = um, {}
        self._FJT = FollowJointTrajectory

        def on_js(msg):
            if all(n in msg.name for n in um.DH_ORDER):
                self._js["q"] = um.to_dh(msg.position, msg.name)
        node.create_subscription(JointState, "/joint_states", on_js, 10)
        self.ac = ActionClient(node, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        if not self.ac.wait_for_server(timeout_sec=10.0):
            raise ExecRefusal(f"no /{controller}/follow_joint_trajectory action server -- "
                              "is the UR driver up (and is that the active controller)?")

    def q(self):
        t_end = time.time() + 5.0
        while "q" not in self._js and time.time() < t_end:
            time.sleep(0.05)
        if "q" not in self._js:
            raise ExecRefusal("no /joint_states with all six UR arm joints")
        time.sleep(0.2)                           # a fresh sample after motion
        return self._js["q"].copy()

    def execute(self, seg):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint
        g = self._FJT.Goal()
        g.trajectory.joint_names = list(self._um.DH_ORDER)
        for k in range(1, len(seg.Q)):            # point 0 is where the robot is
            p = JointTrajectoryPoint(positions=seg.Q[k].tolist(), velocities=seg.V[k].tolist())
            sec = float(seg.t[k])
            p.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
            g.trajectory.points.append(p)
        gh = _wait(self.ac.send_goal_async(g))
        if not gh.accepted:
            return False, "goal rejected"

        def cancel():
            gh.cancel_goal_async()
            print("\n[exec] Ctrl-C: trajectory goal cancelled")
        r = _wait(gh.get_result_async(), cancel).result
        return r.error_code == 0, f"error_code {r.error_code} {r.error_string}".strip()


class RosGripper:
    """control_msgs/GripperCommand: position = jaw gap (m), stalled = gripped."""

    def __init__(self, node, action):
        from control_msgs.action import GripperCommand
        from rclpy.action import ActionClient
        self._GC = GripperCommand
        self.ac = ActionClient(node, GripperCommand, action)
        if not self.ac.wait_for_server(timeout_sec=10.0):
            raise ExecRefusal(f"no {action} action server -- is the gripper node up on the robot "
                              "laptop? (--mock for an in-process stand-in)")

    def command(self, gap_m, effort_n):
        g = self._GC.Goal()
        g.command.position, g.command.max_effort = float(gap_m), float(effort_n)
        gh = _wait(self.ac.send_goal_async(g))
        if not gh.accepted:
            raise ExecRefusal("gripper goal rejected")
        r = _wait(gh.get_result_async()).result
        return {"position": r.position, "effort": r.effort, "stalled": bool(r.stalled),
                "reached_goal": bool(r.reached_goal)}


def _ros_main(a) -> int:
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray

    from manip_bridge import ur_motion as um

    run_dir = Path(a.run)
    rec = json.loads((run_dir / "m3" / "reach.json").read_text())
    if not rec.get("plan"):
        raise ExecRefusal(f"{run_dir / 'm3/reach.json'} has no successful plan -- nothing to execute")
    ch = um.chain()
    segs = build(rec["plan"], ch, a.speed, a.line_speed)
    print(f"[exec] {len(segs)} segments built (speed {a.speed:.0%}, lines {a.line_speed*100:.0f} cm/s):")
    for s in segs:
        print("   ", s.summary())

    node = start_node("grasp_exec")

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, "/grasp/exec_markers", latched)
    ma = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
    colors = {"reach": (0.1, 0.9, 0.1), "approach": (1.0, 0.8, 0.0), "lift": (0.2, 0.4, 1.0)}
    for i, s in enumerate(x for x in segs if x.name in colors):
        m = Marker(header=Header(frame_id="base_link"), ns="exec", id=i, type=Marker.LINE_STRIP)
        m.scale.x = 0.004
        c = colors[s.name]
        m.color = ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=1.0)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in s.path]
        ma.markers.append(m)
    pub.publish(ma)
    print("[exec] tool0 paths latched on /grasp/exec_markers (base_link)")
    if a.dry:
        time.sleep(1.0)
        rclpy.shutdown()
        return 0

    log = []
    status = "error"
    try:
        status = run(segs, RosArm(node, a.controller),
                     MockGripper() if a.mock else RosGripper(node, a.gripper_action), ask, log,
                     mock=a.mock, speed=a.speed, ch=ch)
    finally:
        out = run_dir / "m3" / "exec.json"
        out.write_text(json.dumps({"status": status, "speed": a.speed, "line_speed": a.line_speed,
                                   "controller": a.controller, "mock": a.mock, "lift_m": LIFT_M,
                                   "segments": log}, indent=1, default=float) + "\n")
        print(f"[exec] {status} -> {out}")
        rclpy.shutdown()
    return 0 if status == "done" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--speed", type=float, default=SPEED_FRAC, help="fraction of joint limits")
    ap.add_argument("--line-speed", type=float, default=LINE_SPEED_M_S, help="m/s, straight lines")
    ap.add_argument("--controller", default="scaled_joint_trajectory_controller")
    ap.add_argument("--gripper-action", default="/gripper/gripper_cmd")
    ap.add_argument("--mock", action="store_true",
                    help="in-process gripper, and allowed to move to the plan start (mock hardware only)")
    ap.add_argument("--dry", action="store_true", help="build, print, publish markers; no motion")
    a = ap.parse_args(argv)
    if not 0 < a.speed <= 0.5:
        ap.error("--speed must be in (0, 0.5]")
    from manip_bridge.ur_motion import MotionRefusal
    try:
        return _ros_main(a)
    except (ExecRefusal, MotionRefusal) as e:
        print(f"[exec] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
