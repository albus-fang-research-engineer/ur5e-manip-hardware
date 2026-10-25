#!/usr/bin/env python3
"""First real motion: tool0 straight up --dz from wherever the arm is, then
straight back down, slowly, each move behind a gate. Exercises everything
grasp_exec's moves go through -- the DH-chain line IK, the retiming, the UR
driver's trajectory action, the arrival check -- with no plan, no perception
and nothing near the table.

Also prints the nominal-vs-calibrated kinematics gap: tool0 by the DH chain
(nominal UR5e, what the executor's lines use) against tool0 on /tf (the
driver's robot_state_publisher, the robot's factory calibration), at the
start and after each move.

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/jog_line.py                     # 3 cm up and back at 1 cm/s
    python3 /data/runs/jog_line.py --dz 0.05 --line-speed 0.02
"""
import argparse
import importlib.util
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


def build(ge, ch, q0, dz, speed, line_speed):
    """(up, down) segments for tool0 +dz in base z from q0 and back."""
    from manip_bridge import ur_motion as um
    p0 = ch.fk(q0)[:3, 3]
    Q, info = um.line(ch, q0, p0 + np.array([0.0, 0.0, dz]))
    up = ge._traj("up", ch, Q, speed, min_duration=abs(dz) / line_speed,
                  note=f"(straight, {dz*100:+.1f} cm in base z)", info=info)
    down = ge._traj("down", ch, Q[::-1], speed, min_duration=abs(dz) / line_speed,
                    note="(the same path reversed)")
    return up, down


def _tf_tool0(node, buf):
    try:
        import rclpy.time
        t = buf.lookup_transform("base_link", "tool0", rclpy.time.Time()).transform.translation
        return np.array([t.x, t.y, t.z])
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dz", type=float, default=0.03, help="m, + is up")
    ap.add_argument("--line-speed", type=float, default=0.01, help="m/s")
    ap.add_argument("--speed", type=float, default=0.10, help="fraction of joint limits")
    ap.add_argument("--controller", default="scaled_joint_trajectory_controller")
    a = ap.parse_args(argv)
    if not 0 < abs(a.dz) <= 0.10:
        ap.error("--dz must be within +-0.10 m")

    ge = _grasp_exec()
    from manip_bridge import ur_motion as um
    try:
        import tf2_ros
        node = ge.start_node("jog_line")
        buf = tf2_ros.Buffer()
        node.tf_listener = tf2_ros.TransformListener(buf, node)   # held, or it is collected
        arm = ge.RosArm(node, a.controller)
        ch = um.chain()
        q0 = arm.q()
        up, down = build(ge, ch, q0, a.dz, a.speed, a.line_speed)

        def report(label):
            q = arm.q()
            p_dh = ch.fk(q)[:3, 3]
            time.sleep(0.3)
            p_tf = _tf_tool0(node, buf)
            gap = f"{np.linalg.norm(p_dh - p_tf)*1000:.1f} mm" if p_tf is not None else "no /tf"
            print(f"[jog] {label}: q {np.round(q, 4).tolist()}\n      tool0 DH {np.round(p_dh, 4).tolist()}"
                  f"  /tf {np.round(p_tf, 4).tolist() if p_tf is not None else '-'}  (gap {gap})")

        report("start")
        for seg in (up, down):
            if not ge.ask(seg):
                print("[jog] stopped")
                return 1
            ge._do_traj(seg, arm, log := [])
            print(f"[jog] {seg.name} done in {log[0]['wall_s']} s (planned {seg.t[-1]:.1f} s), "
                  f"arrival {log[0]['arrival_err_rad']:.4f} rad")
            report(f"after {seg.name}")
        return 0
    except (ge.ExecRefusal, um.MotionRefusal) as e:
        print(f"[jog] REFUSED: {e}", file=sys.stderr)
        return 2
    finally:
        import rclpy
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
