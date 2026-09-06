"""ROS2 <-> curobo sidecar bridge for the constrained (CBiRRT) planner.

    service   /planner/plan_constrained   manip_interfaces/PlanConstrained
    subs      joints_topic                sensor_msgs/JointState  (start when request.start empty)
    pubs      /planner/trajectory         trajectory_msgs/JointTrajectory (latched)
              /planner/ee_path            nav_msgs/Path  tool0 positions, base_frame (latched)
              /planner/body_path          nav_msgs/Path  body positions, base_frame (latched)

Thin by design: this node roots TSRs into base_frame through TF, picks the
start config, forwards the request to the sidecar's "plan_constrained"
command and republishes the reply. The goal funnel, IK, collision and the
planner itself live in the sidecar (curobo_server/plan_constrained.py); the
geometry lives in manip_tsr / manip_cbirrt. Nothing here imports either.

TSR frames: subgoal / path TSRs arrive with header.frame_id = the frame their
t0_w is expressed in -- the tracked object frame for grasp-anchored regions
(t0_w = canonical frame at the anchor, in the body frame), base_frame for
world-anchored ones. Each is rooted by T_base_frame @ t0_w with a latest TF
lookup at request time; that lookup IS the "frozen at stage entry".

Same env/param conventions as curobo_bridge: CUROBO_ADDR, base_frame,
joints_topic. First call may take ~10 s while the sidecar builds its
collision + IK oracles (warp JIT is cached across runs).
"""

import json
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from manip_interfaces.srv import PlanConstrained

from .zmq_client import SidecarClient, SidecarError

CUROBO_ADDR = os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671")
ORACLE_BUILD_MS = 300_000       # first call: oracle construction + IK warmup


def _tf_to_mat(tf):
    q, t = tf.transform.rotation, tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _pose_to_mat(pose):
    q, p = pose.orientation, pose.position
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.x, p.y, p.z]
    return T


class ConstrainedPlannerBridge(Node):
    def __init__(self):
        super().__init__("constrained_planner")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("joints_topic", "/joint_states")
        self.declare_parameter("waypoint_dt", 0.5)     # s between path waypoints in the JointTrajectory
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.client = SidecarClient(CUROBO_ADDR, timeout_ms=ORACLE_BUILD_MS)
        self._joints = None
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

        self.create_subscription(JointState, gp("joints_topic"), self._on_joints,
                                 qos_profile_sensor_data)
        # single-threaded executor: the plan call blocks the node for its
        # duration (seconds); the start config is captured before the call
        self.create_service(PlanConstrained, "/planner/plan_constrained", self._on_plan)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_traj = self.create_publisher(JointTrajectory, "/planner/trajectory", latched)
        self.pub_ee = self.create_publisher(Path, "/planner/ee_path", latched)
        self.pub_body = self.create_publisher(Path, "/planner/body_path", latched)
        self.get_logger().info(f"constrained_planner up -> {CUROBO_ADDR} "
                               f"(base {gp('base_frame')}, joints {gp('joints_topic')})")

    def _on_joints(self, msg):
        if msg.name and msg.position:
            self._joints = (list(msg.name), list(msg.position))

    # ---------------------------------------------------------------- frames
    def _root(self, tsr_msg, base):
        """TSR message -> flat dict with t0_w rooted in base_frame."""
        T0_w = np.asarray(tsr_msg.t0_w, float).reshape(4, 4)
        fid = tsr_msg.header.frame_id
        if fid and fid != base:
            tf = self._tf.lookup_transform(base, fid, Time())      # raises if absent
            T0_w = _tf_to_mat(tf) @ T0_w
        return {"t0_w": T0_w.reshape(-1).tolist(), "tw_e": list(tsr_msg.tw_e),
                "bw": list(tsr_msg.bw), "name": tsr_msg.name, "frame_id": fid}

    # --------------------------------------------------------------- service
    def _on_plan(self, req, res):
        base = self.get_parameter("base_frame").value
        try:
            subgoal = self._root(req.subgoal, base)
            paths = [self._root(p, base) for p in req.path]
        except Exception as e:
            res.success, res.message = False, f"TSR frame not resolvable on TF into {base}: {e}"
            return res

        if req.start.name and req.start.position:
            names, q = list(req.start.name), list(req.start.position)
        elif self._joints is not None:
            names, q = self._joints
        else:
            res.success, res.message = False, "no start: request.start empty and no /joint_states yet"
            return res

        spheres = np.asarray(req.attached_spheres, np.float32)
        if spheres.size % 4:
            res.success, res.message = False, f"attached_spheres must be 4 per sphere, got {spheres.size}"
            return res

        msg = {"cmd": "plan_constrained",
               "q_start": np.asarray(q, np.float32), "joint_names": names,
               "T_ee_body": _pose_to_mat(req.t_ee_body).reshape(-1).tolist(),
               "subgoal": subgoal, "path": paths,
               "n_goal_samples": int(req.n_goal_samples) or 60,
               "timeout": float(req.timeout) or 20.0,
               "eps": float(req.eps) or 0.10,
               "constraint_tol": float(req.constraint_tol) or 2e-3,
               "clearance_margin": float(req.clearance_margin),
               "seed": int(req.seed),
               "attached_spheres": spheres.reshape(-1, 4) if spheres.size else None}
        self.get_logger().info(
            f"plan_constrained: subgoal '{subgoal['name']}' ({subgoal['frame_id'] or base}) "
            f"x {len(paths)} path TSR(s), {len(spheres) // 4} attached spheres, "
            f"n={msg['n_goal_samples']} timeout={msg['timeout']}s")
        try:
            rep = self.client.call(msg, timeout_ms=int(msg["timeout"] * 1000) + ORACLE_BUILD_MS)
        except (TimeoutError, SidecarError) as e:
            res.success, res.message = False, f"sidecar: {e}"
            self.get_logger().error(res.message)
            return res

        f = rep.get("funnel", {})
        res.funnel_json = json.dumps(f)
        res.tree_sizes = [int(x) for x in rep.get("tree_sizes", [])]
        res.n_collision_calls = int(rep.get("n_collision_calls", 0))
        res.total_time = float(rep.get("total_time", 0.0))
        self.get_logger().info(
            f"funnel: {f.get('n_sampled')} sampled (acc {f.get('acceptance_rate', 0):.2f}) -> "
            f"{f.get('n_ik')} IK -> {f.get('n_collision_free')} free -> {f.get('n_contained')} contained; "
            f"{f.get('n_plan_attempts')} plan attempt(s)")
        if not rep.get("success"):
            res.success, res.message = False, str(rep.get("reason", "planner failed"))
            self.get_logger().warn(f"plan_constrained failed: {res.message}")
            return res

        hdr = Header(frame_id=base, stamp=self.get_clock().now().to_msg())
        dt = float(self.get_parameter("waypoint_dt").value)
        traj = JointTrajectory(header=hdr, joint_names=list(rep["joint_names"]))
        for i, row in enumerate(np.asarray(rep["positions"], np.float64)):
            pt = JointTrajectoryPoint(positions=row.tolist())
            pt.time_from_start.sec = int(i * dt)
            pt.time_from_start.nanosec = int(((i * dt) % 1.0) * 1e9)
            traj.points.append(pt)

        def path_of(key):
            pth = Path(header=hdr)
            for p in np.asarray(rep[key], np.float64):
                ps = PoseStamped(header=hdr)
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
                ps.pose.orientation.w = 1.0
                pth.poses.append(ps)
            return pth

        res.success, res.message = True, ""
        res.trajectory = traj
        res.ee_path = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in rep["ee_path"]]
        res.body_path = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in rep["body_path"]]
        res.max_excess = float(rep.get("max_excess", 0.0))
        self.pub_traj.publish(traj)
        self.pub_ee.publish(path_of("ee_path"))
        self.pub_body.publish(path_of("body_path"))
        self.get_logger().info(
            f"plan ok: {len(traj.points)} waypoints, max_excess {res.max_excess:.4f}, "
            f"trees {res.tree_sizes}, {res.n_collision_calls} collision calls, {res.total_time:.1f}s")
        return res


def main():
    rclpy.init(args=sys.argv)
    node = ConstrainedPlannerBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
