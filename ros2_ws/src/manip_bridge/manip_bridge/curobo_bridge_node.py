"""ROS2 Humble <-> cuRoboV2 mapper/planner bridge.

Streams bag (or live) RGB-D + joint states into the curobo sidecar's TSDF
with UR5e self-masking, publishes the ESDF for RViz2, and plans to commanded
poses with the MotionPlanner against the live ESDF.

Subscribes
    depth_topic   sensor_msgs/Image        16UC1 (mm) or 32FC1 (m)
    info_topic    sensor_msgs/CameraInfo
    joints_topic  sensor_msgs/JointState   arm joints (extras ignored)
    /curobo/goal_pose  geometry_msgs/PoseStamped   tool0 goal -> plan

Publishes
    /curobo/esdf_cloud  sensor_msgs/PointCloud2   voxels with esdf <= esdf_max_dist
                        (obstacle shell; color by the "esdf" field in RViz)
    /curobo/esdf_slice  sensor_msgs/PointCloud2   dense z-layer at esdf_slice_z
                        (distance-field heatmap; NaN param = off)
    /curobo/trajectory  trajectory_msgs/JointTrajectory
    /curobo/ee_path     nav_msgs/Path             tool positions of the plan

FRAME CONTRACT (matches the server): the mapper's map frame IS base_frame.
The camera pose sent on every integrate is T_base_cam, looked up on TF
(base_frame <- depth header frame_id, override with camera_frame param); if
your bag has no TF, set the static t_base_cam parameter (row-major 16). All
ESDF clouds and plans are therefore in base_frame -- set RViz Fixed Frame
accordingly.

Bag replay:
    ros2 bag play <bag> --clock
    ros2 run manip_bridge curobo_bridge --ros-args -p use_sim_time:=true \
        -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
        -p info_topic:=/camera/color/camera_info
    rviz2 -d test/curobo_esdf.rviz
    # then, e.g.:
    ros2 topic pub -1 /curobo/goal_pose geometry_msgs/PoseStamped \
        '{header: {frame_id: base_link}, pose: {position: {x: 0.4, y: 0.1, \
        z: 0.3}, orientation: {x: 1.0, w: 0.0}}}'

Self-masking starts with the first integrated frame (the node holds frames
until a joint state + TF are available), so the arm never enters the TSDF.
The first plan request builds the MotionPlanner in the sidecar (warmup +
CUDA-graph capture -- expect ~a minute; the 5 min plan timeout covers it).

Joint states and depth are paired by "latest at integrate time", not a
time-synchronizer -- fine at the default 2 Hz integrate rate with tabletop
arm speeds; tighten if you integrate faster while the arm is moving.
"""

import os

import numpy as np
import rclpy
import zmq
import msgpack
import msgpack_numpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation

msgpack_numpy.patch()

CUROBO_ADDR = os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671")
INTEGRATE_TIMEOUT_MS = 300_000   # first integrate pays NVRTC + warp JIT
PLAN_TIMEOUT_MS = 300_000        # first plan pays planner build + warmup


def _depth_to_m(msg):
    h, w = msg.height, msg.width
    if msg.encoding == "16UC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        return d.reshape(h, msg.step // 2)[:, :w].astype(np.float32) * 1e-3
    if msg.encoding == "32FC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.float32)
        return np.nan_to_num(d.reshape(h, msg.step // 4)[:, :w].copy())
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


def _tf_to_mat(tf):
    q = tf.transform.rotation
    t = tf.transform.translation
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _pose_to_mat(pose):
    q, p = pose.orientation, pose.position
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def _xyzi_cloud(header, xyz, intensity, field_name="esdf"):
    """PointCloud2 from (N,3) float32 + (N,) float32, built as one buffer."""
    n = xyz.shape[0]
    data = np.empty((n, 4), np.float32)
    data[:, :3] = xyz
    data[:, 3] = intensity
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name=field_name, offset=12, datatype=PointField.FLOAT32,
                   count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


class CuroboBridge(Node):
    def __init__(self):
        super().__init__("curobo_bridge")

        # same env convention as bridges.launch.py / run_scene: .env sets the
        # topics once per bag, `ros2 run` picks them up without launch args
        self.declare_parameter("depth_topic", os.environ.get(
            "DEPTH_TOPIC", "/camera/camera/aligned_depth_to_color/image_raw"))
        self.declare_parameter("info_topic", os.environ.get(
            "INFO_TOPIC", "/camera/camera/color/camera_info"))
        self.declare_parameter("joints_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "")     # "" = depth frame_id
        self.declare_parameter("t_base_cam", [0.0] * 16)  # static fallback;
        # all-zeros = disabled, use TF
        self.declare_parameter("self_mask", True)
        self.declare_parameter("integrate_rate_hz", 2.0)
        self.declare_parameter("esdf_rate_hz", 0.5)
        self.declare_parameter("esdf_max_dist", 0.03)  # obstacle-shell cut
        self.declare_parameter("esdf_slice_z", float("nan"))  # NaN = off

        self._zctx = zmq.Context()
        self._sock = None
        self._connect()

        self._depth = None          # (msg_header, HxW float32 m)
        self._K = None
        self._joints = None         # (names, positions)
        self._robot_loaded = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        gp = lambda n: self.get_parameter(n).value
        self.create_subscription(Image, gp("depth_topic"), self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, gp("info_topic"), self._on_info, 1)
        self.create_subscription(JointState, gp("joints_topic"),
                                 self._on_joints, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/curobo/goal_pose",
                                 self._on_goal, 1)

        self._pub_cloud = self.create_publisher(PointCloud2,
                                                "/curobo/esdf_cloud", 1)
        self._pub_slice = self.create_publisher(PointCloud2,
                                                "/curobo/esdf_slice", 1)
        self._pub_traj = self.create_publisher(JointTrajectory,
                                               "/curobo/trajectory", 1)
        self._pub_path = self.create_publisher(Path, "/curobo/ee_path", 1)

        self.create_timer(1.0 / gp("integrate_rate_hz"), self._integrate_tick)
        self.create_timer(1.0 / gp("esdf_rate_hz"), self._esdf_tick)
        self.get_logger().info(f"curobo bridge up -> {CUROBO_ADDR}")

    # ------------------------------------------------------------------ zmq
    def _connect(self):
        if self._sock is not None:
            self._sock.close(0)
        self._sock = self._zctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, INTEGRATE_TIMEOUT_MS)
        self._sock.connect(CUROBO_ADDR)

    def _rpc(self, req, timeout_ms=None):
        try:
            if timeout_ms is not None:
                self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self._sock.send(msgpack.packb(req, use_bin_type=True))
            return msgpack.unpackb(self._sock.recv(), raw=False)
        except zmq.Again:
            self._connect()   # REQ socket is stuck; rebuild
            return {"ok": False, "error": "timeout"}
        finally:
            self._sock.setsockopt(zmq.RCVTIMEO, INTEGRATE_TIMEOUT_MS)

    # ------------------------------------------------------------ callbacks
    def _on_depth(self, msg):
        self._depth = (msg.header, _depth_to_m(msg))

    def _on_info(self, msg):
        self._K = np.array(msg.k, np.float32).reshape(3, 3)

    def _on_joints(self, msg):
        if msg.name and msg.position:
            self._joints = (list(msg.name), list(msg.position))

    # ------------------------------------------------------------- T_base_cam
    def _t_base_cam(self, header):
        static = np.array(self.get_parameter("t_base_cam").value, np.float32)
        if np.any(static):
            return static.reshape(4, 4)
        cam = self.get_parameter("camera_frame").value or header.frame_id
        base = self.get_parameter("base_frame").value
        try:
            tf = self._tf_buffer.lookup_transform(base, cam, Time())
        except Exception as e:
            self.get_logger().warn(
                f"no TF {base} <- {cam} ({e}); set t_base_cam param if the "
                f"bag has no TF", throttle_duration_sec=5.0)
            return None
        return _tf_to_mat(tf)

    # -------------------------------------------------------------- integrate
    def _integrate_tick(self):
        if self._depth is None or self._K is None:
            return
        header, depth = self._depth
        T = self._t_base_cam(header)
        if T is None:
            return
        mask = self.get_parameter("self_mask").value
        if mask and self._joints is None:
            # hold frames until the arm state arrives so it never bakes in
            self.get_logger().warn("waiting for joint states before first "
                                   "integrate (self_mask on)",
                                   throttle_duration_sec=5.0)
            return
        req = {"cmd": "integrate", "depth": depth, "intrinsics": self._K,
               "pose": T}
        if mask:
            names, q = self._joints
            req["q"] = np.array(q, np.float32)
            req["joint_names"] = names
        rep = self._rpc(req)
        if not rep.get("ok"):
            self.get_logger().error(f"integrate: {rep.get('error')}")
        elif not self._robot_loaded and "n_masked" in rep:
            self._robot_loaded = True
            self.get_logger().info(
                f"self-mask active ({rep['n_masked']} robot px on first frame)")
        self._depth = None   # each frame integrates at most once

    # ------------------------------------------------------------------ esdf
    def _esdf_tick(self):
        req = {"cmd": "esdf",
               "sparse_below": float(self.get_parameter("esdf_max_dist").value)}
        slice_z = float(self.get_parameter("esdf_slice_z").value)
        if not np.isnan(slice_z):
            req["slice_z"] = slice_z
        rep = self._rpc(req)
        if not rep.get("ok"):
            self.get_logger().warn(f"esdf: {rep.get('error')}",
                                   throttle_duration_sec=5.0)
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.get_parameter("base_frame").value
        low = np.asarray(rep["low"], np.float32)
        vs = float(rep["voxel_size"])

        idx = np.asarray(rep["sparse_idx"], np.float32)
        if idx.size:
            xyz = low[None, :] + (idx + 0.5) * vs
            self._pub_cloud.publish(
                _xyzi_cloud(header, xyz, np.asarray(rep["sparse_vals"])))

        if "slice" in rep:
            sl = np.asarray(rep["slice"], np.float32)
            nx, ny = sl.shape
            gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            xyz = np.empty((nx * ny, 3), np.float32)
            xyz[:, 0] = low[0] + (gx.ravel() + 0.5) * vs
            xyz[:, 1] = low[1] + (gy.ravel() + 0.5) * vs
            xyz[:, 2] = low[2] + (rep["slice_k"] + 0.5) * vs
            self._pub_slice.publish(_xyzi_cloud(header, xyz, sl.ravel()))

    # ------------------------------------------------------------------ plan
    def _on_goal(self, msg):
        if self._joints is None:
            self.get_logger().error("plan requested but no joint states yet")
            return
        base = self.get_parameter("base_frame").value
        T_goal = _pose_to_mat(msg.pose)
        if msg.header.frame_id and msg.header.frame_id != base:
            try:
                tf = self._tf_buffer.lookup_transform(base, msg.header.frame_id,
                                                      Time())
            except Exception as e:
                self.get_logger().error(
                    f"goal in {msg.header.frame_id}, no TF to {base}: {e}")
                return
            T_goal = _tf_to_mat(tf) @ T_goal
        names, q = self._joints
        self.get_logger().info("planning (first call builds the planner; "
                               "may take ~1 min)...")
        rep = self._rpc({"cmd": "plan", "q": np.array(q, np.float32),
                         "joint_names": names, "T_base_goal": T_goal},
                        timeout_ms=PLAN_TIMEOUT_MS)
        if not rep.get("ok"):
            self.get_logger().error(f"plan rpc: {rep.get('error')}")
            return
        if not rep.get("success"):
            self.get_logger().warn(f"plan failed: {rep.get('error')}")
            return

        dt = float(rep["dt"])
        traj = JointTrajectory()
        traj.header.frame_id = base
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(rep["joint_names"])
        pos = np.asarray(rep["positions"], np.float64)
        vel = np.asarray(rep["velocities"], np.float64)
        for i in range(pos.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = pos[i].tolist()
            pt.velocities = vel[i].tolist()
            t = i * dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
            traj.points.append(pt)
        self._pub_traj.publish(traj)

        path = Path()
        path.header = traj.header
        for p in np.asarray(rep["ee_path"], np.float64):
            ps = PoseStamped()
            ps.header = traj.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self._pub_path.publish(path)
        self.get_logger().info(
            f"plan ok: {pos.shape[0]} pts, {pos.shape[0] * dt:.2f} s, "
            f"pos_err {rep.get('position_error')}, "
            f"rot_err {rep.get('rotation_error')}, "
            f"solve {rep.get('solve_time'):.2f} s")


def main():
    rclpy.init()
    node = CuroboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
