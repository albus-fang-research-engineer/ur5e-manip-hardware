"""Live RGB-D -> sam3 mask -> AnyGrasp sidecar -> rviz2 grippers + PoseArray.

    ros2 run manip_bridge grasp_viz --ros-args -p use_sim_time:=true -- --prompt mug --once

Twin of oriany_viz_node: same sam3 path, then the masked pixels are
back-projected with the colour intrinsics into the camera optical frame,
which is the frame the SDK requires (metres, float32, z forward).

The FULL scene cloud is sent; the mask only sets the sidecar's `lims` box
(3D bbox of the masked points + --margin). The server turns lims into a
region-steering mask, so candidates are confined to the object while
collision_detection still sees the table. --crop sends masked points only,
which lets AnyGrasp propose grasps through the tabletop; it is here for A/B.

Publishes, in rgb.header.frame_id, latched:
  /grasp/<prompt>/poses   PoseArray     top-K grasps, AnyGrasp frame:
                                        x = approach (into object),
                                        y = closing (width), z = normal.
                                        Order = score desc. Base-frame
                                        composition is the consumer's job.
  /grasp/markers          MarkerArray   gripper outlines, green = best score
                                        in this frame, red = 0, zero-stamped
                                        so they survive `bag play --loop`.
  /grasp/<prompt>/cloud   PointCloud2   what was sent (crop or full), for
                                        checking the lims box in rviz2.

rviz2: Fixed Frame = camera_color_optical_frame (or base_link with the bag's
TF); MarkerArray + PoseArray on the topics above with Durability: Transient
Local; the cloud topic with Reliability: Best Effort.

Env:
    SAM3_ADDR     tcp://127.0.0.1:5670
    GRASP_ADDR    tcp://127.0.0.1:5666
"""

import argparse
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .img import camera_info_to_K, image_to_depth_m, image_to_rgb
from .zmq_client import SidecarClient, SidecarError

SAM3_ADDR = os.environ.get("SAM3_ADDR", "tcp://127.0.0.1:5670")
GRASP_ADDR = os.environ.get("GRASP_ADDR", "tcp://127.0.0.1:5666")


def backproject(depth, K, valid):
    """HxW depth (m) + 3x3 K -> Nx3 float32 camera-frame points at `valid`."""
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float32), (v, u)


def gripper_lines(t, R, width, depth, height=0.02):
    """Wireframe in the AnyGrasp grasp frame: U of fingers along +x, tail = approach."""
    w = width / 2
    p = [(-height, -w, 0), (0, -w, 0), (0, -w, 0), (0, w, 0), (0, w, 0), (-height, w, 0),
         (0, -w, 0), (depth, -w, 0), (0, w, 0), (depth, w, 0),
         (-height, 0, 0), (-height - 0.02, 0, 0)]
    return [R @ np.array(q) + t for q in p]


def xyzrgb_to_cloud(pts, cols, header):
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(pts)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
    msg.fields.append(PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1))
    rgb = (cols * 255).astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    buf = np.empty(len(pts), dtype=[("xyz", np.float32, 3), ("rgb", np.uint32)])
    buf["xyz"], buf["rgb"] = pts, packed
    msg.point_step, msg.row_step = 16, 16 * len(pts)
    msg.is_dense = True
    msg.data = buf.tobytes()
    return msg


class GraspViz(Node):
    def __init__(self, a):
        super().__init__("grasp_viz")
        self.a, self.i = a, 0
        self.K, self.depth = None, None
        self.sam3 = SidecarClient(SAM3_ADDR, 60_000)
        self.grasp = SidecarClient(GRASP_ADDR, 120_000, codec="pickle")
        self.get_logger().info(
            f"sam3 {SAM3_ADDR} {'alive' if self.sam3.ping() else 'NOT responding'}; "
            f"grasp {GRASP_ADDR} {'alive' if self.grasp.ping() else 'NOT responding'}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        tag = a.prompt.replace(" ", "_")
        self.pub_poses = self.create_publisher(PoseArray, f"/grasp/{tag}/poses", latched)
        self.pub_mark = self.create_publisher(MarkerArray, "/grasp/markers", latched)
        self.pub_cloud = self.create_publisher(PointCloud2, f"/grasp/{tag}/cloud", latched)
        self.create_subscription(CameraInfo, a.info, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, a.depth, self.on_depth, qos_profile_sensor_data)
        self.sub = self.create_subscription(Image, a.rgb, self.on_rgb, qos_profile_sensor_data)
        self.get_logger().info(f"waiting for {a.rgb} + {a.depth} + {a.info}")

    def on_info(self, msg):
        self.K = camera_info_to_K(msg)

    def on_depth(self, msg):
        self.depth = image_to_depth_m(msg)

    def on_rgb(self, msg):
        self.i += 1
        if self.a.every > 1 and self.i % self.a.every != 1:
            return
        if self.K is None or self.depth is None:
            self.get_logger().warn("rgb before depth/camera_info; skipping")
            return
        rgb, depth, K = image_to_rgb(msg), self.depth, self.K
        if depth.shape != rgb.shape[:2]:
            self.get_logger().error(
                f"depth {depth.shape} != rgb {rgb.shape[:2]}: bag needs aligned depth")
            return

        try:
            rep = self.sam3.call({"cmd": "segment", "rgb": rgb, "prompt": self.a.prompt})
            masks = np.asarray(rep["masks"])
            if not masks.shape[0]:
                self.get_logger().warn(f"sam3: nothing for '{self.a.prompt}'")
                return
            mask = masks[0].astype(bool)
        except (TimeoutError, SidecarError) as e:
            self.get_logger().error(f"sam3 {type(e).__name__}: {e}")
            return

        valid = (depth > 0.05) & (depth < 3.0)
        obj_pts, _ = backproject(depth, K, valid & mask)
        if len(obj_pts) < 50:
            self.get_logger().warn(f"only {len(obj_pts)} valid depth px under mask")
            return
        lo, hi = obj_pts.min(0) - self.a.margin, obj_pts.max(0) + self.a.margin
        lims = [float(x) for x in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])]

        sel = valid & mask if self.a.crop else valid
        pts, (v, u) = backproject(depth, K, sel)
        cols = (rgb[v, u].astype(np.float32) / 255.0)

        try:
            rep = self.grasp.call({"points": pts, "colors": cols, "lims": lims,
                                   "collision_detection": True})
        except (TimeoutError, SidecarError) as e:
            self.get_logger().error(f"grasp {type(e).__name__}: {e}")
            return

        n = min(self.a.top, int(rep["n"]))
        self.get_logger().info(
            f"frame #{self.i} '{self.a.prompt}' mask={int(mask.sum())}px sent={len(pts)} "
            f"lims={np.round(lims, 3).tolist()} -> {rep['n']} grasps"
            + (f", best {rep['scores'][0]:.3f}" if n else " (widen --margin?)"))

        # Zero stamp: under `bag play --loop` sim time wraps and a bag-stamped
        # marker leaves rviz's TF window almost immediately.
        hdr = Header(frame_id=msg.header.frame_id)
        self.pub_cloud.publish(xyzrgb_to_cloud(pts, cols, hdr))

        pa, ma = PoseArray(header=hdr), MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        smax = float(rep["scores"][0]) if n else 1.0
        for k in range(n):
            t = np.asarray(rep["translations"][k], np.float64)
            R = np.asarray(rep["rotations"][k], np.float64).reshape(3, 3)
            q = Rotation.from_matrix(R).as_quat()
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, t)
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = map(float, q)
            pa.poses.append(pose)

            m = Marker()
            m.header = hdr
            m.ns, m.id, m.type, m.action = "grasps", k, Marker.LINE_LIST, Marker.ADD
            m.scale.x = 0.002
            s = float(rep["scores"][k]) / smax
            m.color.r, m.color.g, m.color.a = 1.0 - s, s, 1.0
            for p in gripper_lines(t, R, float(rep["widths"][k]), float(rep["depths"][k])):
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            ma.markers.append(m)
        self.pub_poses.publish(pa)
        self.pub_mark.publish(ma)

        if self.a.once:
            self.destroy_subscription(self.sub)
            self.get_logger().info("latched; leave running for rviz2, Ctrl-C to exit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="mug")
    p.add_argument("--rgb", default=os.environ.get("RGB_TOPIC", "/camera/camera/color/image_raw"))
    p.add_argument("--depth", default=os.environ.get(
        "DEPTH_TOPIC", "/camera/camera/aligned_depth_to_color/image_raw"))
    p.add_argument("--info", default=os.environ.get("INFO_TOPIC", "/camera/camera/color/camera_info"))
    p.add_argument("--every", type=int, default=30, help="run every Nth rgb frame")
    p.add_argument("--once", action="store_true", help="one frame, then latch")
    p.add_argument("--top", type=int, default=20, help="grasps to publish/draw")
    p.add_argument("--margin", type=float, default=0.03, help="lims padding around mask bbox, m")
    p.add_argument("--crop", action="store_true", help="send masked points only (A/B)")
    a = p.parse_args(rclpy.utilities.remove_ros_args()[1:])
    rclpy.init()
    try:
        rclpy.spin(GraspViz(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
