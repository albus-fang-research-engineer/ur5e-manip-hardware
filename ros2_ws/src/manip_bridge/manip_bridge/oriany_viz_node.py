"""Live RGB-D -> sam3 -> Orient Anything V2 -> rviz2 axes + TF.

    ros2 run manip_bridge oriany_viz --ros-args -p use_sim_time:=true -- --prompt mug --once

ROS twin of test/oriany_bag_demo.py: same sam3 -> square_crop -> oriany
path, but publishes instead of drawing. Talks to the sidecars directly
(same SidecarClient the bridge nodes use) so bridges.launch.py need not be
running.

Publishes, in rgb.header.frame_id (OpenCV optical: x right, y down, z fwd):
  /oriany/axes   MarkerArray   front red / lateral green / up blue arrows
                               from the mask's 3D centroid, plus a text
                               marker with az/el/ro/alpha
  TF             oriany_<prompt>   R_cam (x=front, y=lateral, z=up) at that
                               centroid; static under --once so it survives
                               bag loops, else re-broadcast per estimate

alpha == 0 -> front/lateral are not committed predictions; drawn at half
length so the state is visible in rviz2, not just in the log.

The centroid is the mask's pixel centroid at median mask depth, i.e. a point
on the visible surface, not the body centre. Good for judging directions;
not a body frame.

rviz2: Fixed Frame = camera_color_optical_frame; add PointCloud2 on the
cloud topic (Reliability: Best Effort), MarkerArray on /oriany/axes
(Durability: Transient Local -- the publisher is latched and rviz's default
volatile subscriber never sees a --once message published before it
subscribed), TF with oriany_<prompt> ticked.

Env:
    SAM3_ADDR     tcp://127.0.0.1:5670
    ORIANY_ADDR   tcp://127.0.0.1:5673
"""

import argparse
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .img import camera_info_to_K, image_to_depth_m, image_to_rgb
from .oriany_bridge_node import square_crop
from .zmq_client import SidecarClient, SidecarError

SAM3_ADDR = os.environ.get("SAM3_ADDR", "tcp://127.0.0.1:5670")
ORIANY_ADDR = os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673")


class OrianyViz(Node):
    def __init__(self, a):
        super().__init__("oriany_viz")
        self.a, self.i = a, 0
        self.K, self.depth = None, None
        self.sam3 = SidecarClient(SAM3_ADDR, 60_000)
        self.oriany = SidecarClient(ORIANY_ADDR, 120_000)
        self.get_logger().info(
            f"sam3 {SAM3_ADDR} {'alive' if self.sam3.ping() else 'NOT responding'}; "
            f"oriany {ORIANY_ADDR} {'alive' if self.oriany.ping() else 'NOT responding'}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, "/oriany/axes", latched)
        self.tf = StaticTransformBroadcaster(self) if a.once else TransformBroadcaster(self)
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
            img, _ = square_crop(rgb, mask, self.a.fg_ratio, 255)
            rep = self.oriany.call({"cmd": "orient", "image": img, "remove_bkg": False})
        except (TimeoutError, SidecarError) as e:
            self.get_logger().error(f"{type(e).__name__}: {e}")
            return

        z = depth[mask]
        z = z[(z > 0.05) & (z < 3.0)]
        if z.size == 0:
            self.get_logger().warn("no valid depth under mask")
            return
        zc = float(np.median(z))
        ys, xs = np.nonzero(mask)
        uc, vc = xs.mean(), ys.mean()
        P = np.array([(uc - K[0, 2]) * zc / K[0, 0], (vc - K[1, 2]) * zc / K[1, 1], zc])
        R = np.asarray(rep["R_cam"], np.float64).reshape(3, 3)
        alpha = int(rep["alpha"])
        label = (f"{self.a.prompt} az={rep['azimuth']:.0f} el={rep['elevation']:.0f} "
                 f"ro={rep['rotation']:.0f} alpha={alpha}")
        self.get_logger().info(f"frame #{self.i} {label} centroid={np.round(P, 3)}")

        # Zero stamp = "latest transform": under `bag play --loop` sim time
        # wraps every few seconds and a bag-stamped marker falls out of
        # rviz's TF window almost immediately. The TF below keeps the real
        # stamp (a static TF ignores it anyway).
        hdr = Header(frame_id=msg.header.frame_id)
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        axes = [("front", R[:, 0], (1.0, 0.0, 0.0), alpha != 0),
                ("lateral", R[:, 1], (0.0, 0.8, 0.0), alpha != 0),
                ("up", R[:, 2], (0.0, 0.5, 1.0), True)]
        for k, (_, d, col, committed) in enumerate(axes):
            m = Marker()
            m.header = hdr
            m.ns, m.id, m.type, m.action = "oriany", k, Marker.ARROW, Marker.ADD
            L = self.a.length * (1.0 if committed else 0.5)
            q = P + L * d
            m.points = [Point(x=float(P[0]), y=float(P[1]), z=float(P[2])),
                        Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))]
            m.scale.x, m.scale.y, m.scale.z = 0.004, 0.012, 0.015
            m.color.r, m.color.g, m.color.b, m.color.a = *col, 1.0
            ma.markers.append(m)
        t = Marker()
        t.header = hdr
        t.ns, t.id, t.type, t.action = "oriany", 3, Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = (
            float(P[0]), float(P[1] - 0.08), float(P[2]))
        t.scale.z = 0.02
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = label
        ma.markers.append(t)
        self.pub.publish(ma)

        tfm = TransformStamped()
        tfm.header = msg.header
        tfm.child_frame_id = f"oriany_{self.a.prompt.replace(' ', '_')}"
        (tfm.transform.translation.x, tfm.transform.translation.y,
         tfm.transform.translation.z) = map(float, P)
        (tfm.transform.rotation.x, tfm.transform.rotation.y,
         tfm.transform.rotation.z, tfm.transform.rotation.w) = map(
            float, Rotation.from_matrix(R).as_quat())
        self.tf.sendTransform(tfm)

        if self.a.once:
            self.destroy_subscription(self.sub)
            self.get_logger().info("latched; leave running for rviz2, Ctrl-C to exit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="mug")
    p.add_argument("--rgb", default="/camera/camera/color/image_raw")
    p.add_argument("--depth", default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--info", default="/camera/camera/color/camera_info")
    p.add_argument("--every", type=int, default=30, help="estimate every Nth rgb frame")
    p.add_argument("--once", action="store_true", help="one estimate, latch, static TF")
    p.add_argument("--length", type=float, default=0.06, help="arrow length, m")
    p.add_argument("--fg-ratio", type=float, default=0.85)
    a = p.parse_args(rclpy.utilities.remove_ros_args()[1:])
    rclpy.init()
    try:
        rclpy.spin(OrianyViz(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
