#!/usr/bin/env python3
"""Live RGB-D -> sam3 -> Orient Anything V2 -> rviz2 axes + TF.

Smoke test for the oriany sidecar against a bag; the ROS twin of
test/oriany_bag_demo.py. Runs inside Ros2Bridge (no colcon; this dir is
mounted loose). Talks msgpack-ZMQ to sam3 (:5670) and oriany (:5673)
directly, same crop as oriany_bridge_node.square_crop.

  # terminal 1
  ros2 bag play /bags/mug --clock --loop
  # terminal 2
  python3 src/ros2_bridge/oriany_bag_viz.py --ros-args -p use_sim_time:=true -- --prompt mug
  python3 src/ros2_bridge/oriany_bag_viz.py --ros-args -p use_sim_time:=true -- --once
  # terminal 3
  rviz2   # Fixed Frame: the cloud/rgb frame_id (camera_color_optical_frame)
          # add PointCloud2 on the cloud topic, MarkerArray on /oriany/axes,
          # TF and tick oriany_<prompt>

Publishes, in rgb.header.frame_id (OpenCV optical: x right, y down, z fwd):
  /oriany/axes   MarkerArray   front red / lateral green / up blue arrows
                               anchored at the mask's 3D centroid, plus a
                               text marker with az/el/ro/alpha
  TF             oriany_<prompt>   R_cam (x=front, y=lateral, z=up) at that
                               centroid; static when --once so it survives
                               bag loops, else re-broadcast per estimate

alpha == 0 -> front/lateral are not committed predictions (arrows drawn
half-length so it is visible in rviz); judge only up on that frame.
"""
import argparse, os

import msgpack, msgpack_numpy
import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

msgpack_numpy.patch()

SAM3_ADDR = os.environ.get("SAM3_ADDR", "tcp://127.0.0.1:5670")
ORIANY_ADDR = os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673")


def zmq_client(addr, timeout_ms):
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(addr)
    return s


def call(sock, payload):
    sock.send(msgpack.packb(payload, use_bin_type=True))
    return msgpack.unpackb(sock.recv(), raw=False)


def image_to_rgb(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step // 3, 3)[:, :w, :]
        return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    raise ValueError(f"unsupported rgb encoding {msg.encoding}")


def image_to_depth_m(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    if msg.encoding == "16UC1":
        return buf.reshape(h, msg.step)[:, :w * 2].copy().view(np.uint16).reshape(h, w) / 1000.0
    if msg.encoding == "32FC1":
        return buf.reshape(h, msg.step)[:, :w * 4].copy().view(np.float32).reshape(h, w).astype(np.float64)
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def square_crop(rgb, mask, fg_ratio, bg_fill=255):
    """Same as oriany_bridge_node.square_crop."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    fg = rgb[y0:y1, x0:x1].copy()
    fg[~mask[y0:y1, x0:x1]] = bg_fill
    h, w = fg.shape[:2]
    side = max(int(round(max(h, w) / max(fg_ratio, 1e-3))), max(h, w))
    out = np.full((side, side, 3), bg_fill, np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy:oy + h, ox:ox + w] = fg
    return out


def mat_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


class OrianyViz(Node):
    def __init__(self, a):
        super().__init__("oriany_bag_viz")
        self.a, self.i = a, 0
        self.K, self.depth = None, None
        self.sam3 = zmq_client(SAM3_ADDR, 60_000)
        self.oriany = zmq_client(ORIANY_ADDR, 120_000)
        self.get_logger().info(f"sam3 {SAM3_ADDR}: {call(self.sam3, {'cmd': 'ping'})}  "
                               f"oriany {ORIANY_ADDR}: {call(self.oriany, {'cmd': 'ping'})}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, "/oriany/axes", latched)
        self.tf = StaticTransformBroadcaster(self) if a.once else TransformBroadcaster(self)
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, a.info, self.on_info, sensor)
        self.create_subscription(Image, a.depth, self.on_depth, sensor)
        self.sub = self.create_subscription(Image, a.rgb, self.on_rgb, sensor)
        self.get_logger().info(f"waiting for {a.rgb} + {a.depth} + {a.info}")

    def on_info(self, msg):
        self.K = np.asarray(msg.k, np.float64).reshape(3, 3)

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
            self.get_logger().error(f"depth {depth.shape} != rgb {rgb.shape[:2]}: bag needs aligned depth")
            return

        rep = call(self.sam3, {"cmd": "segment", "rgb": rgb, "prompt": self.a.prompt})
        masks = np.asarray(rep["masks"]) if rep.get("ok") else np.zeros((0,))
        if not masks.shape[0]:
            self.get_logger().warn(f"sam3: nothing for '{self.a.prompt}'")
            return
        mask = masks[0].astype(bool)

        rep = call(self.oriany, {"cmd": "orient",
                                 "image": square_crop(rgb, mask, self.a.fg_ratio),
                                 "remove_bkg": False})
        if not rep.get("ok"):
            self.get_logger().error(f"oriany: {rep.get('error')}")
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
        self.get_logger().info(
            f"frame #{self.i} az={rep['azimuth']:.0f} el={rep['elevation']:.0f} "
            f"ro={rep['rotation']:.0f} alpha={alpha} centroid={np.round(P, 3)}")

        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        axes = [("front", R[:, 0], (1.0, 0.0, 0.0), alpha != 0),
                ("lateral", R[:, 1], (0.0, 0.8, 0.0), alpha != 0),
                ("up", R[:, 2], (0.0, 0.5, 1.0), True)]
        for k, (name, d, col, committed) in enumerate(axes):
            m = Marker()
            m.header = msg.header
            m.ns, m.id, m.type, m.action = "oriany", k, Marker.ARROW, Marker.ADD
            L = self.a.length * (1.0 if committed else 0.5)
            m.points = [Point(x=float(P[0]), y=float(P[1]), z=float(P[2])),
                        Point(x=float(P[0] + L * d[0]), y=float(P[1] + L * d[1]), z=float(P[2] + L * d[2]))]
            m.scale.x, m.scale.y, m.scale.z = 0.004, 0.012, 0.015
            m.color.r, m.color.g, m.color.b, m.color.a = *col, 1.0
            ma.markers.append(m)
        t = Marker()
        t.header = msg.header
        t.ns, t.id, t.type, t.action = "oriany", 3, Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = float(P[0]), float(P[1] - 0.08), float(P[2])
        t.scale.z = 0.02
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = f"{self.a.prompt} az={rep['azimuth']:.0f} el={rep['elevation']:.0f} ro={rep['rotation']:.0f} alpha={alpha}"
        ma.markers.append(t)
        self.pub.publish(ma)

        tfm = TransformStamped()
        tfm.header = msg.header
        tfm.child_frame_id = f"oriany_{self.a.prompt.replace(' ', '_')}"
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, P)
        (tfm.transform.rotation.x, tfm.transform.rotation.y,
         tfm.transform.rotation.z, tfm.transform.rotation.w) = map(float, mat_to_quat(R))
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
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--once", action="store_true")
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
