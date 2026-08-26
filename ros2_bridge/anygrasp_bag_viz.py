#!/usr/bin/env python3
"""Live PointCloud2 -> AnyGrasp sidecar -> rviz2 MarkerArray.

Smoke test for the grasp sidecar against a bag. Runs inside Ros2Bridge
(no colcon; this dir is mounted loose). Speaks pickle to :5666 directly --
the grasp server predates the msgpack contract every other sidecar uses.

  # terminal 1
  ros2 bag play /bags/mug --clock --loop
  # terminal 2
  python3 src/ros2_bridge/anygrasp_bag_viz.py                  # every 10th cloud
  python3 src/ros2_bridge/anygrasp_bag_viz.py --once           # first cloud, then latch
  # terminal 3
  rviz2   # Fixed Frame: base_link (tf from bag) or the cloud's frame_id
          # add PointCloud2 on the cloud topic, MarkerArray on /anygrasp/grasps

The cloud must be in a *_optical_frame (z forward, y down): that is what the
SDK's approach prior assumes. realsense-ros publishes depth/color/points in
camera_color_optical_frame when aligned, which is what the mug bag has.
"""
import argparse, os, pickle

import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

GRASP_ADDR = os.environ.get("GRASP_ADDR", "tcp://127.0.0.1:5666")


def cloud_to_xyzrgb(msg):
    off = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    f32 = lambda o: buf[:, o:o + 4].copy().view(np.float32).ravel()
    xyz = np.stack([f32(off["x"]), f32(off["y"]), f32(off["z"])], 1)
    if "rgb" in off:
        u = buf[:, off["rgb"]:off["rgb"] + 4].copy().view(np.uint32).ravel()
        rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    ok = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    return xyz[ok].astype(np.float32), rgb[ok].astype(np.float32)


# AnyGrasp grasp frame: x = approach (into object), y = closing, z = normal.
def gripper_lines(t, R, width, depth, height=0.02):
    w = width / 2
    p = [(-height, -w, 0), (0, -w, 0), (0, -w, 0), (0, w, 0), (0, w, 0), (-height, w, 0),
         (0, -w, 0), (depth, -w, 0), (0, w, 0), (depth, w, 0),
         (-height, 0, 0), (-height - 0.02, 0, 0)]
    return [R @ np.array(q) + t for q in p]


class GraspViz(Node):
    def __init__(self, a):
        super().__init__("anygrasp_bag_viz")
        self.a, self.i = a, 0
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(GRASP_ADDR)
        self.sock.send(pickle.dumps({"cmd": "ping"}))
        self.get_logger().info(f"grasp sidecar {GRASP_ADDR}: {pickle.loads(self.sock.recv())}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, "/anygrasp/grasps", latched)
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(PointCloud2, a.topic, self.on_cloud, sensor)
        self.get_logger().info(f"waiting for {a.topic}")

    def on_cloud(self, msg):
        self.i += 1
        if self.i % self.a.every != 1 and self.a.every > 1:
            return
        pts, cols = cloud_to_xyzrgb(msg)
        self.sock.send(pickle.dumps({"points": pts, "colors": cols, "lims": self.a.lims,
                                     "collision_detection": True}))
        rep = pickle.loads(self.sock.recv())
        if not rep.get("ok"):
            self.get_logger().error(f"sidecar: {rep.get('error')}")
            return
        n = min(self.a.top, rep["n"])
        self.get_logger().info(f"cloud #{self.i} {msg.header.frame_id} n={len(pts)} -> "
                               f"{rep['n']} grasps, best {rep['scores'][0]:.3f}" if n else
                               f"cloud #{self.i} n={len(pts)} -> 0 grasps (check --lims)")
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        smax = float(rep["scores"][0]) if n else 1.0
        for k in range(n):
            m = Marker()
            m.header = msg.header
            m.ns, m.id, m.type, m.action = "grasps", k, Marker.LINE_LIST, Marker.ADD
            m.scale.x = 0.002
            s = float(rep["scores"][k]) / smax
            m.color.r, m.color.g, m.color.a = 1.0 - s, s, 1.0
            for q in gripper_lines(rep["translations"][k], rep["rotations"][k],
                                   float(rep["widths"][k]), float(rep["depths"][k])):
                m.points.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
            ma.markers.append(m)
        self.pub.publish(ma)
        if self.a.once:
            self.destroy_subscription(self.sub)
            self.get_logger().info("latched; leave running for rviz2, Ctrl-C to exit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default=os.environ.get("CLOUD_TOPIC", "/camera/camera/depth/color/points"))
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--once", action="store_true")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--lims", type=float, nargs=6, default=[-0.5, 0.5, -0.5, 0.5, 0.2, 1.2],
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    a = p.parse_args()
    rclpy.init()
    try:
        rclpy.spin(GraspViz(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
