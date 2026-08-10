#!/usr/bin/env python3
"""ROS2 <-> PointSO bridge.

Request/response over topics (no custom .srv needed to get started):

  publish std_msgs/String on /pointso_bridge/request:
      {"instruction": "handle"}
  with the object's segmented cloud arriving on /pointso_bridge/cloud
  (sensor_msgs/PointCloud2 with xyz + rgb fields, object points only).

  Result: geometry_msgs/Vector3Stamped on /pointso_bridge/direction,
  frame_id copied from the cloud (direction is in that frame / the object
  frame, depending on how you feed the cloud).

Swap to a proper service definition once the interface stabilizes.
"""

import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
import zmq
import msgpack
import msgpack_numpy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String

msgpack_numpy.patch()

POINTSO_ADDR = os.environ.get("POINTSO_ADDR", "tcp://127.0.0.1:5668")


def cloud_to_xyzrgb(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> Nx6 float32 (xyz, rgb in [0,1])."""
    pts = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    xyz = pts[:, :3].astype(np.float32)
    # packed float rgb -> 3 channels in [0,1]
    rgb_packed = pts[:, 3].astype(np.float32).view(np.uint32)
    r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_packed >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_packed & 0xFF).astype(np.float32) / 255.0
    return np.concatenate([xyz, np.stack([r, g, b], axis=1)], axis=1)


class PointSOBridge(Node):
    def __init__(self):
        super().__init__("pointso_bridge")
        self.zctx = zmq.Context()
        self.sock = self.zctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 10000)
        self.sock.connect(POINTSO_ADDR)
        self.get_logger().info(f"pointso sidecar: {POINTSO_ADDR}")

        self.latest_cloud = None
        self.create_subscription(PointCloud2, "/pointso_bridge/cloud",
                                 self.on_cloud, 1)
        self.create_subscription(String, "/pointso_bridge/request",
                                 self.on_request, 10)
        self.pub = self.create_publisher(Vector3Stamped,
                                         "/pointso_bridge/direction", 10)

    def on_cloud(self, msg):
        self.latest_cloud = msg

    def on_request(self, msg):
        if self.latest_cloud is None:
            self.get_logger().warn("no cloud received yet")
            return
        instruction = json.loads(msg.data)["instruction"]
        pcd = cloud_to_xyzrgb(self.latest_cloud)

        try:
            self.sock.send(msgpack.packb(
                {"cmd": "orient", "pcd": pcd, "instruction": instruction},
                use_bin_type=True))
            rep = msgpack.unpackb(self.sock.recv(), raw=False)
        except zmq.Again:
            self.sock.close(0)
            self.sock = self.zctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 10000)
            self.sock.connect(POINTSO_ADDR)
            self.get_logger().error("pointso timeout")
            return

        if not rep.get("ok"):
            self.get_logger().error(f"pointso: {rep.get('error')}")
            return

        out = Vector3Stamped()
        out.header = self.latest_cloud.header
        out.vector.x, out.vector.y, out.vector.z = map(float, rep["direction"])
        self.pub.publish(out)
        self.get_logger().info(f"'{instruction}' -> {rep['direction']}")


def main():
    rclpy.init()
    rclpy.spin(PointSOBridge())


if __name__ == "__main__":
    main()