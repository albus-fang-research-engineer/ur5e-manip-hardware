#!/usr/bin/env python3
"""ROS2 <-> FoundationPose bridge.

Subscribes to synced RGB + aligned depth + camera_info, talks ZMQ to the
pose sidecar, publishes PoseStamped and a TF frame per tracked object.

Flow:
  1. Publish a std_msgs/String on /pose_bridge/register with JSON:
         {"obj": "mustard", "mesh": "mustard.obj"}
     The next frame that also has a mask on /pose_bridge/mask (mono8,
     nonzero = object) triggers FoundationPose registration.
  2. After registration, every synced frame is sent as a "track" request
     and the result is published on /pose_bridge/<obj>/pose and TF
     (camera_frame -> <obj>).

This is deliberately a skeleton: mask acquisition (SAM/Florence, or your
sim's ground-truth instance masks) is your pipeline's job. Promote to a
proper colcon package with a .srv for registration once the shape settles.
"""

import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
import zmq
import msgpack
import msgpack_numpy

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation  # apt: python3-scipy

msgpack_numpy.patch()

POSE_ADDR = os.environ.get("POSE_ADDR", "tcp://127.0.0.1:5667")


class PoseBridge(Node):
    def __init__(self):
        super().__init__("foundationpose_bridge")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_scale", 0.001)  # 16UC1 mm -> m

        self.bridge = CvBridge()
        self.zctx = zmq.Context()
        self.sock = self.zctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 10000)
        self.sock.connect(POSE_ADDR)
        self.get_logger().info(f"pose sidecar: {POSE_ADDR}")

        self.K = None
        self.pending_registration = None   # {"obj":..., "mesh":...}
        self.latest_mask = None
        self.tracked = {}                  # obj -> publisher

        gp = lambda n: self.get_parameter(n).value
        self.sub_rgb = Subscriber(self, Image, gp("rgb_topic"))
        self.sub_depth = Subscriber(self, Image, gp("depth_topic"))
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.on_frame)

        self.create_subscription(CameraInfo, gp("info_topic"), self.on_info, 1)
        self.create_subscription(String, "/pose_bridge/register",
                                 self.on_register_cmd, 1)
        self.create_subscription(Image, "/pose_bridge/mask", self.on_mask, 1)
        self.tf_bcast = TransformBroadcaster(self)

    # -- callbacks -------------------------------------------------------

    def on_info(self, msg):
        self.K = np.array(msg.k, np.float32).reshape(3, 3)

    def on_register_cmd(self, msg):
        self.pending_registration = json.loads(msg.data)
        self.get_logger().info(f"registration armed: {self.pending_registration}")

    def on_mask(self, msg):
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def on_frame(self, rgb_msg, depth_msg):
        if self.K is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * \
                self.get_parameter("depth_scale").value
        depth = np.nan_to_num(depth.astype(np.float32))

        # one-shot registration
        if self.pending_registration is not None and self.latest_mask is not None:
            reg = self.pending_registration
            rep = self.rpc({"cmd": "register", "obj": reg["obj"],
                            "mesh": reg["mesh"], "rgb": rgb, "depth": depth,
                            "K": self.K, "mask": self.latest_mask})
            if rep.get("ok"):
                self.tracked[reg["obj"]] = self.create_publisher(
                    PoseStamped, f"/pose_bridge/{reg['obj']}/pose", 10)
                self.publish_pose(reg["obj"], rep["pose"], rgb_msg.header)
                self.get_logger().info(f"registered {reg['obj']}")
            else:
                self.get_logger().error(f"register failed: {rep.get('error')}")
            self.pending_registration = None
            return

        # per-frame tracking
        for obj in list(self.tracked):
            rep = self.rpc({"cmd": "track", "obj": obj, "rgb": rgb,
                            "depth": depth, "K": self.K})
            if rep.get("ok"):
                self.publish_pose(obj, rep["pose"], rgb_msg.header)
            else:
                self.get_logger().warn(f"track {obj}: {rep.get('error')}")

    # -- helpers ---------------------------------------------------------

    def rpc(self, req):
        try:
            self.sock.send(msgpack.packb(req, use_bin_type=True))
            return msgpack.unpackb(self.sock.recv(), raw=False)
        except zmq.Again:
            # REQ socket is now stuck; recreate it
            self.sock.close(0)
            self.sock = self.zctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 10000)
            self.sock.connect(POSE_ADDR)
            return {"ok": False, "error": "timeout"}

    def publish_pose(self, obj, T, header):
        T = np.asarray(T)
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw

        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = \
            map(float, T[:3, 3])
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
        self.tracked[obj].publish(msg)

        tf = TransformStamped()
        tf.header = header
        tf.child_frame_id = obj
        tf.transform.translation.x, tf.transform.translation.y, \
            tf.transform.translation.z = map(float, T[:3, 3])
        (tf.transform.rotation.x, tf.transform.rotation.y,
         tf.transform.rotation.z, tf.transform.rotation.w) = map(float, q)
        self.tf_bcast.sendTransform(tf)


def main():
    rclpy.init()
    rclpy.spin(PoseBridge())


if __name__ == "__main__":
    main()