"""Common shape for the two pose sidecars (FoundationPose, Any6D).

    <ns>/estimate   srv EstimatePose   one-shot registration (slow, seconds..min)
    <ns>/release    srv Release        drop a session
    <ns>/<obj>/pose PoseStamped        cam_T_obj per synced camera frame
    TF              camera_optical     -> <obj>

After a successful estimate the object is added to the tracked set and the
streaming path sends every synced RGB-D frame as a "track" request.

Concurrency model (matters -- the sidecars are single-threaded REP loops):
  * two callback groups, MultiThreadedExecutor: a minutes-long estimate
    must not block the executor thread the subscriber lives on;
  * two ZMQ sockets (REQ sockets strictly alternate send/recv);
  * while an estimate is in flight the tracker skips frames rather than
    queueing behind it on the server side and timing out;
  * the tracker is non-reentrant (drop-if-busy), so a slow track never
    builds a backlog of stale frames.

Parameters:
    rgb_topic, depth_topic, info_topic   camera topics (match `ros2 bag info`)
    sync_slop                            s, ApproximateTimeSynchronizer
    track_refine_iter                    per-frame refiner iterations
    publish_tf                           bool
"""

import threading

import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from manip_interfaces.srv import EstimatePose, Release

from .img import (camera_info_to_K, image_to_depth_m, image_to_mono,
                  image_to_rgb, T_to_pose, T_to_tf)
from .zmq_client import SidecarClient, SidecarError


class TrackerBridge(Node):
    """Subclasses set: NS, ADDR, EST_TIMEOUT_MS, TRACK_TIMEOUT_MS and
    implement estimate_payload()/parse_estimate()."""

    NS = "pose"
    ADDR = "tcp://127.0.0.1:5667"
    EST_TIMEOUT_MS = 300_000
    TRACK_TIMEOUT_MS = 10_000

    def __init__(self, node_name):
        super().__init__(node_name)
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("sync_slop", 0.034)
        self.declare_parameter("track_refine_iter", 2)
        self.declare_parameter("publish_tf", True)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.est_client = SidecarClient(self.ADDR, self.EST_TIMEOUT_MS)
        self.track_client = SidecarClient(self.ADDR, self.TRACK_TIMEOUT_MS)

        self.K = None
        self.tracked: dict[str, object] = {}    # obj -> PoseStamped publisher
        self._state_lock = threading.Lock()
        self._estimating = threading.Event()
        self._track_busy = threading.Lock()

        self.cbg_srv = MutuallyExclusiveCallbackGroup()
        self.cbg_stream = MutuallyExclusiveCallbackGroup()

        self.create_service(EstimatePose, f"{self.NS}/estimate", self.on_estimate,
                            callback_group=self.cbg_srv)
        self.create_service(Release, f"{self.NS}/release", self.on_release,
                            callback_group=self.cbg_srv)

        qos = qos_profile_sensor_data
        self.sub_rgb = Subscriber(self, Image, gp("rgb_topic"), qos_profile=qos,
                                  callback_group=self.cbg_stream)
        self.sub_depth = Subscriber(self, Image, gp("depth_topic"), qos_profile=qos,
                                    callback_group=self.cbg_stream)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=2, slop=float(gp("sync_slop")))
        self.sync.registerCallback(self.on_frame)
        self.create_subscription(CameraInfo, gp("info_topic"), self.on_info, qos,
                                 callback_group=self.cbg_stream)
        self.tf_bcast = TransformBroadcaster(self)

        up = self.est_client.ping()
        self.get_logger().info(
            f"{self.NS} bridge up -> {self.ADDR} "
            f"({'sidecar alive' if up else 'sidecar NOT responding'}); "
            f"rgb={gp('rgb_topic')} depth={gp('depth_topic')}")

    # ---- to be specialised -------------------------------------------------
    def estimate_payload(self, req, rgb, depth, K, mask) -> dict:
        raise NotImplementedError

    def parse_estimate(self, rep, res):
        """Fill extra response fields from the sidecar reply."""

    # ---- services ----------------------------------------------------------
    def on_estimate(self, req, res):
        if not req.obj:
            res.success, res.message = False, "obj must be non-empty"
            return res
        try:
            rgb = image_to_rgb(req.rgb)
            depth = image_to_depth_m(req.depth)
            K = camera_info_to_K(req.camera_info)
            mask = (image_to_mono(req.mask) > 0).astype(np.uint8)
            if mask.sum() == 0:
                raise ValueError("mask is empty")
            payload = self.estimate_payload(req, rgb, depth, K, mask)
        except ValueError as e:
            res.success, res.message = False, f"bad request: {e}"
            return res

        self.get_logger().info(f"estimate '{req.obj}' ...")
        self._estimating.set()
        try:
            rep = self.est_client.call(payload)
        except (TimeoutError, SidecarError) as e:
            res.success, res.message = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(res.message)
            return res
        finally:
            self._estimating.clear()

        T = np.asarray(rep["pose"], np.float64)
        res.pose = T_to_pose(T, req.rgb.header)
        self.parse_estimate(rep, res)

        with self._state_lock:
            if req.obj not in self.tracked:
                self.tracked[req.obj] = self.create_publisher(
                    PoseStamped, f"{self.NS}/{req.obj}/pose", 10)
        self.publish(req.obj, T, req.rgb.header)
        res.success = True
        res.message = res.message or f"registered '{req.obj}', tracking"
        self.get_logger().info(f"{res.message}  t={T[:3, 3].round(3).tolist()}")
        return res

    def on_release(self, req, res):
        with self._state_lock:
            pub = self.tracked.pop(req.obj, None)
        if pub is not None:
            self.destroy_publisher(pub)
        try:
            self.est_client.call({"cmd": "release", "obj": req.obj}, timeout_ms=5000)
        except (TimeoutError, SidecarError) as e:
            res.success, res.message = False, f"{type(e).__name__}: {e}"
            return res
        res.success, res.message = True, f"released '{req.obj}'"
        return res

    # ---- streaming ---------------------------------------------------------
    def on_info(self, msg):
        if self.K is None:
            self.get_logger().info(f"camera_info: {msg.width}x{msg.height}")
        self.K = camera_info_to_K(msg)

    def on_frame(self, rgb_msg, depth_msg):
        with self._state_lock:
            objs = list(self.tracked)
        if not objs or self.K is None or self._estimating.is_set():
            return
        if not self._track_busy.acquire(blocking=False):
            return  # previous track still running: drop this frame
        try:
            rgb = image_to_rgb(rgb_msg)
            depth = image_to_depth_m(depth_msg)
            iters = int(self.get_parameter("track_refine_iter").value)
            for obj in objs:
                try:
                    rep = self.track_client.call(
                        {"cmd": "track", "obj": obj, "rgb": rgb, "depth": depth,
                         "K": self.K, "track_refine_iter": iters})
                except (TimeoutError, SidecarError) as e:
                    self.get_logger().warn(f"track {obj}: {e}", throttle_duration_sec=2.0)
                    continue
                self.publish(obj, np.asarray(rep["pose"], np.float64), rgb_msg.header)
        finally:
            self._track_busy.release()

    def publish(self, obj, T, header):
        with self._state_lock:
            pub = self.tracked.get(obj)
        if pub is None:
            return
        pub.publish(T_to_pose(T, header))
        if self.get_parameter("publish_tf").value:
            self.tf_bcast.sendTransform(T_to_tf(T, header, f"{self.NS}_{obj}"))
