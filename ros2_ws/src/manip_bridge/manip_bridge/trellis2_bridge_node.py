"""ROS2 Humble <-> TRELLIS.2 ZMQ bridge.

Exposes:  /trellis2/generate_mesh   (manip_interfaces/srv/GenerateMesh)

Forwards to the trellis2 sidecar over ZMQ REQ. If the request carries depth
+ camera_info, the sidecar also runs similarity registration and the reply
includes a metric mesh + T_cam_obj (see GenerateMesh.srv).

Generation is SLOW for a ROS service (tens of seconds to minutes on a
3090). The handler blocks up to TRELLIS_TIMEOUT_S; keep it off the control
path and call from a dedicated client / its own callback group.

Env:
    TRELLIS_ADDR        tcp://127.0.0.1:5669 (host-net default)
    TRELLIS_TIMEOUT_S   240
"""

import os

import numpy as np
import rclpy
import zmq
import msgpack
import msgpack_numpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle

from manip_interfaces.srv import GenerateMesh

msgpack_numpy.patch()

TRELLIS_ADDR = os.environ.get("TRELLIS_ADDR", "tcp://127.0.0.1:5669")
TIMEOUT_MS = int(float(os.environ.get("TRELLIS_TIMEOUT_S", 240)) * 1000)


def _image_to_rgb(msg) -> np.ndarray:
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step // 3, 3)[:, :w, :]
        return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        img = buf.reshape(h, msg.step // 4, 4)[:, :w, :3]
        return img[..., ::-1].copy() if msg.encoding == "bgra8" else img.copy()
    raise ValueError(f"unsupported rgb encoding: {msg.encoding}")


def _image_to_mono(msg) -> np.ndarray:
    if msg.encoding != "mono8":
        raise ValueError(f"mask must be mono8, got: {msg.encoding}")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return buf.reshape(msg.height, msg.step)[:, : msg.width].copy()


def _image_to_depth_m(msg) -> np.ndarray:
    """16UC1 mm or 32FC1 m -> HxW float32 meters."""
    h, w = msg.height, msg.width
    if msg.encoding == "16UC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        return d.reshape(h, msg.step // 2)[:, :w].astype(np.float32) * 1e-3
    if msg.encoding == "32FC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.float32)
        return d.reshape(h, msg.step // 4)[:, :w].copy()
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


class Trellis2Bridge(Node):
    def __init__(self):
        super().__init__("trellis2_bridge")
        self._zctx = zmq.Context()
        self._sock = None
        self._connect()
        self._cbg = MutuallyExclusiveCallbackGroup()
        self.create_service(
            GenerateMesh, "trellis2/generate_mesh", self._on_generate,
            callback_group=self._cbg,
        )
        self.get_logger().info(f"trellis2 bridge up -> {TRELLIS_ADDR}")

    def _connect(self):
        # REQ sockets wedge after a timeout; rebuild instead of reusing.
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._zctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
        self._sock.setsockopt(zmq.SNDTIMEO, 5000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(TRELLIS_ADDR)

    def _on_generate(self, req, res):
        try:
            payload = {
                "op": "generate",
                "rgb": _image_to_rgb(req.rgb),
                "seed": int(req.seed),
                "decimation_target": int(req.decimation_target),
                "texture_size": int(req.texture_size),
                "output_name": req.output_name or None,
                "return_glb": False,
            }
            if req.mask.height > 0:
                payload["mask"] = _image_to_mono(req.mask)
            if req.depth.height > 0:
                payload["depth"] = _image_to_depth_m(req.depth)
                payload["K"] = np.asarray(req.camera_info.k,
                                          dtype=np.float64).reshape(3, 3)

            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            reply = msgpack.unpackb(self._sock.recv(), raw=False)
        except zmq.error.Again:
            self._connect()
            res.success = False
            res.message = f"trellis2 sidecar timeout after {TIMEOUT_MS} ms"
            return res
        except Exception as e:  # noqa: BLE001
            self._connect()
            res.success = False
            res.message = f"{type(e).__name__}: {e}"
            return res

        if not reply.get("ok", False):
            res.success = False
            res.message = reply.get("error", "unknown sidecar error")
            return res

        verts = np.asarray(reply["vertices"], dtype=np.float64)
        faces = np.asarray(reply["faces"], dtype=np.uint32)
        res.mesh = Mesh(
            vertices=[Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
                      for v in verts],
            triangles=[MeshTriangle(vertex_indices=f.tolist()) for f in faces],
        )
        res.glb_path = reply["glb_path"]
        res.gen_time = float(reply["gen_time"])

        metric = reply.get("metric")
        res.metric_valid = bool(metric)
        if metric:
            res.metric_glb_path = metric["glb_path"]
            res.scale = float(metric["scale"])
            res.registration_rmse = float(metric["rmse"])
            res.object_pose.header = req.rgb.header  # camera frame + stamp
            p, q = metric["t"], metric["q_xyzw"]
            res.object_pose.pose.position.x = float(p[0])
            res.object_pose.pose.position.y = float(p[1])
            res.object_pose.pose.position.z = float(p[2])
            res.object_pose.pose.orientation.x = float(q[0])
            res.object_pose.pose.orientation.y = float(q[1])
            res.object_pose.pose.orientation.z = float(q[2])
            res.object_pose.pose.orientation.w = float(q[3])

        res.success = True
        res.message = (
            f"{len(verts)} verts / {len(faces)} faces in {res.gen_time:.1f}s"
            + (f"; scale={res.scale:.4f} rmse={res.registration_rmse * 1e3:.1f}mm"
               if res.metric_valid else " (canonical only)")
        )
        return res


def main():
    rclpy.init()
    node = Trellis2Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()