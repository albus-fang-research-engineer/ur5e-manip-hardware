"""ROS2 Humble <-> TRELLIS.2 ZMQ bridge.

Exposes:  /trellis2/generate_mesh   (manip_interfaces/srv/GenerateMesh)

The reply's metric GLB lands on the shared ./trellis2_runtime/outputs mount,
which the pose and any6d containers see at /data/meshes -- pass
`metric_glb_path` as `mesh` to /pose/estimate or /any6d/estimate.

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
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle

from manip_interfaces.srv import GenerateMesh

from .img import camera_info_to_K, image_to_depth_m, image_to_mono, image_to_rgb
from .zmq_client import SidecarClient, SidecarError

TRELLIS_ADDR = os.environ.get("TRELLIS_ADDR", "tcp://127.0.0.1:5669")
TIMEOUT_MS = int(float(os.environ.get("TRELLIS_TIMEOUT_S", 240)) * 1000)


class Trellis2Bridge(Node):
    def __init__(self):
        super().__init__("trellis2_bridge")
        self._client = SidecarClient(TRELLIS_ADDR, TIMEOUT_MS)
        self._cbg = MutuallyExclusiveCallbackGroup()
        self.create_service(
            GenerateMesh, "trellis2/generate_mesh", self._on_generate,
            callback_group=self._cbg,
        )
        up = self._client.ping(cmd_key="op")
        self.get_logger().info(
            f"trellis2 bridge up -> {TRELLIS_ADDR} ({'sidecar alive' if up else 'sidecar NOT responding'})")

    def _on_generate(self, req, res):
        try:
            payload = {
                "op": "generate",
                "rgb": image_to_rgb(req.rgb),
                "seed": int(req.seed),
                "decimation_target": int(req.decimation_target),
                "texture_size": int(req.texture_size),
                "output_name": req.output_name or None,
                "return_glb": False,
            }
            if req.mask.height > 0:
                payload["mask"] = image_to_mono(req.mask)
            if req.depth.height > 0:
                payload["depth"] = image_to_depth_m(req.depth)
                payload["K"] = camera_info_to_K(req.camera_info)

            self.get_logger().info("generate_mesh ...")
            reply = self._client.call(payload)
        except (TimeoutError, SidecarError, ValueError) as e:
            res.success = False
            res.message = f"{type(e).__name__}: {e}"
            self.get_logger().error(res.message)
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
        res.metric_valid = bool(metric) and bool(metric.get("ok", True))
        res.message = (
            f"{len(verts)} verts / {len(faces)} faces in {res.gen_time:.1f}s"
            + (f"; scale={res.scale:.4f} rmse={res.registration_rmse * 1e3:.1f}mm"
               if res.metric_valid else " (canonical only)")
        )
        return res


def main():
    rclpy.init()
    node = Trellis2Bridge()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()