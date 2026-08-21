"""ROS2 <-> Any6D ZMQ bridge (model-free: mesh optional, metric scale out).

    /any6d/estimate   srv EstimatePose   -> sidecar "estimate"
    /any6d/release    srv Release
    /any6d/<obj>/pose PoseStamped        streaming "track"
    TF  camera_optical -> any6d_<obj>

`img_to_3d=true` generates the mesh from the anchor RGB (SAM2 + InstantMesh,
~1-2 min); otherwise `mesh` is a reference mesh Any6D will rescale (CAD or a
TRELLIS.2 output -- need not be metric). The reply carries the scaled mesh
path + AABB extents.

Env:
    ANY6D_ADDR          tcp://127.0.0.1:5672
    ANY6D_EST_TIMEOUT_S 900
"""

import os

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from .tracker_bridge import TrackerBridge


class Any6DBridge(TrackerBridge):
    NS = "any6d"
    ADDR = os.environ.get("ANY6D_ADDR", "tcp://127.0.0.1:5672")
    EST_TIMEOUT_MS = int(float(os.environ.get("ANY6D_EST_TIMEOUT_S", 900)) * 1000)

    def __init__(self):
        super().__init__("any6d_bridge")

    def estimate_payload(self, req, rgb, depth, K, mask):
        p = {"cmd": "estimate", "obj": req.obj, "rgb": rgb, "depth": depth,
             "K": K, "mask": mask, "est_refine_iter": int(req.refine_iter)}
        if req.img_to_3d:
            p["img_to_3d"] = True
        elif req.mesh:
            p["mesh"] = req.mesh
        else:
            raise ValueError("Any6D needs `mesh` or img_to_3d=true")
        return p

    def parse_estimate(self, rep, res):
        res.extents = [float(x) for x in np.asarray(rep.get("extents", []))]
        res.mesh_path = str(rep.get("mesh_path", ""))
        if res.extents:
            e = np.round(res.extents, 3).tolist()
            res.message = f"registered, extents={e} mesh={res.mesh_path}"


def main():
    rclpy.init()
    node = Any6DBridge()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
