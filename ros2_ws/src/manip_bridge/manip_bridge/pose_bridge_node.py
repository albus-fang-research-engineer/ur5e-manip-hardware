"""ROS2 <-> FoundationPose ZMQ bridge (model-based: needs a mesh).

    /pose/estimate   srv EstimatePose   -> sidecar "register"
    /pose/release    srv Release
    /pose/<obj>/pose PoseStamped        streaming "track"
    TF  camera_optical -> pose_<obj>

`mesh` is a filename under the sidecar's /opt/meshes, or an absolute path on
a mount it can see (docker-compose mounts ./trellis2_runtime/outputs at
/data/meshes in the pose container, so a TRELLIS.2 metric GLB can be fed
straight in -- that is the "FoundationPose on a TRELLIS mesh" counterpart
to Any6D).

Env:
    POSE_ADDR          tcp://127.0.0.1:5667
    POSE_EST_TIMEOUT_S 300
"""

import os

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .tracker_bridge import TrackerBridge


class PoseBridge(TrackerBridge):
    NS = "pose"
    ADDR = os.environ.get("POSE_ADDR", "tcp://127.0.0.1:5667")
    EST_TIMEOUT_MS = int(float(os.environ.get("POSE_EST_TIMEOUT_S", 300)) * 1000)

    def __init__(self):
        super().__init__("foundationpose_bridge")

    def estimate_payload(self, req, rgb, depth, K, mask):
        if not req.mesh:
            raise ValueError("FoundationPose needs `mesh` (img_to_3d unsupported)")
        return {"cmd": "register", "obj": req.obj, "mesh": req.mesh,
                "rgb": rgb, "depth": depth, "K": K, "mask": mask,
                "est_refine_iter": int(req.refine_iter)}


def main():
    rclpy.init()
    node = PoseBridge()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
