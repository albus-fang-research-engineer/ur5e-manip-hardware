"""ROS2 <-> FoundationPose ZMQ bridge (model-based: needs a mesh).

    /pose/estimate   srv EstimatePose   -> sidecar "register"
    /pose/release    srv Release
    /pose/<obj>/pose PoseStamped        streaming "track"
    TF  camera_optical -> pose_<obj>

`mesh` is a filename under the sidecar's /opt/meshes, or an absolute path on
a mount it can see. docker-compose mounts ./any6d_runtime/outputs at
/data/any6d (Any6D's scaled final_mesh_<obj>.obj -- the tracker-of-record
input; the pose is expressed in that file's frame) and
./trellis2_runtime/outputs at /data/meshes (a TRELLIS.2 GLB can be
registered directly; the sidecar swaps its PBR material for a simple one
so FoundationPose can read the texture).

decide=true (EstimatePose): the register asks for the re-rank, every refined
hypothesis and a render of each re-rank survivor; then pose_decide.decide()
has Orient Anything read the real masked crop (always, including when the
re-rank declines) and each survivor, selects the survivor whose yaw agrees,
and the response carries the decision record. A hard stop (the re-rank's
top-K disagree on the body) returns the record with success=true but the
object is not tracked.

Env:
    POSE_ADDR          tcp://127.0.0.1:5667
    POSE_EST_TIMEOUT_S 300
    ORIANY_ADDR        tcp://127.0.0.1:5673   (decide=true only)
"""

import json
import os

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

from . import pose_decide
from .tracker_bridge import TrackerBridge
from .zmq_client import SidecarClient


class PoseBridge(TrackerBridge):
    NS = "pose"
    ADDR = os.environ.get("POSE_ADDR", "tcp://127.0.0.1:5667")
    EST_TIMEOUT_MS = int(float(os.environ.get("POSE_EST_TIMEOUT_S", 300)) * 1000)

    ORIANY_ADDR = os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673")

    def __init__(self):
        super().__init__("foundationpose_bridge")
        self.oriany = SidecarClient(self.ORIANY_ADDR, 120_000)

    def estimate_payload(self, req, rgb, depth, K, mask):
        if not req.mesh:
            raise ValueError("FoundationPose needs `mesh` (img_to_3d unsupported)")
        p = {"cmd": "register", "obj": req.obj, "mesh": req.mesh,
             "rgb": rgb, "depth": depth, "K": K, "mask": mask,
             "est_refine_iter": int(req.refine_iter)}
        if req.decide:
            p.update(rerank=True, survivor_crops=True, return_all=True)
        return p

    def post_register(self, req, rep, T, res, rgb, mask):
        if not req.decide:
            return T, True
        orient, select = pose_decide.sidecar_fns(self.oriany, self.est_client, req.obj)
        T_sel, rec, sheet = pose_decide.decide(rep, rgb, mask, orient, select,
                                               want_sheet=bool(req.decide_debug))
        res.decision_hard_stop = bool(rec["hard_stop"])
        res.decision_reason = rec["reason"]
        res.decision_json = json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        if sheet is not None:
            m = Image()
            m.header = req.rgb.header
            m.height, m.width = sheet.shape[:2]
            m.encoding, m.step = "rgb8", sheet.shape[1] * 3
            m.data = np.ascontiguousarray(sheet, np.uint8).tobytes()
            res.decision_sheet = m
        line = f"{req.obj}: " + pose_decide.summary_line(rec)
        (self.get_logger().error if rec["hard_stop"] else self.get_logger().info)(line)
        if rec["hard_stop"]:
            res.message = f"HARD STOP ({rec['reason']}): registration unreliable, not tracking"
        return (T_sel if T_sel is not None else T), not rec["hard_stop"]


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
