"""FoundationPose ZMQ REP server.

Mirrors run_demo.py's estimator lifecycle: one FoundationPose instance per
registered object (register once with a mask, then track frame-to-frame).

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "register",
   "obj":  "mustard",              # session key
   "mesh": "mustard.obj",          # filename under /opt/meshes
   "rgb":  HxWx3 uint8,
   "depth": HxW float32 (meters),
   "K":    3x3 float32,
   "mask": HxW uint8/bool,
   "est_refine_iter": 5}
      -> {"ok": True, "pose": 4x4 float32}   # cam_T_obj

  {"cmd": "track", "obj": "mustard",
   "rgb": ..., "depth": ..., "K": ...,
   "track_refine_iter": 2}
      -> {"ok": True, "pose": 4x4 float32}

  {"cmd": "release", "obj": "mustard"} -> {"ok": True}
  {"cmd": "ping"} -> {"ok": True}

Depth convention: meters, invalid = 0 (matches FoundationPose's readers).
"""

import os
import logging

import numpy as np
import trimesh
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

# FoundationPose imports (PYTHONPATH=/opt/FoundationPose)
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr

MESH_DIR = os.environ.get("MESH_DIR", "/opt/meshes")
PORT = int(os.environ.get("POSE_PORT", "5667"))

logging.basicConfig(level=logging.INFO, format="[pose-server] %(message)s")
log = logging.getLogger(__name__)


class Session:
    """One FoundationPose estimator bound to one mesh."""

    def __init__(self, mesh_path, scorer, refiner, glctx):
        mesh = trimesh.load(mesh_path, force="mesh")
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )

    def register(self, K, rgb, depth, mask, iters):
        pose = self.est.register(K=K, rgb=rgb, depth=depth,
                                 ob_mask=mask.astype(bool), iteration=iters)
        return np.asarray(pose, dtype=np.float32)

    def track(self, K, rgb, depth, iters):
        pose = self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=iters)
        return np.asarray(pose, dtype=np.float32)


def main():
    # Shared across sessions (heavy: loads refiner + scorer weights once)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    sessions = {}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{PORT}")
    log.info("listening on :%d, meshes from %s", PORT, MESH_DIR)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            cmd = req["cmd"]
            if cmd == "ping":
                rep = {"ok": True}

            elif cmd == "register":
                obj = req["obj"]
                mesh_path = os.path.join(MESH_DIR, req["mesh"])
                sessions[obj] = Session(mesh_path, scorer, refiner, glctx)
                pose = sessions[obj].register(
                    K=np.asarray(req["K"], np.float64).reshape(3, 3),
                    rgb=req["rgb"], depth=req["depth"], mask=req["mask"],
                    iters=int(req.get("est_refine_iter", 5)),
                )
                rep = {"ok": True, "pose": pose}

            elif cmd == "track":
                sess = sessions[req["obj"]]
                pose = sess.track(
                    K=np.asarray(req["K"], np.float64).reshape(3, 3),
                    rgb=req["rgb"], depth=req["depth"],
                    iters=int(req.get("track_refine_iter", 2)),
                )
                rep = {"ok": True, "pose": pose}

            elif cmd == "release":
                sessions.pop(req["obj"], None)
                rep = {"ok": True}

            else:
                rep = {"ok": False, "error": f"unknown cmd {cmd!r}"}

        except Exception as e:  # keep REP socket in lockstep
            log.exception("request failed")
            rep = {"ok": False, "error": repr(e)}

        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()