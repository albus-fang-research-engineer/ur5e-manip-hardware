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
   "est_refine_iter": 5,
   "return_all": false}             # optional: also return every refined hypothesis
      -> {"ok": True, "pose": 4x4 float32,   # cam_T_obj
          "hypotheses": Nx4x4 float32,       # only with return_all: all refined poses,
          "scores": N float32,               #   scorer-sorted (best first), same frame
          "texture": "simple"|"pbr->simple"|"none"}   # what the scorer's RGB channel saw

`return_all` exists for the yaw-ambiguity diagnosis: FoundationPose keeps
every refined hypothesis (est.poses, est.scores) and on a near-symmetric
object its scorer is flat across them, so an offline re-rank against the
SAM mask can be evaluated before any in-server re-rank is enabled.

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
        # A textured GLB loads with a PBRMaterial; FoundationPose's
        # make_mesh_tensors reads `material.image`, which only SimpleMaterial
        # has. Any6D's final_mesh_*.obj is untextured so this is a no-op on the
        # normal path; it matters if a TRELLIS GLB is ever registered directly.
        vis = getattr(mesh, "visual", None)
        mat = getattr(vis, "material", None)
        self.texture = "none"
        if vis is not None and vis.kind == "texture" and mat is not None:
            if not hasattr(mat, "image") and hasattr(mat, "to_simple"):
                mesh.visual.material = mat.to_simple()
                self.texture = "pbr->simple"
            else:
                self.texture = "simple"
            if getattr(mesh.visual.material, "image", None) is None:
                self.texture = "none"
        # Report what the scorer's RGB channel will actually see: on a
        # yaw-symmetric body the texture is its only tie-breaker.
        log.info("mesh %s: %d verts, texture=%s", os.path.basename(mesh_path),
                 len(mesh.vertices), self.texture)
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

    def hypotheses(self):
        """All refined hypotheses from the last register, scorer-sorted (best
        first), expressed in the same frame as the returned pose (i.e. with
        FoundationPose's centring compensation applied), plus their scores.
        Read-only: est.poses / est.scores are what register() stored."""
        poses = self.est.poses
        scores = self.est.scores
        if poses is None:
            return None, None
        poses = poses.data.cpu().numpy() if hasattr(poses, "data") else np.asarray(poses)
        scores = scores.data.cpu().numpy() if hasattr(scores, "data") else np.asarray(scores)
        tf = self.est.get_tf_to_centered_mesh()
        tf = tf.data.cpu().numpy() if hasattr(tf, "data") else np.asarray(tf)
        return (poses @ tf).astype(np.float32), scores.astype(np.float32)

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
                rep = {"ok": True, "pose": pose, "texture": sessions[obj].texture}
                if req.get("return_all"):
                    hyp, sc = sessions[obj].hypotheses()
                    if hyp is not None:
                        rep["hypotheses"] = hyp
                        rep["scores"] = sc
                        log.info("register %s: %d hypotheses, scores %.3f..%.3f (top-10 spread %.3f)",
                                 obj, len(sc), float(sc.min()), float(sc.max()),
                                 float(sc[0] - sc[min(9, len(sc) - 1)]))

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