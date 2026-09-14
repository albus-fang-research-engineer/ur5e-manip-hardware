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
   "return_all": false,             # optional: also return every refined hypothesis
   "rerank": false}                 # optional: mask-conditioned re-rank (see rerank.py)
      -> {"ok": True, "pose": 4x4 float32,   # cam_T_obj
          "hypotheses": Nx4x4 float32,       # only with return_all: all refined poses,
          "scores": N float32,               #   scorer-sorted (best first), same frame
          "texture": "simple"|"pbr->simple"|"none",   # what the scorer's RGB channel saw
          "rerank": {...}}                    # only with rerank: RerankRecord (changed, reason, ...)

`rerank` selects among FoundationPose's refined hypotheses by how much of the
SAM mask the scorer's top-K body consensus leaves unexplained -- the part of
the object the scorer cannot see (a mug handle behind the body). Off by
default; when it declines, the scorer's pick is returned and `rerank.reason`
says why. `rerank.u_frac` above ~0.35 means the top-K do not agree on the
body: treat as a failed registration upstream.

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
from Utils import nvdiffrast_render
import nvdiffrast.torch as dr
import torch

from rerank import rerank_hypotheses, record_dict

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

    def register(self, K, rgb, depth, mask, iters, rerank=False):
        """Returns (pose, rerank_record_or_None). With rerank=True the pick
        among the refined hypotheses may be changed by rerank_hypotheses();
        est.pose_last is updated so `track` continues from the chosen one."""
        pose = self.est.register(K=K, rgb=rgb, depth=depth,
                                 ob_mask=mask.astype(bool), iteration=iters)
        if not rerank:
            return np.asarray(pose, dtype=np.float32), None
        poses_c = self.est.poses                          # centred-mesh frame, scorer-sorted
        scores = self.est.scores.data.cpu().numpy() if hasattr(self.est.scores, "data") else np.asarray(self.est.scores)
        H, W = mask.shape[:2]
        sils, Ks, hs, ws = self._silhouettes(K, H, W, poses_c)
        mask_s = np.asarray(mask).astype(bool)[::self.RERANK_STRIDE, ::self.RERANK_STRIDE][:hs, :ws]
        depth_s = np.asarray(depth, np.float32)[::self.RERANK_STRIDE, ::self.RERANK_STRIDE][:hs, :ws]
        poses_np = poses_c.data.cpu().numpy() if hasattr(poses_c, "data") else np.asarray(poses_c)
        chosen, rec = rerank_hypotheses(sils, scores, poses_np, mask_s, depth_s, float(self.est.diameter))
        tf = self.est.get_tf_to_centered_mesh()
        tf_np = tf.data.cpu().numpy() if hasattr(tf, "data") else np.asarray(tf)
        if chosen != 0:
            self.est.pose_last = poses_c[chosen]
            pose = poses_np[chosen] @ tf_np
        log.info("rerank: %s (from rank 0 -> %d, %.0f deg, expl %.2f -> %.2f, U %d px = %.1f%% of mask, %d survivors)",
                 rec.reason, rec.to_rank, rec.rotation_deg, rec.expl_from, rec.expl_to, rec.u_px,
                 100 * rec.u_frac, rec.n_survivors)
        return np.asarray(pose, dtype=np.float32), record_dict(rec)

    RERANK_STRIDE = 2          # silhouettes at half resolution: enough for a part-placement test
    RERANK_CHUNK = 32          # hypotheses per nvdiffrast batch

    def _silhouettes(self, K, H, W, poses_c):
        """Boolean silhouettes of all refined hypotheses at reduced resolution,
        via the same nvdiffrast renderer FoundationPose scores with."""
        s = 1.0 / self.RERANK_STRIDE
        hs, ws = int(H * s), int(W * s)
        Ks = np.asarray(K, np.float64).copy(); Ks[:2] *= s
        out = []
        poses_t = poses_c if torch.is_tensor(poses_c) else torch.as_tensor(poses_c, device="cuda", dtype=torch.float)
        for i in range(0, len(poses_t), self.RERANK_CHUNK):
            _, depth_r, _ = nvdiffrast_render(K=Ks, H=hs, W=ws, ob_in_cams=poses_t[i:i + self.RERANK_CHUNK],
                                              glctx=self.est.glctx, mesh_tensors=self.est.mesh_tensors,
                                              output_size=np.asarray([hs, ws]))
            out.append((depth_r > 0).cpu().numpy())
        return np.concatenate(out, 0), Ks, hs, ws

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
                pose, rr = sessions[obj].register(
                    K=np.asarray(req["K"], np.float64).reshape(3, 3),
                    rgb=req["rgb"], depth=req["depth"], mask=req["mask"],
                    iters=int(req.get("est_refine_iter", 5)),
                    rerank=bool(req.get("rerank", False)),
                )
                rep = {"ok": True, "pose": pose, "texture": sessions[obj].texture}
                if rr is not None:
                    rep["rerank"] = rr
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