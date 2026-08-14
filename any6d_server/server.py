"""Any6D ZMQ REP server.

Mirrors run_demo.py's lifecycle: build (or load) a mesh for the object, then
Any6D.register_any6d() jointly refines pose + metric scale from one RGB-D
anchor view. The final metrically-scaled mesh (est.mesh) is exported so the
grasp/planning side can consume it.

Two ways to get the object mesh:
  - "mesh": filename under MESH_DIR -- a reference mesh (CAD or a TRELLIS.2 /
    InstantMesh output). Any6D will rescale it during registration, so it
    doesn't need to be metric.
  - "img_to_3d": true -- run the in-repo SAM2 (box-prompted mask refine) +
    InstantMesh pipeline on the anchor RGB to generate the mesh first. Slow
    (~1-2 min on the 3090) and needs the sam2/instantmesh checkpoints
    mounted.

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "estimate",
   "obj":  "mustard",              # session key; also names output files
   "rgb":  HxWx3 uint8,
   "depth": HxW float32 (meters),
   "K":    3x3 float32,
   "mask": HxW uint8/bool,
   "mesh": "mustard.obj",          # XOR img_to_3d
   "img_to_3d": False,
   "est_refine_iter": 5}
      -> {"ok": True, "pose": 4x4 float32,       # cam_T_obj
          "mesh_path": "/data/any6d/final_mesh_mustard.obj",
          "extents": 3 float32}                  # scaled mesh AABB (m)

  {"cmd": "track", "obj": "mustard",
   "rgb": ..., "depth": ..., "K": ...,
   "track_refine_iter": 2}
      -> {"ok": True, "pose": 4x4 float32}
      (Any6D subclasses FoundationPose, so frame-to-frame tracking uses the
       scaled mesh from the estimate step.)

  {"cmd": "release", "obj": "mustard"} -> {"ok": True}
  {"cmd": "ping"} -> {"ok": True}

Depth convention: meters, invalid = 0 (matches the pose sidecar).
"""

import os
import logging

import numpy as np
import trimesh
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

# Any6D imports (PYTHONPATH=/opt/Any6D; cwd must be /opt/Any6D for the
# relative config paths inside sam2_instantmesh)
os.chdir(os.environ.get("ANY6D_ROOT", "/opt/Any6D"))

from estimater import Any6D  # noqa: E402
from foundationpose.Utils import get_bounding_box, align_mesh_to_coordinate  # noqa: E402

MESH_DIR = os.environ.get("MESH_DIR", "/opt/meshes")
OUT_DIR = os.environ.get("ANY6D_OUT_DIR", "/data/any6d")
PORT = int(os.environ.get("ANY6D_PORT", "5672"))

logging.basicConfig(level=logging.INFO, format="[any6d-server] %(message)s")
log = logging.getLogger(__name__)


def _img_to_3d(rgb, mask, work_dir, obj):
    """SAM2 box-refine + InstantMesh, exactly the run_demo.py --img_to_3d
    path. Imported lazily so mesh-provided estimates don't pay the SAM2 /
    diffusion model load."""
    from sam2_instantmesh import (running_sam_box, preprocess_image,
                                  diffusion_image_generation,
                                  instant_mesh_process)

    cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
    input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
    mask_refine = running_sam_box(rgb, input_box)
    input_image = preprocess_image(rgb, mask_refine, work_dir, obj)
    images = diffusion_image_generation(work_dir, work_dir, obj,
                                        input_image=input_image)
    instant_mesh_process(images, work_dir, obj)

    mesh = trimesh.load(os.path.join(work_dir, f"mesh_{obj}.obj"))
    mesh = align_mesh_to_coordinate(mesh)
    centered = os.path.join(work_dir, f"center_mesh_{obj}.obj")
    mesh.export(centered)
    return trimesh.load(centered)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sessions = {}   # obj -> Any6D estimator (holds the scaled mesh)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{PORT}")
    log.info("listening on :%d, meshes from %s, outputs to %s",
             PORT, MESH_DIR, OUT_DIR)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            cmd = req["cmd"]
            if cmd == "ping":
                rep = {"ok": True}

            elif cmd == "estimate":
                obj = req["obj"]
                rgb = req["rgb"]
                depth = np.asarray(req["depth"], np.float32)
                K = np.asarray(req["K"], np.float64).reshape(3, 3)
                mask = np.asarray(req["mask"]).astype(bool)
                work_dir = os.path.join(OUT_DIR, obj)
                os.makedirs(work_dir, exist_ok=True)

                if req.get("img_to_3d"):
                    log.info("[%s] img_to_3d: SAM2 + InstantMesh...", obj)
                    mesh = _img_to_3d(rgb, mask, work_dir, obj)
                elif "mesh" in req:
                    mesh = trimesh.load(
                        os.path.join(MESH_DIR, req["mesh"]), force="mesh")
                else:
                    raise ValueError("estimate needs 'mesh' or img_to_3d=True")

                est = Any6D(symmetry_tfs=None, mesh=mesh,
                            debug_dir=work_dir, debug=0)
                pose = est.register_any6d(
                    K=K, rgb=rgb, depth=depth, ob_mask=mask,
                    iteration=int(req.get("est_refine_iter", 5)), name=obj)
                sessions[obj] = est

                mesh_path = os.path.join(OUT_DIR, f"final_mesh_{obj}.obj")
                est.mesh.export(mesh_path)
                rep = {"ok": True,
                       "pose": np.asarray(pose, np.float32),
                       "mesh_path": mesh_path,
                       "extents": np.asarray(est.mesh.extents, np.float32)}

            elif cmd == "track":
                est = sessions[req["obj"]]
                pose = est.track_one(
                    rgb=req["rgb"],
                    depth=np.asarray(req["depth"], np.float32),
                    K=np.asarray(req["K"], np.float64).reshape(3, 3),
                    iteration=int(req.get("track_refine_iter", 2)))
                rep = {"ok": True, "pose": np.asarray(pose, np.float32)}

            elif cmd == "release":
                sessions.pop(req["obj"], None)
                rep = {"ok": True}

            else:
                rep = {"ok": False, "error": f"unknown cmd {cmd!r}"}

        except Exception as e:  # noqa: BLE001 -- REP must always answer
            log.exception("handler failed")
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()