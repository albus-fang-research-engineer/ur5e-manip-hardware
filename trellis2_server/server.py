"""TRELLIS.2 ZMQ sidecar.

REP socket, msgpack-numpy wire format (same convention as the pose /
pointso sidecars).

Request:
    {
      "op":               "generate" | "ping",
      "rgb":              HxWx3 uint8 (RGB order),
      "mask":             HxW uint8, optional. >0 = foreground. If given we
                          compose an RGBA input, which makes the pipeline
                          skip its own matting -- strongly preferred on the
                          robot, where SAM/segmentation masks already exist.
      "seed":             int, optional (default 42)
      "decimation_target":int, optional (default 500_000) -- faces in the GLB
      "texture_size":     int, optional (default 2048)
      "return_glb":       bool, optional (default False) -- inline GLB bytes
      "output_name":      str, optional -- basename for the GLB written to
                          OUT_DIR; default "t2_<unixtime>"
    }

Reply:
    {
      "ok":        bool,
      "error":     str (on failure),
      "glb_path":  container path of the written GLB (host-visible via the
                   shared ./trellis2_runtime/outputs mount),
      "glb":       bytes (only if return_glb),
      "vertices":  Nx3 float32   } decimated geometry re-read from the GLB,
      "faces":     Mx3 uint32    } canonical frame, AABB [-0.5, 0.5]^3
      "gen_time":  float seconds
    }

NOTE ON SCALE: output is a canonical unit-box asset. No metric scale, no
camera-frame pose. Recover scale downstream by matching the mesh extent to
the masked depth cloud, then let FoundationPose solve the pose.

Pipeline knobs beyond seed (sampler steps, cfg, resolution cascade) are
intentionally not plumbed through yet -- see app.py upstream for what
pipeline.run() accepts and extend the request schema as needed.
"""

import io
import os
import time
import traceback

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

PORT = int(os.environ.get("TRELLIS_PORT", 5669))
OUT_DIR = os.environ.get("TRELLIS_OUT_DIR", "/data/meshes")

# nvdiffrast vertex-count ceiling used by upstream example.py
NVDIFFRAST_LIMIT = 16_777_216


def _to_pil(rgb: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    if mask is not None:
        rgba = np.dstack([rgb, np.where(mask > 0, 255, 0).astype(np.uint8)])
        return Image.fromarray(rgba, mode="RGBA")
    return Image.fromarray(rgb, mode="RGB")


def _handle_generate(pipeline, req: dict) -> dict:
    t0 = time.time()
    image = _to_pil(req["rgb"], req.get("mask"))
    seed = int(req.get("seed", 42))

    mesh = pipeline.run(image, seed=seed)[0]
    mesh.simplify(NVDIFFRAST_LIMIT)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(req.get("decimation_target", 500_000)),
        texture_size=int(req.get("texture_size", 2048)),
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    name = req.get("output_name") or f"t2_{int(time.time())}"
    glb_path = os.path.join(OUT_DIR, f"{name}.glb")
    os.makedirs(OUT_DIR, exist_ok=True)
    glb.export(glb_path, extension_webp=True)

    with open(glb_path, "rb") as f:
        glb_bytes = f.read()

    # Re-read decimated geometry for consumers that want raw arrays
    # (the ROS bridge builds shape_msgs/Mesh from these).
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    tm = (
        trimesh.util.concatenate(list(scene.geometry.values()))
        if isinstance(scene, trimesh.Scene)
        else scene
    )

    reply = {
        "ok": True,
        "glb_path": glb_path,
        "vertices": np.asarray(tm.vertices, dtype=np.float32),
        "faces": np.asarray(tm.faces, dtype=np.uint32),
        "gen_time": time.time() - t0,
    }
    if req.get("return_glb", False):
        reply["glb"] = glb_bytes

    # Optional metric scale recovery: depth (HxW float32 m, aligned to rgb)
    # + K (3x3) + mask -> similarity registration onto the canonical mesh.
    if req.get("depth") is not None and req.get("K") is not None:
        from scipy.spatial.transform import Rotation
        import metric_scale

        P_obs = metric_scale.backproject(
            np.asarray(req["depth"], dtype=np.float32),
            np.asarray(req["K"], dtype=np.float64),
            req.get("mask", np.ones(req["depth"].shape, np.uint8)),
        )
        sim = metric_scale.register_similarity(P_obs, tm)

        tm_metric = tm.copy()
        tm_metric.apply_scale(sim["scale"])
        metric_path = glb_path.replace(".glb", "_metric.glb")
        tm_metric.export(metric_path)  # geometry-only; textures stay in the
                                       # canonical GLB (scale in your loader)
        reply["metric"] = {
            "glb_path": metric_path,
            "scale": sim["scale"],
            "t": sim["t"].astype(np.float64),
            "q_xyzw": Rotation.from_matrix(sim["R"]).as_quat(),
            "rmse": sim["rmse"],
            "ok": sim["ok"],
        }

    torch.cuda.empty_cache()
    return reply


def main():
    print(f"[trellis2] loading pipeline (HF_HOME={os.environ.get('HF_HOME')})")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print("[trellis2] pipeline resident")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{PORT}")
    print(f"[trellis2] serving on :{PORT}")

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            op = req.get("op", "generate")
            if op == "ping":
                reply = {"ok": True, "pong": True}
            elif op == "generate":
                reply = _handle_generate(pipeline, req)
            else:
                reply = {"ok": False, "error": f"unknown op: {op}"}
        except Exception as e:  # noqa: BLE001 -- REP must always reply
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(reply, use_bin_type=True))


if __name__ == "__main__":
    main()