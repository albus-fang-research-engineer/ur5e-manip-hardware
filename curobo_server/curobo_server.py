"""cuRoboV2 ZMQ REP server -- ESDF construction from depth, whole module importable.

Wraps cuRoboV2's block-sparse TSDF mapper (curobo.perception.Mapper): depth
frames go in, a dense ESDF VoxelGrid comes out (PBA+ on GPU). The rest of
cuRoboV2 (IK / trajopt / motion gen) is installed in the image; extend this
server or exec in when you need it.

Mapper config comes from env at startup (CUROBO_VOXEL_SIZE, CUROBO_EXTENT,
CUROBO_IMAGE_HEIGHT/WIDTH, CUROBO_NUM_CAMERAS) and can be rebuilt at runtime
via the "configure" cmd.

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "ping"} -> {"ok": True}

  {"cmd": "configure",                # optional; rebuilds the Mapper
   "voxel_size": 0.01,
   "extent": [2.0, 2.0, 1.5],         # meters xyz
   "image_height": 480, "image_width": 848,
   "num_cameras": 1}
      -> {"ok": True, "memory_mb": float}

  {"cmd": "integrate",
   "depth": HxW float32 (meters) or BxHxW,   # B == num_cameras
   "intrinsics": 3x3 float32 or Bx3x3,       # [[fx,0,cx],[0,fy,cy],[0,0,1]]
   "pose": 4x4 float32 or Bx4x4,             # camera-to-world, map frame
   "rgb": HxWx3 uint8 or BxHxWx3 (optional; zeros = geometry-only path)}
      -> {"ok": True}

  {"cmd": "esdf",
   "voxel_size": 0.01,                # optional override (m)
   "origin": [x, y, z]}               # optional override (sliding window)
      -> {"ok": True,
          "esdf": float32 tensor,     # signed distance (m); reshape w/ grid_shape
          "grid_shape": [nx, ny, nz], # X slowest, Z fastest (VoxelGrid conv.)
          "low": [x,y,z], "high": [x,y,z],
          "voxel_size": float, "dims": [x,y,z]}

  {"cmd": "reset"}                              -> {"ok": True}
  {"cmd": "clear_region", "min": [..], "max": [..]} -> {"ok": True, "n_cleared": int}
  {"cmd": "save",  "name": "scene.pt"}          -> {"ok": True, "path": str}
  {"cmd": "load",  "name": "scene.pt"}          -> {"ok": True, "memory_mb": float}
  {"cmd": "stats"}                              -> {"ok": True, "memory_mb": float, ...}

Conventions: depth in METERS float32 (convert RealSense uint16 mm on the
client). Pose is camera-to-world as a 4x4 (converted to cuRobo's wxyz Pose via
Pose.from_matrix internally, so no quaternion-order footguns on the wire).
First integrate/esdf call after container start pays NVRTC + warp JIT
(seconds with a warm /opt/kernel_cache mount, ~a minute cold).
"""

import os
import logging

import numpy as np
import torch
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from curobo.perception import Mapper, MapperCfg
from curobo.types import CameraObservation, Pose

PORT = int(os.environ.get("CUROBO_PORT", "5671"))
BLOCKS_DIR = os.environ.get("CUROBO_BLOCKS_DIR", "/data/tsdf_blocks")

logging.basicConfig(level=logging.INFO, format="[curobo-server] %(message)s")
log = logging.getLogger(__name__)


def default_cfg_kwargs():
    extent = tuple(
        float(x) for x in os.environ.get("CUROBO_EXTENT", "2.0,2.0,1.5").split(",")
    )
    return dict(
        voxel_size=float(os.environ.get("CUROBO_VOXEL_SIZE", "0.01")),
        extent=extent,
        image_height=int(os.environ.get("CUROBO_IMAGE_HEIGHT", "480")),
        image_width=int(os.environ.get("CUROBO_IMAGE_WIDTH", "848")),
        num_cameras=int(os.environ.get("CUROBO_NUM_CAMERAS", "1")),
    )


def build_mapper(voxel_size, extent, image_height, image_width, num_cameras):
    """MapperCfg defaults mirror the upstream volumetric_mapping example;
    truncation = 6 * voxel is the recommended band."""
    cfg = MapperCfg(
        voxel_size=voxel_size,
        extent_meters_xyz=tuple(extent),
        truncation_distance=voxel_size * 6,
        depth_maximum_distance=3.0,       # tabletop scale; raise for room-scale
        depth_minimum_distance=0.05,
        minimum_tsdf_weight=4.0,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        enable_static=True,               # analytic-primitive channel (table etc.)
        num_cameras=num_cameras,
        image_height=image_height,
        image_width=image_width,
        block_size=8,
        color_grid_size=8,
    )
    mapper = Mapper(cfg)
    log.info(
        "mapper: voxel=%.3fm extent=%s img=%dx%d cams=%d (%.1f MB)",
        voxel_size, extent, image_width, image_height, num_cameras,
        mapper.memory_usage_mb(),
    )
    return mapper, cfg


def to_batched(arr, dtype, batch, name):
    """HxW... -> 1xHxW... torch tensor on cuda; validate leading batch dim."""
    t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device="cuda")
    if t.ndim == 2 or (t.ndim == 3 and name == "rgb") or (t.ndim == 2 and name == "intrinsics"):
        t = t.unsqueeze(0)
    if name == "intrinsics" and t.ndim == 2:
        t = t.unsqueeze(0)
    if t.shape[0] != batch:
        raise ValueError(f"{name}: expected batch {batch}, got {tuple(t.shape)}")
    return t


def make_observation(msg, cfg):
    b = cfg.num_cameras
    depth = to_batched(msg["depth"], torch.float32, b, "depth")
    intr = torch.as_tensor(
        np.ascontiguousarray(msg["intrinsics"]), dtype=torch.float32, device="cuda"
    ).reshape(-1, 3, 3)
    if intr.shape[0] == 1 and b > 1:
        intr = intr.expand(b, 3, 3).contiguous()

    pose_mat = torch.as_tensor(
        np.ascontiguousarray(msg["pose"]), dtype=torch.float32, device="cuda"
    ).reshape(-1, 4, 4)
    poses = [Pose.from_matrix(pose_mat[i]) for i in range(pose_mat.shape[0])]
    pose = Pose(
        position=torch.cat([p.position.view(1, 3) for p in poses]),
        quaternion=torch.cat([p.quaternion.view(1, 4) for p in poses]),
    )

    if "rgb" in msg and msg["rgb"] is not None:
        rgb = to_batched(msg["rgb"], torch.uint8, b, "rgb")
    else:
        # geometry-only path: cached zero rgb, same batch/image shape as depth
        rgb = torch.zeros((*depth.shape, 3), dtype=torch.uint8, device="cuda")

    return CameraObservation(
        depth_image=depth, rgb_image=rgb, pose=pose, intrinsics=intr
    )


def handle(msg, state):
    cmd = msg.get("cmd")
    mapper, cfg = state["mapper"], state["cfg"]

    if cmd == "ping":
        return {"ok": True}

    if cmd == "configure":
        kw = default_cfg_kwargs()
        kw.update({k: msg[k] for k in kw if k in msg})
        state["mapper"], state["cfg"] = build_mapper(**kw)
        state["cfg_kwargs"] = kw
        return {"ok": True, "memory_mb": state["mapper"].memory_usage_mb()}

    if cmd == "integrate":
        mapper.integrate(camera_observation=make_observation(msg, cfg))
        return {"ok": True}

    if cmd == "esdf":
        origin = msg.get("origin")
        if origin is not None:
            origin = torch.as_tensor(origin, dtype=torch.float32, device="cuda")
        grid = mapper.compute_esdf(
            esdf_origin=origin, esdf_voxel_size=msg.get("voxel_size")
        )
        grid_shape, low, high = grid.get_grid_shape()
        return {
            "ok": True,
            "esdf": grid.feature_tensor.float().cpu().numpy(),
            "grid_shape": grid_shape,
            "low": low,
            "high": high,
            "voxel_size": float(grid.voxel_size),
            "dims": [float(d) for d in grid.dims],
        }

    if cmd == "reset":
        mapper.reset()
        return {"ok": True}

    if cmd == "clear_region":
        n = mapper.clear_region(msg["min"], msg["max"])
        return {"ok": True, "n_cleared": int(n)}

    if cmd == "save":
        os.makedirs(BLOCKS_DIR, exist_ok=True)
        path = os.path.join(BLOCKS_DIR, os.path.basename(msg.get("name", "blocks.pt")))
        mapper.save_blocks(path)
        return {"ok": True, "path": path}

    if cmd == "load":
        path = os.path.join(BLOCKS_DIR, os.path.basename(msg["name"]))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found in mounted {BLOCKS_DIR}")
        state["mapper"] = Mapper.load_blocks(path, cfg)
        return {"ok": True, "memory_mb": state["mapper"].memory_usage_mb()}

    if cmd == "stats":
        return {"ok": True, "memory_mb": mapper.memory_usage_mb(),
                **{k: v for k, v in (mapper.get_stats() or {}).items()
                   if isinstance(v, (int, float, str))}}

    return {"ok": False, "error": f"unknown cmd: {cmd}"}


def main():
    log.info("building mapper (first CUDA kernel compile happens on first integrate)...")
    kw = default_cfg_kwargs()
    mapper, cfg = build_mapper(**kw)
    state = {"mapper": mapper, "cfg": cfg, "cfg_kwargs": kw}

    if os.environ.get("CUROBO_VISUALIZE"):
        from curobo.viewer import ViserVisualizer  # noqa: F401 -- lazy opt-in
        state["viser"] = ViserVisualizer(connect_port=8080)
        log.info("viser: http://localhost:8080")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{PORT}")
    log.info("listening on :%d", PORT)

    while True:
        msg = msgpack.unpackb(sock.recv())
        try:
            reply = handle(msg, state)
        except Exception as e:  # REP socket must always answer
            log.exception("cmd failed: %s", msg.get("cmd"))
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()