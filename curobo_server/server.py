"""cuRoboV2 ZMQ REP server -- ESDF construction from depth, whole module importable.

Wraps cuRoboV2's block-sparse TSDF mapper (curobo.perception.Mapper): depth
frames go in, a dense ESDF VoxelGrid comes out (PBA+ on GPU). The rest of
cuRoboV2 (IK / trajopt / motion gen) is installed in the image; extend this
server or exec in when you need it.

Mapper config comes from env at startup (CUROBO_VOXEL_SIZE,
CUROBO_ESDF_VOXEL_SIZE, CUROBO_EXTENT, CUROBO_IMAGE_HEIGHT/WIDTH,
CUROBO_NUM_CAMERAS) and can be rebuilt at runtime via the "configure" cmd.

ESDF grid semantics (cuRoboV2 >= 0.8.0.post1): the ESDF grid SHAPE is fixed
at Mapper construction from (extent, esdf_voxel_size) -- the seeding kernel
has fixed launch dims for CUDA-graph safety. The per-call voxel_size override
rescales SPACING on that fixed buffer, so it trades resolution against
coverage (0.02 on a grid built for 0.01 covers 2x the extent). It does NOT
reallocate. Upstream, the override is also sticky (it mutates integrator
state); this server defeats that by always passing an explicit voxel size,
so an un-overridden "esdf" call always returns the configured resolution.

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "ping"} -> {"ok": True}

  {"cmd": "configure",                # optional; rebuilds the Mapper
   "voxel_size": 0.01,                # TSDF voxel (m)
   "esdf_voxel_size": 0.01,           # ESDF voxel (m); default = voxel_size
   "extent": [2.0, 2.0, 1.5],         # meters xyz, both TSDF and ESDF grids
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
   "voxel_size": 0.02,                # optional; rescales spacing on the
                                      #   FIXED grid -> coverage scales too
   "origin": [x, y, z]}               # optional override (sliding window)
      -> {"ok": True,
          "esdf": float32 tensor,     # signed distance (m); reshape w/ grid_shape
          "grid_shape": [nx, ny, nz], # X slowest, Z fastest (VoxelGrid conv.)
          "low": [x,y,z], "high": [x,y,z],
          "voxel_size": float, "dims": [x,y,z]}

  {"cmd": "reset"}                              -> {"ok": True}
  {"cmd": "clear_region", "min": [..], "max": [..]} -> {"ok": True, "n_cleared": int}
  {"cmd": "save",  "name": "scene.pt"}          -> {"ok": True, "path": str}
  {"cmd": "load",  "name": "scene.pt",
   "import_weight": null,             # optional; null preserves saved weights
                                      #   (recommended). If set, must be
                                      #   STRICTLY > minimum_tsdf_weight or
                                      #   every voxel reads as unobserved.
   "force": false}                    # skip saved-config compatibility check
      -> {"ok": True, "n_blocks": int, "memory_mb": float}
  {"cmd": "stats"}                              -> {"ok": True, "memory_mb": float, ...}

Conventions: depth in METERS float32 (convert RealSense uint16 mm on the
client). Pose is camera-to-world as a 4x4 (converted to cuRobo's wxyz Pose via
Pose.from_matrix internally, so no quaternion-order footguns on the wire).
First integrate/esdf call after container start pays NVRTC + warp JIT
(seconds with a warm /opt/kernel_cache mount, ~a minute cold).
"""

import json
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
    voxel_size = float(os.environ.get("CUROBO_VOXEL_SIZE", "0.01"))
    return dict(
        voxel_size=voxel_size,
        # ESDF resolution defaults to TSDF resolution; leaving MapperCfg's
        # own default (0.05 on a grid that ignores `extent`) is how you get
        # a 128^3 / 6.4 m window that silently mismatches the map.
        esdf_voxel_size=float(
            os.environ.get("CUROBO_ESDF_VOXEL_SIZE", str(voxel_size))
        ),
        extent=extent,
        image_height=int(os.environ.get("CUROBO_IMAGE_HEIGHT", "480")),
        image_width=int(os.environ.get("CUROBO_IMAGE_WIDTH", "848")),
        num_cameras=int(os.environ.get("CUROBO_NUM_CAMERAS", "1")),
    )


def build_mapper(voxel_size, esdf_voxel_size, extent, image_height, image_width,
                 num_cameras):
    """MapperCfg defaults mirror the upstream volumetric_mapping example;
    truncation = 6 * voxel is the recommended band. esdf_voxel_size and
    extent_esdf_meters_xyz are set explicitly so the fixed-shape ESDF grid
    actually covers the configured extent at the configured resolution."""
    cfg = MapperCfg(
        voxel_size=voxel_size,
        esdf_voxel_size=esdf_voxel_size,
        extent_meters_xyz=tuple(extent),
        extent_esdf_meters_xyz=tuple(extent),
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
        "mapper: voxel=%.3fm esdf=%.3fm extent=%s img=%dx%d cams=%d (%.1f MB)",
        voxel_size, esdf_voxel_size, extent, image_width, image_height,
        num_cameras, mapper.memory_usage_mb(),
    )
    return mapper, cfg


def to_batched(arr, dtype, batch, name):
    """HxW... -> 1xHxW... torch tensor on cuda; validate leading batch dim."""
    t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device="cuda")
    if t.ndim == 2 or (t.ndim == 3 and name == "rgb"):
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


def _sidecar_path(path):
    return path + ".cfg.json"


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
        # Always pass an explicit voxel size: upstream's override mutates
        # integrator state (sticky), so relying on None here would make this
        # call return whatever resolution the *previous* request asked for.
        vs = msg.get("voxel_size")
        vs = float(vs) if vs is not None else float(cfg.esdf_voxel_size)
        grid = mapper.compute_esdf(esdf_origin=origin, esdf_voxel_size=vs)
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
        # Stash the builder kwargs next to the checkpoint so a load under a
        # different config fails loudly instead of pairing blocks with a
        # mismatched voxel size / extent.
        with open(_sidecar_path(path), "w") as f:
            json.dump({k: list(v) if isinstance(v, tuple) else v
                       for k, v in state["cfg_kwargs"].items()}, f)
        return {"ok": True, "path": path}

    if cmd == "load":
        path = os.path.join(BLOCKS_DIR, os.path.basename(msg["name"]))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found in mounted {BLOCKS_DIR}")

        sidecar = _sidecar_path(path)
        if os.path.isfile(sidecar) and not msg.get("force", False):
            with open(sidecar) as f:
                saved_kw = json.load(f)
            live_kw = {k: list(v) if isinstance(v, tuple) else v
                       for k, v in state["cfg_kwargs"].items()}
            if saved_kw != live_kw:
                raise ValueError(
                    f"checkpoint config {saved_kw} != live config {live_kw}; "
                    f'send {{"cmd": "configure", ...}} to match, or "force": true'
                )

        # import_blocks into the existing (reset) mapper instead of
        # Mapper.load_blocks: keeps the same mapper object (no re-JIT, no
        # integrator state reset to a different ESDF resolution), and returns
        # a block count we can assert on. import_weight=None preserves saved
        # per-voxel weights, which round-trip correctly. If a caller sets it,
        # the value must be STRICTLY greater than minimum_tsdf_weight -- the
        # observation gate is a strict comparison, and weight == threshold
        # marks every voxel unobserved (empirically: whole grid goes to the
        # far sentinel). Note the static analytic-primitive channel is not in
        # the checkpoint; re-stamp obstacles after load if you use it.
        iw = msg.get("import_weight")
        if iw is not None:
            iw = float(iw)
            if iw <= cfg.minimum_tsdf_weight:
                raise ValueError(
                    f"import_weight={iw} must be > minimum_tsdf_weight="
                    f"{cfg.minimum_tsdf_weight} (strict observation gate); "
                    f"omit it to preserve saved weights"
                )
        mapper.reset()
        n = mapper.import_blocks(path, import_weight=iw)
        if n <= 0:
            raise RuntimeError(f"import_blocks({path}) imported {n} blocks")
        return {"ok": True, "n_blocks": int(n),
                "memory_mb": mapper.memory_usage_mb()}

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