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
   "origin": [x, y, z],               # optional override (sliding window)
   "sparse_below": 0.04,              # optional: return only voxels with
                                      #   esdf <= this (indices + values)
                                      #   instead of the dense tensor
   "slice_z": 0.10}                   # optional: also return the dense 2D
                                      #   layer nearest this z (map frame)
      -> {"ok": True,
          "esdf": float32 tensor,     # signed distance (m); reshape w/ grid_shape
          "grid_shape": [nx, ny, nz], # X slowest, Z fastest (VoxelGrid conv.)
          "low": [x,y,z], "high": [x,y,z],
          "voxel_size": float, "dims": [x,y,z]}

  {"cmd": "load_robot",
   "seg_threshold": 0.06,             # optional; env CUROBO_SEG_THRESHOLD
   "force_rebuild": false}            # re-fit spheres even if the yml exists
      -> {"ok": True, "joint_names": [...], "n_spheres": int, "yml": str}

  Robot self-masking: once load_robot has run, an "integrate" request may
  carry the arm state and the robot is REMOVED from the depth before TSDF
  integration (RobotSegmenter: FK spheres vs. backprojected pixels):

  {"cmd": "integrate", ...as above...,
   "q": [6] float,                    # joint positions
   "joint_names": [...],              # names for q; reordered to kinematics
   "T_base_cam": 4x4 float32}         # camera in ROBOT BASE frame; omit to
                                      #   reuse "pose" (map frame == base)
      -> {"ok": True, "n_masked": int}

  {"cmd": "robot_mask",               # debug: mask only, no integration
   "depth": HxW float32 (m), "intrinsics": 3x3,
   "T_base_cam": 4x4, "q": [...], "joint_names": [...]}
      -> {"ok": True, "mask": HxW bool, "n_masked": int}

  {"cmd": "plan",                     # MotionPlanner vs. the LIVE ESDF
   "q": [6] float, "joint_names": [...],
   "T_base_goal": 4x4 float32}        # tool0 goal in base frame
      -> {"ok": True, "success": bool,
          "joint_names": [...],
          "positions": Nxdof f32, "velocities": Nxdof f32,
          "dt": float,               # interpolation dt of the trajectory
          "ee_path": Nx3 f32,        # tool positions along the trajectory
          "position_error": float, "rotation_error": float,
          "solve_time": float}

  {"cmd": "plan_constrained",         # CBiRRT over a TSR pair (manip_cbirrt)
   "q_start": [6], "T_ee_body": 4x4,   #   vs. the LIVE ESDF; see
   "subgoal": TSR, "path": [TSR,...],  #   plan_constrained.py for the full
   ...}                                #   request/reply and the goal funnel
      -> {"ok": True, "success": bool, "reason": str, "positions": Nx6,
          "ee_path": Nx3, "body_path": Nx3, "max_excess": float,
          "funnel": {...}, "tree_sizes": [a, b], ...}

  FRAME CONTRACT for masking + planning: the map frame IS the robot base
  frame (base_link at the map origin). The planner plans in base frame and
  reads the mapper's VoxelGrid coordinates as-is; the segmenter needs the
  camera in base frame. Send "pose" = T_base_cam on integrate and the same
  matrix serves the mapper, the segmenter, and the planner. If your map
  frame is elsewhere, pass "T_base_cam" separately for masking -- but the
  planner will still treat map coords as base coords, so don't plan in that
  configuration. NOTE: enable masking from the FIRST integrated frame (or
  send "reset" after load_robot) -- arm surfaces already baked into the TSDF
  are not retroactively removed.

  The MotionPlanner is built lazily on the first "plan" (against the current
  ESDF grid buffer -- later compute_esdf calls refresh it in place, the
  upstream live-mapping pattern) and pays IK/trajopt warmup + CUDA-graph
  capture once. "configure" invalidates the planner (grid buffer is
  reallocated); the robot/segmenter survive configure but assume a fixed
  depth image shape once used.

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

# Robot config: a prebuilt yml, or the ur5e_curobo_config builder module dir
# (compose mounts test/curobo_incontainer there). The builder writes the yml
# into ROBOT_YML's directory via CUROBO_TEST_CACHE, so the sphere fit runs
# once and persists across restarts.
ROBOT_YML = os.environ.get("CUROBO_ROBOT_YML", "/data/robot/ur5e.yml")
ROBOT_BUILDER_DIR = os.environ.get("CUROBO_ROBOT_BUILDER_DIR", "/opt/robot_builder")
SEG_THRESHOLD = float(os.environ.get("CUROBO_SEG_THRESHOLD", "0.06"))
PLAN_CUDA_GRAPH = os.environ.get("CUROBO_PLAN_CUDA_GRAPH", "1") == "1"

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


# ------------------------------------------------------------ robot / planner
def _robot_config(force_rebuild=False):
    """Prebuilt yml if present, else run the ur5e_curobo_config builder
    (single source of truth, lives in test/curobo_incontainer) with its cache
    pointed at ROBOT_YML's directory."""
    from curobo._src.util_file import load_yaml

    if os.path.isfile(ROBOT_YML) and not force_rebuild:
        return load_yaml(ROBOT_YML)

    import sys
    if ROBOT_BUILDER_DIR not in sys.path:
        sys.path.insert(0, ROBOT_BUILDER_DIR)
    os.environ["CUROBO_TEST_CACHE"] = os.path.dirname(ROBOT_YML)
    try:
        from ur5e_curobo_config import build_ur5e_config
    except ImportError as e:
        raise RuntimeError(
            f"no robot yml at {ROBOT_YML} and ur5e_curobo_config not importable "
            f"from {ROBOT_BUILDER_DIR} (mount test/curobo_incontainer there): {e}"
        )
    log.info("fitting UR5e collision spheres (one-time, cached to %s)...",
             ROBOT_YML)
    return build_ur5e_config(force=force_rebuild)


def _load_robot(state, seg_threshold=None, force_rebuild=False):
    from curobo.perception import RobotSegmenter

    cfg = _robot_config(force_rebuild)
    seg = RobotSegmenter.from_robot_file(
        cfg,
        distance_threshold=float(seg_threshold if seg_threshold is not None
                                 else SEG_THRESHOLD),
        use_cuda_graph=True,   # streaming path: fixed depth shape per session
    )
    # cuRobo main regression: default ops_dtype=bfloat16 fails its own
    # fp16/fp32 tensor check during graph-capture warmup, and
    # from_robot_file doesn't forward ops_dtype. Force fp32.
    seg._ops_dtype = torch.float32
    state["robot_cfg"] = cfg
    state["segmenter"] = seg
    n = sum(len(v) for v in
            cfg["kinematics"].get("collision_spheres", {}).values())
    return {"ok": True, "joint_names": list(seg._kinematics.joint_names),
            "n_spheres": n, "yml": ROBOT_YML}


def _joint_state(q, joint_names, target_names):
    from curobo.types import JointState

    name_to_q = dict(zip(list(joint_names), [float(x) for x in np.asarray(q).ravel()]))
    missing = [n for n in target_names if n not in name_to_q]
    if missing:
        raise ValueError(f"joint state lacks {missing}; has {list(name_to_q)}")
    t = torch.tensor([name_to_q[n] for n in target_names],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    return JointState.from_position(t, joint_names=list(target_names))


def _robot_mask(state, depth_np, K_np, T_base_cam_np, q, joint_names):
    """(mask bool HxW on GPU, filtered depth 1xHxW on GPU)."""
    from curobo.types import CameraObservation, Pose

    seg = state["segmenter"]
    depth = torch.as_tensor(np.ascontiguousarray(depth_np),
                            dtype=torch.float32, device="cuda")
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    K = torch.as_tensor(np.ascontiguousarray(K_np), dtype=torch.float32,
                        device="cuda").reshape(-1, 3, 3)
    T = torch.as_tensor(np.ascontiguousarray(T_base_cam_np),
                        dtype=torch.float32, device="cuda").reshape(4, 4)
    cam = CameraObservation(depth_image=depth, intrinsics=K,
                            pose=Pose.from_matrix(T), depth_to_meter=1.0)
    js = _joint_state(q, joint_names, seg._kinematics.joint_names)
    mask, filtered = seg.get_robot_mask(cam, js)
    return mask, filtered


def _get_planner(state):
    """Build MotionPlanner once against the mapper's live ESDF VoxelGrid.
    compute_esdf() writes into the same buffer afterwards (upstream
    live_volumetric_mapping pattern), so refreshing the world is just a
    compute_esdf before each plan."""
    if state.get("planner") is not None:
        return state["planner"]

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene

    if state.get("robot_cfg") is None:
        _load_robot(state)
    mapper, cfg = state["mapper"], state["cfg"]
    grid = mapper.compute_esdf(esdf_voxel_size=float(cfg.esdf_voxel_size))

    log.info("building MotionPlanner (IK/trajopt warmup + graph capture)...")
    pcfg = MotionPlannerCfg.create(
        robot=state["robot_cfg"],
        scene_model=Scene(voxel=[grid]),
        use_cuda_graph=PLAN_CUDA_GRAPH,
    )
    planner = MotionPlanner(pcfg)
    planner.warmup()
    state["planner"] = planner
    log.info("planner ready: tool=%s dof=%d",
             planner.tool_frames, planner.action_dim)
    return planner


def _plan(state, msg):
    from curobo.types import GoalToolPose, Pose

    planner = _get_planner(state)
    mapper, cfg = state["mapper"], state["cfg"]
    # refresh the aliased ESDF buffer to the latest integrated scene, at the
    # resolution the planner captured (defeat any sticky viz-time rescale)
    mapper.compute_esdf(esdf_voxel_size=float(cfg.esdf_voxel_size))

    js = _joint_state(msg["q"], msg["joint_names"], planner.joint_names)
    T = np.asarray(msg["T_base_goal"], np.float32).reshape(4, 4)
    g = Pose.from_matrix(torch.as_tensor(T, dtype=torch.float32,
                                         device="cuda"))
    goal = Pose(position=g.position.view(1, 3),
                quaternion=g.quaternion.view(1, 4))
    tool = planner.tool_frames[0]
    goal_tool = GoalToolPose.from_poses(
        {tool: goal}, ordered_tool_frames=planner.tool_frames, num_goalset=1)

    res = planner.plan_pose(goal_tool, js)
    ok = bool(res is not None and res.success is not None
              and bool(res.success.any()))
    if not ok:
        return {"ok": True, "success": False,
                "error": "planning failed (IK or trajopt unsuccessful)"}

    traj = res.interpolated_trajectory
    pos = traj.position
    vel = traj.velocity if traj.velocity is not None else torch.zeros_like(pos)
    # squeeze any leading batch/seed dims down to [steps, dof]
    while pos.ndim > 2:
        pos, vel = pos[0], vel[0]
    if res.interpolated_last_tstep is not None:
        last = int(res.interpolated_last_tstep.ravel()[0])
        if last > 0:
            pos, vel = pos[:last], vel[:last]

    # tool positions along the trajectory, for RViz path display
    from curobo.types import JointState as _JS
    traj_js = _JS.from_position(pos.contiguous(),
                                joint_names=list(planner.joint_names))
    kin = planner.compute_kinematics(traj_js)
    ee = kin.tool_poses.get_link_pose(tool).position.reshape(-1, 3)

    def f(t):
        return None if t is None else float(torch.as_tensor(t).ravel()[0])

    return {
        "ok": True, "success": True,
        "joint_names": list(planner.joint_names),
        "positions": pos.detach().cpu().numpy().astype(np.float32),
        "velocities": vel.detach().cpu().numpy().astype(np.float32),
        "dt": float(planner.config.trajopt_solver_config.interpolation_dt),
        "ee_path": ee.detach().cpu().numpy().astype(np.float32),
        "position_error": f(res.position_error),
        "rotation_error": f(res.rotation_error),
        "solve_time": float(res.solve_time or 0.0),
    }


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
        # the planner (and the cbirrt oracles) hold an alias of the OLD
        # mapper's ESDF buffer
        state["planner"] = None
        state["cbirrt"] = None
        return {"ok": True, "memory_mb": state["mapper"].memory_usage_mb()}

    if cmd == "integrate":
        n_masked = None
        if msg.get("q") is not None:
            if state.get("segmenter") is None:
                _load_robot(state)
            T_bc = msg.get("T_base_cam", msg["pose"])  # map==base default
            mask, filtered = _robot_mask(
                state, msg["depth"], msg["intrinsics"], T_bc,
                msg["q"], msg["joint_names"])
            n_masked = int(mask.sum().item())
            msg = dict(msg)
            msg["depth"] = filtered.squeeze(0).detach().cpu().numpy()
        mapper.integrate(camera_observation=make_observation(msg, cfg))
        rep = {"ok": True}
        if n_masked is not None:
            rep["n_masked"] = n_masked
        return rep

    if cmd == "load_robot":
        return _load_robot(state, msg.get("seg_threshold"),
                           msg.get("force_rebuild", False))

    if cmd == "robot_mask":
        if state.get("segmenter") is None:
            _load_robot(state)
        mask, _ = _robot_mask(state, msg["depth"], msg["intrinsics"],
                              msg["T_base_cam"], msg["q"], msg["joint_names"])
        m = mask.squeeze(0).detach().cpu().numpy().astype(bool)
        return {"ok": True, "mask": m, "n_masked": int(m.sum())}

    if cmd == "plan":
        return _plan(state, msg)

    if cmd == "plan_constrained":
        import plan_constrained as _pc      # sidecar dir is the CWD / on sys.path
        if state.get("robot_cfg") is None:
            _load_robot(state)
        return _pc.handle(state, msg)

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
        rep = {
            "ok": True,
            "grid_shape": grid_shape,
            "low": low,
            "high": high,
            "voxel_size": float(grid.voxel_size),
            "dims": [float(d) for d in grid.dims],
        }
        f = grid.feature_tensor.float().reshape(grid_shape)
        thr = msg.get("sparse_below")
        if thr is not None:
            sel = f <= float(thr)
            idx = sel.nonzero().to(torch.int32)
            rep["sparse_idx"] = idx.cpu().numpy()
            rep["sparse_vals"] = f[sel].cpu().numpy().astype(np.float32)
        z = msg.get("slice_z")
        if z is not None:
            gvs = float(grid.voxel_size)
            k = int(np.clip(round((float(z) - low[2]) / gvs - 0.5),
                            0, grid_shape[2] - 1))
            rep["slice"] = f[:, :, k].cpu().numpy().astype(np.float32)
            rep["slice_k"] = k
        if thr is None and z is None:
            rep["esdf"] = grid.feature_tensor.float().cpu().numpy()
        return rep

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