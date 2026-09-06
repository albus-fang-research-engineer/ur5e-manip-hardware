"""Isolate the CUDA fault seen on the first live-ESDF plan_constrained call.

CONCLUSION (see variant notes below): an RSC voxel query between the IK
solver's CUDA-graph capture and replay faults the replay. Fix applied:
CuroboIK(use_cuda_graph=False) by default (variant J: 84 ms / batch of 20).
MotionPlanner is unaffected (variant K). Kept as a regression harness.

Traceback pointed at IKSolver._get_result during a scene-aware IK solve on
Scene(voxel=[mapper.compute_esdf()]). The same grid passed through
RobotSceneCollision moments earlier without fault, and IK on a Cuboid scene
is fine in the tests. Variables to separate:

    A  IK, voxel scene, mapper NEVER integrated
    B  IK, voxel scene, mapper after ONE synthetic depth frame (a plane)
    C  IK, no scene at all (collision filter happens afterwards anyway)
    D  RobotSceneCollision on the never-integrated grid (control; expected ok)
    E  the EXACT plan_constrained command the bridge sent, via server.handle
       (max_batch = CuroboIK default 64, unlike A-C which used 32)
    F  same as E with CuroboIK max_batch forced to 32
    -- E and F fault, A-D pass: the trigger is RobotSceneCollision AND IK
       built on the SAME VoxelGrid object (direct-reference, no copy). --
    G  RSC + IK on the same grid, IK solve only (no collision query): confirms
       coexistence alone is the trigger
    H  RSC on the live grid, IK on grid.clone(): candidate fix 1 (scene-aware
       IK on a private copy, refreshed with update_world per call)
    I  RSC on the live grid, IK with scene=None: candidate fix 2
    -- G ok, H FAULT, I ok: the trigger is an RSC VOXEL query between the IK
       solver's CUDA-graph capture and a later replay; a cloned grid does not
       help, a cuboid scene never faults, no-scene IK never faults. --
    J  as H but IK built with use_cuda_graph=False: candidate fix 3
    K  MotionPlanner (server's own, cuda graph) plan -> RSC voxel query ->
       plan again: is the existing "plan" command exposed too?

Run with CUDA_LAUNCH_BLOCKING=1 so the report names the faulting kernel
rather than the next API call:

    docker compose run --rm -e CUDA_LAUNCH_BLOCKING=1 -v $PWD/test:/opt/test curobo \\
        bash -lc "python /opt/test/curobo_incontainer/repro_ik_live_esdf.py"

Each variant runs in its own subprocess: an illegal-instruction error poisons
the CUDA context, so they cannot share one.
"""

import os
import subprocess
import sys
import traceback

VARIANTS = ["J", "K"]


def child(variant: str):
    import numpy as np
    import torch

    sys.path.insert(0, "/opt/curobo_server")
    sys.path.insert(0, "/opt/robot_builder")
    import server  # noqa: E402  (mapper build + robot cfg helpers)
    from curobo.scene import Scene  # noqa: E402
    from cbirrt_backend import CuroboCollision, CuroboIK, dh_chain  # noqa: E402

    kw = server.default_cfg_kwargs()
    mapper, cfg = server.build_mapper(**kw)
    state = {"mapper": mapper, "cfg": cfg, "cfg_kwargs": kw}
    server._load_robot(state)
    robot_cfg = state["robot_cfg"]
    print(f"[{variant}] mapper {kw['extent']} @ {kw['voxel_size']} m, esdf voxel {cfg.esdf_voxel_size}")

    if variant == "B":
        # one synthetic frame: a plane 0.9 m in front of a camera looking along +z,
        # camera at the map/base origin looking +x (T_base_cam rotates z->x)
        h, w = kw["image_height"], kw["image_width"]
        depth = np.full((h, w), 0.9, np.float32)
        K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]], np.float32)
        T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.5], [0, 0, 0, 1]], np.float32)
        server.handle({"cmd": "integrate", "depth": depth, "intrinsics": K, "pose": T}, state)
        print(f"[{variant}] integrated one synthetic frame")

    grid = mapper.compute_esdf(esdf_voxel_size=float(cfg.esdf_voxel_size))
    ft = grid.feature_tensor
    print(f"[{variant}] esdf feature tensor {tuple(ft.shape)} {ft.dtype}: "
          f"min {float(ft.min()):.3f} max {float(ft.max()):.3f} "
          f"nan {int(torch.isnan(ft).sum())} inf {int(torch.isinf(ft).sum())}")
    print(f"[{variant}] grid dims {grid.dims} voxel {grid.voxel_size} pose {grid.pose}")
    torch.cuda.synchronize()

    q0 = np.array([-1.0, -1.2, 1.5, -1.9, -1.57, 0.0])
    if variant in ("E", "F"):
        import functools
        import plan_constrained as pc
        if variant == "F":
            pc.CuroboIK = functools.partial(CuroboIK, max_batch=32)
        msg = {"cmd": "plan_constrained", "q_start": q0.astype(np.float32),
               "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
               "T_ee_body": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.12, 0, 0, 0, 1],
               "subgoal": {"t0_w": [1, 0, 0, 0.35, 0, 1, 0, 0.55, 0, 0, 1, 0.45, 0, 0, 0, 1],
                           "tw_e": list(np.eye(4).ravel()),
                           "bw": [-.01, .01, -.01, .01, -.01, .01, -3.2, 3.2, -3.2, 3.2, -3.2, 3.2],
                           "name": "transport/subgoal"},
               "path": [], "n_goal_samples": 40, "timeout": 20.0, "eps": 0.1,
               "constraint_tol": 0.002, "clearance_margin": 0.0, "seed": 0, "attached_spheres": None}
        rep = server.handle(msg, state)
        torch.cuda.synchronize()
        f = rep.get("funnel", {})
        print(f"[{variant}] success={rep.get('success')} reason='{rep.get('reason')}' "
              f"funnel {f.get('n_sampled')}->{f.get('n_ik')}->{f.get('n_collision_free')}->{f.get('n_contained')} "
              f"waypoints {len(rep.get('positions', []))}")
        return
    if variant == "K":
        goal = np.array([[1, 0, 0, 0.35], [0, 1, 0, 0.55], [0, 0, 1, 0.45], [0, 0, 0, 1]], np.float32)
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        req = {"cmd": "plan", "q": q0.astype(np.float32), "joint_names": names, "T_base_goal": goal}
        r1 = server.handle(req, state); torch.cuda.synchronize()
        print(f"[{variant}] MotionPlanner plan #1: success={r1.get('success')} {r1.get('error', '')}")
        col = CuroboCollision(robot_cfg, Scene(voxel=[grid]))
        print(f"[{variant}] RSC voxel query: in_collision(q0) = {col.in_collision(q0)}"); torch.cuda.synchronize()
        r2 = server.handle(req, state); torch.cuda.synchronize()
        print(f"[{variant}] MotionPlanner plan #2 (after RSC query): success={r2.get('success')} {r2.get('error', '')}")
        return
    if variant == "D":
        col = CuroboCollision(robot_cfg, Scene(voxel=[grid]))
        print(f"[{variant}] collision on live grid: in_collision(q0) = {col.in_collision(q0)}, "
              f"pen = {col.signed_penetration(q0[None])[0][0]:+.4f}")
        torch.cuda.synchronize()
        return

    col = None
    if variant in ("G", "H", "I", "J"):
        col = CuroboCollision(robot_cfg, Scene(voxel=[grid]))       # the planner's oracle, live grid
        print(f"[{variant}] RSC built on the live grid")
    if variant in ("C", "I"):
        scene, label = None, "None"
    elif variant == "H":
        scene, label = Scene(voxel=[grid.clone()]), "voxel CLONE"
    elif variant == "J":
        scene, label = Scene(voxel=[grid]), "voxel (shared), NO cuda graph"
    else:
        scene, label = Scene(voxel=[grid]), "voxel (shared)"
    print(f"[{variant}] building IK (scene={label}) ...")
    ik = CuroboIK(robot_cfg, scene, max_batch=32, num_seeds=32, use_cuda_graph=(variant != "J"))
    if col is not None and variant != "G":
        print(f"[{variant}] collision query on live grid first: in_collision(q0) = {col.in_collision(q0)}")
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(f"[{variant}] IK built + warmed in {ik.warmup_time:.1f}s")
    ch = dh_chain()
    T = np.stack([ch.fk(q0 + np.array([d, 0, 0, 0, 0, 0])) for d in np.linspace(0, 2.0, 20)])
    res = ik.solve(T, q_seed_dh=q0)
    torch.cuda.synchronize()
    print(f"[{variant}] IK solved {res.success.sum()}/20 OK")
    if col is not None:
        print(f"[{variant}] collision after IK: in_collision(q0) = {col.in_collision(q0)}; "
              f"batch of solutions free: {(~col.in_collision_batch(res.q[res.success])).sum()}/{res.success.sum()}")
        torch.cuda.synchronize()
        if variant == "J":
            import time
            t0 = time.time()
            for _ in range(5):
                ik.solve(T, q_seed_dh=q0)
            torch.cuda.synchronize()
            print(f"[{variant}] no-graph IK steady state: {(time.time() - t0) / 5 * 1e3:.1f} ms per batch of 20")
        if variant == "H":
            # refresh the clone from the live grid the way the command would per call
            ik.update_world(Scene(voxel=[mapper.compute_esdf(esdf_voxel_size=float(cfg.esdf_voxel_size)).clone()]))
            res2 = ik.solve(T, q_seed_dh=q0); torch.cuda.synchronize()
            print(f"[{variant}] after update_world(clone): IK solved {res2.success.sum()}/20 OK")


def main():
    if len(sys.argv) > 1:
        try:
            child(sys.argv[1])
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING=os.environ.get("CUDA_LAUNCH_BLOCKING", "1"))
    for v in VARIANTS:
        print(f"\n===== variant {v} =====", flush=True)
        r = subprocess.run([sys.executable, __file__, v], env=env)
        print(f"===== variant {v}: {'OK' if r.returncode == 0 else 'FAILED'} =====", flush=True)


if __name__ == "__main__":
    main()
