"""Isolate the CUDA fault seen on the first live-ESDF plan_constrained call.

Traceback pointed at IKSolver._get_result during a scene-aware IK solve on
Scene(voxel=[mapper.compute_esdf()]). The same grid passed through
RobotSceneCollision moments earlier without fault, and IK on a Cuboid scene
is fine in the tests. Variables to separate:

    A  IK, voxel scene, mapper NEVER integrated
    B  IK, voxel scene, mapper after ONE synthetic depth frame (a plane)
    C  IK, no scene at all (collision filter happens afterwards anyway)
    D  RobotSceneCollision on the never-integrated grid (control; expected ok)

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

VARIANTS = ["D", "A", "B", "C"]


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
    if variant == "D":
        col = CuroboCollision(robot_cfg, Scene(voxel=[grid]))
        print(f"[{variant}] collision on live grid: in_collision(q0) = {col.in_collision(q0)}, "
              f"pen = {col.signed_penetration(q0[None])[0][0]:+.4f}")
        torch.cuda.synchronize()
        return

    scene = None if variant == "C" else Scene(voxel=[grid])
    print(f"[{variant}] building IK (scene={'voxel' if scene is not None else 'None'}) ...")
    ik = CuroboIK(robot_cfg, scene, max_batch=32, num_seeds=32)
    torch.cuda.synchronize()
    print(f"[{variant}] IK built + warmed in {ik.warmup_time:.1f}s")
    ch = dh_chain()
    T = np.stack([ch.fk(q0 + np.array([d, 0, 0, 0, 0, 0])) for d in np.linspace(0, 2.0, 20)])
    res = ik.solve(T, q_seed_dh=q0)
    torch.cuda.synchronize()
    print(f"[{variant}] IK solved {res.success.sum()}/20 OK")


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
