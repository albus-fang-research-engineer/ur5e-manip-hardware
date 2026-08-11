"""Diagnose (a) esdf_voxel_size override semantics, (b) save/load weight gating.
Synthetic scene: camera 1 m above a flat plane at z=0, looking straight down.
Free space above the plane, solid below."""
import inspect, tempfile, os
import numpy as np
import torch
from curobo.perception import Mapper, MapperCfg
from curobo.types import CameraObservation, Pose

H, W = 480, 848
CFG = dict(voxel_size=0.01, extent_meters_xyz=(2.0, 2.0, 1.5),
           truncation_distance=0.06, depth_maximum_distance=3.0,
           depth_minimum_distance=0.05, minimum_tsdf_weight=4.0,
           decay_factor=1.0, frustum_decay_factor=1.0, enable_static=True,
           num_cameras=1, image_height=H, image_width=W,
           block_size=8, color_grid_size=8)

def obs():
    depth = torch.full((1, H, W), 1.0, dtype=torch.float32, device="cuda")
    K = torch.tensor([[[600., 0, W/2], [0, 600., H/2], [0, 0, 1]]], device="cuda")
    T = torch.tensor([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 1.0],
                      [0, 0, 0, 1.]], device="cuda")  # cam at z=1, looking down
    rgb = torch.zeros((1, H, W, 3), dtype=torch.uint8, device="cuda")
    return CameraObservation(depth_image=depth, rgb_image=rgb,
                             pose=Pose.from_matrix(T), intrinsics=K)

def probe(grid, pts):
    shape, low, _ = grid.get_grid_shape()
    g = grid.feature_tensor.float().cpu().numpy().reshape(shape)
    low = np.asarray(low, np.float64); vs = float(grid.voxel_size)
    out = []
    for p in pts:
        idx = np.floor((np.asarray(p) - low) / vs).astype(int)
        out.append(float(g[tuple(idx)]) if np.all(idx >= 0) and np.all(idx < g.shape)
                   else float("nan"))
    return out

def report(tag, grid):
    shape, low, high = grid.get_grid_shape()
    low, high = np.asarray(low, float), np.asarray(high, float)
    implied = (high - low) / np.asarray(shape)
    d = probe(grid, [(0, 0, 0.12), (0, 0, -0.04)])
    e = grid.feature_tensor.float()
    print(f"{tag}: shape={list(shape)} vs={float(grid.voxel_size):.4f} "
          f"implied={np.round(implied,4)} span={np.round(high-low,3)}\n"
          f"    free(+12cm)={d[0]:+.4f} occ(-4cm)={d[1]:+.4f} "
          f"range=[{e.min():.4f},{e.max():.4f}]")

print("=== compute_esdf source ===")
print(inspect.getsource(Mapper.compute_esdf))
try:
    print(inspect.getsource(type(Mapper(MapperCfg(**CFG)).integrator).compute_esdf))
except Exception as ex:
    print(f"(integrator source: {ex})")

print("\n=== A) esdf override semantics (cfg defaults: esdf_voxel_size unset) ===")
m = Mapper(MapperCfg(**CFG))
o = obs()
m.integrate(camera_observation=o)
report("default ", m.compute_esdf())
report("vs=0.01 ", m.compute_esdf(esdf_voxel_size=0.01))
report("vs=0.02 ", m.compute_esdf(esdf_voxel_size=0.02))

print("\n=== A2) with esdf_voxel_size/extent set explicitly in cfg ===")
m2 = Mapper(MapperCfg(**CFG, esdf_voxel_size=0.01, extent_esdf_meters_xyz=(2.0, 2.0, 1.5)))
m2.integrate(camera_observation=o)
report("cfg=0.01", m2.compute_esdf())
report("ovr=0.02", m2.compute_esdf(esdf_voxel_size=0.02))

print("\n=== B) save/load, 1 frame vs 5 frames, import_weight None vs min_weight ===")
path = os.path.join(tempfile.gettempdir(), "diag_blocks.pt")
for n_frames in (1, 5):
    mm = Mapper(MapperCfg(**CFG))
    for _ in range(n_frames):
        mm.integrate(camera_observation=obs())
    report(f"live n={n_frames} ", mm.compute_esdf())
    mm.save_blocks(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if torch.is_tensor(v) and v.is_floating_point() and v.numel():
                print(f"    ckpt[{k}]: shape={list(v.shape)} dtype={v.dtype} "
                      f"min={v.min():.3f} max={v.max():.3f} mean={v.float().mean():.3f}")
            else:
                print(f"    ckpt[{k}]: {type(v).__name__}")
    for iw in (None, CFG["minimum_tsdf_weight"]):
        ld = Mapper.load_blocks(path, MapperCfg(**CFG), import_weight=iw)
        report(f"  load n={n_frames} iw={iw} ", ld.compute_esdf())