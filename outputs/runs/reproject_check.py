#!/usr/bin/env python3
"""Project a tracked mesh through its cam_T_obj onto the saved run_scene frame
and compare with (a) the SAM mask and (b) the measured depth.

(a) silhouette IoU / centroid offset: evidence that the mesh and the pose are
    in the SAME FRAME (a centring or scaling mismatch between the exported
    mesh and the returned pose shows up as a centroid offset of tens of px).
(b) front-surface depth residual: evidence about SCALE. A silhouette cannot
    separate "right size at the right depth" from "10% too big, 10% too far";
    the depth image can. For every mask pixel that received projected
    vertices, the nearest projected z is compared with depth_mm.png. Right
    pose+scale -> small, structureless residual. Scale error -> residual
    that grows toward the silhouette edge (see the heatmap).

Neither replaces a ruler on the real object for the extents themselves.

  python3 reproject_check.py outputs/runs/20260903_203531 \\
      --mesh any6d_runtime/outputs/final_mesh_mug.obj --object mug \\
      --pose-json outputs/runs/20260903_203531/any6d_mug.json
  # or, as before:  --t X Y Z --rpy R P Y   (xyz Euler, degrees)

Writes <run>/reproject_<object>.png (mask contour green, projected vertices
red) and <run>/reproject_<object>_depth.png (signed residual heatmap), and
prints the numbers.
"""
import argparse, json, os, sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_dilation, binary_erosion

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir")
ap.add_argument("--mesh", required=True)
ap.add_argument("--object", default="mug")
ap.add_argument("--pose-json", default=None,
                help='JSON with "cam_T_obj" (4x4 list) -- e.g. written by any6d_from_mesh.py')
ap.add_argument("--t", type=float, nargs=3, metavar=("X", "Y", "Z"))
ap.add_argument("--rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                help="xyz Euler in degrees, as any6d_from_mesh.py prints it")
a = ap.parse_args()

T = np.eye(4)
if a.pose_json:
    T = np.asarray(json.load(open(a.pose_json))["cam_T_obj"], dtype=np.float64).reshape(4, 4)
elif a.t is not None and a.rpy is not None:
    T[:3, :3] = Rotation.from_euler("xyz", a.rpy, degrees=True).as_matrix()
    T[:3, 3] = a.t
else:
    ap.error("--pose-json or --t/--rpy")

rgb = np.array(Image.open(os.path.join(a.run_dir, "rgb.png")).convert("RGB"))[..., ::-1].copy()  # BGR on disk
mask = np.array(Image.open(os.path.join(a.run_dir, f"mask_{a.object.replace(' ', '_')}.png")).convert("L")) > 0
depth = np.array(Image.open(os.path.join(a.run_dir, "depth_mm.png")))
if depth.dtype != np.uint16:
    sys.exit(f"depth_mm.png decoded as {depth.dtype}, expected uint16")
depth = depth.astype(np.float64) * 1e-3
with open(os.path.join(a.run_dir, "summary.json")) as f:
    K = np.asarray(json.load(f)["frame"]["K"], dtype=np.float64)

V = trimesh.load(a.mesh, force="mesh").vertices
Pc = V @ T[:3, :3].T + T[:3, 3]
if (Pc[:, 2] <= 0).any():
    sys.exit("some vertices project behind the camera -- pose is not cam_T_obj")
uv = Pc @ K.T
uv = uv[:, :2] / uv[:, 2:3]
H, W = mask.shape
inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
ui = np.round(uv[inside]).astype(int)
zi = Pc[inside, 2]

# ---- (a) silhouette -------------------------------------------------------
proj = np.zeros_like(mask)
proj[ui[:, 1], ui[:, 0]] = True
proj = binary_dilation(proj, iterations=3)
proj = binary_erosion(proj, iterations=2)
inter = (proj & mask).sum(); union = (proj | mask).sum()
frac_in = mask[ui[:, 1], ui[:, 0]].mean()
cm_p = np.argwhere(proj).mean(0)[::-1]; cm_m = np.argwhere(mask).mean(0)[::-1]
print(f"vertices in frame        {inside.mean()*100:.1f}%")
print(f"projected verts on mask  {frac_in*100:.1f}%")
print(f"silhouette IoU           {inter/union:.3f}")
print(f"centroid offset (px)     {np.linalg.norm(cm_p - cm_m):.1f}   proj {cm_p.round(1)}  mask {cm_m.round(1)}")

# ---- (b) front-surface depth: nearest projected z per pixel vs measured ----
zbuf = np.full((H, W), np.inf)
np.minimum.at(zbuf, (ui[:, 1], ui[:, 0]), zi)            # z-buffer: nearest vertex wins
valid = np.isfinite(zbuf) & mask & (depth > 0)
resid = np.full((H, W), np.nan)
resid[valid] = zbuf[valid] - depth[valid]                # +: mesh surface behind measured depth
r = resid[valid] * 1e3
print(f"depth residual (mesh - measured), {valid.sum()} px:")
print(f"  median {np.median(r):+.1f} mm   p10/p90 {np.percentile(r, 10):+.1f} / {np.percentile(r, 90):+.1f} mm"
      f"   |median| < ~5 mm and a narrow spread = pose+scale consistent with the depth image")
# scale diagnostic: residual vs distance from silhouette centre (in px). A pure
# scale error tilts this; a pure depth offset does not.
d_px = np.linalg.norm(np.argwhere(valid)[:, ::-1] - cm_m, axis=1)
if len(d_px) > 50:
    slope = np.polyfit(d_px, r, 1)[0]
    print(f"  residual trend vs radius  {slope*100:+.2f} mm per 100 px  "
          f"(|trend| well above the noise = suspect scale, not pose)")

im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
edge = mask & ~binary_erosion(mask, iterations=1)
for y, x in np.argwhere(edge): d.point((x, y), fill=(0, 255, 0))
for x, y in ui[::4]: d.point((x, y), fill=(255, 0, 0))
out = os.path.join(a.run_dir, f"reproject_{a.object}.png"); im.save(out)

heat = np.zeros((H, W, 3), np.uint8) + 30
lim = 20.0  # mm colour range
rr = np.clip(resid * 1e3, -lim, lim) / lim
pos, neg = np.nan_to_num(np.clip(rr, 0, 1)), np.nan_to_num(np.clip(-rr, 0, 1))
heat[..., 0] = np.where(valid, 30 + 225 * pos, 30)       # red   = mesh behind measured depth
heat[..., 2] = np.where(valid, 30 + 225 * neg, 30)       # blue  = mesh in front
heat[..., 1] = np.where(valid, 30 + 225 * (1 - np.abs(np.nan_to_num(rr))), 30)   # green = agreement
Image.fromarray(heat).save(os.path.join(a.run_dir, f"reproject_{a.object}_depth.png"))
print("wrote", out, "and", os.path.join(a.run_dir, f"reproject_{a.object}_depth.png"))
