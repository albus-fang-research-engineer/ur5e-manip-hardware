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
ap.add_argument("--pre-rotate", nargs=2, metavar=("AXIS", "DEG"), default=None,
                help="rotate the mesh before applying the pose. AXIS = body (about the FOUND symmetry "
                     "axis through the body centre -- the yaw test) or x|y|z (about a mesh coordinate "
                     "axis through the origin)")
ap.add_argument("--axis", type=float, nargs=3, metavar=("DX", "DY", "DZ"), default=None,
                help="force the symmetry-axis direction (mesh frame); default: searched over the "
                     "hemisphere for the direction with the thinnest trimmed wall ring")
ap.add_argument("--axis-search", type=int, default=800, metavar="N",
                help="number of hemisphere directions in the coarse axis search (then refined locally)")
ap.add_argument("--yaw-sweep", type=float, default=10.0, metavar="STEP_DEG",
                help="sweep yaw about the body axis in this step and report the best-fitting "
                     "yaw offset (0 disables)")
a = ap.parse_args()

T = np.eye(4)
if a.pose_json:
    T = np.asarray(json.load(open(a.pose_json))["cam_T_obj"], dtype=np.float64).reshape(4, 4)
elif a.t is not None and a.rpy is not None:
    T[:3, :3] = Rotation.from_euler("xyz", a.rpy, degrees=True).as_matrix()
    T[:3, 3] = a.t
else:
    ap.error("--pose-json or --t/--rpy")

rgb = np.array(Image.open(os.path.join(a.run_dir, "rgb.png")).convert("RGB")).copy()  # run_scene writes a normal RGB PNG (cv2.imwrite of a BGR view); PIL reads it correctly -- no flip
mask = np.array(Image.open(os.path.join(a.run_dir, f"mask_{a.object.replace(' ', '_')}.png")).convert("L")) > 0
depth = np.array(Image.open(os.path.join(a.run_dir, "depth_mm.png")))
if depth.dtype != np.uint16:
    sys.exit(f"depth_mm.png decoded as {depth.dtype}, expected uint16")
depth = depth.astype(np.float64) * 1e-3
with open(os.path.join(a.run_dir, "summary.json")) as f:
    K = np.asarray(json.load(f)["frame"]["K"], dtype=np.float64)

_m = trimesh.load(a.mesh, force="mesh")
V0, FACES = np.asarray(_m.vertices), np.asarray(_m.faces)

# area-weighted surface samples for the axis statistics: vertex density is not
# uniform on a reconstructed mesh, and rims/handles are usually oversampled
S0 = np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(V0, FACES, process=False), 60000, seed=0)[0])

N_SLICES = 8

def ring_fit(d, pts=None):
    """Trimmed-body fit about unit direction d, SLICE-WISE along the axis.

    Pooling the whole height into one ring conflates two things: a tilted
    axis smears the ring, but so does a TAPERED body about its TRUE axis (rim
    radius != base radius -> pooled radial spread ~10% either way; that is why
    the coordinate-axis table read ~11% about all three axes). Per slice, a
    tapered mug is still a thin circle about its true axis and a smeared
    ellipse about any other direction, so slicing separates the two.

    Returns: 2D basis B, common in-plane body centre c, wall radius R (median
    over slices), per-point radial distance r and r_norm = r / (that point's
    slice wall radius), spread = median over slices of wall std / slice R,
    beyond = fraction with r_norm > 1.2, extent along d."""
    d = d / np.linalg.norm(d)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    B = np.c_[e1, e2]
    X = S0 if pts is None else pts
    P = X @ B
    h = X @ d
    c = P.mean(0)
    for _ in range(3):
        rr = np.linalg.norm(P - c, axis=1)
        Rw = np.percentile(rr, 60)
        c = P[rr < 1.2 * Rw].mean(0)
    rr = np.linalg.norm(P - c, axis=1)
    edges = np.quantile(h, np.linspace(0, 1, N_SLICES + 1))
    R_slice = np.full(len(X), np.nan)
    spreads, Rs = [], []
    for i in range(N_SLICES):
        m = (h >= edges[i]) & (h <= edges[i + 1])
        if m.sum() < 50:
            continue
        Ri = np.percentile(rr[m], 60)
        wall = rr[m][(rr[m] > 0.8 * Ri) & (rr[m] < 1.2 * Ri)]
        if len(wall) < 20:
            continue
        spreads.append(wall.std() / Ri); Rs.append(Ri); R_slice[m] = Ri
    R_slice = np.where(np.isnan(R_slice), np.nanmedian(R_slice), R_slice)
    r_norm = rr / R_slice
    return dict(d=d, B=B, P=P, c=c, R=float(np.median(Rs)), r=rr, r_norm=r_norm,
                spread=float(np.median(spreads)), beyond=float((r_norm > 1.2).mean()),
                extent=np.ptp(h))

def fib_hemisphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]

up_scores = -(T[:3, :3].T)[:, 1]                          # dot(mesh axis_k, -y_cam)
print("axis test (mesh geometry), coordinate axes for reference:")
for kk in range(3):
    f = ring_fit(np.eye(3)[kk])
    print(f"  about {'xyz'[kk]}: wall radius {f['R']*1e3:5.1f} mm  ring spread {f['spread']*100:5.1f}% of R  "
          f"beyond 1.2R {f['beyond']*100:5.1f}%  extent {f['extent']*1e3:5.1f} mm  cam-up cos {up_scores[kk]:+.2f}")

if a.axis is not None:
    fb = ring_fit(np.asarray(a.axis, float)); how = "FORCED by --axis"
else:
    # coarse hemisphere search, then local refinement around the best direction
    D = fib_hemisphere(a.axis_search)
    sc = np.array([ring_fit(d)["spread"] for d in D])
    d0 = D[sc.argmin()]
    for step in (0.08, 0.03, 0.01):
        cand = d0 + step * np.random.default_rng(0).normal(size=(60, 3))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        cand = np.vstack([d0, cand])
        scc = np.array([ring_fit(d)["spread"] for d in cand])
        d0 = cand[scc.argmin()]
    fb = ring_fit(d0); how = f"searched over {a.axis_search} directions + local refine"
d_axis = fb["d"]
d_cam = T[:3, :3] @ d_axis                                 # axis direction in the camera frame
if d_cam[1] > 0: d_axis, d_cam = -d_axis, -d_cam           # orient so it points "up" in the image
tilt = np.degrees(np.arccos(np.clip(np.abs(d_axis), 0, 1)))
print(f"symmetry axis (mesh)     [{d_axis[0]:+.3f} {d_axis[1]:+.3f} {d_axis[2]:+.3f}]  ({how})")
print(f"  ring spread {fb['spread']*100:.1f}% of R  beyond 1.2R {fb['beyond']*100:.1f}%  wall radius {fb['R']*1e3:.1f} mm  "
      f"extent along axis {fb['extent']*1e3:.1f} mm")
print(f"  angle to mesh x/y/z      {tilt[0]:.0f} / {tilt[1]:.0f} / {tilt[2]:.0f} deg   "
      f"(the canonical frame is tilted; a coordinate-axis sweep was rotating about the wrong line)")
print(f"  cam-up cos               {-d_cam[1]:+.2f}   (world-up would be ~cos(camera pitch))")
if fb["spread"] > 0.08:
    print("  <- even the best direction is not a thin ring PER SLICE: the body itself is not round "
          "(elliptical reconstruction or per-axis scale); handle detection below is unreliable")

B, c_b, R = fb["B"], fb["c"], fb["R"]
fv = ring_fit(d_axis, pts=V0)                              # same fit, evaluated on the vertices
P2 = V0 @ B                                                # vertices in the plane normal to the axis
r = fv["r"]
body = fv["r_norm"] < 1.2                                  # per-slice wall radius: taper-safe
handle = fv["r_norm"] > 1.2
ev = np.sort(np.linalg.eigvalsh(np.cov(P2[body].T)))[::-1]
aspect = float(np.sqrt(ev[0] / ev[1]))
print(f"body cross-section       aspect {aspect:.2f} (1.00 = round) about the found axis")
if handle.sum() > 0.01 * len(V0):
    dh2 = (P2[handle].mean(0) - c_b); dh2 /= np.linalg.norm(dh2)
    dh = B @ dh2                                           # handle direction in mesh frame
    print(f"handle in mesh           {handle.mean()*100:.1f}% of vertices, direction "
          f"[{dh[0]:+.2f} {dh[1]:+.2f} {dh[2]:+.2f}] (mesh frame), "
          f"max reach {r[handle].max()*1e3:.0f} mm vs wall radius {R*1e3:.0f} mm")
else:
    dh = None
    print(f"handle in mesh           NOT FOUND (<1% of vertices beyond 1.2 x wall radius {R*1e3:.0f} mm)")

# 3D body-axis point: in-plane centre c_b plus the mean along-axis coordinate of body vertices
c3 = B @ c_b + d_axis * float(np.mean((V0 @ d_axis)[body]))

def yaw_about_body(V, deg):
    """rotate about the FOUND symmetry axis through the body centre"""
    Rm = Rotation.from_rotvec(d_axis * np.radians(deg)).as_matrix()
    return (V - c3) @ Rm.T + c3

V = V0
if a.pre_rotate:
    ax, deg = a.pre_rotate[0].lower(), float(a.pre_rotate[1])
    if ax == "body":
        V = yaw_about_body(V, deg)
        print(f"pre-rotated mesh {deg:g} deg about its symmetry axis through the body centre")
    else:
        V = V @ Rotation.from_euler(ax, deg, degrees=True).as_matrix().T
        print(f"pre-rotated mesh {deg:g} deg about mesh {ax} through the origin")
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
# rasterise the FACES (PIL, ~1 s for 200k faces), not the vertices: a dot
# cloud with a morphological close reads as holes wherever triangles are
# larger than the closing radius, and that corrupts IoU for sparse meshes
def silhouette(uv_all):
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    pts = np.clip(uv_all[FACES].reshape(-1, 6), -1e6, 1e6)
    for row in pts.tolist():
        d.polygon(row, fill=1)
    return np.array(img, dtype=bool)
proj = silhouette(uv)
inter = (proj & mask).sum(); union = (proj | mask).sum()
frac_in = mask[ui[:, 1], ui[:, 0]].mean()
cm_p = np.argwhere(proj).mean(0)[::-1]; cm_m = np.argwhere(mask).mean(0)[::-1]
print(f"vertices in frame        {inside.mean()*100:.1f}%")
print(f"projected verts on mask  {frac_in*100:.1f}%")
prec, rec = inter / proj.sum(), inter / mask.sum()
print(f"silhouette IoU           {inter/union:.3f}   precision {prec:.3f}  recall {rec:.3f}"
      + ("   <- part of the mask is uncovered (a handle hidden behind the body?)"
         if rec < 0.93 and prec - rec > 0.04 else ""))
print(f"centroid offset (px)     {np.linalg.norm(cm_p - cm_m):.1f}   proj {cm_p.round(1)}  mask {cm_m.round(1)}")

# ---- where did the mesh's handle go? ---------------------------------------
zbuf = np.full((H, W), np.inf)
np.minimum.at(zbuf, (ui[:, 1], ui[:, 0]), zi)            # z-buffer: nearest vertex wins
if dh is not None:
    hi = handle[inside]
    hu = ui[hi]; hz = zi[hi]
    on = mask[hu[:, 1], hu[:, 0]].mean()
    occl = (hz > zbuf[hu[:, 1], hu[:, 0]] + 0.005).mean()
    hc = hu.mean(0)
    print(f"mesh handle projects to  ({hc[0]:.0f}, {hc[1]:.0f}) px, {on*100:.0f}% on mask, "
          f"{occl*100:.0f}% occluded by the body  "
          f"{'-> HIDDEN BEHIND THE BODY: yaw is wrong' if occl > 0.5 else ''}")

def sil_scores(Vr, align=True):
    """IoU/recall of the projected silhouette. With align=True the silhouette
    is first shifted so its centroid matches the mask centroid: a few degrees
    of axis error or a few mm of body-centre error displaces the rotated body
    by ~2*delta*sin(yaw/2) in the image, which would swamp the handle signal;
    yaw is about SHAPE agreement, so score shape."""
    Pr = Vr @ T[:3, :3].T + T[:3, 3]
    if (Pr[:, 2] <= 0).any():
        return 0.0, 0.0, 0.0
    u = Pr @ K.T; u = u[:, :2] / u[:, 2:3]
    pr = silhouette(u)
    shift = 0.0
    if align and pr.any():
        dxy = np.round(cm_m - np.argwhere(pr).mean(0)[::-1]).astype(int)
        shift = float(np.linalg.norm(dxy))
        pr = np.roll(pr, (dxy[1], dxy[0]), axis=(0, 1))
    i = (pr & mask).sum()
    return i / (pr | mask).sum(), i / mask.sum(), shift

if a.yaw_sweep > 0:
    angs = np.arange(0, 360, a.yaw_sweep)
    print(f"yaw sweep: {len(angs)} silhouettes of {len(FACES)} faces ...", end="", flush=True)
    scores = np.array([sil_scores(yaw_about_body(V0, g)) for g in angs])
    print(" done")
    j = int(scores[:, 0].argmax())
    print(f"yaw sweep about symmetry axis (centroid-aligned): best IoU {scores[j,0]:.3f} "
          f"(recall {scores[j,1]:.3f}, image shift {scores[j,2]:.0f} px) at {angs[j]:g} deg from the "
          f"given pose; IoU at 0 deg {scores[0,0]:.3f}  "
          f"{'-> pose yaw is off by ~' + format(angs[j], 'g') + ' deg' if scores[j,0] - scores[0,0] > 0.03 else '-> yaw is already the best-fitting one'}")
    top = np.argsort(-scores[:, 0])[:5]
    print("  top-5: " + "  ".join(f"{angs[t]:g}deg={scores[t,0]:.3f}" for t in top))

# ---- (b) front-surface depth: nearest projected z per pixel vs measured ----
valid = np.isfinite(zbuf) & mask & (depth > 0)
resid = np.full((H, W), np.nan)
resid[valid] = zbuf[valid] - depth[valid]                # +: mesh surface behind measured depth
r = resid[valid] * 1e3
print(f"depth residual (mesh - measured), {valid.sum()} px:")
print(f"  median {np.median(r):+.1f} mm   p10/p90 {np.percentile(r, 10):+.1f} / {np.percentile(r, 90):+.1f} mm"
      f"   |median| < ~5 mm and a narrow spread = pose+scale consistent with the depth image")
# radial structure of the residual: with a render-and-compare refiner the pose
# is fit to the whole visible surface, so BOTH a size error and a shape error
# (e.g. anisotropic squash) leave a radial trend. It says "mesh shape/size does
# not match the observed surface", not which one -- read the heatmap.
d_px = np.linalg.norm(np.argwhere(valid)[:, ::-1] - cm_m, axis=1)
if len(d_px) > 50:
    slope = np.polyfit(d_px, r, 1)[0]
    print(f"  residual trend vs radius  {slope*100:+.2f} mm per 100 px  "
          f"(a clear trend = mesh shape/size mismatch with the observed surface; see heatmap)")

im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
edge = mask & ~binary_erosion(mask, iterations=1)
for y, x in np.argwhere(edge): d.point((x, y), fill=(0, 255, 0))
for x, y in ui[::4]: d.point((x, y), fill=(255, 0, 0))
if dh is not None:
    for x, y in ui[handle[inside]][::2]: d.point((x, y), fill=(255, 255, 0))   # handle: yellow
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
