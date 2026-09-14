#!/usr/bin/env python3
"""Score every refined FoundationPose hypothesis against the SAM mask.

Input: the .npz written by `fp_from_mesh.py --all` (all N refined poses,
scorer-sorted, plus scores), the mesh they were refined on, and the saved
run_scene frame. For each hypothesis the mesh is projected, its silhouette
rasterised from the faces (at --scale of full resolution), and compared with
the mask: precision, recall, IoU. Also reports each hypothesis's rotation
distance from the scorer's pick.

This is the pre-check that decides whether an in-sidecar re-rank can work at
all: if some hypothesis reaches recall ~0.95+ with the handle visible, the
refiner left a correct candidate in the set and selecting it is enough; if
no hypothesis does, the refiner collapsed everything onto the wrong yaw and
the re-rank has to move to the coarse stage. It also shows which criterion
separates the right pose from the scorer's pick (recall, IoU, or neither).

  python3 outputs/runs/rank_hypotheses.py outputs/runs/20260903_203531 \
      --mesh any6d_runtime/outputs/final_mesh_mug.obj --object mug

Writes <run>/fp_<obj>_hypotheses.csv and <run>/fp_<obj>_best_recall.json (the
pose of the best-recall hypothesis, for reproject_check.py --pose-json).
"""
import argparse, csv, json, os, sys, time
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir")
ap.add_argument("--mesh", required=True)
ap.add_argument("--object", default="mug")
ap.add_argument("--npz", default=None, help="default <run>/fp_<object>_hypotheses.npz")
ap.add_argument("--scale", type=float, default=0.5, help="raster scale (0.5 = half resolution, ~4x faster)")
ap.add_argument("--recall-gate", type=float, default=0.95)
ap.add_argument("--precision-floor", type=float, default=0.95)
ap.add_argument("--rerank", action="store_true",
                help="also run pose_server/rerank.py's selection on these silhouettes exactly as the "
                     "sidecar would (same code), and report what it would have picked")
a = ap.parse_args()

npz = a.npz or os.path.join(a.run_dir, f"fp_{a.object}_hypotheses.npz")
d = np.load(npz, allow_pickle=True)
poses, scores = d["poses"].astype(np.float64), d["scores"].astype(np.float64)
mask = np.array(Image.open(os.path.join(a.run_dir, f"mask_{a.object.replace(' ', '_')}.png")).convert("L")) > 0
K = np.asarray(json.load(open(os.path.join(a.run_dir, "summary.json")))["frame"]["K"], np.float64)
H, W = mask.shape
s = a.scale
Hs, Ws = int(round(H * s)), int(round(W * s))
mask_s = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((Ws, Hs), Image.NEAREST)) > 0
Ks = K.copy(); Ks[:2] *= s

m = trimesh.load(a.mesh, force="mesh")
V, F = np.asarray(m.vertices), np.asarray(m.faces)
print(f"{len(poses)} hypotheses, mesh {len(V)} v {len(F)} f, raster {Ws}x{Hs}")

def silhouette(T):
    P = V @ T[:3, :3].T + T[:3, 3]
    if (P[:, 2] <= 0).any():
        return None
    u = P @ Ks.T; u = u[:, :2] / u[:, 2:3]
    img = Image.new("1", (Ws, Hs), 0); dr = ImageDraw.Draw(img)
    for row in np.clip(u[F].reshape(-1, 6), -1e6, 1e6).tolist():
        dr.polygon(row, fill=1)
    return np.array(img, dtype=bool)

from scipy.ndimage import distance_transform_edt

def rendered_depth(T, sil):
    """dense rendered depth inside the silhouette from the vertex z-buffer
    (nearest-filled) -- the offline stand-in for the sidecar's nvdiffrast depth"""
    P = V @ T[:3, :3].T + T[:3, 3]; u = P @ Ks.T; u = u[:, :2] / u[:, 2:3]
    q = np.round(u).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < Ws) & (q[:, 1] >= 0) & (q[:, 1] < Hs)
    zb = np.full((Hs, Ws), np.inf); np.minimum.at(zb, (q[ok, 1], q[ok, 0]), P[ok, 2])
    valid = np.isfinite(zb)
    if not valid.any():
        return np.zeros((Hs, Ws), np.float32)
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.where(sil, zb[idx[0], idx[1]], 0.0).astype(np.float32)

R0 = poses[0][:3, :3]
rows = []
sils = np.zeros((len(poses), Hs, Ws), bool)
depths = np.zeros((len(poses), Hs, Ws), np.float32)
t0 = time.time()
for i, (T, sc) in enumerate(zip(poses, scores)):
    sil = silhouette(T)
    if sil is None:
        rows.append(dict(rank=i, score=sc, precision=0, recall=0, iou=0, rot_deg=np.nan)); continue
    sils[i] = sil
    depths[i] = rendered_depth(T, sil)
    inter = (sil & mask_s).sum()
    prec = inter / max(sil.sum(), 1); rec = inter / max(mask_s.sum(), 1); iou = inter / max((sil | mask_s).sum(), 1)
    rot = np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3] @ R0.T).as_rotvec()))
    rows.append(dict(rank=i, score=float(sc), precision=float(prec), recall=float(rec), iou=float(iou), rot_deg=float(rot)))
    if i % 50 == 0:
        print(f"  {i}/{len(poses)}  {time.time()-t0:.0f}s", flush=True)

csv_path = os.path.join(a.run_dir, f"fp_{a.object}_hypotheses.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

rec = np.array([r["recall"] for r in rows]); prec = np.array([r["precision"] for r in rows])
iou = np.array([r["iou"] for r in rows]); rot = np.array([r["rot_deg"] for r in rows])
print(f"\nscorer pick (rank 0):     precision {prec[0]:.3f}  recall {rec[0]:.3f}  IoU {iou[0]:.3f}")
jb = int(np.nanargmax(rec)); ji = int(np.nanargmax(iou))
print(f"best recall (rank {jb}):   precision {prec[jb]:.3f}  recall {rec[jb]:.3f}  IoU {iou[jb]:.3f}  "
      f"rot from pick {rot[jb]:.0f} deg  scorer {scores[jb]:.2f} (pick {scores[0]:.2f})")
print(f"best IoU    (rank {ji}):   precision {prec[ji]:.3f}  recall {rec[ji]:.3f}  IoU {iou[ji]:.3f}  "
      f"rot from pick {rot[ji]:.0f} deg")
# ---- the RELATIVE read (the absolute gate below is informational only) -------
# The mesh's handle may be undersized (TRELLIS), so even the correct pose can
# cap below any fixed recall. What says "a correct candidate exists" is:
#   (1) best recall clearly above the scorer's pick (several points, not one),
#   (2) the top-by-recall hypotheses CLUSTER at one rotation from the pick
#       while the top-by-scorer scatter,
#   (3) reproject_check.py --pose-json fp_<obj>_best_recall.json shows the
#       handle on the mask.  The picture is the verdict.
gap = rec[jb] - rec[0]
top_rec = np.argsort(-rec)[:10]
rots = rot[top_rec]
rot_med = float(np.nanmedian(rots)); rot_mad = float(np.nanmedian(np.abs(rots - rot_med)))
top_sc_rots = rot[:10]
print(f"\nrelative read:")
print(f"  best-recall gap over pick  {gap:+.3f}   (yesterday's manual 140-deg rotation gave ~+0.05 on this frame)")
print(f"  top-10 by recall: rotation from pick median {rot_med:.0f} deg, MAD {rot_mad:.0f} deg  "
      f"{'<- CLUSTERED: a correct candidate exists' if rot_mad < 20 and rot_med > 30 else ''}")
print(f"  top-10 by scorer: rotations from pick {np.round(top_sc_rots).astype(int).tolist()}")
if gap < 0.02:
    print("  -> best recall is not meaningfully above the pick: no refined hypothesis covers the handle "
          "region better than the pick does -- the refiner collapsed onto the pick's yaw (or the mask "
          "has no handle). Re-rank would have to move to the coarse stage.")
elif rot_mad < 20 and rot_med > 30:
    print(f"  -> a correct candidate exists at ~{rot_med:.0f} deg from the pick; a recall gate over the refined "
          "set is enough. Set --recall-gate just below the cluster's recall.")
else:
    print("  -> recall separates hypotheses but they do not cluster in rotation: look at the picture before deciding.")

gate = (rec >= a.recall_gate) & (prec >= a.precision_floor)
print(f"\nabsolute gate (informational): recall >= {a.recall_gate} & precision >= {a.precision_floor}: "
      f"{gate.sum()} of {len(rows)}"
      + (f"; best scorer rank among them {int(np.argmax(gate))} (scorer {scores[np.argmax(gate)]:.2f})" if gate.any() else
         " -- 0 here does NOT by itself mean the refiner collapsed; an undersized handle caps recall. Use the relative read."))
print(f"recall distribution:  max {rec.max():.3f}  p90 {np.percentile(rec, 90):.3f}  median {np.median(rec):.3f}")
print(f"IoU spread across hypotheses: {iou.min():.3f}..{iou.max():.3f}   recall spread: {rec.min():.3f}..{rec.max():.3f}")

order = np.argsort(-rec)[:10]
print("\ntop-10 by recall:  rank  scorer  prec   recall  IoU    rot_from_pick")
for j in order:
    print(f"                   {j:4d}  {scores[j]:6.2f}  {prec[j]:.3f}  {rec[j]:.3f}  {iou[j]:.3f}  {rot[j]:6.0f}")
print("top-10 by scorer:  rank  scorer  prec   recall  IoU    rot_from_pick")
for j in range(min(10, len(rows))):
    print(f"                   {j:4d}  {scores[j]:6.2f}  {prec[j]:.3f}  {rec[j]:.3f}  {iou[j]:.3f}  {rot[j]:6.0f}")

if a.rerank:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_server"))
    from rerank import rerank_hypotheses
    depth_p = os.path.join(a.run_dir, "depth_mm.png")
    depth_s = None
    if os.path.exists(depth_p):
        dep = np.array(Image.open(depth_p)).astype(np.float64) * 1e-3
        depth_s = np.array(Image.fromarray(dep.astype(np.float32)).resize((Ws, Hs), Image.NEAREST))
    diameter = float(np.linalg.norm(m.bounding_box.extents)) if not hasattr(m, "bounding_sphere") else 2 * float(m.bounding_sphere.primitive.radius)
    chosen, rr = rerank_hypotheses(sils, scores, poses, mask_s, depth_s, diameter, depths=depths)
    print(f"\n=== sidecar re-rank (pose_server/rerank.py, same code) ===")
    for k, v in rr.__dict__.items():
        print(f"  {k:16s} {v}")
    print(f"  -> would return rank {chosen}"
          + (f" ({rr.rotation_deg:.0f} deg from the scorer's pick)" if rr.changed else " (scorer's pick unchanged)"))
    json.dump({"cam_T_obj": poses[chosen].tolist(), "mesh": str(d["mesh"]), "rank": chosen,
               "rerank": rr.__dict__},
              open(os.path.join(a.run_dir, f"fp_{a.object}_rerank.json"), "w"), indent=2)
    print(f"  wrote {os.path.join(a.run_dir, f'fp_{a.object}_rerank.json')}   (reproject_check.py --pose-json to see it)")

best = os.path.join(a.run_dir, f"fp_{a.object}_best_recall.json")
json.dump({"cam_T_obj": poses[jb].tolist(), "mesh": str(d["mesh"]), "rank": jb,
           "recall": float(rec[jb]), "precision": float(prec[jb]), "scorer": float(scores[jb])},
          open(best, "w"), indent=2)
print(f"\nwrote {csv_path}\nwrote {best}   (reproject_check.py --pose-json to see it)")
