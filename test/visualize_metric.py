#!/usr/bin/env python3
"""Side-by-side visual check of the TRELLIS.2 metric-scale branch.

Panels along +x:

  [A] canonical mesh (gray) inside the unit-box wireframe -- unit-box contract
  [B] metric mesh at the RECOVERED scale (blue) vs GT sim mesh (green),
      both AABB-centered                                  -- the scale check
  [C] same metric mesh (blue) with the observed depth cloud (red) mapped
      into the mesh frame via the recovered (s, R, t)     -- the FIT check;
      per-point residual stats show WHERE the RMSE lives

The registration is read from <name>_metric_reg.npz next to the metric GLB
if the server saved one, otherwise recomputed host-side from the packet
(needs open3d + cv2). The on-disk <name>_metric.glb is loaded only to
cross-check its scale against the recovered one: the server exports it even
when registration flags ok=False, so a stale/garbage-scaled GLB on the
shared mount is a real failure mode this script now detects.

Usage (from ur5e-manip-hardware root):

  python3 test/visualize_metric.py --name test_mug --obj mug \
      [--packet test/data/packet/packet.npz] [--export scene.glb]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "trellis2_runtime" / "outputs"

GRAY = np.array([0.70, 0.70, 0.70])
BLUE = np.array([0.27, 0.47, 0.86])
GREEN = np.array([0.24, 0.75, 0.35])
RED = np.array([0.90, 0.24, 0.24])
ORANGE = np.array([0.95, 0.60, 0.15])


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"{path}: did not load as a single mesh")
    return m


def _centered(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    return m


def _backproject_obs(packet_path: Path, obj: str):
    sys.path.insert(0, str(REPO_ROOT / "trellis2_server"))
    import metric_scale  # needs cv2
    pk = np.load(packet_path, allow_pickle=False)
    return metric_scale.backproject(
        pk["depth"].astype(np.float32), pk["K"].astype(np.float64),
        pk[f"mask_{obj}"])


def _load_or_recompute_reg(args, canonical: trimesh.Trimesh):
    reg_path = OUTPUTS / f"{args.name}_metric_reg.npz"
    if reg_path.exists():
        d = np.load(reg_path)
        reg = {k: d[k] for k in d.files}
        return reg, "saved"
    if args.packet is None:
        return None, "no _reg.npz and no --packet given"
    sys.path.insert(0, str(REPO_ROOT / "trellis2_server"))
    try:
        import metric_scale  # needs open3d, cv2, scipy
    except (ImportError, TypeError) as e:
        # TypeError: PEP 604 annotations on py<3.10 -- add
        # `from __future__ import annotations` to metric_scale.py
        return None, f"could not import metric_scale host-side ({e})"
    P_obs = _backproject_obs(args.packet, args.obj)
    sim = metric_scale.register_similarity(P_obs, canonical)
    sim["P_obs"] = P_obs
    return sim, "recomputed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test_mug")
    ap.add_argument("--obj", default="mug")
    ap.add_argument("--packet", type=Path,
                    default=REPO_ROOT / "test" / "data" / "packet" / "packet.npz")
    ap.add_argument("--export", type=Path, default=None)
    args = ap.parse_args()
    if args.packet is not None and not args.packet.exists():
        args.packet = None

    canonical = _load_mesh(OUTPUTS / f"{args.name}.glb")
    can_ext = float(max(canonical.extents))

    gt = None
    if args.packet is not None:
        gt_path = args.packet.parent / f"{args.obj}.obj"
        if gt_path.exists():
            gt = _load_mesh(gt_path)

    reg, reg_src = _load_or_recompute_reg(args, canonical)

    # ---------------- report ----------------
    print(f"canonical max extent : {can_ext:.4f}  (unit-box contract)")
    s_rec = None
    if reg is not None and np.isfinite(float(reg["rmse"])):
        s_rec = float(reg["scale"])
        rmse = float(reg["rmse"])
        met_ext = can_ext * s_rec
        print(f"recovered scale      : {s_rec:.4f}  -> extent {met_ext*100:.2f} cm  "
              f"(rmse {rmse*1000:.1f} mm = {rmse/met_ext:.1%} of extent) [{reg_src}]")
        if gt is not None:
            gt_ext = float(max(gt.extents))
            print(f"GT        max extent : {gt_ext*100:.2f} cm   "
                  f"rel err {abs(met_ext-gt_ext)/gt_ext:.1%}")
    else:
        print(f"registration unavailable: {reg_src}")

    # cross-check the exported GLB against the recovered scale
    disk_metric = None
    disk_path = OUTPUTS / f"{args.name}_metric.glb"
    if disk_path.exists():
        disk_metric = _load_mesh(disk_path)
        s_disk = float(max(disk_metric.extents)) / can_ext
        if s_rec is not None and abs(s_disk - s_rec) / s_rec > 0.05:
            print(f"WARNING: on-disk {disk_path.name} implies scale {s_disk:.4f} "
                  f"!= recovered {s_rec:.4f} -- stale export or a run where "
                  f"ok=False was exported anyway. Shown in ORANGE in panel B; "
                  f"do not feed this GLB to FoundationPose.")
        else:
            disk_metric = None  # consistent; no need to draw it twice

    # ---------------- scene ----------------
    # geoms: list of (vertices/faces or points, color, is_cloud)
    geoms = []
    ref_ext = (can_ext * s_rec) if s_rec else 0.15
    gap = max(ref_ext, 0.15) * 1.8

    a = _centered(canonical)
    geoms.append((a, GRAY, False))
    geoms.append((trimesh.creation.box(extents=[1, 1, 1]), None, False))  # wire

    if s_rec is not None:
        metric = canonical.copy()
        metric.apply_scale(s_rec)
        b = _centered(metric)
        b.apply_translation([gap, 0, 0])
        geoms.append((b, BLUE, False))
        if gt is not None:
            g = _centered(gt)
            g.apply_translation([gap, 0, 0])
            geoms.append((g, GREEN, False))
        if disk_metric is not None:
            dm = _centered(disk_metric)
            dm.apply_translation([gap, 0, 0])
            geoms.append((dm, ORANGE, False))

        # [C] fit check in the metric-mesh frame (canonical*s, uncentered)
        c = metric.copy()
        c.apply_translation([2 * gap, 0, 0])
        geoms.append((c, BLUE, False))
        P_obs = reg.get("P_obs")
        if P_obs is None and args.packet is not None:
            try:
                P_obs = _backproject_obs(args.packet, args.obj)
            except (ImportError, TypeError):
                P_obs = None
        if P_obs is not None:
            R, t = np.asarray(reg["R"], float), np.asarray(reg["t"], float)
            Q = (P_obs - t) @ R  # rows: R.T (p - t) -> metric-mesh frame
            geoms.append((Q + [2 * gap, 0, 0], RED, True))
            _, dist, _ = trimesh.proximity.closest_point(c, Q + [2 * gap, 0, 0])
            print(f"obs->mesh residuals  : median {np.median(dist)*1000:.1f} mm, "
                  f"p90 {np.quantile(dist, 0.9)*1000:.1f} mm, "
                  f"max {dist.max()*1000:.1f} mm")

    if args.export:
        scene = trimesh.Scene()
        for i, (g, col, is_cloud) in enumerate(geoms):
            if is_cloud:
                scene.add_geometry(trimesh.PointCloud(
                    g, colors=(np.append(col, 1.0) * 255).astype(np.uint8)))
            else:
                m = g.copy()
                m.visual = trimesh.visual.ColorVisuals(
                    m, face_colors=(np.append(col if col is not None else GRAY,
                                              0.6 if col is None else 1.0)
                                    * 255).astype(np.uint8))
                scene.add_geometry(m, node_name=f"g{i}")
        scene.export(str(args.export))
        print(f"wrote {args.export}")
        return

    import open3d as o3d
    draw = []
    for g, col, is_cloud in geoms:
        if is_cloud:
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(g)))
            p.paint_uniform_color(col)
            draw.append(p)
        elif col is None:  # unit box as a wireframe
            draw.append(o3d.geometry.LineSet.create_from_triangle_mesh(
                o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(g.vertices)),
                    o3d.utility.Vector3iVector(np.asarray(g.faces)))))
        else:
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(g.vertices)),
                o3d.utility.Vector3iVector(np.asarray(g.faces)))
            m.compute_vertex_normals()
            m.paint_uniform_color(col)
            draw.append(m)
    o3d.visualization.draw_geometries(draw, window_name="TRELLIS metric check")


if __name__ == "__main__":
    main()