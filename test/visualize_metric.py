#!/usr/bin/env python3
"""Side-by-side visual check of the TRELLIS.2 metric-scale branch.

Three panels along +x:

  [A] canonical mesh (gray) inside the unit-box wireframe  -- the raw
      TRELLIS output, scale-free contract check
  [B] metric mesh (blue) overlaid with the GT sim mesh (green wireframe),
      both AABB-centered                                    -- the scale check
  [C] metric mesh (blue) with the observed depth cloud (red) mapped into
      the mesh frame via the recovered (s, R, t)            -- the FIT check;
      this is the panel that shows WHERE the RMSE lives

Panel C needs the registration transform. It is read from
<name>_metric_reg.npz next to the metric GLB if the server saved one,
otherwise recomputed host-side from the packet (needs open3d + cv2).

Usage (from ur5e-manip-hardware root):

  python tools/visualize_metric.py --name test_mug --obj mug \
      [--packet test/data/packet/packet.npz] [--export scene.glb]

Interactive open3d window by default; --export writes a GLB you can open
anywhere (three.js viewer, Blender) if you're on a headless box.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "trellis2_runtime" / "outputs"

GRAY = [180, 180, 180, 255]
BLUE = [70, 120, 220, 255]
GREEN = [60, 190, 90, 255]
RED = [230, 60, 60, 255]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"{path}: did not load as a single mesh")
    return m


def _centered(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    return m


def _unit_box_wire() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[1, 1, 1]).outline().to_mesh() \
        if hasattr(trimesh.creation.box(extents=[1, 1, 1]).outline(), "to_mesh") \
        else trimesh.creation.box(extents=[1, 1, 1])


def _load_or_recompute_reg(args, canonical: trimesh.Trimesh):
    reg_path = OUTPUTS / f"{args.name}_metric_reg.npz"
    if reg_path.exists():
        d = np.load(reg_path)
        return {k: d[k] for k in ("scale", "R", "t", "rmse")}, "saved"
    if args.packet is None:
        return None, "no _reg.npz and no --packet given"
    # Recompute host-side: same code path as the server.
    sys.path.insert(0, str(REPO_ROOT / "trellis2_server"))
    try:
        import metric_scale  # noqa: needs open3d, cv2, scipy
    except ImportError as e:
        return None, f"recompute needs open3d/cv2 host-side ({e})"
    pk = np.load(args.packet, allow_pickle=False)
    P_obs = metric_scale.backproject(
        pk["depth"].astype(np.float32), pk["K"].astype(np.float64),
        pk[f"mask_{args.obj}"])
    sim = metric_scale.register_similarity(P_obs, canonical)
    sim["P_obs"] = P_obs
    return sim, "recomputed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test_mug",
                    help="output_name used in the generate request")
    ap.add_argument("--obj", default="mug", help="object key in the packet")
    ap.add_argument("--packet", type=Path,
                    default=REPO_ROOT / "test" / "data" / "packet" / "packet.npz")
    ap.add_argument("--export", type=Path, default=None,
                    help="write combined scene GLB instead of opening a window")
    args = ap.parse_args()
    if args.packet is not None and not args.packet.exists():
        args.packet = None

    canonical = _load_mesh(OUTPUTS / f"{args.name}.glb")
    metric = _load_mesh(OUTPUTS / f"{args.name}_metric.glb")

    gt = None
    if args.packet is not None:
        gt_path = args.packet.parent / f"{args.obj}.obj"
        if gt_path.exists():
            gt = _load_mesh(gt_path)

    reg, reg_src = _load_or_recompute_reg(args, canonical)

    # ---------------- report ----------------
    can_ext = float(max(canonical.extents))
    met_ext = float(max(metric.extents))
    print(f"canonical max extent : {can_ext:.4f}  (unit-box contract)")
    print(f"metric    max extent : {met_ext * 100:.2f} cm")
    if gt is not None:
        gt_ext = float(max(gt.extents))
        rel = abs(met_ext - gt_ext) / gt_ext
        print(f"GT        max extent : {gt_ext * 100:.2f} cm   rel err {rel:.1%}")
    if reg is not None:
        print(f"registration ({reg_src}): scale {float(reg['scale']):.4f}, "
              f"rmse {float(reg['rmse']) * 1000:.1f} mm")

    # ---------------- scene ----------------
    scene = trimesh.Scene()
    gap = max(met_ext, 0.15) * 1.8

    # [A] canonical + unit box
    a = _centered(canonical)
    a.visual.face_colors = GRAY
    scene.add_geometry(a, node_name="A_canonical")
    box = trimesh.creation.box(extents=[1, 1, 1])
    box.visual.face_colors = [120, 120, 120, 40]
    scene.add_geometry(box, node_name="A_unitbox")

    # [B] metric vs GT, both centered
    b = _centered(metric)
    b.visual.face_colors = BLUE
    b.apply_translation([gap, 0, 0])
    scene.add_geometry(b, node_name="B_metric")
    if gt is not None:
        g = _centered(gt)
        g.visual.face_colors = GREEN
        g.apply_translation([gap, 0, 0])
        scene.add_geometry(g, node_name="B_gt")

    # [C] metric mesh + obs cloud in mesh frame
    if reg is not None and np.isfinite(float(reg["rmse"])):
        c = metric.copy()  # metric = canonical * s, NOT centered: frames must match
        c.visual.face_colors = BLUE
        c.apply_translation([2 * gap, 0, 0])
        scene.add_geometry(c, node_name="C_metric")
        P_obs = reg.get("P_obs")
        if P_obs is None and args.packet is not None:
            sys.path.insert(0, str(REPO_ROOT / "trellis2_server"))
            try:
                import metric_scale
                pk = np.load(args.packet, allow_pickle=False)
                P_obs = metric_scale.backproject(
                    pk["depth"].astype(np.float32),
                    pk["K"].astype(np.float64), pk[f"mask_{args.obj}"])
            except ImportError:
                P_obs = None
        if P_obs is not None:
            s, R, t = float(reg["scale"]), np.asarray(reg["R"]), np.asarray(reg["t"])
            # p_cam = s R p_mesh + t  ->  metric-mesh frame: R^T (p_cam - t)
            Q = (P_obs - t) @ R  # == R.T applied row-wise
            cloud = trimesh.PointCloud(Q + [2 * gap, 0, 0], colors=RED)
            scene.add_geometry(cloud, node_name="C_obs")
            # per-point residuals against the metric mesh -> where the rmse lives
            closest, dist, _ = trimesh.proximity.closest_point(c, Q + [2 * gap, 0, 0])
            print(f"obs->mesh residuals  : median {np.median(dist)*1000:.1f} mm, "
                  f"p90 {np.quantile(dist, 0.9)*1000:.1f} mm, "
                  f"max {dist.max()*1000:.1f} mm")
    elif reg is None:
        print(f"[C] skipped: {reg_src}")

    if args.export:
        scene.export(str(args.export))
        print(f"wrote {args.export}")
        return
    try:
        import open3d as o3d
        geoms = []
        for name, geom in scene.geometry.items():
            T = scene.graph.get(name)[0] if name in scene.graph.nodes_geometry else np.eye(4)
            if isinstance(geom, trimesh.PointCloud):
                p = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(np.asarray(geom.vertices)))
                p.paint_uniform_color(np.array(RED[:3]) / 255)
                geoms.append(p)
            else:
                m = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(geom.vertices)),
                    o3d.utility.Vector3iVector(np.asarray(geom.faces)))
                m.compute_vertex_normals()
                col = geom.visual.face_colors[0][:3] / 255.0 \
                    if hasattr(geom.visual, "face_colors") else [0.7, 0.7, 0.7]
                m.paint_uniform_color(col)
                geoms.append(m)
        o3d.visualization.draw_geometries(geoms, window_name="TRELLIS metric check")
    except ImportError:
        scene.show()  # trimesh/pyglet fallback


if __name__ == "__main__":
    main()