import numpy as np, trimesh, open3d as o3d
from scipy.spatial.transform import Rotation
from metric_scale import register_similarity

rng = np.random.default_rng(0)

def unitize(m):
    """TRELLIS-style: center at origin, longest extent -> 1.0"""
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(1.0 / m.extents.max())
    return m

def make_meshes():
    out = {}
    out["box"] = trimesh.creation.box(extents=[0.06, 0.09, 0.15])
    out["cylinder"] = trimesh.creation.cylinder(radius=0.035, height=0.12,
                                                sections=64)
    out["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=0.045)
    # mug-like: cylinder + torus handle (asymmetric, curvature-rich)
    body = trimesh.creation.cylinder(radius=0.04, height=0.10, sections=64)
    handle = trimesh.creation.torus(major_radius=0.028, minor_radius=0.007,
                                    major_sections=48, minor_sections=16)
    handle.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))
    handle.apply_translation([0.042, 0, 0])
    out["mug"] = trimesh.util.concatenate([body, handle])
    # bumpy blob: rich local geometry, best case for FPFH
    blob = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    v = blob.vertices
    r = np.linalg.norm(v, axis=1, keepdims=True)
    blob.vertices = v * (1 + 0.22 * np.sin(4 * v[:, 0:1] / 0.05)
                           * np.cos(3 * v[:, 1:2] / 0.05))
    out["blob"] = blob
    # thin sheet: degenerate case (voxel-averaged normals cancel)
    out["thin_sheet"] = trimesh.creation.box(extents=[0.12, 0.09, 0.002])
    return out

def simulate_view(mesh_metric_cam, n_pts=60000, noise_mm=1.5):
    """Sample surface, keep only points visible from camera at origin."""
    pts = mesh_metric_cam.sample(n_pts)
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    diameter = np.linalg.norm(np.asarray(p.get_max_bound())
                              - np.asarray(p.get_min_bound()))
    _, idx = p.hidden_point_removal(np.zeros(3), diameter * 200)
    vis = pts[np.asarray(idx)]
    # axial (z) noise, like a depth sensor
    vis = vis + np.stack([np.zeros(len(vis)), np.zeros(len(vis)),
                          rng.normal(0, noise_mm / 1000, len(vis))], 1)
    return vis

def run(name, mesh_metric, seed):
    r = np.random.default_rng(seed)
    canon = unitize(mesh_metric)
    s_true = mesh_metric.extents.max()          # unit -> metric factor
    R_true = Rotation.random(random_state=seed).as_matrix()
    t_true = np.array([r.uniform(-0.05, 0.05), r.uniform(-0.05, 0.05),
                       r.uniform(0.45, 0.7)])

    placed = canon.copy()
    T = np.eye(4); T[:3, :3] = s_true * R_true; T[:3, 3] = t_true
    placed.apply_transform(T)
    P_obs = simulate_view(placed)

    res = register_similarity(P_obs, canon, verbose=True)
    s_err = 100 * (res["scale"] / s_true - 1)
    rot_err = np.degrees(np.arccos(np.clip(
        (np.trace(res["R"].T @ R_true) - 1) / 2, -1, 1)))
    t_err = 1000 * np.linalg.norm(res["t"] - t_true)
    print(f"{name:12s} obs={len(P_obs):6d}  s_true={s_true*1000:6.1f}mm  "
          f"s_est={res['scale']*1000:6.1f}mm  Δs={s_err:+7.2f}%  "
          f"Δrot={rot_err:6.1f}deg  Δt={t_err:6.1f}mm  "
          f"rmse={res['rmse']*1000:6.2f}mm  ok={res['ok']}  "
          f"path={res.get('path','-')}")
    return s_err, rot_err, res["ok"]

if __name__ == "__main__":
    meshes = make_meshes()
    for seed in (1, 2, 3):
        print(f"--- seed {seed} ---")
        for name, m in meshes.items():
            try:
                run(name, m, seed)
            except Exception as e:
                print(f"{name:12s} EXCEPTION {type(e).__name__}: {e}")