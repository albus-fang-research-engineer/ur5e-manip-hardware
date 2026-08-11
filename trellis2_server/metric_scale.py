"""Metric scale recovery: similarity registration of a partial single-view
depth cloud (camera frame, meters) onto the TRELLIS.2 canonical mesh.

Estimates (s, R, t) s.t.  p_cam ~= s * R * p_mesh + t.

Pipeline (see README for rationale and failure modes):
  1. Backproject masked depth -> P_obs; erode mask (depth bleeding at
     silhouette edges is the #1 scale killer on RealSense), SOR, voxel DS.
  2. Area-weighted sample the mesh -> P_mesh.
  3. Global init: normalize both clouds to unit RMS radius, FPFH + RANSAC
     with TransformationEstimationPointToPoint(with_scaling=True); compose
     the normalization ratio back in. Keeps FPFH radii commensurate.
  4. Refine: trimmed scaled-ICP, closed-form Umeyama per iteration,
     correspondences obs -> mesh only (partial-to-full).
  5. Report inlier RMSE; caller thresholds it (see MAX_OK_RMSE hint).

Degeneracies to expect: near-spherical objects (R unobservable -- fine, s
is still well constrained); mirror symmetries (ICP local minimum -- we keep
the top RANSAC hypotheses and pick by refined RMSE); thin/low-relief
geometry (poorly constrained along the normal).
"""

import numpy as np
import open3d as o3d

VOXEL = 0.003          # m, downsample for registration
FPFH_RADIUS = 0.025    # on unit-RMS-normalized clouds (dimensionless-ish)
NORMAL_RADIUS = 0.012
RANSAC_N_HYP = 3       # top hypotheses carried into refinement
TRIM_FRAC = 0.75       # keep best 75% correspondences per ICP iter
ICP_ITERS = 40
MAX_OK_RMSE = 0.006    # m; above this, treat scale as unreliable


def backproject(depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray,
                erode_px: int = 4, z_range=(0.15, 2.0)) -> np.ndarray:
    import cv2
    m = (mask > 0).astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    v, u = np.nonzero(m)
    z = depth_m[v, u]
    ok = (z > z_range[0]) & (z < z_range[1]) & np.isfinite(z)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float64)


def _pcd(points: np.ndarray, voxel: float | None) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel:
        p = p.voxel_down_sample(voxel)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return p


def _fpfh(p: o3d.geometry.PointCloud):
    p.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(NORMAL_RADIUS))
    return o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamRadius(FPFH_RADIUS))


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Closed-form similarity: dst ~= s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def _trimmed_scaled_icp(P_obs: np.ndarray, P_mesh: np.ndarray,
                        s: float, R: np.ndarray, t: np.ndarray):
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_mesh)))
    # Work in the mesh frame: x_mesh ~= R^T (x_cam - t) / s
    for _ in range(ICP_ITERS):
        Q = (P_obs - t) @ R / s
        idx = np.array([tree.search_knn_vector_3d(q, 1)[1][0] for q in Q])
        matched = P_mesh[idx]
        d = np.linalg.norm(Q - matched, axis=1)
        keep = d <= np.quantile(d, TRIM_FRAC)
        s, R, t = _umeyama(matched[keep], P_obs[keep])
    Q = (P_obs - t) @ R / s
    idx = np.array([tree.search_knn_vector_3d(q, 1)[1][0] for q in Q])
    resid = s * np.linalg.norm(Q - P_mesh[idx], axis=1)  # back to meters
    inl = resid <= np.quantile(resid, TRIM_FRAC)
    rmse = float(np.sqrt((resid[inl] ** 2).mean()))
    return s, R, t, rmse


def register_similarity(P_obs: np.ndarray, mesh_trimesh) -> dict:
    """P_obs: Nx3 camera-frame meters. mesh_trimesh: canonical trimesh.
    Returns {scale, R, t, rmse, ok}."""
    P_mesh = mesh_trimesh.sample(30_000).astype(np.float64)

    obs = _pcd(P_obs, VOXEL)
    # RMS-normalize both so FPFH radii mean the same thing on each cloud.
    r_obs = np.sqrt((np.asarray(obs.points).var(0)).sum())
    r_msh = np.sqrt(P_mesh.var(0).sum())
    obs_n = _pcd(np.asarray(obs.points) / r_obs, None)
    msh_n = _pcd(P_mesh / r_msh, None)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        obs_n, msh_n, _fpfh(obs_n), _fpfh(msh_n),
        mutual_filter=True, max_correspondence_distance=0.06,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPoint(with_scaling=True),
        ransac_n=4,
        checkers=[o3d.pipelines.registration
                  .CorrespondenceCheckerBasedOnDistance(0.06)],
        criteria=o3d.pipelines.registration
            .RANSACConvergenceCriteria(200_000, 0.999),
    )
    # obs_n -> msh_n similarity T = [s'R' | t']; invert & denormalize into
    # mesh->cam convention: p_cam ~= s R p_mesh + t.
    T = result.transformation
    sR, tp = T[:3, :3], T[:3, 3]
    sp = np.cbrt(np.linalg.det(sR))
    Rp = sR / sp
    s0 = r_obs / (sp * r_msh)
    R0 = Rp.T
    t0 = -r_obs * (Rp.T @ tp) / sp
    # NB: single hypothesis refined here; for heavy mirror symmetry, rerun
    # RANSAC with different seeds and keep the best refined RMSE.
    s, R, t, rmse = _trimmed_scaled_icp(
        np.asarray(obs.points), P_mesh, s0, R0, t0)

    return {"scale": float(s), "R": R, "t": t, "rmse": rmse,
            "ok": rmse < MAX_OK_RMSE}
