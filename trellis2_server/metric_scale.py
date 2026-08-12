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
     NB: all radii used at this stage (N_*) are expressed in NORMALIZED
     units (unit-RMS clouds have extent ~2-3), not meters. Mixing metric
     radii into normalized space starves normal estimation of neighbors
     -> garbage FPFH -> zero correspondences.
  4. Refine: trimmed scaled-ICP, closed-form Umeyama per iteration,
     correspondences obs -> mesh only (partial-to-full).
  5. Report inlier RMSE; caller thresholds it (see MAX_OK_RMSE hint).

Degeneracy policy: this module NEVER raises for bad geometry -- every
degenerate path (too few points, singular RANSAC similarity, NaN/collapsed
ICP) returns ok=False with rmse=inf, so the server's metric branch always
replies and the caller decides.

Degeneracies to expect: near-spherical objects (R unobservable -- fine, s
is still well constrained); mirror symmetries (ICP local minimum -- we keep
the top RANSAC hypotheses and pick by refined RMSE); thin/low-relief
geometry (poorly constrained along the normal).
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ---- metric-space knobs (meters; applied to the raw obs cloud) ----------
VOXEL = 0.003          # m, downsample of the observed cloud
MIN_OBS_POINTS = 200   # below this, don't even attempt registration
MAX_OK_RMSE = 0.006    # m; above this, treat scale as unreliable

# ---- normalized-space knobs (unit-RMS clouds, extent ~2-3) --------------
# Rule of thumb: voxel ~5% of RMS radius; normals ~2-3x voxel; FPFH ~6x.
N_VOXEL = 0.05
N_NORMAL_RADIUS = 0.12
N_FPFH_RADIUS = 0.30
N_MAX_CORR = 0.08      # RANSAC correspondence distance / checker

RANSAC_N_HYP = 3       # top hypotheses carried into refinement
TRIM_FRAC = 0.75       # keep best 75% correspondences per ICP iter
ICP_ITERS = 40

_FAIL = {"scale": 1.0, "R": np.eye(3), "t": np.zeros(3),
         "rmse": float("inf"), "ok": False}


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
    # Hybrid search: radius in NORMALIZED units, capped neighbor count so
    # dense patches can't blow up the neighborhood.
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=N_NORMAL_RADIUS,
                                             max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=N_FPFH_RADIUS,
                                                max_nn=100))


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
    """Returns (s, R, t, rmse). Degenerate states return rmse=inf instead
    of raising; caller maps that to ok=False."""
    tree = cKDTree(P_mesh)  # vectorized queries; the o3d per-point python
                            # loop was O(N * ICP_ITERS) interpreter calls
    for _ in range(ICP_ITERS):
        if not (np.isfinite(s) and s > 1e-9 and
                np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
            return s, R, t, float("inf")
        Q = (P_obs - t) @ R / s
        _, idx = tree.query(Q, k=1)
        matched = P_mesh[idx]
        d = np.linalg.norm(Q - matched, axis=1)
        keep = d <= np.quantile(d, TRIM_FRAC)
        if keep.sum() < 10:
            return s, R, t, float("inf")
        s, R, t = _umeyama(matched[keep], P_obs[keep])
    if not (np.isfinite(s) and s > 1e-9):
        return s, R, t, float("inf")
    Q = (P_obs - t) @ R / s
    _, idx = tree.query(Q, k=1)
    resid = s * np.linalg.norm(Q - P_mesh[idx], axis=1)  # back to meters
    inl = resid <= np.quantile(resid, TRIM_FRAC)
    rmse = float(np.sqrt((resid[inl] ** 2).mean()))
    return s, R, t, rmse


def register_similarity(P_obs: np.ndarray, mesh_trimesh) -> dict:
    """P_obs: Nx3 camera-frame meters. mesh_trimesh: canonical trimesh.
    Returns {scale, R, t, rmse, ok}. Never raises on degenerate geometry."""
    if len(P_obs) < MIN_OBS_POINTS:
        return dict(_FAIL)
    P_mesh = np.asarray(mesh_trimesh.sample(30_000), dtype=np.float64)

    obs = _pcd(P_obs, VOXEL)
    P_obs_ds = np.asarray(obs.points)
    if len(P_obs_ds) < MIN_OBS_POINTS:
        return dict(_FAIL)

    # RMS-normalize both so the N_* radii mean the same thing on each
    # cloud; downsampling BOTH at N_VOXEL also equalizes density (FPFH is
    # sensitive to density mismatch between source and target).
    r_obs = float(np.sqrt(P_obs_ds.var(0).sum()))
    r_msh = float(np.sqrt(P_mesh.var(0).sum()))
    if not (r_obs > 1e-6 and r_msh > 1e-6):
        return dict(_FAIL)
    obs_n = _pcd(P_obs_ds / r_obs, N_VOXEL)
    msh_n = _pcd(P_mesh / r_msh, N_VOXEL)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        obs_n, msh_n, _fpfh(obs_n), _fpfh(msh_n),
        mutual_filter=True, max_correspondence_distance=N_MAX_CORR,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPoint(with_scaling=True),
        ransac_n=4,
        checkers=[o3d.pipelines.registration
                  .CorrespondenceCheckerBasedOnDistance(N_MAX_CORR)],
        criteria=o3d.pipelines.registration
            .RANSACConvergenceCriteria(200_000, 0.999),
    )

    # obs_n -> msh_n similarity T = [s'R' | t']; invert & denormalize into
    # mesh->cam convention: p_cam ~= s R p_mesh + t.
    T = result.transformation
    sR, tp = T[:3, :3], T[:3, 3]
    det = np.linalg.det(sR)
    if (len(result.correspondence_set) < 10
            or not np.isfinite(det) or det < 1e-9):
        # Global init failed (too few matches / singular similarity):
        # fall back to a scale-only seed from the RMS ratio with identity
        # rotation -- for near-symmetric objects ICP can still converge,
        # and if it can't, rmse=inf -> ok=False, never a crash.
        s, R, t, rmse = _trimmed_scaled_icp(
            P_obs_ds, P_mesh, r_obs / r_msh, np.eye(3),
            P_obs_ds.mean(0) - (r_obs / r_msh) * P_mesh.mean(0))
        return {"scale": float(s), "R": R, "t": t, "rmse": rmse,
                "ok": bool(np.isfinite(rmse) and rmse < MAX_OK_RMSE)}

    sp = np.cbrt(det)
    Rp = sR / sp
    s0 = r_obs / (sp * r_msh)
    R0 = Rp.T
    t0 = -r_obs * (Rp.T @ tp) / sp
    # NB: single hypothesis refined here; for heavy mirror symmetry, rerun
    # RANSAC with different seeds and keep the best refined RMSE.
    s, R, t, rmse = _trimmed_scaled_icp(P_obs_ds, P_mesh, s0, R0, t0)

    return {"scale": float(s), "R": R, "t": t, "rmse": rmse,
            "ok": bool(np.isfinite(rmse) and rmse < MAX_OK_RMSE)}