"""Metric scale recovery: similarity registration of a partial single-view
depth cloud (camera frame, meters) onto the TRELLIS.2 canonical mesh.

Estimates (s, R, t) s.t.  p_cam ~= s * R * p_mesh + t.

Pipeline (see README for rationale and failure modes):
  1. Backproject masked depth -> P_obs; erode mask (depth bleeding at
     silhouette edges is the #1 scale killer on RealSense), SOR, voxel DS.
  2. Area-weighted sample the mesh -> P_mesh, carrying TRUE face normals
     (never estimated -- estimated normals on mesh samples have arbitrary
     sign and shred FPFH similarity against the obs cloud).
  3. Global init: normalize both clouds to unit RMS radius; FPFH with
     CONSISTENTLY ORIENTED normals (obs: toward camera at origin; mesh:
     outward face normals -- outward == camera-facing on the visible
     surface, so the conventions agree). Correspondences are computed
     OURSELVES with a mutual-NN filter: Open3D's built-in mutual filter
     silently falls back to noisy one-way matches whenever mutual pairs
     < 10% of source points ("Too few correspondences ... fall back"),
     which is the wrong heuristic for partial-to-full -- a few dozen
     mutual matches is a high-precision set and plenty for ransac_n=4.
     RANSAC (with_scaling=True) is rerun RANSAC_N_HYP times and every
     hypothesis is refined; best refined RMSE wins (mirror symmetries).
     NB: all radii used at this stage (N_*) are expressed in NORMALIZED
     units (unit-RMS clouds have extent ~2-3), not meters.
  4. Refine: trimmed scaled-ICP, closed-form Umeyama per iteration,
     correspondences obs -> mesh only (partial-to-full).
  5. Report inlier RMSE; caller thresholds it (see MAX_OK_RMSE hint).

Degeneracy policy: this module NEVER raises for bad geometry -- every
degenerate path (too few points, singular RANSAC similarity, NaN/collapsed
ICP) returns ok=False with rmse=inf, so the server's metric branch always
replies and the caller decides.

Degeneracies to expect: near-spherical objects (R unobservable -- fine, s
is still well constrained); mirror symmetries (handled by multi-hypothesis
RANSAC + refined-RMSE selection); thin/low-relief geometry (poorly
constrained along the normal; note voxel DS averages opposing face normals
on thin sheets toward zero -- those points contribute weak FPFH, which is
honest).
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

MIN_MUTUAL_CORR = 10   # accept a mutual set this small (vs o3d's 10% of N)
RANSAC_N_HYP = 3       # independent RANSAC restarts, all refined
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


def _pcd(points: np.ndarray, voxel: float | None,
         normals: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        p.normals = o3d.utility.Vector3dVector(normals)
    if voxel:
        p = p.voxel_down_sample(voxel)  # averages normals if present
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return p


def _fpfh(p: o3d.geometry.PointCloud):
    """FPFH with consistently oriented normals. If the cloud already
    carries normals (mesh sample: true outward face normals), keep them;
    otherwise (obs cloud, camera frame scaled about the origin) estimate
    and orient toward the camera at the origin. FPFH angles are taken
    relative to the normal, so sign consistency ACROSS the two clouds is
    what makes descriptors comparable at all."""
    if not p.has_normals():
        # Hybrid search: radius in NORMALIZED units, capped neighbor count
        # so dense patches can't blow up the neighborhood.
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=N_NORMAL_RADIUS,
                                                 max_nn=30))
        p.orient_normals_towards_camera_location(np.zeros(3))
    p.normalize_normals()  # voxel averaging leaves non-unit normals
    return o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=N_FPFH_RADIUS,
                                                max_nn=100))


def _correspondences(f_src, f_dst) -> np.ndarray:
    """(M, 2) int array of (src_idx, dst_idx). Mutual-NN in FPFH space if
    the mutual set has >= MIN_MUTUAL_CORR pairs, else one-way src->dst.
    Version-independent replacement for o3d's built-in mutual filter,
    whose >=10%-of-source fallback discards small-but-precise mutual sets
    in the partial-to-full regime (backside mesh points break mutuality by
    construction)."""
    A = np.asarray(f_src.data).T  # (Ns, 33)
    B = np.asarray(f_dst.data).T  # (Nd, 33)
    _, j = cKDTree(B).query(A)    # src -> dst
    _, i = cKDTree(A).query(B)    # dst -> src
    src_idx = np.arange(len(A))
    mutual = src_idx[i[j] == src_idx]
    if len(mutual) >= MIN_MUTUAL_CORR:
        return np.stack([mutual, j[mutual]], axis=1)
    return np.stack([src_idx, j], axis=1)


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


def _scale_only_fallback(P_obs_ds, P_mesh, r_obs, r_msh):
    """Scale seed from the RMS ratio with identity rotation -- for
    near-symmetric objects ICP can still converge, and if it can't,
    rmse=inf -> ok=False, never a crash."""
    s0 = r_obs / r_msh
    s, R, t, rmse = _trimmed_scaled_icp(
        P_obs_ds, P_mesh, s0, np.eye(3),
        P_obs_ds.mean(0) - s0 * P_mesh.mean(0))
    return {"scale": float(s), "R": R, "t": t, "rmse": rmse,
            "ok": bool(np.isfinite(rmse) and rmse < MAX_OK_RMSE)}


def register_similarity(P_obs: np.ndarray, mesh_trimesh) -> dict:
    """P_obs: Nx3 camera-frame meters. mesh_trimesh: canonical trimesh.
    Returns {scale, R, t, rmse, ok}. Never raises on degenerate geometry."""
    if len(P_obs) < MIN_OBS_POINTS:
        return dict(_FAIL)
    pts, fid = mesh_trimesh.sample(30_000, return_index=True)
    P_mesh = np.asarray(pts, dtype=np.float64)
    N_mesh = np.asarray(mesh_trimesh.face_normals[fid], dtype=np.float64)

    obs = _pcd(P_obs, VOXEL)
    P_obs_ds = np.asarray(obs.points)
    if len(P_obs_ds) < MIN_OBS_POINTS:
        return dict(_FAIL)

    # RMS-normalize both so the N_* radii mean the same thing on each
    # cloud; downsampling BOTH at N_VOXEL also equalizes density (FPFH is
    # sensitive to density mismatch between source and target). Uniform
    # scaling about the origin leaves normal directions AND the camera
    # location (origin) unchanged, so orientation logic still holds.
    r_obs = float(np.sqrt(P_obs_ds.var(0).sum()))
    r_msh = float(np.sqrt(P_mesh.var(0).sum()))
    if not (r_obs > 1e-6 and r_msh > 1e-6):
        return dict(_FAIL)
    obs_n = _pcd(P_obs_ds / r_obs, N_VOXEL)
    msh_n = _pcd(P_mesh / r_msh, N_VOXEL, normals=N_mesh)

    corr = _correspondences(_fpfh(obs_n), _fpfh(msh_n))
    if len(corr) < MIN_MUTUAL_CORR:
        return _scale_only_fallback(P_obs_ds, P_mesh, r_obs, r_msh)

    estimation = (o3d.pipelines.registration
                  .TransformationEstimationPointToPoint(with_scaling=True))
    checkers = [o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnDistance(N_MAX_CORR)]
    criteria = (o3d.pipelines.registration
                .RANSACConvergenceCriteria(200_000, 0.999))
    corr_o3d = o3d.utility.Vector2iVector(corr.astype(np.int32))

    # Multi-hypothesis: RANSAC restarts are cheap relative to a wrong
    # mirror-symmetric basin; refine every hypothesis, keep best RMSE.
    best = None
    for _ in range(RANSAC_N_HYP):
        result = (o3d.pipelines.registration
                  .registration_ransac_based_on_correspondence(
                      obs_n, msh_n, corr_o3d,
                      max_correspondence_distance=N_MAX_CORR,
                      estimation_method=estimation, ransac_n=4,
                      checkers=checkers, criteria=criteria))
        # obs_n -> msh_n similarity T = [s'R' | t']; invert & denormalize
        # into mesh->cam convention: p_cam ~= s R p_mesh + t.
        T = result.transformation
        sR, tp = T[:3, :3], T[:3, 3]
        det = np.linalg.det(sR)
        if (len(result.correspondence_set) < 10
                or not np.isfinite(det) or det < 1e-9):
            continue  # singular / unsupported hypothesis
        sp = np.cbrt(det)
        Rp = sR / sp
        s0 = r_obs / (sp * r_msh)
        R0 = Rp.T
        t0 = -r_obs * (Rp.T @ tp) / sp
        s, R, t, rmse = _trimmed_scaled_icp(P_obs_ds, P_mesh, s0, R0, t0)
        if best is None or rmse < best[3]:
            best = (s, R, t, rmse)
        if np.isfinite(rmse) and rmse < MAX_OK_RMSE:
            break  # good enough; skip remaining restarts

    if best is None:
        # Every RANSAC hypothesis degenerate: scale-only seed.
        return _scale_only_fallback(P_obs_ds, P_mesh, r_obs, r_msh)

    s, R, t, rmse = best
    return {"scale": float(s), "R": R, "t": t, "rmse": rmse,
            "ok": bool(np.isfinite(rmse) and rmse < MAX_OK_RMSE)}