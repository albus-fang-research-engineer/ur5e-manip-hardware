"""Metric scale recovery: similarity registration of a partial single-view
depth cloud (camera frame, meters) onto a TRELLIS-style canonical mesh.

Estimates (s, R, t) s.t.  p_cam ~= s * R * p_mesh + t.

Differences from v1, all driven by measured failures (reproduce with test_roundtrip.py
and test_objective.py):

  1. FPFH + RANSAC initialization is GONE. On partial-to-full it produced
     mutual-NN sets of 3-97 pairs out of ~3000 points, fell through to
     one-way matching, and returned scale seeds spanning 0.2x to 24x the
     truth. Replaced by a deterministic SO(3) grid multi-start seeded from
     the RMS-radius ratio -- which is biased low by a predictable ~10-20%
     (a single-view patch has smaller spread about its own centroid than
     the full surface does) and is corrected for.

  2. ICP is BIDIRECTIONAL with a visibility test. v1 scored only
     obs -> mesh, which is unbounded upward: inflate the mesh and slide
     the observed patch onto a locally flat region and the cost barely
     moves. Front-facing mesh points inside the observed region now also
     have to find an observation. This is what constrains scale for
     low-relief geometry; curvature alone does it for rounded objects.

  3. Acceptance is SIZE-RELATIVE. v1's fixed 6 mm gate passed solutions
     with +500% to +800% scale error. Residual at truth is ~0.9 mm here,
     so the gate is now a fraction of the estimated object diameter.

Degeneracy policy unchanged: never raises, returns ok=False on any
degenerate path so the caller's metric branch always replies.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# ---- metric-space knobs (meters) ---------------------------------------
VOXEL = 0.003
MIN_OBS_POINTS = 200
MAX_OK_REL_RMSE = 0.025   # fraction of estimated object diameter
MAX_OK_ABS_RMSE = 0.008   # meters, hard ceiling

# ---- registration knobs ------------------------------------------------
S_PARTIAL_CORRECTION = 1.15  # RMS ratio underestimates on single views
N_AXIS = 42                  # icosphere(1) vertices
N_ROLL = 8                   # in-plane samples per axis  -> 336 rotations
COARSE_ITERS = 6
COARSE_STRIDE = 4
TOPK = 8
ICP_ITERS = 40
TRIM_FRAC = 0.75
REV_WEIGHT = 0.5             # weight on the visible mesh -> obs term
VIS_MARGIN = -0.40           # cos(normal, line-of-sight) cutoff; grazing
                             # mesh points near the silhouette are excluded
                             # because mask erosion removed their support
                             # and demanding it shrinks the mesh
MESH_SAMPLES = 30_000

_FAIL = {"scale": 1.0, "R": np.eye(3), "t": np.zeros(3),
         "rmse": float("inf"), "ok": False, "path": "fail"}


def _so3_grid(n_axis=N_AXIS, n_roll=N_ROLL):
    ico = trimesh.creation.icosphere(subdivisions=1)
    axes = np.asarray(ico.vertices)[:n_axis]
    out = []
    for a in axes:
        z = a / np.linalg.norm(a)
        tmp = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
        x = np.cross(tmp, z); x /= np.linalg.norm(x)
        B = np.stack([x, np.cross(z, x), z], axis=1)
        for r in np.linspace(0, 2 * np.pi, n_roll, endpoint=False):
            c, sn = np.cos(r), np.sin(r)
            out.append(B @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.]]))
    return np.asarray(out)


_GRID = _so3_grid()


def backproject(depth_m, K, mask, erode_px=4, z_range=(0.15, 2.0)):
    """Masked metric depth -> camera-frame points. Erosion matters: depth
    bleeding at silhouette edges drags points into free space and biases
    scale upward."""
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


def _clean(points, voxel):
    """Voxel downsample + statistical outlier removal, numpy in/out."""
    import open3d as o3d
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel:
        p = p.voxel_down_sample(voxel)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(p.points)


def _umeyama(src, dst, w=None):
    """Closed-form similarity dst ~= s R src + t (Umeyama 1991), with
    optional per-correspondence weights. src should be the CLEAN set (the
    mesh): the scale estimator divides by src variance, so noise on src
    attenuates s while noise on dst does not."""
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (w[:, None] * dc).T @ sc
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (w * (sc ** 2).sum(1)).sum()
    if var_s < 1e-18:
        return np.nan, R, np.zeros(3)
    s = np.trace(np.diag(D) @ S) / var_s
    return s, R, mu_d - s * R @ mu_s


def _visible_mask(P_mesh, N_mesh, s, R, t, margin=VIS_MARGIN):
    """Front-facing test in camera frame: a mesh point is expected to be
    observed iff its outward normal points back toward the origin."""
    p_cam = s * (P_mesh @ R.T) + t
    n_cam = N_mesh @ R.T
    los = p_cam / np.linalg.norm(p_cam, axis=1, keepdims=True)
    return (n_cam * los).sum(1) < margin


def _bidir_icp(P_obs, P_mesh, N_mesh, tree_mesh, s, R, t,
               iters, rev_weight=REV_WEIGHT, trim=TRIM_FRAC):
    """Returns (s, R, t, rmse). Forward term: every observed point must lie
    on the mesh. Reverse term: every front-facing mesh point must be
    explained by an observation. The reverse term is what stops the mesh
    from inflating."""
    tree_obs = cKDTree(P_obs)
    for it in range(iters + 1):
        if not (np.isfinite(s) and s > 1e-9
                and np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
            return s, R, t, float("inf")

        # forward: obs -> mesh, evaluated in canonical units
        Q = (P_obs - t) @ R / s
        df, idx_f = tree_mesh.query(Q, k=1)
        keep_f = df <= np.quantile(df, trim)
        if keep_f.sum() < 10:
            return s, R, t, float("inf")
        src = [P_mesh[idx_f][keep_f]]
        dst = [P_obs[keep_f]]
        w = [np.ones(int(keep_f.sum()))]

        # reverse: visible mesh -> obs, evaluated in metric units
        vis = _visible_mask(P_mesh, N_mesh, s, R, t)
        dr = np.array([])
        if vis.sum() >= 10 and rev_weight > 0:
            Mv = P_mesh[vis]
            dr, idx_r = tree_obs.query(s * (Mv @ R.T) + t, k=1)
            keep_r = dr <= np.quantile(dr, trim)
            if keep_r.sum() >= 10:
                src.append(Mv[keep_r])
                dst.append(P_obs[idx_r][keep_r])
                # match total influence, then apply the user weight
                w.append(np.full(int(keep_r.sum()),
                                 rev_weight * keep_f.sum() / keep_r.sum()))
                dr = dr[keep_r]

        if it == iters:
            break
        s, R, t = _umeyama(np.concatenate(src), np.concatenate(dst),
                           np.concatenate(w))

    fwd = (df[keep_f] * s) ** 2
    rmse = float(np.sqrt(np.concatenate([fwd, dr ** 2]).mean()))
    return s, R, t, rmse


def register_similarity(P_obs, mesh_trimesh, rev_weight=REV_WEIGHT,
                        return_transform=False):
    """P_obs: (N,3) camera-frame meters. mesh_trimesh: canonical (unit) mesh.

    Returns {scale, R, t, rmse, rel_rmse, ok, path}. With
    return_transform=True also returns 'T', the 4x4 taking canonical mesh
    vertices straight to camera-frame meters, so the caller can do
    mesh.copy().apply_transform(T) and sample interaction points on it.
    """
    if P_obs is None or len(P_obs) < MIN_OBS_POINTS:
        return dict(_FAIL)

    pts, fid = mesh_trimesh.sample(MESH_SAMPLES, return_index=True)
    P_mesh = np.asarray(pts, dtype=np.float64)
    N_mesh = np.asarray(mesh_trimesh.face_normals[fid], dtype=np.float64)
    N_mesh /= np.linalg.norm(N_mesh, axis=1, keepdims=True) + 1e-12
    tree_mesh = cKDTree(P_mesh)

    P = _clean(P_obs, VOXEL)
    if len(P) < MIN_OBS_POINTS:
        return dict(_FAIL)

    r_obs = float(np.sqrt(P.var(0).sum()))
    r_msh = float(np.sqrt(P_mesh.var(0).sum()))
    if not (r_obs > 1e-6 and r_msh > 1e-6):
        return dict(_FAIL)

    s0 = (r_obs / r_msh) * S_PARTIAL_CORRECTION
    c_obs, c_msh = P.mean(0), P_mesh.mean(0)

    # coarse pass over the rotation grid on a subsample
    sub = P[::COARSE_STRIDE]
    if len(sub) < 50:
        sub = P
    # coarse ranking uses a decimated mesh and the forward term only:
    # the reverse term costs a visibility pass over every mesh sample and
    # is not needed to discriminate grossly wrong rotations.
    step = max(1, len(P_mesh) // 4000)
    Pm_c, Nm_c = P_mesh[::step], N_mesh[::step]
    tree_c = cKDTree(Pm_c)
    cand = []
    for R0 in _GRID:
        t0 = c_obs - s0 * R0 @ c_msh
        s_, R_, t_, e_ = _bidir_icp(sub, Pm_c, Nm_c, tree_c,
                                    s0, R0, t0, COARSE_ITERS, 0.0)
        cand.append((s_, R_, t_, e_))
    cand.sort(key=lambda c: c[3] if np.isfinite(c[3]) else np.inf)

    best = None
    for s, R, t, _ in cand[:TOPK]:
        out = _bidir_icp(P, P_mesh, N_mesh, tree_mesh, s, R, t,
                         ICP_ITERS, rev_weight)
        if best is None or out[3] < best[3]:
            best = out
    if best is None or not np.isfinite(best[3]):
        return dict(_FAIL)

    s, R, t, rmse = best
    diam = s * float(np.linalg.norm(P_mesh.max(0) - P_mesh.min(0)))
    rel = rmse / max(diam, 1e-9)
    ok = bool(np.isfinite(rmse) and s > 1e-9
              and rmse < MAX_OK_ABS_RMSE and rel < MAX_OK_REL_RMSE)
    out = {"scale": float(s), "R": R, "t": t, "rmse": rmse,
           "rel_rmse": float(rel), "ok": ok, "path": "multistart"}
    if return_transform:
        T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
        out["T"] = T
    return out