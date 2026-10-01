"""Canonical frame of a tracked mesh: the rotation that makes it upright.

Why: the sim's grounding stack assumes assets are converted upright
(ground_parts.py UP_BODY = [0, 0, 1], canonical_cameras' z-up look-at), and
hardware meshes keep the TRELLIS output frame -- the mug's up is 67 deg off
+z. Rather than thread a rotation through the sim, hardware builds a
render-only canonical COPY of the mesh (V_c = V @ R) that satisfies the
sim's assumption by construction, grounds on it, and maps the result back
(x_body = R @ x_c). The tracked mesh, cam_T_obj, frames.json and check_symbols
all stay in the tracked mesh's body frame; this rotation is grounding's
working frame and nothing else. `to_canonical` / `to_body` below are the one
place that algebra lives.

    R = [x | y | z]   columns are the canonical axes IN BODY COORDINATES
        z = up        table-plane normal through the registration, refined
                      by a revolution-axis fit on the mesh (accepted only
                      within UP_SNAP_MAX_DEG of the table normal, else the
                      table normal itself)
        x = front     Orient Anything's real-crop front through the same
                      registration, projected perpendicular to z (semantic
                      route; metric azimuth needs grounded parts, which this
                      frame exists to produce -- refining it here would be
                      circular). No usable front -> deterministic arbitrary
                      x, azimuth.accepted False; z is still right, so views
                      are still upright.
        y = z x x     right-handed, det +1

Up is seeded from the TABLE, not from Orient Anything: the table normal is a
measurement, while OA's up judges object semantics and called a tumbled mug
upright (learnings 2026-09-14). OA's up is recorded as a disagreement angle
only. For an object resting on the table this is also the sim's convention:
the canonical frame is the rest frame (the marker asset is its scanned rest
pose), not a semantic frame.

Pure numpy apart from the sim refinement it calls through refine_compat.
That import is deferred into canonical_frame() so that to_canonical /
to_body -- the only rotation algebra, which render_asset uses to write the
canonical copy -- import without the sim mount. refine_compat's diagnostic
ImportError (empty mount, shadowed package) still fires, at the first
canonical_frame() call.
"""
import numpy as np

# (value, basis) -- n=1 calibrations recorded as such (check_symbols convention).
PARAMS = {
    "up_snap_max_deg": (5.0, "= check_symbols up_axis_tol_deg: the refined up may not move further "
                             "from the table normal than C's own up gate tolerates; C measured the "
                             "mug's ring-fit up 3.1 deg off the table normal on 20260928_190618; n=1"),
    "n_samples":       (6000, "area-weighted surface samples for the revolution fit; the fit is "
                              "multistart least squares, cost linear in samples"),
    "sample_seed":     (0, "fixed: the frame must be reproducible run to run"),
}


def values(params=None):
    v = {k: val for k, (val, _) in PARAMS.items()}
    if params:
        v.update(params)
    return v


def to_canonical(X, R):
    """Body-frame points/directions (N,3) -> canonical frame. V_c = V @ R."""
    return np.asarray(X, float) @ np.asarray(R, float)


def to_body(Xc, R):
    """Canonical-frame points/directions (N,3) -> body frame. x_body = R @ x_c."""
    return np.asarray(Xc, float) @ np.asarray(R, float).T


def surface_samples(V, F, n, rng):
    """Area-weighted points on the triangle surface. Vertex density on a
    TRELLIS mesh is not uniform, and the revolution fit weights every point
    equally, so sample by area rather than taking vertices."""
    V = np.asarray(V, float)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("mesh has no surface area")
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return ((1 - s)[:, None] * a[tri] + (s * (1 - r2))[:, None] * b[tri]
            + (s * r2)[:, None] * c[tri])


def _unit(v, what):
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"{what} is zero or non-finite: {v.tolist()}")
    return v / n


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def canonical_frame(V, F, cam_T_obj, n_cam, oa_real=None, params=None):
    """(R, record) for the mesh (V, F) in its tracked body frame.

    cam_T_obj  4x4 registration of THIS mesh (summary pose_on_final -- the
               same pose check_symbols judges against)
    n_cam      table-plane normal in the camera frame, oriented toward the
               camera = up (frame_packet's fit_table_plane convention)
    oa_real    the decision record's Orient Anything reading of the real crop
               ({"front_cam", "up_cam", "alpha", ...}) or None. front_body is
               RECOMPUTED here through cam_T_obj rather than taken from
               decision.front, so it cannot come from a different
               registration than the one everything else uses.
    """
    from manip_bridge import refine_compat as rc     # deferred: see module docstring

    p = values(params)
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    T = np.asarray(cam_T_obj, float).reshape(4, 4)
    R_cam = T[:3, :3]
    if abs(np.linalg.det(R_cam) - 1.0) > 1e-3:
        raise ValueError(f"cam_T_obj rotation is not proper (det {np.linalg.det(R_cam):.4f})")

    # ---- up: table normal through the registration, revolution-refined
    up_table = _unit(R_cam.T @ _unit(n_cam, "table normal"), "table normal (body)")
    P = surface_samples(V, F, int(p["n_samples"]), np.random.default_rng(int(p["sample_seed"])))
    up_res = rc.refine_axis(P, up_table, "revolution", max_snap_deg=float(p["up_snap_max_deg"]))
    up = _unit(up_res.direction, "up")

    # ---- front: Orient Anything's real-crop front, semantic route only
    cands, front_body_raw, alpha, az_note = [], None, None, ""
    oa_up_angle = None
    if oa_real is None:
        az_note = "no Orient Anything reading of the real crop"
    else:
        alpha = oa_real.get("alpha")
        if oa_real.get("up_cam") is not None:
            oa_up = _unit(R_cam.T @ _unit(oa_real["up_cam"], "OA up"), "OA up (body)")
            oa_up_angle = _angle_deg(oa_up, up)
        if alpha == 0:
            az_note = "alpha 0: Orient Anything reports no confident front"
        elif oa_real.get("front_cam") is None:
            az_note = "Orient Anything reading has no front_cam"
        else:
            front_body_raw = _unit(R_cam.T @ _unit(oa_real["front_cam"], "OA front"),
                                   "OA front (body)")
            cands.append(rc.azimuth_from_semantic(front_body_raw, up))
    fr = rc.assemble_frame(up_res, cands)
    R = np.asarray(fr.R, float)

    # assemble_frame builds [x, z x x, z]; check what downstream relies on
    if not (np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1.0) < 1e-9):
        raise RuntimeError(f"assembled frame is not a rotation: det {np.linalg.det(R)}")
    if _angle_deg(R[:, 2], up) > 1e-6:
        raise RuntimeError("assembled frame's z is not the chosen up")

    az = fr.azimuth
    record = {
        "R_body_from_canon": R.tolist(),
        "convention": "columns [x=front, y=up x front, z=up] in the tracked mesh's body frame; "
                      "canonical copy V_c = V @ R; map back x_body = R @ x_c",
        "up": {
            "source": "refined" if up_res.accepted else "table_normal",
            "seed": "table_normal_through_pose",
            "direction_body": up.tolist(),
            "table_normal_body": up_table.tolist(),
            "angle_to_table_deg": _angle_deg(up, up_table),
            "fit_snap_deg": float(up_res.snap_deg),
            "fit_sigma_deg": float(up_res.sigma_deg),
            "fit_rms_m": float(up_res.residual_rms),
            "fit_inliers": int(up_res.inliers),
            "fit_note": up_res.note,
        },
        "oa_up_angle_deg": oa_up_angle,
        "azimuth": {
            "route": az.route if cands else "none",
            "accepted": bool(fr.accepted),
            "sigma_deg": float(az.sigma_deg) if fr.accepted else None,
            "conditioning": float(az.conditioning) if cands else None,
            "alpha_reported": alpha,
            "front_body_oa": None if front_body_raw is None else front_body_raw.tolist(),
            "note": az_note or (az.note if not fr.accepted else ""),
        },
        "note": fr.note,
        "params": {k: {"value": v, "basis": b} for k, (v, b) in PARAMS.items()},
    }
    if params:
        record["params_overridden"] = dict(params)
    return R, record
