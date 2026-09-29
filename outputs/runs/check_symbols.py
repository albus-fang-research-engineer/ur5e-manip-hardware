#!/usr/bin/env python3
"""Milestone-1 acceptance instrument: project the symbols of a frames.json
through a registration's cam_T_obj onto the saved run_scene frame and check
each one against the measured depth and the mesh itself.

`reproject_check.py` answers "is the REGISTRATION right" (whole-mesh
silhouette + depth residual). This tool answers "are the SYMBOLS right,
given that registration": per-point pixel placement and depth residuals,
per-axis angle against a named physical referent. It is the gate grounding
output must pass before any VLM call consumes real geometry.

Residual semantics (per point, both through the same cam_T_obj):
  vs_measured = z_symbol - median(depth at the symbol's pixel window)
  vs_mesh     = z_symbol - front surface of the mesh along that ray
                (vertex z-buffer, window median; assumes a densely
                reconstructed mesh -- TRELLIS/Any6D outputs are)
  diff        = vs_measured - vs_mesh = z_front_mesh - depth_measured
                -- symbol-independent: the local registration/fidelity
                residual at that pixel (z_symbol cancels).

Gating depends on the symbol's `depth_check` field in frames.json
(default "surface"):
  surface   on the mesh surface: |vs_mesh| and |vs_measured| small.
  interior  a centerline/interior point: vs_mesh in [lo, hi] ("inside the
            mesh along this ray"; a correct point reads ~ +half the local
            thickness on BOTH residuals) and |diff| small.
  skip      free-space symbol (e.g. opening_center: the ray hits the far
            inner wall, both residuals are meaningless): drawn and gated on
            in-mask placement only.
All points must land inside the (slightly dilated) instance mask.

Axes are gated per named referent: up_axis against the frame packet's table
plane normal (camera frame; fit_table_plane orients it toward the camera,
i.e. upward). Other axes are report-only for now -- a front-class referent
is decision.front, itself an estimate; a tight bound there would fail
correct groundings on decider noise.

Triage on a failure:
  diff large                -> registration or mesh fidelity at that pixel
                               (undersized handle etc.): reach for the ruler,
                               not the grounding code.
  vs_mesh out of its band   -> the symbol is not on/inside the mesh where it
                               claims to be: grounding bug.
  not_in_mask / off_mesh    -> gross placement error: grounding bug.

Preconditions: single-layer, non-occluded scene capture (the yaw-decider
precondition; there is deliberately no occlusion machinery here).

  python3 check_symbols.py outputs/runs/<run> --key m2 \\
      --frames outputs/runs/fixtures/mug_frames_hand.json
  # pose/mesh default to summary["objects"][<key>]["pose_on_final"];
  # --pose-json fp_mug.json overrides for driver-world registrations.

Writes <run>/symbols_<key>.png (overlay) and <run>/symbols_<key>.json
(per-symbol record, thresholds included), prints a table, exits nonzero on
a failed verdict.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

# (value, basis) -- n=1 calibrations are recorded as such, revisited when the
# teapot runs; do not turn these into scaling laws off one object.
PARAMS = {
    "surface_tol_m":    (0.012, "RealSense D435 + accepted-registration depth residual p90 ~8 mm; n=1 (mug, 20260928_190618)"),
    "interior_lo_m":    (-0.005, "vertex z-buffer window discretisation slack"),
    "interior_hi_m":    (0.030, "mug handle thickness ~15 mm + margin; n=1"),
    "diff_tol_m":       (0.012, "same registration/fidelity bound as surface_tol_m"),
    "up_axis_tol_deg":  (5.0,  "matches frame_packet plane_up_warn_deg; table fit + registration; n=1"),
    "window_px":        (5,    "odd side of the depth / z-buffer sampling window"),
    "mask_dilate_frac": (0.02, "in-mask gate dilation, fraction of mask bbox diagonal"),
}


def values(params=None):
    v = {k: val for k, (val, _) in PARAMS.items()}
    if params:
        v.update(params)
    return v


def _py(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def project(K, X):
    """(N,3) camera-frame points -> (N,2) pixel coords + (N,) z."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    z = X[:, 2]
    px = np.c_[K[0, 0] * X[:, 0] / z + K[0, 2],
               K[1, 1] * X[:, 1] / z + K[1, 2]]
    return px, z


def vertex_zbuffer(V_cam, K, H, W):
    """Front-surface depth per pixel from projected vertices (nearest wins).
    Same mechanism as reproject_check.py; valid where any vertex lands."""
    z = V_cam[:, 2]
    keep = z > 1e-6
    px, z = project(K, V_cam[keep])[0], z[keep]
    ui = np.round(px).astype(int)
    ok = (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (ui[ok, 1], ui[ok, 0]), z[ok])
    return zbuf


def _window(img, u, v, half):
    H, W = img.shape[:2]
    return img[max(0, v - half):min(H, v + half + 1),
               max(0, u - half):min(W, u + half + 1)]


def evaluate(frames, T, V_body, mask, depth, K, plane, params=None):
    """Pure evaluation -> record dict. mask: bool HxW; depth: meters, 0/nan
    invalid; plane: {"n": [...] } in the CAMERA frame or None."""
    prm = values(params)
    H, W = mask.shape
    R, t = np.asarray(T, np.float64)[:3, :3], np.asarray(T, np.float64)[:3, 3]
    V_cam = V_body @ R.T + t
    zbuf = vertex_zbuffer(V_cam, K, H, W)
    diam = float(np.linalg.norm(np.ptp(V_body, axis=0)))
    half = int(prm["window_px"]) // 2
    depth = np.where(np.isfinite(depth), depth, 0.0)

    ys, xs = np.nonzero(mask)
    bbox_diag = float(np.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 0.0
    it = max(1, int(round(prm["mask_dilate_frac"] * bbox_diag)))
    mask_dil = binary_dilation(mask, iterations=it)

    rec = {"params": {k: {"value": v, "basis": PARAMS[k][1]} for k, v in values().items()},
           "mesh_diameter_m": diam, "points": {}, "axes": {}, "quantities": {}}
    if params:
        for k, v in params.items():
            rec["params"][k] = {"value": v, "basis": "override"}
    failures = []

    for name, p in frames.get("points", {}).items():
        check = p.get("depth_check", "surface")
        r = {"depth_check": check, "xyz_body": list(map(float, p["xyz"]))}
        X_cam = R @ np.asarray(p["xyz"], np.float64) + t
        reasons = []
        if X_cam[2] <= 1e-6:
            r.update(status="fail", fail_reasons=["behind_camera"])
            rec["points"][name] = r
            failures.append(f"point {name}: behind_camera")
            continue
        px, z_sym = project(K, X_cam[None])
        u, v = int(round(px[0, 0])), int(round(px[0, 1]))
        z_sym = float(z_sym[0])
        r.update(px=[float(px[0, 0]), float(px[0, 1])], z_sym_m=z_sym)
        if not (0 <= u < W and 0 <= v < H):
            r.update(status="fail", fail_reasons=["off_image"])
            rec["points"][name] = r
            failures.append(f"point {name}: off_image")
            continue

        r["in_mask"] = bool(mask_dil[v, u])
        if not r["in_mask"]:
            reasons.append("not_in_mask")

        zw = _window(zbuf, u, v, half)
        zw = zw[np.isfinite(zw)]
        vs_mesh = float(z_sym - np.median(zw)) if len(zw) else None
        r["vs_mesh_mm"] = None if vs_mesh is None else 1e3 * vs_mesh

        dw = _window(depth, u, v, half)[_window(mask, u, v, half)]
        dw = dw[dw > 0]
        r["n_depth_px"] = int(len(dw))
        vs_meas = float(z_sym - np.median(dw)) if len(dw) else None
        r["vs_measured_mm"] = None if vs_meas is None else 1e3 * vs_meas
        diff = None if (vs_meas is None or vs_mesh is None) else vs_meas - vs_mesh
        r["diff_mm"] = None if diff is None else 1e3 * diff

        if check != "skip":
            if vs_mesh is None:
                reasons.append("off_mesh")          # no mesh surface near the pixel
            elif check == "surface":
                if abs(vs_mesh) > prm["surface_tol_m"]:
                    reasons.append("vs_mesh_band")
                if vs_meas is not None and abs(vs_meas) > prm["surface_tol_m"]:
                    reasons.append("vs_measured")
            elif check == "interior":
                if not (prm["interior_lo_m"] <= vs_mesh <= prm["interior_hi_m"]):
                    reasons.append("vs_mesh_band")
                if diff is not None and abs(diff) > prm["diff_tol_m"]:
                    reasons.append("diff")
            else:
                reasons.append(f"unknown_depth_check:{check}")
            if vs_meas is None:
                r["note"] = "no valid depth in window: vs_measured not gated"

        r["status"] = "fail" if reasons else ("pass" if check != "skip" else "pass(skip)")
        r["fail_reasons"] = reasons
        rec["points"][name] = r
        if reasons:
            failures.append(f"point {name}: {','.join(reasons)}")

    # axes ---------------------------------------------------------------
    anchor_body = np.asarray(frames.get("points", {}).get("opening_center", {})
                             .get("xyz", V_body.mean(axis=0)), np.float64)
    n_up = None
    if plane is not None and "n" in plane:
        n_up = np.asarray(plane["n"], np.float64)
        n_up = n_up / np.linalg.norm(n_up)          # fit_table_plane: toward camera = up
    for name, a in frames.get("axes", {}).items():
        d_body = np.asarray(a["xyz"], np.float64)
        d_body = d_body / np.linalg.norm(d_body)
        d_cam = R @ d_body
        r = {"xyz_body": d_body.tolist(), "anchor_body": anchor_body.tolist()}
        if name == "up_axis" and n_up is not None:
            ang = float(np.degrees(np.arccos(np.clip(d_cam @ n_up, -1, 1))))
            gated = ang <= prm["up_axis_tol_deg"]
            r.update(referent="table_plane_normal", angle_deg=ang,
                     status="pass" if gated else "fail")
            if not gated:
                r["fail_reasons"] = ["angle"]
                failures.append(f"axis {name}: {ang:.1f} deg vs table normal")
        else:
            r.update(referent=None, status="report_only",
                     note="no measured referent (front-class referents are "
                          "decider estimates; gate deferred)" if name != "up_axis"
                     else "no table plane in packet")
        rec["axes"][name] = r

    for name, q in frames.get("quantities", {}).items():
        rec["quantities"][name] = {"value": q.get("value"), "status": "drawn_only"}

    rec["verdict"] = {"pass": not failures, "failures": failures}
    return rec, {"zbuf": zbuf, "mask_dil": mask_dil, "anchor_body": anchor_body,
                 "R": R, "t": t, "diam": diam}


def draw_overlay(rgb, rec, aux, frames, K, out_png):
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)
    mask_edge = aux["mask_dil"] & ~binary_erosion(aux["mask_dil"], iterations=2)
    arr = np.array(im)
    arr[mask_edge] = (0, 200, 0)
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    col = {"pass": (0, 220, 0), "pass(skip)": (0, 200, 220), "fail": (255, 40, 40)}
    for name, r in rec["points"].items():
        if "px" not in r:
            continue
        u, v = r["px"]
        c = col.get(r["status"], (255, 255, 0))
        dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=c, width=2)
        dr.line([u - 7, v, u + 7, v], fill=c, width=1)
        dr.line([u, v - 7, u, v + 7], fill=c, width=1)
        lab = name
        if r.get("vs_measured_mm") is not None:
            lab += f" {r['vs_measured_mm']:+.0f}mm"
        dr.text((u + 6, v + 4), lab, fill=c)
    R, t, L = aux["R"], aux["t"], 0.6 * aux["diam"]
    for name, a in rec["axes"].items():
        p0 = R @ aux["anchor_body"] + t
        p1 = p0 + R @ (np.asarray(a["xyz_body"]) * L)
        (px, _) = project(K, np.stack([p0, p1]))
        c = (255, 220, 0) if a["status"] != "fail" else (255, 40, 40)
        dr.line([tuple(px[0]), tuple(px[1])], fill=c, width=2)
        lab = name + (f" {a['angle_deg']:.1f}deg" if "angle_deg" in a else "")
        dr.text(tuple(px[1] + 3), lab, fill=c)
    oc, up = frames.get("points", {}).get("opening_center"), frames.get("axes", {}).get("up_axis")
    rr = frames.get("quantities", {}).get("rim_radius", {}).get("value")
    if oc and up and rr:
        d = np.asarray(up["xyz"], np.float64)
        d = d / np.linalg.norm(d)
        e1 = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        th = np.linspace(0, 2 * np.pi, 64)
        C = np.asarray(oc["xyz"]) + rr * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
        (px, _) = project(K, C @ R.T + t)
        dr.line([tuple(p) for p in px] + [tuple(px[0])], fill=(220, 0, 220), width=1)
    im.save(out_png)


def resolve_mesh(args, rd, rec_obj):
    for cand in ([args.mesh] if args.mesh else []) + \
                ([rec_obj.get("pose_on_final", {}).get("mesh")] if rec_obj else []):
        if cand and os.path.exists(cand):
            return cand
    for pat in (f"objects/{args.key}/*final*.obj", f"objects/{args.key}/*.obj",
                f"objects/{args.key}/*.glb"):
        g = sorted(glob.glob(os.path.join(rd, pat)))
        if g:
            return g[0]
    sys.exit(f"no mesh found: pass --mesh (tried the summary record and objects/{args.key}/)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frames", required=True, help="frames.json to check")
    ap.add_argument("--key", default="m2", help="mark key in summary['objects'] (mask_<key>.png)")
    ap.add_argument("--object", default=None, help="legacy mask name fallback (mask_<object>.png)")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--pose-json", default=None, help='JSON with "cam_T_obj" (driver-world override)')
    ap.add_argument("--out-prefix", default=None, help="default <run>/symbols_<key>")
    a = ap.parse_args(argv)
    rd = a.run_dir

    with open(os.path.join(rd, "summary.json")) as f:
        summary = json.load(f)
    K = np.asarray(summary["frame"]["K"], np.float64)
    plane = summary.get("packet", {}).get("plane")
    rec_obj = summary.get("objects", {}).get(a.key)

    if a.pose_json:
        T = np.asarray(json.load(open(a.pose_json))["cam_T_obj"], np.float64).reshape(4, 4)
        pose_src = a.pose_json
    else:
        if not rec_obj or "pose_on_final" not in rec_obj:
            sys.exit(f"summary has no objects[{a.key}].pose_on_final: pass --pose-json")
        T = np.asarray(rec_obj["pose_on_final"]["cam_T_obj"], np.float64).reshape(4, 4)
        pose_src = f"summary objects[{a.key}].pose_on_final"

    mesh_path = resolve_mesh(a, rd, rec_obj)
    V = np.asarray(trimesh.load(mesh_path, force="mesh").vertices, np.float64)

    rgb = np.array(Image.open(os.path.join(rd, "rgb.png")).convert("RGB")).copy()
    for mname in [f"mask_{a.key}.png"] + ([f"mask_{a.object.replace(' ', '_')}.png"] if a.object else []):
        mp = os.path.join(rd, mname)
        if os.path.exists(mp):
            mask = np.array(Image.open(mp).convert("L")) > 0
            break
    else:
        sys.exit(f"no mask found (tried mask_{a.key}.png"
                 + (f", mask_{a.object}.png" if a.object else "") + ")")
    depth = np.array(Image.open(os.path.join(rd, "depth_mm.png")))
    if depth.dtype != np.uint16:
        sys.exit(f"depth_mm.png decoded as {depth.dtype}, expected uint16")
    depth = depth.astype(np.float64) * 1e-3

    frames = json.load(open(a.frames))
    rec, aux = evaluate(frames, T, V, mask, depth, K, plane)
    rec.update(run=os.path.abspath(rd), key=a.key, frames=os.path.abspath(a.frames),
               mesh=os.path.abspath(mesh_path), pose_source=pose_src,
               plane_frame="camera (fit_table_plane, normal toward camera = up)")

    pre = a.out_prefix or os.path.join(rd, f"symbols_{a.key}")
    draw_overlay(rgb, rec, aux, frames, K, pre + ".png")
    with open(pre + ".json", "w") as f:
        json.dump(rec, f, indent=1, default=_py)

    print(f"pose: {pose_src}\nmesh: {mesh_path}")
    print(f"{'symbol':<18s}{'check':<10s}{'px':<14s}{'vs_meas':>9s}{'vs_mesh':>9s}"
          f"{'diff':>8s}  status")
    for name, r in rec["points"].items():
        px = "-" if "px" not in r else f"{r['px'][0]:.0f},{r['px'][1]:.0f}"
        fmt = lambda k: "-" if r.get(k) is None else f"{r[k]:+.1f}"
        print(f"{name:<18s}{r['depth_check']:<10s}{px:<14s}{fmt('vs_measured_mm'):>9s}"
              f"{fmt('vs_mesh_mm'):>9s}{fmt('diff_mm'):>8s}  {r['status']}"
              + ("" if not r["fail_reasons"] else f" ({','.join(r['fail_reasons'])})"))
    for name, r in rec["axes"].items():
        ang = f"{r['angle_deg']:.1f} deg vs {r['referent']}" if "angle_deg" in r else r["status"]
        print(f"axis {name:<13s}{ang}  {r['status']}")
    v = rec["verdict"]
    print(("PASS" if v["pass"] else "FAIL: " + "; ".join(v["failures"]))
          + f"   -> {pre}.png / .json")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
