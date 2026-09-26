"""SAM3 instances on a real frame -> the objects that get marks.

Set-of-mark on hardware: a label-free SAM3 prompt ("object") returns every
instance above a permissive score; this module decides which instances are
objects the pipeline can act on, and writes them as the sim's canonical
mark set (marks_compat.build_marks) so plan_stages.py reads it unchanged.

Rejection is geometric, not by score. The score threshold is set low on
purpose -- a missing mark is an object the VLM cannot choose, an extra one
is a distractor it can ignore -- and depth does the rejecting:

    fit_table_plane   RANSAC plane over the frame's depth, oriented so the
                      camera is on the positive side (height = n.p + d > 0
                      above the table), with the in-plane extent of its
                      inliers (1st..99th percentile)

    select_instances  per instance, every reason that applies is recorded:
      below_score       score < score_min
      below_min_area    area < min_area_frac of the image
      truncated         bbox within truncation_px of the image border: the
                        pipeline acts only on fully visible objects (same
                        principle as the single-layer precondition)
      arm               >= arm_frac of the instance inside the arm mask
                        (cuRobo robot_mask, or the "robot arm" prompt as a
                        fallback). A fraction, not IoU: the instance may
                        extend past the arm mask (gripper)
      no_depth          too few valid depth pixels to measure it
      table             >= table_frac of its points within plane_tol_m of
                        the plane: a surface, not an object on it
      out_of_workspace  median height not in (plane_tol_m, ws_max_height_m],
                        or median in-plane position outside the table
                        extent + ws_margin_m: walls, floor past the table
                        edge, shelves -- whatever the score
    then, among instances with no reason so far:
      duplicate_of:<i>  IoU >= dup_iou with a higher-scoring kept instance
      part_of:<i>       >= part_of of it inside a larger kept instance (a
                        lid in a teapot). Safe ONLY under the single-layer,
                        non-occluded precondition: without it, 2D containment
                        could be an object in front of another

    write_mark_set    build_marks + marks.sam3.json (mark id -> SAM3 prompt,
                      score). The prompt strings live in that separate file
                      because they must never reach the VLM: plan_stages.py
                      reads only marks.json (and marks.gt.json in sim).

Every parameter carries a basis string, and the table goes into the run's
record, so each run says which thresholds were measured (n=1 so far) and
which are still provisional.
"""
import json
from pathlib import Path

import numpy as np

REF_FRAME = "20260903_203531"
PARAMS = {
    "score_min": (0.2, f"n=1 ({REF_FRAME}): generic 'object' scores 0.10-0.43, real objects "
                       "0.29-0.37; set in the widest gap (0.166-0.285) because a missing mark "
                       "is unrecoverable and depth rejects background"),
    "bg_score_min": (0.4, f"n=1 ({REF_FRAME}): 'robot arm' scored 0.527; a specific noun "
                          "scores high, so the fallback arm mask needs no permissive threshold"),
    "min_area_frac": (150 / (640 * 480), "sim MIN_AREA_PX 150 at 640x480 as a fraction "
                                         f"(450 px at 1280x720); n=1, nothing near it on {REF_FRAME}"),
    "truncation_px": (2, f"n=1 ({REF_FRAME}): three instances touch x=0 or y=0"),
    "arm_frac": (0.5, f"n=1 ({REF_FRAME}): the arm instance is 0.854 inside the 'robot arm' mask"),
    "dup_iou": (0.8, f"partly measured ({REF_FRAME}): cross-prompt duplicates were IoU >= 0.995; "
                     "generic 'object' produced none"),
    "part_of": (0.9, "provisional, untested: needs the multi-object frame (lid, handle)"),
    "plane_tol_m": (0.008, "provisional: ~3 sigma of D435 depth noise near 0.5 m; "
                           "flat objects (a marker lying down) untested"),
    "plane_min_inlier_frac": (0.15, "provisional: below this the dominant plane is not trusted "
                                    "and geometric tests are skipped (recorded)"),
    "table_frac": (0.8, "provisional, untested"),
    "ws_max_height_m": (0.5, "provisional: workspace ceiling above the table"),
    "ws_margin_m": (0.02, "provisional: lateral slack beyond the table's inlier extent"),
    "min_depth_px": (100, "provisional"),
    "min_depth_frac": (0.2, "provisional"),
    "ransac_iters": (300, "fixed; seeded, so the fit is deterministic per frame"),
}


def values(params=None):
    p = {k: v for k, (v, _) in PARAMS.items()}
    if params:
        unknown = set(params) - set(p)
        if unknown:
            raise KeyError(f"unknown scene_marks params: {sorted(unknown)}")
        p.update(params)
    return p


def params_report(params=None):
    """{name: {value, basis}} for the run record; overrides are marked."""
    v = values(params)
    return {k: {"value": v[k], "basis": b if not params or k not in params else "override: " + b}
            for k, (_, b) in PARAMS.items()}


# ---------------------------------------------------------------- geometry

def backproject(depth, K, mask=None):
    """Nx3 camera-frame points (OpenCV: x right, y down, z forward) for the
    valid depth pixels, optionally inside `mask`."""
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)


def _plane_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1)], axis=1)          # 3x2: in-plane coordinates


def fit_table_plane(depth, K, params=None, up_hint=None, max_points=20000, seed=0):
    """Dominant plane of the frame, or None if there aren't enough points.

    Returns {n, d, inlier_frac, n_points, n_inliers, rms_m, basis (3x2),
    extent_lo, extent_hi}. A low rms with a small n_inliers is the signature
    of a fit that grabbed a box top instead of the table.
    plus angle_to_up_deg when `up_hint` (base +z in the camera frame, from
    T_base_cam) is given -- recorded as a check, not used as a gate."""
    P = values(params)
    pts = backproject(np.asarray(depth, np.float64), np.asarray(K, np.float64))
    if len(pts) < 100:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    tol = P["plane_tol_m"]
    best = None
    for _ in range(int(P["ransac_iters"])):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        cnt = np.count_nonzero(np.abs(pts @ n - n @ a) < tol)
        if best is None or cnt > best[0]:
            best = (cnt, n, -n @ a)
    if best is None:
        return None
    _, n, d = best
    for _ in range(2):                                      # least-squares refit on inliers
        inl = pts[np.abs(pts @ n + d) < tol]
        if len(inl) < 3:
            return None
        c = inl.mean(axis=0)
        n = np.linalg.svd(inl - c, full_matrices=False)[2][2]
        d = -n @ c
    if d < 0:                                               # camera on the positive side
        n, d = -n, -d
    r = pts @ n + d
    inl = pts[np.abs(r) < tol]
    E = _plane_basis(n)
    uv = inl @ E
    out = {"n": n.tolist(), "d": float(d), "inlier_frac": len(inl) / len(pts),
           "n_points": int(len(pts)), "n_inliers": int(len(inl)),
           "rms_m": float(np.sqrt(np.mean(r[np.abs(r) < tol] ** 2))), "basis": E.tolist(),
           "extent_lo": np.percentile(uv, 1, axis=0).tolist(),
           "extent_hi": np.percentile(uv, 99, axis=0).tolist()}
    if up_hint is not None:
        u = np.asarray(up_hint, float) / np.linalg.norm(up_hint)
        out["angle_to_up_deg"] = float(np.degrees(np.arccos(np.clip(abs(u @ n), -1, 1))))
    return out


def instance_geometry(mask, depth, K, plane, params=None):
    P = values(params)
    pts = backproject(depth, K, mask)
    area = int(mask.sum())
    g = {"n_valid": int(len(pts)), "valid_frac": len(pts) / max(area, 1)}
    if len(pts) == 0:
        return g
    n, d = np.asarray(plane["n"]), plane["d"]
    h = pts @ n + d
    uv = pts @ np.asarray(plane["basis"])
    g.update(median_h=float(np.median(h)), in_band_frac=float(np.mean(np.abs(h) < P["plane_tol_m"])),
             median_uv=np.median(uv, axis=0).tolist())
    return g


# --------------------------------------------------------------- selection

def _summary(it):
    g = it.get("geometry")
    return {"idx": it["idx"], "prompt": it["prompt"], "score": round(it["score"], 4),
            "area": it["area"], "bbox": it["bbox"], "reasons": it["reasons"],
            "geometry": None if g is None else {k: (np.round(v, 4).tolist() if isinstance(v, list)
                                                    else round(v, 4) if isinstance(v, float) else v)
                                                for k, v in g.items()}}


def select_instances(instances, depth=None, K=None, plane=None, arm_mask=None,
                     arm_source=None, params=None):
    """instances: [{prompt, score, mask (HxW bool)}]. Returns (kept, report).

    kept: the input dicts that survive, with idx (1-based, score order --
    the probe's numbering for the same input), area, bbox, geometry.
    report: {params, geometry_used, geometry_note, arm_source, plane,
    instances: [summary per instance, reasons empty when kept]}."""
    P = values(params)
    insts = sorted(({**it, "mask": np.asarray(it["mask"], bool), "score": float(it["score"])}
                    for it in instances), key=lambda it: -it["score"])
    if not insts:
        return [], {"params": params_report(params), "geometry_used": False,
                    "geometry_note": "no instances", "arm_source": arm_source,
                    "plane": plane, "instances": []}
    H, W = insts[0]["mask"].shape
    if depth is None or K is None or plane is None:
        geo_note = "skipped: no depth / K / plane"
    elif plane["inlier_frac"] < P["plane_min_inlier_frac"]:
        geo_note = (f"skipped: plane inlier_frac {plane['inlier_frac']:.2f} "
                    f"< {P['plane_min_inlier_frac']}")
    else:
        geo_note = "ok"
    use_geo = geo_note == "ok"
    if use_geo:
        depth = np.asarray(depth, np.float64)
        K = np.asarray(K, np.float64)
        lo = np.asarray(plane["extent_lo"]) - P["ws_margin_m"]
        hi = np.asarray(plane["extent_hi"]) + P["ws_margin_m"]

    for i, it in enumerate(insts, start=1):
        m = it["mask"]
        it["idx"], it["reasons"], it["geometry"] = i, [], None
        it["area"] = int(m.sum())
        if it["area"] == 0:
            it["bbox"] = None
            it["reasons"].append("empty")
            continue
        vs, us = np.nonzero(m)
        x0, y0, x1, y1 = int(us.min()), int(vs.min()), int(us.max()), int(vs.max())
        it["bbox"] = [x0, y0, x1, y1]
        r = it["reasons"]
        if it["score"] < P["score_min"]:
            r.append("below_score")
        if it["area"] < P["min_area_frac"] * H * W:
            r.append("below_min_area")
        t = P["truncation_px"]
        if x0 <= t or y0 <= t or x1 >= W - 1 - t or y1 >= H - 1 - t:
            r.append("truncated")
        if arm_mask is not None and (m & arm_mask).sum() / it["area"] >= P["arm_frac"]:
            r.append("arm")
        if use_geo:
            g = it["geometry"] = instance_geometry(m, depth, K, plane, params)
            if g["n_valid"] < P["min_depth_px"] or g["valid_frac"] < P["min_depth_frac"]:
                r.append("no_depth")
            elif g["in_band_frac"] >= P["table_frac"]:
                r.append("table")
            elif (not P["plane_tol_m"] < g["median_h"] <= P["ws_max_height_m"]
                  or np.any(np.asarray(g["median_uv"]) < lo)
                  or np.any(np.asarray(g["median_uv"]) > hi)):
                r.append("out_of_workspace")

    kept = []                                               # duplicates: score order
    for it in (it for it in insts if not it["reasons"]):
        for k in kept:
            inter = (it["mask"] & k["mask"]).sum()
            if inter / (it["area"] + k["area"] - inter) >= P["dup_iou"]:
                it["reasons"].append(f"duplicate_of:{k['idx']}")
                break
        else:
            kept.append(it)
    final = []                                              # parts: largest first
    for it in sorted(kept, key=lambda it: -it["area"]):
        for k in final:
            if (it["mask"] & k["mask"]).sum() / it["area"] >= P["part_of"]:
                it["reasons"].append(f"part_of:{k['idx']}")
                break
        else:
            final.append(it)
    final.sort(key=lambda it: it["idx"])

    report = {"params": params_report(params), "geometry_used": use_geo, "geometry_note": geo_note,
              "arm_source": arm_source if arm_mask is not None else None,
              "plane": plane, "instances": [_summary(it) for it in insts]}
    return final, report


# ------------------------------------------------------------------ output

def write_mark_set(rgb, kept, out_dir):
    """The sim's canonical mark set from `kept`, plus marks.sam3.json.
    Returns (MarkSet, {mark_id: instance idx}). IDs are build_marks' reading
    order; every kept instance gets exactly one mark (selection already
    applied the area floor, so build_marks' own floor is disabled)."""
    from manip_bridge.marks_compat import build_marks        # lazy: selection needs no sim

    out_dir = Path(out_dir)
    ms = build_marks(np.asarray(rgb, np.uint8), [it["mask"] for it in kept], "sam", out_dir,
                     min_area=1)
    if len(ms.ids()) != len(kept):
        raise RuntimeError(f"build_marks wrote {len(ms.ids())} marks for {len(kept)} instances")
    id_map, unclaimed = {}, list(kept)
    for mid in ms.ids():
        m = ms.load_mask(mid)
        hit = [it for it in unclaimed if np.array_equal(it["mask"], m)]
        if len(hit) != 1:
            raise RuntimeError(f"mark {mid}: {len(hit)} instances with an identical mask")
        unclaimed.remove(hit[0])
        id_map[mid] = hit[0]["idx"]
    by_idx = {it["idx"]: it for it in kept}
    (out_dir / "marks.sam3.json").write_text(json.dumps({
        "note": "SAM3 provenance per mark. Never read by prompt builders: prompt strings "
                "reaching the VLM would turn marking into naming.",
        "marks": {str(mid): {"instance": i, "prompt": by_idx[i]["prompt"],
                             "score": round(by_idx[i]["score"], 4)}
                  for mid, i in id_map.items()}}, indent=2) + "\n")
    return ms, id_map
