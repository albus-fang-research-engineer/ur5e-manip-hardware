"""Mask-conditioned re-rank of FoundationPose's refined hypotheses.

Why. On a body of revolution with a small attached part (mug + handle,
teapot + spout) FoundationPose's scorer is flat across yaw: its inputs (RGB +
XYZ crops at 160x160) barely change when the object spins about its axis, so
the pick among the ~250 refined hypotheses is decided by noise, and it can
land with the handle hidden behind the body. Global silhouette statistics
against the SAM mask (IoU, recall, precision) do not separate the yaw
families either: the part is small relative to the body, so what it adds
when correctly placed is the same size as the body-fit jitter between
hypotheses. Measured on run 20260903_203531: IoU 0.892 wrong vs 0.889 right.

What does separate them is placement of the part, conditioned on the body:

  1. The scorer's top-K hypotheses agree on the body (it is good at
     translation and tilt) and disagree only about the part. Their
     CONSENSUS silhouette -- pixels covered by >= consensus_frac of them,
     dilated by a fraction of the mask's bounding-box diagonal -- is the
     body, without any part detector.
  2. U = mask minus consensus is the region of the mask the body cannot
     explain: the part, if the mask contains it. Pixels whose measured depth
     is farther than half the mesh diameter from the consensus region's
     median depth are dropped from U (a SAM leak onto the table or a shadow
     is not the object).
  3. Per hypothesis, expl = |sil AND U| / |U|: how much of the unexplained
     mask its silhouette covers. Wrong-yaw hypotheses score 0.00 (part
     hidden behind the body); correct-yaw ones score well above (0.5-0.7 on
     the mug frame). A tumbled hypothesis can cover U with its BODY and
     score 1.0 -- those have poor precision (|sil AND mask| / |sil|), hence
     the precision floor.
  4. Survivors: precision >= precision_floor, scorer score within
     scorer_margin of the pick's (the scorer is overridden only where it is
     flat: on the mug frame the correct family sits 0.25 below the pick, the
     inverted family 1.4 below), and -- when rendered depth is available -- a
     DEPTH gate: the fraction of
     silhouette pixels whose rendered depth is off the measured depth by more
     than depth_bad_diam * diameter must stay below depth_bad_max. A silhouette
     cannot tell an upright cup from an inverted one (a frustum's outline is
     the same either way, and the handle lands in the same place), but the
     visible cavity is centimetres deep where the inverted mesh puts a flat
     base: measured on run 20260903_203531, that fraction is 0.02 for the
     upright pose and 0.44 (min 0.08) for the inverted family. THEN, over
     the hypotheses that pass those gates, expl >= expl_rel * max(expl) --
     the relative threshold is taken after gating so a disqualified tumbled
     body at expl 1.0 cannot set it. Among survivors, the scorer's own best.

Declines (the scorer's pick is returned unchanged, with the reason recorded):
  - |U| below u_floor of the mask: the top-K explain the whole mask, nothing
    to re-rank on (also the symmetric-object case, where yaw is meaningless)
  - max(expl) below expl_floor: the mask has an unexplained region but no
    hypothesis reaches it -- part missing from the mesh, or the refined set
    collapsed; re-ranking cannot help and should not guess
  - no survivor passes the precision floor / depth gate
  - u_frac above u_frac_max: the top-K do not even agree on the body; the
    registration is bad for reasons a re-rank cannot fix -- callers should
    treat this as a hard stop, not proceed to grounding / VLM calls

Pure numpy; silhouettes come from the caller (nvdiffrast in the sidecar, PIL
rasterisation offline) so the selection is testable without a GPU.
Validated offline on run 20260903_203531 (252 refined hypotheses): U = 570 px
at half resolution vs a geometric handle blob of 569 px; picks rank 18
(155 deg from the scorer's pick, handle visible) -- see test_rerank.py and
outputs/runs/rank_hypotheses.py --rerank.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class RerankParams:
    top_k: int = 10                 # scorer-top hypotheses that define the body consensus
    consensus_frac: float = 0.9     # fraction of the top-K that must cover a pixel
    dilate_frac: float = 0.01       # dilation of the consensus, as a fraction of the mask bbox diagonal
    u_floor: float = 0.02           # |U| below this fraction of the mask -> decline (nothing unexplained)
    u_frac_max: float = 0.35        # |U| above this fraction of the mask -> body disagreement, decline + flag
    depth_band_diam: float = 0.5    # U pixels farther than this * mesh diameter from the body depth are dropped
    expl_floor: float = 0.10        # max(expl) below this -> decline (no hypothesis reaches U)
    expl_rel: float = 0.6           # survivors: expl >= expl_rel * max(expl)
    precision_floor: float = 0.9    # survivors: |sil AND mask| / |sil| >= this (rejects tumbled bodies covering U)
    depth_bad_diam: float = 0.15    # a pixel is "wrong surface" if |rendered - measured| > this * diameter
    depth_bad_max: float = 0.10     # survivors: fraction of wrong-surface pixels <= this (rejects inverted bodies)
    scorer_margin: float = 1.0      # survivors: score >= pick's score - this (override the scorer only where it is flat)


@dataclass
class RerankRecord:
    enabled: bool
    changed: bool
    reason: str
    from_rank: int
    to_rank: int
    rotation_deg: float
    expl_from: float
    expl_to: float
    n_survivors: int
    mask_px: int
    u_px_raw: int          # |mask minus consensus| before dilation
    u_px: int              # after dilation and the depth guard -- what expl is measured against
    u_frac: float          # u_px / mask_px
    u_depth_dropped: int   # U pixels removed by the depth guard
    dilate_px: int
    depth_band_m: float
    max_expl: float
    scorer_from: float
    scorer_to: float
    depth_bad_from: float = -1.0   # wrong-surface fraction of the scorer's pick (-1: no rendered depth given)
    depth_bad_to: float = -1.0     # ... of the chosen hypothesis
    n_gated: int = 0               # hypotheses passing precision / scorer-margin / depth gates (before expl)
    max_expl_gated: float = 0.0    # max(expl) over the gated set -- what the relative threshold is taken from


def _rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rerank_hypotheses(sils: np.ndarray, scores: np.ndarray, poses: np.ndarray, mask: np.ndarray,
                      depth: np.ndarray | None, diameter: float,
                      params: RerankParams | None = None,
                      depths: np.ndarray | None = None) -> tuple[int, RerankRecord]:
    """
    sils    (N, h, w) bool   silhouettes of the refined hypotheses, scorer-sorted (index 0 = pick)
    scores  (N,)             scorer scores, same order
    poses   (N, 4, 4)        same order (only used to report the rotation between pick and choice)
    mask    (h, w) bool      SAM mask at the same resolution
    depth   (h, w) float m   measured depth at the same resolution, 0 = invalid; None disables the
                             depth guard on U and the depth gate on survivors
    diameter                 mesh bounding diameter in metres (FoundationPose's est.diameter)
    depths  (N, h, w) float  rendered depth of each hypothesis, metres, <=0 or non-finite where
                             the mesh does not cover the pixel; None disables the depth gate
    returns (chosen_index, record). chosen_index == 0 whenever the re-rank declines.
    """
    p = params or RerankParams()
    N, h, w = sils.shape
    mask = mask.astype(bool)
    mask_px = int(mask.sum())
    rec = dict(enabled=True, changed=False, reason="", from_rank=0, to_rank=0, rotation_deg=0.0,
               expl_from=0.0, expl_to=0.0, n_survivors=0, mask_px=mask_px, u_px_raw=0, u_px=0,
               u_frac=0.0, u_depth_dropped=0, dilate_px=0, depth_band_m=float(p.depth_band_diam * diameter),
               max_expl=0.0, scorer_from=float(scores[0]), scorer_to=float(scores[0]))

    def decline(reason):
        rec["reason"] = reason
        return 0, RerankRecord(**rec)

    if N < p.top_k or mask_px == 0:
        return decline("too few hypotheses or empty mask")

    # 1. body consensus of the scorer's top-K, dilated by a fraction of the mask scale
    ys, xs = np.nonzero(mask)
    diag = float(np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    dil = max(1, int(round(p.dilate_frac * diag)))
    rec["dilate_px"] = dil
    consensus = sils[:p.top_k].mean(0) >= p.consensus_frac
    u_raw = mask & ~consensus
    rec["u_px_raw"] = int(u_raw.sum())
    body = binary_dilation(consensus, iterations=dil)
    U = mask & ~body

    # 2. depth guard: U must sit at the object's depth, not on the table behind it
    if depth is not None:
        dv = depth[mask & consensus]
        dv = dv[dv > 0]
        if dv.size >= 20:
            zb = float(np.median(dv))
            du = depth[U]
            far = (du > 0) & (np.abs(du - zb) > rec["depth_band_m"])
            rec["u_depth_dropped"] = int(far.sum())
            if far.any():
                idx = np.argwhere(U)[far]
                U[idx[:, 0], idx[:, 1]] = False
    u_px = int(U.sum())
    rec["u_px"] = u_px
    rec["u_frac"] = u_px / mask_px

    if rec["u_frac"] > p.u_frac_max:
        return decline("top-K disagree on the body (u_frac too large): registration unreliable, hard stop upstream")
    if rec["u_frac"] < p.u_floor:
        return decline("nothing unexplained by the body consensus (|U| below floor)")

    # 3. per-hypothesis coverage of U, and body precision
    expl = (sils & U).sum(axis=(1, 2)) / u_px
    sil_px = sils.sum(axis=(1, 2))
    precision = (sils & mask).sum(axis=(1, 2)) / np.maximum(sil_px, 1)
    rec["expl_from"] = float(expl[0])
    rec["max_expl"] = float(expl.max())
    if rec["max_expl"] < p.expl_floor:
        return decline("no hypothesis reaches the unexplained region (max expl below floor): part missing or set collapsed")

    # 4. gates first (precision, scorer margin, depth), THEN the relative expl
    #    threshold over the gated set, then the scorer picks among survivors
    ok = (precision >= p.precision_floor) & (scores >= scores[0] - p.scorer_margin)
    depth_bad = None
    if depths is not None and depth is not None:
        depth_bad = np.ones(N)
        thr = p.depth_bad_diam * diameter
        valid = mask & (depth > 0)
        for i in range(N):
            di = depths[i]
            sel = sils[i] & valid & np.isfinite(di) & (di > 0)
            if sel.sum() >= 50:
                depth_bad[i] = float((np.abs(di[sel] - depth[sel]) > thr).mean())
        ok &= depth_bad <= p.depth_bad_max
        rec["depth_bad_from"] = float(depth_bad[0])
    rec["n_gated"] = int(ok.sum())
    if not ok.any():
        return decline("no hypothesis passes the precision / scorer-margin / depth gates")
    rec["max_expl_gated"] = float(expl[ok].max())
    if rec["max_expl_gated"] < p.expl_floor:
        return decline("no gated hypothesis reaches the unexplained region (max expl below floor): "
                       "part missing, set collapsed, or only bad bodies cover it")
    ok &= expl >= p.expl_rel * rec["max_expl_gated"]
    surv = np.nonzero(ok)[0]
    rec["n_survivors"] = int(len(surv))
    chosen = int(surv[np.argmax(scores[surv])])
    if depth_bad is not None:
        rec["depth_bad_to"] = float(depth_bad[chosen])
    rec.update(to_rank=chosen, expl_to=float(expl[chosen]), scorer_to=float(scores[chosen]),
               rotation_deg=_rotation_deg(poses[0][:3, :3], poses[chosen][:3, :3]),
               changed=chosen != 0, reason="re-ranked" if chosen != 0 else "scorer pick already covers U")
    return chosen, RerankRecord(**rec)


def record_dict(rec: RerankRecord) -> dict:
    return asdict(rec)
