"""Which of FoundationPose's surviving hypotheses has the right yaw -- the
decision the offline driver (outputs/runs/fp_from_mesh.py --oriany) and
the live bridge (pose_bridge_node, /pose/estimate with decide=true) both
make, implemented once here so the driver is a regression test of the
bridge rather than a second implementation that happens to agree today.

Transport is injected: decide() takes `orient(image) -> Orient Anything
reply` and `select(rank) -> 4x4`. The driver passes its own sockets, the
bridge passes its SidecarClients (sidecar_fns). Nothing here does I/O.

Inputs are FoundationPose's register reply with rerank + survivor_crops +
return_all: the scorer-sorted hypotheses, the mask-conditioned re-rank
record (pose_server/rerank.py), and a textured render of each re-rank
survivor in the mask's crop framing.

The decision (patch0010; the earlier alpha-gated rule never fired on real
data -- the mug's photo read alpha 2 with one obvious handle, and clean renders
of the same mesh read 0 / 2 / 4, while the FRONT head was consistent: 13 deg
for the correct pose, 72-108 deg for wrong ones, orient_rel agreeing):
  1. Orient Anything reads the REAL masked crop -- always, including when
     the re-rank declined (on a cup that is the only semantic azimuth).
     Real-crop alpha 0 (no confident front) abstains; otherwise alpha is
     recorded and never used.
  2. It reads each survivor's render in the same framing; each survivor's
     front is compared with the real crop's as an UNFOLDED yaw about the
     real crop's up. Render alpha is recorded, never gating.
  3. The best-agreeing survivor must read within FIRE_YAW_DEG, else abstain
     (oa_no_agreement).
  4. Ambiguity from the front head's own behaviour, not its alpha: another
     survivor within AMBIG_YAW_DEG of the best's agreement but more than
     FAMILY_DEG away as a pose -> abstain (oa_ambiguous). If the front head
     cannot tell front from back on some object, a flipped survivor reads
     near 0 too and this fires; if it can, the flip reads ~180 and loses.
  5. If the best survivor is in the re-rank pick's pose family (within
     FAMILY_DEG), the re-rank's pick stands: oriany_agrees. A few degrees of
     front noise must not move the pose within a family.
  6. Otherwise Orient Anything proposes an OVERRIDE of the re-rank. That
     needs a second estimation path: orient_rel(real, render) -- a different
     head of the same model, comparing the two images directly instead of
     reading absolute peaks -- must also put the new pick closer to the real
     crop than the re-rank's pick (else oa_rel_disagrees). This guards the
     one failure the ambiguity check cannot see: on a real crop with a
     two-peaked front distribution, the front head picking the wrong peak,
     which makes the flipped survivor read ~0 and the correct one ~180.

Framing: the survivor renders were cropped with the sidecar's crop_box, so
the real crop uses that box whenever the reply carries one. When it
doesn't (re-rank declined, no survivors), crop_box() recomputes it from the
mask with the sidecar's own formula. Both are recorded when both exist.

Outcome classes (record["reason"]):
  hard stop:   rerank_u_frac          top-K disagree on the body (u_frac above
                                      the sidecar's u_frac_max): the registration
                                      itself is unreliable
  degrade (the re-rank / scorer pick stands, recorded):
               rerank_missing, rerank_u_floor (nothing unexplained -- the cup),
               rerank_declined (other gates), oriany_unavailable, crops_missing,
               oa_alpha0, oa_no_candidates, oa_no_hypotheses, oa_no_agreement,
               oa_ambiguous, oa_rel_unavailable, oa_rel_disagrees, select_failed
  fired:       oriany_agrees (confirms the re-rank; agreement_deg recorded)
               oriany_changed (overrides it; corroboration recorded)

The front for grounding (record["front"]): front_body = R_selᵀ f_real
whenever the real crop reads alpha != 0, with alpha_reported and a caveat
when it is > 1 (self-reported folds were unreliable on the mug). `independent`
is False exactly when this reading CHANGED the yaw (oriany_changed); a
confirmation leaves it an independent measurement.
"""
import numpy as np

CROP_MARGIN = 0.2          # pose_server Session.survivor_crops default
FIRE_YAW_DEG = 30.0        # best survivor must agree within this
AMBIG_YAW_DEG = 20.0       # a far-apart rival this close in agreement -> ambiguous
FAMILY_DEG = 30.0          # rotation within which two hypotheses are one pose family
PARAMS = {
    "fire_yaw_deg": (FIRE_YAW_DEG, "n=1 (mug, 20260926_234500): correct pose 13 deg, wrong 72-108 deg"),
    "ambig_yaw_deg": (AMBIG_YAW_DEG, "provisional: no flipped survivor observed yet"),
    "family_deg": (FAMILY_DEG, "n=1: one family's survivors read -13/-9 deg; wrong poses were >= 59 deg away"),
}

# ------------------------------------------------------------------ geometry

def crop_box(mask, margin=CROP_MARGIN):
    """[y0, y1, x0, x1]: pose_server's survivor_crops framing, verbatim --
    the mask's bbox centre, square half-side max(ptp) * (0.5 + margin),
    clipped to the image."""
    m = np.asarray(mask).astype(bool)
    H, W = m.shape[:2]
    ys, xs = np.nonzero(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = int(max(np.ptp(ys), np.ptp(xs)) * (0.5 + margin))
    return [max(0, int(cy - half)), min(H, int(cy + half)),
            max(0, int(cx - half)), min(W, int(cx + half))]


def real_crop(rgb, mask, box):
    y0, y1, x0, x1 = box
    m = np.asarray(mask).astype(bool)
    return np.where(m[..., None], rgb, 255).astype(np.uint8)[y0:y1, x0:x1]


def geo(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def yaw_about(up, fa, fb):
    """Signed angle between two fronts projected onto the plane normal to `up`."""
    up = up / np.linalg.norm(up)
    pa = fa - up * (fa @ up)
    pb = fb - up * (fb @ up)
    pa /= max(np.linalg.norm(pa), 1e-9)
    pb /= max(np.linalg.norm(pb), 1e-9)
    return float(np.degrees(np.arctan2(np.cross(pa, pb) @ up, pa @ pb)))


def _oa(o):
    return {"alpha": int(o["alpha"]), "front_cam": [float(x) for x in np.asarray(o["front_cam"]).ravel()],
            "up_cam": [float(x) for x in np.asarray(o["up_cam"]).ravel()],
            "R_cam": np.asarray(o["R_cam"], float).reshape(3, 3).tolist()}


# ------------------------------------------------------------------ decision

def choose(rows, alpha_real, hyp_poses, sidecar_rank, rel):
    """-> (chosen_rank, used, note, reason, corroboration or None).
    rel(rank) -> orient_rel azimuth (deg, wrapped) of that survivor's render
    against the real crop; only called when an override is proposed."""
    if alpha_real == 0:
        return sidecar_rank, "scorer", "real crop alpha 0: Orient Anything has no confident front", "oa_alpha0", None
    if not rows:
        return sidecar_rank, "scorer", "no survivor readings", "oa_no_candidates", None
    if hyp_poses is None:
        return (sidecar_rank, "scorer", "no hypotheses in the reply: pose families unknown",
                "oa_no_hypotheses", None)

    def apart(ra, rb):
        return geo(hyp_poses[ra][:3, :3], hyp_poses[rb][:3, :3])

    cand = sorted(rows, key=lambda r: r["yaw_abs"])
    best = cand[0]
    if best["yaw_abs"] > FIRE_YAW_DEG:
        return sidecar_rank, "scorer", (
            f"no survivor reads within {FIRE_YAW_DEG:.0f} deg of the real crop "
            f"(best: rank {best['rank']} at {best['yaw_abs']:.0f})"), "oa_no_agreement", None
    for r in cand[1:]:
        if r["yaw_abs"] - best["yaw_abs"] >= AMBIG_YAW_DEG:
            break
        d = apart(best["rank"], r["rank"])
        if d > FAMILY_DEG:
            return sidecar_rank, "scorer", (
                f"oa_ambiguous: ranks {best['rank']} and {r['rank']} read {best['yaw_abs']:.0f} and "
                f"{r['yaw_abs']:.0f} deg but are {d:.0f} deg apart as poses"), "oa_ambiguous", None
    if apart(best["rank"], sidecar_rank) <= FAMILY_DEG:
        return sidecar_rank, "oriany", "", "oriany_agrees", None
    if rel is None:
        return sidecar_rank, "scorer", "override proposed but no orient_rel transport", "oa_rel_unavailable", None
    try:
        rb, rp = rel(best["rank"]), rel(sidecar_rank)
    except Exception as e:
        return sidecar_rank, "scorer", f"orient_rel failed: {type(e).__name__}: {e}", "oa_rel_unavailable", None
    corr = {"proposed": {"rank": int(best["rank"]), "rel_az": rb},
            "rerank_pick": {"rank": int(sidecar_rank), "rel_az": rp}}
    if abs(rb) < abs(rp):
        return int(best["rank"]), "oriany", "", "oriany_changed", corr
    return sidecar_rank, "scorer", (
        f"orient_rel disagrees with the override: proposed rank {best['rank']} rel_az {rb:+.0f}, "
        f"re-rank pick {sidecar_rank} rel_az {rp:+.0f}"), "oa_rel_disagrees", corr


def _rerank_class(rr):
    """(hard_stop, reason) for the re-rank record alone, or None if it produced survivors."""
    if rr is None:
        return False, "rerank_missing"
    if rr.get("survivors"):
        return None
    why = rr.get("reason", "")
    if "u_frac too large" in why:
        return True, "rerank_u_frac"
    if "nothing unexplained" in why:
        return False, "rerank_u_floor"
    return False, "rerank_declined"


def decide(rep, rgb, mask, orient, select, orient_rel=None, want_sheet=False):
    """-> (T_selected or None, record, sheet or None).

    rep         pose sidecar register reply (rerank, survivor_crops, return_all)
    orient      callable(image uint8 HxWx3) -> Orient Anything reply; may raise
    select      callable(rank) -> 4x4 cam_T_obj; may raise
    orient_rel  callable(ref image, tgt image) -> orient_rel reply; may raise.
                Needed only to corroborate an override (None -> no overrides)
    T_selected is None when the sidecar's pose stands (nothing selected)."""
    rr = rep.get("rerank")
    sidecar_rank = int(rr["to_rank"]) if rr else 0
    own_box = crop_box(mask)
    sc_box = [int(v) for v in rep["crop_box"]] if rep.get("crop_box") is not None else None
    box = sc_box or own_box
    rec = {"version": 1, "rerank": rr, "sidecar_rank": sidecar_rank,
           "crop_box": box, "crop_box_source": "sidecar" if sc_box else "bridge",
           "crop_box_match": (sc_box == own_box) if sc_box else None,
           "oa_real": None, "rows": [], "decider": "scorer", "note": "", "chosen_rank": sidecar_rank,
           "decider_fired": False, "hard_stop": False, "reason": "", "agreement_deg": None,
           "corroboration": None,
           "params": {k: {"value": v, "basis": b} for k, (v, b) in PARAMS.items()}}

    real = real_crop(rgb, mask, box)
    o_real, oa_err = None, None
    try:
        o_real = orient(real)
    except Exception as e:                                   # sidecar down, timeout, error reply
        oa_err = f"{type(e).__name__}: {e}"
    if o_real is not None:
        rec["oa_real"] = _oa(o_real)

    survivors = list(rr.get("survivors") or []) if rr else []
    crops = rep.get("crops")
    oa_survivors = []
    if o_real is not None and survivors and crops is not None:
        R_real, up_real = np.asarray(o_real["R_cam"], float), np.asarray(o_real["up_cam"], float)
        f_real = np.asarray(o_real["front_cam"], float)
        for k, rk in enumerate(survivors):
            try:
                o = orient(np.asarray(crops[k], np.uint8))
            except Exception as e:
                oa_err = f"{type(e).__name__}: {e}"
                rec["rows"], oa_survivors = [], []
                break
            oa_survivors.append(o)
            up_o, f_o = np.asarray(o["up_cam"], float), np.asarray(o["front_cam"], float)
            yaw = yaw_about(up_real, f_real, f_o)             # the real crop's up for BOTH sides
            rec["rows"].append({
                "rank": int(rk), "geo": geo(R_real, np.asarray(o["R_cam"], float)),
                "up": float(np.degrees(np.arccos(np.clip(up_real @ up_o, -1, 1)))),
                "yaw": yaw, "yaw_abs": abs(yaw), "alpha": int(o["alpha"])})

    cls = _rerank_class(rr)
    if cls is not None:                                     # re-rank declined: nothing to choose among
        rec["hard_stop"], rec["reason"] = cls
        rec["note"] = (rr or {}).get("reason", "no re-rank record in the reply")
        if oa_err is not None:
            rec["note"] += f" | real-crop Orient Anything failed: {oa_err}"
    elif oa_err is not None:
        rec["reason"], rec["note"] = "oriany_unavailable", oa_err
    elif crops is None:
        rec["reason"], rec["note"] = "crops_missing", "re-rank survivors but no survivor_crops in the reply"
    else:
        hyp = np.asarray(rep["hypotheses"], np.float64) if rep.get("hypotheses") is not None else None
        by_rank = {int(rk): np.asarray(crops[k], np.uint8) for k, rk in enumerate(survivors)}

        def rel(rank):
            r = orient_rel(real, by_rank[int(rank)])
            return float(((float(r["rel_azimuth"]) + 180.0) % 360.0) - 180.0)

        chosen, used, note, reason, corr = choose(rec["rows"], int(o_real["alpha"]), hyp, sidecar_rank,
                                                  rel if orient_rel is not None else None)
        rec.update(chosen_rank=chosen, decider=used, note=note, reason=reason, corroboration=corr,
                   decider_fired=used == "oriany")
        row = next((r for r in rec["rows"] if r["rank"] == chosen), None)
        if used == "oriany" and row is not None:
            rec["agreement_deg"] = row["yaw_abs"]

    T_sel = None
    if not rec["hard_stop"] and rec["chosen_rank"] != sidecar_rank:
        try:
            T_sel = np.asarray(select(rec["chosen_rank"]), np.float64).reshape(4, 4)
        except Exception as e:
            rec["note"] += f" | select({rec['chosen_rank']}) failed: {type(e).__name__}: {e}"
            rec.update(reason="select_failed", chosen_rank=sidecar_rank, decider="scorer",
                       decider_fired=False)

    T_final = T_sel if T_sel is not None else np.asarray(rep["pose"], np.float64).reshape(4, 4)
    rec["front"] = front_record(rec["oa_real"], T_final, rec["reason"] == "oriany_changed")
    if rec["hard_stop"]:                                    # no reliable pose to express it in
        rec["front"].update(front_body=None, up_body=None, independent=None,
                            note="hard stop: the registration is unreliable, so there is no body "
                                 "frame to express the front in (the reading itself is oa_real)")
    sheet = contact_sheet(real, rec, [np.asarray(c, np.uint8) for c in crops[:len(oa_survivors)]]
                          if crops is not None else [], oa_survivors) if want_sheet and o_real is not None else None
    return T_sel, rec, sheet


def front_record(oa_real, T, changed_yaw):
    """The semantic front handed to grounding, in the body frame of the
    selected pose, whenever the real crop reads alpha != 0. A reported fold
    > 1 is attached as a caveat, not used to withhold the front: on the mug
    the photo read alpha 2 while its front was the correct one."""
    if oa_real is None:
        return {"source": None, "alpha_reported": None, "front_body": None, "up_body": None,
                "independent": None, "note": "no Orient Anything reading of the real crop"}
    R = np.asarray(T, float)[:3, :3]
    up_body = (R.T @ np.asarray(oa_real["up_cam"], float)).tolist()
    a = oa_real["alpha"]
    if a == 0:
        return {"source": "oriany_real_crop", "alpha_reported": 0, "front_body": None, "up_body": up_body,
                "independent": None, "note": "alpha 0: no confident front"}
    note = "the same reading changed the yaw" if changed_yaw else "independent of the yaw choice"
    if a > 1:
        note += (f"; Orient Anything reports alpha {a} (front up to {360 // a} deg) -- self-reported "
                 "folds were unreliable on the mug, whose correct front read alpha 2")
    return {"source": "oriany_real_crop", "alpha_reported": a,
            "front_body": (R.T @ np.asarray(oa_real["front_cam"], float)).tolist(), "up_body": up_body,
            "independent": not changed_yaw, "note": note}


# -------------------------------------------------------------------- output

def contact_sheet(real, rec, crops, oa_survivors):
    """Real crop + each survivor render, front (green) and up (blue) drawn."""
    from PIL import Image, ImageDraw

    def annotate(img, o, title, line):
        im = Image.fromarray(img).convert("RGB")
        d = ImageDraw.Draw(im)
        h, w = img.shape[:2]
        c = np.array([w / 2, h / 2])
        L = 0.25 * min(h, w)
        for vec, col in ((np.asarray(o["front_cam"], float), (0, 200, 0)),
                         (np.asarray(o["up_cam"], float), (40, 90, 255))):
            tip = c + L * vec[:2]
            d.line([tuple(c), tuple(tip)], fill=col, width=3)
            d.ellipse([tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4], fill=col)
        d.text((4, 2), title, fill=(0, 0, 0))
        d.text((4, h - 12), line, fill=(0, 0, 0))
        return im

    panels = [annotate(real, rec["oa_real"], "real", f"alpha {rec['oa_real']['alpha']}")]
    for crop, o, r in zip(crops, oa_survivors, rec["rows"]):
        mark = " <-" if r["rank"] == rec["chosen_rank"] else ""
        panels.append(annotate(crop, o, f"rank {r['rank']}{mark}",
                               f"yaw {r['yaw']:+.0f} rot {r['geo']:.0f} up {r['up']:.0f} a{r['alpha']}"))
    Wp, Hp = max(p.size[0] for p in panels), max(p.size[1] for p in panels)
    sheet = Image.new("RGB", (len(panels) * (Wp + 6), Hp), "white")
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (Wp + 6), 0))
    return np.asarray(sheet)


def summary_line(rec):
    fr = rec["front"]
    agree = f" at {rec['agreement_deg']:.0f} deg" if rec.get("agreement_deg") is not None else ""
    return (f"decider {rec['decider']} ({rec['reason']}{agree}): rank {rec['sidecar_rank']} -> {rec['chosen_rank']}"
            f"{'  HARD STOP' if rec['hard_stop'] else ''}; front "
            + ("none" if fr["front_body"] is None else ("independent" if fr["independent"] else "not independent"))
            + (f" -- {rec['note']}" if rec["note"] else ""))


def sidecar_fns(oriany_client, pose_client, obj):
    """(orient, select, orient_rel) over SidecarClients: the bridge's transport,
    and the driver's (fp_from_mesh --oriany), so every path fails the same way."""
    def orient(img):
        return oriany_client.call({"cmd": "orient", "image": np.ascontiguousarray(img, np.uint8),
                                   "remove_bkg": False})

    def select(rank):
        return pose_client.call({"cmd": "select", "obj": obj, "rank": int(rank)})["pose"]

    def orient_rel(ref, tgt):
        return oriany_client.call({"cmd": "orient_rel", "image_ref": np.ascontiguousarray(ref, np.uint8),
                                   "image_tgt": np.ascontiguousarray(tgt, np.uint8), "remove_bkg": False})
    return orient, select, orient_rel
