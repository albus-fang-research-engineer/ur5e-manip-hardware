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

The decision (unchanged from the validated driver):
  1. Orient Anything reads the REAL masked crop -- always, including when
     the re-rank declined. That reading is the object's semantic front,
     and on the decline path (a cup: nothing unexplained, U below floor) it
     is the only semantic azimuth there is. Skipping it there is the bug
     this module fixes relative to the driver it replaces.
  2. It reads each survivor's render in the same framing. A survivor is
     confirmable when its alpha equals the real crop's and is not 0.
  3. Among confirmable survivors, the one whose front is closest in yaw
     about the real crop's up (folded by the symmetry order) wins -- unless
     the two best read within 10 deg of each other but are > 30 deg apart
     as poses (oa_ambiguous), in which case the re-rank pick stands.

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
               oa_alpha0, oa_none_confirmable, oa_ambiguous, select_failed
  fired:       oriany_agrees / oriany_changed

The front for grounding (record["front"]): front_body = R_selᵀ f_real when
the real crop reads alpha 1, else None. `independent` is False exactly
when the decider fired: then that same reading chose the yaw, and the front
is not a second measurement of it.
"""
import numpy as np

CROP_MARGIN = 0.2          # pose_server Session.survivor_crops default
AMBIG_YAW_DEG = 10.0
AMBIG_APART_DEG = 30.0


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


def fold(deg, alpha):
    """|yaw| modulo the symmetry order: on an alpha-fold object Orient
    Anything's front is defined only up to 360/alpha."""
    period = 360.0 / max(int(alpha), 1)
    d = abs(deg) % period
    return float(min(d, period - d))


def _oa(o):
    return {"alpha": int(o["alpha"]), "front_cam": [float(x) for x in np.asarray(o["front_cam"]).ravel()],
            "up_cam": [float(x) for x in np.asarray(o["up_cam"]).ravel()],
            "R_cam": np.asarray(o["R_cam"], float).reshape(3, 3).tolist()}


# ------------------------------------------------------------------ decision

def choose(rows, alpha_real, hyp_poses, sidecar_rank):
    """-> (chosen_rank, used, note, reason). The validated driver rule."""
    if alpha_real == 0:
        return sidecar_rank, "scorer", "real crop alpha 0: Orient Anything has no confident front", "oa_alpha0"
    cand = sorted((r for r in rows if r["confirmable"]), key=lambda r: r["yaw_folded"])
    if not cand:
        return (sidecar_rank, "scorer", "no survivor render is confirmable (alpha 0 or alpha != real)",
                "oa_none_confirmable")
    best = cand[0]
    if len(cand) > 1 and hyp_poses is not None:
        second = cand[1]
        apart = geo(hyp_poses[best["rank"]][:3, :3], hyp_poses[second["rank"]][:3, :3])
        if abs(best["yaw_folded"] - second["yaw_folded"]) < AMBIG_YAW_DEG and apart > AMBIG_APART_DEG:
            return sidecar_rank, "scorer", (
                f"oa_ambiguous: ranks {best['rank']} and {second['rank']} read {best['yaw_folded']:.0f} vs "
                f"{second['yaw_folded']:.0f} deg but are {apart:.0f} deg apart (alpha {alpha_real} fold)"), "oa_ambiguous"
    rank = int(best["rank"])
    return rank, "oriany", "", "oriany_changed" if rank != sidecar_rank else "oriany_agrees"


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


def decide(rep, rgb, mask, orient, select, want_sheet=False):
    """-> (T_selected or None, record, sheet or None).

    rep     pose sidecar register reply (rerank, survivor_crops, return_all)
    orient  callable(image uint8 HxWx3) -> Orient Anything reply; may raise
    select  callable(rank) -> 4x4 cam_T_obj; may raise
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
           "decider_fired": False, "hard_stop": False, "reason": ""}

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
        f_real, alpha_real = np.asarray(o_real["front_cam"], float), int(o_real["alpha"])
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
                "yaw": yaw, "yaw_folded": fold(yaw, alpha_real), "alpha": int(o["alpha"]),
                "confirmable": bool(int(o["alpha"]) != 0 and int(o["alpha"]) == alpha_real)})

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
        chosen, used, note, reason = choose(rec["rows"], int(o_real["alpha"]), hyp, sidecar_rank)
        rec.update(chosen_rank=chosen, decider=used, note=note, reason=reason,
                   decider_fired=used == "oriany")

    T_sel = None
    if not rec["hard_stop"] and rec["chosen_rank"] != sidecar_rank:
        try:
            T_sel = np.asarray(select(rec["chosen_rank"]), np.float64).reshape(4, 4)
        except Exception as e:
            rec["note"] += f" | select({rec['chosen_rank']}) failed: {type(e).__name__}: {e}"
            rec.update(reason="select_failed", chosen_rank=sidecar_rank, decider="scorer",
                       decider_fired=False)

    T_final = T_sel if T_sel is not None else np.asarray(rep["pose"], np.float64).reshape(4, 4)
    rec["front"] = front_record(rec["oa_real"], T_final, rec["decider_fired"])
    if rec["hard_stop"]:                                    # no reliable pose to express it in
        rec["front"].update(front_body=None, up_body=None, independent=None,
                            note="hard stop: the registration is unreliable, so there is no body "
                                 "frame to express the front in (the reading itself is oa_real)")
    sheet = contact_sheet(real, rec, [np.asarray(c, np.uint8) for c in crops[:len(oa_survivors)]]
                          if crops is not None else [], oa_survivors) if want_sheet and o_real is not None else None
    return T_sel, rec, sheet


def front_record(oa_real, T, decider_fired):
    """The semantic front handed to grounding, in the body frame of the
    selected pose. None unless the real crop reads alpha 1."""
    if oa_real is None:
        return {"source": None, "alpha": None, "front_body": None, "up_body": None,
                "independent": None, "note": "no Orient Anything reading of the real crop"}
    R = np.asarray(T, float)[:3, :3]
    up_body = (R.T @ np.asarray(oa_real["up_cam"], float)).tolist()
    a = oa_real["alpha"]
    if a != 1:
        return {"source": "oriany_real_crop", "alpha": a, "front_body": None, "up_body": up_body,
                "independent": None,
                "note": "alpha 0: no confident front" if a == 0 else
                        f"alpha {a}: front defined only up to {360 // a} deg"}
    return {"source": "oriany_real_crop", "alpha": 1,
            "front_body": (R.T @ np.asarray(oa_real["front_cam"], float)).tolist(), "up_body": up_body,
            "independent": not decider_fired,
            "note": "the same reading chose the yaw" if decider_fired else
                    "independent of the yaw choice"}


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
    return (f"decider {rec['decider']} ({rec['reason']}): rank {rec['sidecar_rank']} -> {rec['chosen_rank']}"
            f"{'  HARD STOP' if rec['hard_stop'] else ''}; front "
            + ("none" if fr["front_body"] is None else ("independent" if fr["independent"] else "not independent"))
            + (f" -- {rec['note']}" if rec["note"] else ""))


def sidecar_fns(oriany_client, pose_client, obj):
    """(orient, select) over SidecarClients: the bridge's transport."""
    def orient(img):
        return oriany_client.call({"cmd": "orient", "image": np.ascontiguousarray(img, np.uint8),
                                   "remove_bkg": False})

    def select(rank):
        return pose_client.call({"cmd": "select", "obj": obj, "rank": int(rank)})["pose"]
    return orient, select
