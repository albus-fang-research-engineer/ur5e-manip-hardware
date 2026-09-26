"""The frame packet: one still scene, captured once, with everything the
rest of the pipeline assumes about it written down.

run_scene gathers raw inputs (messages, joint samples, a TF lookup); this
module turns them into (median depth, packet record, robot mask) and
decides what stops the run. Pure numpy, so every gate is testable offline.

What a packet records, and why:

  frames      N synced RGB/depth pairs; the reference is the middle one.
              Stamp span and RGB->depth deltas, so a bad sync is visible.
  depth       the per-pixel median over valid samples (a pixel needs
              min_valid_frac of N valid samples, else it is a hole), and the
              contract every stage since the reprojection check has assumed:
              topic, encoding, scale, and that depth is ALIGNED to the colour
              frame whose K is used (same frame_id, same size). Misalignment
              is a stop: it decodes fine, pairs with the colour K, and makes
              every pose plausible and wrong. Distortion is recorded because
              nothing here undistorts: K is used as a pinhole.
  stillness   the median, q-at-a-stamp and the plane fit all assume nothing
              moved. Joint displacement over the frame span above
              still_tol_rad is a stop; depth that varies by more than
              unstable_m across N at a pixel is recorded as unstable_frac (a
              hand in the scene moves no joint).
  joints      q + names nearest the reference stamp, with the delta.
              Missing /joint_states is a DEGRADE: no robot mask, so arm
              removal falls back to the --bg prompt.
  t_base_cam  camera in the robot base frame. Live: from TF (at the stamp,
              else latest, delta recorded), else the static --t-base-cam
              (source "param", the same fallback curobo_bridge documents),
              else STOP -- without hand-eye nothing downstream means anything.
              From a saved run that predates packets: "absent_in_source", a
              degrade, because replaying an old frame creates no new data.
  robot_mask  cuRobo FK spheres vs the depth (needs q and T_base_cam).
              Excluded from the plane fit; later the arm test in scene_marks.
  plane       scene_marks.fit_table_plane on the median depth minus the robot.
              None, or too few inliers: STOP. Its normal through T_base_cam
              should be near base +z; the angle is recorded and warned above
              plane_up_warn_deg, never a stop on its own (a table can be
              slightly off level) -- but a large one means either the fit
              grabbed something that isn't the table, or the hand-eye is wrong.
"""
import math

import numpy as np

from manip_bridge import scene_marks as sm

PACKET_PARAMS = {
    "n_frames": (10, "provisional: ~0.33 s at 30 Hz; static scene assumed and checked"),
    "min_valid_frac": (0.5, "provisional: a pixel valid in fewer than half the frames is a hole"),
    "unstable_m": (0.02, "provisional: per-pixel depth range across N that counts as motion"),
    "still_tol_rad": (0.005, "provisional: ~0.3 deg total joint displacement over the frame span"),
    "plane_up_warn_deg": (5.0, "provisional: table normal vs base +z; warn only"),
    "sync_warn_s": (0.02, "provisional: RGB->depth pair delta worth a warning"),
    "joint_dt_warn_s": (0.05, "provisional: RGB->joint_states delta worth a warning"),
    "tf_dt_warn_s": (0.1, "provisional: RGB->TF delta worth a warning (static TF: n/a)"),
}


class GateStop(Exception):
    """A condition under which nothing downstream is meaningful. The run
    stops with the summary still written and summary['stop'] set."""

    def __init__(self, stage, reason):
        super().__init__(f"[{stage}] {reason}")
        self.stage, self.reason = stage, reason


def pvalues(params=None):
    p = {k: v for k, (v, _) in PACKET_PARAMS.items()}
    if params:
        unknown = set(params) - set(p)
        if unknown:
            raise KeyError(f"unknown frame_packet params: {sorted(unknown)}")
        p.update(params)
    return p


# ------------------------------------------------------------------ pieces

def median_depth(depths, min_valid_frac=0.5, unstable_m=0.02):
    """Per-pixel median over valid (> 0, finite) samples. A pixel with fewer
    than ceil(min_valid_frac * N) valid samples is 0. Returns (HxW float32,
    stats). Holes that are systematic (a mug cavity shadowing the projector)
    stay holes: this removes flicker and noise, not viewpoint problems."""
    D = np.stack([np.asarray(d, np.float32) for d in depths])
    N = D.shape[0]
    D = np.where(np.isfinite(D) & (D > 0), D, np.nan)
    count = np.sum(~np.isnan(D), axis=0)
    need = max(1, math.ceil(min_valid_frac * N))
    ok = count >= need
    med = np.zeros(D.shape[1:], np.float32)
    rng = np.zeros(D.shape[1:], np.float32)
    if ok.any():
        sub = D[:, ok]
        med[ok] = np.nanmedian(sub, axis=0)
        rng[ok] = np.nanmax(sub, axis=0) - np.nanmin(sub, axis=0)
    single_valid = float(np.mean(count > 0))
    return med, {"n": int(N), "min_valid_samples": int(need),
                 "valid_frac": float(ok.mean()), "any_sample_valid_frac": single_valid,
                 "unstable_frac": float(np.mean(rng[ok] > unstable_m)) if ok.any() else 0.0,
                 "unstable_m": unstable_m}


UR_ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def arm_sample(t, names, pos, vel=None, arm=UR_ARM_JOINTS):
    """One /joint_states message -> (t, arm, q, v) in canonical arm order, or
    None if it doesn't carry every arm joint. A gripper driver publishing its
    own /joint_states (finger joint only) on the same topic must not become
    "the nearest q"."""
    idx = {n: i for i, n in enumerate(names)}
    if not all(n in idx for n in arm) or len(pos) < len(names):
        return None
    q = [float(pos[idx[n]]) for n in arm]
    v = [float(vel[idx[n]]) for n in arm] if vel is not None and len(vel) == len(names) else None
    return (float(t), list(arm), q, v)


def nearest(samples, t):
    """samples: [(t, names, pos, vel)] -> (sample, t_sample - t) or (None, None)."""
    if not samples:
        return None, None
    s = min(samples, key=lambda x: abs(x[0] - t))
    return s, s[0] - t


def joint_motion(samples, t0, t1, pad=0.05):
    """Max displacement (rad) of any joint over [t0 - pad, t1 + pad], and the
    max |velocity| where reported. Samples are matched by joint name, so a
    reordered message can't fake motion."""
    win = [s for s in samples if t0 - pad <= s[0] <= t1 + pad]
    if len(win) < 2:
        return {"n_samples": len(win), "max_displacement_rad": None, "max_velocity_rad_s": None}
    names = list(win[0][1])
    P = np.array([[dict(zip(s[1], s[2]))[n] for n in names] for s in win])
    vel = [np.abs(np.asarray(s[3], float)).max() for s in win if s[3] is not None and len(s[3])]
    return {"n_samples": len(win),
            "max_displacement_rad": float(np.abs(P - P[0]).max()),
            "max_velocity_rad_s": float(max(vel)) if vel else None}


DEPTH_SCALE = {"16UC1": 0.001, "32FC1": 1.0}


def depth_contract(rgb, depth, info):
    """rgb/depth/info: {topic, frame_id, width, height[, encoding]} (+ info:
    distortion_model, D). Returns the record; raises GateStop if the depth is
    not aligned to the colour frame the K belongs to."""
    rec = {"rgb_topic": rgb.get("topic"), "depth_topic": depth.get("topic"),
           "info_topic": info.get("topic"), "rgb_frame_id": rgb["frame_id"],
           "depth_frame_id": depth["frame_id"], "info_frame_id": info["frame_id"],
           "size": [rgb["width"], rgb["height"]], "encoding": depth.get("encoding"),
           "scale_m_per_unit": DEPTH_SCALE.get(depth.get("encoding")),
           "distortion_model": info.get("distortion_model"),
           "D": [float(x) for x in (info.get("D") or [])]}
    rec["distortion_nonzero"] = bool(np.any(np.abs(rec["D"]) > 1e-9)) if rec["D"] else False
    problems = []
    if depth["frame_id"] != rgb["frame_id"]:
        problems.append(f"depth frame_id '{depth['frame_id']}' != rgb '{rgb['frame_id']}'")
    if info["frame_id"] and info["frame_id"] != rgb["frame_id"]:
        problems.append(f"camera_info frame_id '{info['frame_id']}' != rgb '{rgb['frame_id']}'")
    for name, m in (("depth", depth), ("camera_info", info)):
        if (m["width"], m["height"]) != (rgb["width"], rgb["height"]):
            problems.append(f"{name} {m['width']}x{m['height']} != rgb {rgb['width']}x{rgb['height']}")
    if rec["scale_m_per_unit"] is None:
        problems.append(f"unsupported depth encoding {depth.get('encoding')!r}")
    rec["aligned"] = not problems
    if problems:
        raise GateStop("depth_contract",
                       "; ".join(problems) + " -- depth must be aligned to the colour frame "
                       "whose K is used (realsense align_depth.enable:=true)")
    return rec


# ---------------------------------------------------------------- assembly

def build(*, depths, K, source, contract, stamps=None, joints=None, t_base_cam=None,
          robot_mask_fn=None, params=None, mark_params=None):
    """Assemble the packet. Raises GateStop.

    depths      list of HxW depth (m); the median is taken (N = 1 from a run)
    source      "live" or {"from_run": dir}
    contract    depth_contract() record (live) or the source run's
    stamps      live: {"ref": t, "rgb": [...], "depth": [...]}; None otherwise
    joints      live: {"samples": [arm_sample(...)], "topic": ...} (empty
                samples -> degrade); from a run: {"names", "q"} or None
    t_base_cam  {"T": 4x4, "source": "tf"|"tf_latest"|"param"|..., "dt_s",
                "base_frame", "camera_frame"} or None
    robot_mask_fn  callable(depth, K, T, q, names) -> HxW bool; may raise
    Returns (median_depth, packet, robot_mask or None)."""
    P = pvalues(params)
    live = source == "live"
    pk = {"version": 1, "source": source, "params": {k: {"value": P[k], "basis": b}
                                                     for k, (_, b) in PACKET_PARAMS.items()},
          "depth": dict(contract), "degrades": [], "warnings": []}
    K = np.asarray(K, np.float64)
    pk["K"] = K.tolist()
    if contract.get("aligned") is None:
        pk["degrades"].append("depth-to-colour alignment not recorded by the source run")
    if contract.get("distortion_nonzero"):
        pk["warnings"].append(f"camera_info has nonzero distortion {contract.get('D')}: "
                              "K is used as a pinhole and nothing undistorts")

    # frames + depth
    depth, st = median_depth(depths, P["min_valid_frac"], P["unstable_m"])
    pk["depth"].update(median=st)
    if st["unstable_frac"] > 0.01:
        pk["warnings"].append(f"{100 * st['unstable_frac']:.1f}% of pixels changed by "
                              f"> {P['unstable_m']} m across the {st['n']} frames: something moved")
    if stamps:
        rgb_t, dep_t = np.asarray(stamps["rgb"]), np.asarray(stamps["depth"])
        dt = np.abs(rgb_t - dep_t)
        pk["frames"] = {"n": len(rgb_t), "ref_stamp": stamps["ref"],
                        "span_s": float(rgb_t.max() - rgb_t.min()),
                        "rgb_depth_dt_max_s": float(dt.max())}
        if dt.max() > P["sync_warn_s"]:
            pk["warnings"].append(f"RGB->depth pair delta up to {dt.max() * 1e3:.0f} ms")
    else:
        pk["frames"] = {"n": st["n"], "ref_stamp": None, "span_s": None,
                        "rgb_depth_dt_max_s": None}

    # joints + stillness
    q = names = None
    if live:
        samples = (joints or {}).get("samples") or []
        s, dtq = nearest(samples, stamps["ref"]) if stamps else (None, None)
        if s is None:
            pk["joints"] = {"status": "missing", "topic": (joints or {}).get("topic")}
            pk["degrades"].append("no /joint_states: no robot mask; arm removal falls back "
                                  "to the --bg prompt")
        else:
            names, q = list(s[1]), [float(x) for x in s[2]]
            mot = joint_motion(samples, min(stamps["rgb"]), max(stamps["rgb"]))
            pk["joints"] = {"status": "ok", "topic": joints.get("topic"), "names": names, "q": q,
                            "dt_to_ref_s": dtq, **mot, "still_tol_rad": P["still_tol_rad"]}
            if abs(dtq) > P["joint_dt_warn_s"]:
                pk["warnings"].append(f"nearest joint_states is {dtq * 1e3:+.0f} ms from the frame")
            if (mot["max_displacement_rad"] is not None
                    and mot["max_displacement_rad"] > P["still_tol_rad"]):
                raise GateStop("stillness",
                               f"a joint moved {mot['max_displacement_rad']:.4f} rad during the "
                               f"{pk['frames']['span_s']:.2f} s capture (> {P['still_tol_rad']}): "
                               "the median depth and the robot mask would both be wrong")
    elif joints and joints.get("q") is not None:
        names, q = list(joints["names"]), [float(x) for x in joints["q"]]
        pk["joints"] = {"status": "from_source", "names": names, "q": q}
    else:
        pk["joints"] = {"status": "absent_in_source"}
        pk["degrades"].append("source run has no joint state: no robot mask")

    # hand-eye
    T = None
    if t_base_cam is not None:
        T = np.asarray(t_base_cam["T"], np.float64)
        pk["t_base_cam"] = {**{k: v for k, v in t_base_cam.items() if k != "T"}, "T": T.tolist()}
        dtt = t_base_cam.get("dt_s")
        if dtt is not None and abs(dtt) > P["tf_dt_warn_s"]:
            pk["warnings"].append(f"T_base_cam is {dtt * 1e3:+.0f} ms from the frame")
    elif live:
        raise GateStop("t_base_cam", "no base<-camera transform on TF and no --t-base-cam: "
                                     "without hand-eye nothing downstream is meaningful")
    else:
        pk["t_base_cam"] = {"source": "absent_in_source"}
        pk["degrades"].append("source run has no T_base_cam: no robot mask, no plane-vs-base check")

    # robot mask
    rmask = None
    if q is None or T is None:
        pk["robot_mask"] = {"status": "unavailable: needs q and T_base_cam"}
    elif robot_mask_fn is None:
        pk["robot_mask"] = {"status": "unavailable: no cuRobo client"}
        pk["degrades"].append("no robot mask (no cuRobo client): arm removal via --bg prompt")
    else:
        try:
            rmask = np.asarray(robot_mask_fn(depth, K, T, q, names), bool)
            pk["robot_mask"] = {"status": "ok", "n_masked": int(rmask.sum())}
        except Exception as e:                              # sidecar down, timeout, bad reply
            pk["robot_mask"] = {"status": f"error: {e}"}
            pk["degrades"].append(f"robot mask failed ({e}): arm removal via --bg prompt")

    # plane
    fit_depth = depth if rmask is None else np.where(rmask, 0.0, depth)
    up = None if T is None else T[:3, :3].T @ np.array([0.0, 0.0, 1.0])   # base +z in camera
    plane = sm.fit_table_plane(fit_depth, K, mark_params, up_hint=up)
    mp = sm.values(mark_params)
    if plane is None:
        raise GateStop("plane", "too few depth points to fit a table plane")
    plane["fit_excludes_robot"] = rmask is not None
    pk["plane"] = plane
    if plane["inlier_frac"] < mp["plane_min_inlier_frac"]:
        raise GateStop("plane", f"dominant plane has {100 * plane['inlier_frac']:.0f}% inliers "
                                f"(< {100 * mp['plane_min_inlier_frac']:.0f}%): no trustworthy table")
    ang = plane.get("angle_to_up_deg")
    if ang is not None and ang > P["plane_up_warn_deg"]:
        pk["warnings"].append(f"table normal is {ang:.1f} deg from base +z: the fit grabbed "
                              "something other than the table, or the hand-eye is off")
    return depth, pk, rmask


# ---------------------------------------------------------------- from a run

def load_run(run_dir):
    """A saved run as build() inputs: rgb (HxWx3 uint8), depth (m), K,
    frame_id, stamp, contract, joints, t_base_cam. A run written with a
    packet replays its recorded q / T_base_cam / contract; an older run has
    none of them, and says so (the fields become degrades in build())."""
    import json
    import os

    from PIL import Image

    with open(os.path.join(run_dir, "summary.json")) as f:
        summ = json.load(f)
    fr = summ["frame"]
    rgb = np.asarray(Image.open(os.path.join(run_dir, "rgb.png")).convert("RGB"), np.uint8)
    depth = np.asarray(Image.open(os.path.join(run_dir, "depth_mm.png")), np.float64) / 1000.0
    if depth.shape != rgb.shape[:2]:
        raise GateStop("from_run", f"depth {depth.shape} != rgb {rgb.shape[:2]} in {run_dir}")
    src = summ.get("packet")
    if src:
        contract = dict(src["depth"])
        contract.pop("median", None)
        contract["note"] = f"recorded by the source run ({src.get('source')})"
        j = src.get("joints") or {}
        joints = {"names": j["names"], "q": j["q"]} if j.get("q") is not None else None
        t = src.get("t_base_cam") or {}
        t_base_cam = ({**t, "source": f"source run ({t.get('source')})"}
                      if t.get("T") is not None else None)
    else:
        H, W = depth.shape
        contract = {"rgb_frame_id": fr.get("frame_id"), "depth_frame_id": None,
                    "size": [W, H], "encoding": "16UC1 (depth_mm.png)", "scale_m_per_unit": 0.001,
                    "aligned": None,
                    "note": "source run predates frame packets: alignment was not recorded"}
        joints = t_base_cam = None
    return {"rgb": rgb, "depth": depth, "K": fr["K"], "frame_id": fr.get("frame_id", ""),
            "stamp": fr.get("stamp"), "contract": contract, "joints": joints,
            "t_base_cam": t_base_cam, "has_packet": bool(src)}
