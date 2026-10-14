#!/usr/bin/env python3
"""Write a hand-eye calibration into a run that was captured without one,
where run_scene would have recorded it: summary.json packet.t_base_cam.

    <calib>             an easy_handeye2 .calib file (YAML of
                        easy_handeye2_msgs/HandeyeCalibration: parameters +
                        transform{translation, rotation}); handeye_publisher
                        broadcasts it as robot_base_frame -> tracking_base_frame,
                        i.e. the camera's pose in the base: T_base_cam
    <run>/summary.json  packet.t_base_cam <- {source: "param", T, calib: {...}}

Once written, every later step reads it from the run (grasp_tsr, and the
t_base_cam the step-3/4 nodes are given) -- one place, no flags.

THE CALIBRATION MUST DESCRIBE THE CAMERA AT THE RUN'S TIME. Nothing here can
prove it does: grasp_tsr's pixel check cancels T_base_cam. The one
independent check available is printed and recorded: the run's table normal
(packet.plane.n, fitted in the camera frame at capture) against base +z
through this T, the same angle and warning threshold frame_packet uses on a
live capture. A calibration from a different camera pose usually tilts the
table; a large angle means the wrong calibration (or the fit grabbed
something other than the table). It warns; it does not refuse.

Refusals (exit 2), each naming its fix:
  type     calibration_type must be eye_on_base, robot_base_frame base_link
  frame    tracking_base_frame must equal the run's camera frame
           (summary frame.frame_id). easy_handeye2 is often run against
           camera_link while depth and K are in the optical frame; the two
           differ by a fixed rotation, and writing one for the other would
           point every grasp the wrong way, silently.
  rigid    the quaternion must be unit (|q| within 1e-3)
  existing never replaces a T the run already carries; the same T again is
           a no-op (exit 0)

summary.json is backed up once to summary.json.pre_hand_eye. No gate hashes
summary.json, so earlier artifacts (grounding, plan, selections) stay valid.

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/set_hand_eye.py --run /data/runs/<stamp> --calib <name>.calib
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

try:
    from manip_bridge import frame_packet as fp
except ImportError as e:
    sys.exit(f"set_hand_eye: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

BASE_FRAME = "base_link"
SAME_T_TOL = 1e-9


class HandEyeRefusal(Exception):
    """An input the script will not write; the message names the fix."""


def read_calib(path: Path):
    """-> (T_base_cam 4x4, parameters dict)."""
    try:
        doc = yaml.safe_load(Path(path).read_text())
        par, tf = doc["parameters"], doc["transform"]
        t, q = tf["translation"], tf["rotation"]
        xyz = np.array([t["x"], t["y"], t["z"]], float)
        quat = np.array([q["x"], q["y"], q["z"], q["w"]], float)
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        raise HandEyeRefusal(f"{path}: not an easy_handeye2 calibration (needs parameters + "
                             f"transform.translation/rotation): {e!r}") from None
    if par.get("calibration_type") != "eye_on_base":
        raise HandEyeRefusal(f"{path}: calibration_type {par.get('calibration_type')!r} -- this "
                             "cell's camera is fixed: an eye_on_base calibration is required")
    if par.get("robot_base_frame") != BASE_FRAME:
        raise HandEyeRefusal(f"{path}: robot_base_frame {par.get('robot_base_frame')!r}, the "
                             f"pipeline plans in {BASE_FRAME!r}")
    if abs(np.linalg.norm(quat) - 1.0) > 1e-3:
        raise HandEyeRefusal(f"{path}: rotation quaternion has norm {np.linalg.norm(quat):.4f}")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    T[:3, 3] = xyz
    return T, par


def plane_angle_deg(summary: dict, T: np.ndarray):
    """frame_packet's check: the run's table normal vs base +z, through T."""
    n = ((summary.get("packet") or {}).get("plane") or {}).get("n")
    if n is None:
        return None
    n = np.asarray(n, float) / np.linalg.norm(n)
    up_cam = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(abs(up_cam @ n), -1, 1))))


def set_hand_eye(run, calib):
    """Returns (exit_code, report)."""
    run, calib = Path(run), Path(calib)
    sp = run / "summary.json"
    summary = json.loads(sp.read_text())
    T, par = read_calib(calib)

    cam = (summary.get("frame") or {}).get("frame_id")
    if not cam:
        raise HandEyeRefusal(f"{sp}: no frame.frame_id -- cannot tell which camera frame the "
                             "run's depth and K are in")
    if par.get("tracking_base_frame") != cam:
        raise HandEyeRefusal(f"{calib}: tracking_base_frame {par.get('tracking_base_frame')!r} "
                             f"but the run's depth and K are in {cam!r}. Compose the camera "
                             f"driver's static {par.get('tracking_base_frame')} -> {cam} "
                             "transform first; writing one for the other points every grasp "
                             "the wrong way")

    packet = summary.setdefault("packet", {})
    old = packet.get("t_base_cam") or {}
    report = {"run": str(run), "calib": str(calib), "camera_frame": cam,
              "T_base_cam": T.tolist(), "plane_angle_to_base_up_deg": plane_angle_deg(summary, T),
              "warn_deg": fp.pvalues()["plane_up_warn_deg"]}
    if old.get("T") is not None:
        if np.allclose(np.asarray(old["T"], float).reshape(4, 4), T, atol=SAME_T_TOL):
            report["written"] = False
            return 0, report
        raise HandEyeRefusal(f"{sp}: the run already carries T_base_cam (source "
                             f"{old.get('source')!r}) and it differs from {calib} -- a recorded "
                             "hand-eye is never replaced")

    ang = report["plane_angle_to_base_up_deg"]
    rec = {"source": "param", "T": T.tolist(),
           "calib": {"path": str(calib.resolve()), "name": par.get("name"),
                     "sha256": hashlib.sha256(calib.read_bytes()).hexdigest()},
           "note": "written after capture by set_hand_eye.py; the calibration is asserted, not "
                   "measured, to describe the camera at the run's time",
           "plane_angle_to_base_up_deg": ang}
    if old.get("source"):
        rec["replaces_source"] = old["source"]
    packet["t_base_cam"] = rec
    if ang is not None and ang > report["warn_deg"]:
        packet.setdefault("warnings", []).append(
            f"table normal is {ang:.1f} deg from base +z through the hand-eye written by "
            "set_hand_eye.py: the calibration may not describe the camera at the run's time")
    backup = sp.with_name("summary.json.pre_hand_eye")
    if not backup.exists():
        shutil.copy2(sp, backup)
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    report["written"], report["backup"] = True, str(backup)
    return 0, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path, help="easy_handeye2 .calib (YAML)")
    a = ap.parse_args(argv)
    try:
        code, r = set_hand_eye(a.run, a.calib)
    except HandEyeRefusal as e:
        print(f"[set-hand-eye] REFUSED: {e}", file=sys.stderr)
        return 2
    T = np.asarray(r["T_base_cam"])
    print(f"[set-hand-eye] {r['calib']} -> {r['run']}/summary.json packet.t_base_cam "
          f"({'written' if r['written'] else 'already set, unchanged'})")
    print(f"    camera frame: {r['camera_frame']}")
    print(f"    camera in base (m): {np.round(T[:3, 3], 4).tolist()}, optical axis in base: "
          f"{np.round(T[:3, 2], 3).tolist()}")
    ang = r["plane_angle_to_base_up_deg"]
    if ang is None:
        print("    table check: the run has no packet.plane -- nothing to check against")
    else:
        flag = "OK" if ang <= r["warn_deg"] else "WARNING: is this the calibration from the run's time?"
        print(f"    table check: run's table normal is {ang:.1f} deg from base +z "
              f"(warn > {r['warn_deg']:g})  {flag}")
    return code


if __name__ == "__main__":
    sys.exit(main())
