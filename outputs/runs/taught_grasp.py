#!/usr/bin/env python3
"""Taught grasps: gripper poses written once in a library mesh's BODY frame,
carried into a scene by that scene's registration, and published where
grasp_reach reads filtered grasps. The AnyGrasp -> grasp_filter path is
untouched; this publishes under its own tag.

    <library>.taught_grasps.json   grasps in the body frame of ONE mesh file
                                   (its sha256 is recorded and checked)
    <run>/summary.json             objects[key].final_mesh_override (the file
                                   registered on), .pose_on_final.cam_T_obj,
                                   packet.t_base_cam.T
            |
            v  T_base_e = T_base_cam . cam_T_obj . T_body_e
    /grasp/<tag>/filtered          GraspArray, base_link, latched: the e frame
                                   grasp_reach expects (+z approach, +x
                                   closing, origin = pad centre), in file order
    <run>/m3/taught_grasps.json    what was published, for the record

A taught grasp is NOT the pipeline's stage-0 grasp: AnyGrasp proposes and the
grasp TSR selects there. It exists to get a first reliable grasp on hardware
and to exercise reach/exec; a result obtained with it is recorded as taught.

Why the body frame: pose_on_final maps the body frame of the exact mesh file
FoundationPose registered on, so a pose written in that frame needs no other
input. The run's canonical frame (up from the table normal, front from Orient
Anything) is rebuilt per run, so it is used only when TEACHING, never stored.

Refused (exit 2): the run registered the key on a different file than the
library's (sha256), or with no --final-mesh at all; a grasp whose approach is
not a unit vector or whose closing axis is not perpendicular to it.

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/taught_grasp.py --run /data/runs/<stamp> --key m2 \\
        --grasps /data/runs/library/lib_teapot.taught_grasps.json
    python3 /data/runs/grasp_reach.py --run /data/runs/<stamp> --obj m2_taught
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ORTHO_TOL = 1e-3


class TaughtRefusal(Exception):
    """An input this script will not publish from; the message names the fix."""


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def T_body_e(g: dict) -> np.ndarray:
    """e frame from a grasp record: z = approach, x = closing, y = z x x,
    origin = the pad centre point, all in the body frame."""
    z = np.asarray(g["approach_body"], float)
    x = np.asarray(g["closing_body"], float)
    if abs(np.linalg.norm(z) - 1) > ORTHO_TOL or abs(np.linalg.norm(x) - 1) > ORTHO_TOL:
        raise TaughtRefusal(f"grasp {g.get('name')!r}: approach/closing must be unit vectors")
    if abs(z @ x) > ORTHO_TOL:
        raise TaughtRefusal(f"grasp {g.get('name')!r}: closing is {np.degrees(np.arcsin(abs(z @ x))):.2f} "
                            "deg off perpendicular to the approach")
    x = x - (z @ x) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = np.asarray(g["point_body"], float)
    return T


def resolve(summary: dict, key: str, lib: dict, mesh_sha=None) -> list[dict]:
    """[{name, T_base_e, approach_base, height_above_table?}] in file order.
    mesh_sha: the sha256 of the file the run registered on (None = read it)."""
    o = (summary.get("objects") or {}).get(key)
    if o is None:
        raise TaughtRefusal(f"summary has no objects[{key!r}]")
    mesh = o.get("final_mesh_override")
    if not mesh:
        raise TaughtRefusal(f"objects[{key!r}] was not registered with --final-mesh: a taught grasp "
                            "is only defined on its library mesh")
    if mesh_sha is None:
        try:
            mesh_sha = _sha256(mesh)
        except OSError as e:
            raise TaughtRefusal(f"cannot hash {mesh} ({e.strerror})") from None
    if mesh_sha != lib["mesh_sha256"]:
        raise TaughtRefusal(f"the run registered {key} on {mesh} (sha256 {mesh_sha[:12]}), but the "
                            f"grasps are taught on sha256 {lib['mesh_sha256'][:12]} -- a different mesh")
    try:
        T_bc = np.asarray(summary["packet"]["t_base_cam"]["T"], float).reshape(4, 4)
        T_co = np.asarray(o["pose_on_final"]["cam_T_obj"], float).reshape(4, 4)
    except (KeyError, TypeError, ValueError) as e:
        raise TaughtRefusal(f"summary lacks packet.t_base_cam.T or objects[{key!r}].pose_on_final "
                            f"({e!r})") from None
    T_bo = T_bc @ T_co
    out = []
    for g in lib["grasps"]:
        T = T_bo @ T_body_e(g)
        out.append({"name": g["name"], "T_base_e": T, "approach_base": T[:3, 2].copy(),
                    "elevation_deg": float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1))))})
    return out


def _T_to_pose(T):
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--key", required=True, help="the registered object key, e.g. m2")
    ap.add_argument("--grasps", required=True, type=Path, help="<library>.taught_grasps.json")
    ap.add_argument("--tag", default=None, help="publishes /grasp/<tag>/filtered; default <key>_taught")
    a = ap.parse_args(argv)
    tag = a.tag or f"{a.key}_taught"
    try:
        summary = json.loads((a.run / "summary.json").read_text())
        lib = json.loads(a.grasps.read_text())
        rows = resolve(summary, a.key, lib)
    except (OSError, ValueError) as e:
        print(f"[taught] REFUSED: {e}", file=sys.stderr)
        return 2
    except TaughtRefusal as e:
        print(f"[taught] REFUSED: {e}", file=sys.stderr)
        return 2
    for r in rows:
        print(f"[taught] {r['name']}: pad centre {np.round(r['T_base_e'][:3, 3], 3).tolist()}  approach "
              f"{np.round(r['approach_base'], 2).tolist()} ({r['elevation_deg']:+.0f} deg below horizontal)")
    out = a.run / "m3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "taught_grasps.json").write_text(json.dumps({
        "key": a.key, "tag": tag, "library": str(a.grasps.resolve()), "mesh_sha256": lib["mesh_sha256"],
        "note": "TAUGHT grasps, not AnyGrasp + TSR: record any result obtained with them as taught",
        "grasps": [{"name": r["name"], "T_base_e": r["T_base_e"].tolist(),
                    "elevation_deg": r["elevation_deg"]} for r in rows]}, indent=1) + "\n")

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import Header

    from manip_interfaces.msg import Grasp, GraspArray
    rclpy.init()
    node = rclpy.create_node("taught_grasp")
    pub = node.create_publisher(GraspArray, f"/grasp/{tag}/filtered",
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    msg = GraspArray(header=Header(frame_id="base_link", stamp=node.get_clock().now().to_msg()))
    n = len(rows)
    for i, r in enumerate(rows):                      # file order = try order: scores descending
        msg.grasps.append(Grasp(pose=_T_to_pose(r["T_base_e"]), score=float(n - i) / n,
                                width=float(lib.get("width_m", 0.0)), depth=0.0))
    pub.publish(msg)
    print(f"[taught] {n} grasp(s) latched on /grasp/{tag}/filtered (base_link) -> "
          f"grasp_reach --obj {tag}; record: {out / 'taught_grasps.json'}; Ctrl-C to retract")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
