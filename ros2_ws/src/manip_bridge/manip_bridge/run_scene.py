"""One-shot scene pipeline over the bridge services -- the online pipeline's
orchestrator in embryo, and the manual "does it work, show me" harness.

    1. grab one synced (rgb, depth, camera_info) frame from the camera topics
    2. /sam3/segment            prompts -> masks            (writes overlays)
    3. per object:
         a. /oriany/orient           rgb+mask -> az/el/ro + alpha (semantic)
         b. /trellis2/generate_mesh  rgb+mask+depth+K -> canonical + metric GLB
         c. /any6d/estimate          img_to_3d  (or --any6d-mesh trellis)
         d. /pose/estimate           mesh = TRELLIS metric GLB   (FP counterpart)
    4. summary table + summary.json; every artifact under --out/<stamp>/
    5. --watch: keep spinning, print tracked poses as they stream

Every stage is optional (--skip sam3,oriany,trellis2,any6d,pose) and tolerant: a
failed or absent service is logged and the rest continues, so you can run
it with a partial sidecar stack.

Run inside the Ros2Bridge container after `colcon build`:

    ros2 bag play /bags/<bag> --clock --loop &
    ros2 launch manip_bridge bridges.launch.py use_sim_time:=true
    ros2 run manip_bridge run_scene --ros-args -p use_sim_time:=true \\
        -- --prompts teapot mug "robot arm" --watch
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from manip_interfaces.srv import EstimatePose, GenerateMesh, Orient, Segment

from . import DEPTH_TOPIC, INFO_TOPIC, RGB_TOPIC
from .img import (image_to_depth_m, image_to_mono, image_to_rgb,
                  mono_to_image)

PALETTE = [(255, 80, 80), (80, 200, 80), (80, 120, 255), (240, 200, 40),
           (200, 80, 220), (40, 220, 220)]


def pose_to_T(p):
    q = p.pose.orientation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    return T


class SceneRunner(Node):
    def __init__(self, args):
        super().__init__("run_scene")
        self.args = args
        self.declare_parameter("rgb_topic", RGB_TOPIC)
        self.declare_parameter("depth_topic", DEPTH_TOPIC)
        self.declare_parameter("info_topic", INFO_TOPIC)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.frame = None
        self.info = None
        self._got = threading.Event()
        qos = qos_profile_sensor_data
        self.sub_rgb = Subscriber(self, Image, gp("rgb_topic"), qos_profile=qos)
        self.sub_depth = Subscriber(self, Image, gp("depth_topic"), qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=2, slop=0.034)
        self.sync.registerCallback(self._on_frame)
        self.create_subscription(CameraInfo, gp("info_topic"), self._on_info, qos)

        self.cli = {
            "sam3": self.create_client(Segment, "sam3/segment"),
            "oriany": self.create_client(Orient, "oriany/orient"),
            "trellis2": self.create_client(GenerateMesh, "trellis2/generate_mesh"),
            "any6d": self.create_client(EstimatePose, "any6d/estimate"),
            "pose": self.create_client(EstimatePose, "pose/estimate"),
        }
        self.tracks = {}  # topic -> list of (stamp, T)

    # ---- frame capture -----------------------------------------------------
    def _on_info(self, msg):
        self.info = msg

    def _on_frame(self, rgb, depth):
        if self.frame is None and self.info is not None:
            self.frame = (rgb, depth, self.info)
            self._got.set()

    def wait_frame(self, timeout):
        self.get_logger().info("waiting for a synced rgb+depth+camera_info ...")
        if not self._got.wait(timeout):
            self.get_logger().error(
                "no synced frame. Check `ros2 topic hz` on the camera topics, the "
                "rgb/depth/info_topic params, and that bag play uses --clock with "
                "use_sim_time:=true here.")
            return None
        rgb, depth, info = self.frame
        self.get_logger().info(
            f"frame {rgb.width}x{rgb.height} rgb={rgb.encoding} depth={depth.encoding} "
            f"stamp={rgb.header.stamp.sec}.{rgb.header.stamp.nanosec:09d} "
            f"frame_id={rgb.header.frame_id}")
        if (rgb.header.frame_id and depth.header.frame_id
                and rgb.header.frame_id != depth.header.frame_id):
            self.get_logger().error(
                f"depth frame_id '{depth.header.frame_id}' != rgb "
                f"'{rgb.header.frame_id}' -- this depth stream is NOT aligned to "
                "colour. It decodes fine and pairs with the colour K, so every "
                "pose below will look plausible and be wrong. Re-record with "
                "align_depth.enable:=true, or register depth to colour first.")
        return self.frame

    # ---- service helper ----------------------------------------------------
    def call(self, name, req, timeout):
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{name}: service {cli.srv_name} not available (bridge down?)")
            return None
        t0 = time.time()
        fut = cli.call_async(req)
        while not fut.done():
            if time.time() - t0 > timeout:
                self.get_logger().error(f"{name}: timeout after {timeout}s")
                return None
            time.sleep(0.05)
        res = fut.result()
        self.get_logger().info(f"{name}: {'ok' if res.success else 'FAIL'} "
                               f"({time.time() - t0:.1f}s) {res.message}")
        return res if res.success else None

    # ---- tracking watch ----------------------------------------------------
    def watch(self, objs, nss):
        for ns in nss:
            for obj in objs:
                topic = f"{ns}/{obj}/pose"
                self.tracks[topic] = []
                self.create_subscription(
                    PoseStamped, topic,
                    lambda m, t=topic: self.tracks[t].append(
                        (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, pose_to_T(m))),
                    10)

    def report_tracks(self):
        for topic, xs in self.tracks.items():
            if len(xs) < 2:
                print(f"  {topic:32s} {len(xs)} msgs")
                continue
            t = np.array([x[0] for x in xs])
            P = np.array([x[1][:3, 3] for x in xs])
            R = [x[1][:3, :3] for x in xs]
            ang = [np.degrees(np.arccos(np.clip((np.trace(R[0].T @ r) - 1) / 2, -1, 1)))
                   for r in R]
            hz = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
            print(f"  {topic:32s} {len(xs):4d} msgs  {hz:5.1f} Hz  "
                  f"pos std {1e3 * P.std(0).round(1).tolist()} mm  "
                  f"rot drift max {max(ang):.2f} deg")


def draw_overlay(rgb, masks, labels):
    out = rgb.copy()
    for i, (m, lab) in enumerate(zip(masks, labels)):
        c = np.array(PALETTE[i % len(PALETTE)], np.uint8)
        sel = m > 0
        out[sel] = (0.55 * out[sel] + 0.45 * c).astype(np.uint8)
        ys, xs = np.nonzero(sel)
        if len(xs):
            cv2.putText(out, lab, (int(xs.min()), max(int(ys.min()) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(int(v) for v in c), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", default=["object"],
                    help="SAM3 prompts; each becomes an object (except --bg prompts)")
    ap.add_argument("--bg", nargs="*", default=["robot arm"],
                    help="prompts to segment but NOT reconstruct/pose (masked out of depth)")
    ap.add_argument("--skip", default="",
                    help="comma list of sam3,oriany,trellis2,any6d,pose")
    ap.add_argument("--oriany-matting", action="store_true",
                    help="send oriany the FULL frame with an empty mask so the model "
                         "mattes it itself (upstream demo path) instead of a SAM3 crop. "
                         "A/B this against the default before trusting either.")
    ap.add_argument("--any6d-mesh", choices=["img_to_3d", "trellis"], default="img_to_3d",
                    help="Any6D mesh source: its own SAM2+InstantMesh, or the TRELLIS metric GLB")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out", default=os.environ.get("RUN_OUT_DIR", "/data/runs"))
    ap.add_argument("--watch", type=float, nargs="?", const=20.0, default=None,
                    help="after estimates, watch tracked poses for N seconds (default 20)")
    ap.add_argument("--frame-timeout", type=float, default=30.0)
    argv = [a for a in sys.argv[1:]]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = ap.parse_args(rclpy.utilities.remove_ros_args(argv))
    skip = set(filter(None, args.skip.split(",")))

    rclpy.init(args=sys.argv)
    node = SceneRunner(args)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    out = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    summary = {"out": out, "objects": {}}
    log = node.get_logger()

    try:
        fr = node.wait_frame(args.frame_timeout)
        if fr is None:
            return 1
        rgb_msg, depth_msg, info = fr
        rgb = image_to_rgb(rgb_msg)
        depth = image_to_depth_m(depth_msg)
        cv2.imwrite(f"{out}/rgb.png", rgb[..., ::-1])
        cv2.imwrite(f"{out}/depth_mm.png", (depth * 1000).astype(np.uint16))
        dv = np.clip(depth / max(np.percentile(depth[depth > 0], 99), 1e-3), 0, 1)
        cv2.imwrite(f"{out}/depth_viz.png", cv2.applyColorMap((dv * 255).astype(np.uint8),
                                                             cv2.COLORMAP_TURBO))
        summary["frame"] = {"stamp": f"{rgb_msg.header.stamp.sec}.{rgb_msg.header.stamp.nanosec:09d}",
                            "frame_id": rgb_msg.header.frame_id,
                            "K": np.asarray(info.k).reshape(3, 3).tolist(),
                            "depth_valid_frac": float((depth > 0).mean())}
        if summary["frame"]["depth_valid_frac"] < 0.2:
            log.warn(f"only {100 * summary['frame']['depth_valid_frac']:.0f}% of "
                     "depth pixels are valid -- wrong depth topic, or a bag "
                     "recorded before the sensor settled?")

        # ---- SAM3 ----------------------------------------------------------
        masks = {}
        if "sam3" not in skip:
            req = Segment.Request()
            req.rgb = rgb_msg
            req.prompts = list(args.prompts) + list(args.bg)
            req.threshold = float(args.threshold)
            res = node.call("sam3", req, 120)
            if res is not None:
                for p, m, s in zip(res.prompt, res.masks, res.scores):
                    if p not in masks:  # first = best (sorted by score desc)
                        masks[p] = image_to_mono(m)
                        cv2.imwrite(f"{out}/mask_{p.replace(' ', '_')}.png", masks[p])
                        summary.setdefault("sam3", {})[p] = float(s)
                cv2.imwrite(f"{out}/overlay.png",
                            draw_overlay(rgb, list(masks.values()), list(masks))[..., ::-1])
                missing = [p for p in req.prompts if p not in masks]
                if missing:
                    log.warn(f"sam3 found nothing for: {missing}")
        if not masks:
            log.error("no masks; nothing downstream can run")
            return 1

        # background (robot arm) removed from the depth the pose models see
        depth_clean = depth.copy()
        for p in args.bg:
            if p in masks:
                depth_clean[masks[p] > 0] = 0.0
        depth_clean_msg = Image()
        depth_clean_msg.header = depth_msg.header
        depth_clean_msg.height, depth_clean_msg.width = depth_clean.shape
        depth_clean_msg.encoding = "32FC1"
        depth_clean_msg.step = depth_clean.shape[1] * 4
        depth_clean_msg.data = depth_clean.astype(np.float32).tobytes()

        # ---- per object ----------------------------------------------------
        objs = [p for p in args.prompts if p in masks]
        for obj in objs:
            key = obj.replace(" ", "_")
            rec = summary["objects"].setdefault(obj, {})
            mask_msg = mono_to_image(masks[obj], rgb_msg.header)
            trellis_metric = ""

            if "oriany" not in skip:
                req = Orient.Request()
                req.rgb = rgb_msg
                req.mask = Image() if args.oriany_matting else mask_msg
                res = node.call("oriany", req, 180)
                if res is not None:
                    q = res.orientation.quaternion
                    rec["oriany"] = {
                        "azimuth": res.azimuth, "elevation": res.elevation,
                        "rotation": res.rotation, "alpha": res.alpha,
                        "matting": bool(args.oriany_matting),
                        "bbox_xyxy": list(res.bbox_xyxy),
                        "R_cam": Rotation.from_quat(
                            [q.x, q.y, q.z, q.w]).as_matrix().tolist()}

            if "trellis2" not in skip:
                req = GenerateMesh.Request()
                req.rgb, req.mask, req.depth, req.camera_info = rgb_msg, mask_msg, depth_clean_msg, info
                req.output_name = f"{key}_{os.path.basename(out)}"
                res = node.call("trellis2", req, 400)
                if res is not None:
                    rec["trellis2"] = {"glb": res.glb_path, "gen_time": res.gen_time,
                                       "metric_valid": res.metric_valid}
                    if res.metric_valid:
                        trellis_metric = res.metric_glb_path
                        rec["trellis2"].update(
                            metric_glb=res.metric_glb_path, scale=res.scale,
                            rmse_mm=1e3 * res.registration_rmse,
                            cam_T_obj=pose_to_T(res.object_pose).tolist())

            if "any6d" not in skip:
                req = EstimatePose.Request()
                req.rgb, req.depth, req.camera_info, req.mask = rgb_msg, depth_clean_msg, info, mask_msg
                req.obj = key
                if args.any6d_mesh == "trellis" and trellis_metric:
                    req.mesh = trellis_metric
                else:
                    req.img_to_3d = True
                res = node.call("any6d", req, 1000)
                if res is not None:
                    rec["any6d"] = {"cam_T_obj": pose_to_T(res.pose).tolist(),
                                    "extents": list(res.extents), "mesh": res.mesh_path}

            if "pose" not in skip:
                if not trellis_metric:
                    log.warn(f"pose: no TRELLIS metric mesh for '{obj}', skipping FP")
                else:
                    req = EstimatePose.Request()
                    req.rgb, req.depth, req.camera_info, req.mask = rgb_msg, depth_clean_msg, info, mask_msg
                    req.obj, req.mesh = key, trellis_metric
                    res = node.call("pose", req, 400)
                    if res is not None:
                        rec["pose_on_trellis"] = {"cam_T_obj": pose_to_T(res.pose).tolist()}

        # ---- summary -------------------------------------------------------
        print("\n==== scene summary ====")
        for obj, rec in summary["objects"].items():
            print(f"[{obj}]")
            o = rec.get("oriany")
            if o:
                print(f"  {'oriany':16s} az={o['azimuth']:6.1f} el={o['elevation']:6.1f} "
                      f"ro={o['rotation']:7.1f} alpha={o['alpha']} "
                      f"({'matting' if o['matting'] else 'masked crop'})"
                      + ("   <- alpha != 1: front axis defined only up to a "
                         "symmetry group" if o["alpha"] != 1 else ""))
            ts = {}
            for k in ("trellis2", "any6d", "pose_on_trellis"):
                r = rec.get(k)
                if r and "cam_T_obj" in r:
                    T = np.array(r["cam_T_obj"])
                    ts[k] = T
                    extra = ""
                    if k == "trellis2":
                        extra = f" scale={r['scale']:.4f} rmse={r['rmse_mm']:.1f}mm"
                    if k == "any6d":
                        extra = f" extents={np.round(r['extents'], 3).tolist()}"
                    print(f"  {k:16s} t={T[:3, 3].round(3).tolist()}{extra}")
            if "any6d" in ts and "pose_on_trellis" in ts:
                d = np.linalg.norm(ts["any6d"][:3, 3] - ts["pose_on_trellis"][:3, 3])
                Ra, Rb = ts["any6d"][:3, :3], ts["pose_on_trellis"][:3, :3]
                ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
                print(f"  any6d vs FP(trellis): dt={1e3 * d:.1f} mm  dR={ang:.1f} deg  "
                      "(body frames differ -> dR only meaningful if both use the TRELLIS mesh)")
        with open(f"{out}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"artifacts: {out}")

        if args.watch:
            keys = [o.replace(" ", "_") for o in objs]
            node.watch(keys, [ns for ns in ("any6d", "pose") if ns not in skip])
            print(f"\nwatching tracked poses for {args.watch:.0f}s ...")
            time.sleep(args.watch)
            node.report_tracks()
        return 0
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
