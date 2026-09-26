"""One-shot scene pipeline over the bridge services -- the online pipeline's
orchestrator in embryo, and the manual "does it work, show me" harness.

    1. frame packet (manip_bridge/frame_packet.py): N synced frames -> median
       depth, joint state, T_base_cam, robot mask, table plane, and the depth
       contract, with stop-at-gate: misaligned depth, a moving arm, no
       hand-eye, or no table plane ends the run (summary.json still written,
       summary["stop"] says why). --from-run DIR replays a saved frame instead
    2. set-of-mark: /sam3/segment with a label-free prompt ("object") at a
       low floor -> scene_marks keeps the instances the pipeline can act on
       (depth, plane, robot mask from the packet) -> the sim's canonical mark
       set in <out>/marks/ (marked.png is what the VLM will see). Every
       instance and why it was or wasn't marked goes into summary["marks"].
       --reuse-marks (with --from-run) takes the source run's mark set
       instead, so mark ids stay fixed across invocations
    3. per REGISTERED mark (--register none|all|<ids>, default none), keyed
       m<id> (service session key, TF child frame, file stem):
         a. /oriany/orient           rgb+mask -> az/el/ro + alpha (semantic)
         b. /trellis2/generate_mesh  rgb+mask -> canonical (unit-box) GLB
         c. /any6d/estimate          mesh = that GLB; Any6D does the metric
                                     scaling itself and exports
                                     final_mesh_<obj>.obj (--any6d-mesh
                                     img_to_3d keeps its own InstantMesh path)
         d. /pose/estimate           mesh = Any6D's final_mesh -> FoundationPose
                                     is the tracker of record on that file.
                                     decide=true (default; --no-decide to A/B):
                                     re-rank + Orient Anything choose the yaw
                                     (manip_bridge/pose_decide.py); the record is
                                     summary.objects.m<id>.decision. A hard stop
                                     (re-rank top-K disagree) ends the run, exit 2.
       3a's Orient Anything call is a DIAGNOSTIC: its square-padded framing is
       not the decider's, and nothing reads it (the decider's real-crop reading
       is decision.oa_real / decision.front).
    4. summary table + summary.json; every artifact under --out/<stamp>/,
       including each registered object's GLB / final mesh copied into
       objects/m<id>/ (the shared sidecar paths are overwritten by the next
       scene) and mask_m<id>.png per mark for the per-object drivers
    5. --watch: keep spinning, print tracked poses as they stream

The body frame of record is Any6D's final_mesh_<obj>.obj: FoundationPose
registers and tracks on it, and the render asset the grounding renderers
load is built from it (manip_bridge/render_asset.py). The TRELLIS sidecar's
own metric-scale branch (_metric.glb, metric_scale.py) is recorded in the
summary when the bridge computes it, but nothing here consumes it.

Every stage after the packet is optional (--skip sam3,oriany,trellis2,any6d,pose)
and tolerant: a failed or absent service is logged and the rest continues, so
you can run it with a partial sidecar stack. The packet is not optional: its
gates are what make everything after it meaningful.

Run inside the Ros2Bridge container after `colcon build`:

    ros2 bag play /bags/<bag> --clock --loop &
    ros2 launch manip_bridge bridges.launch.py use_sim_time:=true
    ros2 run manip_bridge run_scene --ros-args -p use_sim_time:=true \\
        -- --skip oriany,trellis2,any6d,pose          # packet + marks
    ros2 run manip_bridge run_scene -- --from-run /data/runs/<stamp> \\
        --reuse-marks --register 2 --watch            # register mark 2
"""

import argparse
import json
import os
import shutil
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
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener

from manip_interfaces.srv import EstimatePose, GenerateMesh, Orient, Segment

from . import DEPTH_TOPIC, INFO_TOPIC, RGB_TOPIC
from . import frame_packet as fp
from . import scene_marks as sm
from .pose_decide import summary_line as decide_line
from .img import (image_to_depth_m, image_to_mono, image_to_rgb,
                  mono_to_image)
from .zmq_client import SidecarClient

CUROBO_ADDR = os.environ.get("CUROBO_ADDR", "tcp://127.0.0.1:5671")

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
        self.declare_parameter("joints_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gp = gp

        self.frames = []            # [(rgb_msg, depth_msg)], filled while collecting
        self.n_want = 0
        self.info = None
        self._got = threading.Event()
        self._flock = threading.Lock()
        self.joint_samples = []     # fp.arm_sample tuples, bounded below
        self._jlock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = qos_profile_sensor_data
        self.sub_rgb = Subscriber(self, Image, gp("rgb_topic"), qos_profile=qos)
        self.sub_depth = Subscriber(self, Image, gp("depth_topic"), qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=2, slop=0.034)
        self.sync.registerCallback(self._on_frame)
        self.create_subscription(CameraInfo, gp("info_topic"), self._on_info, qos)
        self.create_subscription(JointState, gp("joints_topic"), self._on_joints, qos)

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
        with self._flock:
            if self.info is None or len(self.frames) >= self.n_want:
                return
            self.frames.append((rgb, depth))
            if len(self.frames) >= self.n_want:
                self._got.set()

    def _on_joints(self, msg):
        s = fp.arm_sample(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                          list(msg.name), list(msg.position),
                          list(msg.velocity) if len(msg.velocity) else None)
        if s is not None:
            with self._jlock:
                self.joint_samples.append(s)
                del self.joint_samples[:-2000]

    def wait_frames(self, n, timeout):
        """n synced (rgb, depth) pairs, or None."""
        with self._flock:
            self.frames, self.n_want = [], int(n)
            self._got.clear()
        self.get_logger().info(f"collecting {n} synced rgb+depth frames (+camera_info) ...")
        if not self._got.wait(timeout):
            self.get_logger().error(
                f"got {len(self.frames)}/{n} synced frames. Check `ros2 topic hz` on the camera "
                "topics, the rgb/depth/info_topic params, and that bag play uses --clock with "
                "use_sim_time:=true here.")
            return None
        with self._flock:
            frames, self.n_want = list(self.frames), 0
        rgb = frames[len(frames) // 2][0]
        self.get_logger().info(
            f"frames {rgb.width}x{rgb.height} rgb={rgb.encoding} depth={frames[0][1].encoding} "
            f"ref stamp={rgb.header.stamp.sec}.{rgb.header.stamp.nanosec:09d} "
            f"frame_id={rgb.header.frame_id}")
        return frames

    def lookup_t_base_cam(self, cam_frame, stamp_msg):
        """base <- camera at the frame stamp, else the latest; None if neither."""
        base = self.gp("base_frame")
        t_ref = stamp_msg.sec + stamp_msg.nanosec * 1e-9
        for src, when in (("tf", Time.from_msg(stamp_msg)), ("tf_latest", Time())):
            try:
                tf = self.tf_buffer.lookup_transform(base, cam_frame, when,
                                                     timeout=Duration(seconds=2.0))
            except Exception as e:  # tf2 Lookup/Extrapolation/Connectivity
                self.get_logger().warn(f"TF {base} <- {cam_frame} ({src}): {e}")
                continue
            t = tf.transform
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z,
                                            t.rotation.w]).as_matrix()
            T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
            ts = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            return {"T": T, "source": src, "base_frame": base, "camera_frame": cam_frame,
                    "dt_s": (ts - t_ref) if (src == "tf_latest" and ts > 0) else
                            (0.0 if src == "tf" else None)}
        return None

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


def _header(frame_id, stamp):
    h = Header()
    h.frame_id = frame_id or ""
    if stamp:
        sec, _, nsec = str(stamp).partition(".")
        h.stamp = TimeMsg(sec=int(sec), nanosec=int((nsec or "0").ljust(9, "0")[:9]))
    return h


def _rgb_msg(rgb, header):
    m = Image()
    m.header = header
    m.height, m.width = rgb.shape[:2]
    m.encoding, m.step = "rgb8", rgb.shape[1] * 3
    m.data = np.ascontiguousarray(rgb, np.uint8).tobytes()
    return m


def _info_msg(K, W, H, header):
    m = CameraInfo()
    m.header = header
    m.width, m.height = int(W), int(H)
    m.k = [float(x) for x in np.asarray(K, float).ravel()]
    m.p = [m.k[0], 0.0, m.k[2], 0.0, 0.0, m.k[4], m.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return m


def _meta(msg, topic, encoding=True):
    d = {"topic": topic, "frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height}
    if encoding:
        d["encoding"] = msg.encoding
    return d


def _robot_mask_fn(log):
    """cuRobo robot_mask over ZMQ, or None when the sidecar is not up (a
    degrade recorded by the packet, not a stop)."""
    cli = SidecarClient(CUROBO_ADDR, timeout_ms=5000)
    if not cli.ping(timeout_ms=3000):
        log.warn(f"cuRobo sidecar not answering at {CUROBO_ADDR}: no robot mask")
        return None

    def fn(depth, K, T, q, names):
        rep = cli.call({"cmd": "robot_mask", "depth": np.asarray(depth, np.float32),
                        "intrinsics": np.asarray(K, np.float32),
                        "T_base_cam": np.asarray(T, np.float32), "q": list(q),
                        "joint_names": list(names)},
                       timeout_ms=180000)          # first call loads the robot model
        return np.asarray(rep["mask"], bool)
    return fn


def build_packet(node, args, log):
    """-> (rgb, rgb_msg, info, header, depth, packet, robot_mask). Raises GateStop."""
    static_T = None
    if args.t_base_cam:
        static_T = {"T": np.asarray(args.t_base_cam, float).reshape(4, 4), "source": "param",
                    "base_frame": node.gp("base_frame"), "camera_frame": None, "dt_s": None}
    if args.from_run:
        src = fp.load_run(args.from_run)
        rgb, K = src["rgb"], np.asarray(src["K"], float)
        header = _header(src["frame_id"], src["stamp"])
        rgb_msg, info = _rgb_msg(rgb, header), _info_msg(K, rgb.shape[1], rgb.shape[0], header)
        kw = dict(depths=[src["depth"]], source={"from_run": os.path.abspath(args.from_run)},
                  contract=src["contract"], stamps=None, joints=src["joints"],
                  t_base_cam=static_T or src["t_base_cam"])
    else:
        frames = node.wait_frames(args.frames, args.frame_timeout)
        if frames is None:
            raise fp.GateStop("capture", "no synced frames")
        info = node.info
        rgb_msg, depth_msg = frames[len(frames) // 2]
        contract = fp.depth_contract(
            _meta(rgb_msg, node.gp("rgb_topic")), _meta(depth_msg, node.gp("depth_topic")),
            {**_meta(info, node.gp("info_topic"), encoding=False),
             "distortion_model": info.distortion_model, "D": list(info.d)})
        rgb, header, K = image_to_rgb(rgb_msg), rgb_msg.header, np.asarray(info.k, float).reshape(3, 3)
        ts = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9  # noqa: E731
        with node._jlock:
            samples = list(node.joint_samples)
        kw = dict(depths=[image_to_depth_m(d) for _, d in frames], source="live",
                  contract=contract,
                  stamps={"ref": ts(header), "rgb": [ts(r.header) for r, _ in frames],
                          "depth": [ts(d.header) for _, d in frames]},
                  joints={"samples": samples, "topic": node.gp("joints_topic")},
                  t_base_cam=static_T or node.lookup_t_base_cam(header.frame_id, header.stamp))
    need_mask = kw["joints"] is not None and kw["t_base_cam"] is not None
    depth, packet, rmask = fp.build(K=K, robot_mask_fn=_robot_mask_fn(log) if need_mask else None,
                                    **kw)
    return rgb, rgb_msg, info, header, depth, packet, rmask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", default=["object"],
                    help="SAM3 marking prompts. Label-free by default: naming objects is "
                         "the VLM's job on the marked image, not the segmenter's")
    ap.add_argument("--bg", nargs="*", default=["robot arm"],
                    help="fallback arm prompt(s), used only when the packet has no robot "
                         "mask; the arm is dropped from marks and removed from depth")
    ap.add_argument("--register", default="none",
                    help="marks to reconstruct + register: none (default), all, or ids "
                         "like 2,3 (ids from <out>/marks/marked.png)")
    ap.add_argument("--reuse-marks", action="store_true",
                    help="with --from-run: take the source run's mark set (and arm mask) "
                         "instead of re-running SAM3, so the ids you chose stay valid")
    ap.add_argument("--no-decide", action="store_true",
                    help="FoundationPose without the re-rank / Orient Anything yaw decision "
                         "(the pre-0007 path, for A/B only)")
    ap.add_argument("--decide-debug", action="store_true",
                    help="also save the decider's annotated contact sheet per object")
    ap.add_argument("--skip", default="",
                    help="comma list of sam3,oriany,trellis2,any6d,pose")
    ap.add_argument("--oriany-matting", action="store_true",
                    help="send oriany the FULL frame with an empty mask so the model "
                         "mattes it itself (upstream demo path) instead of a SAM3 crop. "
                         "A/B this against the default before trusting either.")
    ap.add_argument("--any6d-mesh", choices=["img_to_3d", "trellis"], default="trellis",
                    help="Any6D mesh source: the TRELLIS.2 canonical GLB (default; Any6D "
                         "rescales it), or Any6D's own SAM2+InstantMesh")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="SAM3 request floor. Selection applies scene_marks score_min "
                         "(0.2) and records the instances in between as below_score. "
                         "<= 0 means the sidecar default 0.5, where 'object' finds nothing")
    ap.add_argument("--out", default=os.environ.get("RUN_OUT_DIR", "/data/runs"))
    ap.add_argument("--watch", type=float, nargs="?", const=20.0, default=None,
                    help="after estimates, watch tracked poses for N seconds (default 20)")
    ap.add_argument("--frame-timeout", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=fp.PACKET_PARAMS["n_frames"][0],
                    help="synced frames in the packet; depth is their per-pixel median")
    ap.add_argument("--from-run", default=None, metavar="DIR",
                    help="replay a saved run's frame (rgb.png, depth_mm.png, summary.json) "
                         "instead of the camera; fields it never recorded are degrades")
    ap.add_argument("--t-base-cam", type=float, nargs=16, default=None, metavar="T",
                    help="static camera-in-base transform, row-major 4x4, when TF has no "
                         "hand-eye (recorded as source 'param'; same contract as curobo_bridge)")
    argv = [a for a in sys.argv[1:]]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = ap.parse_args(rclpy.utilities.remove_ros_args(argv))
    skip = set(filter(None, args.skip.split(",")))
    if args.reuse_marks and not args.from_run:
        ap.error("--reuse-marks needs --from-run (the run whose marks to reuse)")
    if args.register.strip().lower() != "none" and "sam3" in skip and not args.reuse_marks:
        ap.error("--register needs marks: don't skip sam3, or use --from-run ... --reuse-marks")

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

    def write_summary():
        with open(f"{out}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    try:
        return _run(node, args, skip, out, summary, log, write_summary)
    except fp.GateStop as e:
        summary["stop"] = {"stage": e.stage, "reason": e.reason}
        write_summary()
        log.error(f"STOP at {e.stage}: {e.reason}")
        print(f"stopped at gate '{e.stage}'; summary: {out}/summary.json")
        return 2
    finally:
        _shutdown(ex, spin, node, log)


def _shutdown(ex, spin, node, log):
    """Tear down in dependency order. rclpy.shutdown() alone, with the
    executor still spinning in its thread, lets the interpreter finalize
    while a callback (TF, joint_states) is inside rcl, and a C++ destructor
    calls std::terminate: "terminate called without an active exception",
    exit 250 -- which also destroys the 0 / 2 exit code the gates report."""
    try:
        ex.shutdown(timeout_sec=2.0)       # wakes the wait set, drains callbacks, stops the pool
    except Exception as e:                  # never let teardown mask the run's exit code
        log.warn(f"executor shutdown: {e}")
    spin.join(timeout=3.0)
    if spin.is_alive():
        log.warn("executor thread still alive after shutdown")
    try:
        node.destroy_node()
    except Exception as e:
        log.warn(f"destroy_node: {e}")
    if rclpy.ok():
        rclpy.shutdown()


def _keep(out, key, paths):
    """Copy an object's sidecar artifacts into <out>/objects/<key>/. The
    sidecar paths are shared (final_mesh_m1.obj is the next scene's m1 too),
    so the run dir is the record. A path this container can't see is noted,
    not an error: the sidecars pass paths among themselves."""
    rec, dst = {}, os.path.join(out, "objects", key)
    for name, p in paths.items():
        if not p:
            continue
        if not os.path.isfile(p):
            rec[name] = f"not visible from run_scene: {p}"
            continue
        os.makedirs(dst, exist_ok=True)
        rec[name] = shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
    return rec


def _run(node, args, skip, out, summary, log, write_summary):
    rgb, rgb_msg, info, header, depth, packet, rmask = build_packet(node, args, log)
    summary["packet"] = packet
    cv2.imwrite(f"{out}/rgb.png", rgb[..., ::-1])
    cv2.imwrite(f"{out}/depth_mm.png", np.round(depth * 1000).astype(np.uint16))
    dv = np.clip(depth / max(np.percentile(depth[depth > 0], 99), 1e-3), 0, 1)
    cv2.imwrite(f"{out}/depth_viz.png", cv2.applyColorMap((dv * 255).astype(np.uint8),
                                                         cv2.COLORMAP_TURBO))
    if rmask is not None:
        cv2.imwrite(f"{out}/robot_mask.png", rmask.astype(np.uint8) * 255)
    summary["frame"] = {"stamp": f"{header.stamp.sec}.{header.stamp.nanosec:09d}",
                        "frame_id": header.frame_id,
                        "K": np.asarray(info.k).reshape(3, 3).tolist(),
                        "depth_valid_frac": float((depth > 0).mean())}
    pl = packet["plane"]
    log.info(f"packet: {packet['frames']['n']} frame(s), plane n={np.round(pl['n'], 3).tolist()} "
             f"d={pl['d']:.3f} m inliers={100 * pl['inlier_frac']:.0f}% rms={1e3 * pl['rms_m']:.1f} mm"
             + (f" vs base z {pl['angle_to_up_deg']:.1f} deg" if "angle_to_up_deg" in pl else ""))
    for w in packet["warnings"]:
        log.warn(f"packet: {w}")
    for d in packet["degrades"]:
        log.warn(f"packet DEGRADE: {d}")
    if summary["frame"]["depth_valid_frac"] < 0.2:
        log.warn(f"only {100 * summary['frame']['depth_valid_frac']:.0f}% of "
                 "depth pixels are valid -- wrong depth topic, or a bag "
                 "recorded before the sensor settled?")
    depth_msg = Image()
    depth_msg.header = header

    # ---- marks (set-of-mark) ---------------------------------------------
    K = np.asarray(packet["K"], float)
    arm_mask, arm_source = (rmask, "robot_mask") if rmask is not None else (None, None)
    mark_dir = os.path.join(out, "marks")
    if args.reuse_marks:
        src_marks = os.path.join(args.from_run, "marks")
        if not os.path.isfile(os.path.join(src_marks, "marks.json")):
            raise fp.GateStop("marks", f"--reuse-marks: no mark set at {src_marks}")
        shutil.copytree(src_marks, mark_dir)
        marks, _ = sm.load_mark_set(mark_dir)
        with open(os.path.join(args.from_run, "summary.json")) as f:
            src_rec = json.load(f).get("marks")
        summary["marks"] = {"reused_from": os.path.abspath(src_marks), "source_record": src_rec}
        src_arm = os.path.join(args.from_run, "arm_mask.png")
        if arm_mask is None and os.path.isfile(src_arm):
            arm_mask = cv2.imread(src_arm, cv2.IMREAD_GRAYSCALE) > 127
            arm_source = f"reused from source run ({(src_rec or {}).get('arm_source')})"
        log.info(f"marks: reused {sorted(marks)} from {src_marks}")
    elif "sam3" in skip:                         # packet-only run: done, not a failure
        write_summary()
        print(f"packet only (--skip sam3); summary: {out}/summary.json")
        return 0
    else:
        use_bg = rmask is None and bool(args.bg)
        req = Segment.Request()
        req.rgb = rgb_msg
        req.prompts = list(args.prompts) + (list(args.bg) if use_bg else [])
        req.threshold = float(args.threshold)
        res = node.call("sam3", req, 120)
        if res is None:
            raise fp.GateStop("sam3", "segmentation failed (service down or error)")
        inst, bg, n_bg = sm.split_instances(
            res.prompt, [image_to_mono(m) for m in res.masks], res.scores,
            set(args.prompts), set(args.bg) if use_bg else set(), sm.values()["bg_score_min"])
        if use_bg:
            arm_mask = bg if bg is not None else np.zeros(depth.shape, bool)
            arm_source = f"bg_prompt {list(args.bg)} ({n_bg} instance(s))"
        kept, report = sm.select_instances(inst, depth=depth, K=K, plane=packet["plane"],
                                           arm_mask=arm_mask, arm_source=arm_source)
        report.pop("plane", None)                # recorded once, in summary["packet"]
        summary["marks"] = {**report, "sam3": {"prompts": list(req.prompts),
                                               "threshold": req.threshold,
                                               "instances": len(inst)}}
        if not kept:
            raise fp.GateStop("marks", f"no instance survived selection ({len(inst)} from "
                                       "SAM3); summary['marks'] lists every drop reason")
        from . import marks_compat               # loud if the sim mount is missing
        ms, id_map = sm.write_mark_set(rgb, kept, mark_dir)
        marks = {mid: ms.load_mask(mid) for mid in ms.ids()}
        summary["marks"].update(id_map={str(k): v for k, v in id_map.items()},
                                sim=marks_compat.provenance())
        inv = {v: k for k, v in id_map.items()}
        print(f"\n==== marks: {len(kept)} of {len(inst)} instances  (arm: {arm_source}) ====")
        for r in report["instances"]:
            res_s = (f"MARK {inv[r['idx']]}" if r["idx"] in inv
                     else "drop: " + ", ".join(r["reasons"]))
            print(f"  {r['idx']:3d}  {r['score']:.3f}  {r['area']:8d}  {res_s}")
        print(f"  {mark_dir}/marked.png")
    for mid, m in marks.items():                 # per-object drivers read mask_<object>.png
        cv2.imwrite(f"{out}/mask_{sm.obj_key(mid)}.png", m.astype(np.uint8) * 255)
    if arm_mask is not None:
        cv2.imwrite(f"{out}/arm_mask.png", arm_mask.astype(np.uint8) * 255)
    summary["arm"] = {"source": arm_source,
                      "pixels": int(arm_mask.sum()) if arm_mask is not None else 0}

    # arm removed from the depth the pose models see
    depth_clean = depth.copy()
    if arm_mask is not None:
        depth_clean[arm_mask] = 0.0
    depth_clean_msg = Image()
    depth_clean_msg.header = depth_msg.header
    depth_clean_msg.height, depth_clean_msg.width = depth_clean.shape
    depth_clean_msg.encoding = "32FC1"
    depth_clean_msg.step = depth_clean.shape[1] * 4
    depth_clean_msg.data = depth_clean.astype(np.float32).tobytes()

    # ---- per registered mark ------------------------------------------
    try:
        reg = sm.parse_register(args.register, marks)
    except ValueError as e:
        raise fp.GateStop("register", str(e))
    objs = [sm.obj_key(mid) for mid in reg]
    summary["registered"] = objs
    if not objs:
        log.info("--register none: marks only (pick ids from marks/marked.png, then "
                 "--from-run <this run> --reuse-marks --register <ids>)")
    for mid in reg:
        obj = key = sm.obj_key(mid)
        rec = summary["objects"].setdefault(key, {"mark": mid})
        mask_msg = mono_to_image(marks[mid], rgb_msg.header)
        trellis_glb = ""       # canonical GLB -> Any6D's input
        final_mesh = ""        # Any6D's scaled export -> FoundationPose's input

        if "oriany" not in skip:
            req = Orient.Request()
            req.rgb = rgb_msg
            req.mask = Image() if args.oriany_matting else mask_msg
            res = node.call("oriany", req, 180)
            if res is not None:
                q = res.orientation.quaternion
                rec["oriany"] = {
                    "role": "diagnostic (square-padded framing; not read by the decider)",
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
                trellis_glb = res.glb_path
                rec["trellis2"] = {"glb": res.glb_path, "gen_time": res.gen_time,
                                   "metric_valid": res.metric_valid}
                if res.metric_valid:   # sidecar's own scale branch: recorded, not used
                    rec["trellis2"].update(
                        metric_glb=res.metric_glb_path, scale=res.scale,
                        rmse_mm=1e3 * res.registration_rmse,
                        cam_T_obj=pose_to_T(res.object_pose).tolist())

        if "any6d" not in skip:
            req = EstimatePose.Request()
            req.rgb, req.depth, req.camera_info, req.mask = rgb_msg, depth_clean_msg, info, mask_msg
            req.obj = key
            if args.any6d_mesh == "trellis":
                if not trellis_glb:
                    log.warn(f"any6d: no TRELLIS GLB for '{obj}', skipping "
                             "(--any6d-mesh img_to_3d to use InstantMesh instead)")
                    res = None
                else:
                    req.mesh = trellis_glb
                    res = node.call("any6d", req, 1000)
            else:
                req.img_to_3d = True
                res = node.call("any6d", req, 1000)
            if res is not None:
                final_mesh = res.mesh_path
                rec["any6d"] = {"cam_T_obj": pose_to_T(res.pose).tolist(),
                                "extents": list(res.extents), "mesh": res.mesh_path,
                                "source": args.any6d_mesh}

        if "pose" not in skip:
            if not final_mesh:
                log.warn(f"pose: no Any6D final mesh for '{obj}', skipping FP "
                         "(FoundationPose registers on Any6D's scaled export)")
            else:
                req = EstimatePose.Request()
                req.rgb, req.depth, req.camera_info, req.mask = rgb_msg, depth_clean_msg, info, mask_msg
                req.obj, req.mesh = key, final_mesh
                req.decide, req.decide_debug = not args.no_decide, bool(args.decide_debug)
                res = node.call("pose", req, 600)
                if res is not None:
                    rec["pose_on_final"] = {"cam_T_obj": pose_to_T(res.pose).tolist(),
                                            "mesh": final_mesh, "decide": bool(req.decide)}
                    if req.decide:
                        rec["decision"] = json.loads(res.decision_json) if res.decision_json else None
                        if res.decision_sheet.width:
                            os.makedirs(os.path.join(out, "objects", key), exist_ok=True)
                            cv2.imwrite(os.path.join(out, "objects", key, "decision_sheet.png"),
                                        image_to_rgb(res.decision_sheet)[..., ::-1])
                        if rec["decision"] is None:
                            raise fp.GateStop("decision", f"{key}: decide=true but no decision record "
                                                          "(stale pose bridge? rebuild + restart it)")
                        if res.decision_hard_stop:
                            rec["copied"] = _keep(out, key, {"trellis_glb": trellis_glb,
                                                             "final_mesh": final_mesh})
                            raise fp.GateStop("decision", f"{key}: {res.decision_reason} -- "
                                                          f"{rec['decision'].get('note', '')}")

        rec["copied"] = _keep(out, key, {"trellis_glb": trellis_glb, "final_mesh": final_mesh})

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
        for k in ("trellis2", "any6d", "pose_on_final"):
            r = rec.get(k)
            if r and "cam_T_obj" in r:
                T = np.array(r["cam_T_obj"])
                ts[k] = T
                extra = ""
                if k == "trellis2":
                    extra = f" scale={r['scale']:.4f} rmse={r['rmse_mm']:.1f}mm (sidecar branch, unused)"
                if k == "any6d":
                    extra = f" extents={np.round(r['extents'], 3).tolist()}"
                print(f"  {k:16s} t={T[:3, 3].round(3).tolist()}{extra}")
        dec = rec.get("decision")
        if dec:
            print("  " + decide_line(dec))
        if "any6d" in ts and "pose_on_final" in ts:
            # Same mesh, same body frame (Any6D's last reset_object is on
            # the already-centred scaled mesh, so its pose compensation is
            # the identity and final_mesh IS the frame): this is a
            # registration-consistency check, and a large gap means one of
            # the two registrations is wrong -- not a frame difference.
            d = np.linalg.norm(ts["any6d"][:3, 3] - ts["pose_on_final"][:3, 3])
            Ra, Rb = ts["any6d"][:3, :3], ts["pose_on_final"][:3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
            flag = "   <- disagree: inspect both registrations" if (d > 0.01 or ang > 5) else ""
            print(f"  any6d vs FP(final): dt={1e3 * d:.1f} mm  dR={ang:.1f} deg  "
                  f"(same mesh, same frame){flag}")
    write_summary()
    print(f"artifacts: {out}")

    if args.watch:
        keys = [o.replace(" ", "_") for o in objs]
        node.watch(keys, [ns for ns in ("any6d", "pose") if ns not in skip])
        print(f"\nwatching tracked poses for {args.watch:.0f}s ...")
        time.sleep(args.watch)
        node.report_tracks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
