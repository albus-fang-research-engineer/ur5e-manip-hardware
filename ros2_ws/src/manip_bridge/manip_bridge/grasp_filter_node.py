"""AnyGrasp proposals x stage-1 grasp TSR -> classified grasps.

Thin rclpy shell around manip_bridge.grasp_filter (pure numpy, tested
offline in test/test_grasp_filter.py). Everything geometric happens there;
this file only moves messages, looks up TF, and draws.

    /grasp/<obj>/grasps          GraspArray   in   grasp_viz output: camera
                                                   optical frame, AnyGrasp axes
    <tsr_topic>                  TSR          in   default /tsr/<obj>/grasp
    /grasp/<obj>/filtered        GraspArray   out  base_frame, e axes (+z approach,
                                                   +x closing), pad-centre origin;
                                                   an EMPTY array on route `empty`
                                                   so a stale list never survives
    /grasp/<obj>/filter_report   String       out  JSON: FilterResult.to_dict()
                                                   + frames + route (latched)
    /grasp/<obj>/filter_markers  MarkerArray  out  rviz2: green kept, amber
                                                   fallback, grey rejected, RGB
                                                   triad at w

SCOPING. Only the STAGE-1 GRASP TSR belongs on tsr_topic. Path and subgoal
TSRs constrain the object BODY frame and go to the constrained planner;
filtering gripper proposals against one is meaningless but would not
crash, so the node refuses any TSR whose name does not start with "grasp".
TSR topics are namespaced by role for the same reason:
/tsr/<obj>/grasp here, /tsr/<obj>/<stage>/{path,subgoal} for the planner.

WHAT THE TSR MESSAGE CONTAINS (for the producer). The tracked object frame
(FoundationPose / Any6D TF child `<obj>`) is an arbitrary mesh frame -- a
rigid carrier, nothing more. w for the grasp TSR is the object's CANONICAL
frame (Orient Anything: z = up, x = front, a fixed rotation in the body
frame computed once at registration) anchored at the call-#2 interaction
point (e.g. handle_center, a fixed body-frame point from grounding):

    t0_w             = T_body_w = [R_body_canonical | p_body_anchor]
    header.frame_id  = "<obj>"   (the tracker's TF child)

World-anchored TSRs (upright) use header.frame_id = base_frame and t0_w in
base. Either way this node roots the TSR by TF at evaluation time:
T_base_w = T_base_frame_id @ t0_w. Bw rows are therefore expressed in the
canonical axes, which is what gives them meaning.

FRAMES. Everything is brought into base_frame via TF: AnyGrasp poses
through base <- camera (eye-to-base calibration; static t_base_cam param
as fallback for bags without it, same as curobo_bridge), the TSR through
base <- header.frame_id (no fallback: an object frame only exists on TF).
grasp_viz ZERO-STAMPS its output on purpose (to survive `bag play --loop`),
so lookups use latest TF, not time-matched. Fine for a static pre-grasp
object; the report records the lookup time. This lookup IS the
"frozen at stage entry": w is rooted once, at detection.

QOS. Both subscriptions are TRANSIENT_LOCAL so a node started after the
detection / TSR was published still receives them. DDS requires the
publisher to be at least as durable: TSR producers (tsr_from_yaml, the
future compiled-TSR node) MUST publish latched, as grasp_viz already does,
or this node silently never receives the TSR.

UPDATE SEMANTICS. Both inputs are latched; whenever either arrives and both
are present the filter runs and all three outputs are re-published. A new
TSR re-filters the last detection without re-running AnyGrasp
(detect-once-at-stage-entry); a new detection under the standing TSR is
filtered immediately.

Parameters
    obj            object tag; sets the /grasp/<obj>/* topics      (required)
    tsr_topic      default /tsr/<obj>/grasp
    base_frame     base_link
    camera_frame   "" = GraspArray header.frame_id
    t_base_cam     float64[16] row-major static fallback; all-zero = use TF
    pad_offset     m along approach from AnyGrasp's surface centre to the
                   Robotiq pad centre (TCP). Gripper constant, measure once
                   against tcp_link. Default 0.
    tol            containment tolerance on TSR distance (1e-6)
    max_distance   fallback cap, metres+radians mixed (0.03)
    fallback_k     max survivors on the tsr_distance route (5)

    ros2 run manip_bridge grasp_filter --ros-args -p obj:=mug
"""

import json
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA, Header, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from manip_interfaces.msg import Grasp, GraspArray, TSR

from .grasp_filter import (ROUTE_CONTAINED, ROUTE_DISTANCE, e_to_anygrasp,
                           filter_grasps, grasps_from_anygrasp, tsr_from_flat)
from .grasp_viz_node import gripper_lines

COLOR = {"kept": (0.1, 0.8, 0.2, 1.0), "fallback": (1.0, 0.65, 0.0, 1.0),
         "rejected": (0.6, 0.6, 0.6, 0.35)}


def _tf_to_mat(tf):
    q, t = tf.transform.rotation, tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _pose_to_Rt(p: Pose):
    q = p.orientation
    return (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
            np.array([p.position.x, p.position.y, p.position.z]))


def _T_to_pose(T) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def _lines_marker(hdr, ns, mid, pts, rgba, width=0.002) -> Marker:
    m = Marker()
    m.header, m.ns, m.id = hdr, ns, mid
    m.type, m.action = Marker.LINE_LIST, Marker.ADD
    m.scale.x = width
    m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
    return m


class GraspFilter(Node):
    def __init__(self):
        super().__init__("grasp_filter")
        self.declare_parameter("obj", "")
        self.declare_parameter("tsr_topic", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("t_base_cam", [0.0] * 16)
        self.declare_parameter("pad_offset", 0.0)
        self.declare_parameter("tol", 1e-6)
        self.declare_parameter("max_distance", 0.03)
        self.declare_parameter("fallback_k", 5)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        obj = gp("obj").replace(" ", "_")
        if not obj:
            raise SystemExit("grasp_filter: -p obj:=<tag> is required "
                             "(the tag grasp_viz was run with)")
        self.obj = obj
        tsr_topic = gp("tsr_topic") or f"/tsr/{obj}/grasp"

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._grasps = None      # last GraspArray
        self._tsr = None         # last TSR

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(GraspArray, f"/grasp/{obj}/grasps", self._on_grasps, latched)
        self.create_subscription(TSR, tsr_topic, self._on_tsr, latched)
        self.pub_filtered = self.create_publisher(GraspArray, f"/grasp/{obj}/filtered", latched)
        self.pub_report = self.create_publisher(String, f"/grasp/{obj}/filter_report", latched)
        self.pub_markers = self.create_publisher(MarkerArray, f"/grasp/{obj}/filter_markers", latched)
        self.get_logger().info(
            f"grasp_filter[{obj}]: /grasp/{obj}/grasps x {tsr_topic} -> "
            f"/grasp/{obj}/filtered in {gp('base_frame')} (pad_offset={gp('pad_offset')})")

    # ------------------------------------------------------------ inputs
    def _on_grasps(self, msg: GraspArray):
        self._grasps = msg
        self._run("grasps")

    def _on_tsr(self, msg: TSR):
        if not msg.name.startswith("grasp"):
            self.get_logger().error(
                f"refusing TSR '{msg.name}' on the grasp filter: only the stage-1 grasp "
                f"TSR belongs here (path/subgoal TSRs constrain the object body frame "
                f"and go to the planner). Not stored.")
            return
        self._tsr = msg
        self._run("tsr")

    # ---------------------------------------------------------------- TF
    def _lookup(self, base, frame):
        try:
            return _tf_to_mat(self._tf.lookup_transform(base, frame, Time()))
        except Exception as e:  # tf2 raises several unrelated types
            self.get_logger().warn(f"no TF {base} <- {frame} ({e})", throttle_duration_sec=5.0)
            return None

    def _t_base_cam(self, header):
        static = np.array(self.get_parameter("t_base_cam").value, float)
        if np.any(static):
            return static.reshape(4, 4)
        cam = self.get_parameter("camera_frame").value or header.frame_id
        return self._lookup(self.get_parameter("base_frame").value, cam)

    # -------------------------------------------------------------- run
    def _run(self, trigger):
        if self._grasps is None or self._tsr is None:
            self.get_logger().info(f"{trigger} received; waiting for "
                                   f"{'TSR' if self._tsr is None else 'grasps'}")
            return
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        base = gp("base_frame")
        ga, tm = self._grasps, self._tsr

        T_base_cam = self._t_base_cam(ga.header)
        if T_base_cam is None:
            return
        if tm.header.frame_id in ("", base):
            T_base_w = np.eye(4)
        else:
            T_base_w = self._lookup(base, tm.header.frame_id)
            if T_base_w is None:
                return
        tsr = tsr_from_flat(tm.t0_w, tm.tw_e, tm.bw, tm.name, T_ref_frame=T_base_w)

        n = len(ga.grasps)
        Rs = np.empty((n, 3, 3))
        ts = np.empty((n, 3))
        for k, g in enumerate(ga.grasps):
            Rs[k], ts[k] = _pose_to_Rt(g.pose)
        scores = [g.score for g in ga.grasps]
        widths = [g.width for g in ga.grasps]
        depths = [g.depth for g in ga.grasps]
        pad = float(gp("pad_offset"))
        grasps = grasps_from_anygrasp(Rs, ts, scores, widths, depths,
                                      T_ref_cam=T_base_cam, pad_offset=pad)
        res = filter_grasps(tsr, grasps, tol=float(gp("tol")),
                            max_distance=float(gp("max_distance")),
                            fallback_k=int(gp("fallback_k")))
        self.get_logger().info(f"[{trigger}] {res.summary()}")

        # ---- /filtered: survivors in route order, e-frame, base ------
        now = self.get_clock().now().to_msg()
        hdr = Header(frame_id=base, stamp=now)
        out = GraspArray(header=hdr)
        for g in res.survivors:
            out.grasps.append(Grasp(pose=_T_to_pose(g.T0_e), score=float(g.score),
                                    width=float(g.width), depth=float(g.depth)))
        self.pub_filtered.publish(out)

        # ---- report -------------------------------------------------
        rep = res.to_dict()
        rep.update(obj=self.obj, trigger=trigger, base_frame=base,
                   camera_frame=ga.header.frame_id, tsr_frame=tm.header.frame_id,
                   tsr_stamp=f"{tm.header.stamp.sec}.{tm.header.stamp.nanosec:09d}",
                   tf_lookup="latest", pad_offset=pad,
                   stamp=f"{now.sec}.{now.nanosec:09d}")
        self.pub_report.publish(String(data=json.dumps(rep)))

        # ---- markers: zero-stamped like grasp_viz, base frame ---------
        mh = Header(frame_id=base)
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        kept = {g.index for g in res.survivors}
        verdict = ("kept" if res.route == ROUTE_CONTAINED else "fallback") \
            if res.route in (ROUTE_CONTAINED, ROUTE_DISTANCE) else "rejected"
        for g in grasps:
            R_ag, t_ag = e_to_anygrasp(g.T0_e, pad)          # glyph is drawn in AnyGrasp convention
            cls = verdict if g.index in kept else "rejected"
            ma.markers.append(_lines_marker(
                mh, f"grasp_{cls}", g.index,
                gripper_lines(t_ag, R_ag, g.width, g.depth), COLOR[cls],
                width=0.003 if cls != "rejected" else 0.0015))
        o, Rw = tsr.T0_w[:3, 3], tsr.T0_w[:3, :3]
        for i, c in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1))):
            ma.markers.append(_lines_marker(mh, "tsr_w", i, [o, o + 0.05 * Rw[:, i]], c, 0.004))
        self.pub_markers.publish(ma)


def main():
    rclpy.init(args=sys.argv)
    node = GraspFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
