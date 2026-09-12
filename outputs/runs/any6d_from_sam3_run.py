#!/usr/bin/env python3
"""Replay a saved run_scene artifact dir into /any6d/estimate (ROS service).

Same idea as trellis2_from_run.py and `grasp_viz --run`: no sam3, the mask is
the one run_scene already wrote. Unlike the ZMQ scripts this goes THROUGH
any6d_bridge, so after the estimate returns the bridge keeps tracking on the
live bag frames and publishes /any6d/<obj>/pose + TF any6d_<obj>, which is
what rviz2 shows.

Inside Ros2Bridge, with the bag looping and bridges.launch.py up:

    python3 /data/runs/any6d_from_run.py /data/runs/20260822_213318 \
        --ros-args -p use_sim_time:=true -- --object mug

Release when done (tracking otherwise continues until the bridge exits):

    python3 /data/runs/any6d_from_run.py --release --ros-args ... -- --object mug
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image

from manip_bridge.img import mono_to_image
from manip_interfaces.srv import EstimatePose, Release


def load_run(run_dir, obj):
    paths = {k: os.path.join(run_dir, v) for k, v in {
        "rgb": "rgb.png", "depth": "depth_mm.png",
        "mask": f"mask_{obj.replace(' ', '_')}.png", "summary": "summary.json"}.items()}
    for p in paths.values():
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    rgb = cv2.imread(paths["rgb"], cv2.IMREAD_COLOR)[..., ::-1].copy()   # disk is BGR
    depth_mm = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
    if depth_mm.dtype != np.uint16:
        sys.exit(f"depth_mm.png decoded as {depth_mm.dtype}, expected uint16")
    mask = cv2.imread(paths["mask"], cv2.IMREAD_GRAYSCALE) > 127
    with open(paths["summary"]) as f:
        fr = json.load(f)["frame"]
    return rgb, depth_mm, mask, np.asarray(fr["K"], np.float64), fr["frame_id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="e.g. /data/runs/20260822_213318")
    ap.add_argument("--object", default="mug")
    ap.add_argument("--mesh", default="",
                    help="reference mesh (sidecar path); default is img_to_3d")
    ap.add_argument("--refine-iter", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=1000.0)
    ap.add_argument("--release", action="store_true", help="call /any6d/release and exit")
    a = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = Node("any6d_from_run")

    if a.release:
        cli = node.create_client(Release, "any6d/release")
        if not cli.wait_for_service(5.0):
            sys.exit("any6d/release not available -- is bridges.launch.py up?")
        req = Release.Request(); req.obj = a.object
        fut = cli.call_async(req); rclpy.spin_until_future_complete(node, fut, timeout_sec=10)
        print(fut.result()); return 0

    if not a.run_dir:
        sys.exit("run_dir required")
    rgb, depth_mm, mask, K, frame_id = load_run(a.run_dir, a.object)
    md = depth_mm[mask & (depth_mm > 0)]
    print(f"mask fg={int(mask.sum())}px  masked depth {md.size}px "
          f"{md.min() / 1e3:.3f}-{md.max() / 1e3:.3f} m median {np.median(md) / 1e3:.3f} m")

    # Header stamp = now (sim time from the bag clock). The bridge only uses
    # the header for the reply/TF, and tracking picks up whatever bag frame
    # comes next; the bag loops, so the anchor scene is the same.
    from std_msgs.msg import Header
    hdr = Header(); hdr.stamp = node.get_clock().now().to_msg(); hdr.frame_id = frame_id

    rgb_msg = Image(header=hdr, height=rgb.shape[0], width=rgb.shape[1],
                    encoding="rgb8", step=rgb.shape[1] * 3, data=rgb.tobytes())
    depth_msg = Image(header=hdr, height=depth_mm.shape[0], width=depth_mm.shape[1],
                      encoding="16UC1", step=depth_mm.shape[1] * 2, data=depth_mm.tobytes())
    info = CameraInfo(header=hdr, height=rgb.shape[0], width=rgb.shape[1])
    info.k = K.reshape(-1).tolist()

    req = EstimatePose.Request()
    req.rgb, req.depth, req.camera_info = rgb_msg, depth_msg, info
    req.mask = mono_to_image(mask, hdr)
    req.obj = a.object
    req.refine_iter = a.refine_iter
    if a.mesh:
        req.mesh = a.mesh
    else:
        req.img_to_3d = True

    cli = node.create_client(EstimatePose, "any6d/estimate")
    if not cli.wait_for_service(5.0):
        sys.exit("any6d/estimate not available -- is bridges.launch.py up?")
    print(f"-> /any6d/estimate obj={a.object} "
          f"{'mesh=' + a.mesh if a.mesh else 'img_to_3d'} (timeout {a.timeout:.0f}s) ...")
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=a.timeout)
    res = fut.result()
    if res is None:
        sys.exit("no reply -- check `docker compose logs -f any6d`")
    if not res.success:
        sys.exit(f"estimate failed: {res.message}")

    p, q = res.pose.pose.position, res.pose.pose.orientation
    print(f"ok\n  t       {p.x:.4f} {p.y:.4f} {p.z:.4f} m   (cf. masked depth median above)")
    print(f"  q(xyzw) {q.x:.4f} {q.y:.4f} {q.z:.4f} {q.w:.4f}")
    print(f"  extents {np.round(res.extents, 4).tolist()} m")
    print(f"  mesh    {res.mesh_path}")
    print("bridge is now tracking -> /any6d/%s/pose + TF any6d_%s; "
          "--release to stop" % (a.object, a.object))
    node.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())