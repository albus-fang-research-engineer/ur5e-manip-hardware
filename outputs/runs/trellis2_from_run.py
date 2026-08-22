#!/usr/bin/env python3
"""Drive the TRELLIS.2 sidecar from a saved run_scene artifact directory.

No ROS, no bag, no SAM3 -- just the PNGs run_scene already wrote. This lets
you test trellis2 as the ONLY model resident on the GPU, which matters on a
shared box where the img_to_3d path OOMs.

Run inside the Ros2Bridge container (it has pyzmq/msgpack/numpy/opencv, is
host-networked so 127.0.0.1:5669 reaches the sidecar, and mounts
./outputs/runs at /data/runs and ./trellis2_runtime/outputs at /data/meshes):

    docker exec -it Ros2Bridge bash
    python3 /data/runs/trellis2_from_run.py /data/runs/20260822_210159 \
        --object mug --texture-size 1024 --decimation-target 200000

Wire format notes (these differ from the other sidecars -- see
trellis2_server/server.py):
  * the key is "op", not "cmd"
  * depth must be float32 METERS; run_scene saved uint16 millimetres
  * the metric block is only computed when BOTH depth and K are supplied,
    and it is nested under reply["metric"], not flat like GenerateMesh.srv
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def load_frame(run_dir, object_name):
    """Read back exactly what run_scene wrote, in the units the sidecar wants."""
    rgb_path = os.path.join(run_dir, "rgb.png")
    mask_path = os.path.join(run_dir, f"mask_{object_name.replace(' ', '_')}.png")
    depth_path = os.path.join(run_dir, "depth_mm.png")
    summary_path = os.path.join(run_dir, "summary.json")

    for p in (rgb_path, mask_path, depth_path, summary_path):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    # run_scene wrote rgb[..., ::-1], i.e. BGR on disk. Flip back to RGB.
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[..., ::-1].copy()

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # IMREAD_UNCHANGED or the 16-bit depth silently truncates to 8-bit.
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_mm.dtype != np.uint16:
        sys.exit(f"depth_mm.png decoded as {depth_mm.dtype}, expected uint16")
    depth = (depth_mm.astype(np.float32) * 1e-3)

    with open(summary_path) as f:
        summary = json.load(f)
    K = np.asarray(summary["frame"]["K"], dtype=np.float64)

    return rgb, mask, depth, K, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="e.g. /data/runs/20260822_210159")
    ap.add_argument("--object", default="mug", help="matches mask_<object>.png")
    ap.add_argument("--addr", default=os.environ.get("TRELLIS_ADDR",
                                                     "tcp://127.0.0.1:5669"))
    ap.add_argument("--seed", type=int, default=42)
    # Both defaults are LOWER than the sidecar's (500k / 2048). Texture baking
    # and decimation are the tail of the memory curve; shrink them for a first
    # smoke test, then raise once you know the pipeline runs at all.
    ap.add_argument("--decimation-target", type=int, default=200_000)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--no-metric", action="store_true",
                    help="skip depth+K, canonical mesh only (isolates whether a "
                         "failure is generation or metric_scale registration)")
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()

    import msgpack
    import msgpack_numpy
    import zmq
    msgpack_numpy.patch()

    rgb, mask, depth, K, summary = load_frame(args.run_dir, args.object)
    fg = int((mask > 0).sum())
    print(f"rgb {rgb.shape} mask fg={fg}px "
          f"({100.0 * fg / mask.size:.1f}%) depth valid="
          f"{100.0 * (depth > 0).mean():.1f}%")
    print(f"K fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    if fg == 0:
        sys.exit("mask has no foreground pixels -- wrong --object?")

    # Depth inside the mask is what metric_scale actually registers against.
    # A mask over a hole in the depth image produces a confident-looking scale
    # from a handful of points, so surface the count before trusting it.
    md = depth[(mask > 0) & (depth > 0)]
    if md.size:
        print(f"masked depth: {md.size} px, "
              f"{md.min():.3f}-{md.max():.3f} m, median {np.median(md):.3f} m")
    else:
        print("WARNING: no valid depth under the mask; metric scale will fail")

    req = {
        "op": "generate",
        "rgb": rgb,
        "mask": mask,
        "seed": args.seed,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "output_name": f"{args.object}_offline_{int(time.time())}",
        "return_glb": False,          # path is shared via the mount; don't
                                      # ship megabytes back over ZMQ
    }
    if not args.no_metric:
        req["depth"] = depth
        req["K"] = K

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.addr)

    print(f"-> {args.addr}  generate (timeout {args.timeout:.0f}s) ...")
    t0 = time.time()
    sock.send(msgpack.packb(req, use_bin_type=True))
    try:
        rep = msgpack.unpackb(sock.recv(), raw=False)
    except zmq.error.Again:
        sys.exit(f"no reply after {args.timeout:.0f}s -- check "
                 "`docker compose logs -f trellis2`")
    dt = time.time() - t0

    if not rep.get("ok"):
        sys.exit(f"sidecar error after {dt:.1f}s: {rep.get('error')}")

    V = np.asarray(rep["vertices"])
    F = np.asarray(rep["faces"])
    print(f"\nok in {dt:.1f}s (gen_time {rep.get('gen_time', float('nan')):.1f}s)")
    print(f"  glb        {rep['glb_path']}")
    print(f"  geometry   {len(V)} verts, {len(F)} faces")
    # Canonical output is a unit box by construction; if this is not ~1.0 the
    # GLB is not in the frame the metric step assumes.
    print(f"  canonical AABB extent {np.round(V.max(0) - V.min(0), 3).tolist()}"
          "   (expect ~[1, 1, 1] before scaling)")

    m = rep.get("metric")
    if m is None:
        print("  metric     not requested")
        return 0

    ext = (V.max(0) - V.min(0)) * m["scale"]
    print(f"\n  metric ok={m['ok']}")
    print(f"  scale      {m['scale']:.5f}  (canonical -> meters)")
    print(f"  rmse       {1e3 * m['rmse']:.1f} mm")
    print(f"  t          {np.round(m['t'], 4).tolist()} m")
    print(f"  metric glb {m['glb_path']}")
    print(f"  SCALED EXTENTS {np.round(ext, 4).tolist()} m")
    print("\n  ^ this is the number to sanity-check with calipers. A plausible")
    print("    pose with wrong extents means registration locked onto the")
    print("    wrong similarity, and everything downstream inherits it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
