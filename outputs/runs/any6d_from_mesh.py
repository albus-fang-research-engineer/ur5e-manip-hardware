#!/usr/bin/env python3
"""Drive the Any6D sidecar against a SUPPLIED mesh, from a saved run_scene frame.

Companion to trellis2_from_run.py. Skips the img_to_3d path entirely --
no SAM2, no Zero123++, no InstantMesh -- which is the whole point on a shared
box where InstantMesh's FlexiCubes gather wants 15 GiB in one allocation.

The mesh path is passed straight through: any6d_server does
    trimesh.load(os.path.join(MESH_DIR, req["mesh"]), force="mesh")
and os.path.join returns an absolute second argument unchanged, so an
absolute /data/meshes/... path bypasses MESH_DIR (/opt/meshes) cleanly.
./trellis2_runtime/outputs is mounted into the any6d container at
/data/meshes:ro, so TRELLIS.2 output is already visible there.

Run inside the Ros2Bridge container:

    docker exec -it Ros2Bridge bash
    python3 /data/runs/any6d_from_mesh.py /data/runs/20260822_210159 \
        --mesh /data/meshes/mug_offline_1787436994.glb \
        --object mug \
        --expect-extents 0.1154 0.1027 0.1043 \
        --expect-t -0.0771 0.0133 0.8715

Wire format notes (these differ from the trellis2 sidecar):
  * the key is "cmd", not "op"; port 5672, not 5669
  * depth must be float32 METERS; run_scene saved uint16 millimetres
  * "mesh" and "img_to_3d" are mutually exclusive; supplying neither is an error
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
    depth = depth_mm.astype(np.float32) * 1e-3

    with open(summary_path) as f:
        summary = json.load(f)
    K = np.asarray(summary["frame"]["K"], dtype=np.float64)

    return rgb, mask, depth, K


def call(sock, msgpack, payload, timeout_s, label):
    t0 = time.time()
    sock.send(msgpack.packb(payload, use_bin_type=True))
    import zmq
    try:
        rep = msgpack.unpackb(sock.recv(), raw=False)
    except zmq.error.Again:
        sys.exit(f"{label}: no reply after {timeout_s:.0f}s -- check "
                 "`docker compose logs -f any6d`")
    dt = time.time() - t0
    if not rep.get("ok"):
        sys.exit(f"{label} failed after {dt:.1f}s: {rep.get('error')}")
    return rep, dt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="e.g. /data/runs/20260822_210159")
    ap.add_argument("--mesh", required=True,
                    help="absolute path visible to the any6d container, or a "
                         "bare filename resolved under MESH_DIR (/opt/meshes "
                         "= ./foundationpose_runtime/meshes)")
    ap.add_argument("--object", default="mug",
                    help="session key; also matches mask_<object>.png and names "
                         "the output final_mesh_<object>.obj")
    ap.add_argument("--addr", default=os.environ.get("ANY6D_ADDR",
                                                     "tcp://127.0.0.1:5672"))
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track", type=int, default=0, metavar="N",
                    help="after estimating, re-send the SAME frame N times as "
                         "track requests. Degenerate by construction -- the "
                         "pose should barely move -- so it tests the tracking "
                         "code path and its latency, not tracking accuracy.")
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--expect-extents", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="metres, e.g. TRELLIS's scaled extents, to diff against")
    ap.add_argument("--expect-t", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="metres, e.g. TRELLIS's metric t, to diff against")
    ap.add_argument("--release", action="store_true",
                    help="release the session at the end (frees the estimator)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    import msgpack
    import msgpack_numpy
    import zmq
    from scipy.spatial.transform import Rotation
    msgpack_numpy.patch()

    rgb, mask, depth, K = load_frame(args.run_dir, args.object)
    fg = int((mask > 0).sum())
    print(f"rgb {rgb.shape}  mask fg={fg}px  depth valid={100.0*(depth>0).mean():.1f}%")
    if fg == 0:
        sys.exit("mask has no foreground pixels -- wrong --object?")
    md = depth[(mask > 0) & (depth > 0)]
    if md.size == 0:
        sys.exit("no valid depth under the mask -- registration has nothing to fit")
    print(f"masked depth: {md.size} px, median {np.median(md):.3f} m")
    print(f"mesh: {args.mesh}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.addr)

    rep, dt = call(sock, msgpack, {"cmd": "ping"}, args.timeout, "ping")
    print(f"sidecar alive ({1e3*dt:.0f} ms)\n")

    print(f"-> estimate '{args.object}' (est_refine_iter={args.est_refine_iter}) ...")
    rep, dt = call(sock, msgpack, {
        "cmd": "estimate",
        "obj": args.object,
        "rgb": rgb,
        "depth": depth,
        "K": K.astype(np.float32),
        "mask": mask,
        "mesh": args.mesh,
        "est_refine_iter": args.est_refine_iter,
    }, args.timeout, "estimate")

    pose = np.asarray(rep["pose"], np.float64).reshape(4, 4)
    ext = np.asarray(rep["extents"], np.float64)
    t = pose[:3, 3]
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

    print(f"\nok in {dt:.1f}s")
    print(f"  mesh_path  {rep['mesh_path']}")
    print(f"  extents    {np.round(1e3*ext, 1).tolist()} mm")
    print(f"  t          {np.round(t, 4).tolist()} m")
    print(f"  rpy        {np.round(rpy, 1).tolist()} deg")

    # Persist the pose at full precision so reproject_check.py (and the
    # render-asset / FoundationPose consistency checks) read it instead of
    # retyping a 0.1-degree print.
    pose_json = os.path.join(args.run_dir, f"any6d_{args.object}.json")
    with open(pose_json, "w") as f:
        json.dump({"object": args.object, "mesh": args.mesh,
                   "final_mesh": rep["mesh_path"], "cam_T_obj": pose.tolist(),
                   "extents": ext.tolist(), "est_refine_iter": args.est_refine_iter,
                   "seconds": dt}, f, indent=2)
    print(f"  pose json  {pose_json}")

    if args.expect_extents is not None:
        exp = np.asarray(args.expect_extents, np.float64)
        d = ext - exp
        # Extents are AXIS-ORDERED, so a large per-axis delta with a small
        # sorted delta means the mesh came out in a permuted orientation --
        # not a scale failure. Report both.
        ds = np.sort(ext) - np.sort(exp)
        print(f"\n  extents delta      {np.round(1e3*d, 1).tolist()} mm "
              f"(ratio {np.round(ext/exp, 3).tolist()})")
        print(f"  sorted-axis delta  {np.round(1e3*ds, 1).tolist()} mm")
        if np.abs(ds).max() > 0.01:
            print("  ^ >10mm even after sorting: the two estimators disagree on "
                  "SIZE, not just axis order. Suspect the metric fit.")

    if args.expect_t is not None:
        exp_t = np.asarray(args.expect_t, np.float64)
        # Approximate: TRELLIS's sim t maps the canonical mesh centroid into the
        # camera frame, Any6D's is the mesh ORIGIN. They agree only to however
        # well-centred the mesh is -- treat a couple of cm as agreement.
        print(f"  t delta            {np.round(1e3*(t-exp_t), 1).tolist()} mm "
              f"(|d| = {1e3*np.linalg.norm(t-exp_t):.1f} mm)")

    if args.track:
        print(f"\n-> track x{args.track} (same frame, so pose should be static)")
        prev, dts = pose, []
        for i in range(args.track):
            rep, dt = call(sock, msgpack, {
                "cmd": "track",
                "obj": args.object,
                "rgb": rgb,
                "depth": depth,
                "K": K.astype(np.float32),
                "track_refine_iter": args.track_refine_iter,
            }, args.timeout, f"track[{i}]")
            p = np.asarray(rep["pose"], np.float64).reshape(4, 4)
            dpos = 1e3 * np.linalg.norm(p[:3, 3] - prev[:3, 3])
            drot = np.degrees(np.linalg.norm(
                Rotation.from_matrix(p[:3, :3] @ prev[:3, :3].T).as_rotvec()))
            print(f"   [{i}] {1e3*dt:6.0f} ms   d={dpos:5.2f} mm  {drot:5.2f} deg")
            prev, _ = p, dts.append(dt)
        print(f"   mean {1e3*np.mean(dts):.0f} ms -> ~{1.0/np.mean(dts):.1f} Hz "
              "ceiling (bag gives you ~12 Hz of synced pairs regardless)")

    if args.release:
        call(sock, msgpack, {"cmd": "release", "obj": args.object},
             args.timeout, "release")
        print("\nsession released")

    print("\nMeasure the real mug. Extents is the number that catches a "
          "registration\nthat locked onto the wrong similarity -- a plausible "
          "pose with wrong\nextents is the failure mode that looks like success.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
