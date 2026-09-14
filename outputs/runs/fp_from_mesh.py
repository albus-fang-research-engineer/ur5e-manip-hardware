#!/usr/bin/env python3
"""Register FoundationPose (pose sidecar, :5667) on a SUPPLIED mesh from a saved
run_scene frame -- the standalone counterpart of any6d_from_mesh.py.

Any6D subclasses FoundationPose (same hypothesis generator, refiner, scorer)
and wraps a scale loop around it. This drives the plain FoundationPose
`register` on a metric mesh -- e.g. Any6D's own final_mesh_<obj>.obj -- so a
yaw picked by Any6D can be compared with the one FoundationPose picks on
its own for the same mesh and frame. Writes <run>/fp_<obj>.json with the
full-precision pose for reproject_check.py --pose-json.

Run inside the Ros2Bridge container (it has the frame, zmq/msgpack, cv2):

    docker exec -it Ros2Bridge bash
    python3 /data/runs/fp_from_mesh.py /data/runs/20260903_203531 \
        --mesh /data/any6d/final_mesh_mug.obj --object mug --release

Then, host side:

    python3 outputs/runs/reproject_check.py outputs/runs/20260903_203531 \
        --mesh any6d_runtime/outputs/final_mesh_mug.obj --object mug \
        --pose-json outputs/runs/20260903_203531/fp_mug.json

Wire format (pose_server): cmd register/track/release; rgb uint8 HxWx3, depth
float32 METRES, K float 3x3, mask uint8; `mesh` is a filename under the
sidecar's /opt/meshes or an absolute path on a mount it can see
(/data/any6d, /data/meshes).
"""
import argparse, json, os, sys, time
import numpy as np


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from any6d_from_mesh import load_frame

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="e.g. /data/runs/20260903_203531")
    ap.add_argument("--mesh", required=True,
                    help="absolute path visible to the pose container, e.g. /data/any6d/final_mesh_mug.obj")
    ap.add_argument("--object", default="mug", help="session key; matches mask_<object>.png")
    ap.add_argument("--addr", default=os.environ.get("POSE_ADDR", "tcp://127.0.0.1:5667"))
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track", type=int, default=0, metavar="N",
                    help="re-send the same frame N times as track requests (latency / code path only)")
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=None, help="pose json path; default <run_dir>/fp_<object>.json")
    ap.add_argument("--rerank", action="store_true",
                    help="ask the sidecar to apply its mask-conditioned re-rank (pose_server/rerank.py) "
                         "and print the record")
    ap.add_argument("--all", action="store_true",
                    help="ask the sidecar for every refined hypothesis + scorer score (return_all) and "
                         "save them to <run_dir>/fp_<object>_hypotheses.npz for rank_hypotheses.py")
    args = ap.parse_args()

    import msgpack, msgpack_numpy, zmq
    from scipy.spatial.transform import Rotation
    msgpack_numpy.patch()

    rgb, mask, depth, K = load_frame(args.run_dir, args.object)
    print(f"rgb {rgb.shape}  mask fg={int((mask > 0).sum())}px  depth valid={100.0*(depth>0).mean():.1f}%")
    print(f"mesh: {args.mesh}")

    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000)); sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.addr)

    def call(payload, label):
        t0 = time.time()
        sock.send(msgpack.packb(payload, use_bin_type=True))
        try:
            rep = msgpack.unpackb(sock.recv(), raw=False)
        except zmq.error.Again:
            sys.exit(f"{label}: no reply after {args.timeout:.0f}s -- `docker compose logs -f pose`")
        if not rep.get("ok"):
            sys.exit(f"{label} failed: {rep.get('error')}")
        return rep, time.time() - t0

    call({"cmd": "ping"}, "ping"); print("sidecar alive\n")
    print(f"-> register '{args.object}' (est_refine_iter={args.est_refine_iter}) ...")
    rep, dt = call({"cmd": "register", "obj": args.object, "rgb": rgb, "depth": depth,
                    "K": K.astype(np.float32), "mask": mask, "mesh": args.mesh,
                    "est_refine_iter": args.est_refine_iter, "return_all": bool(args.all),
                    "rerank": bool(args.rerank)}, "register")
    T = np.asarray(rep["pose"], np.float64).reshape(4, 4)
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\nok in {dt:.1f}s   (scorer saw texture: {rep.get('texture', '?')})")
    if "rerank" in rep:
        rr = rep["rerank"]
        print(f"  rerank: {rr['reason']}  changed={rr['changed']}  rank 0 -> {rr['to_rank']}  "
              f"{rr['rotation_deg']:.0f} deg  expl {rr['expl_from']:.2f} -> {rr['expl_to']:.2f}  "
              f"U {rr['u_px']} px ({100*rr['u_frac']:.1f}% of mask, {rr['u_depth_dropped']} dropped by depth)  "
              f"{rr['n_survivors']} survivors")
    if args.all and "hypotheses" in rep:
        hyp = np.asarray(rep["hypotheses"], np.float64); sc = np.asarray(rep["scores"], np.float64)
        stem = os.path.splitext(os.path.basename(args.out))[0] if args.out else f"fp_{args.object}"
        npz = os.path.join(args.run_dir, f"{stem}_hypotheses.npz")   # follows --out: gray/tex runs keep both
        np.savez(npz, poses=hyp, scores=sc, mesh=args.mesh)
        print(f"  hypotheses {hyp.shape[0]}, scorer {sc.min():.2f}..{sc.max():.2f}, "
              f"top-10 spread {sc[0]-sc[min(9,len(sc)-1)]:.2f} -> {npz}")
    print(f"  t    {np.round(T[:3, 3], 4).tolist()} m")
    print(f"  rpy  {np.round(rpy, 1).tolist()} deg")

    out = args.out or os.path.join(args.run_dir, f"fp_{args.object}.json")
    with open(out, "w") as f:
        json.dump({"object": args.object, "mesh": args.mesh, "cam_T_obj": T.tolist(),
                   "est_refine_iter": args.est_refine_iter, "seconds": dt,
                   "estimator": "foundationpose", "rerank": rep.get("rerank")}, f, indent=2)
    print(f"  pose json  {out}")

    if args.track:
        prev = T
        for i in range(args.track):
            rep, dt = call({"cmd": "track", "obj": args.object, "rgb": rgb, "depth": depth,
                            "K": K.astype(np.float32), "track_refine_iter": args.track_refine_iter},
                           f"track[{i}]")
            P = np.asarray(rep["pose"], np.float64).reshape(4, 4)
            dpos = 1e3 * np.linalg.norm(P[:3, 3] - prev[:3, 3])
            drot = np.degrees(np.linalg.norm(Rotation.from_matrix(P[:3, :3] @ prev[:3, :3].T).as_rotvec()))
            print(f"   [{i}] {1e3*dt:6.0f} ms   d={dpos:5.2f} mm  {drot:5.2f} deg")
            prev = P

    if args.release:
        call({"cmd": "release", "obj": args.object}, "release"); print("\nsession released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
