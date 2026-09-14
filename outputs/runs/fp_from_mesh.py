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
    ap.add_argument("--oriany", action="store_true",
                    help="after --rerank, have Orient Anything decide among the gate survivors: it judges the "
                         "real masked crop and a textured render of each survivor (same framing), the survivor "
                         "whose full rotation agrees best is selected via the sidecar's `select`. Writes "
                         "<run>/oriany_<object>.png (annotated contact sheet) and records the table in the pose json.")
    ap.add_argument("--oriany-addr", default=os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673"))
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
                    "est_refine_iter": args.est_refine_iter, "return_all": bool(args.all or args.oriany),
                    "rerank": bool(args.rerank or args.oriany),
                    "survivor_crops": bool(args.oriany)}, "register")
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

    oriany_table = None
    if args.oriany and rep.get("rerank", {}).get("survivors") and "crops" in rep:
        from PIL import Image, ImageDraw
        survivors = list(rep["rerank"]["survivors"]); crops = np.asarray(rep["crops"]); y0, y1, x0, x1 = rep["crop_box"]
        real_crop = np.where((mask > 0)[..., None], rgb, 255).astype(np.uint8)[y0:y1, x0:x1]
        osock = zmq.Context().socket(zmq.REQ); osock.setsockopt(zmq.RCVTIMEO, 120000); osock.setsockopt(zmq.LINGER, 0)
        osock.connect(args.oriany_addr)

        def ocall(payload):
            osock.send(msgpack.packb(payload, use_bin_type=True))
            r = msgpack.unpackb(osock.recv(), raw=False)
            if not r.get("ok"):
                sys.exit(f"oriany: {r.get('error')}")
            return r

        def geo(Ra, Rb):
            return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))

        def yaw_about(up, fa, fb):
            """signed angle between two fronts projected onto the plane normal to `up`"""
            up = up / np.linalg.norm(up)
            pa = fa - up * (fa @ up); pb = fb - up * (fb @ up)
            pa /= max(np.linalg.norm(pa), 1e-9); pb /= max(np.linalg.norm(pb), 1e-9)
            ang = np.degrees(np.arctan2(np.cross(pa, pb) @ up, pa @ pb))
            return float(ang)

        def fold(deg, alpha):
            """|yaw| modulo the object's symmetry order: Orient Anything's front on an
            alpha-fold object is defined only up to 360/alpha, so a disagreement of
            360/alpha is no disagreement"""
            period = 360.0 / max(int(alpha), 1)
            d = abs(deg) % period
            return float(min(d, period - d))

        def annotate(img, o, title, line):
            im = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(im)
            h, w = img.shape[:2]; c = np.array([w / 2, h / 2]); L = 0.25 * min(h, w)
            for vec, col in ((np.asarray(o["front_cam"], float), (0, 200, 0)), (np.asarray(o["up_cam"], float), (40, 90, 255))):
                tip = c + L * vec[:2]; d.line([tuple(c), tuple(tip)], fill=col, width=3)
                d.ellipse([tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4], fill=col)
            d.text((4, 2), title, fill=(0, 0, 0)); d.text((4, h - 12), line, fill=(0, 0, 0))
            return im

        ocall({"cmd": "ping"})
        o_real = ocall({"cmd": "orient", "image": real_crop, "remove_bkg": False})
        R_real = np.asarray(o_real["R_cam"], float); up_real = np.asarray(o_real["up_cam"], float)
        f_real = np.asarray(o_real["front_cam"], float); alpha_real = int(o_real["alpha"])
        panels = [annotate(real_crop, o_real, "real", f"alpha {alpha_real}")]
        rows = []
        hyp_poses = np.asarray(rep["hypotheses"], np.float64) if "hypotheses" in rep else None
        for k, rk in enumerate(survivors):
            o = ocall({"cmd": "orient", "image": crops[k], "remove_bkg": False})
            R_o = np.asarray(o["R_cam"], float); up_o = np.asarray(o["up_cam"], float); f_o = np.asarray(o["front_cam"], float)
            g = geo(R_real, R_o)
            upa = float(np.degrees(np.arccos(np.clip(up_real @ up_o, -1, 1))))
            yaw = yaw_about(up_real, f_real, f_o)                 # real crop's up for BOTH sides
            confirmable = int(o["alpha"]) != 0 and int(o["alpha"]) == alpha_real
            rows.append(dict(rank=int(rk), geo=g, up=upa, yaw=yaw, yaw_folded=fold(yaw, alpha_real),
                             alpha=int(o["alpha"]), confirmable=bool(confirmable)))
            panels.append(annotate(crops[k], o, f"rank {rk}", f"yaw {yaw:+.0f} rot {g:.0f} up {upa:.0f} a{o['alpha']}"))

        # ---- decision: rank by folded yaw-about-up among confirmable survivors ----
        sidecar_rank = int(rep["rerank"]["to_rank"])
        cand = [r for r in rows if r["confirmable"]]
        used, best, note = "oriany", None, ""
        if alpha_real == 0:
            used, note = "scorer", "real crop alpha 0: Orient Anything has no confident front"
        elif not cand:
            used, note = "scorer", "no survivor render is confirmable (alpha 0 or alpha != real)"
        else:
            cand.sort(key=lambda r: r["yaw_folded"])
            best = cand[0]
            if len(cand) > 1 and hyp_poses is not None:
                second = cand[1]
                Ra, Rb = hyp_poses[best["rank"]][:3, :3], hyp_poses[second["rank"]][:3, :3]
                apart = geo(Ra, Rb)
                if abs(best["yaw_folded"] - second["yaw_folded"]) < 10.0 and apart > 30.0:
                    used, note = "scorer", (f"oa_ambiguous: ranks {best['rank']} and {second['rank']} read "
                                            f"{best['yaw_folded']:.0f} vs {second['yaw_folded']:.0f} deg but are {apart:.0f} deg apart "
                                            f"(alpha {alpha_real} fold)")
                    best = None
        print(f"\n  oriany: real crop alpha {alpha_real}; {len(rows)} survivors "
              f"({len(cand)} confirmable); decider: {used}" + (f" -- {note}" if note else ""))
        print("    rank   yaw-about-up  folded  full-rot   up  alpha")
        for r in sorted(rows, key=lambda r: r["yaw_folded"]):
            flag = "   <- chosen" if best is not None and r is best else ("   (not confirmable)" if not r["confirmable"] else "")
            print(f"    {r['rank']:4d}   {r['yaw']:+8.1f}    {r['yaw_folded']:6.1f}   {r['geo']:6.1f}  {r['up']:5.1f}   {r['alpha']}{flag}")
        chosen_rank = int(best["rank"]) if best is not None else sidecar_rank
        if chosen_rank != sidecar_rank:
            sel = call({"cmd": "select", "obj": args.object, "rank": chosen_rank}, "select")
            T = np.asarray(sel["pose"], np.float64).reshape(4, 4)
            rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            print(f"  selected rank {chosen_rank} via {used} (sidecar's rerank pick was rank {rep['rerank']['to_rank']})")
            print(f"  t    {np.round(T[:3, 3], 4).tolist()} m\n  rpy  {np.round(rpy, 1).tolist()} deg")
        else:
            print(f"  oriany agrees with the sidecar's rerank pick (rank {chosen_rank})")
        Wp = max(p.size[0] for p in panels); Hp = max(p.size[1] for p in panels)
        sheet = Image.new("RGB", (len(panels) * (Wp + 6), Hp), "white")
        for i, pnl in enumerate(panels):
            sheet.paste(pnl, (i * (Wp + 6), 0))
        sheet_path = os.path.join(args.run_dir, f"oriany_{args.object}.png"); sheet.save(sheet_path)
        print(f"  wrote {sheet_path}")
        oriany_table = {"real_alpha": alpha_real, "rows": rows, "chosen_rank": chosen_rank, "decider": used, "note": note}
    elif args.oriany:
        print("  oriany: no survivors returned (rerank declined) -- nothing to decide")

    out = args.out or os.path.join(args.run_dir, f"fp_{args.object}.json")
    with open(out, "w") as f:
        json.dump({"object": args.object, "mesh": args.mesh, "cam_T_obj": T.tolist(),
                   "est_refine_iter": args.est_refine_iter, "seconds": dt,
                   "estimator": "foundationpose", "rerank": rep.get("rerank"), "oriany": oriany_table}, f, indent=2)
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
