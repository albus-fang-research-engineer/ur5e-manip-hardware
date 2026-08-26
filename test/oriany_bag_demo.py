"""One bag frame -> Orient Anything V2 sidecar -> axis overlay PNG.

No ROS install needed: reads the mcap directly with mcap-ros2-support.
Crop comes from the sam3 sidecar (prompt "mug") if it answers, else --bbox.

  python oriany_bag_demo.py ros2bags/mug/mug_0.mcap --out outputs/oriany
  python oriany_bag_demo.py ros2bags/mug/mug_0.mcap --frame 40 --bbox 300 200 420 340

Writes <out>/frame.png, crop.png, crop_masked.png (if sam3), overlay.png
and prints azimuth/elevation/rotation/alpha for both background paths.

Overlay draws up_cam (blue) / front_cam (red) as served. If front points
away from the handle on a frame where alpha == 1, restart oriany with
ORIANY_AZ_SIGN=-1 and pin it in compose.
"""
import argparse, json, os, sys
import numpy as np
import zmq, msgpack, msgpack_numpy
from PIL import Image, ImageDraw
from mcap_ros2.reader import read_ros2_messages

msgpack_numpy.patch()

RGB_T = "/camera/camera/color/image_raw"
DEP_T = "/camera/camera/aligned_depth_to_color/image_raw"
INFO_T = "/camera/camera/color/camera_info"


def zmq_call(addr, payload, timeout_ms=120_000):
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms); s.setsockopt(zmq.LINGER, 0)
    s.connect(addr)
    s.send(msgpack.packb(payload, use_bin_type=True))
    try:
        return msgpack.unpackb(s.recv(), raw=False)
    except zmq.error.Again:
        return None


def decode_image(m):
    buf = np.frombuffer(bytes(m.data), np.uint8)
    h, w, enc = m.height, m.width, m.encoding
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, m.step)[:, : w * 3].reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "bgr8" else img.copy(), enc
    if enc == "16UC1":
        d = buf.reshape(h, m.step)[:, : w * 2].copy().view(np.uint16).reshape(h, w)
        return d.astype(np.float32) / 1000.0, enc
    if enc == "32FC1":
        return buf.reshape(h, m.step)[:, : w * 4].copy().view(np.float32).reshape(h, w), enc
    raise ValueError(f"unhandled encoding {enc}")


def read_frame(path, frame_idx):
    rgbs, deps, info = [], [], None
    for r in read_ros2_messages(path, topics=[RGB_T, DEP_T, INFO_T]):
        t = r.channel.topic
        if t == INFO_T and info is None:
            info = r.ros_msg
        elif t == RGB_T:
            rgbs.append((r.log_time_ns, r.ros_msg))
        elif t == DEP_T:
            deps.append((r.log_time_ns, r.ros_msg))
    if not rgbs:
        sys.exit(f"no messages on {RGB_T}")
    t_rgb, m_rgb = rgbs[min(frame_idx, len(rgbs) - 1)]
    rgb, enc = decode_image(m_rgb)
    depth, denc = (None, None)
    if deps:
        _, m_dep = min(deps, key=lambda x: abs(x[0] - t_rgb))
        depth, denc = decode_image(m_dep)
    K = np.asarray(info.k, np.float64).reshape(3, 3) if info is not None else None
    print(f"rgb {rgb.shape} {enc} | depth {None if depth is None else depth.shape} {denc} "
          f"| {len(rgbs)} color frames, using #{frame_idx}")
    return rgb, depth, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--bbox", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--prompt", default="mug")
    ap.add_argument("--oriany", default="tcp://127.0.0.1:5673")
    ap.add_argument("--sam3", default="tcp://127.0.0.1:5670")
    ap.add_argument("--out", default="outputs/oriany")
    ap.add_argument("--pad", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rgb, depth, K = read_frame(a.mcap, a.frame)
    Image.fromarray(rgb).save(f"{a.out}/frame.png")

    # ---- crop: sam3 if reachable, else bbox -------------------------------
    mask = None
    rep = zmq_call(a.sam3, {"cmd": "ping"}, timeout_ms=3_000)
    if rep and rep.get("ok"):
        rep = zmq_call(a.sam3, {"cmd": "segment", "rgb": rgb, "prompt": a.prompt})
        masks = np.asarray(rep["masks"]) if rep and rep.get("ok") else np.zeros((0,))
        if masks.shape[0]:
            mask = masks[0].astype(bool)
            ys, xs = np.nonzero(mask)
            bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            print(f"sam3 '{a.prompt}': {masks.shape[0]} inst, score {float(np.asarray(rep['scores'])[0]):.2f}")
        else:
            print(f"sam3 returned nothing for '{a.prompt}'")
    else:
        print(f"sam3 unreachable at {a.sam3}; falling back to --bbox")
    if mask is None:
        if a.bbox is None:
            sys.exit("no sam3 mask; pass --bbox X0 Y0 X1 Y1 (pixels)")
        bbox = a.bbox
    h, w = rgb.shape[:2]
    x0, y0 = max(0, bbox[0] - a.pad), max(0, bbox[1] - a.pad)
    x1, y1 = min(w, bbox[2] + a.pad), min(h, bbox[3] + a.pad)
    crop = rgb[y0:y1, x0:x1]
    Image.fromarray(crop).save(f"{a.out}/crop.png")

    # ---- orient ------------------------------------------------------------
    assert zmq_call(a.oriany, {"cmd": "ping"}, 5_000), "oriany not answering"
    results = {}
    results["rembg"] = zmq_call(a.oriany, {"cmd": "orient", "image": crop, "remove_bkg": True})
    if mask is not None:
        masked = crop.copy(); masked[~mask[y0:y1, x0:x1]] = 255
        Image.fromarray(masked).save(f"{a.out}/crop_masked.png")
        results["sam3mask"] = zmq_call(a.oriany, {"cmd": "orient", "image": masked, "remove_bkg": False})
    for k, r in results.items():
        if not r or not r.get("ok"):
            print(f"{k}: FAILED {r and r.get('error')}"); continue
        up, fr = np.asarray(r["up_cam"]), np.asarray(r["front_cam"])
        print(f"{k:9s} az={r['azimuth']:7.1f} el={r['elevation']:6.1f} ro={r['rotation']:7.1f} "
              f"alpha={r['alpha']} | up_cam={np.round(up, 2)} front_cam={np.round(fr, 2)} "
              f"|up.front|={abs(up @ fr):.1e}")
    json.dump({k: {kk: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                   for kk, v in r.items()} for k, r in results.items() if r},
              open(f"{a.out}/result.json", "w"), indent=1)

    # ---- overlay: up/front_cam at the object's 3D centroid ------------------
    r = results.get("sam3mask") or results["rembg"]
    if not (r and r.get("ok") and K is not None and depth is not None):
        print("skipping overlay (need ok result + K + depth)"); return
    sel = mask if mask is not None else np.zeros((h, w), bool)
    if mask is None:
        sel[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True
    z = depth[sel]; z = z[(z > 0.05) & (z < 3.0)]
    if z.size == 0:
        print("no valid depth under mask; skipping overlay"); return
    zc = float(np.median(z))
    ys, xs = np.nonzero(sel)
    uc, vc = xs.mean(), ys.mean()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([(uc - cx) * zc / fx, (vc - cy) * zc / fy, zc])
    def proj(p):
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, x1, y1], outline=(255, 255, 0))
    o = proj(P)
    for key, col, name in [("up_cam", (0, 128, 255), "up"), ("front_cam", (255, 0, 0), "front")]:
        e = proj(P + 0.06 * np.asarray(r[key], np.float64))
        d.line([o, e], fill=col, width=3); d.text(e, name, fill=col)
    d.text((10, 10), f"az={r['azimuth']:.0f} el={r['elevation']:.0f} ro={r['rotation']:.0f} alpha={r['alpha']}",
           fill=(255, 255, 255))
    im.save(f"{a.out}/overlay.png")
    print(f"wrote {a.out}/overlay.png  (alpha != 1 -> front is not a committed "
          f"prediction; judge only 'up' on this frame)")


if __name__ == "__main__":
    main()
