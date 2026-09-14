#!/usr/bin/env python3
"""Can Orient Anything tell a correct registration from a wrong one?

Two-sided test: the same model judges the REAL masked crop and a RENDER of
the mesh at each candidate pose, so whatever "front" convention it uses
cancels -- for the correct pose the rendered mug faces the same way as the
real one, for a wrong-yaw pose the fronts disagree by the yaw error. Uses
the sidecar's own `orient_rel` (target w.r.t. reference azimuth) with the
real crop as reference, plus `orient` on each image for alpha (symmetry
order: 0 = no confident front) and elevation.

Run inside Ros2Bridge (it reaches the oriany sidecar on :5673):

    docker exec -it Ros2Bridge python3 /data/runs/oriany_check.py /data/runs/oriany_test

where the directory holds real_crop.png and render_rank<N>.png (same crop
framing, white background). Expect: |rel_azimuth| small for the correct
rank, ~150 deg for the handle-behind pick, and alpha != 0 on the real crop
if the prior is usable on this object at all.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("image_dir")
ap.add_argument("--real", default="real_crop.png")
ap.add_argument("--addr", default=os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673"))
ap.add_argument("--remove-bkg", action="store_true", help="let the sidecar run rembg (crops here are already white-background)")
a = ap.parse_args()

import zmq, msgpack, msgpack_numpy
msgpack_numpy.patch()
sock = zmq.Context().socket(zmq.REQ); sock.setsockopt(zmq.RCVTIMEO, 120000); sock.connect(a.addr)

def call(payload):
    sock.send(msgpack.packb(payload, use_bin_type=True))
    rep = msgpack.unpackb(sock.recv(), raw=False)
    if not rep.get("ok"):
        sys.exit(f"sidecar error: {rep.get('error')}")
    return rep

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.uint8)

def geo_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))

def yaw_about(up, fa, fb):
    """angle between two front vectors projected onto the plane normal to `up`"""
    up = up / np.linalg.norm(up)
    pa = fa - up * (fa @ up); pb = fb - up * (fb @ up)
    pa /= max(np.linalg.norm(pa), 1e-9); pb /= max(np.linalg.norm(pb), 1e-9)
    return float(np.degrees(np.arccos(np.clip(pa @ pb, -1, 1))))

call({"cmd": "ping"})
real = load(os.path.join(a.image_dir, a.real))
r = call({"cmd": "orient", "image": real, "remove_bkg": a.remove_bkg})
R_real = np.asarray(r["R_cam"], float); up_real = np.asarray(r["up_cam"], float); f_real = np.asarray(r["front_cam"], float)
print(f"real crop : azimuth {r['azimuth']:6.1f}  elevation {r['elevation']:6.1f}  rotation {r['rotation']:6.1f}  "
      f"alpha {r['alpha']}   up_cam {np.round(up_real, 2).tolist()}   "
      f"{'<- alpha 0: no confident front on the real object' if r['alpha'] == 0 else ''}")
files = sorted(glob.glob(os.path.join(a.image_dir, "render_rank*.png")), key=lambda p: int(p.split("rank")[-1].split(".")[0]))
print("\nrender      alpha | full rotation  up-vector  yaw-about-up  (deg, render vs real, from R_cam) | orient_rel az / el")
for p in files:
    img = load(p)
    o = call({"cmd": "orient", "image": img, "remove_bkg": a.remove_bkg})
    R_o = np.asarray(o["R_cam"], float); up_o = np.asarray(o["up_cam"], float); f_o = np.asarray(o["front_cam"], float)
    rel = call({"cmd": "orient_rel", "image_ref": real, "image_tgt": img, "remove_bkg": a.remove_bkg})
    raz = ((rel["rel_azimuth"] + 180) % 360) - 180
    name = os.path.basename(p).replace(".png", "")
    up_ang = float(np.degrees(np.arccos(np.clip(up_real @ up_o / (np.linalg.norm(up_real) * np.linalg.norm(up_o)), -1, 1))))
    print(f"{name:11s}   {o['alpha']}   |   {geo_deg(R_real, R_o):6.1f}       {up_ang:6.1f}      {yaw_about(up_real, f_real, f_o):6.1f}"
          f"                                    | {raz:+7.1f} / {rel['rel_elevation']:+6.1f}")
print("\nread: correct pose -> small full-rotation angle (yaw-about-up near 0, up-vector near 0); handle-behind pick -> "
      "yaw ~150 with up near 0; inverted body -> up-vector ~180. If the real crop reads alpha 0 or the angles do not "
      "separate the ranks, Orient Anything stays a tie-breaker with a confidence gate; if they do, it becomes the decider "
      "over the gated survivors.")
