"""Orient Anything V2 ZMQ REP server.

Thin wrapper around the upstream inference path (utils.app_utils), same
recipe as app.py: VGGT_OriAny_Ref(out_dim=900, nopretrain=True) + the full
demo checkpoint (VGGT() builds with random weights; the 5GB state dict
overwrites everything, so nothing else is downloaded at model init).

  ORIANY_PORT   listen port                    (default 5673)
  ORIANY_CKPT   local .pt path; if unset/missing, hf_hub_download
                Viglong/OriAnyV2_ckpt :: demo_ckpts/rotmod_realrotaug_best.pt
                into HF_HOME (mounted cache)

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "orient",
   "image": HxWx3 uint8,          # RGB object crop
   "remove_bkg": True}            # optional, default True (demo behavior);
                                  # pass False for SAM3-masked crops that are
                                  # already background-free
      -> {"ok": True,
          "azimuth": float,       # deg, [0, 360)
          "elevation": float,     # deg, [-90, 90]
          "rotation": float,      # deg, [-180, 180], in-plane
          "alpha": int,           # symmetry order of the front-face
                                  # distribution: 1 / 2 / 4 discrete folds,
                                  # 0 = no confident single front (continuous
                                  # symmetry or flat distribution)
          "up_cam":      3 float32,  # object canonical up, OpenCV camera
          "front_cam":   3 float32,  #   frame (x right, y down, z fwd)
          "lateral_cam": 3 float32,  # up x front
          "R_cam":     3x3 float32,  # columns (front, lateral, up) in the
                                     # OpenCV camera frame: a proper
                                     # camera-frame rotation of the object's
                                     # canonical (x=front, y=lateral, z=up)
          "R_blender": 3x3 float32}  # upstream azi_ele_rot_to_Obj_Rmatrix_batch:
                                     # Rx(rot)Ry(ele)Rz(-azi), the gizmo
                                     # rotation in the demo's z-up Blender
                                     # world. Kept for parity with app.py.

  {"cmd": "orient_rel",
   "image_ref": HxWx3 uint8,
   "image_tgt": HxWx3 uint8,
   "remove_bkg": True}
      -> {"ok": True, ...all "orient" fields for the ref view...,
          "rel_azimuth": float, "rel_elevation": float,
          "rel_rotation": float}                       # tgt w.r.t. ref, deg

  {"cmd": "ping"} -> {"ok": True}

Angle semantics (Orient Anything): elevation = camera height above the
object's horizontal plane, azimuth = camera bearing around the object's up
axis measured from its front face, rotation = in-plane roll.

Camera-frame axes are NOT hand-rolled from the angles. The only place
upstream commits to a handedness is the demo overlay: utils/axis_renderer.py
sets the gizmo's Blender rotation to R_blender (ZYX Euler = (rot, ele,
-azi)) and renders it with the camera in assets/axis_render.blend, which
sits at world (100, 0, 0) with rotation_euler XYZ (90deg, 0, 90deg): it
looks down -X with Z up, so image-right = +Y_w and image-up = +Z_w. The
gizmo at identity has front = +X_w (facing that camera at az = 0) and
up = +Z_w. Re-expressing those world axes in the OpenCV frame of the same
camera is the fixed change of basis BLENDER_TO_CAM below; up_cam / front_cam
are the columns of BLENDER_TO_CAM @ R_blender. A previous hand-rolled
mapping here had the image-x sign of both azimuth and roll mirrored relative
to that gizmo (front drawn on the wrong side of the object for az != 0),
which is why it is gone and why there is no sign knob.

Angles are the model's native discretized outputs (1 deg bins); treat them
as a coarse semantic init for refine_frame.py, not as metric ground truth --
the geometry-based refinement owns the sub-degree regime.

The alpha head is the reason V2 is here at all: alpha != 1 tells the caller
the front axis is only defined up to a symmetry group, i.e. azimuth for that
object should widen (or drop) the corresponding TSR rotational bound instead
of trusting a single sampled mode.
"""

import os
import logging

import numpy as np
import torch
import zmq
import msgpack
import msgpack_numpy
from PIL import Image

msgpack_numpy.patch()

# PYTHONPATH=/opt/OrientAnythingV2
from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import (
    inf_single_case,
    remove_background,
    resize_foreground,
)
from utils.utils import azi_ele_rot_to_Obj_Rmatrix_batch

PORT = int(os.environ.get("ORIANY_PORT", "5673"))

# Blender world of assets/axis_render.blend -> OpenCV camera frame of its
# camera (at +X_w looking -X_w, Z_w up): x_cam = +Y_w, y_cam = -Z_w,
# z_cam = -X_w. det = +1.
BLENDER_TO_CAM = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0]], np.float32)

HF_REPO = "Viglong/OriAnyV2_ckpt"
HF_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"

logging.basicConfig(level=logging.INFO, format="[oriany-server] %(message)s")
log = logging.getLogger(__name__)

# rembg session is created lazily and reused; app.py's background_preprocess
# builds a fresh session (u2net load) per call, which is fine for a demo and
# wrong for a server.
_REMBG_SESSION = None


def load_model():
    ckpt = os.environ.get("ORIANY_CKPT", "")
    if not (ckpt and os.path.isfile(ckpt)):
        from huggingface_hub import hf_hub_download
        log.info("downloading %s :: %s (5GB on first pull)", HF_REPO, HF_FILE)
        ckpt = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE,
                               repo_type="model")

    # sm86 supports bf16; keep app.py's capability check anyway
    dtype = (torch.bfloat16
             if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model.to("cuda")


def to_pil(arr, remove_bkg):
    img = Image.fromarray(np.asarray(arr, np.uint8)).convert("RGB")
    if remove_bkg:
        global _REMBG_SESSION
        if _REMBG_SESSION is None:
            import rembg
            _REMBG_SESSION = rembg.new_session()
        img = remove_background(img, _REMBG_SESSION)
        img = resize_foreground(img, 0.85)  # demo's fg ratio
    return img


def angles_to_R(az, el, ro):
    R = azi_ele_rot_to_Obj_Rmatrix_batch(
        torch.tensor([az]), torch.tensor([el]), torch.tensor([ro]))[0]
    return R.cpu().numpy().astype(np.float32)


def blender_to_cam_axes(R_blender):
    """R_blender (gizmo rotation, z-up Blender world) -> (up, front, lateral,
    R_cam) in the OpenCV camera frame. Gizmo canonical: x = front, y =
    lateral, z = up; R_cam keeps that column order."""
    R_cam = (BLENDER_TO_CAM @ np.asarray(R_blender, np.float32)).astype(np.float32)
    front, lat, up = R_cam[:, 0], R_cam[:, 1], R_cam[:, 2]
    return up, front, lat, R_cam


def ref_fields(ans):
    az = float(ans["ref_az_pred"])
    el = float(ans["ref_el_pred"])
    ro = float(ans["ref_ro_pred"])
    R_blender = angles_to_R(az, el, ro)
    up, front, lat, R_cam = blender_to_cam_axes(R_blender)
    return {
        "azimuth": az, "elevation": el, "rotation": ro,
        "alpha": int(ans["ref_alpha_pred"]),
        "up_cam": up, "front_cam": front, "lateral_cam": lat,
        "R_cam": R_cam, "R_blender": R_blender,
    }


def main():
    log.info("loading Orient Anything V2...")
    model = load_model()
    log.info("model ready, listening on :%d", PORT)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{PORT}")

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            cmd = req["cmd"]
            if cmd == "ping":
                rep = {"ok": True}

            elif cmd == "orient":
                rm = bool(req.get("remove_bkg", True))
                pil = to_pil(req["image"], rm)
                with torch.no_grad():
                    ans = inf_single_case(model, pil, None)
                rep = {"ok": True, **ref_fields(ans)}

            elif cmd == "orient_rel":
                rm = bool(req.get("remove_bkg", True))
                pil_ref = to_pil(req["image_ref"], rm)
                pil_tgt = to_pil(req["image_tgt"], rm)
                with torch.no_grad():
                    ans = inf_single_case(model, pil_ref, pil_tgt)
                raz = float(ans["rel_az_pred"])
                rel = float(ans["rel_el_pred"])
                rro = float(ans["rel_ro_pred"])
                rep = {"ok": True, **ref_fields(ans),
                       "rel_azimuth": raz, "rel_elevation": rel,
                       "rel_rotation": rro}

            else:
                rep = {"ok": False, "error": f"unknown cmd {cmd!r}"}

        except Exception as e:
            log.exception("request failed")
            rep = {"ok": False, "error": repr(e)}

        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()