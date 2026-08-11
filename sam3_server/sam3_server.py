"""SAM 3 ZMQ REP server (open-vocabulary instance segmentation).

Wraps Sam3Processor around build_sam3_image_model(). One model instance,
stateless per request: every "segment" call runs set_image + set_text_prompt.
The image encoder is the expensive part, so "segment_multi" amortizes one
set_image across several prompts (mirrors orient/orient_batch on pointso).

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "segment",
   "rgb": HxWx3 uint8,
   "prompt": "mug handle",          # open-vocab noun phrase
   "threshold": 0.5}                # optional; processor default 0.5
      -> {"ok": True,
          "masks":  NxHxW bool,     # all instances above threshold,
          "boxes":  Nx4 float32,    #   sorted by score desc
          "scores": N float32}      # boxes are xyxy in pixels

  {"cmd": "segment_multi",
   "rgb": HxWx3 uint8,
   "prompts": ["mug", "handle"],
   "threshold": 0.5}
      -> {"ok": True, "results": [{"masks": ..., "boxes": ..., "scores": ...},
                                  ...]}   # one entry per prompt, same order

  {"cmd": "ping"} -> {"ok": True}

N == 0 (empty arrays) is a valid answer: SAM 3's presence token actively
suppresses concepts that aren't in the image, so don't treat it as an error.

Env:
  SAM3_PORT       default 5670
  SAM3_VERSION    "sam3" (default) or "sam3.1" -- HF repo to pull
  SAM3_CKPT       local .pt path override; skips the HF download entirely
  SAM3_THRESHOLD  default confidence threshold (per-request value wins)

The masks are what the pose bridge wants for FoundationPose registration:
pick masks[0] (highest-scoring instance) and publish it on /pose_bridge/mask.
"""

import os
import logging

import numpy as np
import zmq
import msgpack
import msgpack_numpy
from PIL import Image

msgpack_numpy.patch()

# PYTHONPATH=/opt/sam3
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

PORT = int(os.environ.get("SAM3_PORT", "5670"))
DEFAULT_THRESH = float(os.environ.get("SAM3_THRESHOLD", "0.5"))

logging.basicConfig(level=logging.INFO, format="[sam3-server] %(message)s")
log = logging.getLogger(__name__)


def load_processor():
    """Resolve checkpoint from env (local override or gated HF pull)."""
    ckpt = os.environ.get("SAM3_CKPT")
    if ckpt:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM3_CKPT={ckpt} not found in container")
        log.info("ckpt=%s (local)", ckpt)
    else:
        version = os.environ.get("SAM3_VERSION", "sam3")
        log.info("pulling %s from HF (gated -- needs HF_TOKEN + approved "
                 "access to huggingface.co/facebook/%s)", version, version)
        ckpt = download_ckpt_from_hf(version=version)
        log.info("ckpt=%s", ckpt)

    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=DEFAULT_THRESH)


def run_prompt(proc, state, prompt):
    """set_text_prompt + convert to numpy, score-sorted."""
    state = proc.set_text_prompt(prompt=prompt, state=state)
    scores = state["scores"].detach().cpu().numpy().astype(np.float32)
    order = np.argsort(-scores)
    masks = state["masks"].detach().cpu().numpy()  # (N,1,H,W) bool
    masks = masks[:, 0][order]
    boxes = state["boxes"].detach().cpu().numpy().astype(np.float32)[order]
    return {"masks": masks, "boxes": boxes, "scores": scores[order]}


def main():
    log.info("loading SAM 3...")
    proc = load_processor()
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

            elif cmd in ("segment", "segment_multi"):
                proc.confidence_threshold = float(
                    req.get("threshold", DEFAULT_THRESH))
                rgb = np.ascontiguousarray(req["rgb"], np.uint8)
                state = proc.set_image(Image.fromarray(rgb))

                if cmd == "segment":
                    rep = {"ok": True, **run_prompt(proc, state, req["prompt"])}
                else:
                    results = []
                    for p in req["prompts"]:
                        # new text prompt overwrites the old one in-state;
                        # image features are reused across the loop
                        results.append(run_prompt(proc, state, p))
                    rep = {"ok": True, "results": results}

            else:
                rep = {"ok": False, "error": f"unknown cmd {cmd!r}"}

        except Exception as e:
            log.exception("request failed")
            rep = {"ok": False, "error": repr(e)}

        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()