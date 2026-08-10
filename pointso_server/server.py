"""PointSO ZMQ REP server.

Thin wrapper around SoFar's own serving helpers (serve/pointso.py), which
expose get_model() and pred_orientation(model, pcd, instruction). The cfg /
checkpoint pair is selected via env vars (overriding the globals at the top
of serve/pointso.py before get_model() reads them):

  POINTSO_CKPT  path to .pth   (default: repo's small.pth)
  POINTSO_CFG   path to yaml   (default: derived from ckpt name --
                                base* -> base.yaml, else small.yaml)

Wire protocol (msgpack + msgpack_numpy, REQ/REP):

  {"cmd": "orient",
   "pcd": Nx6 float32,          # xyz + rgb (rgb in [0,1]), object-centric
   "instruction": "handle"}     # open-vocab semantic direction query
      -> {"ok": True, "direction": 3-vector float32}   # unit vector, obj frame

  {"cmd": "orient_batch",
   "pcd": Nx6 float32,
   "instructions": ["top", "handle", "pouring water"]}
      -> {"ok": True, "directions": Mx3 float32}

  {"cmd": "ping"} -> {"ok": True}

Point cloud convention follows SoFar's demo: sampled object points from the
segmented instance, xyz typically normalized to the unit sphere by the model
pipeline internally -- feed raw metric object points and let pred_orientation
handle preprocessing (it asserts shape[1] == 6).
"""

import os
import logging

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

# PYTHONPATH=/opt/SoFar
import serve.pointso as pointso
from serve.pointso import pred_orientation

PORT = int(os.environ.get("POINTSO_PORT", "5668"))

logging.basicConfig(level=logging.INFO, format="[pointso-server] %(message)s")
log = logging.getLogger(__name__)


def load_model():
    """Resolve cfg/ckpt from env, patch serve.pointso globals, load."""
    ckpt = os.environ.get("POINTSO_CKPT", pointso.CHECKPOINT_PATH)
    variant = "base" if os.path.basename(ckpt).startswith("base") else "small"
    cfg = os.environ.get("POINTSO_CFG", f"orientation/cfgs/train/{variant}.yaml")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"{ckpt} not found -- is ./pointso_runtime/checkpoints mounted? "
            "wget -c https://huggingface.co/qizekun/PointSO/resolve/main/"
            f"{os.path.basename(ckpt)} -P pointso_runtime/checkpoints/"
        )

    pointso.CHECKPOINT_PATH = ckpt
    pointso.CFG_PATH = cfg
    log.info("cfg=%s ckpt=%s", cfg, ckpt)
    return pointso.get_model()


def main():
    log.info("loading PointSO...")
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
                pcd = np.asarray(req["pcd"], np.float32)
                # pred_orientation takes a LIST of instructions (n =
                # len(instruction)); a bare string is iterated per-character.
                d = pred_orientation(model, pcd, [req["instruction"]])[0]
                rep = {"ok": True, "direction": np.asarray(d, np.float32)}

            elif cmd == "orient_batch":
                pcd = np.asarray(req["pcd"], np.float32)
                # One call per instruction on purpose: upstream
                # pred_orientation with n>1 instructions interleaves
                # [i0,i1]*12 against vote-major point clouds, so the
                # 12-vote mean averages ACROSS instructions. n=1 is the
                # only pairing that is unambiguously correct.
                ds = [np.asarray(pred_orientation(model, pcd, [ins])[0],
                                 np.float32)
                      for ins in req["instructions"]]
                rep = {"ok": True, "directions": np.stack(ds)}

            else:
                rep = {"ok": False, "error": f"unknown cmd {cmd!r}"}

        except Exception as e:
            log.exception("request failed")
            rep = {"ok": False, "error": repr(e)}

        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()