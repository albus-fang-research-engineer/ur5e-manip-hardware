"""AnyGrasp ZMQ sidecar (hardware wing).

Wire format is pickle, not msgpack like the other sidecars in this repo --
this is the contract manip_sim/perception/grasp_client.py already speaks, so
the sim wing points ANYGRASP_ADDR at this server and nothing else changes.

  in : {"points": Nx3 f32, "colors": Nx3 f32, "lims": [6] | "region_steering",
        "dense_grasp", "collision_detection", "approach_steering",
        "approach_thresh"}
  out: {"ok": True, "n": k, "translations", "rotations", "widths", "depths",
        "scores"}  |  {"ok": False, "error": repr}

Points are expected in the CAMERA frame, metres, float32 (SDK requirement).
On hardware that means the depth-camera optical frame; whatever composes the
grasp into the base frame does so on the client side.
"""
import os, pickle, argparse
import numpy as np
import zmq
from gsnet import create_detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", default="/opt/anygrasp/checkpoints/checkpoint_detection.tar")
    p.add_argument("--max_gripper_width", type=float,
                   default=float(os.environ.get("GRASP_MAX_WIDTH", 0.085)))  # match your gripper
    p.add_argument("--gripper_height", type=float,
                   default=float(os.environ.get("GRASP_HEIGHT", 0.03)))
    p.add_argument("--top_down_grasp", action="store_true")
    p.add_argument("--debug", action="store_true")
    cfgs = p.parse_args()
    cfgs.max_gripper_width = max(0.0, min(0.1, cfgs.max_gripper_width))

    detector = create_detector(cfgs)
    if detector is None:
        raise RuntimeError("create_detector failed (license validation or checkpoint issue)")

    port = os.environ.get("GRASP_PORT", "5666")
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"[grasp_server] model ready, listening on :{port}", flush=True)

    while True:
        req = pickle.loads(sock.recv())

        if req.get("cmd") == "ping":
            sock.send(pickle.dumps({"ok": True, "service": "anygrasp"}))
            continue

        try:
            points = req["points"].astype(np.float32)

            # lims -> region_steering mask (workspace filtering moved into the mask API)
            region_steering = req.get("region_steering")
            lims = req.get("lims")
            if region_steering is None and lims is not None:
                xmin, xmax, ymin, ymax, zmin, zmax = lims
                region_steering = (
                    (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                    (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                    (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
                )

            optional_params = {
                "dense_grasp": req.get("dense_grasp", False),
                "collision_detection": req.get("collision_detection", True),
                "region_steering": region_steering,
                "approach_steering": req.get("approach_steering",
                                             [0, 0, 1] if cfgs.top_down_grasp else None),
                "approach_thresh": req.get("approach_thresh",
                                           np.pi / 6 if cfgs.top_down_grasp else np.pi),
            }

            gg = detector.get_grasp(points, optional_params)

            if gg is None or len(gg) == 0:
                sock.send(pickle.dumps({"ok": True, "n": 0}))
                continue
            if not optional_params["dense_grasp"]:
                gg = gg.nms()
            gg = gg.sort_by_score()
            sock.send(pickle.dumps({
                "ok": True, "n": len(gg),
                "translations": gg.translations,
                "rotations": gg.rotation_matrices,
                "widths": gg.widths, "depths": gg.depths, "scores": gg.scores,
            }))
        except Exception as e:
            sock.send(pickle.dumps({"ok": False, "error": repr(e)}))


if __name__ == "__main__":
    main()
