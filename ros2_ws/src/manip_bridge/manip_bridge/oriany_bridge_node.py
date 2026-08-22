"""ROS2 <-> Orient Anything V2 ZMQ bridge.

    /oriany/orient   srv Orient   -> sidecar "orient"

A service, not a tracker, and deliberately so. The sidecar is stateless
(no session dict, no release), and semantic orientation is a scene-time
question: once a body frame is registered, its orientation is carried
per-frame by the pose tracker, so re-answering "which way is the front"
every frame buys nothing. Same shape as sam3 / trellis2.

No TF is published: the model returns a rotation only, and a TF frame
needs an origin. Compose R_obj with the pose sidecar's translation
client-side if you want a frame.

CROPPING lives here rather than in the caller, because getting it wrong is
silent. Upstream runs resize_foreground(img, 0.85) *inside* the
remove_bkg path only; a tight bbox crop with remove_bkg=False is a framing
the model never saw in training. So on the masked path this node
reproduces that framing itself: bbox -> fill background -> square pad to
`fg_ratio`.

Mask convention matches GenerateMesh:
    empty (0x0) mask -> whole image, remove_bkg=True   (upstream app.py)
    non-empty mask   -> masked square crop, remove_bkg=False

`bg_fill` is the one number here that is NOT verified against upstream --
rembg emits RGBA and the demo's compositing colour isn't legible from the
server wrapper. It is a parameter for that reason; sweep it against the
matting path (`run_scene --oriany-matting`) before trusting either.

Env:
    ORIANY_ADDR        tcp://127.0.0.1:5673
    ORIANY_TIMEOUT_S   120

Parameters:
    fg_ratio   foreground fraction of the padded square (upstream: 0.85)
    bg_fill    0-255 grey level painted outside the mask
"""

import os

import numpy as np
import rclpy
from geometry_msgs.msg import QuaternionStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from manip_interfaces.srv import Orient

from .img import image_to_mono, image_to_rgb
from .zmq_client import SidecarClient, SidecarError

ADDR = os.environ.get("ORIANY_ADDR", "tcp://127.0.0.1:5673")
TIMEOUT_MS = int(float(os.environ.get("ORIANY_TIMEOUT_S", 120)) * 1000)


def square_crop(rgb, mask, fg_ratio, bg_fill):
    """Mask bbox -> background-filled square whose foreground occupies
    `fg_ratio` of the side. Mirrors upstream resize_foreground()."""
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    fg = rgb[y0:y1, x0:x1].copy()
    fg[mask[y0:y1, x0:x1] == 0] = bg_fill
    h, w = fg.shape[:2]
    side = max(int(round(max(h, w) / max(fg_ratio, 1e-3))), max(h, w))
    out = np.full((side, side, 3), bg_fill, np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy:oy + h, ox:ox + w] = fg
    return out, [x0, y0, x1, y1]


class OrianyBridge(Node):
    def __init__(self):
        super().__init__("oriany_bridge")
        self.declare_parameter("fg_ratio", 0.85)
        self.declare_parameter("bg_fill", 255)
        self.client = SidecarClient(ADDR, TIMEOUT_MS)
        self.create_service(Orient, "oriany/orient", self.on_orient)
        up = self.client.ping()
        self.get_logger().info(
            f"oriany bridge up -> {ADDR} "
            f"({'sidecar alive' if up else 'sidecar NOT responding'})")

    def on_orient(self, req, res):
        try:
            rgb = image_to_rgb(req.rgb)
            if req.mask.height and req.mask.width:
                mask = image_to_mono(req.mask) > 0
                if mask.sum() == 0:
                    raise ValueError("mask is non-empty but has no set pixels")
                if mask.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"mask {mask.shape} does not match rgb {rgb.shape[:2]}")
                img, bbox = square_crop(
                    rgb, mask,
                    float(self.get_parameter("fg_ratio").value),
                    int(self.get_parameter("bg_fill").value))
                remove_bkg = False
            else:
                img, bbox, remove_bkg = rgb, [], True
        except ValueError as e:
            res.success, res.message = False, f"bad request: {e}"
            return res

        try:
            rep = self.client.call(
                {"cmd": "orient", "image": img, "remove_bkg": remove_bkg})
        except (TimeoutError, SidecarError) as e:
            res.success, res.message = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(res.message)
            return res

        res.azimuth = float(rep["azimuth"])
        res.elevation = float(rep["elevation"])
        res.rotation = float(rep["rotation"])
        res.alpha = int(rep["alpha"])
        res.bbox_xyxy = [int(v) for v in bbox]

        q = Rotation.from_matrix(
            np.asarray(rep["R_obj"], np.float64).reshape(3, 3)).as_quat()
        qs = QuaternionStamped()
        qs.header = req.rgb.header
        (qs.quaternion.x, qs.quaternion.y,
         qs.quaternion.z, qs.quaternion.w) = map(float, q)
        res.orientation = qs

        res.success = True
        res.message = (f"az={res.azimuth:.1f} el={res.elevation:.1f} "
                       f"ro={res.rotation:.1f} alpha={res.alpha} "
                       f"{'matting' if remove_bkg else 'masked crop'}")
        self.get_logger().info(res.message)
        return res


def main():
    rclpy.init()
    node = OrianyBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
