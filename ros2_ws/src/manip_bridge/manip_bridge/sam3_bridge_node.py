"""ROS2 <-> SAM 3 ZMQ bridge.

Exposes:  /sam3/segment   (manip_interfaces/srv/Segment)

Stateless service: the caller supplies the RGB frame. One prompt maps to
the sidecar's "segment", several to "segment_multi". Zero instances for a
prompt is a valid (non-failure) reply.

Env:
    SAM3_ADDR        tcp://127.0.0.1:5670
    SAM3_TIMEOUT_S   60
"""

import os

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from manip_interfaces.srv import Segment

from .img import image_to_rgb, mono_to_image
from .zmq_client import SidecarClient, SidecarError

SAM3_ADDR = os.environ.get("SAM3_ADDR", "tcp://127.0.0.1:5670")
TIMEOUT_MS = int(float(os.environ.get("SAM3_TIMEOUT_S", 60)) * 1000)


class Sam3Bridge(Node):
    def __init__(self):
        super().__init__("sam3_bridge")
        self.client = SidecarClient(SAM3_ADDR, TIMEOUT_MS)
        self.create_service(Segment, "sam3/segment", self.on_segment,
                            callback_group=MutuallyExclusiveCallbackGroup())
        up = self.client.ping()
        self.get_logger().info(
            f"sam3 bridge up -> {SAM3_ADDR} ({'sidecar alive' if up else 'sidecar NOT responding'})")

    def on_segment(self, req, res):
        prompts = list(req.prompts)
        if not prompts:
            res.success, res.message = False, "no prompts"
            return res
        try:
            rgb = image_to_rgb(req.rgb)
            payload = {"rgb": rgb}
            if req.threshold > 0:
                payload["threshold"] = float(req.threshold)
            if len(prompts) == 1:
                payload.update(cmd="segment", prompt=prompts[0])
                results = [self.client.call(payload)]
            else:
                payload.update(cmd="segment_multi", prompts=prompts)
                results = self.client.call(payload)["results"]
        except (TimeoutError, SidecarError, ValueError) as e:
            res.success, res.message = False, f"{type(e).__name__}: {e}"
            return res

        counts = []
        for p, r in zip(prompts, results):
            masks = np.asarray(r["masks"])
            scores = np.asarray(r["scores"], np.float32).reshape(-1)
            boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
            counts.append(len(scores))
            for i in range(len(scores)):
                res.prompt.append(p)
                res.masks.append(mono_to_image(masks[i], req.rgb.header))
                res.scores.append(float(scores[i]))
                res.boxes_xyxy.extend(map(float, boxes[i]))
        res.success = True
        res.message = ", ".join(f"{p}:{n}" for p, n in zip(prompts, counts))
        return res


def main():
    rclpy.init()
    node = Sam3Bridge()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
