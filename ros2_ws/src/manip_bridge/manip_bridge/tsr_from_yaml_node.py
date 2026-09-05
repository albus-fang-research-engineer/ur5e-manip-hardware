"""Publish a hand-authored TSR from a YAML file, latched, and stay alive.

    ros2 run manip_bridge tsr_from_yaml -- config/tsr_grasp_example.yaml
    ros2 run manip_bridge tsr_from_yaml -- my.yaml --topic /tsr/mug/grasp

The HAND arm of the TSR ablation, and the test harness for grasp_filter
before any TSR producer exists. Spec schema: manip_bridge/tsr_spec.py.

Latched (TRANSIENT_LOCAL) on purpose: grasp_filter subscribes latched, and
DDS drops a volatile publisher on that subscription silently. The node
therefore must keep running -- a latched sample lives only as long as its
publisher. Ctrl-C to retract.

header.stamp is set to now at publish time: for a snapshot TSR that is
"when it was frozen", which is what grasp_filter's report records.
"""

import argparse
import sys

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from manip_interfaces.msg import TSR

from .grasp_filter import tsr_to_flat
from .tsr_spec import tsr_from_spec


class TsrFromYaml(Node):
    def __init__(self, path: str, topic_override: str | None):
        super().__init__("tsr_from_yaml")
        with open(path) as f:
            spec = yaml.safe_load(f)
        tsr, frame_id, topic = tsr_from_spec(spec)
        topic = topic_override or topic
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(TSR, topic, latched)
        msg = TSR()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.t0_w, msg.tw_e, msg.bw = tsr_to_flat(tsr)
        msg.name = tsr.name
        self.pub.publish(msg)
        lo, hi = tsr.Bw[:, 0], tsr.Bw[:, 1]
        self.get_logger().info(
            f"latched {tsr.name!r} on {topic} in frame '{frame_id}' from {path}\n"
            f"  w origin {tsr.T0_w[:3, 3].round(4).tolist()}\n"
            f"  Bw trans lo {lo[:3].round(4).tolist()} hi {hi[:3].round(4).tolist()} m\n"
            f"  Bw rot   lo {[round(float(v), 1) for v in lo[3:] * 180 / 3.141592653589793]} "
            f"hi {[round(float(v), 1) for v in hi[3:] * 180 / 3.141592653589793]} deg\n"
            f"  staying alive so the latch persists; Ctrl-C to retract")


def main():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml", help="TSR spec file (see manip_bridge/tsr_spec.py)")
    ap.add_argument("--topic", default=None, help="override the spec's topic")
    a = ap.parse_args(argv)
    rclpy.init(args=sys.argv)
    node = TsrFromYaml(a.yaml, a.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
