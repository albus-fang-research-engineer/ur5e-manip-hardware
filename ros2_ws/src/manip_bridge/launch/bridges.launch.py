"""Single entry point for every bridge node.

Packaged nodes launch as proper ROS2 nodes. The not-yet-migrated pose and
pointso scripts run as plain processes from the legacy ros2_bridge/ mount,
which compose places inside the workspace at src/ros2_bridge (it has a
COLCON_IGNORE, so colcon never tries to build it). To migrate one: move it
into manip_bridge/manip_bridge/, add its console_script in setup.py, and
swap ExecuteProcess -> Node here.

Sidecar addresses (POSE_ADDR, POINTSO_ADDR, TRELLIS_ADDR, ...) come from the
container environment set in docker-compose.yml.
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

LEGACY_DIR = os.path.expanduser("~/ros2_ws/src/ros2_bridge")


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="manip_bridge",
            executable="trellis2_bridge",
            name="trellis2_bridge",
            output="screen",
        ),
        ExecuteProcess(
            cmd=["python3", os.path.join(LEGACY_DIR, "pose_bridge_node.py")],
            name="pose_bridge",
            output="screen",
        ),
        ExecuteProcess(
            cmd=["python3", os.path.join(LEGACY_DIR, "pointso_bridge_node.py")],
            name="pointso_bridge",
            output="screen",
        ),
    ])
