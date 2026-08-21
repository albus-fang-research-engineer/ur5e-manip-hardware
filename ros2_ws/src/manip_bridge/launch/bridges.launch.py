"""Single entry point for every bridge node.

Services (called once per scene by an orchestrator, e.g. `run_scene`):
    /sam3/segment              Segment
    /trellis2/generate_mesh    GenerateMesh
Service + streaming tracker (estimate once, then track every camera frame):
    /any6d/estimate  /any6d/release  /any6d/<obj>/pose
    /pose/estimate   /pose/release   /pose/<obj>/pose
Legacy (still topic-JSON, not yet promoted): pointso.

Launch args:
    use_sim_time   true when driving from `ros2 bag play --clock`
    rgb_topic / depth_topic / info_topic   camera topics; defaults match the
                   current realsense-ros naming (node name repeated). Read
                   them off `ros2 bag info` and override if they differ.
    camera_frame   (informational) optical frame the poses are expressed in

Sidecar addresses come from the container environment (docker-compose.yml).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LEGACY_DIR = os.path.expanduser("~/ros2_ws/src/ros2_bridge")


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    cam = {
        "rgb_topic": LaunchConfiguration("rgb_topic"),
        "depth_topic": LaunchConfiguration("depth_topic"),
        "info_topic": LaunchConfiguration("info_topic"),
        "use_sim_time": use_sim_time,
    }
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("rgb_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic",
                              default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("info_topic", default_value="/camera/camera/color/camera_info"),

        Node(package="manip_bridge", executable="sam3_bridge", name="sam3_bridge",
             output="screen", parameters=[{"use_sim_time": use_sim_time}]),
        Node(package="manip_bridge", executable="trellis2_bridge", name="trellis2_bridge",
             output="screen", parameters=[{"use_sim_time": use_sim_time}]),
        Node(package="manip_bridge", executable="any6d_bridge", name="any6d_bridge",
             output="screen", parameters=[cam]),
        Node(package="manip_bridge", executable="pose_bridge", name="foundationpose_bridge",
             output="screen", parameters=[cam]),

        ExecuteProcess(
            cmd=["python3", os.path.join(LEGACY_DIR, "pointso_bridge_node.py")],
            name="pointso_bridge", output="screen"),
    ])
