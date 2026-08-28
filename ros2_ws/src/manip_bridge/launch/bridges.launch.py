"""Single entry point for every bridge node.

Services (called once per scene by an orchestrator, e.g. `run_scene`):
    /sam3/segment              Segment
    /trellis2/generate_mesh    GenerateMesh
    /oriany/orient             Orient
Service + streaming tracker (estimate once, then track every camera frame):
    /any6d/estimate  /any6d/release  /any6d/<obj>/pose
    /pose/estimate   /pose/release   /pose/<obj>/pose
Legacy (still topic-JSON, not yet promoted): pointso.

Launch args:
    use_sim_time   true when driving from `ros2 bag play --clock`
    rgb_topic / depth_topic / info_topic   camera topics. Defaults come from
                   the RGB_TOPIC / DEPTH_TOPIC / INFO_TOPIC container env
                   (set once in .env for a given bag), falling back to the
                   current realsense-ros naming. Read the real names off
                   `ros2 bag info`. Setting them in .env also covers
                   `run_scene`, which is `ros2 run` and so does NOT inherit
                   these launch arguments.
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
        DeclareLaunchArgument("rgb_topic", default_value=os.environ.get(
            "RGB_TOPIC", "/camera/camera/color/image_raw")),
        DeclareLaunchArgument("depth_topic", default_value=os.environ.get(
            "DEPTH_TOPIC", "/camera/camera/aligned_depth_to_color/image_raw")),
        DeclareLaunchArgument("info_topic", default_value=os.environ.get(
            "INFO_TOPIC", "/camera/camera/color/camera_info")),

        Node(package="manip_bridge", executable="sam3_bridge", name="sam3_bridge",
             output="screen", parameters=[{"use_sim_time": use_sim_time}]),
        Node(package="manip_bridge", executable="trellis2_bridge", name="trellis2_bridge",
             output="screen", parameters=[{"use_sim_time": use_sim_time}]),
        Node(package="manip_bridge", executable="oriany_bridge", name="oriany_bridge",
             output="screen", parameters=[{"use_sim_time": use_sim_time}]),
        Node(package="manip_bridge", executable="any6d_bridge", name="any6d_bridge",
             output="screen", parameters=[cam]),
        Node(package="manip_bridge", executable="pose_bridge", name="foundationpose_bridge",
             output="screen", parameters=[cam]),
        Node(package="manip_bridge", executable="curobo_bridge", name="curobo_bridge",
             output="screen", parameters=[{
                 "depth_topic": LaunchConfiguration("depth_topic"),
                 "info_topic": LaunchConfiguration("info_topic"),
                 "use_sim_time": use_sim_time,
             }]),


        ExecuteProcess(
            cmd=["python3", os.path.join(LEGACY_DIR, "pointso_bridge_node.py")],
            name="pointso_bridge", output="screen"),
    ])
