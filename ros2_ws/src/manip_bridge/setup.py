import os
from glob import glob

from setuptools import setup

package_name = "manip_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="albus",
    maintainer_email="albus@localhost",
    description="ZMQ <-> ROS2 bridge nodes for the perception sidecars",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sam3_bridge = manip_bridge.sam3_bridge_node:main",
            "trellis2_bridge = manip_bridge.trellis2_bridge_node:main",
            "any6d_bridge = manip_bridge.any6d_bridge_node:main",
            "pose_bridge = manip_bridge.pose_bridge_node:main",
            "oriany_bridge = manip_bridge.oriany_bridge_node:main",
            "run_scene = manip_bridge.run_scene:main",
            # still legacy (ros2_bridge/pointso_bridge_node.py via ExecuteProcess)
        ],
    },
)