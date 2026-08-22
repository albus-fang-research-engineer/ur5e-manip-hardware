"""Camera topic defaults, in one place.

Every node here declares rgb/depth/info topic parameters, and `run_scene`
is launched with `ros2 run` so it does NOT inherit the launch file's
arguments -- which means a bag whose topics differ from the realsense-ros
default has to be spelled out twice, and forgetting the second one gives
you a node that silently never fires a callback.

The env vars below are read from the container environment (set them once
in .env / docker-compose.yml for a given bag) and used as the *defaults*
for those parameters, so one setting covers both. Explicit launch args and
`-p` overrides still win.

bridges.launch.py deliberately re-reads the same env vars rather than
importing this module: a launch file that imports the package it launches
breaks in ways that are annoying to debug.
"""

import os

RGB_TOPIC = os.environ.get("RGB_TOPIC", "/camera/camera/color/image_raw")
DEPTH_TOPIC = os.environ.get(
    "DEPTH_TOPIC", "/camera/camera/aligned_depth_to_color/image_raw")
INFO_TOPIC = os.environ.get("INFO_TOPIC", "/camera/camera/color/camera_info")
