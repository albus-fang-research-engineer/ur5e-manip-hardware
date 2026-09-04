#!/usr/bin/env bash
# Vendor the Robotiq 2F-85 description into test/curobo_incontainer/robotiq/
# as a FROZEN (all joints fixed at open) URDF + meshes, for the cuRobo sphere
# fit. Source of truth: ros-humble-robotiq-description (apt, PickNik via the
# ROS build farm) installed in the Ros2Bridge image -- see
# docker/Dockerfile.ros2-bridge.
#
# Run from the repo root on the host, with the Ros2Bridge container up:
#     ./scripts/vendor_robotiq.sh
#
# Produces:
#     test/curobo_incontainer/robotiq/robotiq_2f85_frozen.urdf   (commit this)
#     test/curobo_incontainer/robotiq/meshes/                    (gitignored;
#         regenerate per checkout with this script -- binaries stay out of git)
#
# Mesh URIs are rewritten to /opt/robot_builder/robotiq/meshes/... -- the
# path test/curobo_incontainer is mounted at inside CuroboServer. Both URI
# styles the package has shipped are handled (package:// and file://<prefix>).
set -euo pipefail

CONTAINER="${ROBOTIQ_BRIDGE_CONTAINER:-Ros2Bridge}"
DEST="test/curobo_incontainer/robotiq"
SHARE="/opt/ros/humble/share/robotiq_description"
MESH_ROOT_IN_CUROBO="/opt/robot_builder/robotiq/meshes"

[ -d test/curobo_incontainer ] || {
    echo "run from the repo root (test/curobo_incontainer not found)" >&2
    exit 1
}
docker exec "$CONTAINER" test -d "$SHARE" || {
    echo "robotiq_description not installed in $CONTAINER -- rebuild the" >&2
    echo "bridge image (docker compose up -d --build ros2-bridge)" >&2
    exit 1
}

mkdir -p "$DEST"

# 1. xacro-expand + freeze + URI rewrite, all inside the bridge container
docker exec -e MESH_ROOT="$MESH_ROOT_IN_CUROBO" "$CONTAINER" bash -c '
set -euo pipefail
source /opt/ros/humble/setup.bash
xacro '"$SHARE"'/urdf/robotiq_2f_85_gripper.urdf.xacro > /tmp/robotiq_raw.urdf
python3 - <<PY
import os
import xml.etree.ElementTree as ET

MESH_ROOT = os.environ["MESH_ROOT"]
tree = ET.parse("/tmp/robotiq_raw.urdf")
root = tree.getroot()

for tag in ("ros2_control", "transmission", "gazebo"):
    for el in list(root.findall(tag)):
        root.remove(el)

n_frozen = 0
for joint in root.findall("joint"):
    if joint.get("type") in ("revolute", "continuous", "prismatic"):
        joint.set("type", "fixed")   # q=0 == 2F-85 fully OPEN
        n_frozen += 1
        for child in ("mimic", "limit", "dynamics", "safety_controller"):
            el = joint.find(child)
            if el is not None:
                joint.remove(el)

n_mesh = 0
for mesh in root.iter("mesh"):
    fn = mesh.get("filename", "")
    for prefix in ("package://robotiq_description/meshes",
                   "file:///opt/ros/humble/share/robotiq_description/meshes"):
        if fn.startswith(prefix):
            mesh.set("filename", MESH_ROOT + fn[len(prefix):])
            n_mesh += 1
            break

assert n_frozen > 0, "no joints frozen -- xacro output changed?"
assert n_mesh > 0, "no mesh URIs rewritten -- unknown URI style in package"
tree.write("/tmp/robotiq_2f85_frozen.urdf")
links = [l.get("name") for l in root.findall("link")]
print(f"froze {n_frozen} joints, rewrote {n_mesh} mesh URIs, "
      f"{len(links)} links: {links}")
PY
'

# 2. pull artifacts onto the host
docker cp "$CONTAINER":/tmp/robotiq_2f85_frozen.urdf "$DEST/robotiq_2f85_frozen.urdf"
rm -rf "$DEST/meshes"
docker cp "$CONTAINER":"$SHARE"/meshes "$DEST/meshes"

# 3. resolvability check from CuroboServer's viewpoint (skip if not running)
if docker exec CuroboServer true 2>/dev/null; then
    docker exec CuroboServer python3 - <<'PY'
import os
import xml.etree.ElementTree as ET
t = ET.parse("/opt/robot_builder/robotiq/robotiq_2f85_frozen.urdf")
missing = [m.get("filename") for m in t.getroot().iter("mesh")
           if not os.path.isfile(m.get("filename", ""))]
assert not missing, f"unresolved mesh paths in CuroboServer: {missing}"
print("all mesh paths resolve inside CuroboServer")
PY
else
    echo "CuroboServer not running -- skipped in-container mesh check"
fi

echo "vendored -> $DEST (urdf: commit; meshes: gitignored, regenerable)"
