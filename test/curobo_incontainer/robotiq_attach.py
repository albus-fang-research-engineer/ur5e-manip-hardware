"""Attach the frozen Robotiq 2F-85 chain to a generated UR5e URDF.

Stdlib-only on purpose (no curobo import): the merge is pure XML surgery, so
it is unit-testable on any host while ur5e_curobo_config -- which imports
curobo at module top -- stays container-only.

Inputs and provenance:

  robotiq/robotiq_2f85_frozen.urdf   produced by scripts/vendor_robotiq.sh
      from the apt package ros-humble-robotiq-description (PickNik, ROS build
      farm): xacro-expanded, every gripper joint converted to type="fixed" at
      q=0 (2F-85 fully OPEN), <mimic>/<limit>/<ros2_control>/<transmission>
      stripped, mesh URIs rewritten to /opt/robot_builder/robotiq/meshes/...
      (the path test/curobo_incontainer is mounted at inside CuroboServer).
      The file keeps the standalone wrapper's dummy "world" root; this module
      drops it during the merge.

  Mount chain (mirrors the installed package's ur_to_robotiq_adapter.urdf.xacro
  so frames cross-check against a live robot_description later):

      tool0
        --[ur_to_robotiq_joint, rpy z=rotation]--> ur_to_robotiq_link
        --[gripper_side_joint, xyz z=+0.011]-----> gripper_mount_link
        --[robotiq_85_base_joint, origin PRESERVED from the frozen URDF]
                                                -> robotiq_85_base_link -> ...

  "rotation" is the physical coupler clocking about tool0 z. It is 0 for the
  current cell; if the coupler is ever re-mounted clocked, change
  COUPLER_ROTATION (env CUROBO_ROBOTIQ_COUPLER_ROT) -- and remember every
  grasp transform measured against tool0 shifts with it.

The gripper is attached RIGID AT OPEN: locked joints mean cuRobo never sees a
mimic chain, the segmenter needs no gripper joint state, and the fingers are
wrist geometry for both self-masking and planning. Grasp-width reasoning does
not belong in this model.
"""

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROBOTIQ_DIR = Path(__file__).resolve().parent / "robotiq"
FROZEN_URDF = ROBOTIQ_DIR / "robotiq_2f85_frozen.urdf"

# Mesh root as seen from inside CuroboServer (test/curobo_incontainer is
# mounted at /opt/robot_builder). vendor_robotiq.sh writes URIs against this.
MESH_ROOT = "/opt/robot_builder/robotiq/meshes"

# gripper_side_joint z from the INSTALLED adapter xacro (apt snapshot):
# <origin xyz="0 0 0.011" .../> -- the 11 mm UR-to-Robotiq adapter plate.
ADAPTER_THICKNESS = 0.011

COUPLER_ROTATION = float(os.environ.get("CUROBO_ROBOTIQ_COUPLER_ROT", "0.0"))

# The 9 links of the frozen 2F-85 (installed-package naming), asserted both
# at merge time and against the fitted collision_spheres afterwards --
# RobotBuilder's _fit_single_link SILENTLY SKIPS links whose mesh path does
# not resolve, so presence must be checked loudly, never assumed.
GRIPPER_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link",
]

# Links added by the merge itself (adapter plate carries real collision
# geometry and is fitted too; the mount link is a bare frame).
ADAPTER_LINK = "ur_to_robotiq_link"
MOUNT_LINK = "gripper_mount_link"

BASE_JOINT = "robotiq_85_base_joint"
DUMMY_ROOT = "world"


class RobotiqAttachError(RuntimeError):
    pass


def _subelem(parent, tag, **attrib):
    el = ET.SubElement(parent, tag)
    for k, v in attrib.items():
        el.set(k, v)
    return el


def _make_adapter_link():
    """ur_to_robotiq_link with the plate's visual/collision meshes."""
    link = ET.Element("link", {"name": ADAPTER_LINK})
    for tag, sub in (("visual", "visual/ur_to_robotiq_adapter.dae"),
                     ("collision", "collision/ur_to_robotiq_adapter.stl")):
        sec = _subelem(link, tag)
        _subelem(sec, "origin", xyz="0 0 0", rpy="0 0 0")
        geo = _subelem(sec, "geometry")
        _subelem(geo, "mesh", filename=f"{MESH_ROOT}/{sub}")
    return link


def attach_robotiq(robot_root, frozen_path=None, parent_link="tool0",
                   rotation=None):
    """Splice the frozen gripper chain onto `parent_link` of a URDF tree.

    Mutates and returns robot_root (an ET <robot> Element). Raises
    RobotiqAttachError on any structural surprise -- a merge that half
    succeeds would fit half a gripper silently.
    """
    frozen_path = Path(frozen_path) if frozen_path else FROZEN_URDF
    rotation = COUPLER_ROTATION if rotation is None else float(rotation)

    if not frozen_path.is_file():
        raise RobotiqAttachError(
            f"vendored gripper URDF not found: {frozen_path}\n"
            "run scripts/vendor_robotiq.sh (needs the Ros2Bridge container "
            "with ros-humble-robotiq-description installed)")

    grip = ET.parse(str(frozen_path)).getroot()

    have = {l.get("name") for l in robot_root.findall("link")}
    if parent_link not in have:
        raise RobotiqAttachError(f"parent link {parent_link!r} not in robot")
    clash = (set(GRIPPER_LINKS) | {ADAPTER_LINK, MOUNT_LINK}) & have
    if clash:
        raise RobotiqAttachError(
            f"links already present (double attach?): {sorted(clash)}")

    glinks = {l.get("name"): l for l in grip.findall("link")}
    gjoints = {j.get("name"): j for j in grip.findall("joint")}

    missing = [n for n in GRIPPER_LINKS if n not in glinks]
    if missing:
        raise RobotiqAttachError(
            f"frozen URDF lacks expected links {missing}; package layout "
            f"changed? has {sorted(glinks)}")
    if BASE_JOINT not in gjoints:
        raise RobotiqAttachError(f"frozen URDF lacks {BASE_JOINT}")

    bad = [(n, j.get("type")) for n, j in gjoints.items()
           if j.get("type") != "fixed"]
    if bad:
        raise RobotiqAttachError(
            f"frozen URDF has non-fixed joints {bad}; re-run "
            "scripts/vendor_robotiq.sh (freeze step)")

    # -- mount chain -------------------------------------------------------
    robot_root.append(_make_adapter_link())
    j = _subelem(robot_root, "joint", name="ur_to_robotiq_joint", type="fixed")
    _subelem(j, "parent", link=parent_link)
    _subelem(j, "child", link=ADAPTER_LINK)
    _subelem(j, "origin", xyz="0 0 0", rpy=f"0 0 {rotation}")

    robot_root.append(ET.Element("link", {"name": MOUNT_LINK}))
    j = _subelem(robot_root, "joint", name="gripper_side_joint", type="fixed")
    _subelem(j, "parent", link=ADAPTER_LINK)
    _subelem(j, "child", link=MOUNT_LINK)
    _subelem(j, "origin", xyz=f"0 0 {ADAPTER_THICKNESS}", rpy="0 0 0")

    # -- gripper chain: drop dummy root, retarget base joint, keep origin --
    for name, link in glinks.items():
        if name == DUMMY_ROOT:
            continue
        robot_root.append(copy.deepcopy(link))
    for name, joint in gjoints.items():
        jc = copy.deepcopy(joint)
        if name == BASE_JOINT:
            p = jc.find("parent")
            if p is None or p.get("link") != DUMMY_ROOT:
                raise RobotiqAttachError(
                    f"{BASE_JOINT} parent is "
                    f"{None if p is None else p.get('link')!r}, expected "
                    f"{DUMMY_ROOT!r}; wrapper convention changed?")
            p.set("link", MOUNT_LINK)
        robot_root.append(jc)

    _validate(robot_root, parent_link)
    return robot_root


def _validate(root, parent_link):
    links = [l.get("name") for l in root.findall("link")]
    if len(links) != len(set(links)):
        dupes = sorted({n for n in links if links.count(n) > 1})
        raise RobotiqAttachError(f"duplicate links after merge: {dupes}")
    lset = set(links)
    if DUMMY_ROOT in lset:
        raise RobotiqAttachError(f"dummy root {DUMMY_ROOT!r} survived merge")

    child_of = {}
    for j in root.findall("joint"):
        pa, ch = j.find("parent"), j.find("child")
        if pa is None or ch is None:
            continue  # transmission-style refs -- not kinematic
        for end in (pa.get("link"), ch.get("link")):
            if end not in lset:
                raise RobotiqAttachError(
                    f"joint {j.get('name')!r} references missing link {end!r}")
        child_of[ch.get("link")] = pa.get("link")

    # every gripper link must reach parent_link through the merged tree
    targets = GRIPPER_LINKS + [ADAPTER_LINK, MOUNT_LINK]
    for name in targets:
        cur, seen = name, set()
        while cur in child_of and cur not in seen:
            seen.add(cur)
            cur = child_of[cur]
            if cur == parent_link:
                break
        else:
            raise RobotiqAttachError(
                f"{name!r} not connected to {parent_link!r} after merge")


def assert_gripper_spheres(collision_spheres):
    """Loud guard against _fit_single_link's silent skip: every gripper link
    (adapter plate included) must have >=1 fitted sphere."""
    empty = [n for n in GRIPPER_LINKS + [ADAPTER_LINK]
             if not collision_spheres.get(n)]
    if empty:
        raise RobotiqAttachError(
            f"no collision spheres fitted for {empty} -- mesh paths "
            f"unresolved inside the container? Expected roots under "
            f"{MESH_ROOT}. (_fit_single_link skips silently; this assert "
            "exists so it cannot.)")
