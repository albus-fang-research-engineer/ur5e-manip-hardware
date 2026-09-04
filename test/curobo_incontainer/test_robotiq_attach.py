"""Tests for robotiq_attach -- ALL offline (stdlib + pytest only, no curobo,
no CUDA, no meshes needed). A synthetic frozen-URDF fixture mirrors the real
vendored artifact's structure (same link/joint names, box geometry) so the
merge logic is fully exercised in CI; tests against the real vendored file
and the fitted yml run when those artifacts exist and skip otherwise.

Run from repo root:  python -m pytest test/curobo_incontainer/test_robotiq_attach.py -v
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotiq_attach import (  # noqa: E402
    ADAPTER_LINK,
    ADAPTER_THICKNESS,
    BASE_JOINT,
    FROZEN_URDF,
    GRIPPER_LINKS,
    MOUNT_LINK,
    RobotiqAttachError,
    assert_gripper_spheres,
    attach_robotiq,
)

BASE_JOINT_ORIGIN = "0 0 0.004"  # arbitrary non-identity: must be PRESERVED


def _synthetic_frozen(tmp_path, base_parent="world", joint_type="fixed",
                      drop_link=None):
    """Minimal frozen-gripper URDF with the real names."""
    lines = ['<robot name="robotiq_2f85">', '<link name="world"/>']
    for name in GRIPPER_LINKS:
        if name == drop_link:
            continue
        lines.append(
            f'<link name="{name}"><collision><geometry>'
            f'<box size="0.01 0.01 0.01"/></geometry></collision></link>')
    joints = [
        (BASE_JOINT, base_parent, "robotiq_85_base_link", BASE_JOINT_ORIGIN),
        ("robotiq_85_left_knuckle_joint", "robotiq_85_base_link",
         "robotiq_85_left_knuckle_link", "0 0 0"),
        ("robotiq_85_right_knuckle_joint", "robotiq_85_base_link",
         "robotiq_85_right_knuckle_link", "0 0 0"),
        ("robotiq_85_left_finger_joint", "robotiq_85_left_knuckle_link",
         "robotiq_85_left_finger_link", "0 0 0"),
        ("robotiq_85_right_finger_joint", "robotiq_85_right_knuckle_link",
         "robotiq_85_right_finger_link", "0 0 0"),
        ("robotiq_85_left_inner_knuckle_joint", "robotiq_85_base_link",
         "robotiq_85_left_inner_knuckle_link", "0 0 0"),
        ("robotiq_85_right_inner_knuckle_joint", "robotiq_85_base_link",
         "robotiq_85_right_inner_knuckle_link", "0 0 0"),
        ("robotiq_85_left_finger_tip_joint", "robotiq_85_left_finger_link",
         "robotiq_85_left_finger_tip_link", "0 0 0"),
        ("robotiq_85_right_finger_tip_joint", "robotiq_85_right_finger_link",
         "robotiq_85_right_finger_tip_link", "0 0 0"),
    ]
    for name, parent, child, xyz in joints:
        if drop_link in (parent, child):
            continue
        lines.append(
            f'<joint name="{name}" type="{joint_type}">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="0 0 0"/></joint>')
    lines.append("</robot>")
    p = tmp_path / "frozen.urdf"
    p.write_text("\n".join(lines))
    return p


def _arm_root():
    """Stub UR5e tree: just the links the merge touches."""
    return ET.fromstring(
        '<robot name="ur5e">'
        '<link name="base_link"/><link name="wrist_3_link"/>'
        '<link name="tool0"/>'
        '<joint name="wrist_3-tool0" type="fixed">'
        '<parent link="wrist_3_link"/><child link="tool0"/></joint>'
        "</robot>")


def _joints(root):
    return {j.get("name"): j for j in root.findall("joint")}


def _links(root):
    return {l.get("name") for l in root.findall("link")}


# ------------------------------------------------------------- merge behavior
def test_merge_adds_all_links_and_drops_world(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path))
    have = _links(root)
    assert set(GRIPPER_LINKS) <= have
    assert ADAPTER_LINK in have and MOUNT_LINK in have
    assert "world" not in have


def test_mount_chain_topology_and_rotation(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path),
                          rotation=0.25)
    j = _joints(root)
    ur2rq = j["ur_to_robotiq_joint"]
    assert ur2rq.find("parent").get("link") == "tool0"
    assert ur2rq.find("child").get("link") == ADAPTER_LINK
    assert ur2rq.find("origin").get("rpy").split()[-1] == "0.25"

    side = j["gripper_side_joint"]
    assert side.find("parent").get("link") == ADAPTER_LINK
    assert side.find("child").get("link") == MOUNT_LINK
    assert float(side.find("origin").get("xyz").split()[-1]) == pytest.approx(
        ADAPTER_THICKNESS)


def test_base_joint_retargeted_origin_preserved(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path))
    bj = _joints(root)[BASE_JOINT]
    assert bj.find("parent").get("link") == MOUNT_LINK
    assert bj.find("origin").get("xyz") == BASE_JOINT_ORIGIN


def test_all_added_joints_fixed(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path))
    pre = {"wrist_3-tool0"}
    for name, j in _joints(root).items():
        if name not in pre:
            assert j.get("type") == "fixed", name


def test_every_gripper_link_chains_to_tool0(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path))
    child_of = {j.find("child").get("link"): j.find("parent").get("link")
                for j in root.findall("joint")}
    for name in GRIPPER_LINKS + [ADAPTER_LINK, MOUNT_LINK]:
        cur = name
        for _ in range(20):
            if cur == "tool0":
                break
            cur = child_of[cur]
        assert cur == "tool0", f"{name} does not reach tool0"


# ------------------------------------------------------------------- failures
def test_double_attach_raises(tmp_path):
    root = attach_robotiq(_arm_root(), _synthetic_frozen(tmp_path))
    with pytest.raises(RobotiqAttachError, match="already present"):
        attach_robotiq(root, _synthetic_frozen(tmp_path))


def test_missing_vendored_file_raises_with_pointer(tmp_path):
    with pytest.raises(RobotiqAttachError, match="vendor_robotiq.sh"):
        attach_robotiq(_arm_root(), tmp_path / "nope.urdf")


def test_unfrozen_joint_raises(tmp_path):
    frozen = _synthetic_frozen(tmp_path, joint_type="revolute")
    with pytest.raises(RobotiqAttachError, match="non-fixed"):
        attach_robotiq(_arm_root(), frozen)


def test_missing_gripper_link_raises(tmp_path):
    frozen = _synthetic_frozen(tmp_path,
                               drop_link="robotiq_85_left_finger_tip_link")
    with pytest.raises(RobotiqAttachError, match="lacks expected links"):
        attach_robotiq(_arm_root(), frozen)


def test_sphere_guard_names_empty_links():
    spheres = {n: [{"center": [0, 0, 0], "radius": 0.01}]
               for n in GRIPPER_LINKS + [ADAPTER_LINK]}
    assert_gripper_spheres(spheres)  # complete set passes
    del spheres["robotiq_85_left_finger_tip_link"]
    spheres[ADAPTER_LINK] = []  # fitted-but-empty is also a failure
    with pytest.raises(RobotiqAttachError) as e:
        assert_gripper_spheres(spheres)
    assert "robotiq_85_left_finger_tip_link" in str(e.value)
    assert ADAPTER_LINK in str(e.value)


# ------------------------------------- real vendored artifact (skip if absent)
@pytest.mark.skipif(not FROZEN_URDF.is_file(),
                    reason="vendored artifact absent; run scripts/vendor_robotiq.sh")
def test_vendored_artifact_merges_clean():
    root = attach_robotiq(_arm_root(), FROZEN_URDF)
    have = _links(root)
    assert set(GRIPPER_LINKS) <= have and "world" not in have
    # every mesh URI must target the CuroboServer mount, and the file must
    # exist under the repo-side twin of that mount
    repo_root = Path(__file__).resolve().parent
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if "robotiq" not in fn:
            continue  # arm meshes (relative, resolved by asset_path)
        assert fn.startswith("/opt/robot_builder/robotiq/meshes/"), fn
        twin = repo_root / fn.replace("/opt/robot_builder/", "")
        if (repo_root / "robotiq" / "meshes").is_dir():
            assert twin.is_file(), f"vendored mesh missing on host: {twin}"
