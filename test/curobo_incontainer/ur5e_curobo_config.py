"""Build a cuRoboV2 UR5e robot config at test time.

cuRoboV2 ships UR5e MESHES (content/assets/robot/ur_description/meshes/ur5e)
but only a ur10e config/URDF. The two arms share the exact ur_description
frame conventions and link/joint names — they differ only in the six kinematic
lengths — so we generate ur5e.urdf from the shipped ur10e.urdf by substituting
those parameters and repointing the meshes, then let RobotBuilder fit the
collision spheres. That makes this module double duty:

  1. it produces the UR5e config consumed by the self-masking and attachment
     tests, and
  2. RobotBuilder.fit_collision_spheres() on the real UR5e link meshes IS the
     "sphere approximation of the robot" functionality under test.

UR5e parameters substituted (official UR kinematics, same convention as the
shipped ur10e.urdf: shoulder d1, elbow a2, wrist_1 (a3, d4), wrist_2 d5,
wrist_3 d6):

    ur10e: 0.1807  -0.6127  -0.57155/0.17415  0.11985  0.11655
    ur5e:  0.1625  -0.425   -0.3922 /0.1333   0.0997   0.0996

Left as-is on purpose: link inertials (ur10e values — irrelevant for
kinematics/spheres/segmentation; do NOT reuse this config for the
inverse-dynamics / torque-limit features) and joint effort limits (same
caveat). The ur10e camera_mount link is dropped — it's that rig's hardware.

An "attached_object" virtual link is injected following the franka.yml
pattern (extra_links + extra_collision_spheres slots + self-collision
entries) so the AttachmentManager tests have sphere slots to write into.

Cache: everything lands under CUROBO_TEST_CACHE (default /tmp/ur5e_curobo)
and is reused across test runs in the same container.
"""

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from curobo._src.util_file import get_assets_path, load_yaml, write_yaml

CACHE = Path(os.environ.get("CUROBO_TEST_CACHE", "/tmp/ur5e_curobo"))
N_ATTACHED_SPHERES = 16
UR5E_HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# joint name -> (origin xyz, origin rpy or None to keep)
UR5E_ORIGINS = {
    "shoulder_pan_joint": ("0 0 0.1625", None),
    "elbow_joint": ("-0.425 0 0", None),
    "wrist_1_joint": ("-0.3922 0 0.1333", None),
    "wrist_2_joint": ("0 -0.0997 0", None),
    "wrist_3_joint": ("0 0.0996 0", None),
}


def _generate_ur5e_urdf(out_path: Path) -> Path:
    src = Path(get_assets_path()) / "robot" / "ur_description" / "ur10e.urdf"
    tree = ET.parse(src)
    root = tree.getroot()

    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name in UR5E_ORIGINS:
            xyz, rpy = UR5E_ORIGINS[name]
            origin = joint.find("origin")
            origin.set("xyz", xyz)
            if rpy is not None:
                origin.set("rpy", rpy)

    # meshes/ur10e/... -> meshes/ur5e/... (same filenames both dirs)
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if "meshes/ur10e/" in fn:
            mesh.set("filename", fn.replace("meshes/ur10e/", "meshes/ur5e/"))

    # drop the ur10e rig's camera mount (link + fixed joint)
    for tag in ("link", "joint"):
        for el in [e for e in root.findall(tag) if "camera_mount" in (e.get("name") or "")]:
            root.remove(el)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path)
    return out_path


def _inject_attached_object(kin: dict) -> None:
    """Mirror the franka.yml attached_object pattern onto the built config."""
    kin.setdefault("collision_link_names", [])
    if "attached_object" not in kin["collision_link_names"]:
        kin["collision_link_names"].append("attached_object")
    kin["extra_collision_spheres"] = {"attached_object": N_ATTACHED_SPHERES}
    kin["extra_links"] = {
        "attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "link_name": "attached_object",
            "parent_link_name": "tool0",
        }
    }
    kin.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
    ignore = kin.setdefault("self_collision_ignore", {})
    ignore.setdefault("attached_object", [])
    for l in ("tool0", "wrist_3_link", "wrist_2_link"):
        if l not in ignore["attached_object"]:
            ignore["attached_object"].append(l)


def build_ur5e_config(force: bool = False) -> dict:
    """Generate (or load cached) UR5e config with fitted collision spheres.

    Returns the {"kinematics": {...}} dict that both
    RobotSegmenter.from_robot_file and KinematicsCfg.from_data_dict(inner)
    consume. Also writes CACHE/ur5e.yml for inspection / reuse in the server.
    """
    yml = CACHE / "ur5e.yml"
    if yml.exists() and not force:
        return load_yaml(str(yml))

    from curobo._src.geom.sphere_fit.types import SphereFitType
    from curobo.robot_builder import RobotBuilder

    urdf = _generate_ur5e_urdf(CACHE / "ur5e.urdf")
    asset_root = str(Path(get_assets_path()) / "robot" / "ur_description")

    builder = RobotBuilder(str(urdf), asset_path=asset_root, tool_frames=["tool0"])
    fit_name = os.environ.get("CUROBO_TEST_FIT_TYPE", "voxel").upper()
    builder.fit_collision_spheres(
        fit_type=SphereFitType[fit_name],
        sphere_density=1.0,
        use_collision_mesh=True,   # UR collision STLs are clean & watertight
        compute_metrics=True,
    )
    builder.compute_collision_matrix()
    config = builder.build()
    builder.save(config, str(yml))

    data = load_yaml(str(yml))
    kin = data["kinematics"]
    _inject_attached_object(kin)
    if "cspace" in kin:
        kin["cspace"]["default_joint_position"] = list(UR5E_HOME)
    write_yaml(data, str(yml))
    # keep the builder's per-link metrics readable next to the config
    write_yaml(
        {l: {k: float(v) for k, v in vars(m).items() if isinstance(v, (int, float))}
         for l, m in builder.link_metrics.items()},
        str(CACHE / "ur5e_sphere_metrics.yml"),
    )
    return data


def link_meshes(urdf_path=None) -> dict:
    """link name -> trimesh of its collision geometry, for fit-quality checks."""
    import trimesh

    from curobo.robot_parser import UrdfRobotParser

    urdf = Path(urdf_path) if urdf_path else CACHE / "ur5e.urdf"
    if not urdf.exists():
        _generate_ur5e_urdf(urdf)
    asset_root = str(Path(get_assets_path()) / "robot" / "ur_description")
    parser = UrdfRobotParser(str(urdf), mesh_root=asset_root, load_meshes=True)
    out = {}
    for link in parser.get_link_names():
        try:
            geoms = parser.get_link_geometry(link)
        except Exception:
            continue
        meshes = []
        for g in geoms if isinstance(geoms, (list, tuple)) else [geoms]:
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)
            elif hasattr(g, "vertices") and hasattr(g, "faces"):
                meshes.append(trimesh.Trimesh(vertices=g.vertices, faces=g.faces))
        if meshes:
            out[link] = trimesh.util.concatenate(meshes)
    return out


if __name__ == "__main__":
    cfg = build_ur5e_config(force=True)
    n = sum(len(v) for v in cfg["kinematics"].get("collision_spheres", {}).values())
    print(f"built {CACHE/'ur5e.yml'} with {n} fitted spheres "
          f"(+{N_ATTACHED_SPHERES} attached_object slots)")
