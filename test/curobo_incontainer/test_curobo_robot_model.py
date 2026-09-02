"""cuRoboV2 robot-model tests: sphere approximation, robot self-masking from
depth, and mesh attachment to the gripper. UR5e throughout.

These import curobo directly, so they run INSIDE the curobo container:

    docker compose run --rm \
        -v $PWD/test:/opt/test \
        curobo bash -lc "pip install -q pytest && \
            python -m pytest /opt/test/curobo_incontainer -v"

The frame packet must be at /opt/test/data/packet/packet.npz (i.e. keep it
under test/data/ and the single mount above covers it), or set FRAME_PACKET.

What each block proves:

  sphere approximation
      fit_spheres_to_mesh() on the packet's teapot mesh (every SphereFitType),
      checked for containment + surface coverage; and
      RobotBuilder.fit_collision_spheres() on the real UR5e link meshes via
      ur5e_curobo_config (spheres inside inflated link AABBs, sane counts).

  robot self-masking (RobotSegmenter)
      the packet's depth + intrinsics + camera-in-base pose + the packet's own
      joint state -> get_robot_mask, scored against the renderer's GT robot
      mask. Because robosuite's UR5e MJCF and ur_description may disagree on
      the base yaw convention (the base_link / base_link_inertia pi flip), the
      test evaluates yaw 0 and yaw pi about the base z and takes the best —
      and tells you which one won, which is exactly the calibration fact you
      need when wiring the real extrinsics.

  mesh attachment (AttachmentManager)
      fit spheres to the teapot, attach at tool0, verify the attached spheres
      (a) appear in compute_kinematics().robot_spheres, (b) ride rigidly with
      the tool across configurations, and (c) vanish on detach.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from curobo._src.geom.sphere_fit.types import SphereFitType  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.sphere_fit import estimate_sphere_count, fit_spheres_to_mesh  # noqa: E402
from curobo.types import JointState, Pose  # noqa: E402

from ur5e_curobo_config import (  # noqa: E402
    CACHE,
    N_ATTACHED_SPHERES,
    UR5E_LINK_ORIGINS,
    _generate_ur5e_urdf,
    build_ur5e_config,
)

if not torch.cuda.is_available():
    pytest.skip("cuRobo needs CUDA", allow_module_level=True)


# ---------------------------------------------------------------------- packet
def _find_packet():
    env = os.environ.get("FRAME_PACKET")
    cands = [Path(env)] if env else []
    cands += [Path(__file__).resolve().parents[1] / "data" / "packet" / "packet.npz"]
    for p in cands:
        if p.is_file():
            return p
    pytest.skip("frame packet not found — mount test/ with data/packet/ inside, "
                "or set FRAME_PACKET")


@pytest.fixture(scope="module")
def packet():
    p = _find_packet()
    data = np.load(p, allow_pickle=False)
    d = {k: data[k] for k in data.files if k != "meta"}
    d["meta"] = json.loads(str(data["meta"]))
    d["dir"] = p.parent
    return d


@pytest.fixture(scope="module")
def ur5e_cfg():
    return build_ur5e_config()


@pytest.fixture(scope="module")
def kin(ur5e_cfg):
    return Kinematics(KinematicsCfg.from_data_dict(ur5e_cfg["kinematics"]))


def _teapot_mesh(packet):
    trimesh = pytest.importorskip("trimesh")
    p = packet["dir"] / "teapot.obj"
    if not p.exists():
        pytest.skip("teapot.obj not in packet dir")
    return trimesh.load(str(p), force="mesh")


# ------------------------------------------------------- sphere approximation
@pytest.mark.parametrize("fit_type", [SphereFitType.SURFACE,
                                      SphereFitType.VOXEL,
                                      SphereFitType.MORPHIT])
def test_fit_spheres_to_object_mesh(packet, fit_type):
    mesh = _teapot_mesh(packet)
    n = estimate_sphere_count(mesh)
    assert n >= 1
    res = fit_spheres_to_mesh(mesh, num_spheres=min(n, 32), fit_type=fit_type,
                              iterations=60, compute_metrics=True)
    centers = res.centers.detach().cpu().numpy()
    radii = res.radii.detach().cpu().numpy()

    assert len(centers) == len(radii) > 0
    assert np.all(radii > 0)
    # containment: centers inside a modestly inflated mesh AABB
    lo, hi = mesh.bounds
    pad = 0.25 * (hi - lo).max()
    assert np.all(centers >= lo - pad) and np.all(centers <= hi + pad), fit_type
    # coverage: most surface points within reach of some sphere (SURFACE fits
    # tiny beads on the skin; volume types must actually wrap the geometry)
    pts = mesh.sample(2000)
    d = np.linalg.norm(pts[:, None, :] - centers[None], axis=-1) - radii[None]
    slack = 0.02 if fit_type == SphereFitType.SURFACE else 0.01
    frac = float((d.min(axis=1) < slack).mean())
    print(f"[sphere_fit] {fit_type.value}: {len(radii)} spheres, "
          f"coverage@{slack*100:.0f}mm = {frac:.2f}")
    assert frac > 0.85, f"{fit_type}: only {frac:.2f} of surface covered"


def test_robot_builder_fitted_ur5e_spheres(ur5e_cfg):
    from ur5e_curobo_config import link_meshes

    spheres = ur5e_cfg["kinematics"]["collision_spheres"]
    arm_links = {"shoulder_link", "upper_arm_link", "forearm_link",
                 "wrist_1_link", "wrist_2_link", "wrist_3_link"}
    fitted = arm_links & set(spheres)
    assert fitted == arm_links, f"missing sphere sets for {arm_links - fitted}"

    meshes = link_meshes()
    total = 0
    for link in sorted(fitted):
        ss = spheres[link]
        assert len(ss) >= 1
        total += len(ss)
        c = np.array([s["center"] for s in ss])
        r = np.array([s["radius"] for s in ss])
        assert np.all(r > 0.005) and np.all(r < 0.25), f"{link}: silly radii {r}"
        if link in meshes:
            lo, hi = meshes[link].bounds
            pad = 0.05 + 0.2 * (hi - lo).max()
            assert np.all(c >= lo - pad) and np.all(c <= hi + pad), \
                f"{link}: sphere centers escape the link mesh AABB"
    print(f"[robot_builder] UR5e arm fitted with {total} spheres")
    assert 10 <= total <= 400


def test_kinematics_reports_positive_spheres_at_home(kin, ur5e_cfg):
    q = torch.tensor(ur5e_cfg["kinematics"]["cspace"]["default_joint_position"],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    js = JointState.from_position(q, joint_names=kin.joint_names)
    sph = kin.compute_kinematics(js).robot_spheres.view(-1, 4)
    active = sph[sph[:, 3] > 0]
    # attached_object slots are parked (negative radius) until attach
    assert len(active) >= 10
    assert len(active) <= sph.shape[0] - N_ATTACHED_SPHERES + 1


# ------------------------------------------------------- robot self-masking
def _camera_in_base(packet, extra_base_yaw=0.0):
    T_wb = packet["T_world_base"].astype(np.float64)
    if extra_base_yaw:
        c, s = np.cos(extra_base_yaw), np.sin(extra_base_yaw)
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T_wb = T_wb @ Rz
    return np.linalg.inv(T_wb) @ packet["T_world_cam"].astype(np.float64)


def _joint_state_from_packet(packet, joint_names):
    name_to_q = dict(zip(packet["meta"]["joint_names_arm"],
                         packet["qpos_arm"].astype(np.float32)))
    missing = [n for n in joint_names if n not in name_to_q]
    assert not missing, f"packet lacks joints {missing}; has {list(name_to_q)}"
    q = torch.tensor([name_to_q[n] for n in joint_names],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    return JointState.from_position(q, joint_names=list(joint_names))


def test_robot_segmenter_masks_the_arm(packet, ur5e_cfg):
    from curobo.perception import RobotSegmenter
    from curobo.types import CameraObservation

    seg = RobotSegmenter.from_robot_file(
        ur5e_cfg,
        distance_threshold=0.06,
        use_cuda_graph=False,   # one-shot test; skip graph capture warmup
    )
    depth = torch.tensor(packet["depth"], dtype=torch.float32,
                         device="cuda").unsqueeze(0)
    K = torch.tensor(packet["K"], dtype=torch.float32, device="cuda").unsqueeze(0)
    js = _joint_state_from_packet(packet, seg._kinematics.joint_names)
    gt = packet["mask_robot"].astype(bool)
    observed = packet["depth"] > 0

    def run(yaw):
        T_bc = torch.tensor(_camera_in_base(packet, yaw), dtype=torch.float32,
                            device="cuda")
        cam = CameraObservation(depth_image=depth, intrinsics=K,
                                pose=Pose.from_matrix(T_bc),
                                depth_to_meter=1.0)  # packet depth is meters
        mask, filtered = seg.get_robot_mask(cam, js)
        return mask.squeeze(0).cpu().numpy().astype(bool), filtered

    results = {yaw: run(yaw) for yaw in (0.0, np.pi)}
    scores = {}
    for yaw, (pred, _) in results.items():
        tp = (pred & gt & observed).sum()
        recall = tp / max((gt & observed).sum(), 1)
        precision = tp / max((pred & observed).sum(), 1)
        scores[yaw] = (recall, precision)
    best_yaw = max(scores, key=lambda y: scores[y][0] * scores[y][1])
    recall, precision = scores[best_yaw]
    pred, filtered = results[best_yaw]

    print(f"[segmenter] base-yaw {best_yaw:.2f} rad wins: "
          f"recall {recall:.2f}, precision {precision:.2f} "
          f"(other yaw: r={scores[np.pi - best_yaw][0]:.2f})")
    if best_yaw != 0.0:
        print("[segmenter] NOTE: robosuite base frame is pi-yawed vs "
              "ur_description — bake this into the hardware extrinsics.")

    # spheres over-approximate the arm: demand high recall, decent precision
    assert recall > 0.65, f"segmenter recall {recall:.2f} — kinematics/extrinsics off?"
    assert precision > 0.45, f"segmenter precision {precision:.2f} — mask eating the scene"
    # filtered depth actually removed the robot pixels it claimed
    f = filtered.squeeze(0).cpu().numpy()
    assert (f[pred & observed] == 0).mean() > 0.95
    # and left the rest of the scene alone
    keep = observed & ~pred
    assert np.allclose(f[keep], packet["depth"][keep], atol=1e-3)


# --------------------------------------------------------- mesh attachment
def test_attach_mesh_to_tool(kin, packet, ur5e_cfg):
    pytest.importorskip("trimesh")
    from curobo._src.collision.attachment_manager import AttachmentManager
    from curobo.scene import Mesh

    mesh = _teapot_mesh(packet)
    obstacle = Mesh(name="teapot",
                    vertices=np.asarray(mesh.vertices, np.float32),
                    faces=np.asarray(mesh.faces, np.int32).ravel().tolist(),
                    pose=[0, 0, 0, 1, 0, 0, 0])

    q_home = torch.tensor(ur5e_cfg["kinematics"]["cspace"]["default_joint_position"],
                          dtype=torch.float32, device="cuda").unsqueeze(0)
    js_home = JointState.from_position(q_home, joint_names=kin.joint_names)

    def tool_pose(js):
        st = kin.compute_kinematics(js)
        return st.tool_poses.get_link_pose(kin.tool_frames[0])

    def active_spheres(js):
        sph = kin.compute_kinematics(js).robot_spheres.view(-1, 4)
        return sph[sph[:, 3] > 0].detach().cpu().numpy()

    n_before = len(active_spheres(js_home))

    am = AttachmentManager(kin)
    am.attach(js_home, [obstacle], link_name="attached_object",
              num_spheres=N_ATTACHED_SPHERES // 2,
              sphere_fit_type=SphereFitType.VOXEL)

    after = active_spheres(js_home)
    n_added = len(after) - n_before
    print(f"[attach] {n_added} spheres added ({n_before} -> {len(after)})")
    assert 1 <= n_added <= N_ATTACHED_SPHERES

    # attached spheres live near the tool at home config
    Tt = tool_pose(js_home)
    t_home = Tt.position.view(3).detach().cpu().numpy()
    # identify attached spheres as those absent before: compare via radii sets
    # is fragile — instead recompute with the attachment link's slots directly
    kparams = kin.config.kinematics_config
    idx = kparams.get_sphere_index_from_link_name("attached_object").cpu().numpy()
    sph_home = kin.compute_kinematics(js_home).robot_spheres.view(-1, 4)
    att_home = sph_home[idx].detach().cpu().numpy()
    att_home = att_home[att_home[:, 3] > 0]
    assert len(att_home) == n_added
    d_tool = np.linalg.norm(att_home[:, :3] - t_home, axis=1)
    reach = float(max(mesh.extents)) + 0.05
    assert np.all(d_tool < reach), f"attached spheres {d_tool.max():.2f} m from tool"

    # rigidity: move the arm, attached spheres must transform with tool0
    q2 = q_home.clone()
    q2[0, 0] += 0.6
    q2[0, 3] -= 0.4
    js2 = JointState.from_position(q2, joint_names=kin.joint_names)
    sph2 = kin.compute_kinematics(js2).robot_spheres.view(-1, 4)
    att2 = sph2[idx].detach().cpu().numpy()
    att2 = att2[att2[:, 3] > 0]

    def to_tool_frame(att, js):
        T = np.eye(4)
        p = tool_pose(js)
        from scipy.spatial.transform import Rotation as R
        T[:3, :3] = R.from_quat(p.quaternion.view(4).detach().cpu().numpy(),
                                scalar_first=True).as_matrix()
        T[:3, 3] = p.position.view(3).detach().cpu().numpy()
        Ti = np.linalg.inv(T)
        return (Ti[:3, :3] @ att[:, :3].T).T + Ti[:3, 3]

    local1 = to_tool_frame(att_home, js_home)
    local2 = to_tool_frame(att2, js2)
    assert np.allclose(np.sort(local1, axis=0), np.sort(local2, axis=0),
                       atol=5e-3), "attached spheres did not ride rigidly with tool0"
    assert np.allclose(np.sort(att_home[:, 3]), np.sort(att2[:, 3]), atol=1e-5)

    # detach: slots parked again, sphere count restored
    am.detach()
    assert len(active_spheres(js_home)) == n_before


def test_attach_respects_world_pose_offset(kin, packet, ur5e_cfg):
    """Attaching with the object's WORLD pose (grasp-time pose) must place the
    spheres at that world pose for the grasp configuration — the deployment
    path: FoundationPose gives cam_T_obj, we compose to world, attach."""
    from curobo._src.collision.attachment_manager import AttachmentManager
    from curobo.scene import Mesh
    pytest.importorskip("trimesh")

    mesh = _teapot_mesh(packet)
    obstacle = Mesh(name="teapot",
                    vertices=np.asarray(mesh.vertices, np.float32),
                    faces=np.asarray(mesh.faces, np.int32).ravel().tolist(),
                    pose=[0, 0, 0, 1, 0, 0, 0])

    q = torch.tensor(ur5e_cfg["kinematics"]["cspace"]["default_joint_position"],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    js = JointState.from_position(q, joint_names=kin.joint_names)
    tool = kin.compute_kinematics(js).tool_poses.get_link_pose(kin.tool_frames[0])
    t_tool = tool.position.view(3).detach().cpu().numpy()

    # pretend the object sits 10 cm below the tool in base frame at grasp time
    obj_world = t_tool + np.array([0.0, 0.0, -0.10])
    offset = Pose(position=torch.tensor(obj_world, dtype=torch.float32,
                                        device="cuda").view(1, 3),
                  quaternion=torch.tensor([1.0, 0, 0, 0], device="cuda").view(1, 4))

    am = AttachmentManager(kin)
    am.attach(js, [obstacle], link_name="attached_object",
              num_spheres=4, sphere_fit_type=SphereFitType.VOXEL,
              world_objects_pose_offset=offset)
    try:
        kparams = kin.config.kinematics_config
        idx = kparams.get_sphere_index_from_link_name("attached_object").cpu().numpy()
        sph = kin.compute_kinematics(js).robot_spheres.view(-1, 4)
        att = sph[idx].detach().cpu().numpy()
        att = att[att[:, 3] > 0]
        centroid = att[:, :3].mean(0)
        # object mesh centroid was near its own origin; spheres should sit
        # near the commanded world pose, not at the tool origin
        err = np.linalg.norm(centroid - (obj_world + np.asarray(mesh.centroid)))
        print(f"[attach/offset] sphere centroid error vs commanded pose: "
              f"{err*100:.1f} cm")
        assert err < max(mesh.extents), "world_objects_pose_offset ignored"
        assert np.linalg.norm(centroid - t_tool) > 0.03
    finally:
        am.detach()


# --------------------------------------------------------- urdf/config hygiene
def test_link_mesh_origins_are_ur5e_not_ur10e(tmp_path):
    """Joint origins alone are not enough: ur_description also places each
    link's MESH inside its own frame via shoulder_offset/elbow_offset and the
    wrist visual_offsets. Patching only the joints leaves ur5e meshes at
    ur10e offsets -- a visibly disconnected arm and, worse, a sphere fit run
    over misplaced geometry (up to ~38 mm at the elbow).
    """
    import xml.etree.ElementTree as ET

    from curobo._src.util_file import get_assets_path

    gen = _generate_ur5e_urdf(tmp_path / "ur5e.urdf")
    src = Path(get_assets_path()) / "robot" / "ur_description" / "ur10e.urdf"

    def origins(path):
        out = {}
        for link in ET.parse(path).getroot().findall("link"):
            for tag in ("visual", "collision"):
                el = link.find(tag)
                if el is None or el.find("origin") is None:
                    continue
                out[(link.get("name"), tag)] = el.find("origin").get("xyz")
        return out

    got, ur10e = origins(gen), origins(src)
    for link, want in UR5E_LINK_ORIGINS.items():
        for tag in ("visual", "collision"):
            key = (link, tag)
            assert key in got, f"{link} lost its <{tag}> origin"
            assert got[key] == want, f"{link}/{tag}: {got[key]} != {want}"
            assert got[key] != ur10e.get(key), \
                f"{link}/{tag} still carries the ur10e offset {ur10e.get(key)}"

    # <inertial> is ur10e on purpose (see module docstring) -- don't touch it
    for link in ET.parse(gen).getroot().findall("link"):
        if link.get("name") not in UR5E_LINK_ORIGINS:
            continue
        inertial = link.find("inertial")
        if inertial is None or inertial.find("origin") is None:
            continue
        src_link = ET.parse(src).getroot().find(f"./link[@name='{link.get('name')}']")
        assert inertial.find("origin").get("xyz") == \
            src_link.find("inertial").find("origin").get("xyz")


def test_attached_object_is_a_collision_link_but_not_a_mesh_link(ur5e_cfg):
    """builder.save() writes mesh_link_names as a YAML alias of
    collision_link_names, so a naive append registers the virtual attachment
    link as a mesh link too -- it then renders as a stray blob in
    visualize() and enters the self-mask segmenter's link set.
    """
    kin = ur5e_cfg["kinematics"]
    assert "attached_object" in kin["collision_link_names"]
    assert "attached_object" not in kin["mesh_link_names"], \
        "attached_object leaked into mesh_link_names (shared-list alias)"
    assert kin["collision_link_names"] is not kin["mesh_link_names"]


def test_saved_config_has_no_shared_link_name_alias():
    """Guard the on-disk artifact, not just the in-memory dict."""
    from curobo._src.util_file import load_yaml

    yml = CACHE / "ur5e.yml"
    if not yml.exists():
        build_ur5e_config()
    kin = load_yaml(str(yml))["kinematics"]
    assert "attached_object" not in kin["mesh_link_names"]
