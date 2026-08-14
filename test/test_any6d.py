"""Any6D sidecar tests: model-free estimate (+ optional img_to_3d) and track
on the sim frame packet, scored against ground-truth cam_T_obj.

Same frame bookkeeping as test_foundationpose.py: the packet mesh, the
packet's cam_T_<obj>, and the server's returned pose all live in the same
body/CV-camera frames, so a direct comparison is legal. Any6D additionally
rescales the mesh during registration -- with the packet's already-metric
mesh the recovered scale should be ~1, which the extents check pins down.

Tolerances are looser than the pose sidecar's (4 cm / 20 deg): the joint
alignment step can trade a little pose accuracy for scale consistency, and
that's fine for grasp/planning consumption.

The img_to_3d path (SAM2 + InstantMesh generation from the anchor RGB) is a
multi-minute generative test -> marked slow, skipped by -m "not slow".
"""

import shutil

import numpy as np
import pytest

from conftest import REPO_ROOT, rot_err_deg

OBJ = "teapot"
TRANS_TOL = 0.04   # m
ROT_TOL = 20.0     # deg
SCALE_TOL = 0.25   # relative AABB error vs packet mesh (metric input)
EST_TIMEOUT = 600_000    # register_any6d does render-and-compare + alignment
GEN_TIMEOUT = 900_000    # + SAM2 + diffusion + InstantMesh


@pytest.fixture(scope="module")
def mesh_name(packet):
    if OBJ not in packet.objects:
        pytest.skip(f"{OBJ} not in packet")
    src = packet.mesh_path(OBJ)
    if src is None:
        pytest.skip(f"{OBJ}.obj missing from packet dir (meshes are gitignored "
                    "in the sim repo — re-run capture on the machine that has them)")
    dst_dir = REPO_ROOT / "foundationpose_runtime" / "meshes"
    dst = dst_dir / f"{OBJ}_packet.obj"
    if not dst.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return dst.name


@pytest.fixture(scope="module")
def estimated(any6d, packet, mesh_name):
    rep = any6d.ok(
        {"cmd": "estimate", "obj": f"{OBJ}_test", "mesh": mesh_name,
         "rgb": packet["rgb"],
         "depth": packet["depth"].astype(np.float32),
         "K": packet["K"].astype(np.float32),
         "mask": packet[f"mask_{OBJ}"].astype(np.uint8),
         "est_refine_iter": 5},
        timeout_ms=EST_TIMEOUT)
    return rep


def test_estimate_pose_matches_gt(estimated, packet):
    est = np.asarray(estimated["pose"], np.float64)
    gt = np.asarray(packet[f"cam_T_{OBJ}"], np.float64)
    t_err = np.linalg.norm(est[:3, 3] - gt[:3, 3])
    r_err = rot_err_deg(est[:3, :3], gt[:3, :3])
    assert t_err < TRANS_TOL, f"translation error {t_err:.4f} m"
    assert r_err < ROT_TOL, f"rotation error {r_err:.1f} deg"


def test_estimate_recovers_metric_scale(estimated, packet):
    """The packet mesh is metric, so the joint-alignment scale should come
    back ~1: compare the returned scaled-mesh AABB against the input's."""
    import trimesh
    src = packet.mesh_path(OBJ)
    gt_extents = np.sort(trimesh.load(src, force="mesh").extents)
    est_extents = np.sort(np.asarray(estimated["extents"], np.float64))
    rel = np.abs(est_extents - gt_extents) / gt_extents
    assert np.all(rel < SCALE_TOL), \
        f"scaled-mesh extents off by {rel} (est {est_extents}, gt {gt_extents})"


def test_track_after_estimate(any6d, estimated, packet):
    """Same frame re-tracked should stay near the estimate (sanity that the
    FoundationPose tracking path works off the scaled mesh)."""
    rep = any6d.ok(
        {"cmd": "track", "obj": f"{OBJ}_test",
         "rgb": packet["rgb"],
         "depth": packet["depth"].astype(np.float32),
         "K": packet["K"].astype(np.float32),
         "track_refine_iter": 2},
        timeout_ms=120_000)
    est0 = np.asarray(estimated["pose"], np.float64)
    est1 = np.asarray(rep["pose"], np.float64)
    assert np.linalg.norm(est1[:3, 3] - est0[:3, 3]) < 0.01
    assert rot_err_deg(est1[:3, :3], est0[:3, :3]) < 5.0


@pytest.mark.slow
def test_img_to_3d_estimate(any6d, packet):
    """Full model-free path: mesh generated from the anchor RGB by
    SAM2 + InstantMesh, then jointly scaled + posed. Generative meshes are
    rough, so only the translation is scored (rotation of a hallucinated
    mesh's body frame vs the packet body frame isn't well-defined)."""
    if OBJ not in packet.objects:
        pytest.skip(f"{OBJ} not in packet")
    rep = any6d.ok(
        {"cmd": "estimate", "obj": f"{OBJ}_gen", "img_to_3d": True,
         "rgb": packet["rgb"],
         "depth": packet["depth"].astype(np.float32),
         "K": packet["K"].astype(np.float32),
         "mask": packet[f"mask_{OBJ}"].astype(np.uint8),
         "est_refine_iter": 5},
        timeout_ms=GEN_TIMEOUT)
    est = np.asarray(rep["pose"], np.float64)
    gt = np.asarray(packet[f"cam_T_{OBJ}"], np.float64)
    t_err = np.linalg.norm(est[:3, 3] - gt[:3, 3])
    assert t_err < 0.06, f"translation error {t_err:.4f} m"
    any6d.ok({"cmd": "release", "obj": f"{OBJ}_gen"})


def test_release(any6d, estimated):
    any6d.ok({"cmd": "release", "obj": f"{OBJ}_test"})
    rep = any6d.err(
        {"cmd": "track", "obj": f"{OBJ}_test",
         "rgb": np.zeros((8, 8, 3), np.uint8),
         "depth": np.zeros((8, 8), np.float32),
         "K": np.eye(3, dtype=np.float32)})
    assert "KeyError" in rep["error"] or "obj" in rep["error"]