"""FoundationPose sidecar tests: register + track on the sim frame packet,
scored against ground-truth cam_T_obj.

Frame bookkeeping (why this comparison is legal):
  - the packet's exported visual mesh is in the body frame of mjcf body
    '<obj>_object' — the same frame frames.json declares;
  - the pose server loads exactly that mesh and returns FoundationPose's raw
    cam_T_mesh;
  - the packet's cam_T_<obj> is that same body's pose in the same CV camera
    frame (robosuite's extrinsics bake the MuJoCo->CV correction).
  So est ≈ cam_T_<obj> directly, no to_origin games.

The teapot is the pose target (handle + spout kill the symmetry ambiguity a
mug's near-cylinder would introduce). Thresholds are loose-but-meaningful for
clean sim depth: 3 cm / 15 deg.

The mesh must be visible to the server under /opt/meshes, which compose mounts
from ./foundationpose_runtime/meshes — the fixture copies it there from the
packet dir if it's missing (host side, same box).
"""

import shutil

import numpy as np
import pytest

from conftest import REPO_ROOT, rot_err_deg

OBJ = "teapot"
TRANS_TOL = 0.03   # m
ROT_TOL = 15.0     # deg
REG_TIMEOUT = 300_000  # first register pays model warmup


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
def registered(pose, packet, mesh_name):
    rep = pose.ok(
        {"cmd": "register", "obj": f"{OBJ}_test", "mesh": mesh_name,
         "rgb": packet["rgb"],
         "depth": packet["depth"].astype(np.float32),
         "K": packet["K"].astype(np.float32),
         "mask": packet[f"mask_{OBJ}"].astype(np.uint8),
         "est_refine_iter": 5},
        timeout_ms=REG_TIMEOUT)
    yield np.asarray(rep["pose"], np.float64)
    pose.ok({"cmd": "release", "obj": f"{OBJ}_test"})


def test_ping(pose):
    assert pose.ok({"cmd": "ping"})["ok"]


def test_register_matches_ground_truth(registered, packet):
    gt = packet[f"cam_T_{OBJ}"]
    est = registered
    t_err = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
    r_err = rot_err_deg(est[:3, :3], gt[:3, :3])
    print(f"[foundationpose] register: trans err {t_err*1000:.1f} mm, "
          f"rot err {r_err:.2f} deg")
    assert t_err < TRANS_TOL, f"translation error {t_err*1000:.1f} mm"
    assert r_err < ROT_TOL, f"rotation error {r_err:.1f} deg"
    # pose is a proper rigid transform
    R = est[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)
    # object is in front of the camera at a plausible tabletop range
    assert 0.2 < est[2, 3] < 3.0


def test_track_is_consistent_on_same_frame(pose, packet, registered):
    """Tracking the registration frame itself must stay put — this is the
    frame-to-frame path the /pose_bridge node runs at camera rate."""
    rep = pose.ok({"cmd": "track", "obj": f"{OBJ}_test",
                   "rgb": packet["rgb"],
                   "depth": packet["depth"].astype(np.float32),
                   "K": packet["K"].astype(np.float32),
                   "track_refine_iter": 2},
                  timeout_ms=60_000)
    tracked = np.asarray(rep["pose"], np.float64)
    dt = float(np.linalg.norm(tracked[:3, 3] - registered[:3, 3]))
    dr = rot_err_deg(tracked[:3, :3], registered[:3, :3])
    print(f"[foundationpose] track drift on static frame: {dt*1000:.2f} mm, {dr:.2f} deg")
    assert dt < 0.01 and dr < 5.0


def test_track_unregistered_object_errors_cleanly(pose, packet):
    rep = pose.call({"cmd": "track", "obj": "never_registered",
                     "rgb": packet["rgb"],
                     "depth": packet["depth"].astype(np.float32),
                     "K": packet["K"].astype(np.float32)})
    assert rep.get("ok") is False and rep.get("error")


def test_release_then_track_errors(pose, packet, mesh_name):
    pose.ok({"cmd": "register", "obj": "ephemeral", "mesh": mesh_name,
             "rgb": packet["rgb"], "depth": packet["depth"].astype(np.float32),
             "K": packet["K"].astype(np.float32),
             "mask": packet[f"mask_{OBJ}"].astype(np.uint8)},
            timeout_ms=REG_TIMEOUT)
    pose.ok({"cmd": "release", "obj": "ephemeral"})
    rep = pose.call({"cmd": "track", "obj": "ephemeral",
                     "rgb": packet["rgb"],
                     "depth": packet["depth"].astype(np.float32),
                     "K": packet["K"].astype(np.float32)})
    assert rep.get("ok") is False
