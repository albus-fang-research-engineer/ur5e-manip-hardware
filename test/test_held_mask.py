"""curobo_server/held_mask.py -- the held-object depth mask's geometry, offline.

A synthetic depth image (non-square, so a u/v swap cannot pass) of a table
plane with a ball on it, seen by a tilted camera at a known pose in base. The
ball is "held": its spheres are given in tool0 at a known T_base_tool0, as the
sidecar's hold receives them. The mask must drop every ball pixel, keep the
table beyond the threshold, follow tool0 when it moves, and use the same
backprojection convention as the robot mask (x right = u, y down = v, z =
depth along the optical axis).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curobo_server"))
import held_mask as hm  # noqa: E402

H, W = 240, 320
K = np.array([[280.0, 0, 165.0], [0, 285.0, 118.0], [0, 0, 1]])
BALL_C = np.array([0.55, 0.10, 0.04])      # resting on the table z = 0
BALL_R = 0.04
THR = 0.06


def _T(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def camera():
    """Camera 0.6 m above and behind the ball, looking at it (optical z)."""
    eye = np.array([0.15, -0.25, 0.55])
    z = BALL_C - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def render(T_bc, balls=((BALL_C, BALL_R),)):
    """depth (H, W) of the plane z = 0 plus balls; returns (depth, is_ball)."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    d = d_cam @ T_bc[:3, :3].T                 # ray direction in base (t = depth)
    o = T_bc[:3, 3]
    t_plane = np.where(d[..., 2] < 0, -o[2] / d[..., 2], np.inf)
    depth, is_ball = t_plane.copy(), np.zeros((H, W), bool)
    for c, r in balls:
        oc = o - c
        b = d @ oc
        a = np.einsum("...i,...i", d, d)
        disc = b * b - a * (oc @ oc - r * r)
        t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
        closer = (t > 0) & (t < depth)
        depth[closer], is_ball[closer] = t[closer], True
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32), is_ball


def points_base(depth, T_bc):
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)
    return P @ T_bc[:3, :3].T + T_bc[:3, 3]


# tool0 somewhere above the ball, rotated; the held sphere sits at the ball's
# centre, slightly smaller than the ball (an inside fit, like MORPHIT/VOXEL)
T_BT = _T([0.3, -2.6, 0.4], [0.52, 0.12, 0.21])
S_TOOL0 = np.array([[*(np.linalg.inv(T_BT) @ np.r_[BALL_C, 1])[:3], 0.03]])


def test_spheres_to_base_round_trip():
    S = hm.spheres_to_base(S_TOOL0, T_BT)
    np.testing.assert_allclose(S[0, :3], BALL_C, atol=1e-12)
    assert S[0, 3] == 0.03


def test_mask_drops_the_held_ball_and_keeps_the_far_table():
    T_bc = camera()
    depth, is_ball = render(T_bc)
    assert is_ball.sum() > 300 and (depth > 0).sum() > 0.9 * H * W
    m = hm.held_mask(depth, K, T_bc, hm.spheres_to_base(S_TOOL0, T_BT), THR)
    assert m[is_ball].all()                                   # the whole ball goes
    P = points_base(depth, T_bc)
    dist = np.linalg.norm(P - BALL_C, axis=-1) - 0.03          # to the sphere's surface
    table = ~is_ball & (depth > 0)
    assert not m[table & (dist > THR + 1e-3)].any()           # far table stays
    assert m[table & (dist < THR - 1e-3)].all()               # the 6 cm band around it goes
    # (H, W) and (1, H, W) depth are the same image
    np.testing.assert_array_equal(
        m, hm.held_mask(depth[None], K, T_bc, hm.spheres_to_base(S_TOOL0, T_BT), THR))


def test_mask_follows_tool0():
    T_bc = camera()
    moved = BALL_C + [0.0, 0.25, 0.0]
    depth, is_ball = render(T_bc, balls=((BALL_C, BALL_R), (moved, BALL_R)))
    T_shift = np.eye(4)
    T_shift[:3, 3] = [0.0, 0.25, 0.0]
    m = hm.held_mask(depth, K, T_bc, hm.spheres_to_base(S_TOOL0, T_shift @ T_BT), THR)
    P = points_base(depth, T_bc)
    on_moved = is_ball & (np.linalg.norm(P - moved, axis=-1) < BALL_R + 1e-3)
    on_orig = is_ball & ~on_moved
    assert m[on_moved].all() and not m[on_orig].any()


def test_invalid_pixels_and_no_spheres():
    T_bc = camera()
    depth, _ = render(T_bc)
    depth[:20] = 0.0
    m = hm.held_mask(depth, K, T_bc, hm.spheres_to_base(S_TOOL0, T_BT), 10.0)
    assert not m[:20].any() and m[20:].all()                  # depth 0 is never "held"
    assert not hm.held_mask(depth, K, T_bc, np.zeros((0, 4)), THR).any()


def test_uv_convention_matches_a_known_pixel():
    """A tiny sphere at the point seen at pixel (row v=60, column u=250)
    masks that pixel and a handful around it, nothing at (v=180, u=60)."""
    T_bc = camera()
    depth, _ = render(T_bc, balls=())
    P = points_base(depth, T_bc)
    S = np.array([[*P[60, 250], 0.005]])
    m = hm.held_mask(depth, K, T_bc, S, 0.002)
    assert m[60, 250] and m.sum() < 20
    assert not m[180, 60]


@pytest.mark.parametrize("n", [1, 16])
def test_cost_is_fine_at_vga(n):
    import time
    Kv = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    depth = np.full((480, 640), 0.8, np.float32)
    S = np.c_[np.random.default_rng(0).uniform(-0.2, 0.2, (n, 3)) + [0, 0, 0.8], np.full(n, 0.03)]
    t0 = time.time()
    hm.held_mask(depth, Kv, np.eye(4), S, THR)
    assert time.time() - t0 < 1.0                             # 2 Hz integrate budget
