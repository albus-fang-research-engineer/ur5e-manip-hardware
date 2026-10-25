"""Held-object depth mask: drop the pixels that land on the object in the
gripper before TSDF integration, so the map never sees it.

ReKep's approach (it masks the grasped object out of a live nvblox map before
integrating; it needs a 2D tracker for that, we have the object's spheres and
the arm's FK, so the mask is geometric). Same rule as cuRobo's RobotSegmenter
applies to the arm: a pixel is dropped when its backprojected point is within
`threshold` of a sphere's SURFACE, and a dropped pixel is depth 0 -- no
observation, not free space.

Pure numpy, no cuRobo: the server supplies T_base_tool0 (FK at the frame's q)
and the camera pose; this file only does the geometry.

    held_mask(depth, K, T_base_cam, spheres_base, threshold) -> bool (H, W)
    spheres_to_base(spheres_tool0, T_base_tool0)              -> (N, 4)
"""
from __future__ import annotations

import numpy as np


def spheres_to_base(spheres_tool0, T_base_tool0) -> np.ndarray:
    S = np.asarray(spheres_tool0, np.float64).reshape(-1, 4)
    T = np.asarray(T_base_tool0, np.float64).reshape(4, 4)
    out = S.copy()
    out[:, :3] = S[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out


def held_mask(depth, K, T_base_cam, spheres_base, threshold: float) -> np.ndarray:
    """depth (H, W) metres (0 = invalid), K (3, 3), T_base_cam (4, 4) camera in
    base, spheres_base (N, 4). True where a valid pixel's point is within
    `threshold` of any sphere surface."""
    D = np.asarray(depth, np.float32)
    if D.ndim == 3:
        D = D.reshape(D.shape[-2:])
    K = np.asarray(K, np.float64).reshape(3, 3)
    S = np.asarray(spheres_base, np.float64).reshape(-1, 4)
    out = np.zeros(D.shape, bool)
    if len(S) == 0:
        return out
    # sphere centres into the camera frame (N transforms, not H*W)
    T_cb = np.linalg.inv(np.asarray(T_base_cam, np.float64).reshape(4, 4))
    C = (S[:, :3] @ T_cb[:3, :3].T + T_cb[:3, 3]).astype(np.float32)
    reach = (S[:, 3] + float(threshold)).astype(np.float32)
    H, W = D.shape
    v, u = np.nonzero(D > 0)
    z = D[v, u]
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1).astype(np.float32)
    hit = np.zeros(len(P), bool)
    for c, r in zip(C, reach):
        hit |= np.einsum("ij,ij->i", P - c, P - c) < r * r
    out[v[hit], u[hit]] = True
    return out
