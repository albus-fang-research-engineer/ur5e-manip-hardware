"""The scene facts a grasp and a carry need from a run, in base_link: object
poses, the table plane, the box that removes the grasped object from the
cuRobo map, and the carried object's spheres in the tool0 frame.

Pure numpy except object_spheres (one sidecar call, cached). No sim code.

    T_base_obj(summary, key)     T_base_cam . cam_T_obj (pose_on_final), the
                                 composition grasp_tsr and select_scene use
    table_plane_base(summary)    packet.plane (camera frame, n toward the
                                 camera = up, n.p + d = 0) moved to base
    clear_box(V_body, T, plane)  the axis-aligned box (base) that clear_region
                                 takes: the tracked mesh's vertices at the
                                 object's pose, padded by `margin`, the
                                 bottom clipped at the table plane's highest
                                 point under the box + half a voxel, so the
                                 table stays in the map. Clipped ONLY at the
                                 table -- never at another object, so check
                                 conflicts() against the other objects'
                                 meshes: a teapot box reaching into the mug
                                 would delete the mug's voxels, and the mug
                                 is a real obstacle. A conflict is a refusal
                                 (move the objects apart), not a clip.
    T_ee_body(T_tool0, T_obj)    the PLANNED grasp transform tool0 -> body;
                                 the executor replaces it with the measured
                                 one after the gripper closes
    spheres_tool0(S_body, T)     sphere centres moved into tool0, radii kept:
                                 plan_constrained's attached_spheres AND the
                                 sidecar's hold (held-object depth mask) --
                                 one source for both
    object_spheres(...)          the sidecar's fit_spheres on the tracked mesh,
                                 cached per mesh sha256

The clear box comes from the TRACKED mesh, and TRELLIS can undersize thin
parts (the mug's handle covered ~57% of the real one). `margin` is the
lever: after clearing, look at the ESDF around the handle in rviz, and if
voxels of the real handle remain, raise the margin for that run and record
it.

Map ORDER: close -> hold (the sidecar masks the held object out of every
integrated frame; the map stays live) -> blind lift -> clear the pre-grasp
footprint once (the lifted object hides it from the camera, so live depth
cannot) -> carry -> place -> open -> release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VOXEL_M = 0.01            # the mapper's default voxel (CUROBO_VOXEL_SIZE)
CLEAR_MARGIN_M = 0.02     # two voxels around the tracked mesh; raise per run if
                          # real-handle voxels survive (see the docstring)
N_SPHERES = 16            # attached_object slots (ur5e_curobo_config.N_ATTACHED_SPHERES)


def _summary(summary) -> dict:
    return summary if isinstance(summary, dict) else json.loads(Path(summary).read_text())


def T_base_cam(summary) -> np.ndarray:
    s = _summary(summary)
    T = ((s.get("packet") or {}).get("t_base_cam") or {}).get("T")
    if T is None:
        raise ValueError("summary has no packet.t_base_cam.T")
    return np.asarray(T, float).reshape(4, 4)


def T_base_obj(summary, key: str) -> np.ndarray:
    s = _summary(summary)
    po = ((s.get("objects") or {}).get(key) or {}).get("pose_on_final")
    if not po:
        raise ValueError(f"summary has no objects[{key!r}].pose_on_final")
    return T_base_cam(s) @ np.asarray(po["cam_T_obj"], float).reshape(4, 4)


def table_plane_base(summary) -> tuple[np.ndarray, float]:
    """(n, d) in base with n.p + d = 0 and n pointing up (away from the table)."""
    s = _summary(summary)
    pl = (s.get("packet") or {}).get("plane") or {}
    if pl.get("n") is None:
        raise ValueError("summary has no packet.plane")
    T = T_base_cam(s)
    n_c = np.asarray(pl["n"], float)
    k = np.linalg.norm(n_c)
    n_c, d_c = n_c / k, float(pl["d"]) / k          # same plane, unit normal
    n_b = T[:3, :3] @ n_c
    d_b = d_c - float(n_b @ T[:3, 3])
    if n_b[2] < 0:                            # camera-side normal is up for a table
        n_b, d_b = -n_b, -d_b
    return n_b, d_b


def table_z(plane, x, y) -> np.ndarray:
    """Height of the table plane at (x, y) in base."""
    n, d = plane
    return -(n[0] * np.asarray(x) + n[1] * np.asarray(y) + d) / n[2]


def aabb(V_body, T) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(V_body, float) @ T[:3, :3].T + T[:3, 3]
    return W.min(axis=0), W.max(axis=0)


def clear_box(V_body, T_base_object, plane, margin: float = CLEAR_MARGIN_M,
              voxel: float = VOXEL_M) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = aabb(V_body, T_base_object)
    lo, hi = lo - margin, hi + margin
    xs, ys = np.array([lo[0], hi[0], lo[0], hi[0]]), np.array([lo[1], lo[1], hi[1], hi[1]])
    lo[2] = max(lo[2], float(table_z(plane, xs, ys).max()) + 0.5 * voxel)
    if lo[2] >= hi[2]:
        raise ValueError(f"clear box is empty after clipping at the table (bottom {lo[2]:.3f} >= "
                         f"top {hi[2]:.3f}): the object's pose puts it below the table")
    return lo, hi


def overlap_depth(a: tuple, b: tuple) -> np.ndarray:
    """Per-axis overlap (m) of two axis-aligned boxes (lo, hi); a negative
    entry is the gap along that axis. They intersect iff all three > 0."""
    return np.minimum(np.asarray(a[1]), np.asarray(b[1])) - np.maximum(np.asarray(a[0]), np.asarray(b[0]))


def overlap(a: tuple, b: tuple) -> bool:
    return bool(np.all(overlap_depth(a, b) > 0))


def conflicts(box: tuple, others: dict) -> dict:
    """{key: per-axis overlap} for each other object whose mesh AABB at its
    pose (no margin -- its voxels) the clear box reaches into.
    others = {key: (V_body, T_base_obj)}."""
    out = {}
    for k, (V, T) in others.items():
        d = overlap_depth(box, aabb(V, T))
        if np.all(d > 0):
            out[k] = d
    return out


def mesh_path(summary, key: str) -> str:
    """The tracked mesh pose_on_final is a pose of (container path)."""
    po = ((_summary(summary).get("objects") or {}).get(key) or {}).get("pose_on_final") or {}
    if not po.get("mesh"):
        raise ValueError(f"summary objects[{key!r}].pose_on_final has no mesh")
    return po["mesh"]


def T_ee_body(T_base_tool0, T_base_object) -> np.ndarray:
    return np.linalg.inv(np.asarray(T_base_tool0, float)) @ np.asarray(T_base_object, float)


def spheres_tool0(spheres_body, T_tool0_body) -> np.ndarray:
    """(N, 4) body-frame spheres -> tool0 frame (T_tool0_body = T_ee_body)."""
    S = np.asarray(spheres_body, float).reshape(-1, 4)
    T = np.asarray(T_tool0_body, float)
    out = S.copy()
    out[:, :3] = S[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out


def object_spheres(mesh, cache_path, call, num_spheres: int = N_SPHERES) -> np.ndarray:
    """The sidecar's fit_spheres on the tracked mesh, cached at cache_path
    with the mesh's sha256: refit only when the mesh changed. `call(payload)
    -> reply` is the cuRobo sidecar client."""
    import trimesh
    mesh, cache_path = Path(mesh), Path(cache_path)
    sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
    if cache_path.is_file():
        c = json.loads(cache_path.read_text())
        if c.get("mesh_sha256") == sha and c.get("num_spheres") == num_spheres:
            return np.asarray(c["spheres"], float).reshape(-1, 4)
    m = trimesh.load(str(mesh), force="mesh", process=False)
    rep = call({"cmd": "fit_spheres", "vertices": np.asarray(m.vertices, np.float32),
                "faces": np.asarray(m.faces, np.int32), "num_spheres": int(num_spheres)})
    S = np.asarray(rep["spheres"], float).reshape(-1, 4)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "mesh": str(mesh), "mesh_sha256": sha, "num_spheres": num_spheres,
        "fit_type": rep.get("fit_type"), "frame": "the mesh's own (body) frame",
        "spheres": S.tolist()}, indent=1) + "\n")
    return S
