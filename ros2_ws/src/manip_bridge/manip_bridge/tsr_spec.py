"""Hand-authored TSR spec (a YAML/JSON mapping) -> manip_tsr.TSR.

The authoring convenience for tsr_from_yaml: poses as xyz + rpy in
degrees, bounds per axis as [lo, hi] / a scalar / "free", angles in
degrees. Pure numpy; the node loads the file and publishes. This is the
HAND arm -- what a compiled TSR emission replaces -- so the spec is kept
deliberately close to the message: same T0_w / Tw_e / Bw / name / frame_id
fields, no object semantics.

    name: grasp/handle
    frame_id: mug                 # frame t0_w is in: object frame, or base_link
    topic: /tsr/mug/grasp         # optional; default /tsr/<frame_id>/grasp
    t0_w:  {xyz: [0, 0.06, 0.05], rpy_deg: [0, 0, 0]}   # or matrix: 16 or 4x4
    tw_e:  {xyz: [0, 0, 0],       rpy_deg: [0, 0, 0]}   # optional, identity
    bw:                           # metres / degrees; "free" = unbounded
      x: [-0.005, 0.005]
      y: [-0.005, 0.005]
      z: [-0.02, 0.02]
      roll: [-5, 5]
      pitch: [-5, 5]
      yaw: free
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from manip_tsr import FREE_ROT, FREE_TRANS, TSR, bounds, make_pose

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def pose_from_spec(spec) -> np.ndarray:
    """{xyz, rpy_deg} | {matrix: 16 or 4x4} | None (identity) -> 4x4.
    rpy_deg is scipy extrinsic 'xyz' (roll about x, then pitch, then yaw),
    the same convention manip_tsr uses for displacements."""
    if spec is None:
        return np.eye(4)
    if "matrix" in spec:
        return np.asarray(spec["matrix"], dtype=float).reshape(4, 4)
    xyz = np.asarray(spec.get("xyz", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    rpy = np.deg2rad(np.asarray(spec.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=float).reshape(3))
    return make_pose(xyz, Rotation.from_euler("xyz", rpy).as_matrix())


def _row(axis: str, v):
    angular = axis in ("roll", "pitch", "yaw")
    if isinstance(v, str):
        if v.lower() == "free":
            return FREE_ROT if angular else FREE_TRANS
        raise ValueError(f"bw.{axis}: unknown keyword {v!r} (only 'free')")
    scale = np.pi / 180.0 if angular else 1.0
    if np.isscalar(v):
        return float(v) * scale
    lo, hi = (float(x) * scale for x in v)
    if hi < lo:
        raise ValueError(f"bw.{axis}: hi < lo ({v})")
    return (lo, hi)


def bounds_from_spec(spec: dict) -> np.ndarray:
    """Per-axis bounds -> 6x2 Bw. Missing axes are pinned at 0 (same as
    manip_tsr.bounds); say so explicitly rather than rely on it."""
    unknown = set(spec) - set(AXES)
    if unknown:
        raise ValueError(f"bw: unknown axes {sorted(unknown)}; expected {AXES}")
    return bounds(**{a: _row(a, spec[a]) for a in AXES if a in spec})


def tsr_from_spec(spec: dict) -> tuple[TSR, str, str]:
    """-> (TSR, frame_id, topic). frame_id is REQUIRED: a TSR without a
    frame is not a TSR."""
    for k in ("name", "frame_id", "t0_w", "bw"):
        if k not in spec:
            raise ValueError(f"tsr spec missing required key {k!r}")
    frame_id = str(spec["frame_id"])
    tsr = TSR(T0_w=pose_from_spec(spec["t0_w"]), Tw_e=pose_from_spec(spec.get("tw_e")),
              Bw=bounds_from_spec(spec["bw"]), name=str(spec["name"]))
    topic = str(spec.get("topic") or f"/tsr/{frame_id}/grasp")
    return tsr, frame_id, topic
