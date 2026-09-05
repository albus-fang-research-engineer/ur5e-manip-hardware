"""TSR-as-classifier over AnyGrasp proposals. Pure numpy; no rclpy.

Role (the stage-1 contract from ur5e-manip-sim's manip_sim.grasping): the
grasp stage is the one place the TSR is a CLASSIFIER over externally
proposed gripper poses, not a sampler. AnyGrasp proposes; the grasp TSR
keeps what it contains. Containment is binary -- the TSR excess is
identically zero inside the bounds -- so every survivor has
tsr_distance == 0 and the TSR carries no ranking of its own among them.
Survivors therefore keep AnyGrasp's score order; the lookahead probe,
when it exists on hardware, becomes the primary key and the score the
tie-break. TSR distance ranks only the REJECTED set, for the graded
fallback.

Frames
------
AnyGrasp / graspnetAPI (verified against plot_gripper_pro_max source):
    x = approach (fingers extend along +x), y = closing (finger
    separation), z = gripper-plane normal. `t` is the "center of grasp",
    a point on the object surface: the finger boxes run from
    x = -(depth_base + finger_width) = -0.024 to x = +depth around it.
    `t` is NOT the finger root, and grasp_viz's glyph (fingers 0..depth,
    tail at -height) omits the 0.024 m behind t.
e -- the TSR's constrained frame, sim grip-site / Robotiq TCP convention:
    +z = approach, +x = closing, +y = z x x.
The map is the cyclic column permutation e = ag[:, (1, 2, 0)]; cyclic
means even, so det is preserved and a rotation stays a rotation.
Origin: t + pad_offset * approach. `pad_offset` is the fixed distance
from AnyGrasp's surface point to the point between the Robotiq pads that
the TCP frame is defined at -- a gripper constant, not per-grasp (it is
NOT depth/2). Measure it once against tcp_link, alongside the
tool0 -> tcp_link re-expression of the grasp transforms; default 0.

Reference frame: `grasps_from_anygrasp` takes T_ref_cam, the camera pose
in the TSR's frame (base, from TF via the eye-to-base calibration), and
expresses every proposal there. Poses and the TSR must share a frame;
the filter does not check frame_ids -- the node does.

Routes -- reported per attempt (the eval's "which TSR route fired"):
    tsr_contained   >= 1 proposal inside. survivors = the contained
                    ones in score order (desc, stable).
    tsr_distance    none inside. survivors = the nearest rejected with
                    tsr_distance <= max_distance, at most fallback_k,
                    distance order. distance() mixes metres and radians
                    (manip_tsr caveat); max_distance inherits that.
    empty           none inside, none within max_distance. survivors = [].
`rejected` always holds every non-survivor, distance ascending, so the
report can say how far the best AnyGrasp score was from the region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manip_tsr import TSR

# e column j = ag column AG_TO_E[j]:  e_x = ag_y, e_y = ag_z, e_z = ag_x
AG_TO_E: tuple[int, int, int] = (1, 2, 0)
E_TO_AG: tuple[int, int, int] = (2, 0, 1)      # inverse permutation

ROUTE_CONTAINED = "tsr_contained"
ROUTE_DISTANCE = "tsr_distance"
ROUTE_EMPTY = "empty"


# ------------------------------------------------------------------- frames


def anygrasp_to_e(R_ag: np.ndarray, t_ag: np.ndarray,
                  pad_offset: float = 0.0) -> np.ndarray:
    """One AnyGrasp (R, t) -> 4x4 pose of the e frame, same reference frame."""
    R_ag = np.asarray(R_ag, dtype=float).reshape(3, 3)
    t_ag = np.asarray(t_ag, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R_ag[:, AG_TO_E]
    T[:3, 3] = t_ag + pad_offset * R_ag[:, 0]
    return T


def e_to_anygrasp(T_e: np.ndarray,
                  pad_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of anygrasp_to_e (tests, and drawing e-frame grasps with the
    AnyGrasp glyph)."""
    T_e = np.asarray(T_e, dtype=float)
    R_ag = T_e[:3, :3][:, E_TO_AG]
    t_ag = T_e[:3, 3] - pad_offset * R_ag[:, 0]
    return R_ag, t_ag


# ---------------------------------------------------------------- proposals


@dataclass
class Grasp:
    T0_e: np.ndarray            # 4x4 e-frame pose in the TSR's reference frame
    score: float                # AnyGrasp score (already includes 1 - stable_score)
    width: float                # m
    depth: float                # m
    index: int                  # position in the AnyGrasp reply (score-desc order)
    tsr_distance: float = float("nan")


def grasps_from_anygrasp(rotations, translations, scores, widths, depths,
                         T_ref_cam: np.ndarray | None = None,
                         pad_offset: float = 0.0) -> list[Grasp]:
    """AnyGrasp reply arrays (camera frame) -> Grasp list in the TSR's
    reference frame. `T_ref_cam` = camera pose in that frame (identity ->
    poses stay in the camera frame, which is only correct if the TSR is
    expressed there too)."""
    R = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    t = np.asarray(translations, dtype=float).reshape(-1, 3)
    n = len(R)
    if not (len(t) == len(scores) == len(widths) == len(depths) == n):
        raise ValueError(
            f"AnyGrasp arrays disagree: R {len(R)} t {len(t)} scores {len(scores)} "
            f"widths {len(widths)} depths {len(depths)}")
    T_ref_cam = np.eye(4) if T_ref_cam is None else np.asarray(T_ref_cam, dtype=float)
    out: list[Grasp] = []
    for k in range(n):
        T_cam_e = anygrasp_to_e(R[k], t[k], pad_offset)
        out.append(Grasp(T0_e=T_ref_cam @ T_cam_e, score=float(scores[k]),
                         width=float(widths[k]), depth=float(depths[k]), index=k))
    return out


# ------------------------------------------------------------------- filter


@dataclass
class FilterResult:
    route: str
    survivors: list[Grasp]
    rejected: list[Grasp]           # every non-survivor, tsr_distance ascending
    n_in: int
    tol: float
    max_distance: float
    tsr_name: str = ""
    n_contained: int = 0

    def summary(self) -> str:
        best = f"best score {self.survivors[0].score:.3f} (idx {self.survivors[0].index})" \
            if self.survivors else "no survivors"
        near = f", nearest rejected d={self.rejected[0].tsr_distance:.4f}" \
            if self.rejected else ""
        return (f"{self.tsr_name or 'tsr'}: {self.n_in} proposed -> "
                f"{self.n_contained} contained -> {len(self.survivors)} kept "
                f"[{self.route}] {best}{near}")

    def to_dict(self) -> dict:
        """JSON-safe report: per-proposal distance and verdict, in original
        AnyGrasp index order, plus the funnel."""
        kept = {g.index for g in self.survivors}
        allg = sorted(self.survivors + self.rejected, key=lambda g: g.index)
        return {
            "tsr": self.tsr_name,
            "route": self.route,
            "n_in": self.n_in,
            "n_contained": self.n_contained,
            "n_kept": len(self.survivors),
            "tol": self.tol,
            "max_distance": self.max_distance,
            "survivor_indices": [g.index for g in self.survivors],
            "proposals": [{"index": g.index, "score": g.score,
                           "tsr_distance": g.tsr_distance, "kept": g.index in kept}
                          for g in allg],
        }


def filter_grasps(tsr: TSR, grasps: list[Grasp], tol: float = 1e-6,
                  max_distance: float = 0.03, fallback_k: int = 5) -> FilterResult:
    """Classify `grasps` against `tsr` (both in the same reference frame).
    Mutates each Grasp's tsr_distance. See module docstring for routes."""
    for g in grasps:
        g.tsr_distance = float(tsr.distance(g.T0_e))
    by_dist = sorted(grasps, key=lambda g: g.tsr_distance)
    contained = [g for g in grasps if g.tsr_distance <= tol]

    if contained:
        route = ROUTE_CONTAINED
        survivors = sorted(contained, key=lambda g: -g.score)   # stable: ties keep AnyGrasp order
    else:
        near = [g for g in by_dist if g.tsr_distance <= max_distance][:max(fallback_k, 0)]
        route = ROUTE_DISTANCE if near else ROUTE_EMPTY
        survivors = near

    kept = {g.index for g in survivors}
    rejected = [g for g in by_dist if g.index not in kept]
    return FilterResult(route=route, survivors=survivors, rejected=rejected,
                        n_in=len(grasps), tol=tol, max_distance=max_distance,
                        tsr_name=tsr.name, n_contained=len(contained))
