"""cuRoboV2 backends for the manip_cbirrt Kinematics protocol: collision and IK.

The planner's inner loop (fk + jacobian, thousands of calls per plan) runs
on manip_cbirrt.DHChain -- numpy, microseconds. cuRobo answers the two
questions that need the robot's collision geometry and the live scene:

    CuroboCollision   in_collision(q) via RobotSceneCollision on the robot's
                      OWN spheres -- the 276 fitted UR5e + Robotiq 2F-85
                      spheres plus the attached_object slots (292 total), so
                      the gripper is in every check and a grasped object joins
                      once attached. Scene = whatever SceneCfg it was built with
                      (the mapper's live ESDF VoxelGrid on hardware, a Cuboid
                      in tests); update_world() swaps it.
    CuroboIK          batched, seeded IK on tool0 via InverseKinematics; the
                      goal sampler's oracle (sample subgoal INTERSECT path ->
                      IK all samples in one call -> collision filter).
    make_kinematics   DHChain(T_base = Rz(pi), T_tool = I) + CuroboCollision
                      as a CompositeKinematics -- the frames established by
                      test_cbirrt_frames.py: cuRobo's tool0 IS UR DH frame 6,
                      and ur_description's base_link is the DH base yawed pi.

Frames: everything in base_link; e = tool0 (attached_object's parent at
this HEAD). Joint order: the DH chain uses UR's order; cuRobo's
joint_names are matched by NAME, never by position.

DISTANCE CONVENTION (read from curobo/_src/geom/collision/wp_collision_common.py
and wp_collision_kernel.py, and confirmed against geometric ground truth by
test_cbirrt_backend.py). RobotSceneCollision returns PER-SPHERE costs,
(B, 1, n_spheres), summed over obstacles. With eta = activation_distance and
p = a sphere's true penetration (radius - sdf(centre); negative = clearance):

    d = p + eta
    cost = 0                 if d <= 0        clearance >= eta
         = 0.5 d^2 / eta     if 0 < d <= eta  inside the band, not touching
         = d - eta/2         if d > eta       CONTACT: cost = p + eta/2

so a sphere is actually penetrating <=> cost > eta/2, and penetrating by more
than `margin` <=> cost > eta/2 + margin. in_collision reduces with max over
spheres and thresholds there. Self-collision cost is the largest sphere-pair
penetration (positive = overlap), thresholded at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from manip_cbirrt import CompositeKinematics, DHChain, ur5e_chain

DH_JOINT_ORDER = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
T_BASE_LINK_FROM_DH = np.eye(4)
T_BASE_LINK_FROM_DH[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()


def dh_chain() -> DHChain:
    """UR5e chain expressed in base_link with e = tool0 (validated against
    cuRobo FK to 2e-7 m by test_cbirrt_frames.py)."""
    return ur5e_chain(T_base=T_BASE_LINK_FROM_DH, T_tool=np.eye(4))


def _reorder(q_dh: np.ndarray, target_names) -> np.ndarray:
    """(..., 6) in DH order -> (..., 6) in cuRobo's joint order."""
    q_dh = np.asarray(q_dh, dtype=np.float32)
    idx = [DH_JOINT_ORDER.index(n) for n in target_names]
    return np.ascontiguousarray(q_dh[..., idx])      # cuRobo kernels reject non-contiguous input


def _reorder_back(q_curobo: np.ndarray, source_names) -> np.ndarray:
    q_curobo = np.asarray(q_curobo, dtype=np.float64)
    idx = [list(source_names).index(n) for n in DH_JOINT_ORDER]
    return q_curobo[..., idx]


# --------------------------------------------------------------- collision


class CuroboCollision:
    """Kinematics.in_collision backend on cuRobo's robot-scene checker.

    in_collision(q) is the protocol call (one config). in_collision_batch(Q)
    is what a batched ConstrainedExtend will use; it is the same kernel on
    (B, 6) and costs about as much as one call."""

    def __init__(self, robot_cfg: dict, scene, margin: float = 0.0,
                 collision_activation_distance: float = 0.05,
                 max_collision_distance: float = 1.0):
        from curobo._src.collision.collision_robot_scene import (
            RobotSceneCollision, RobotSceneCollisionCfg)

        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config=robot_cfg, scene_model=scene,
            collision_activation_distance=float(collision_activation_distance),
            self_collision_activation_distance=0.0,
            max_collision_distance=float(max_collision_distance))
        self.rsc = RobotSceneCollision(cfg)
        self.joint_names = list(self.rsc.kinematics.joint_names)
        self.margin = float(margin)
        self.activation = float(collision_activation_distance)
        self.n_calls = 0

    @property
    def contact_threshold(self) -> float:
        """Per-sphere cost above which the sphere touches an obstacle."""
        return 0.5 * self.activation

    def update_world(self, scene) -> None:
        self.rsc.update_world(scene)

    def distances(self, Q_dh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, 6) DH-order configs -> (scene (B, S), self (B, K)) raw per-sphere
        costs exactly as cuRobo returns them (see class doc)."""
        Q = np.atleast_2d(np.asarray(Q_dh, dtype=np.float32))
        B = len(Q)
        # cuRobo wants [batch, horizon, dof]; independent configs = horizon 1
        q = torch.as_tensor(_reorder(Q, self.joint_names), dtype=torch.float32,
                            device="cuda").unsqueeze(1).contiguous()
        d_scene, d_self = self.rsc.get_scene_self_collision_distance_from_joints(q)
        self.n_calls += 1
        return (d_scene.detach().float().reshape(B, -1).cpu().numpy(),
                d_self.detach().float().reshape(B, -1).cpu().numpy())

    def penetration(self, Q_dh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B,) worst-sphere scene penetration in metres (negative values are
        NOT clearances -- the cost saturates at 0 beyond eta; treat <= 0 as
        'not touching'), and (B,) worst self-collision penetration."""
        d_scene, d_self = self.distances(Q_dh)
        return d_scene.max(axis=1) - self.contact_threshold, d_self.max(axis=1)

    def in_collision_batch(self, Q_dh: np.ndarray) -> np.ndarray:
        pen_scene, pen_self = self.penetration(Q_dh)
        return (pen_scene > self.margin) | (pen_self > 0.0)

    def in_collision(self, q_dh: np.ndarray) -> bool:
        return bool(self.in_collision_batch(np.asarray(q_dh)[None])[0])

    def __call__(self, q_dh: np.ndarray) -> bool:           # CompositeKinematics.collision
        return self.in_collision(q_dh)


# ---------------------------------------------------------------------- IK


@dataclass
class IKResult:
    q: np.ndarray                 # (B, 6) DH order (NaN rows where !success)
    success: np.ndarray           # (B,) bool
    position_error: np.ndarray    # (B,)
    rotation_error: np.ndarray    # (B,)
    solve_time: float


class CuroboIK:
    """Batched IK on tool0. Built for a fixed max batch (CUDA graphs want
    fixed shapes); solve() pads/truncates to it."""

    def __init__(self, robot_cfg: dict, scene=None, max_batch: int = 64,
                 num_seeds: int = 32, position_tolerance: float = 0.005,
                 orientation_tolerance: float = 0.05, use_cuda_graph: bool = True):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        cfg = InverseKinematicsCfg.create(
            robot=robot_cfg, scene_model=scene, num_seeds=int(num_seeds),
            position_tolerance=float(position_tolerance),
            orientation_tolerance=float(orientation_tolerance),
            max_batch_size=int(max_batch), use_cuda_graph=bool(use_cuda_graph))
        self.ik = InverseKinematics(cfg)
        self.max_batch = int(max_batch)
        self.joint_names = list(self.ik.joint_names)
        self.tool = self.ik.tool_frames[0]
        self.warmup_time = self._warmup()

    def _warmup(self) -> float:
        """First solve pays warp JIT + CUDA-graph capture (~30 s observed);
        do it here, on a reachable dummy pose, so callers see steady-state
        latency."""
        import time
        t0 = time.time()
        T = np.eye(4); T[:3, 3] = [0.45, 0.0, 0.45]
        self.solve(T[None])
        return time.time() - t0

    def update_world(self, scene) -> None:
        self.ik.update_world(scene)

    def solve(self, T_targets: np.ndarray, q_seed_dh: np.ndarray | None = None) -> IKResult:
        """T_targets: (B, 4, 4) tool0 poses in base_link. q_seed_dh: (6,) DH
        order; seeds every problem so solutions stay in the current
        elbow/wrist branch (a lesson from the planner tests: unseeded IK
        lands in branches with no constraint-preserving path from here)."""
        from curobo.types import GoalToolPose, JointState, Pose

        T = np.asarray(T_targets, dtype=np.float64).reshape(-1, 4, 4)
        n = len(T)
        if n > self.max_batch:
            raise ValueError(f"{n} targets > max_batch {self.max_batch}; chunk the call")
        pad = self.max_batch - n
        if pad:
            T = np.concatenate([T, np.repeat(T[-1:], pad, axis=0)], axis=0)
        quat_xyzw = Rotation.from_matrix(T[:, :3, :3]).as_quat()
        quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
        pose = Pose(position=torch.as_tensor(np.ascontiguousarray(T[:, :3, 3]),
                                             dtype=torch.float32, device="cuda"),
                    quaternion=torch.as_tensor(np.ascontiguousarray(quat_wxyz),
                                               dtype=torch.float32, device="cuda"))
        goal = GoalToolPose.from_poses({self.tool: pose},
                                       ordered_tool_frames=list(self.ik.tool_frames),
                                       num_goalset=1)
        current = None
        if q_seed_dh is not None:
            qs = _reorder(np.asarray(q_seed_dh, dtype=np.float32)[None], self.joint_names)
            qs = np.repeat(qs, self.max_batch, axis=0)
            current = JointState.from_position(
                torch.as_tensor(np.ascontiguousarray(qs), dtype=torch.float32, device="cuda"),
                joint_names=self.joint_names)
        res = self.ik.solve_pose(goal, current_state=current, return_seeds=1)

        succ = res.success.detach().reshape(-1).cpu().numpy().astype(bool)[:n]
        sol = res.solution if res.solution is not None else res.js_solution.position
        sol = sol.detach().float().cpu().numpy().reshape(self.max_batch, -1, len(self.joint_names))[:n, 0]
        q = _reorder_back(sol, self.joint_names)
        q[~succ] = np.nan

        def arr(t):
            return (np.full(n, np.nan) if t is None
                    else t.detach().float().reshape(-1).cpu().numpy()[:n])
        return IKResult(q=q, success=succ, position_error=arr(res.position_error),
                        rotation_error=arr(res.rotation_error),
                        solve_time=float(res.solve_time or 0.0))


# ------------------------------------------------------------------ assembly


def make_kinematics(robot_cfg: dict, scene, **collision_kw) -> tuple[CompositeKinematics, CuroboCollision]:
    """The Kinematics the planner plans against: numpy DH chain in base_link
    (e = tool0) + cuRobo collision on the fitted spheres. Returns the
    collision object too so callers can update_world() and read n_calls."""
    col = CuroboCollision(robot_cfg, scene, **collision_kw)
    return CompositeKinematics(dh_chain(), col), col
