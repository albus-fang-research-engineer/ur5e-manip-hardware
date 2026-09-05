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

so a sphere is actually penetrating <=> cost > eta/2. A CLEARANCE margin m
(flag anything closer than m, 0 <= m < eta) is NOT a linear shift of that
threshold: inside the band the cost is quadratic in clearance, so
"clearance < m" <=> cost > 0.5 (eta - m)^2 / eta (which is eta/2 at m = 0).
in_collision reduces with max over spheres and thresholds there;
signed_penetration() inverts the formula exactly (saturating at -eta).
Self-collision cost is the largest sphere-pair penetration (positive =
overlap), thresholded at 0.

ATTACHED OBJECT. The yml reserves sphere slots on the `attached_object` link
(parent tool0, identity offset). attach_spheres() writes a grasped body's
spheres -- given in the tool0 = e frame, i.e. already composed with
T_ee_body -- into those slots through cuRobo's AttachmentManager, so they
ride with the gripper in every collision check. detach() clears them.
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

    def __init__(self, robot_cfg: dict, scene, clearance_margin: float = 0.0,
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
        self.activation = float(collision_activation_distance)
        self.clearance_margin = float(clearance_margin)
        self.n_calls = 0
        self._attach = None

    @property
    def clearance_margin(self) -> float:
        return self._clearance_margin

    @clearance_margin.setter
    def clearance_margin(self, m: float) -> None:
        if not 0.0 <= m < self.activation:
            raise ValueError(f"clearance_margin must be in [0, activation={self.activation}), got {m}")
        self._clearance_margin = float(m)

    @property
    def contact_threshold(self) -> float:
        """Per-sphere cost above which the sphere touches an obstacle."""
        return 0.5 * self.activation

    @property
    def flag_threshold(self) -> float:
        """Per-sphere cost above which clearance < clearance_margin."""
        d = self.activation - self._clearance_margin
        return 0.5 * d * d / self.activation

    # ---- attached object ------------------------------------------------
    def attach_spheres(self, spheres_e: np.ndarray, q_dh: np.ndarray,
                       link_name: str = "attached_object") -> int:
        """Write a grasped body's spheres (N, 4) = (x, y, z, r) IN THE tool0
        FRAME into the attached_object slots. `q_dh` is only needed by the
        manager for batching; the spheres are taken as link-local (no world
        offset). Returns the number of slots available."""
        from curobo._src.collision.attachment_manager import AttachmentManager
        from curobo.types import JointState
        if self._attach is None:
            self._attach = AttachmentManager(self.rsc.kinematics)
        sph = torch.as_tensor(np.ascontiguousarray(np.asarray(spheres_e, dtype=np.float32).reshape(-1, 4)),
                              dtype=torch.float32, device="cuda")
        q = torch.as_tensor(_reorder(np.asarray(q_dh, dtype=np.float32)[None], self.joint_names),
                            dtype=torch.float32, device="cuda")
        js = JointState.from_position(q, joint_names=self.joint_names)
        self._attach.update(sph, js, link_name=link_name, world_objects_pose_offset=None)
        return int(self._attach.kinematics_params.get_sphere_index_from_link_name(link_name).shape[0])

    def detach(self, link_name: str = "attached_object") -> None:
        if self._attach is not None:
            self._attach.detach(link_name=link_name)

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

    def signed_penetration(self, Q_dh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B,) worst-sphere scene penetration in metres, exact inverse of the
        kernel formula: > 0 penetrating, < 0 clearance, saturating at -eta
        (the cost is 0 beyond the band). And (B,) worst self penetration."""
        d_scene, d_self = self.distances(Q_dh)
        c = d_scene.max(axis=1)
        eta = self.activation
        pen = np.where(c > 0.5 * eta, c - 0.5 * eta,
                       np.where(c > 0.0, np.sqrt(2.0 * eta * np.maximum(c, 0.0)) - eta, -eta))
        return pen, d_self.max(axis=1)

    def in_collision_batch(self, Q_dh: np.ndarray) -> np.ndarray:
        d_scene, d_self = self.distances(Q_dh)
        return (d_scene.max(axis=1) > self.flag_threshold) | (d_self.max(axis=1) > 0.0)

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
    """Batched IK on tool0. Built for a fixed max batch and a fixed goal-buffer
    structure (CUDA graphs cannot be re-captured in this build): solve() pads
    every call to max_batch and always supplies a current_state."""

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
        if n > self.max_batch:                      # chunk transparently
            parts = [self.solve(T[i:i + self.max_batch], q_seed_dh)
                     for i in range(0, n, self.max_batch)]
            return IKResult(q=np.concatenate([p.q for p in parts]),
                            success=np.concatenate([p.success for p in parts]),
                            position_error=np.concatenate([p.position_error for p in parts]),
                            rotation_error=np.concatenate([p.rotation_error for p in parts]),
                            solve_time=sum(p.solve_time for p in parts))
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
        # ALWAYS pass a current_state: GoalManager.update_goal_buffer treats
        # current_js going None -> present as a structural change and resets
        # the CUDA graph, which this build cannot do. Seed = caller's config
        # in cuRobo order, else the solver's retract/default posture.
        if q_seed_dh is not None:
            qs = _reorder(np.asarray(q_seed_dh, dtype=np.float32)[None], self.joint_names)
        else:
            qs = self.ik.default_joint_state.position.detach().float().cpu().numpy().reshape(1, -1)
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
