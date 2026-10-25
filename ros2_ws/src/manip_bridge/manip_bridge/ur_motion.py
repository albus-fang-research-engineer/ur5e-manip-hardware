"""Joint-space motion pieces for the UR5e executor: straight-line tool moves by
IK, time parameterisation, and the checks a trajectory must pass before it
is sent. Pure numpy (+ manip_cbirrt's DH chain); no ROS.

    chain()                     manip_cbirrt UR5e DH chain in base_link with
                                e = tool0 -- the planner's own inner-loop
                                kinematics (cbirrt_backend.dh_chain), validated
                                against cuRobo FK to 2e-7 m (test_cbirrt_frames)
    line(ch, q0, p_goal)        tool0 along a straight line to p_goal at fixed
                                orientation, damped least-squares IK every
                                `step` metres from the previous solution; refuses
                                a non-converged step, a joint jump (branch flip,
                                singularity) or a joint-limit violation
    retime(Q, ...)              trapezoidal speed profile along the path,
                                every joint under v_max and a_max, at least
                                min_duration long; returns times and velocities
    check_start(q_now, q0)      refuses a trajectory whose first point is not
                                where the robot is (a stale plan)

Joint order: everything here is UR DH order (DH_ORDER, = ur_description's
names); to_dh / from_dh convert named vectors.

The DH chain is the NOMINAL UR5e, while the laptop's driver runs the robot's
factory-calibrated kinematics. The difference is millimetres: harmless for a
5 cm blind line, and the reason the executor checks arrival in joint space
(what the driver commands) rather than in Cartesian space.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

DH_ORDER = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
UR5E_VMAX = np.full(6, np.pi)            # rad/s, every UR5e joint (180 deg/s)
UR5E_AMAX = np.full(6, np.pi)            # rad/s^2, a conservative scale for slow runs
T_BASE_LINK_FROM_DH = np.diag([-1.0, -1.0, 1.0, 1.0])   # pi about z (cbirrt_backend)


class MotionRefusal(Exception):
    """A motion the executor will not send; the message says why."""


def chain():
    from manip_cbirrt import ur5e_chain
    return ur5e_chain(T_base=T_BASE_LINK_FROM_DH, T_tool=np.eye(4))


def to_dh(q, names) -> np.ndarray:
    q, names = np.asarray(q, float).reshape(-1), list(names)
    missing = [n for n in DH_ORDER if n not in names]
    if missing:
        raise MotionRefusal(f"joint vector lacks {missing} (has {names})")
    return np.array([q[names.index(n)] for n in DH_ORDER])


def to_dh_path(Q, names) -> np.ndarray:
    Q = np.asarray(Q, float)
    idx = [list(names).index(n) for n in DH_ORDER]
    return Q[:, idx]


def _rot_err(R_goal, R) -> np.ndarray:
    """World-frame rotation vector taking R to R_goal (matches Jr's frame)."""
    return Rotation.from_matrix(R_goal @ R.T).as_rotvec()


def ik_to(ch, q_seed, T_goal, tol_p=1e-4, tol_r=1e-3, damping=0.01, iters=100):
    """Damped least-squares IK from q_seed; -> (q, pos_err, rot_err)."""
    q = np.asarray(q_seed, float).copy()
    for _ in range(iters):
        T = ch.fk(q)
        e = np.r_[T_goal[:3, 3] - T[:3, 3], _rot_err(T_goal[:3, :3], T[:3, :3])]
        ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if ep < tol_p and er < tol_r:
            return q, ep, er
        Jp, Jr = ch.jacobian(q)
        J = np.vstack([Jp, Jr])
        q = q + J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
    T = ch.fk(q)
    return (q, float(np.linalg.norm(T_goal[:3, 3] - T[:3, 3])),
            float(np.linalg.norm(_rot_err(T_goal[:3, :3], T[:3, :3]))))


def line(ch, q0, p_goal, step=0.005, max_joint_step=0.15, tol_p=1e-4, tol_r=1e-3):
    """Joint path (K+1, 6), first row q0, moving tool0 in a straight line from
    fk(q0) to p_goal (base_link) with fk(q0)'s orientation held."""
    q0 = np.asarray(q0, float)
    T0 = ch.fk(q0)
    p0, p_goal = T0[:3, 3], np.asarray(p_goal, float)
    L = float(np.linalg.norm(p_goal - p0))
    n = max(1, int(np.ceil(L / step - 1e-9)))
    Q, worst = [q0], {"pos_err": 0.0, "rot_err": 0.0, "joint_step": 0.0}
    lo, hi = ch.joint_range[:, 0], ch.joint_range[:, 1]
    for k in range(1, n + 1):
        T = T0.copy()
        T[:3, 3] = p0 + (p_goal - p0) * k / n
        q, ep, er = ik_to(ch, Q[-1], T, tol_p, tol_r)
        if ep > tol_p or er > tol_r:
            raise MotionRefusal(f"line IK did not converge at {k}/{n} "
                                f"(pos {ep*1e3:.2f} mm, rot {np.degrees(er):.2f} deg)")
        jump = float(np.max(np.abs(q - Q[-1])))
        if jump > max_joint_step:
            raise MotionRefusal(f"line step {k}/{n} needs a {jump:.3f} rad joint jump "
                                f"(> {max_joint_step}): branch flip or near-singular")
        if np.any(q < lo) or np.any(q > hi):
            raise MotionRefusal(f"line step {k}/{n} leaves the joint limits")
        worst = {"pos_err": max(worst["pos_err"], ep), "rot_err": max(worst["rot_err"], er),
                 "joint_step": max(worst["joint_step"], jump)}
        Q.append(q)
    return np.array(Q), {"length_m": L, "n_steps": n, **worst}


def retime(Q, v_max=UR5E_VMAX, a_max=UR5E_AMAX, min_duration=0.0):
    """Times t (K,) from 0 and velocities V (K, 6) for the joint path Q (K, 6):
    a trapezoidal profile in the path parameter s (cumulative max_i |dq_i| /
    v_max_i), so no joint exceeds v_max; acceleration bounded by a_max along
    the path (curvature terms ignored -- dense paths, slow runs). Stretched
    uniformly to min_duration if shorter. V is the finite-difference
    derivative, zero at both ends."""
    Q = np.asarray(Q, float)
    v_max, a_max = np.broadcast_to(v_max, (6,)), np.broadcast_to(a_max, (6,))
    if len(Q) < 2:
        raise MotionRefusal("a trajectory needs at least two points")
    ds = np.max(np.abs(np.diff(Q, axis=0)) / v_max, axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    L = float(s[-1])
    if L <= 0:
        raise MotionRefusal("zero-length trajectory")
    a = float(np.min(a_max / v_max))                 # path acceleration bound (1/s^2)
    v = min(1.0, np.sqrt(a * L))                      # peak path speed (<= 1: joints at v_max)
    t_a = v / a
    s_a = 0.5 * a * t_a ** 2
    T = 2 * t_a + (L - 2 * s_a) / v

    def t_of(x):
        if x <= s_a:
            return np.sqrt(2 * x / a)
        if x <= L - s_a:
            return t_a + (x - s_a) / v
        return T - np.sqrt(max(2 * (L - x) / a, 0.0))
    t = np.array([t_of(x) for x in s])
    if min_duration > T:
        t *= min_duration / T
    t = np.maximum.accumulate(t)
    # strictly increasing for the controller
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-4
    V = np.gradient(Q, t, axis=0)
    V[0] = V[-1] = 0.0
    return t, V


def check_start(q_now_dh, q0_dh, tol=0.01):
    d = float(np.max(np.abs(np.asarray(q_now_dh) - np.asarray(q0_dh))))
    if d > tol:
        raise MotionRefusal(f"the robot is {d:.3f} rad from the trajectory's start (> {tol}): "
                            "stale plan -- re-plan from the current joint state")
    return d


def tool_path(ch, Q) -> np.ndarray:
    return np.array([ch.fk(q)[:3, 3] for q in np.asarray(Q)])
