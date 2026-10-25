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
    time_optimal(Q, ...)        for UNTIMED paths (the lines; CBiRRT's at M4):
                                TOPP-RA under per-joint v_max and a_max, at
                                least min_duration long, its spline checked
                                against the path it times
    scale_timed(Q, V, dt)       for TIMED paths (cuRobo's plan): keep the
                                planner's profile, stretch time uniformly until
                                every joint is under v_max and a_max
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


SAMPLE_DT = 0.02                 # s between the points sent to the controller
SPLINE_TOL = 1e-3                # rad, spline vs the straight joint segments between waypoints


def time_optimal(Q, v_max=UR5E_VMAX, a_max=UR5E_AMAX, min_duration=0.0, dt=SAMPLE_DT):
    """TOPP-RA: the time-optimal timing of the UNTIMED joint path Q (K, 6)
    under per-joint velocity and acceleration limits, exact along the path
    (the path-curvature terms included), from rest to rest. Stretched
    uniformly to min_duration if faster. TOPP-RA times a cubic spline through
    the waypoints, so the spline is first checked against the straight joint
    segments it replaces (SPLINE_TOL at every segment midpoint): a spline that
    swings away from the checked path is refused, not executed. Returns
    (t, Q_s, V_s) sampled every dt, first and last points exactly Q's."""
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    Q = np.asarray(Q, float)
    keep = np.r_[True, np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1e-9]
    Q = Q[keep]
    if len(Q) < 2:
        raise MotionRefusal("zero-length trajectory")
    v_max, a_max = np.broadcast_to(v_max, (6,)), np.broadcast_to(a_max, (6,))
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    ss = d / d[-1]
    path = ta.SplineInterpolator(ss, Q)
    mid = 0.5 * (ss[1:] + ss[:-1])
    dev = float(np.max(np.abs(path(mid) - 0.5 * (Q[1:] + Q[:-1]))))
    if dev > SPLINE_TOL:
        raise MotionRefusal(f"the timing spline leaves the joint path by {dev:.4f} rad "
                            f"(> {SPLINE_TOL}): waypoints too sparse for their turns -- densify")
    inst = algo.TOPPRA([constraint.JointVelocityConstraint(np.stack([-v_max, v_max], 1)),
                        constraint.JointAccelerationConstraint(np.stack([-a_max, a_max], 1))],
                       path, parametrizer="ParametrizeConstAccel")
    traj = inst.compute_trajectory(0.0, 0.0)
    if traj is None:
        raise MotionRefusal("TOPP-RA found no feasible timing for this path")
    T = float(traj.duration)
    k = max(1.0, float(min_duration) / T) if T > 0 else 1.0     # the stretch
    n = max(2, int(np.ceil(T * k / dt)) + 1)                     # dt apart AFTER it
    ts = np.linspace(0.0, T, n)
    Qs, Vs = traj(ts), traj(ts, 1)
    Qs[0], Qs[-1] = Q[0], Q[-1]
    Vs[0] = Vs[-1] = 0.0
    return ts * k, Qs, Vs / k


def scale_timed(Q, V, dt, v_max=UR5E_VMAX, a_max=UR5E_AMAX):
    """A trajectory that is ALREADY timed (cuRobo's: positions, velocities,
    uniform dt) slowed by ONE uniform time factor k >= 1 until every joint is
    under v_max and a_max: t = k * i * dt, V / k (accelerations / k^2). The
    planner's profile is kept; it is only stretched, never reshaped. -> (t, V, k)."""
    Q, V = np.asarray(Q, float), np.asarray(V, float)
    v_max, a_max = np.broadcast_to(v_max, (6,)), np.broadcast_to(a_max, (6,))
    if len(Q) < 2 or V.shape != Q.shape or not dt > 0:
        raise MotionRefusal(f"timed trajectory needs matching positions/velocities and dt > 0 "
                            f"(got {Q.shape}, {V.shape}, dt {dt})")
    t0 = np.arange(len(Q)) * float(dt)
    A = np.gradient(V, t0, axis=0)
    k = max(1.0, float(np.max(np.abs(V) / v_max)), float(np.sqrt(np.max(np.abs(A) / a_max))))
    V = V / k
    V[0] = V[-1] = 0.0
    return t0 * k, V, k


def check_start(q_now_dh, q0_dh, tol=0.01):
    d = float(np.max(np.abs(np.asarray(q_now_dh) - np.asarray(q0_dh))))
    if d > tol:
        raise MotionRefusal(f"the robot is {d:.3f} rad from the trajectory's start (> {tol}): "
                            "stale plan -- re-plan from the current joint state")
    return d


def tool_path(ch, Q) -> np.ndarray:
    return np.array([ch.fk(q)[:3, 3] for q in np.asarray(Q)])
