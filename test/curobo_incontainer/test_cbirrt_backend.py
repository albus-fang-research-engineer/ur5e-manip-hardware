"""cuRobo backend for manip_cbirrt, validated against geometric ground truth.

    collision   a Cuboid obstacle in a SceneCfg; ground truth per config is
                cuRobo's own robot spheres (compute_kinematics().robot_spheres)
                tested against the box analytically. The backend's distance
                SIGN is established from that (which side do colliding configs
                fall on?) and its in_collision must agree with the geometry.
    IK          batched, seeded solve on tool0 poses taken from the DH chain's
                FK: solutions must reproduce the pose and stay in the seed's
                branch.
    plan        a real plan_constrained run: DH chain + cuRobo collision, an
                upright path TSR, start/goal near the working posture, the box
                in the way. Every waypoint must be geometrically collision-free
                and on the manifold. Prints wall time and collision-call count
                (the number Bit 5's batching will drive down).

Runs inside the curobo container (curobo + manip_tsr + manip_cbirrt on
PYTHONPATH via the compose mounts):

    docker compose run --rm -v $PWD/test:/opt/test curobo \\
        bash -lc "pip install -q pytest scipy && \\
                  python -m pytest /opt/test/curobo_incontainer/test_cbirrt_backend.py -v -s"
"""

import sys
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
pytest.importorskip("manip_cbirrt", reason="mount ../manip-cbirrt at /opt/manip-cbirrt (compose)")

sys.path.insert(0, "/opt/robot_builder")
sys.path.insert(0, "/opt/curobo_server")
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402
from curobo.types import JointState  # noqa: E402
from ur5e_curobo_config import build_ur5e_config  # noqa: E402

from manip_tsr import FREE_ROT, FREE_TRANS, TSR, bounds, make_pose  # noqa: E402
from manip_cbirrt import AttachedObject, plan_constrained, project_config  # noqa: E402
from cbirrt_backend import (DH_JOINT_ORDER, CuroboCollision, CuroboIK,  # noqa: E402
                            dh_chain, make_kinematics)

if not torch.cuda.is_available():
    pytest.skip("cuRobo needs CUDA", allow_module_level=True)

# a box in the workspace, base_link frame, axis-aligned
BOX_CENTER = np.array([0.45, 0.0, 0.30])
BOX_DIMS = np.array([0.30, 0.30, 0.30])
Q_NOMINAL = np.array([0.0, -1.2, 1.5, -1.9, -1.57, 0.0])      # DH order, elbow-up
TOL = 2e-3


@pytest.fixture(scope="module")
def robot_cfg():
    return build_ur5e_config()


@pytest.fixture(scope="module")
def scene():
    pose = BOX_CENTER.tolist() + [1.0, 0.0, 0.0, 0.0]            # x y z qw qx qy qz
    return Scene(cuboid=[Cuboid(name="box", pose=pose, dims=BOX_DIMS.tolist())])


@pytest.fixture(scope="module")
def kin_curobo(robot_cfg):
    return Kinematics(KinematicsCfg.from_data_dict(robot_cfg["kinematics"]))


@pytest.fixture(scope="module")
def collision(robot_cfg, scene):
    return CuroboCollision(robot_cfg, scene)


# ------------------------------------------------------------ ground truth
def spheres_at(kin_curobo, q_dh):
    names = list(kin_curobo.joint_names)
    q = torch.zeros((1, len(names)), device="cuda", dtype=torch.float32)
    for i, n in enumerate(DH_JOINT_ORDER):
        q[0, names.index(n)] = float(q_dh[i])
    st = kin_curobo.compute_kinematics(JointState.from_position(q, joint_names=names))
    s = st.robot_spheres.detach().reshape(-1, 4).cpu().numpy()
    return s[s[:, 3] > 0]                       # unused attached_object slots have r <= 0


def box_penetration(spheres):
    """max over spheres of (r - distance(center, box)); > 0 means some sphere
    intersects the box."""
    d = np.abs(spheres[:, :3] - BOX_CENTER) - BOX_DIMS / 2
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(d.max(axis=1), 0.0)          # negative when centre is inside
    dist = outside + inside
    return float(np.max(spheres[:, 3] - dist))


def random_configs(rng, n):
    near = Q_NOMINAL + rng.uniform(-1.0, 1.0, (n // 2, 6))
    wide = rng.uniform(-np.pi, np.pi, (n - n // 2, 6))
    return np.vstack([near, wide])


# --------------------------------------------------------------- collision
def test_collision_sign_and_agreement(kin_curobo, collision):
    rng = np.random.default_rng(0)
    Q = random_configs(rng, 200)
    pen = np.array([box_penetration(spheres_at(kin_curobo, q)) for q in Q])
    hit, free = pen > 0.01, pen < -0.03            # clear cases only
    print(f"\n  ground truth: {hit.sum()} configs hit the box, {free.sum()} clearly free "
          f"(of {len(Q)})")
    assert hit.sum() >= 10 and free.sum() >= 10, "move BOX_CENTER: need both classes"

    d_scene, d_self = collision.distances(Q)
    eta = collision.activation
    mx = d_scene.max(axis=1)
    print(f"  raw shapes: scene {d_scene.shape} self {d_self.shape}; eta = {eta}")
    print(f"  worst-sphere cost  hit: median {np.median(mx[hit]):+.4f} "
          f"[{mx[hit].min():+.4f}, {mx[hit].max():+.4f}]   free: median {np.median(mx[free]):+.4f} "
          f"[{mx[free].min():+.4f}, {mx[free].max():+.4f}]")

    # the kernel formula (wp_collision_common.apply_collision_activation with
    # radius_adjusted = r + eta): contact => cost = pen + eta/2; in-band =>
    # 0.5 (pen + eta)^2 / eta; beyond the band => 0. Same spheres, same box.
    def expected(pen):
        d = pen + eta
        return np.where(d <= 0, 0.0, np.where(d <= eta, 0.5 * d * d / eta, d - 0.5 * eta))
    err = mx - expected(pen)
    print(f"  cost - formula(pen_gt): median {np.median(np.abs(err)):.5f}, max {np.abs(err).max():.5f} m")
    print("  sample (pen_gt, cost, formula):",
          [(round(float(a), 4), round(float(b), 4), round(float(c), 4))
           for a, b, c in list(zip(pen, mx, expected(pen)))[:6]])
    assert np.abs(err).max() < 3e-3, "per-sphere cost does not follow the kernel formula"

    near = np.zeros(len(Q), bool); near[:100] = True
    self_pen = d_self.max(axis=1)
    print(f"  self-collision penetration > 0: near-nominal {np.mean(self_pen[near] > 0):.2f}, "
          f"wide-random {np.mean(self_pen[~near] > 0):.2f}; max {self_pen.max():.4f} m")

    flagged = collision.in_collision_batch(Q)
    assert flagged[hit].all(), f"{(~flagged[hit]).sum()} box-hitting configs passed as free"
    # 'free' w.r.t. the box may still self-collide (wide random configs), so
    # demand agreement on the near-nominal half only, where self-collision is rare
    near = np.zeros(len(Q), bool); near[:100] = True
    fp = flagged[free & near].mean()
    print(f"  flagged among box-free near-nominal configs (self-collision only): {fp:.2f}")
    assert fp < 0.25, "too many near-nominal postures self-collide: check the yml's self_collision_buffer/ignore"
    # Q_NOMINAL's wrist sits 3 cm past the box's +x face, so the gripper spheres
    # DO touch it -- agreement with ground truth is the test, not an assumed answer
    pen_nom = box_penetration(spheres_at(kin_curobo, Q_NOMINAL))
    self_nom = collision.signed_penetration(Q_NOMINAL[None])[1][0]
    print(f"  Q_NOMINAL: box penetration {pen_nom:+.4f} m, self {self_nom:+.4f} m, "
          f"in_collision={collision.in_collision(Q_NOMINAL)}")
    assert collision.in_collision(Q_NOMINAL) == (pen_nom > 0 or self_nom > 0)


def test_signed_penetration_and_clearance_margin(kin_curobo, collision):
    """signed_penetration must reproduce geometric penetration wherever the
    cost has not saturated (clearance < eta); a clearance margin m must flag
    exactly the configs with clearance < m."""
    rng = np.random.default_rng(4)
    Q = random_configs(rng, 200)
    pen_gt = np.array([box_penetration(spheres_at(kin_curobo, q)) for q in Q])
    pen, _ = collision.signed_penetration(Q)
    eta = collision.activation
    live = pen_gt > -eta                            # inside the band or penetrating: cost > 0
    err = np.abs(pen[live] - pen_gt[live])
    print(f"\n  signed_penetration vs geometry on {live.sum()} unsaturated configs: "
          f"max err {err.max():.5f} m; saturated configs report {pen[~live].min():+.3f}..{pen[~live].max():+.3f}")
    assert err.max() < 1e-3
    assert np.all(pen[~live] <= -eta + 1e-6)     # beyond the band the cost is 0 -> reports exactly -eta

    for m in (0.0, 0.01, 0.03):
        collision.clearance_margin = m
        flagged = collision.in_collision_batch(Q)
        should = pen_gt > -m                          # closer than m (or penetrating)
        # self-collision can add flags; it cannot remove them
        assert flagged[should].all(), f"m={m}: {(~flagged[should]).sum()} configs within {m} m passed"
        extra = flagged & ~should
        print(f"  margin {m:.2f}: {should.sum()} within margin all flagged; "
              f"{extra.sum()} extra flags (self-collision or box clearance in ({-m:.2f}, ...])")
    collision.clearance_margin = 0.0
    with pytest.raises(ValueError):
        collision.clearance_margin = eta


def test_single_and_batch_agree(collision):
    rng = np.random.default_rng(1)
    Q = random_configs(rng, 16)
    batch = collision.in_collision_batch(Q)
    single = np.array([collision.in_collision(q) for q in Q])
    assert (batch == single).all()


# ---------------------------------------------------------------------- IK
def test_ik_reproduces_pose_and_keeps_branch(robot_cfg, scene, kin_curobo):
    chain = dh_chain()
    ik = CuroboIK(robot_cfg, scene, max_batch=32, num_seeds=32)
    print(f"\n  IK warmup (JIT + graph capture): {ik.warmup_time:.1f}s")
    rng = np.random.default_rng(2)
    # free, near-nominal configs
    Q = Q_NOMINAL + rng.uniform(-0.6, 0.6, (60, 6))
    Q = np.array([q for q in Q if box_penetration(spheres_at(kin_curobo, q)) < -0.02])[:20]
    assert len(Q) >= 10
    T = np.stack([chain.fk(q) for q in Q])
    t0 = time.time()
    res = ik.solve(T, q_seed_dh=Q_NOMINAL)
    dt = time.time() - t0
    print(f"  IK batch of {len(Q)} (steady state): {res.success.sum()} solved in {dt:.2f}s; "
          f"pos err median {np.nanmedian(res.position_error):.4f} "
          f"rot err median {np.nanmedian(res.rotation_error):.4f}")
    assert res.success.mean() >= 0.8
    assert dt < 5.0, "steady-state IK batch should be well under a second; warmup did not take"
    for q_true, q_sol, ok in zip(Q, res.q, res.success):
        if not ok:
            continue
        A = chain.fk(q_sol)
        assert np.linalg.norm(A[:3, 3] - chain.fk(q_true)[:3, 3]) < 0.006
        assert np.linalg.norm(A[:3, :3] - chain.fk(q_true)[:3, :3]) < 0.1
    # branch retention: seeded from Q_NOMINAL, solutions should be its neighbours
    dq = np.linalg.norm(res.q[res.success] - Q[res.success], axis=1)
    print(f"  |q_sol - q_true| median {np.median(dq):.3f} rad, max {dq.max():.3f}")
    assert np.median(dq) < 0.5


# -------------------------------------------------------------------- plan
def test_cbirrt_plan_around_box(robot_cfg, scene, kin_curobo):
    kin, col = make_kinematics(robot_cfg, scene)
    attached = AttachedObject(T_ee_body=make_pose([0.0, 0.0, 0.12]))
    # "Upright" is anchored to the body's orientation AT THE GRASP (sim's
    # transport_pair: w = the body frame frozen at stage entry), not to the
    # world identity. At Q_NOMINAL the gripper points down; a T0_w = I upright
    # TSR would demand body z UP and the projector would obligingly flip the
    # wrist 180 deg into the box. Verified offline: with this w the projection
    # is the identity at both pan-sweep endpoints.
    R_body_nom = attached.body_pose(kin.fk(Q_NOMINAL))[:3, :3]
    upright = TSR(T0_w=make_pose(rot=R_body_nom),
                  Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS,
                            roll=(-0.26, 0.26), pitch=(-0.26, 0.26), yaw=FREE_ROT),
                  name="upright")
    rng = np.random.default_rng(3)

    def on_manifold_and_free(q0):
        q, ok = project_config(kin, attached, [upright], q0, tol=TOL)
        assert ok, "projection onto upright failed at a pan-sweep endpoint"
        assert np.linalg.norm(q - q0) < 1e-6, \
            f"pan-sweep endpoint should already be on the manifold, projection moved it by {np.linalg.norm(q - q0):.3f}"
        assert not kin.in_collision(q) and box_penetration(spheres_at(kin_curobo, q)) < -0.02, \
            "pan-sweep endpoint is not clear of the box"
        return q

    # start/goal differ ONLY in shoulder pan (same elbow/wrist branch by
    # construction); the pan = 0 midpoint puts the wrist inside the box, so the
    # straight joint-space line is blocked and the planner must detour.
    q_lo, q_hi, q_mid = Q_NOMINAL.copy(), Q_NOMINAL.copy(), Q_NOMINAL.copy()
    q_lo[0], q_hi[0] = -1.0, +1.0
    qs, qg = on_manifold_and_free(q_lo), on_manifold_and_free(q_hi)
    assert box_penetration(spheres_at(kin_curobo, q_mid)) > 0.0, "midpoint must be blocked"
    print(f"\n  start tool {kin.fk(qs)[:3, 3].round(3)}  goal tool {kin.fk(qg)[:3, 3].round(3)}  "
          f"(midpoint pan=0 penetrates box by {box_penetration(spheres_at(kin_curobo, q_mid)):+.3f} m)")

    col.n_calls = 0
    t0 = time.time()
    res = plan_constrained(kin, attached, [upright], qs, qg, timeout=180.0, eps=0.12,
                           constraint_tol=TOL, rng=rng)
    dt = time.time() - t0
    print(f"  plan: ok={res.ok} {res.reason} in {dt:.1f}s, {len(res.path) if res.ok else 0} "
          f"waypoints, {col.n_calls} collision calls ({1e3 * dt / max(col.n_calls, 1):.2f} ms/call), "
          f"tree sizes {res.stats.get('tree_sizes')}")
    assert res.ok, res.reason
    assert res.max_excess <= TOL
    pens = [box_penetration(spheres_at(kin_curobo, q)) for q in res.path]
    print(f"  worst box penetration along path (neg = clear): {max(pens):+.4f} m")
    assert max(pens) <= 0.0
    assert not any(kin.in_collision(q) for q in res.path)
