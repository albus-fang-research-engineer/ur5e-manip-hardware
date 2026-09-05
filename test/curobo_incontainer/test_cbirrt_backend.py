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
    print(f"  raw shapes: scene {d_scene.shape} self {d_self.shape} (per sphere / per pair)")
    mx, mn = d_scene.max(axis=1), d_scene.min(axis=1)
    print(f"  scene MAX over spheres  hit: median {np.median(mx[hit]):+.4f} "
          f"[{mx[hit].min():+.4f}, {mx[hit].max():+.4f}]   free: median {np.median(mx[free]):+.4f} "
          f"[{mx[free].min():+.4f}, {mx[free].max():+.4f}]")
    print(f"  scene MIN over spheres  hit: median {np.median(mn[hit]):+.4f} "
          f"[{mn[hit].min():+.4f}, {mn[hit].max():+.4f}]   free: median {np.median(mn[free]):+.4f} "
          f"[{mn[free].min():+.4f}, {mn[free].max():+.4f}]")
    print(f"  self values             all: [{d_self.min():+.4f}, {d_self.max():+.4f}]  "
          f"nonzero fraction {(np.abs(d_self) > 1e-6).mean():.3f}")
    print("  sample (pen_gt, scene max, scene min):",
          [(round(float(a), 3), round(float(b), 4), round(float(c), 4))
           for a, b, c in list(zip(pen, mx, mn))[:8]])

    # Which reading separates the classes? penetration-positive: MAX(hit) >> MAX(free);
    # clearance-positive: MIN(free) >> MIN(hit).
    sep_pen = np.median(mx[hit]) - np.median(mx[free])
    sep_clr = np.median(mn[free]) - np.median(mn[hit])
    pos_pen = sep_pen > sep_clr
    print(f"  separation: penetration-reading {sep_pen:+.4f}, clearance-reading {sep_clr:+.4f} "
          f"=> data says {'PENETRATION-positive' if pos_pen else 'CLEARANCE-positive'}; "
          f"backend assumes {'PENETRATION' if collision.penetration_positive else 'CLEARANCE'} "
          f"(activation {collision.activation})")
    assert max(sep_pen, sep_clr) > 0.005, "neither reading separates hit from free"
    assert pos_pen == collision.penetration_positive, \
        "flip CuroboCollision.penetration_positive default (see cbirrt_backend docstring)"

    flagged = collision.in_collision_batch(Q)
    assert flagged[hit].all(), f"{(~flagged[hit]).sum()} box-hitting configs passed as free"
    # 'free' w.r.t. the box may still self-collide (wide random configs), so
    # demand agreement on the near-nominal half only, where self-collision is rare
    near = np.zeros(len(Q), bool); near[:100] = True
    fp = flagged[free & near].mean()
    print(f"  false-positive rate on clearly-free near-nominal configs: {fp:.2f}")
    assert fp < 0.15
    assert not collision.in_collision(Q_NOMINAL), "the nominal posture must be free"


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
    rng = np.random.default_rng(2)
    # free, near-nominal configs
    Q = Q_NOMINAL + rng.uniform(-0.6, 0.6, (60, 6))
    Q = np.array([q for q in Q if box_penetration(spheres_at(kin_curobo, q)) < -0.02])[:20]
    assert len(Q) >= 10
    T = np.stack([chain.fk(q) for q in Q])
    t0 = time.time()
    res = ik.solve(T, q_seed_dh=Q_NOMINAL)
    print(f"\n  IK batch of {len(Q)}: {res.success.sum()} solved in {time.time() - t0:.2f}s "
          f"(solver {res.solve_time:.3f}s); pos err median {np.nanmedian(res.position_error):.4f} "
          f"rot err median {np.nanmedian(res.rotation_error):.4f}")
    assert res.success.mean() >= 0.8
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
    upright = TSR(T0_w=np.eye(4), Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS,
                                            roll=(-0.26, 0.26), pitch=(-0.26, 0.26), yaw=FREE_ROT),
                  name="upright")
    rng = np.random.default_rng(3)

    def free_on_manifold(q0):
        q, ok = project_config(kin, attached, [upright], q0, tol=TOL)
        return q if ok and not kin.in_collision(q) \
            and box_penetration(spheres_at(kin_curobo, q)) < -0.02 else None

    starts = [free_on_manifold(Q_NOMINAL + rng.uniform(-0.8, 0.8, 6)) for _ in range(60)]
    starts = [q for q in starts if q is not None]
    assert len(starts) >= 2, "no free on-manifold configs near Q_NOMINAL"
    # pick the pair whose tool positions straddle the box in y, so the straight
    # line is blocked and the planner has to go around
    ys = np.array([kin.fk(q)[1, 3] for q in starts])
    qs, qg = starts[int(np.argmin(ys))], starts[int(np.argmax(ys))]
    print(f"\n  start tool {kin.fk(qs)[:3, 3].round(3)}  goal tool {kin.fk(qg)[:3, 3].round(3)}")

    col.n_calls = 0
    t0 = time.time()
    res = plan_constrained(kin, attached, [upright], qs, qg, timeout=120.0, eps=0.12,
                           constraint_tol=TOL, rng=rng)
    dt = time.time() - t0
    print(f"  plan: ok={res.ok} {res.reason} in {dt:.1f}s, {len(res.path) if res.ok else 0} "
          f"waypoints, {col.n_calls} collision calls, tree sizes {res.stats.get('tree_sizes')}")
    assert res.ok, res.reason
    assert res.max_excess <= TOL
    pens = [box_penetration(spheres_at(kin_curobo, q)) for q in res.path]
    print(f"  worst box penetration along path (neg = clear): {max(pens):+.4f} m")
    assert max(pens) <= 0.0
    assert not any(kin.in_collision(q) for q in res.path)
