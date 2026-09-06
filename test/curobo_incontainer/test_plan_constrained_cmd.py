"""`plan_constrained` command, driven through server.handle() with a SceneCfg
override (no mapper / bag needed). The pan-sweep problem from
test_cbirrt_backend, phrased the way a stage will phrase it: a subgoal TSR
around the target body pose, an upright path TSR, a grasp transform, a start
config -- goals are never given.

    docker compose run --rm -v $PWD/test:/opt/test curobo \\
        bash -lc "pip install -q pytest scipy && \\
                  python -m pytest /opt/test/curobo_incontainer/test_plan_constrained_cmd.py -v -s"
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
from ur5e_curobo_config import build_ur5e_config  # noqa: E402

from manip_tsr import FREE_ROT, FREE_TRANS, TSR, bounds, make_pose  # noqa: E402
from manip_cbirrt import AttachedObject  # noqa: E402
from cbirrt_backend import dh_chain  # noqa: E402
import plan_constrained as pc  # noqa: E402
import server  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("cuRobo needs CUDA", allow_module_level=True)

Q_NOMINAL = np.array([0.0, -1.2, 1.5, -1.9, -1.57, 0.0])
# SceneCfg.create wants name-keyed dicts per primitive type (yaml layout)
SCENE = {"cuboid": {"box": {"pose": [0.45, 0.0, 0.30, 1, 0, 0, 0], "dims": [0.3, 0.3, 0.3]}}}
T_EE_BODY = make_pose([0.0, 0.0, 0.12])


def flat(t: TSR) -> dict:
    return {"t0_w": t.T0_w.ravel().tolist(), "tw_e": t.Tw_e.ravel().tolist(),
            "bw": t.Bw.ravel().tolist(), "name": t.name}


@pytest.fixture(scope="module")
def state():
    return {"mapper": None, "cfg": None, "cfg_kwargs": {}, "robot_cfg": build_ur5e_config()}


@pytest.fixture(scope="module")
def problem():
    ch = dh_chain()
    att = AttachedObject(T_ee_body=T_EE_BODY)
    q_lo, q_hi = Q_NOMINAL.copy(), Q_NOMINAL.copy()
    q_lo[0], q_hi[0] = -1.0, +1.0
    R_nom = att.body_pose(ch.fk(Q_NOMINAL))[:3, :3]
    upright = TSR(T0_w=make_pose(rot=R_nom),
                  Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS,
                            roll=(-0.26, 0.26), pitch=(-0.26, 0.26), yaw=FREE_ROT), name="transport/path")
    T_goal_body = att.body_pose(ch.fk(q_hi))
    subgoal = TSR(T0_w=T_goal_body,
                  Bw=bounds(x=(-.01, .01), y=(-.01, .01), z=(-.01, .01),
                            roll=(-.05, .05), pitch=(-.05, .05), yaw=(-.3, .3)), name="transport/subgoal")
    return dict(q_lo=q_lo, q_hi=q_hi, upright=upright, subgoal=subgoal, T_goal_body=T_goal_body)


def base_msg(problem, **kw):
    m = {"cmd": "plan_constrained", "q_start": problem["q_lo"].tolist(),
         "T_ee_body": T_EE_BODY.ravel().tolist(),
         "subgoal": flat(problem["subgoal"]), "path": [flat(problem["upright"])],
         "n_goal_samples": 40, "timeout": 60.0, "eps": 0.12, "seed": 0, "scene": SCENE}
    m.update(kw)
    return m


def test_plan_constrained_succeeds_with_typed_funnel(state, problem):
    t0 = time.time()
    rep = server.handle(base_msg(problem), state)
    dt = time.time() - t0
    f = rep["funnel"]
    print(f"\n  reply: success={rep['success']} reason='{rep['reason']}' in {dt:.1f}s "
          f"(oracle build included on first call)")
    print(f"  funnel: requested {f['n_requested']} -> sampled {f['n_sampled']} (acc {f['acceptance_rate']:.2f}) "
          f"-> IK {f['n_ik']} -> free {f['n_collision_free']} -> contained {f['n_contained']}; "
          f"escalations {f['escalations']}, plan attempts {f['n_plan_attempts']}, funnel {f['funnel_time']:.2f}s")
    assert rep["ok"] and rep["success"], rep["reason"]
    print(f"  path: {len(rep['positions'])} waypoints, max_excess {rep['max_excess']:.4f}, "
          f"trees {rep['tree_sizes']}, {rep['n_collision_calls']} collision calls, goal #{rep['goal_index']}")
    assert f["n_contained"] >= 1 and f["n_ik"] >= f["n_collision_free"] >= f["n_contained"]
    assert rep["max_excess"] <= 2e-3
    P = np.asarray(rep["positions"])
    np.testing.assert_allclose(P[0], problem["q_lo"], atol=1e-6)
    # the reached goal body pose lies in the subgoal TSR (never the emitted pose itself)
    ch = dh_chain(); att = AttachedObject(T_ee_body=T_EE_BODY)
    assert problem["subgoal"].contains(att.body_pose(ch.fk(P[-1])), tol=pc.CONTAINMENT_TOL)
    assert rep["joint_names"] == pc.DH_JOINT_ORDER
    assert len(rep["ee_path"]) == len(rep["body_path"]) == len(P)


def test_no_path_tsr_means_plain_birrt(state, problem):
    """path: [] must plan (unconstrained), not raise in project_config."""
    rep = server.handle(base_msg(problem, path=[], timeout=30.0), state)
    assert rep["success"], rep["reason"]
    assert rep["tsr_names"] == ["transport/subgoal", "path/free"]
    assert rep["max_excess"] == 0.0


def test_joint_name_reorder_is_honoured(state, problem):
    names = ["wrist_3_joint", "shoulder_pan_joint", "elbow_joint", "wrist_1_joint",
             "wrist_2_joint", "shoulder_lift_joint"]
    q = problem["q_lo"]
    q_perm = [q[pc.DH_JOINT_ORDER.index(n)] for n in names]
    rep = server.handle(base_msg(problem, q_start=q_perm, joint_names=names, timeout=30.0), state)
    assert rep["success"], rep["reason"]
    np.testing.assert_allclose(np.asarray(rep["positions"])[0], q, atol=1e-6)


def test_thin_intersection_is_typed(state, problem):
    far = TSR(T0_w=make_pose([0.4, 0.4, 1.4]), Bw=bounds(), name="transport/subgoal")   # pinned, off-manifold
    rep = server.handle(base_msg(problem, subgoal=flat(far), n_goal_samples=20), state)
    print(f"\n  thin: success={rep['success']} reason='{rep['reason']}' stopped={rep['funnel']['stopped']}")
    assert not rep["success"]
    assert rep["funnel"]["stopped"] == "thin_intersection" and "INTERSECT" in rep["reason"]


def test_start_in_collision_is_typed(state, problem):
    rep = server.handle(base_msg(problem, q_start=Q_NOMINAL.tolist()), state)   # wrist in the box
    assert not rep["success"] and rep["reason"].startswith("q_start in collision")


def test_attached_spheres_are_checked(state, problem):
    """A fat sphere attached at the body makes the (otherwise free) start
    config collide with the box; detaching restores it."""
    kin, col, ik = pc.get_oracles(state, {"scene": SCENE})
    q = problem["q_lo"]
    col.detach()
    assert not col.in_collision(q)
    sph = np.array([[0.0, 0.0, 0.12, 0.35]], np.float32)     # centre at the body, r = 35 cm
    slots = col.attach_spheres(sph, q)
    print(f"\n  attached_object slots: {slots}")
    assert slots >= 1
    assert col.in_collision(q), "attached sphere should reach the box"
    col.detach()
    assert not col.in_collision(q)
    # and through the command: a start that is free without the sphere fails typed with it
    rep = server.handle(base_msg(problem, attached_spheres=sph), state)
    assert not rep["success"] and rep["reason"].startswith("q_start in collision")
    assert rep["attached_slots"] >= 1
    rep = server.handle(base_msg(problem), state)          # no spheres -> detached -> plans
    assert rep["success"], rep["reason"]
