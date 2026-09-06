"""`plan_constrained` command: TSR pair -> goal funnel -> CBiRRT, in the cuRobo sidecar.

The invariant this module owns: GOAL CONFIGS ARE SAMPLED FROM subgoal INTERSECT
path AND NEVER EMITTED. A caller hands over the two TSRs (plus any extra path
TSRs), the grasp transform and the start config; everything from the body
pose samples to the joint path happens here, and the reply carries the funnel
so a failure is typed (thin intersection vs IK vs collision vs planner) rather
than a bare False. This is ur5e-manip-sim's plan_pour_tea funnel
(_goal_funnel / _sample_funnel_escalating), same constants, on cuRobo oracles.

Request (msgpack; matrices row-major; joints in UR order unless joint_names
is given):
  {"cmd": "plan_constrained",
   "q_start": [6],            "joint_names": [...] optional,
   "T_ee_body": 4x4,          measured grasp transform, e = tool0
   "subgoal": TSR, "path": [TSR, ...],   TSR = {"t0_w":16, "tw_e":16, "bw":12, "name"}
                              all in base_link (the caller roots object-frame
                              TSRs through TF before sending)
   "n_goal_samples": 60,      escalates x2 up to GOAL_SAMPLE_MAX on starvation
   "timeout": 20.0,           CBiRRT wall budget, shared across goal attempts
   "eps": 0.10, "constraint_tol": 2e-3, "clearance_margin": 0.0, "seed": 0,
   "max_goal_attempts": 5,
   "attached_spheres": (N,4) in tool0 frame | omit,   the grasped body's spheres
   "scene": SceneCfg dict | omit (= the mapper's live ESDF)}
             SceneCfg.create layout: {"cuboid": {"<name>": {"pose": [x y z qw qx qy qz],
             "dims": [dx dy dz]}}, ...} -- name-keyed per primitive type.
Reply:
  {"ok": True, "success": bool, "reason": str,
   "joint_names": UR order, "positions": (n,6) float32, "ee_path": (n,3),
   "body_path": (n,3), "max_excess": float, "solve_time": float,
   "funnel": {...counts...}, "tree_sizes": [a,b], "n_collision_calls": int}
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from manip_tsr import FREE_ROT, FREE_TRANS, TSR, bounds, sample_intersection
from manip_cbirrt import AttachedObject, plan_constrained

from cbirrt_backend import DH_JOINT_ORDER, CuroboCollision, CuroboIK, make_ik, make_kinematics

log = logging.getLogger("curobo_server.plan_constrained")

# Same constants as ur5e-manip-sim scripts/plan_pour_tea.py -- not flags: the
# ablation varies n_goal_samples, and a per-run cap would be a hidden variable.
GOAL_SAMPLE_MAX = 480
MIN_ACCEPT_RATE = 0.02
CONTAINMENT_TOL = 5e-3          # achieved-config containment check

# "No path TSR" = unconstrained motion = CBiRRT degenerates to plain BiRRT. The
# planner needs at least one TSR to project onto (max() over an empty list),
# so an all-free region stands in; it constrains nothing and is contained
# everywhere. manip_cbirrt stays verbatim with sim, which always has a path.
FREE_PATH = TSR(T0_w=np.eye(4), Bw=bounds(x=FREE_TRANS, y=FREE_TRANS, z=FREE_TRANS,
                                          roll=FREE_ROT, pitch=FREE_ROT, yaw=FREE_ROT),
                name="path/free")


# ------------------------------------------------------------------ oracles


def _live_scene(state):
    from curobo.scene import Scene
    mapper, cfg = state["mapper"], state["cfg"]
    grid = mapper.compute_esdf(esdf_voxel_size=float(cfg.esdf_voxel_size))
    return Scene(voxel=[grid])


def get_oracles(state, msg):
    """(kin, collision, ik) against the live ESDF (cached; the VoxelGrid is a
    direct reference so compute_esdf refreshes it in place) or against a
    per-request SceneCfg dict (tests; cached by content)."""
    if "scene" in msg and msg["scene"] is not None:
        from curobo.scene import Scene
        key = "cbirrt_scene:" + json.dumps(msg["scene"], sort_keys=True, default=str)
        if state.get(key) is None:
            scene = Scene.create(msg["scene"])
            kin, col = make_kinematics(state["robot_cfg"], scene)
            state[key] = (kin, col, make_ik(state["robot_cfg"], col))   # ONE checker, shared
        return state[key]
    if state.get("cbirrt") is None:
        scene = _live_scene(state)
        kin, col = make_kinematics(state["robot_cfg"], scene)
        state["cbirrt"] = (kin, col, make_ik(state["robot_cfg"], col))    # ONE checker, shared
        log.info("cbirrt oracles built on the live ESDF")
    else:
        _live_scene(state)                          # refresh the aliased buffer
    return state["cbirrt"]


# -------------------------------------------------------------------- funnel


def _tsr(d) -> TSR:
    return TSR(T0_w=np.asarray(d["t0_w"], float).reshape(4, 4),
               Tw_e=np.asarray(d.get("tw_e", np.eye(4).ravel()), float).reshape(4, 4),
               Bw=np.asarray(d["bw"], float).reshape(6, 2), name=str(d.get("name", "")))


def goal_funnel(rep, ik: CuroboIK, kin, col: CuroboCollision, attached, containment,
                q_ref, funnel: dict):
    """sampled body poses -> batched IK -> collision -> containment of the
    ACHIEVED config; sorted nearest q_ref first. Mirrors sim's _goal_funnel."""
    if not rep.accepted:
        return []
    T_ee = np.stack([T @ np.linalg.inv(attached.T_ee_body) for T in rep.accepted])
    res = ik.solve(T_ee, q_seed_dh=q_ref)
    Q = res.q[res.success]
    funnel["n_ik"] += int(res.success.sum())
    if len(Q) == 0:
        return []
    free = ~col.in_collision_batch(Q)
    Q = Q[free]
    funnel["n_collision_free"] += int(free.sum())
    goals = [q for q in Q
             if all(t.contains(attached.body_pose(kin.fk(q)), tol=CONTAINMENT_TOL) for t in containment)]
    funnel["n_contained"] += len(goals)
    goals.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
    return goals


def sample_funnel_escalating(subgoal, paths, n, rng, ik, kin, col, attached, q_ref, funnel):
    """sample_intersection -> goal_funnel, doubling n on starvation up to
    GOAL_SAMPLE_MAX; only re-draws when the intersection itself is healthy."""
    while True:
        rep = sample_intersection(subgoal, paths, n=n, rng=rng)
        funnel["n_requested"] += n
        funnel["n_sampled"] += len(rep.accepted)
        funnel["acceptance_rate"] = float(rep.acceptance_rate)
        funnel["per_constraint_rejections"] = {k: int(v) for k, v in rep.per_constraint_rejections.items()}
        goals = goal_funnel(rep, ik, kin, col, attached, [subgoal] + list(paths), q_ref, funnel)
        if goals or n >= GOAL_SAMPLE_MAX:
            return goals, n
        if rep.acceptance_rate < MIN_ACCEPT_RATE:
            funnel["stopped"] = "thin_intersection"
            return goals, n
        n = min(2 * n, GOAL_SAMPLE_MAX)
        funnel["escalations"] += 1


# ------------------------------------------------------------------- handler


def handle(state, msg) -> dict:
    t_all = time.time()
    if state.get("robot_cfg") is None:
        raise RuntimeError("robot not loaded: send load_robot first")
    kin, col, ik = get_oracles(state, msg)

    names = list(msg.get("joint_names") or DH_JOINT_ORDER)
    q_in = np.asarray(msg["q_start"], float).reshape(-1)
    q_start = np.array([q_in[names.index(n)] for n in DH_JOINT_ORDER])
    attached = AttachedObject(T_ee_body=np.asarray(msg["T_ee_body"], float).reshape(4, 4))
    subgoal = _tsr(msg["subgoal"])
    paths = [_tsr(p) for p in msg.get("path", [])] or [FREE_PATH]
    rng = np.random.default_rng(int(msg.get("seed", 0)))
    col.clearance_margin = float(msg.get("clearance_margin", 0.0))
    col.n_calls = 0

    if msg.get("attached_spheres") is not None:
        n_slots = col.attach_spheres(np.asarray(msg["attached_spheres"], np.float32), q_start)
    else:
        col.detach()
        n_slots = 0

    funnel = dict(n_requested=0, n_sampled=0, acceptance_rate=0.0, n_ik=0,
                  n_collision_free=0, n_contained=0, escalations=0, stopped=None,
                  n_plan_attempts=0, per_constraint_rejections={})
    base = {"ok": True, "joint_names": list(DH_JOINT_ORDER), "funnel": funnel,
            "attached_slots": n_slots, "tsr_names": [subgoal.name] + [p.name for p in paths]}

    if col.in_collision(q_start):
        pen = col.signed_penetration(q_start[None])
        return {**base, "success": False,
                "reason": f"q_start in collision (scene pen {pen[0][0]:+.4f} m, self {pen[1][0]:+.4f} m)",
                "n_collision_calls": col.n_calls}

    t0 = time.time()
    goals, n_used = sample_funnel_escalating(
        subgoal, paths, int(msg.get("n_goal_samples", 60)), rng, ik, kin, col, attached, q_start, funnel)
    funnel["n_used_samples"] = n_used
    funnel["funnel_time"] = time.time() - t0
    log.info("goal funnel: %d sampled -> %d IK -> %d free -> %d contained (%.1fs, acc %.3f)",
             funnel["n_sampled"], funnel["n_ik"], funnel["n_collision_free"],
             funnel["n_contained"], funnel["funnel_time"], funnel["acceptance_rate"])
    if not goals:
        why = ("subgoal INTERSECT path is (nearly) empty: emission inconsistency, not attrition"
               if funnel["stopped"] == "thin_intersection" else
               "no goal survived the funnel (see counts)")
        return {**base, "success": False, "reason": why, "n_collision_calls": col.n_calls}

    timeout = float(msg.get("timeout", 20.0))
    deadline = time.time() + timeout
    last = None
    for gi, q_goal in enumerate(goals[:int(msg.get("max_goal_attempts", 5))]):
        remaining = deadline - time.time()
        if remaining <= 0.5:
            break
        funnel["n_plan_attempts"] += 1
        res = plan_constrained(kin, attached, paths, q_start, q_goal, timeout=remaining,
                               eps=float(msg.get("eps", 0.10)),
                               constraint_tol=float(msg.get("constraint_tol", 2e-3)), rng=rng)
        last = res
        if res.ok:
            path = np.asarray(res.path, np.float32)
            ee = np.stack([kin.fk(q)[:3, 3] for q in res.path]).astype(np.float32)
            body = np.stack([attached.body_pose(kin.fk(q))[:3, 3] for q in res.path]).astype(np.float32)
            return {**base, "success": True, "reason": "",
                    "positions": path, "ee_path": ee, "body_path": body,
                    "goal_index": gi, "max_excess": float(res.max_excess),
                    "solve_time": float(res.solve_time), "total_time": time.time() - t_all,
                    "tree_sizes": list(res.stats.get("tree_sizes", [])),
                    "n_collision_calls": col.n_calls}
    return {**base, "success": False,
            "reason": f"CBiRRT failed on {funnel['n_plan_attempts']} goal(s): "
                      f"{last.reason if last else 'no attempt fit in the time budget'}",
            "tree_sizes": list(last.stats.get("tree_sizes", [])) if last else [],
            "n_collision_calls": col.n_calls, "total_time": time.time() - t_all}
