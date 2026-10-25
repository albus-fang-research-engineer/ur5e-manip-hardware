"""outputs/runs/grasp_exec.py -- the grasp-only run's segments and their order,
offline: a synthetic reach plan, a fake arm that lands wherever it is sent,
and the in-process mock gripper. Needs manip_cbirrt
(PYTHONPATH=../manip-cbirrt from the repo root)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "manip_bridge"))
pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
pytest.importorskip("toppra", reason="pip install toppra")
from manip_bridge import ur_motion as um  # noqa: E402

Q0 = np.array([2.8166, -1.2194, 1.0823, -0.1403, 2.4661, 1.0577])
CUROBO_ORDER = [um.DH_ORDER[i] for i in (2, 0, 1, 5, 3, 4)]    # any order, names attached


@pytest.fixture(scope="module")
def ge():
    spec = importlib.util.spec_from_file_location("grasp_exec", REPO / "outputs" / "runs" / "grasp_exec.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ch():
    return um.chain()


def plan_for(ge, ch, standoff=0.05):
    """A reach plan from Q0 to a nearby q_end, with the grasp placed so that
    q_end is exactly its standoff."""
    gr = ge._grasp_reach()
    q_end = Q0 + np.array([0.15, -0.1, 0.12, 0.05, -0.1, 0.2])
    T_goal = ch.fk(q_end)
    fwd = np.eye(4)
    fwd[2, 3] = standoff
    T_e = T_goal @ gr.T_tool0_pad() @ fwd
    idx = [um.DH_ORDER.index(n) for n in CUROBO_ORDER]
    # cuRobo-style timing: smooth (zero end velocity) at dt, fast enough that
    # the 10 % caps must stretch it
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 40))
    P = Q0 + np.outer(s, q_end - Q0)
    dt = 0.02
    V = np.gradient(P, dt, axis=0)
    return {"positions": P[:, idx].tolist(), "velocities": V[:, idx].tolist(), "dt": dt,
            "joint_names": CUROBO_ORDER, "T_base_e": T_e.tolist(),
            "T_base_tool0_goal": T_goal.tolist(), "standoff_m": standoff, "grasp": 0}


def test_build_segments(ge, ch):
    gr = ge._grasp_reach()
    plan = plan_for(ge, ch)
    segs = ge.build(plan, ch)
    assert [s.name for s in segs] == ["open", "reach", "approach", "close", "lift", "hold",
                                      "lower", "release", "retreat"]
    by = {s.name: s for s in segs}
    np.testing.assert_allclose(by["reach"].Q[0], Q0)                     # names honoured
    # approach ends with the pads on the grasp
    T_app_end = ch.fk(by["approach"].Q[-1])
    np.testing.assert_allclose(T_app_end[:3, 3],
                               gr.tool0_goal(np.asarray(plan["T_base_e"]), 0.0)[:3, 3], atol=2e-4)
    assert by["approach"].info["length_m"] == pytest.approx(0.05, abs=1e-3)
    # lift: 5 cm straight up in base z
    dz = ch.fk(by["lift"].Q[-1])[:3, 3] - T_app_end[:3, 3]
    np.testing.assert_allclose(dz, [0, 0, ge.LIFT_M], atol=2e-4)
    # put back = the same paths reversed: same ends, swapped, and the same line
    for back, fwd_ in (("lower", "lift"), ("retreat", "approach")):
        np.testing.assert_array_equal(by[back].Q[0], by[fwd_].Q[-1])
        np.testing.assert_array_equal(by[back].Q[-1], by[fwd_].Q[0])
        P, a0, a1 = by[back].path, by[fwd_].path[0], by[fwd_].path[-1]
        u = (a1 - a0) / np.linalg.norm(a1 - a0)
        assert np.linalg.norm((P - a0) - np.outer((P - a0) @ u, u), axis=1).max() < 3e-4
    # blind lines go at the line speed at most
    for n in ("approach", "lift", "lower", "retreat"):
        s = by[n]
        length = np.linalg.norm(s.path[-1] - s.path[0])
        assert s.t[-1] >= length / ge.LINE_SPEED_M_S - 1e-6
        assert np.abs(s.V).max() <= np.pi * ge.SPEED_FRAC * 1.02
    assert by["lower"].gate is False and by["hold"].gate is True        # one gate before putting back


def test_build_refuses_a_plan_whose_names_lie(ge, ch):
    plan = plan_for(ge, ch)
    plan["joint_names"] = list(um.DH_ORDER)          # positions are in CUROBO_ORDER
    with pytest.raises(ge.ExecRefusal, match="joint order or frame"):
        ge.build(plan, ch)


class FakeArm:
    def __init__(self, q):
        self._q, self.sent = np.asarray(q, float), []

    def q(self):
        return self._q.copy()

    def execute(self, seg):
        self.sent.append(seg.name)
        self._q = seg.Q[-1].copy()
        return True, "ok"


class Gripper(object):
    def __init__(self, ge, grips=True):
        self.mock, self.grips, self.cmds = ge.MockGripper(), grips, []

    def command(self, gap, effort):
        self.cmds.append(gap)
        r = self.mock.command(gap, effort)
        if gap < 0.04 and not self.grips:
            r.update(stalled=False, reached_goal=True, position=0.0)
        return r


def test_run_order_and_outcomes(ge, ch):
    segs = ge.build(plan_for(ge, ch), ch)
    arm, grip, log = FakeArm(Q0), Gripper(ge), []
    assert ge.run(segs, arm, grip, lambda s: True, log) == "done"
    assert arm.sent == ["reach", "approach", "lift", "lower", "retreat"]
    assert grip.cmds == [ge.OPEN_GAP_M, 0.0, ge.OPEN_GAP_M]
    assert [e["segment"] for e in log] == ["open", "reach", "approach", "close", "lift", "hold",
                                           "lower", "release", "retreat"]
    # nothing gripped: open and back out, no lift
    segs = ge.build(plan_for(ge, ch), ch)
    arm, grip = FakeArm(Q0), Gripper(ge, grips=False)
    assert ge.run(segs, arm, grip, lambda s: True, []) == "no object"
    assert arm.sent == ["reach", "approach", "retreat"]
    # a gate says no: nothing after it moves
    segs = ge.build(plan_for(ge, ch), ch)
    arm = FakeArm(Q0)
    assert ge.run(segs, arm, Gripper(ge), lambda s: s.name != "approach", []) == "stopped at approach"
    assert arm.sent == ["reach"]


def test_run_refuses_a_stale_start_unless_mock(ge, ch):
    segs = ge.build(plan_for(ge, ch), ch)
    with pytest.raises(ge.ExecRefusal, match="stale plan"):
        ge.run(segs, FakeArm(Q0 + 0.1), Gripper(ge), lambda s: True, [])
    arm = FakeArm(Q0 + 0.1)
    assert ge.run(segs, arm, Gripper(ge), lambda s: True, [], mock=True, ch=ch) == "done"
    assert arm.sent[0].startswith("to_start")


def test_run_stops_when_the_arm_does_not_arrive(ge, ch):
    segs = ge.build(plan_for(ge, ch), ch)

    class Short(FakeArm):
        def execute(self, seg):
            self.sent.append(seg.name)
            self._q = seg.Q[-1] + 0.05
            return True, "ok"
    with pytest.raises(ge.ExecRefusal, match="ended"):
        ge.run(segs, Short(Q0), Gripper(ge), lambda s: True, [])


def test_jog_line_goes_straight_up_and_back(ge, ch):
    spec = importlib.util.spec_from_file_location("jog_line", REPO / "outputs" / "runs" / "jog_line.py")
    jl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jl)
    up, down = jl.build(ge, ch, Q0, 0.03, 0.1, 0.01)
    np.testing.assert_allclose(up.path[-1] - up.path[0], [0, 0, 0.03], atol=2e-4)
    np.testing.assert_array_equal(down.Q[0], up.Q[-1])
    np.testing.assert_array_equal(down.Q[-1], up.Q[0])
    d = up.path - up.path[0]
    assert np.abs(d[:, :2]).max() < 3e-4                           # straight up in base z
    assert up.t[-1] >= 3.0 - 1e-6 and down.t[-1] >= 3.0 - 1e-6       # 3 cm at 1 cm/s
    arm = FakeArm(Q0)
    for s in (up, down):
        ge._do_traj(s, arm, [])
    np.testing.assert_allclose(arm.q(), Q0)


def test_reach_keeps_curobos_timing_stretched_uniformly(ge, ch):
    plan = plan_for(ge, ch)
    reach = {s.name: s for s in ge.build(plan, ch)}["reach"]
    k = reach.info["time_scale"]
    assert k > 1                                                     # the caps did bind
    np.testing.assert_allclose(reach.t, np.arange(40) * plan["dt"] * k)   # same profile, stretched
    V_dh = um.to_dh_path(plan["velocities"], plan["joint_names"])
    np.testing.assert_allclose(reach.V[1:-1], V_dh[1:-1] / k)
    v_cap = np.pi * ge.SPEED_FRAC
    assert np.abs(reach.V).max() <= v_cap * (1 + 1e-9)
    A = np.gradient(reach.V, reach.t, axis=0)
    assert np.abs(A).max() <= np.pi * ge.SPEED_FRAC * 1.05
    # a slow plan is not slowed further
    slow = dict(plan, velocities=(np.asarray(plan["velocities"]) * 0).tolist())
    assert {s.name: s for s in ge.build(slow, ch)}["reach"].info["time_scale"] == 1.0
    with pytest.raises(ge.ExecRefusal, match="velocities"):
        ge.build({k2: v for k2, v in plan.items() if k2 != "velocities"}, ch)
