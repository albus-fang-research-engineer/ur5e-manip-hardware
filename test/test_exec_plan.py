"""outputs/runs/exec_plan.py -- the general executor, offline: synthetic
steps, a fake arm that lands wherever it is sent, the in-process mock
gripper. Needs manip_cbirrt (PYTHONPATH=../manip-cbirrt from the repo root)."""
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
OTHER_ORDER = [um.DH_ORDER[i] for i in (2, 0, 1, 5, 3, 4)]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "outputs" / "runs" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ep():
    return _load("exec_plan")


@pytest.fixture(scope="module")
def ge(ep):
    return ep._grasp_exec()


@pytest.fixture(scope="module")
def ch():
    return um.chain()


def path(q0, dq, n=30):
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    return q0 + np.outer(s, dq)


def traj(P, order=um.DH_ORDER, dt=None):
    idx = [um.DH_ORDER.index(n) for n in order]
    t = {"joint_names": list(order), "positions": P[:, idx].tolist()}
    if dt is not None:
        t["velocities"] = np.gradient(P, dt, axis=0)[:, idx].tolist()
        t["dt"] = dt
    return t


def spec_two_stages():
    P1 = path(Q0, [0.15, -0.1, 0.12, 0.05, -0.1, 0.2])
    P2 = path(P1[-1], [-0.1, 0.05, -0.05, 0.0, 0.1, -0.1])
    return {"steps": [
        {"name": "reach", "traj": traj(P1, OTHER_ORDER, dt=0.02)},
        {"name": "close", "gripper": {"gap_m": 0.0, "effort_n": 60, "expect_object": True}},
        {"name": "carry", "traj": traj(P2)},
        {"name": "open", "gripper": {"gap_m": 0.085}, "gate": False},
    ]}, P1, P2


class FakeArm:
    def __init__(self, q):
        self._q, self.sent = np.asarray(q, float), []

    def q(self):
        return self._q.copy()

    def execute(self, seg):
        self.sent.append(seg.name)
        self._q = seg.Q[-1].copy()
        return True, "ok"


class Gripper:
    def __init__(self, ge, grips=True):
        self.mock, self.grips, self.cmds = ge.MockGripper(), grips, []

    def command(self, gap, effort):
        self.cmds.append(gap)
        r = self.mock.command(gap, effort)
        if gap < 0.04 and not self.grips:
            r.update(stalled=False, reached_goal=True, position=0.0)
        return r


def test_build_timing_names_and_gates(ep, ge, ch):
    spec, P1, P2 = spec_two_stages()
    segs = ep.build(spec, ch, ge, 0.1)
    assert [(s.name, s.kind, s.gate) for s in segs] == [
        ("reach", "traj", True), ("close", "gripper", True),
        ("carry", "traj", True), ("open", "gripper", False)]
    by = {s.name: s for s in segs}
    np.testing.assert_allclose(by["reach"].Q, P1)                 # names honoured
    assert "time_scale" in by["reach"].info                      # planner timing kept
    np.testing.assert_allclose(by["carry"].Q[[0, -1]], P2[[0, -1]])
    for s in (by["reach"], by["carry"]):
        assert np.abs(s.V).max() <= np.pi * 0.1 * 1.02
    assert by["open"].effort_n == ge.CLOSE_EFFORT_N              # default effort


def test_build_refuses_bad_plans(ep, ge, ch):
    spec, P1, P2 = spec_two_stages()
    gap = dict(spec, steps=[spec["steps"][0], {"name": "carry", "traj": traj(P2 + 0.05)}])
    with pytest.raises(ge.ExecRefusal, match="starts .* from where the previous"):
        ep.build(gap, ch, ge, 0.1)
    for bad, msg in (({"steps": []}, "no 'steps'"),
                     ({"steps": [{"name": "x", "gripper": {"gap_m": 0.0}}]}, "no traj step"),
                     ({"steps": [{"name": "x"}]}, "exactly one of"),
                     ({"steps": [{"name": "x", "gripper": {}}]}, "gap_m")):
        with pytest.raises(ge.ExecRefusal, match=msg):
            ep.build(bad, ch, ge, 0.1)


def test_run_order_no_object_and_gates(ep, ge, ch):
    spec, _, _ = spec_two_stages()
    arm, grip, log = FakeArm(Q0), Gripper(ge), []
    assert ep.run(ep.build(spec, ch, ge, 0.1), arm, grip, lambda s: True, log, ge) == "done"
    assert arm.sent == ["reach", "carry"] and grip.cmds == [0.0, 0.085]
    assert [e["segment"] for e in log] == ["reach", "close", "carry", "open"]
    # nothing gripped: stop where it is, no carry
    arm = FakeArm(Q0)
    assert ep.run(ep.build(spec, ch, ge, 0.1), arm, Gripper(ge, grips=False),
                  lambda s: True, [], ge) == "no object at close"
    assert arm.sent == ["reach"]
    # a gate says no
    arm, asked = FakeArm(Q0), []
    gate = lambda s: asked.append(s.name) or s.name != "carry"       # noqa: E731
    assert ep.run(ep.build(spec, ch, ge, 0.1), arm, Gripper(ge), gate, [], ge) == "stopped at carry"
    assert arm.sent == ["reach"] and asked == ["reach", "close", "carry"]


def test_run_refuses_a_stale_start_unless_mock(ep, ge, ch):
    spec, _, _ = spec_two_stages()
    segs = ep.build(spec, ch, ge, 0.1)
    with pytest.raises(ge.ExecRefusal, match="stale plan"):
        ep.run(segs, FakeArm(Q0 + 0.1), Gripper(ge), lambda s: True, [], ge)
    arm = FakeArm(Q0 + 0.1)
    assert ep.run(segs, arm, Gripper(ge), lambda s: True, [], ge, mock=True, ch=ch) == "done"
    assert arm.sent[0].startswith("to_start")


def test_arm_only_plan_needs_no_gripper(ep, ge, ch):
    P = path(Q0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.05], n=2)
    arm = FakeArm(Q0)
    segs = ep.build({"steps": [{"name": "wrist", "traj": traj(P)}]}, ch, ge, 0.1)
    assert ep.run(segs, arm, None, lambda s: True, [], ge) == "done"
    np.testing.assert_allclose(arm.q(), P[-1])
