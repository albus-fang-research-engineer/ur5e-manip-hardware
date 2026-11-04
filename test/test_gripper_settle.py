"""outputs/runs/grasp_exec.py settle(): the laptop's gripper node answers each
GripperCommand goal with its state from when the goal arrived (bench, 11/03:
close from open -> 0.0839 m not stalled; the same close again -> 0.0100 m
stalled on the strap). settle() re-sends the goal until the jaws stop.
Offline: a simulated node on a fake clock."""
import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OPEN = 0.0839          # what the node reports fully open
STRAP = 0.0100         # what it reported holding the teapot strap


@pytest.fixture(scope="module")
def ge():
    spec = importlib.util.spec_from_file_location("grasp_exec", REPO / "outputs" / "runs" / "grasp_exec.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Clock:
    def __init__(self):
        self.t = 0.0

    def sleep(self, dt):
        self.t += dt

    def now(self):
        return self.t


class StaleNode:
    """Answers with the state at arrival, then moves toward the goal at
    speed (m/s) after lag (s); closing stops, stalled, at the object gap."""

    def __init__(self, clock, pos, goal=None, obj=None, speed=0.1, lag=0.1):
        self.c, self.obj, self.speed, self.lag = clock, obj, speed, lag
        self.from_pos, self.goal, self.t_goal = pos, pos if goal is None else goal, -1e9

    def state(self):
        d = self.goal - self.from_pos
        step = min(abs(d), self.speed * max(0.0, self.c.t - self.t_goal - self.lag))
        pos, stalled = self.from_pos + math.copysign(step, d), False
        if self.obj is not None and d < 0 and pos <= self.obj:
            pos, stalled = self.obj, True
        return {"position": pos, "stalled": stalled, "reached_goal": abs(pos - self.goal) < 0.002,
                "effort": 60.0}

    def send(self, goal):
        r = self.state()
        if goal != self.goal:
            self.from_pos, self.goal, self.t_goal = r["position"], goal, self.c.t
        return r


def run(ge, node, clock, goal):
    return ge.settle(lambda: node.send(goal), goal, sleep=clock.sleep, clock=clock.now)


def test_close_on_the_strap_reports_the_grip_not_the_stale_open(ge):
    first = StaleNode(Clock(), OPEN, obj=STRAP).send(0.0)            # the bench symptom
    assert first["position"] == OPEN and not first["stalled"]
    c = Clock()
    r = run(ge, StaleNode(c, OPEN, obj=STRAP), c, 0.0)
    assert r["stalled"] and r["position"] == pytest.approx(STRAP)
    assert r["sends"] > 2 and r["settle_s"] < 2.0


def test_close_on_nothing_settles_closed_and_not_stalled(ge):
    c = Clock()
    r = run(ge, StaleNode(c, OPEN), c, 0.0)
    assert not r["stalled"] and r["position"] == pytest.approx(0.0, abs=1e-9)


def test_open_from_holding_is_not_taken_for_the_stale_stall_even_with_a_slow_start(ge):
    c = Clock()
    node = StaleNode(c, STRAP, goal=0.0, obj=STRAP, lag=0.4)      # holding the strap
    assert node.state()["stalled"]
    r = run(ge, node, c, 0.085)
    assert not r["stalled"] and r["position"] == pytest.approx(0.085, abs=1e-6)


def test_closing_again_while_holding_returns_the_hold_after_the_stable_window(ge):
    c = Clock()
    r = run(ge, StaleNode(c, STRAP, goal=0.0, obj=STRAP), c, 0.0)
    assert r["stalled"] and r["position"] == pytest.approx(STRAP)
    assert r["settle_s"] >= ge.SETTLE_STABLE_S


def test_jaws_that_never_stop_are_refused(ge):
    c = Clock()
    with pytest.raises(ge.ExecRefusal, match="still moving"):
        run(ge, StaleNode(c, OPEN, speed=0.002), c, 0.0)
