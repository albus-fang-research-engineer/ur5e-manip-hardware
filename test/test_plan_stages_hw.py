"""outputs/runs/plan_stages_hw.py -- offline pieces (no sidecar). Needs the
sim checkout and manip_cbirrt on PYTHONPATH (../ur5e-manip-sim, ../manip-cbirrt)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "manip_bridge"))
pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
pytest.importorskip("manip_sim", reason="PYTHONPATH=../ur5e-manip-sim")


@pytest.fixture(scope="module")
def hw():
    spec = importlib.util.spec_from_file_location("plan_stages_hw", REPO / "outputs" / "runs" / "plan_stages_hw.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_joint_travel_flags_a_wrist_that_winds_and_unwinds(hw):
    q0 = np.array([2.8, -1.2, 1.1, -0.1, 2.5, 1.0])
    s = np.linspace(0, 1, 50)[:, None]
    out = q0 + s * [0.2, 0, 0, 0, 0, np.pi / 2]                 # shoulder pan +11.5, wrist 3 +90
    back = out[-1] + s[1:] * [0, 0, 0, 0, 0, -np.pi / 2]       # wrist 3 back -90
    jt = hw.joint_travel(np.vstack([out, back]))
    assert list(jt) == list(hw.um.DH_ORDER)
    w3, pan = jt[hw.um.DH_ORDER[5]], jt[hw.um.DH_ORDER[0]]
    assert w3 == {"total": 180.0, "net": 0.0, "range": 90.0}
    assert pan["total"] == pytest.approx(11.5, abs=0.1) and pan["net"] == pan["total"]
    assert all(v == {"total": 0.0, "net": 0.0, "range": 0.0} for v in list(jt.values())[1:5])
    assert "(net +0)" in hw._fmt_travel(jt)


def test_report_records_sim_provenance(hw):
    p = hw.psc.provenance()
    assert p["files"] and any(f.endswith("manip_sim/pour_stages.py") for f in p["files"])
    assert "commit" in p and "dirty" in p
