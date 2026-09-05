"""Offline tests for manip_bridge.tsr_spec (hand-authored YAML -> TSR) and
the shipped example spec. Pure numpy; no ROS.

    python -m pytest test/test_tsr_spec.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "manip_bridge"))

from manip_tsr import FREE_ROT, FREE_TRANS                              # noqa: E402
from manip_bridge.grasp_filter import tsr_from_flat, tsr_to_flat         # noqa: E402
from manip_bridge.tsr_spec import (bounds_from_spec, pose_from_spec,     # noqa: E402
                                   tsr_from_spec)

EXAMPLE = ROOT / "ros2_ws" / "src" / "manip_bridge" / "config" / "tsr_grasp_example.yaml"


def test_pose_from_spec_conventions():
    np.testing.assert_allclose(pose_from_spec(None), np.eye(4))
    T = pose_from_spec({"xyz": [1, 2, 3], "rpy_deg": [90, 0, 0]})
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(T[:3, :3], R.from_euler("x", np.pi / 2).as_matrix(), atol=1e-12)
    M = np.arange(16, dtype=float)
    np.testing.assert_allclose(pose_from_spec({"matrix": M.tolist()}), M.reshape(4, 4))
    np.testing.assert_allclose(pose_from_spec({"matrix": M.reshape(4, 4).tolist()}), M.reshape(4, 4))


def test_bounds_from_spec_units_and_free():
    Bw = bounds_from_spec({"x": [-0.01, 0.02], "y": 0.003, "roll": [-10, 10], "yaw": "free",
                           "pitch": "FREE"})
    np.testing.assert_allclose(Bw[0], [-0.01, 0.02])
    np.testing.assert_allclose(Bw[1], [0.003, 0.003])
    np.testing.assert_allclose(Bw[2], [0.0, 0.0])                      # missing -> pinned
    np.testing.assert_allclose(Bw[3], np.deg2rad([-10, 10]))
    np.testing.assert_allclose(Bw[4], FREE_ROT)
    np.testing.assert_allclose(Bw[5], FREE_ROT)
    np.testing.assert_allclose(bounds_from_spec({"z": "free"})[2], FREE_TRANS)


def test_bounds_from_spec_rejects_garbage():
    with pytest.raises(ValueError):
        bounds_from_spec({"x": [0.02, -0.01]})           # hi < lo
    with pytest.raises(ValueError):
        bounds_from_spec({"x": "loose"})                 # unknown keyword
    with pytest.raises(ValueError):
        bounds_from_spec({"sideways": [0, 1]})           # unknown axis


def test_tsr_from_spec_requires_frame_and_defaults_topic():
    spec = {"name": "grasp/x", "frame_id": "mug", "t0_w": {"xyz": [0, 0, 0.1]},
            "bw": {"z": [-0.02, 0.02], "yaw": "free"}}
    tsr, frame, topic = tsr_from_spec(spec)
    assert (frame, topic, tsr.name) == ("mug", "/tsr/mug/grasp", "grasp/x")
    np.testing.assert_allclose(tsr.Tw_e, np.eye(4))
    for missing in ("name", "frame_id", "t0_w", "bw"):
        bad = dict(spec); bad.pop(missing)
        with pytest.raises(ValueError):
            tsr_from_spec(bad)


def test_example_yaml_loads_and_round_trips_the_wire():
    spec = yaml.safe_load(EXAMPLE.read_text())
    tsr, frame, topic = tsr_from_spec(spec)
    assert frame == "mug" and topic == "/tsr/mug/grasp" and tsr.name.startswith("grasp")
    t0, te, bw = tsr_to_flat(tsr)
    back = tsr_from_flat(t0, te, bw, tsr.name)
    np.testing.assert_allclose(back.T0_w, tsr.T0_w)
    np.testing.assert_allclose(back.Bw, tsr.Bw)
    # sampling from it must produce poses it contains (sanity of the authored bounds)
    rng = np.random.default_rng(0)
    assert all(tsr.contains(tsr.sample(rng), tol=1e-9) for _ in range(50))
