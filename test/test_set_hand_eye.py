"""outputs/runs/set_hand_eye.py -- writing an easy_handeye2 calibration into
a run captured without one. Offline: a synthetic summary.json and .calib
files in easy_handeye2's own format (HandeyeCalibration as YAML). No sim.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
OPTICAL = "camera_color_optical_frame"

# camera 1 m up, 0.5 m back, looking 45 deg down at the table: optical z is
# the viewing direction, so the table normal in the camera frame follows
R_BC = Rotation.from_euler("xyz", [-135.0, 0.0, -90.0], degrees=True).as_matrix()
T_BC = np.eye(4)
T_BC[:3, :3], T_BC[:3, 3] = R_BC, [-0.5, 0.0, 1.0]


@pytest.fixture(scope="module")
def she():
    sys.path.insert(0, str(BRIDGE))
    spec = importlib.util.spec_from_file_location(
        "set_hand_eye", REPO_ROOT / "outputs" / "runs" / "set_hand_eye.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    yield m
    sys.path.remove(str(BRIDGE))


def _calib(tmp, T=T_BC, ctype="eye_on_base", base="base_link", cam=OPTICAL, name="cell"):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    doc = {"parameters": {"name": name, "calibration_type": ctype, "robot_base_frame": base,
                          "robot_effector_frame": "tool0", "tracking_base_frame": cam,
                          "tracking_marker_frame": "marker", "freehand_robot_movement": False,
                          "move_group_namespace": "/", "move_group": "manipulator"},
           "transform": {"translation": dict(zip("xyz", map(float, T[:3, 3]))),
                         "rotation": dict(zip("xyzw", map(float, q)))}}
    p = Path(tmp) / f"{name}.calib"
    p.write_text(yaml.safe_dump(doc))
    return p


def _run(tmp, plane_n=None, t_base_cam=None):
    run = Path(tmp) / "run"
    run.mkdir()
    n = R_BC.T @ [0, 0, 1] if plane_n is None else plane_n          # level table, true camera
    packet = {"plane": {"n": list(map(float, n)), "d": 1.0}, "warnings": [],
              "t_base_cam": t_base_cam or {"source": "absent_in_source"}}
    (run / "summary.json").write_text(json.dumps(
        {"frame": {"frame_id": OPTICAL, "K": np.eye(3).tolist()}, "packet": packet,
         "objects": {"m2": {"pose_on_final": {"cam_T_obj": np.eye(4).tolist()}}}}))
    return run


def test_writes_where_run_scene_would_and_checks_the_table(she, tmp_path):
    run, calib = _run(tmp_path), _calib(tmp_path)
    before = (run / "summary.json").read_text()
    code, r = she.set_hand_eye(run, calib)
    assert code == 0 and r["written"]
    s = json.loads((run / "summary.json").read_text())
    rec = s["packet"]["t_base_cam"]
    np.testing.assert_allclose(rec["T"], T_BC, atol=1e-12)
    assert rec["source"] == "param" and rec["replaces_source"] == "absent_in_source"
    assert rec["calib"]["name"] == "cell" and len(rec["calib"]["sha256"]) == 64
    assert r["plane_angle_to_base_up_deg"] < 1e-6 and s["packet"]["warnings"] == []
    assert s["objects"] == json.loads(before)["objects"]                 # nothing else touched
    assert (run / "summary.json.pre_hand_eye").read_text() == before
    # the same calibration again: a no-op
    after = (run / "summary.json").read_text()
    code, r2 = she.set_hand_eye(run, calib)
    assert code == 0 and r2["written"] is False
    assert (run / "summary.json").read_text() == after


def test_a_calibration_from_another_camera_pose_warns(she, tmp_path):
    moved = T_BC.copy()
    moved[:3, :3] = Rotation.from_euler("x", 12, degrees=True).as_matrix() @ R_BC
    run = _run(tmp_path)
    code, r = she.set_hand_eye(run, _calib(tmp_path, T=moved))
    assert code == 0 and r["plane_angle_to_base_up_deg"] == pytest.approx(12.0, abs=1e-6)
    s = json.loads((run / "summary.json").read_text())
    assert any("may not describe the camera" in w for w in s["packet"]["warnings"])


@pytest.mark.parametrize("kw,match", [
    ({"cam": "camera_link"}, "tracking_base_frame 'camera_link' but the run's depth and K"),
    ({"ctype": "eye_in_hand"}, "eye_on_base calibration is required"),
    ({"base": "world"}, "robot_base_frame 'world'"),
])
def test_refuses_the_wrong_calibration(she, tmp_path, kw, match):
    run = _run(tmp_path)
    before = (run / "summary.json").read_text()
    with pytest.raises(she.HandEyeRefusal, match=match):
        she.set_hand_eye(run, _calib(tmp_path, **kw))
    assert (run / "summary.json").read_text() == before


def test_never_replaces_a_recorded_hand_eye(she, tmp_path):
    other = T_BC.copy()
    other[0, 3] += 0.01
    run = _run(tmp_path, t_base_cam={"source": "tf", "T": other.tolist()})
    with pytest.raises(she.HandEyeRefusal, match="never replaced"):
        she.set_hand_eye(run, _calib(tmp_path))
