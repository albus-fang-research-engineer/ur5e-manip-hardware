"""outputs/runs/grasp_tsr.py -- the stage-1 grasp TSR on a hardware run
(M3 step 2) -- and grasping_compat.

Offline, on the same synthetic bound run test_select_scene builds (canned
call #2 pick), plus a hand-eye in summary.json's packet. Needs the sim
checkout at $SIM_DIR or ../ur5e-manip-sim and manip_tsr; skips without them.
"""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("manip_tsr")
yaml = pytest.importorskip("yaml")
import test_select_scene as tss  # noqa: E402  -- its synthetic run, R, T, K, HANDLE, PICK

pytestmark = pytest.mark.skipif(not tss.HAVE_SIM, reason=f"no sim checkout at {tss.SIM_DIR}")

# camera -> base: translation only, so camera z (= canonical up in the
# fixture's cam_T_obj) is base up and the elevation is measurable in base
T_BC = np.eye(4)
T_BC[:3, 3] = [0.40, -0.20, 0.30]


@pytest.fixture(scope="module")
def mods():
    paths = [str(tss.BRIDGE), str(tss.SIM_DIR), str(tss.RUNS)]
    for p in paths:
        sys.path.insert(0, p)
    import test_plan_scene as tps
    yield tss._load("plan_scene"), tss._load("select_scene"), tps, tss._load("grasp_tsr")
    for p in paths:
        sys.path.remove(p)


def _selected_run(mods, tmp, t_base_cam=T_BC):
    ps, ss, tps, _ = mods
    run = tss._bound_run((ps, ss, tps), tmp)
    code, _, _ = ss.select_scene(run, transport=tps.Canned(tss._resp(tss.PICK)),
                                 render=tss.StubRender())
    assert code == 0
    s = json.loads((run / "summary.json").read_text())
    s["packet"] = {"t_base_cam": ({"source": "tf", "T": t_base_cam.tolist()}
                                  if t_base_cam is not None else {"source": "absent_in_source"})}
    (run / "summary.json").write_text(json.dumps(s))
    return run


def test_tsr_is_the_sims_placed_at_the_pick_in_base(mods, tmp_path):
    *_, gt = mods
    run = _selected_run(mods, tmp_path)
    code, r, sim = gt.grasp_tsr(run)
    assert code == 0

    # origin = T_base_cam . cam_T_obj . R x_canon of the picked mark
    x_c = np.asarray(tss.HANDLE[tss.PICK])
    want = T_BC[:3, :3] @ (tss.T[:3, :3] @ (tss.R @ x_c) + tss.T[:3, 3]) + T_BC[:3, 3]
    np.testing.assert_allclose(r["origin_base_m"], want, atol=1e-12)
    # select_scene logs the pixel rounded per axis: at most sqrt(0.5) px off
    assert r["pixel"]["distance"] <= np.sqrt(0.5) and r["pixel"]["select_scene"] == tss._px(x_c)
    assert "point-only default" in r["axis_source"]
    np.testing.assert_allclose(r["bar_axis_base"], [0, 0, 1], atol=1e-12)   # fixture: level table
    assert r["bar_tilt_deg"] < 1e-6
    assert abs(r["approach_nominal_base"][2]) < 1e-12                       # horizontal nominal
    np.testing.assert_allclose(np.dot(r["closing_base"], r["bar_axis_base"]), 0, atol=1e-12)

    # the sim's default bounds except the roll row (= the elevation, horizontal..down)
    roll = r["bounds"].pop("roll")
    assert roll in ([0.0, 90.0], [-90.0, 0.0]) and r["elevation_roll_row_deg"] == roll
    assert r["bounds"] == {"x": [-0.015, 0.015], "y": [-0.015, 0.015], "z": [-0.02, 0.02],
                           "pitch": pytest.approx([-np.rad2deg(0.09), np.rad2deg(0.09)]),
                           "yaw": pytest.approx([-45.0, 45.0])}
    head = (run / "m3" / "grasp_tsr.yaml").read_text().splitlines()[:4]
    assert any("TWO departures" in h for h in head)

    # what tsr_from_yaml will publish classifies exactly as the sim's TSR does
    from manip_bridge import grasping_compat as gc
    from manip_bridge.tsr_spec import tsr_from_spec
    from manip_bridge import selection_compat as sc
    spec = yaml.safe_load((run / "m3" / "grasp_tsr.yaml").read_text())
    hw, frame_id, topic = tsr_from_spec(spec)
    assert (frame_id, topic, hw.name) == ("base_link", "/tsr/mug/grasp", "grasp/handle")
    rng = np.random.default_rng(0)
    poses = [sim.sample(rng) for _ in range(50)]
    for k, dx in enumerate((0.04, -0.04, 0.05)):                    # outside: lateral, slide
        T = poses[k].copy()
        T[:3, 3] += sim.T0_w[:3, k % 3] * dx
        poses.append(T)
    verdicts = [(sim.contains(T), hw.contains(T)) for T in poses]
    assert all(a == b for a, b in verdicts)
    assert sum(a for a, _ in verdicts) == 50
    assert sc.provenance() and gc.provenance()


def _grasp(r, elev_deg=35.0, azimuth_deg=0.0, closing_tilt_deg=0.0, offset=(0, 0, 0)):
    """A pad-centre pose in base: approach pitched elev_deg below the nominal
    horizontal (negative = from below), swung azimuth_deg about the bar,
    the closing axis tilted toward the bar by closing_tilt_deg."""
    from scipy.spatial.transform import Rotation as Rot
    up, c, a0 = (np.asarray(r[k], float) for k in ("bar_axis_base", "closing_base",
                                                    "approach_nominal_base"))
    yaw = Rot.from_rotvec(np.deg2rad(azimuth_deg) * up)
    c, a0 = yaw.apply(c), yaw.apply(a0)
    e = np.deg2rad(elev_deg)
    a = np.cos(e) * a0 - np.sin(e) * up                    # descends for elev > 0
    c = Rot.from_rotvec(np.deg2rad(closing_tilt_deg) * a).apply(c)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([c, np.cross(a, c), a])
    T[:3, 3] = np.asarray(r["origin_base_m"]) + np.asarray(offset, float)
    return T


def test_elevation_is_free_and_position_is_the_question(mods, tmp_path):
    """Any elevation from outside passes; what still rejects is position (the
    grounding question) and fingers not straddling the bar. Also the numeric
    check that the roll row's sign descends: 35 deg from BELOW is rejected."""
    *_, gt = mods
    run = _selected_run(mods, tmp_path)
    _, r, _ = gt.grasp_tsr(run)
    from manip_bridge.tsr_spec import tsr_from_spec
    hw, _, _ = tsr_from_spec(yaml.safe_load((run / "m3" / "grasp_tsr.yaml").read_text()))
    radial = np.asarray(r["approach_nominal_base"])        # horizontal, across the bar
    for elev in (0.0, 35.0, 80.0, 89.9):
        assert hw.contains(_grasp(r, elev)), elev
    assert hw.contains(_grasp(r, 35.0, azimuth_deg=40.0))
    assert hw.contains(_grasp(r, 35.0, offset=0.012 * radial))         # inside +-15 mm
    rejected = {"from below": _grasp(r, -35.0),
                "from the body side": _grasp(r, 35.0, azimuth_deg=180.0),
                "closing tilted 15 deg toward the bar": _grasp(r, 35.0, closing_tilt_deg=15.0),
                "20 mm off the anchor sideways": _grasp(r, 35.0, offset=0.020 * radial)}
    for why, T in rejected.items():
        assert not hw.contains(T), why


def test_refuses_a_run_without_t_base_cam(mods, tmp_path):
    *_, gt = mods
    run = _selected_run(mods, tmp_path, t_base_cam=None)
    with pytest.raises(gt.GraspTsrRefusal, match="no T_base_cam .absent_in_source.*set_hand_eye"):
        gt.grasp_tsr(run)
    assert not (run / "m3" / "grasp_tsr.yaml").exists()


def test_pixel_gate_catches_a_wrong_chain(mods, tmp_path, monkeypatch):
    """A transposed R_body_from_canon moves the origin off select_scene's pixel."""
    _, _, _, gt = mods
    run = _selected_run(mods, tmp_path)
    real = gt.ss.load_inputs

    def transposed(*a, **k):
        b, doc, objs = real(*a, **k)
        for o in objs.values():
            o["R"] = o["R"].T
        return b, doc, objs
    monkeypatch.setattr(gt.ss, "load_inputs", transposed)
    with pytest.raises(gt.GraspTsrRefusal, match="composed wrong"):
        gt.grasp_tsr(run)


def test_refuses_selections_from_another_select_scene_run(mods, tmp_path):
    *_, gt = mods
    run = _selected_run(mods, tmp_path)
    lp = run / "call2" / "selections.log.json"
    log = json.loads(lp.read_text())
    log["resolved"]["grasp"]["candidate_id"] = tss.PICK + 1
    lp.write_text(json.dumps(log))
    with pytest.raises(gt.GraspTsrRefusal, match="different select_scene runs"):
        gt.grasp_tsr(run)


def test_set_hand_eye_then_grasp_tsr(mods, tmp_path):
    """The real order on an old run: refused, set_hand_eye writes the
    calibration into the run, then grasp_tsr reads it from there."""
    *_, gt = mods
    run = _selected_run(mods, tmp_path, t_base_cam=None)
    s = json.loads((run / "summary.json").read_text())
    s["frame"]["frame_id"] = "camera_color_optical_frame"
    (run / "summary.json").write_text(json.dumps(s))
    with pytest.raises(gt.GraspTsrRefusal):
        gt.grasp_tsr(run)
    calib = tmp_path / "cell.calib"
    calib.write_text(yaml.safe_dump({
        "parameters": {"name": "cell", "calibration_type": "eye_on_base",
                       "robot_base_frame": "base_link", "robot_effector_frame": "tool0",
                       "tracking_base_frame": "camera_color_optical_frame",
                       "tracking_marker_frame": "marker"},
        "transform": {"translation": dict(zip("xyz", map(float, T_BC[:3, 3]))),
                      "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}))
    import importlib.util
    spec = importlib.util.spec_from_file_location("set_hand_eye", tss.RUNS / "set_hand_eye.py")
    she = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(she)
    assert she.set_hand_eye(run, calib)[0] == 0
    code, r, _ = gt.grasp_tsr(run)
    assert code == 0 and r["pixel"]["distance"] <= np.sqrt(0.5)
