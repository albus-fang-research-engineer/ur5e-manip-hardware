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
    assert abs(r["elevation_deg_in_base"] - 35.0) < 1e-6
    assert "point-only default" in r["axis_source"]

    # the sim's default bounds, unchanged
    assert r["bounds"] == {"x": [-0.005, 0.005], "y": [-0.005, 0.005], "z": [-0.02, 0.02],
                           "roll": pytest.approx([-np.rad2deg(0.09), np.rad2deg(0.09)]),
                           "pitch": pytest.approx([-np.rad2deg(0.09), np.rad2deg(0.09)]),
                           "yaw": pytest.approx([-45.0, 45.0])}

    # what tsr_from_yaml will publish classifies exactly as the sim's TSR does
    from manip_bridge import grasping_compat as gc
    from manip_bridge.tsr_spec import tsr_from_spec
    from manip_bridge import selection_compat as sc
    spec = yaml.safe_load((run / "m3" / "grasp_tsr.yaml").read_text())
    hw, frame_id, topic = tsr_from_spec(spec)
    assert (frame_id, topic, hw.name) == ("base_link", "/tsr/mug/grasp", "grasp/handle")
    rng = np.random.default_rng(0)
    poses = [sim.sample(rng) for _ in range(50)]
    for k, dx in enumerate((0.008, -0.008, 0.03)):                  # outside: lateral, slide
        T = poses[k].copy()
        T[:3, 3] += sim.T0_w[:3, k % 3] * dx
        poses.append(T)
    verdicts = [(sim.contains(T), hw.contains(T)) for T in poses]
    assert all(a == b for a, b in verdicts)
    assert sum(a for a, _ in verdicts) == 50
    assert sc.provenance() and gc.provenance()


def test_refuses_a_run_without_t_base_cam(mods, tmp_path):
    *_, gt = mods
    run = _selected_run(mods, tmp_path, t_base_cam=None)
    with pytest.raises(gt.GraspTsrRefusal, match="no T_base_cam .absent_in_source."):
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
