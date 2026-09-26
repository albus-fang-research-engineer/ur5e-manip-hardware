"""frame_packet: median depth, joint filtering, the depth contract, and
every gate / degrade in build(), plus replaying saved runs.

Synthetic scene: camera looking straight down (OpenCV frame) at a table at
z = 0.6 m with a 10 cm cube on it. Base +z (up) in this camera frame is
(0, 0, -1), i.e. R_base_cam = diag(1, -1, -1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"))
from manip_bridge import frame_packet as fp  # noqa: E402

H, W = 120, 160
K = [[150.0, 0, 80.0], [0, 150.0, 60.0], [0, 0, 1]]
T_LEVEL = np.diag([1.0, -1.0, -1.0, 1.0]); T_LEVEL[:3, 3] = [0.5, 0.0, 0.6]
ARM = list(fp.UR_ARM_JOINTS)


def table(seed=0, z=0.6):
    d = np.full((H, W), z) + np.random.default_rng(seed).normal(0, 0.001, (H, W))
    d[50:70, 60:90] = 0.5
    return d


def contract(**over):
    rgb = {"topic": "/c", "frame_id": "cam", "width": W, "height": H, "encoding": "rgb8"}
    dep = {"topic": "/d", "frame_id": "cam", "width": W, "height": H, "encoding": "16UC1"}
    info = {"topic": "/i", "frame_id": "cam", "width": W, "height": H,
            "distortion_model": "plumb_bob", "D": [0, 0, 0, 0, 0]}
    for k, v in over.items():
        which, key = k.split("__")
        {"rgb": rgb, "dep": dep, "info": info}[which][key] = v
    return fp.depth_contract(rgb, dep, info)


def still_samples(t0=10.0, n=20, q=(0.1, -1.2, 1.3, -1.6, -1.57, 0.0), jitter=0.0):
    rng = np.random.default_rng(3)
    return [fp.arm_sample(t0 + 0.03 * i, ARM, list(np.add(q, rng.normal(0, jitter, 6))),
                          [0.0] * 6) for i in range(n)]


def live(**over):
    kw = dict(depths=[table(s) for s in range(5)], K=K, source="live", contract=contract(),
              stamps={"ref": 10.06, "rgb": [10.0, 10.03, 10.06, 10.09, 10.12],
                      "depth": [10.001, 10.031, 10.061, 10.091, 10.121]},
              joints={"samples": still_samples(9.9), "topic": "/joint_states"},
              t_base_cam={"T": T_LEVEL, "source": "tf", "dt_s": 0.0},
              robot_mask_fn=lambda depth, K, T, q, n: np.zeros(depth.shape, bool))
    kw.update(over)
    return fp.build(**kw)


# ------------------------------------------------------------------ pieces

def test_median_depth_fills_flicker_keeps_systematic_holes_and_flags_motion():
    base = table()
    frames = [base.copy() for _ in range(6)]
    frames[0][0:10, 0:10] = 0                         # flicker: 1 of 6 missing
    for f in frames:
        f[100:110, 100:110] = 0                        # systematic hole
    for f in frames[3:]:
        f[20:30, 20:30] = 0.45                         # a hand arrives mid-capture
    med, st = fp.median_depth(frames, min_valid_frac=0.5, unstable_m=0.02)
    assert np.all(med[0:10, 0:10] > 0) and np.all(med[100:110, 100:110] == 0)
    assert st["min_valid_samples"] == 3 and st["n"] == 6
    assert st["unstable_frac"] == pytest.approx(100 / (H * W - 100), rel=0.01)


def test_arm_sample_rejects_gripper_only_and_reorders():
    assert fp.arm_sample(1.0, ["finger_joint"], [0.3]) is None
    names = ARM[::-1] + ["finger_joint"]
    t, n, q, v = fp.arm_sample(1.0, names, [6, 5, 4, 3, 2, 1, 0.3], [0] * 7)
    assert n == ARM and q == [1, 2, 3, 4, 5, 6] and v == [0.0] * 6


def test_joint_motion_is_by_name_and_windowed():
    s = still_samples(10.0)
    assert fp.joint_motion(s, 10.0, 10.5)["max_displacement_rad"] == 0.0
    moved = s + [fp.arm_sample(10.2, ARM, [0.1, -1.2, 1.3, -1.6, -1.57, 0.02])]
    assert fp.joint_motion(moved, 10.0, 10.5)["max_displacement_rad"] == pytest.approx(0.02)
    assert fp.joint_motion(moved, 11.0, 11.5)["n_samples"] == 0


def test_depth_contract_records_and_gates():
    c = contract()
    assert c["aligned"] and c["scale_m_per_unit"] == 0.001 and not c["distortion_nonzero"]
    with pytest.raises(fp.GateStop, match="depth frame_id"):
        contract(dep__frame_id="depth_optical")
    with pytest.raises(fp.GateStop, match="depth 160x100"):
        contract(dep__height=100)
    with pytest.raises(fp.GateStop, match="unsupported"):
        contract(dep__encoding="mono16")
    assert contract(info__D=[0.1, 0, 0, 0, 0])["distortion_nonzero"]


# --------------------------------------------------------------- build: ok

def test_live_packet_records_everything():
    depth, pk, rmask = live()
    assert pk["depth"]["aligned"] and pk["depth"]["median"]["n"] == 5
    assert pk["frames"]["span_s"] == pytest.approx(0.12) and pk["frames"]["rgb_depth_dt_max_s"] < 0.002
    assert pk["joints"]["status"] == "ok" and pk["joints"]["max_displacement_rad"] == 0.0
    assert pk["robot_mask"]["status"] == "ok" and pk["plane"]["fit_excludes_robot"]
    assert pk["plane"]["angle_to_up_deg"] < 1.0 and pk["plane"]["rms_m"] < 0.002
    assert pk["plane"]["n_inliers"] > 0.8 * pk["plane"]["n_points"]
    assert pk["degrades"] == [] and pk["warnings"] == []
    assert np.median(depth[50:70, 60:90]) == pytest.approx(0.5, abs=0.002)


def test_plane_fit_excludes_the_robot():
    """A large flat robot part (60% of the view, 20 cm above the table)
    wins the RANSAC unless the robot mask removes it first."""
    d = table()
    arm = np.zeros((H, W), bool); arm[:, :96] = True
    d[arm] = 0.4
    kw = dict(depths=[d], stamps=None)
    kw_live = dict(stamps={"ref": 10.0, "rgb": [10.0], "depth": [10.0]})
    _, pk, _ = live(depths=[d], **kw_live, robot_mask_fn=lambda *a: np.zeros((H, W), bool))
    assert pk["plane"]["d"] == pytest.approx(0.4, abs=0.01)          # grabbed the arm
    _, pk, _ = live(depths=[d], **kw_live, robot_mask_fn=lambda *a: arm)
    assert pk["plane"]["d"] == pytest.approx(0.6, abs=0.01)          # the table
    assert pk["robot_mask"]["n_masked"] == arm.sum()
    del kw


# ------------------------------------------------------------ build: gates

def test_stop_without_hand_eye_live():
    with pytest.raises(fp.GateStop) as e:
        live(t_base_cam=None)
    assert e.value.stage == "t_base_cam"


def test_stop_when_the_arm_moved():
    moving = still_samples(9.9)
    moving[8] = fp.arm_sample(moving[8][0], ARM, [0.1, -1.2, 1.3, -1.6, -1.57, 0.01])
    with pytest.raises(fp.GateStop) as e:
        live(joints={"samples": moving, "topic": "/joint_states"})
    assert e.value.stage == "stillness"


def test_stop_without_a_table():
    rng = np.random.default_rng(0)
    clutter = [rng.uniform(0.3, 1.5, (H, W))]
    with pytest.raises(fp.GateStop) as e:
        live(depths=clutter, stamps={"ref": 10.0, "rgb": [10.0], "depth": [10.0]})
    assert e.value.stage == "plane"


# --------------------------------------------------------- build: degrades

def test_missing_joint_states_is_a_degrade():
    called = []
    _, pk, rmask = live(joints={"samples": [], "topic": "/joint_states"},
                        robot_mask_fn=lambda *a: called.append(1))
    assert pk["joints"]["status"] == "missing" and rmask is None and not called
    assert any("joint_states" in d for d in pk["degrades"])
    assert not pk["plane"]["fit_excludes_robot"]


def test_robot_mask_failure_is_a_degrade():
    def boom(*a):
        raise TimeoutError("tcp://127.0.0.1:5671: no reply to 'robot_mask'")
    _, pk, rmask = live(robot_mask_fn=boom)
    assert rmask is None and pk["robot_mask"]["status"].startswith("error")
    assert any("robot mask failed" in d for d in pk["degrades"])


def test_warnings_tilt_sync_and_distortion():
    tilt = np.eye(4)
    tilt[:3, :3] = T_LEVEL[:3, :3] @ np.array([[1, 0, 0], [0, np.cos(0.35), -np.sin(0.35)],
                                               [0, np.sin(0.35), np.cos(0.35)]])
    _, pk, _ = live(t_base_cam={"T": tilt, "source": "tf", "dt_s": 0.0},
                    stamps={"ref": 10.06, "rgb": [10.0, 10.03, 10.06, 10.09, 10.12],
                            "depth": [10.0, 10.03, 10.1, 10.09, 10.12]},
                    contract=contract(info__D=[0.05, 0, 0, 0, 0]))
    w = " | ".join(pk["warnings"])
    assert "deg from base +z" in w and "RGB->depth" in w and "distortion" in w
    assert pk["plane"]["angle_to_up_deg"] == pytest.approx(20.05, abs=0.5)


# ---------------------------------------------------------------- from-run

def _save_run(tmp_path, packet=None):
    from PIL import Image
    run = tmp_path / "run"
    run.mkdir()
    Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(run / "rgb.png")
    Image.fromarray(np.round(table() * 1000).astype(np.uint16)).save(run / "depth_mm.png")
    summ = {"frame": {"stamp": "1725400000.123000000", "frame_id": "cam", "K": K}}
    if packet:
        summ["packet"] = packet
    (run / "summary.json").write_text(json.dumps(summ))
    return run


def test_old_run_replays_with_every_missing_field_a_degrade(tmp_path):
    pytest.importorskip("PIL")
    src = fp.load_run(_save_run(tmp_path))
    assert not src["has_packet"] and src["contract"]["aligned"] is None
    depth, pk, rmask = fp.build(depths=[src["depth"]], K=src["K"], source={"from_run": "x"},
                                contract=src["contract"], joints=src["joints"],
                                t_base_cam=src["t_base_cam"])
    assert rmask is None and pk["joints"]["status"] == "absent_in_source"
    assert pk["t_base_cam"]["source"] == "absent_in_source"
    d = " | ".join(pk["degrades"])
    assert "alignment not recorded" in d and "no joint state" in d and "no T_base_cam" in d
    assert "angle_to_up_deg" not in pk["plane"] and pk["frames"]["span_s"] is None


def test_packet_run_replays_its_q_and_hand_eye(tmp_path):
    pytest.importorskip("PIL")
    _, live_pk, _ = live()
    src = fp.load_run(_save_run(tmp_path, packet=json.loads(json.dumps(live_pk))))
    assert src["has_packet"] and src["contract"]["aligned"]
    assert src["joints"]["q"] == live_pk["joints"]["q"]
    assert np.allclose(src["t_base_cam"]["T"], T_LEVEL)
    _, pk, rmask = fp.build(depths=[src["depth"]], K=src["K"], source={"from_run": "x"},
                            contract=src["contract"], joints=src["joints"],
                            t_base_cam=src["t_base_cam"],
                            robot_mask_fn=lambda d, *a: np.zeros(d.shape, bool))
    assert pk["joints"]["status"] == "from_source" and pk["robot_mask"]["status"] == "ok"
    assert pk["degrades"] == [] and pk["plane"]["angle_to_up_deg"] < 1.0
