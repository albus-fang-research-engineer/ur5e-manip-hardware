"""pose_decide: the yaw decision, every outcome class, and parity between
the offline driver (outputs/runs/fp_from_mesh.py --oriany) and the bridge's
transport (pose_decide.sidecar_fns over SidecarClients).

Synthetic register reply: 12 scorer-sorted hypotheses rotating about the
camera y axis; the re-rank keeps ranks 0, 5, 9. Each survivor render is a
flat colour; a fake Orient Anything answers by the crop's dominant colour,
so every survivor has a known front. The real crop is red and reads yaw 0.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
sys.path.insert(0, str(BRIDGE))
from manip_bridge import pose_decide as pd  # noqa: E402

H, W = 120, 160
UP = [0.0, -1.0, 0.0]                        # camera y is down; up is -y


def Ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def front(deg):
    return (Ry(deg) @ np.array([1.0, 0, 0])).tolist()


def oa(deg, alpha=1):
    R = Ry(deg)
    return {"ok": True, "alpha": alpha, "front_cam": front(deg), "up_cam": UP,
            "R_cam": np.stack([R @ [1, 0, 0], R @ [0, 0, 1], np.array(UP)], 1).tolist()}


REAL = (200, 40, 40)
CROP_COLOURS = {0: (40, 40, 200), 5: (40, 200, 40), 9: (200, 200, 40)}


def make_frame():
    rgb = np.full((H, W, 3), 90, np.uint8)
    mask = np.zeros((H, W), bool)
    mask[40:90, 50:110] = True
    rgb[mask] = REAL
    return rgb, mask


def make_reply(mask, survivors=(0, 5, 9), to_rank=0, reason="scorer pick already covers U",
               crops=True, u_frac=0.12):
    hyp = np.stack([np.eye(4) for _ in range(12)])
    for i in range(12):
        hyp[i, :3, :3] = Ry(30 * i)
        hyp[i, :3, 3] = [0.1, 0.0, 0.5]
    # every RerankRecord field (pose_server/rerank.py), as the sidecar sends it
    rr = {"enabled": True, "changed": to_rank != 0, "reason": reason, "from_rank": 0, "to_rank": to_rank,
          "rotation_deg": 30.0 * to_rank, "expl_from": 0.2, "expl_to": 0.8, "n_survivors": len(survivors),
          "mask_px": 3000, "u_px_raw": 500, "u_px": int(3000 * u_frac), "u_frac": u_frac,
          "u_depth_dropped": 0, "dilate_px": 3, "depth_band_m": 0.02, "max_expl": 0.8,
          "scorer_from": 1.0, "scorer_to": 0.9, "depth_bad_from": 0.1, "depth_bad_to": 0.1,
          "n_gated": 6, "max_expl_gated": 0.8, "survivors": list(survivors) or None}
    rep = {"ok": True, "pose": hyp[to_rank].astype(np.float32), "hypotheses": hyp.astype(np.float32),
           "scores": np.linspace(1, 0.5, 12).astype(np.float32), "rerank": rr, "texture": "simple"}
    if crops and survivors:
        box = pd.crop_box(mask)
        h, w = box[1] - box[0], box[3] - box[2]
        rep["crops"] = np.stack([np.full((h, w, 3), CROP_COLOURS.get(r, (1, 2, 3)), np.uint8)
                                 for r in survivors])
        rep["crop_box"] = box
    return rep


def dominant(img):
    px = np.asarray(img).reshape(-1, 3)
    px = px[(px != 255).any(1)]
    vals, counts = np.unique(px, axis=0, return_counts=True)
    return tuple(int(v) for v in vals[counts.argmax()])


def make_orient(table):
    """table: colour -> OA reply; records every call."""
    calls = []

    def orient(img):
        calls.append(dominant(img))
        return table[dominant(img)]
    return orient, calls


DEFAULT_TABLE = {REAL: oa(0), CROP_COLOURS[0]: oa(180), CROP_COLOURS[5]: oa(6), CROP_COLOURS[9]: oa(90)}


# ------------------------------------------------------------------ pieces

def test_crop_box_is_the_sidecars_formula_and_clips():
    m = np.zeros((H, W), bool)
    m[40:90, 50:110] = True                   # ptp 49 x 59 -> half int(59 * 0.7) = 41
    assert pd.crop_box(m) == [int(64.5 - 41), int(64.5 + 41), int(79.5 - 41), int(79.5 + 41)]
    m = np.zeros((H, W), bool)
    m[0:30, 0:30] = True
    y0, y1, x0, x1 = pd.crop_box(m)
    assert y0 == 0 and x0 == 0 and y1 <= H and x1 <= W


def test_yaw_about_and_fold():
    up = np.array(UP)
    # front(30) = Ry(30) x; about up = -y that is a -30 deg turn
    assert pd.yaw_about(up, np.array(front(0)), np.array(front(30))) == pytest.approx(-30, abs=1e-6)
    assert pd.fold(185, 1) == pytest.approx(175) and pd.fold(185, 2) == pytest.approx(5)
    assert pd.fold(-95, 4) == pytest.approx(5)


# ---------------------------------------------------------------- outcomes

def run(rep=None, table=None, select=None, mask_frame=None):
    rgb, mask = make_frame()
    rep = rep if rep is not None else make_reply(mask)
    orient, calls = make_orient(table or DEFAULT_TABLE)
    sel_calls = []

    def default_select(rank):
        sel_calls.append(rank)
        return rep["hypotheses"][rank]
    T, rec, _ = pd.decide(rep, rgb, mask, orient, select or default_select)
    return T, rec, calls, sel_calls


def test_fires_and_selects_the_survivor_whose_yaw_agrees():
    T, rec, calls, sel = run()
    assert rec["reason"] == "oriany_changed" and rec["decider_fired"] and not rec["hard_stop"]
    assert rec["chosen_rank"] == 5 and sel == [5]
    assert np.allclose(T, make_reply(make_frame()[1])["hypotheses"][5])
    assert calls[0] == REAL and len(calls) == 4                    # real crop first, then 3 survivors
    assert rec["crop_box_source"] == "sidecar" and rec["crop_box_match"] is True
    assert rec["front"]["independent"] is False and rec["front"]["alpha"] == 1


def test_agrees_without_select():
    rgb, mask = make_frame()
    T, rec, _, sel = run(rep=make_reply(mask, to_rank=5, reason="re-ranked"))
    assert rec["reason"] == "oriany_agrees" and T is None and sel == []


def test_front_is_in_the_selected_body_frame():
    T, rec, *_ = run()
    fb = np.asarray(rec["front"]["front_body"])
    assert np.allclose(np.asarray(T)[:3, :3] @ fb, front(0), atol=1e-6)


def test_hard_stop_on_body_disagreement_still_reads_the_real_crop():
    rgb, mask = make_frame()
    rep = make_reply(mask, survivors=(), reason="top-K disagree on the body (u_frac too large): "
                     "registration unreliable, hard stop upstream", u_frac=0.5)
    T, rec, calls, sel = run(rep=rep)
    assert rec["hard_stop"] and rec["reason"] == "rerank_u_frac" and T is None and sel == []
    assert calls == [REAL] and rec["oa_real"]["alpha"] == 1
    assert rec["front"]["front_body"] is None and "hard stop" in rec["front"]["note"]


def test_decline_at_u_floor_still_measures_front_with_the_bridges_box():
    """The cup: nothing unexplained, no survivors, no sidecar crop_box. The
    driver used to skip Orient Anything here; the front must be measured."""
    rgb, mask = make_frame()
    rep = make_reply(mask, survivors=(), reason="nothing unexplained by the body consensus (|U| below floor)")
    T, rec, calls, _ = run(rep=rep)
    assert rec["reason"] == "rerank_u_floor" and not rec["hard_stop"] and T is None
    assert calls == [REAL] and rec["crop_box_source"] == "bridge" and rec["crop_box"] == pd.crop_box(mask)
    assert rec["front"]["front_body"] is not None and rec["front"]["independent"] is True


def test_real_crop_alpha0_degrades():
    _, rec, *_ = run(table={**DEFAULT_TABLE, REAL: oa(0, alpha=0)})
    assert rec["reason"] == "oa_alpha0" and rec["chosen_rank"] == 0
    assert rec["front"]["front_body"] is None and "alpha 0" in rec["front"]["note"]


def test_none_confirmable_degrades():
    t = {**DEFAULT_TABLE, CROP_COLOURS[0]: oa(180, 2), CROP_COLOURS[5]: oa(6, 0), CROP_COLOURS[9]: oa(90, 2)}
    _, rec, *_ = run(table=t)
    assert rec["reason"] == "oa_none_confirmable" and rec["chosen_rank"] == 0


def test_ambiguous_degrades():
    """Ranks 5 and 9 both read ~6 deg but are 120 deg apart as poses."""
    _, rec, *_ = run(table={**DEFAULT_TABLE, CROP_COLOURS[9]: oa(8)})
    assert rec["reason"] == "oa_ambiguous" and rec["chosen_rank"] == 0 and not rec["decider_fired"]


def test_oriany_unavailable_degrades():
    def dead(img):
        raise TimeoutError("tcp://127.0.0.1:5673: no reply to 'orient'")
    rgb, mask = make_frame()
    T, rec, _ = pd.decide(make_reply(mask), rgb, mask, dead, lambda r: None)
    assert rec["reason"] == "oriany_unavailable" and T is None and rec["front"]["source"] is None


def test_select_failure_keeps_the_sidecar_pick():
    def broken(rank):
        raise RuntimeError("no session")
    T, rec, *_ = run(select=broken)
    assert rec["reason"] == "select_failed" and rec["chosen_rank"] == 0 and T is None
    assert rec["front"]["independent"] is True                     # nothing fired in the end


def test_alpha2_front_is_withheld():
    t = {REAL: oa(0, 2), CROP_COLOURS[0]: oa(180, 2), CROP_COLOURS[5]: oa(6, 2), CROP_COLOURS[9]: oa(90, 2)}
    _, rec, *_ = run(table=t)
    assert rec["front"]["front_body"] is None and "180 deg" in rec["front"]["note"]


def test_crops_missing_is_named():
    rgb, mask = make_frame()
    _, rec, *_ = run(rep=make_reply(mask, crops=False))
    assert rec["reason"] == "crops_missing"


def test_record_is_json_serialisable():
    _, rec, *_ = run()
    assert json.loads(json.dumps(rec)) == rec


# ------------------------------------------------------------------ parity

def _serve(sock, handler, stop):
    import msgpack
    while not stop.is_set():
        if sock.poll(100):
            req = msgpack.unpackb(sock.recv(), raw=False)
            sock.send(msgpack.packb(handler(req), use_bin_type=True))


@pytest.fixture(params=["ok", "select_error", "oriany_error"])
def fake_sidecars(request):
    zmq = pytest.importorskip("zmq")
    pytest.importorskip("msgpack_numpy").patch()
    rgb, mask = make_frame()
    rep = make_reply(mask)
    ctx = zmq.Context.instance()
    pose, ori = ctx.socket(zmq.REP), ctx.socket(zmq.REP)
    pp, op = pose.bind_to_random_port("tcp://127.0.0.1"), ori.bind_to_random_port("tcp://127.0.0.1")

    def pose_h(req):
        c = req["cmd"]
        if c in ("ping", "release"):
            return {"ok": True}
        if c == "register":
            assert req["rerank"] and req["survivor_crops"] and req["return_all"]
            return rep
        if c == "select":
            if request.param == "select_error":
                return {"ok": False, "error": f"no session '{req['obj']}'"}
            return {"ok": True, "pose": rep["hypotheses"][int(req["rank"])]}
        return {"ok": False, "error": c}

    def ori_h(req):
        if req["cmd"] == "ping":
            return {"ok": True}
        if request.param == "oriany_error":
            return {"ok": False, "error": "CUDA out of memory"}
        return DEFAULT_TABLE[dominant(req["image"])]

    stop = threading.Event()
    ts = [threading.Thread(target=_serve, args=(s, h, stop), daemon=True)
          for s, h in ((pose, pose_h), (ori, ori_h))]
    for t in ts:
        t.start()
    yield f"tcp://127.0.0.1:{pp}", f"tcp://127.0.0.1:{op}", rgb, mask, request.param
    stop.set()
    for t in ts:
        t.join(1)
    pose.close(0)
    ori.close(0)


def test_driver_and_bridge_transport_make_the_same_decision(fake_sidecars, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from manip_bridge.zmq_client import SidecarClient
    """On every path -- success, a select error reply, an Orient Anything
    error reply -- the driver writes its JSON and it matches the bridge's
    record exactly (error text included: same clients, same exceptions)."""
    pose_addr, ori_addr, rgb, mask, mode = fake_sidecars

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cv2.imwrite(str(run_dir / "rgb.png"), rgb[..., ::-1])
    cv2.imwrite(str(run_dir / "mask_m2.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(run_dir / "depth_mm.png"), np.full((H, W), 500, np.uint16))
    (run_dir / "summary.json").write_text(json.dumps({"frame": {"K": [[150, 0, 80], [0, 150, 60], [0, 0, 1]]}}))

    env = {**os.environ, "PYTHONPATH": str(BRIDGE)}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "outputs" / "runs" / "fp_from_mesh.py"), str(run_dir),
                        "--mesh", "/data/any6d/final_mesh_m2.obj", "--object", "m2", "--oriany",
                        "--addr", pose_addr, "--oriany-addr", ori_addr, "--timeout", "20"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    driver = json.loads((run_dir / "fp_m2.json").read_text())

    # the bridge's transport: PoseBridge.estimate_payload's flags, then post_register's calls
    pose_c, ori_c = SidecarClient(pose_addr, 20000), SidecarClient(ori_addr, 20000)
    rep = pose_c.call({"cmd": "register", "obj": "m2", "mesh": "/data/any6d/final_mesh_m2.obj",
                       "rgb": rgb, "depth": np.full((H, W), 0.5, np.float32),
                       "K": np.eye(3), "mask": mask.astype(np.uint8), "est_refine_iter": 5,
                       "rerank": True, "survivor_crops": True, "return_all": True})
    orient, select = pd.sidecar_fns(ori_c, pose_c, "m2")
    T_sel, bridge, _ = pd.decide(rep, rgb, mask, orient, select)

    assert json.loads(json.dumps(bridge)) == driver["decision"]
    expect = {"ok": ("oriany_changed", 5), "select_error": ("select_failed", 0),
              "oriany_error": ("oriany_unavailable", 0)}[mode]
    assert (bridge["reason"], bridge["chosen_rank"]) == expect
    if mode == "ok":
        assert np.allclose(np.asarray(driver["cam_T_obj"]), T_sel, atol=1e-6)
    else:
        assert T_sel is None and "SidecarError" in bridge["note"]
