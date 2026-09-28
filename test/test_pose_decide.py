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


def oa(deg, alpha=1, rel=None):
    """A fake Orient Anything reply reading front at `deg`; `rel` is what
    orient_rel reports for this image against the real crop (default: deg)."""
    R = Ry(deg)
    return {"ok": True, "alpha": alpha, "front_cam": front(deg), "up_cam": UP,
            "R_cam": np.stack([R @ [1, 0, 0], R @ [0, 0, 1], np.array(UP)], 1).tolist(),
            "_rel": deg if rel is None else rel}


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
    """table: colour -> OA reply. Returns (orient, calls, orient_rel, rel_calls)."""
    calls, rel_calls = [], []

    def orient(img):
        calls.append(dominant(img))
        return table[dominant(img)]

    def orient_rel(ref, tgt):
        rel_calls.append(dominant(tgt))
        return {"ok": True, "rel_azimuth": table[dominant(tgt)]["_rel"] - table[dominant(ref)]["_rel"]}
    return orient, calls, orient_rel, rel_calls


def colour(rank):
    return (10 + (37 * rank) % 200, 60, 230 - (53 * rank) % 150)


def scenario(real, survivors, pick, n_hyp=40):
    """real: (read_deg, alpha); survivors: {rank: (pose_deg, read_deg, alpha[, rel_deg])}.
    -> (rep, table). Hypothesis `rank` is Ry(pose_deg); the re-rank picked `pick`."""
    rgb, mask = make_frame()
    hyp = np.stack([np.eye(4) for _ in range(n_hyp)])
    for rk, v in survivors.items():
        hyp[rk, :3, :3] = Ry(v[0])
    box = pd.crop_box(mask)
    h, w = box[1] - box[0], box[3] - box[2]
    ranks = list(survivors)
    rep = {"ok": True, "pose": hyp[pick].astype(np.float32), "hypotheses": hyp.astype(np.float32),
           "rerank": {"reason": "re-ranked", "to_rank": pick, "survivors": ranks, "u_frac": 0.1},
           "crops": np.stack([np.full((h, w, 3), colour(r), np.uint8) for r in ranks]), "crop_box": box}
    table = {REAL: oa(*real)}
    for rk, v in survivors.items():
        table[colour(rk)] = oa(v[1], v[2], v[3] if len(v) > 3 else None)
    return rep, table


def decide_on(rep, table, with_rel=True, rel_override=None):
    rgb, mask = make_frame()
    orient, calls, orient_rel, rel_calls = make_orient(table)
    sel = []

    def select(rank):
        sel.append(rank)
        return rep["hypotheses"][rank]
    T, rec, _ = pd.decide(rep, rgb, mask, orient, select,
                          orient_rel=(rel_override or orient_rel) if with_rel else None)
    return T, rec, sel, rel_calls


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


def test_yaw_about():
    up = np.array(UP)
    # front(30) = Ry(30) x; about up = -y that is a -30 deg turn
    assert pd.yaw_about(up, np.array(front(0)), np.array(front(30))) == pytest.approx(-30, abs=1e-6)
    assert abs(pd.yaw_about(up, np.array(front(0)), np.array(front(180)))) == pytest.approx(180, abs=1e-6)


# ---------------------------------------------------------------- outcomes

def run(rep=None, table=None, select=None, mask_frame=None):
    rgb, mask = make_frame()
    rep = rep if rep is not None else make_reply(mask)
    orient, calls, orient_rel, _ = make_orient(table or DEFAULT_TABLE)
    sel_calls = []

    def default_select(rank):
        sel_calls.append(rank)
        return rep["hypotheses"][rank]
    T, rec, _ = pd.decide(rep, rgb, mask, orient, select or default_select, orient_rel=orient_rel)
    return T, rec, calls, sel_calls


def test_fires_and_selects_the_survivor_whose_yaw_agrees():
    T, rec, calls, sel = run()
    assert rec["reason"] == "oriany_changed" and rec["decider_fired"] and not rec["hard_stop"]
    assert rec["chosen_rank"] == 5 and sel == [5]
    assert np.allclose(T, make_reply(make_frame()[1])["hypotheses"][5])
    assert calls[0] == REAL and len(calls) == 4                    # real crop first, then 3 survivors
    assert rec["crop_box_source"] == "sidecar" and rec["crop_box_match"] is True
    assert rec["front"]["independent"] is False and rec["front"]["alpha_reported"] == 1
    assert rec["corroboration"]["proposed"] == {"rank": 5, "rel_az": pytest.approx(6)}


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


def test_render_alpha_never_gates():
    """Renders reading alpha 2 / 0 / 2 against a real crop reading 1: the
    alpha-equality rule would have discarded rank 5; the front decides."""
    t = {**DEFAULT_TABLE, CROP_COLOURS[0]: oa(180, 2), CROP_COLOURS[5]: oa(6, 0), CROP_COLOURS[9]: oa(90, 2)}
    _, rec, *_ = run(table=t)
    assert rec["reason"] == "oriany_changed" and rec["chosen_rank"] == 5
    assert [r["alpha"] for r in rec["rows"]] == [2, 0, 2]               # recorded all the same


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


def test_alpha2_front_is_given_with_a_caveat():
    t = {REAL: oa(0, 2), CROP_COLOURS[0]: oa(180, 2), CROP_COLOURS[5]: oa(6, 2), CROP_COLOURS[9]: oa(90, 2)}
    _, rec, *_ = run(table=t)
    fr = rec["front"]
    assert fr["front_body"] is not None and fr["alpha_reported"] == 2
    assert "unreliable" in fr["note"] and "180 deg" in fr["note"]


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
        if c == "crops":
            box = pd.crop_box(np.asarray(req["mask"]) > 0)
            h, w = box[1] - box[0], box[3] - box[2]
            return {"ok": True, "crop_box": box, "ranks": list(req["ranks"]),
                    "crops": np.stack([np.full((h, w, 3), CROP_COLOURS.get(int(r), (1, 2, 3)), np.uint8)
                                       for r in req["ranks"]])}
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
        if req["cmd"] == "orient_rel":
            ref, tgt = DEFAULT_TABLE[dominant(req["image_ref"])], DEFAULT_TABLE[dominant(req["image_tgt"])]
            return {"ok": True, "rel_azimuth": tgt["_rel"] - ref["_rel"]}
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
    orient, select, orient_rel = pd.sidecar_fns(ori_c, pose_c, "m2")
    T_sel, bridge, _ = pd.decide(rep, rgb, mask, orient, select, orient_rel=orient_rel)

    assert json.loads(json.dumps(bridge)) == driver["decision"]
    expect = {"ok": ("oriany_changed", 5), "select_error": ("select_failed", 0),
              "oriany_error": ("oriany_unavailable", 0)}[mode]
    assert (bridge["reason"], bridge["chosen_rank"]) == expect
    if mode == "ok":
        assert np.allclose(np.asarray(driver["cam_T_obj"]), T_sel, atol=1e-6)
    else:
        assert T_sel is None and "SidecarError" in bridge["note"]


@pytest.mark.parametrize("fake_sidecars", ["ok"], indirect=True)
def test_driver_dumps_crops_for_oriany_check(fake_sidecars, tmp_path):
    """--dump-crops writes what oriany_check.py reads: real_crop.png and
    render_rank<N>.png for rank 0, the final pose and --crop-ranks, all in
    one framing (same size)."""
    cv2 = pytest.importorskip("cv2")
    from PIL import Image
    pose_addr, ori_addr, rgb, mask, _ = fake_sidecars
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cv2.imwrite(str(run_dir / "rgb.png"), rgb[..., ::-1])
    cv2.imwrite(str(run_dir / "mask_m2.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(run_dir / "depth_mm.png"), np.full((H, W), 500, np.uint16))
    (run_dir / "summary.json").write_text(json.dumps({"frame": {"K": [[150, 0, 80], [0, 150, 60], [0, 0, 1]]}}))
    dump = tmp_path / "oa"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "outputs" / "runs" / "fp_from_mesh.py"), str(run_dir),
                        "--mesh", "/m.obj", "--object", "m2", "--oriany", "--addr", pose_addr,
                        "--oriany-addr", ori_addr, "--timeout", "20", "--dump-crops", str(dump),
                        "--crop-ranks", "9"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(BRIDGE)}, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    names = sorted(p.name for p in dump.iterdir())
    assert names == ["real_crop.png", "render_rank0.png", "render_rank5.png", "render_rank9.png"]
    sizes = {Image.open(dump / n).size for n in names}
    assert len(sizes) == 1                                   # one framing for all
    real = np.asarray(Image.open(dump / "real_crop.png"))
    assert dominant(real) == REAL                            # the masked object, white elsewhere


# ------------------------------------------------ the 20260926_234500 dump

DUMP = {16: (13, 13, 4), 0: (72, 72, 2), 1: (108, 108, 0), 2: (108, 108, 2)}  # rank: pose, read, alpha


def test_dump_numbers_old_rule_fails_new_rule_confirms_rank16():
    """Real crop alpha 2; renders read 13 (a4) / 72 (a2) / 108 (a0) / 108 (a2).
    The alpha-equality rule would have kept only ranks 0 and 2 -- both wrong,
    the correct rank 16 excluded. The new rule confirms rank 16 at 13 deg."""
    old_confirmable = {rk for rk, (_, _, a) in DUMP.items() if a != 0 and a == 2}
    assert old_confirmable == {0, 2}
    rep, table = scenario((0, 2), DUMP, pick=16)
    T, rec, sel, rel = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oriany_agrees", 16)
    assert rec["agreement_deg"] == pytest.approx(13, abs=1e-6)
    assert T is None and sel == [] and rel == []                      # confirming needs no orient_rel
    assert rec["front"]["front_body"] is not None and rec["front"]["independent"] is True


def test_dump_numbers_override_the_scorers_pick():
    """Same readings, but the re-rank had kept the scorer's rank 0: the
    decider overrides to rank 16, corroborated by orient_rel."""
    rep, table = scenario((0, 2), DUMP, pick=0)
    T, rec, sel, rel = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oriany_changed", 16) and sel == [16]
    assert rec["corroboration"]["proposed"]["rel_az"] == pytest.approx(13)
    assert rec["corroboration"]["rerank_pick"]["rel_az"] == pytest.approx(72)
    assert rec["front"]["independent"] is False


# --------------------------------------------------- flipped survivors

def test_flipped_survivor_reading_180_is_rejected():
    """The re-rank picked the handle flip (rank 9, 180 deg from rank 3);
    the front head reads it at 180: the decider takes rank 3."""
    rep, table = scenario((0, 2), {3: (90, 5, 2), 9: (270, 180, 2)}, pick=9)
    T, rec, sel, _ = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oriany_changed", 3) and sel == [3]


def test_flipped_survivor_reading_near_0_is_ambiguous():
    """If the front head cannot tell front from back, the flip reads near 0
    too: two candidates within the margin, 180 deg apart -> abstain."""
    rep, table = scenario((0, 2), {3: (90, 5, 2), 9: (270, 3, 2)}, pick=9)
    T, rec, sel, _ = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oa_ambiguous", 9) and sel == [] and T is None
    assert "180 deg apart" in rec["note"]


def test_wrong_peak_is_caught_by_orient_rel():
    """The front head landed on the wrong peak: the flip (rank 3) reads 5,
    the correct pick (rank 9) reads 180. orient_rel, comparing the images
    directly, disagrees -> no override."""
    rep, table = scenario((0, 2), {3: (90, 5, 2, 170), 9: (270, 180, 2, 8)}, pick=9)
    T, rec, sel, rel = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oa_rel_disagrees", 9) and sel == []
    assert sorted(rel) == sorted([colour(3), colour(9)])              # exactly the two calls


def test_override_without_orient_rel_abstains():
    rep, table = scenario((0, 2), DUMP, pick=0)
    _, rec, sel, _ = decide_on(rep, table, with_rel=False)
    assert rec["reason"] == "oa_rel_unavailable" and rec["chosen_rank"] == 0 and sel == []

    def dead(ref, tgt):
        raise TimeoutError("no reply to 'orient_rel'")
    _, rec, sel, _ = decide_on(rep, table, rel_override=dead)
    assert rec["reason"] == "oa_rel_unavailable" and "TimeoutError" in rec["note"] and sel == []


def test_same_family_noise_does_not_move_the_pose():
    """Your frame: survivors of one family read -13 and -9 deg. The -9 one
    must not replace the re-rank's pick on 4 deg of front noise."""
    rep, table = scenario((0, 2), {18: (0, -13, 4), 32: (8, -9, 4)}, pick=18)
    T, rec, sel, _ = decide_on(rep, table)
    assert (rec["reason"], rec["chosen_rank"]) == ("oriany_agrees", 18) and sel == [] and T is None
    assert rec["agreement_deg"] == pytest.approx(13)


def test_no_survivor_agrees():
    rep, table = scenario((0, 2), {0: (72, 72, 2), 1: (108, 108, 0)}, pick=0)
    _, rec, sel, _ = decide_on(rep, table)
    assert rec["reason"] == "oa_no_agreement" and rec["chosen_rank"] == 0 and sel == []
