"""scene_marks: plane fit, every drop reason, and the mark-set writer.

Synthetic scene, camera looking straight down (OpenCV frame, +z forward):
table at z = 0.6 m over the left 70% of the image, floor at z = 1.4 m past
its edge. Heights above the table are therefore 0.6 - z. Selection tests
need no sim checkout; the writer test uses it via marks_compat and skips
without it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"))
from manip_bridge import scene_marks as sm  # noqa: E402

SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "manip_sim" / "perception" / "marks.py").is_file()

H, W = 240, 320
K = [[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]]
TABLE_Z, FLOOR_Z, EDGE = 0.6, 1.4, 224


def box(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(1)
    depth = np.full((H, W), TABLE_Z)
    depth[:, EDGE:] = FLOOR_Z
    obj = {
        "cube":      (box(100, 140, 60, 100), 0.50),    # 10 cm tall, on the table
        "teapot":    (box(60, 100, 120, 180), 0.45),
        "lid":       (box(60, 72, 140, 160), 0.43),     # inside the teapot's mask
        "shelf":     (box(100, 140, 285, 315), 0.40),   # 20 cm up, but beside the table
        "tall":      (box(170, 200, 120, 150), 0.05),   # 55 cm above the table
        "edge":      (box(150, 190, 0, 30), 0.50),      # touches x = 0
        "arm":       (box(0, 40, 100, 160), 0.30),      # touches y = 0 too
    }
    for name in ("cube", "teapot", "lid", "shelf", "tall", "edge", "arm"):
        m, z = obj[name]
        depth[m] = z
    depth = depth + rng.normal(0, 0.001, depth.shape)
    hole = box(200, 230, 180, 210)                      # a no-depth object (glass)
    depth[hole] = 0.0
    obj["glass"] = (hole, None)
    table = np.zeros((H, W), bool)
    table[:, :EDGE] = True
    for name in ("cube", "teapot", "shelf", "tall", "edge", "arm", "glass"):
        table &= ~obj[name][0]
    floor = box(0, 60, 240, 320)
    return depth, obj, table, floor


def inst(prompt, mask, score):
    return {"prompt": prompt, "score": score, "mask": mask}


def run(scene, instances, **kw):
    depth, *_ = scene
    plane = sm.fit_table_plane(depth, K)
    kept, rep = sm.select_instances(instances, depth=depth, K=K, plane=plane, **kw)
    reasons = {r["prompt"]: r["reasons"] for r in rep["instances"]}
    return kept, rep, reasons, plane


# ------------------------------------------------------------------- plane

def test_plane_fit_finds_the_table_oriented_toward_the_camera(scene):
    depth, *_ = scene
    p = sm.fit_table_plane(depth, K, up_hint=[0, 0, -1])
    assert np.allclose(p["n"], [0, 0, -1], atol=0.01)       # camera side positive
    assert p["d"] == pytest.approx(TABLE_Z, abs=0.003)
    assert p["angle_to_up_deg"] < 1.0
    # extent: the table spans image columns 0..EDGE at z = 0.6
    E = np.asarray(p["basis"])
    j = int(np.argmax(np.abs(E[0])))                      # the in-plane axis along camera x
    xs = [(c - 160) * TABLE_Z / 300 for c in (0, EDGE - 1)]
    lo, hi = sorted(np.sign(E[0, j]) * x for x in xs)
    assert p["extent_lo"][j] == pytest.approx(lo, abs=0.02)
    assert p["extent_hi"][j] == pytest.approx(hi, abs=0.02)


def test_plane_fit_is_deterministic(scene):
    depth, *_ = scene
    assert sm.fit_table_plane(depth, K) == sm.fit_table_plane(depth, K)


def test_plane_fit_needs_points():
    assert sm.fit_table_plane(np.zeros((H, W)), K) is None


# --------------------------------------------------------------- selection

def test_every_drop_reason(scene):
    depth, obj, table, floor = scene
    arm_mask = obj["arm"][0]
    instances = [inst("cube", obj["cube"][0], 0.37), inst("teapot", obj["teapot"][0], 0.35),
                 inst("lid", obj["lid"][0], 0.30), inst("shelf", obj["shelf"][0], 0.33),
                 inst("tall", obj["tall"][0], 0.31), inst("edge", obj["edge"][0], 0.40),
                 inst("arm", obj["arm"][0], 0.43), inst("glass", obj["glass"][0], 0.28),
                 inst("table", table, 0.29), inst("floor", floor, 0.25),
                 inst("faint", obj["cube"][0], 0.15), inst("speck", box(120, 125, 190, 195), 0.3),
                 inst("cube_again", obj["cube"][0], 0.26)]
    kept, rep, reasons, _ = run(scene, instances, arm_mask=arm_mask, arm_source="bg_prompt")
    assert {it["prompt"] for it in kept} == {"cube", "teapot"}
    assert reasons["lid"] == [f"part_of:{reasons_idx(rep, 'teapot')}"]
    assert reasons["shelf"] == ["out_of_workspace"]
    assert reasons["tall"] == ["out_of_workspace"]
    assert reasons["floor"] == ["truncated", "out_of_workspace"]
    assert reasons["edge"] == ["truncated"]
    assert reasons["arm"] == ["truncated", "arm"]           # every reason, not just the first
    assert reasons["glass"] == ["no_depth"]
    assert reasons["table"] == ["truncated", "table"]
    assert reasons["faint"] == ["below_score"]          # dedup only among clean instances
    assert reasons["speck"] == ["below_min_area", "no_depth"]   # 25 px < min_depth_px too
    assert reasons["cube_again"] == [f"duplicate_of:{reasons_idx(rep, 'cube')}"]
    assert rep["geometry_used"] and rep["arm_source"] == "bg_prompt"
    cube = next(r for r in rep["instances"] if r["prompt"] == "cube")
    assert cube["geometry"]["median_h"] == pytest.approx(0.10, abs=0.005)


def reasons_idx(rep, prompt):
    return next(r["idx"] for r in rep["instances"] if r["prompt"] == prompt)


def test_idx_is_score_order_like_the_probe(scene):
    _, obj, *_ = scene
    kept, rep, *_ = run(scene, [inst("b", obj["teapot"][0], 0.2), inst("a", obj["cube"][0], 0.4)])
    assert [(r["idx"], r["prompt"]) for r in rep["instances"]] == [(1, "a"), (2, "b")]


def test_two_same_category_instances_stay_two(scene):
    """The prompt-keyed collapse bug: same prompt, disjoint masks, both kept."""
    _, obj, *_ = scene
    kept, *_ = run(scene, [inst("object", obj["cube"][0], 0.37),
                           inst("object", obj["teapot"][0], 0.35)])
    assert len(kept) == 2


def test_arm_is_a_fraction_of_the_instance_not_iou(scene):
    """An instance 85% inside a SMALLER arm mask (arm + gripper overshoot):
    IoU would be ~0.85 too here, so also check the case IoU misses -- the
    instance mostly inside a much larger arm mask."""
    _, obj, *_ = scene
    cube = obj["cube"][0]
    big_arm = box(90, 150, 50, 110)                      # contains the whole cube
    _, _, reasons, _ = run(scene, [inst("object", cube, 0.4)], arm_mask=big_arm)
    assert "arm" in reasons["object"]                    # IoU(cube, big_arm) is only ~0.44
    small = cube.copy(); small[130:140] = False          # 75% of the cube
    _, _, reasons, _ = run(scene, [inst("object", cube, 0.4)], arm_mask=small)
    assert "arm" in reasons["object"]
    tiny = cube.copy(); tiny[112:] = False               # 30% of the cube
    _, _, reasons, _ = run(scene, [inst("object", cube, 0.4)], arm_mask=tiny)
    assert reasons["object"] == []


def test_geometry_skipped_without_depth_or_trusted_plane(scene):
    depth, obj, table, _ = scene
    kept, rep = sm.select_instances([inst("table", table, 0.3), inst("cube", obj["cube"][0], 0.4)])
    assert rep["geometry_note"].startswith("skipped") and not rep["geometry_used"]
    assert all("table" not in r["reasons"] for r in rep["instances"])
    plane = dict(sm.fit_table_plane(depth, K), inlier_frac=0.05)
    _, rep = sm.select_instances([inst("cube", obj["cube"][0], 0.4)], depth=depth, K=K, plane=plane)
    assert "inlier_frac" in rep["geometry_note"] and not rep["geometry_used"]


def test_params_carry_a_basis_and_overrides_are_marked(scene):
    assert all(isinstance(b, str) and b for _, b in sm.PARAMS.values())
    _, obj, *_ = scene
    _, rep, reasons, _ = run(scene, [inst("object", obj["cube"][0], 0.25)], params={"score_min": 0.3})
    assert reasons["object"] == ["below_score"]
    assert rep["params"]["score_min"] == {"value": 0.3, "basis": "override: " + sm.PARAMS["score_min"][1]}
    with pytest.raises(KeyError):
        sm.values({"score_mni": 0.3})


def test_empty_input():
    kept, rep = sm.select_instances([])
    assert kept == [] and rep["instances"] == []


# ------------------------------------------------------------------ writer

@pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
def test_write_mark_set_maps_ids_and_keeps_prompts_out_of_marks_json(scene, tmp_path):
    pytest.importorskip("PIL")
    sys.path.insert(0, str(SIM_DIR))
    try:
        _, obj, *_ = scene
        kept, _, _, _ = run(scene, [inst("teapot-ish", obj["teapot"][0], 0.35),
                                    inst("cube-ish", obj["cube"][0], 0.37)])
        rgb = np.full((H, W, 3), 128, np.uint8)
        ms, id_map = sm.write_mark_set(rgb, kept, tmp_path)
    finally:
        sys.path.remove(str(SIM_DIR))
    # reading order: the teapot (rows 60-100) comes before the cube (rows 100-140)
    by_mid = json.loads((tmp_path / "marks.sam3.json").read_text())["marks"]
    assert by_mid["1"]["prompt"] == "teapot-ish" and by_mid["2"]["prompt"] == "cube-ish"
    assert id_map == {1: 2, 2: 1}                              # mark id -> instance idx
    marks_json = (tmp_path / "marks.json").read_text()
    assert "teapot-ish" not in marks_json and "cube-ish" not in marks_json
    assert ms.ids() == (1, 2) and (tmp_path / "marked.png").is_file()


# ------------------------------------------------------ run_scene plumbing

def test_obj_key_and_register_parsing():
    assert sm.obj_key(3) == "m3"
    ids = [3, 1, 2]
    assert sm.parse_register("none", ids) == [] and sm.parse_register(None, ids) == []
    assert sm.parse_register("all", ids) == [1, 2, 3]
    assert sm.parse_register("2", ids) == [2]
    assert sm.parse_register(" 3, 1 ", ids) == [1, 3] == sm.parse_register("3 1 3", ids)
    with pytest.raises(ValueError, match=r"\[4\]: not marks"):
        sm.parse_register("2,4", ids)                    # never silently shrink the set
    with pytest.raises(ValueError, match="none, all, or mark ids"):
        sm.parse_register("mug", ids)


def test_split_instances_keeps_every_object_score_and_thresholds_only_bg():
    a, b, arm1, arm2 = (box(10, 20, 10, 20), box(30, 40, 30, 40),
                        box(0, 5, 0, 50), box(0, 5, 60, 100))
    inst, bg, n = sm.split_instances(
        ["object", "object", "robot arm", "robot arm", "table"],
        [a.astype(np.uint8) * 255, b.astype(np.uint8) * 255, arm1 * 255, arm2 * 255, a * 255],
        [0.35, 0.12, 0.53, 0.2], {"object"}, {"robot arm"}, 0.4)
    assert [(i["prompt"], i["score"]) for i in inst] == [("object", 0.35), ("object", pytest.approx(0.12))]
    assert inst[0]["mask"].dtype == bool and np.array_equal(inst[0]["mask"], a)
    assert n == 1 and np.array_equal(bg, arm1)           # 0.2 arm instance below bg_score_min
    inst, bg, n = sm.split_instances(["object"], [a], [0.3], {"object"}, set(), 0.4)
    assert bg is None and n == 0


@pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
def test_load_mark_set_round_trips_write_mark_set(scene, tmp_path):
    pytest.importorskip("PIL")
    sys.path.insert(0, str(SIM_DIR))
    try:
        _, obj, *_ = scene
        kept, *_ = run(scene, [inst("object", obj["teapot"][0], 0.35),
                               inst("object", obj["cube"][0], 0.37)])
        sm.write_mark_set(np.full((H, W, 3), 128, np.uint8), kept, tmp_path)
        masks, sam = sm.load_mark_set(tmp_path)
    finally:
        sys.path.remove(str(SIM_DIR))
    assert sorted(masks) == [1, 2]
    assert np.array_equal(masks[1], obj["teapot"][0]) and np.array_equal(masks[2], obj["cube"][0])
    assert sam["marks"]["2"]["score"] == pytest.approx(0.37)
