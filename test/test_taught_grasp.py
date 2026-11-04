"""outputs/runs/taught_grasp.py -- the pure part: library grasps in the body
frame of one mesh file, carried into a run by its registration. Offline (no
ROS). Numbers from run 20261104_051339 (hand-eye and the teapot's
pose_on_final, verbatim) and the committed library file.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "outputs" / "runs" / "library" / "lib_teapot.taught_grasps.json"

T_BASE_CAM = [[0.039391162832567356, 0.5475137822277278, -0.8358690056231219, 0.8705860389909645],
              [0.999029125690202, -0.005065195912056494, 0.043762424670521605, -0.14222476195496275],
              [0.019726690380516387, -0.8367813346753504, -0.5471817391192533, 0.3430058143987959],
              [0, 0, 0, 1]]
CAM_T_TEAPOT = [[-0.24339346671461576, -0.14124790922946723, -0.9595877492438852, 0.12351386994123459],
                [0.6693488467240101, -0.7404576252269514, -0.060783440446402826, 0.06606866419315338],
                [-0.7019485321231743, -0.6570932455759746, 0.27476667350418094, 0.484902560710907],
                [0, 0, 0, 1]]


@pytest.fixture(scope="module")
def tg():
    spec = importlib.util.spec_from_file_location("taught_grasp", REPO_ROOT / "outputs" / "runs" / "taught_grasp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def lib():
    return json.loads(LIB.read_text())


def summary(mesh="/data/any6d/lib_teapot.obj"):
    o = {"pose_on_final": {"cam_T_obj": CAM_T_TEAPOT}}
    if mesh:
        o["final_mesh_override"] = mesh
    return {"packet": {"t_base_cam": {"T": T_BASE_CAM}}, "objects": {"m2": o}}


def test_level_grasp_lands_on_the_strap_with_a_horizontal_approach(tg, lib):
    rows = tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"])
    assert [r["name"] for r in rows] == [g["name"] for g in lib["grasps"]]     # file order kept
    level = rows[0]
    # the strap centroid measured on the mesh, placed through this run's pose
    np.testing.assert_allclose(level["T_base_e"][:3, 3], [0.503, 0.089, 0.042], atol=2e-3)
    # level: approach within ~2 deg of horizontal; from +y (outside the loop) toward the body
    assert abs(level["elevation_deg"]) < 2.5
    assert level["approach_base"][1] < -0.99
    # the tilted ones descend by their tilt, same pad centre and closing axis
    for r, tilt in zip(rows[1:], (15, 30)):
        assert r["elevation_deg"] == pytest.approx(tilt, abs=2.5)
        np.testing.assert_allclose(r["T_base_e"][:3, 3], level["T_base_e"][:3, 3], atol=1e-9)
        np.testing.assert_allclose(r["T_base_e"][:3, 0], level["T_base_e"][:3, 0], atol=1e-6)
    for r in rows:                                                           # proper rotations
        Rm = r["T_base_e"][:3, :3]
        np.testing.assert_allclose(Rm.T @ Rm, np.eye(3), atol=1e-5)
        assert np.linalg.det(Rm) == pytest.approx(1.0, abs=1e-5)


def test_refuses_another_mesh_or_a_run_without_final_mesh(tg, lib):
    with pytest.raises(tg.TaughtRefusal, match="different mesh"):
        tg.resolve(summary(), "m2", lib, mesh_sha="0" * 64)
    with pytest.raises(tg.TaughtRefusal, match="not registered with --final-mesh"):
        tg.resolve(summary(mesh=None), "m2", lib, mesh_sha=lib["mesh_sha256"])
    with pytest.raises(tg.TaughtRefusal, match="no objects"):
        tg.resolve(summary(), "m7", lib, mesh_sha=lib["mesh_sha256"])


def test_refuses_a_closing_axis_not_perpendicular_to_the_approach(tg, lib):
    bad = dict(lib, grasps=[dict(lib["grasps"][0], closing_body=lib["grasps"][0]["approach_body"])])
    with pytest.raises(tg.TaughtRefusal, match="off perpendicular"):
        tg.resolve(summary(), "m2", bad, mesh_sha=lib["mesh_sha256"])


def test_deeper_moves_each_pad_centre_along_its_own_approach(tg, lib):
    base = tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"])
    deep = tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"], deeper=0.007)
    for b, d in zip(base, deep):
        np.testing.assert_allclose(d["T_base_e"][:3, 3] - b["T_base_e"][:3, 3], 0.007 * b["approach_base"], atol=1e-12)
        np.testing.assert_allclose(d["T_base_e"][:3, :3], b["T_base_e"][:3, :3], atol=1e-12)
    for bad in (-0.001, 0.03):
        with pytest.raises(tg.TaughtRefusal, match="must be within"):
            tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"], deeper=bad)


def test_up_moves_each_pad_centre_along_base_z_only(tg, lib):
    base = tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"])
    up = tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"], up=0.01, deeper=0.007)
    for b, u in zip(base, up):
        np.testing.assert_allclose(u["T_base_e"][:3, 3] - b["T_base_e"][:3, 3],
                                   0.007 * b["approach_base"] + [0, 0, 0.01], atol=1e-12)
        np.testing.assert_allclose(u["T_base_e"][:3, :3], b["T_base_e"][:3, :3], atol=1e-12)
    with pytest.raises(tg.TaughtRefusal, match="must be within"):
        tg.resolve(summary(), "m2", lib, mesh_sha=lib["mesh_sha256"], up=0.05)
