"""TRELLIS.2 sidecar tests: image(+mask) -> canonical mesh, plus the metric
scale recovery branch scored against the sim mesh's true extents.

Generation is minutes-long GPU work, so everything heavy is @slow (deselect
with -m "not slow"). One generation is shared module-wide via a fixture.

Scale check logic: the server registers the masked depth cloud onto the
canonical unit-box mesh, so (canonical extent x recovered scale) should land
near the GT visual mesh's max extent. TRELLIS geometry from one view is
approximate — 35% relative tolerance is the "not off by 2x / not unit-box
confusion" test, which is what actually bites downstream (FoundationPose
registration with a wrongly scaled mesh).
"""

import numpy as np
import pytest

OBJ = "mug"  # solid-of-revolution-ish: friendliest case for single-view genai
GEN_TIMEOUT = 900_000


def _gt_extent(packet):
    trimesh = pytest.importorskip("trimesh")
    mesh_path = packet.mesh_path(OBJ)
    if mesh_path is None:
        pytest.skip(f"{OBJ}.obj not in packet dir")
    return float(max(trimesh.load(str(mesh_path), force="mesh").extents))


@pytest.fixture(scope="module")
def generated(trellis2, packet):
    if OBJ not in packet.objects:
        pytest.skip(f"{OBJ} not in packet")
    rep = trellis2.ok(
        {"op": "generate",
         "rgb": packet["rgb"],
         "mask": packet[f"mask_{OBJ}"].astype(np.uint8),
         "depth": packet["depth"].astype(np.float32),
         "K": packet["K"].astype(np.float64),
         "seed": 42,
         "decimation_target": 50_000,   # plenty for tests, much faster
         "texture_size": 512,
         "output_name": f"test_{OBJ}"},
        timeout_ms=GEN_TIMEOUT)
    return rep


def test_ping(trellis2):
    assert trellis2.ok({"op": "ping"})["ok"]


def test_unknown_op_errors_cleanly(trellis2):
    rep = trellis2.call({"op": "definitely_not_an_op"})
    assert rep.get("ok") is False and "unknown" in rep.get("error", "")


@pytest.mark.slow
def test_generate_canonical_geometry(generated):
    v = np.asarray(generated["vertices"], np.float32)
    f = np.asarray(generated["faces"], np.int64)
    assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 100
    assert f.ndim == 2 and f.shape[1] == 3 and len(f) > 100
    assert f.min() >= 0 and f.max() < len(v)
    # canonical frame contract: AABB inside [-0.5, 0.5]^3 (small slack for
    # decimation/remesh wiggle)
    assert v.min() >= -0.55 and v.max() <= 0.55
    # and it should actually USE the box, not be a speck
    assert float(max(v.max(0) - v.min(0))) > 0.5
    assert generated["gen_time"] > 0
    assert generated["glb_path"].endswith(".glb")


@pytest.mark.slow
def test_glb_written_and_readable(generated, packet):
    """The bridge reads GLBs from the shared trellis2_runtime/outputs mount —
    verify the file landed host-side and parses."""
    from pathlib import Path

    from conftest import REPO_ROOT

    host = REPO_ROOT / "trellis2_runtime" / "outputs" / Path(generated["glb_path"]).name
    if not host.exists():
        pytest.skip(f"{host} not found — outputs mount differs on this box")
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(str(host))
    assert len(scene.geometry) >= 1 if hasattr(scene, "geometry") else len(scene.vertices) > 0


@pytest.mark.slow
def test_metric_scale_recovery(generated, packet):
    if "metric" not in generated:
        pytest.skip("server did not run the metric branch (depth/K path)")
    m = generated["metric"]
    if not m["ok"]:
        pytest.skip(f"similarity registration flagged unreliable "
                    f"(rmse {m['rmse']*1000:.1f} mm) — geometry too degenerate "
                    "for this view; not a wire-protocol failure")
    v = np.asarray(generated["vertices"], np.float32)
    canonical_extent = float(max(v.max(0) - v.min(0)))
    recovered = canonical_extent * float(m["scale"])
    gt = _gt_extent(packet)
    rel = abs(recovered - gt) / gt
    print(f"[trellis2] metric extent: recovered {recovered*100:.1f} cm "
          f"vs GT {gt*100:.1f} cm (rel err {rel:.1%}, rmse {m['rmse']*1000:.1f} mm)")
    assert rel < 0.35
    # rotation part of the similarity must be a rotation
    from scipy.spatial.transform import Rotation as R
    q = np.asarray(m["q_xyzw"], np.float64)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-3)
    R.from_quat(q)  # raises if degenerate
