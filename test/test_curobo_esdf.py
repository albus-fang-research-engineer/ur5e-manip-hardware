"""cuRobo mapper sidecar tests: depth -> TSDF integrate -> dense ESDF, over
the ZMQ wire, with sign/geometry checks derived from the packet's own depth.

Map frame: the Mapper's extent is centered on the map origin, so we place the
map frame at the table top (world shifted by -table_offset). The camera pose
sent to "integrate" is therefore T_map_cam = trans(-table_offset) @ T_world_cam.

Geometry check strategy (robust to any tabletop layout): pick an observed
table pixel, backproject it to the surface point along the camera ray, then
probe the ESDF
  - 12 cm in front of the surface (toward the camera): observed free space,
    ESDF should be clearly positive;
  - 4 cm behind the surface (away from the camera): carved/occupied, ESDF
    should be smaller than the free probe and at/below ~0.
This tests the whole chain -- depth units, intrinsics, pose convention, PBA+
ESDF sign convention -- without hardcoding scene coordinates.

ESDF grid semantics under test (cuRoboV2 >= 0.8.0.post1): the grid SHAPE is
fixed at Mapper construction from (extent, esdf_voxel_size); the per-call
voxel_size override rescales spacing on that fixed buffer, so element count
is constant and coverage scales with voxel size. The server passes an
explicit voxel size on every compute so upstream's sticky override cannot
leak one request's resolution into the next (regression-tested below).

The first integrate/esdf pays NVRTC + Warp JIT (docstring says up to ~a
minute cold), hence the fat timeouts. The module restores the server's
env-default mapper config at the end so a long-running sidecar isn't left
reconfigured to test settings.
"""

import numpy as np
import pytest

FIRST_CALL_TIMEOUT = 300_000
CALL_TIMEOUT = 60_000

EXTENT = [2.0, 2.0, 1.5]
VOXEL = 0.01


@pytest.fixture(scope="module")
def mapper(curobo, packet):
    h, w = packet["depth"].shape
    rep = curobo.ok({"cmd": "configure",
                     "voxel_size": VOXEL,
                     "esdf_voxel_size": VOXEL,
                     "extent": EXTENT,
                     "image_height": int(h), "image_width": int(w),
                     "num_cameras": 1},
                    timeout_ms=CALL_TIMEOUT)
    assert rep["memory_mb"] > 0
    yield curobo
    # leave the long-running sidecar in its env-default configuration
    curobo.ok({"cmd": "configure"}, timeout_ms=CALL_TIMEOUT)


def _T_map_cam(packet):
    T_map_world = np.eye(4)
    T_map_world[:3, 3] = -np.asarray(packet.meta["table_offset"], np.float64)
    return (T_map_world @ packet["T_world_cam"]).astype(np.float32)


def _integrate(mapper, packet, timeout):
    mapper.ok({"cmd": "integrate",
               "depth": packet["depth"].astype(np.float32),
               "intrinsics": packet["K"].astype(np.float32),
               "pose": _T_map_cam(packet),
               "rgb": packet["rgb"]},
              timeout_ms=timeout)


def _surface_probe_points(packet):
    """(free_point, occupied_point) in MAP frame, derived from an observed
    table pixel near the table-mask centroid."""
    depth, K = packet["depth"], packet["K"]
    table = packet["mask_table"] & (depth > 0)
    assert table.any(), "no observed table pixels in packet"
    vs, us = np.nonzero(table)
    cv_, cu_ = vs.mean(), us.mean()
    i = int(np.argmin((vs - cv_) ** 2 + (us - cu_) ** 2))
    v, u = int(vs[i]), int(us[i])
    z = float(depth[v, u])
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0],
                      (v - K[1, 2]) * z / K[1, 1], z])
    T = _T_map_cam(packet).astype(np.float64)
    p_map = T[:3, :3] @ p_cam + T[:3, 3]
    ray = T[:3, :3] @ (p_cam / np.linalg.norm(p_cam))  # camera->surface dir
    return p_map - 0.12 * ray, p_map + 0.04 * ray


def _esdf_lookup(rep, points):
    grid = np.asarray(rep["esdf"], np.float32).reshape(rep["grid_shape"])
    low = np.asarray(rep["low"], np.float64)
    vs = float(rep["voxel_size"])
    out = []
    for p in points:
        idx = np.floor((np.asarray(p) - low) / vs).astype(int)
        assert np.all(idx >= 0) and np.all(idx < grid.shape), \
            f"probe {p} outside grid low={low} shape={grid.shape}"
        out.append(float(grid[tuple(idx)]))  # X slowest, Z fastest (VoxelGrid)
    return out


def _check_grid_consistency(rep, expect_vs=None):
    """Invariants every esdf reply must satisfy: spacing implied by
    (high-low)/shape matches the reported voxel_size (catches mislabeled
    grids -- a client doing low + idx*vs silently reads wrong space
    otherwise), and optionally that voxel_size is the one requested."""
    shape = np.asarray(rep["grid_shape"])
    low, high = np.asarray(rep["low"]), np.asarray(rep["high"])
    implied = (high - low) / shape
    assert np.allclose(implied, rep["voxel_size"], atol=1e-4), \
        f"implied spacing {implied} != reported voxel_size {rep['voxel_size']}"
    if expect_vs is not None:
        assert np.isclose(rep["voxel_size"], expect_vs, atol=1e-6), \
            f"voxel_size {rep['voxel_size']} != expected {expect_vs}"


def test_ping(curobo):
    assert curobo.ok({"cmd": "ping"})["ok"]


def test_integrate_and_esdf_geometry(mapper, packet):
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    rep = mapper.ok({"cmd": "esdf"}, timeout_ms=FIRST_CALL_TIMEOUT)

    grid_shape = list(rep["grid_shape"])
    esdf = np.asarray(rep["esdf"], np.float32)
    assert esdf.size == int(np.prod(grid_shape))
    assert np.isfinite(esdf).all()
    low, high = np.asarray(rep["low"]), np.asarray(rep["high"])
    assert np.all(high > low)
    _check_grid_consistency(rep, expect_vs=VOXEL)

    # the ESDF window must cover the configured extent at the configured
    # resolution -- an unconfigured MapperCfg gives a 128^3 window that
    # silently crops the map, which no probe-based check would catch as
    # long as probes happen to land inside it
    span = high - low
    assert np.all(span >= np.asarray(EXTENT) - 2 * VOXEL), \
        f"ESDF span {span} does not cover configured extent {EXTENT}"

    p_free, p_occ = _surface_probe_points(packet)
    d_free, d_occ = _esdf_lookup(rep, [p_free, p_occ])
    print(f"[curobo] esdf probes: free {d_free*100:+.1f} cm, "
          f"behind-surface {d_occ*100:+.1f} cm")
    assert d_free > 0.03, f"observed free space should be clearly positive, got {d_free:.3f}"
    assert d_occ < d_free, "carved space not closer-to-surface than free probe"
    assert d_occ < 0.03, f"point 4 cm behind observed surface reads {d_occ:.3f} m"


def test_geometry_only_path(mapper, packet):
    """No rgb -> the server substitutes zeros; geometry must be unaffected."""
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    mapper.ok({"cmd": "integrate",
               "depth": packet["depth"].astype(np.float32),
               "intrinsics": packet["K"].astype(np.float32),
               "pose": _T_map_cam(packet)},
              timeout_ms=FIRST_CALL_TIMEOUT)
    rep = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    p_free, p_occ = _surface_probe_points(packet)
    d_free, d_occ = _esdf_lookup(rep, [p_free, p_occ])
    assert d_free > 0.03 and d_occ < d_free


def test_esdf_voxel_size_override(mapper, packet):
    """The override rescales spacing on a fixed-shape grid: element count is
    constant, coverage scales with voxel size. Assert exactly that, plus the
    spacing/label consistency that a mislabeled grid would violate."""
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    fine = mapper.ok({"cmd": "esdf", "voxel_size": 0.01}, timeout_ms=CALL_TIMEOUT)
    coarse = mapper.ok({"cmd": "esdf", "voxel_size": 0.02}, timeout_ms=CALL_TIMEOUT)

    _check_grid_consistency(fine, expect_vs=0.01)
    _check_grid_consistency(coarse, expect_vs=0.02)

    # fixed-dims by design (CUDA-graph-safe seeding kernel)
    assert list(fine["grid_shape"]) == list(coarse["grid_shape"])
    span_f = np.asarray(fine["high"]) - np.asarray(fine["low"])
    span_c = np.asarray(coarse["high"]) - np.asarray(coarse["low"])
    assert np.allclose(span_c, 2.0 * span_f, rtol=1e-3), \
        f"coarse coverage {span_c} should be 2x fine coverage {span_f}"


def test_esdf_override_not_sticky(mapper, packet):
    """Regression: upstream's per-call override mutates integrator state, so
    a naive server would return 0.02-resolution grids to every subsequent
    un-overridden request. This is the exact mechanism that broke the
    save/load roundtrip test (before at 0.02 vs after at 0.05)."""
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    mapper.ok({"cmd": "esdf", "voxel_size": 0.02}, timeout_ms=CALL_TIMEOUT)
    rep = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    _check_grid_consistency(rep, expect_vs=VOXEL)


def test_reset_clears_observations(mapper, packet):
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    rep = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    p_free, p_occ = _surface_probe_points(packet)
    d_free, d_occ = _esdf_lookup(rep, [p_free, p_occ])
    # after reset both probes are unobserved -- no surface should separate them
    assert abs(d_free - d_occ) < 0.08, (
        f"reset map still distinguishes surface sides "
        f"({d_free:.3f} vs {d_occ:.3f})")


def test_save_load_roundtrip(mapper, packet):
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    before = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    saved = mapper.ok({"cmd": "save", "name": "test_packet_blocks.pt"},
                      timeout_ms=CALL_TIMEOUT)
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    loaded = mapper.ok({"cmd": "load", "name": "test_packet_blocks.pt"},
                       timeout_ms=CALL_TIMEOUT)
    assert loaded["n_blocks"] > 0, "load imported zero blocks"
    after = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)

    # both grids must be at the same (configured) resolution before any
    # value comparison means anything -- the original failure here was a
    # 0.02-vs-0.05 resolution mismatch misread as data drift
    _check_grid_consistency(before, expect_vs=VOXEL)
    _check_grid_consistency(after, expect_vs=VOXEL)
    assert list(before["grid_shape"]) == list(after["grid_shape"])

    # full-grid comparison: a global clamp, sign flip, or offset is legible
    # from the stats in the failure message, unlike two probe values
    b = np.asarray(before["esdf"], np.float32)
    a = np.asarray(after["esdf"], np.float32)
    assert np.allclose(a, b, atol=0.005), (
        f"save/load drift ({saved['path']}): "
        f"before [{b.min():.4f}, {b.max():.4f}] "
        f"after [{a.min():.4f}, {a.max():.4f}], "
        f"max|diff|={np.abs(a - b).max():.4f}")

    p_free, p_occ = _surface_probe_points(packet)
    bp = _esdf_lookup(before, [p_free, p_occ])
    ap = _esdf_lookup(after, [p_free, p_occ])
    assert np.allclose(ap, bp, atol=0.005), f"probe drift: {bp} -> {ap}"


def test_load_rejects_mismatched_config(mapper, packet):
    """The save sidecar records the builder config; loading under a
    different one must fail loudly instead of pairing blocks with the wrong
    voxel size / extent. err() expects a {"ok": False} reply; if the client
    helper lacks it, assert on ok({...}) raising instead."""
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    mapper.ok({"cmd": "save", "name": "test_cfg_guard.pt"}, timeout_ms=CALL_TIMEOUT)

    h, w = packet["depth"].shape
    mapper.ok({"cmd": "configure", "voxel_size": 0.02, "esdf_voxel_size": 0.02,
               "extent": EXTENT, "image_height": int(h), "image_width": int(w),
               "num_cameras": 1}, timeout_ms=CALL_TIMEOUT)
    rep = mapper.err({"cmd": "load", "name": "test_cfg_guard.pt"},
                     timeout_ms=FIRST_CALL_TIMEOUT)
    assert "config" in rep["error"]

    # restore the module's test config for any remaining tests
    mapper.ok({"cmd": "configure", "voxel_size": VOXEL, "esdf_voxel_size": VOXEL,
               "extent": EXTENT, "image_height": int(h), "image_width": int(w),
               "num_cameras": 1}, timeout_ms=CALL_TIMEOUT)


def test_stats(mapper):
    rep = mapper.ok({"cmd": "stats"}, timeout_ms=CALL_TIMEOUT)
    assert rep["memory_mb"] > 0