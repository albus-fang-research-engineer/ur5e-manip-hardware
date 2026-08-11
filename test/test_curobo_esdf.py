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
This tests the whole chain — depth units, intrinsics, pose convention, PBA+
ESDF sign convention — without hardcoding scene coordinates.

The first integrate/esdf pays NVRTC + Warp JIT (docstring says up to ~a
minute cold), hence the fat timeouts. The module restores the server's
env-default mapper config at the end so a long-running sidecar isn't left
reconfigured to test settings.
"""

import numpy as np
import pytest

FIRST_CALL_TIMEOUT = 300_000
CALL_TIMEOUT = 60_000


@pytest.fixture(scope="module")
def mapper(curobo, packet):
    h, w = packet["depth"].shape
    rep = curobo.ok({"cmd": "configure",
                     "voxel_size": 0.01,
                     "extent": [2.0, 2.0, 1.5],
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
    assert rep["voxel_size"] > 0

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
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    fine = mapper.ok({"cmd": "esdf", "voxel_size": 0.01}, timeout_ms=CALL_TIMEOUT)
    coarse = mapper.ok({"cmd": "esdf", "voxel_size": 0.02}, timeout_ms=CALL_TIMEOUT)
    assert np.isclose(coarse["voxel_size"], 0.02, atol=1e-6)
    assert np.prod(coarse["grid_shape"]) < np.prod(fine["grid_shape"])


def test_reset_clears_observations(mapper, packet):
    _integrate(mapper, packet, FIRST_CALL_TIMEOUT)
    mapper.ok({"cmd": "reset"}, timeout_ms=CALL_TIMEOUT)
    rep = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    p_free, p_occ = _surface_probe_points(packet)
    d_free, d_occ = _esdf_lookup(rep, [p_free, p_occ])
    # after reset both probes are unobserved — no surface should separate them
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
    mapper.ok({"cmd": "load", "name": "test_packet_blocks.pt"},
              timeout_ms=CALL_TIMEOUT)
    after = mapper.ok({"cmd": "esdf"}, timeout_ms=CALL_TIMEOUT)
    p_free, p_occ = _surface_probe_points(packet)
    b = _esdf_lookup(before, [p_free, p_occ])
    a = _esdf_lookup(after, [p_free, p_occ])
    assert np.allclose(a, b, atol=0.02), f"save/load drift: {b} -> {a} ({saved['path']})"


def test_stats(mapper):
    rep = mapper.ok({"cmd": "stats"}, timeout_ms=CALL_TIMEOUT)
    assert rep["memory_mb"] > 0
