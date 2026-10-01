"""The ONE place hardware imports the sim's geometric axis refinement.

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
When the shared modules move to a library, this file's import lines change
and nothing else does.

Two sim modules, one compat, deliberately: manip_sim.refine_frame imports
manip_sim.refine (RefineResult, _unit, ...) at module level, so they cannot
fail independently -- a second file would isolate nothing. Both are
numpy/scipy only (refine_frame also pulls manip_sim.tsr, itself
numpy/scipy), so importing this module is GL-free and works outside the
ros2 container whenever the sim checkout is on PYTHONPATH.

Contract used by canonical_frame.py:
  refine_axis(points, coarse, feature, max_snap_deg) -> RefineResult
      fitted line signed toward `coarse`; rejection is a typed result
      (accepted=False, direction = coarse), never a SystemExit
  azimuth_from_semantic(front, up) -> AzimuthResult
      projection only, sigma = SEMANTIC_SIGMA_DEG / conditioning
  assemble_frame(up_result, candidates) -> FrameResult
      R columns [x=front, y=up x front, z=up]; empty/rejected candidates
      give a deterministic arbitrary x with accepted=False

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""

try:
    import manip_sim.refine as _refine
    import manip_sim.refine_frame as _refine_frame
    from manip_sim.refine import MAX_SNAP_DEG, RefineResult, refine_axis
    from manip_sim.refine_frame import (SEMANTIC_SIGMA_DEG, AzimuthResult, FrameResult,
                                        assemble_frame, azimuth_from_semantic)
except ImportError as e:
    raise ImportError(
        "refine_compat: cannot import manip_sim.refine / manip_sim.refine_frame. The sim "
        "checkout must be bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro). A missing host path makes Docker "
        "mount an EMPTY dir without any error -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/refine.py\n"
        "    echo $PYTHONPATH\n"
        "and on the host:  docker compose config | grep -B1 -A1 manip-sim") from e

from manip_bridge.sim_provenance import provenance as _provenance

SOURCES = (_refine.__file__, _refine_frame.__file__)

__all__ = ["MAX_SNAP_DEG", "SEMANTIC_SIGMA_DEG", "RefineResult", "AzimuthResult", "FrameResult",
           "refine_axis", "azimuth_from_semantic", "assemble_frame", "provenance", "SOURCES"]


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files imported here; goes into canonical.json per asset."""
    return _provenance(SOURCES)
