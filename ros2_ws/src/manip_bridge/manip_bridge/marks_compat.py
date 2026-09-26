"""The ONE place hardware imports the sim's mark-set code.

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
When the shared modules move to a library, this file's import line changes
and nothing else does. One file per module keeps import failures isolated:
a later compat module that pulls in mujoco breaking does not break marking.

The names below are the contract a future library must provide. marks.py
is numpy + PIL only (PIL imported lazily), so nothing sim-side enters the
bridge image. load_gt is deliberately NOT exported: marks.gt.json is the
sim verifier's ground truth and has no meaning on hardware.

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""

try:
    import manip_sim.perception.marks as _marks
    from manip_sim.perception.marks import MIN_AREA_PX, build_marks, load_marks, render_marks
except ImportError as e:
    raise ImportError(
        "marks_compat: cannot import manip_sim.perception.marks. The sim checkout must be "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro). A missing host path makes Docker "
        "mount an EMPTY dir without any error -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/perception/marks.py\n"
        "    echo $PYTHONPATH\n"
        "and on the host:  docker compose config | grep -B1 -A1 manip-sim") from e

from manip_bridge.sim_provenance import provenance as _provenance

SOURCES = (_marks.__file__,)

__all__ = ["MIN_AREA_PX", "build_marks", "load_marks", "render_marks", "provenance", "SOURCES"]


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files imported here; goes into summary.json per run."""
    return _provenance(SOURCES)
