"""The ONE place hardware imports the sim's grounding renderer.

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
When the shared modules move to a library, this file's import line changes
and nothing else does. One file per module keeps import failures isolated:
this module breaking does not break marking (marks_compat) or anything else.

Render-only: `render_depth_views` imports mujoco and scripts.render_candidates
lazily AT CALL TIME, so importing this module stays GL-free; MuJoCo + osmesa
live in the ros2 container (docker/Dockerfile.ros2), which is where callers
run. View parity with `ground_object` is structural, not enforced here:
ground_object calls this same `render_depth_views` (scripts/ground_parts.py)
with cameras derived deterministically from the vertex bounds, so two calls
on the same asset dir agree by construction.

Import note: sim's scripts/ has no __init__.py, so `scripts.ground_parts`
resolves as a NAMESPACE package off /opt/manip-sim -- any other `scripts/`
dir earlier on PYTHONPATH shadows it silently. The ImportError below covers
both that and the empty-mount case.

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""

try:
    import manip_sim.proposal as _proposal
    import scripts.ground_parts as _gp
    from manip_sim.proposal import load_obj
    from scripts.ground_parts import VIEW_PX, render_depth_views
except ImportError as e:
    raise ImportError(
        "render_compat: cannot import scripts.ground_parts / manip_sim.proposal. The sim "
        "checkout must be bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro). A missing host path makes Docker "
        "mount an EMPTY dir without any error, and sim's scripts/ is a NAMESPACE package, so "
        "any other scripts/ dir earlier on PYTHONPATH shadows it silently -- check inside "
        "the container:\n"
        "    ls /opt/manip-sim/scripts/ground_parts.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'\n"
        "    echo $PYTHONPATH\n"
        "and on the host:  docker compose config | grep -B1 -A1 manip-sim") from e

from manip_bridge.sim_provenance import provenance as _provenance

SOURCES = (_gp.__file__, _proposal.__file__)

__all__ = ["VIEW_PX", "render_depth_views", "load_obj", "provenance", "SOURCES"]


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files imported here; goes into the masks manifest per run."""
    return _provenance(SOURCES)
