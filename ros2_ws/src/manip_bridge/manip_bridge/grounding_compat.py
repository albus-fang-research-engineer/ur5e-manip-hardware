"""The ONE place hardware imports the sim's part grounding (ground_object).

Rule: every sim capability hardware uses gets its own *_compat.py, and
nothing else in this repo imports manip_sim (test/test_sim_imports.py
enforces it). When the shared modules move to a library, this file's import
lines change and nothing else does.

Declared deviation from one-compat-per-module: this file and render_compat.py
both import the SAME sim module, scripts/ground_parts.py. They are split by
capability, not by module: render_compat is render-only (task D's views, GL
at call time, nothing grounded), this one grounds (lift + fit + symbols +
depth semantics). Folding them would make render_compat's render-only
contract false for the masks driver that depends on it, for no runtime
saving (Python caches the module; both names refer to the same object).
Accepted cost: at extraction, two files re-point instead of one.

Requires the sim checkout to carry the grounding seam (branch
`grounding-seam` off insert-marker a23de1e): ground_object raising
GroundingError instead of SystemExit, taking up_sigma_deg, and annotating
every point with depth_check + winding (part_grounding.annotate_depth_checks,
with provenance.depth_semantics from mesh_orientation). An older checkout
imports fine and would ground silently WITHOUT those fields, so the
capability check below refuses it at import, naming the branch. The
container mounts whatever ~/ur5e-manip-sim has checked out -- switching that
checkout changes what this imports, which is why every output records sim
provenance.

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""
import inspect
import os

try:
    import manip_sim.part_grounding as _pg
    import manip_sim.proposal as _proposal
    import scripts.ground_parts as _gp
    from scripts.ground_parts import ground_object
except ImportError as e:
    raise ImportError(
        "grounding_compat: cannot import scripts.ground_parts / manip_sim.part_grounding. The "
        "sim checkout must be bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro). A missing host path makes Docker "
        "mount an EMPTY dir without any error, and sim's scripts/ is a NAMESPACE package, so "
        "any other scripts/ dir earlier on PYTHONPATH shadows it silently -- check inside "
        "the container:\n"
        "    ls /opt/manip-sim/scripts/ground_parts.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'\n"
        "    echo $PYTHONPATH") from e

_missing = [n for n, ok in (
    ("scripts.ground_parts.GroundingError", hasattr(_gp, "GroundingError")),
    ("ground_object(up_sigma_deg=...)",
     "up_sigma_deg" in inspect.signature(ground_object).parameters),
    ("manip_sim.part_grounding.annotate_depth_checks", hasattr(_pg, "annotate_depth_checks")),
    ("manip_sim.part_grounding.mesh_orientation(V, F)",
     hasattr(_pg, "mesh_orientation")
     and list(inspect.signature(_pg.mesh_orientation).parameters)[:2] == ["V", "F"]),
) if not ok]
if _missing:
    raise ImportError(
        "grounding_compat: the mounted sim checkout predates the grounding seam (missing "
        f"{', '.join(_missing)}). Check out the seam branch on the host:\n"
        "    cd ~/ur5e-manip-sim && git checkout grounding-seam\n"
        f"(imported from {_gp.__file__})")

GroundingError = _gp.GroundingError

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

# the sim files whose code decides a grounding result: driver, lift/fit/
# symbols/depth semantics, mesh loading + sampling + pool, the view cameras.
# render_candidates imports mujoco at module level (ground_parts imports it
# lazily at render time), so it is located, not imported: importing this
# compat stays GL-free.
SOURCES = (_gp.__file__, _pg.__file__, _proposal.__file__,
           os.path.join(os.path.dirname(_gp.__file__), "render_candidates.py"))

__all__ = ["ground_object", "GroundingError", "provenance", "SOURCES"]


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files that decide a grounding result; goes into every frames.json
    the seam writes."""
    return _provenance(SOURCES)
