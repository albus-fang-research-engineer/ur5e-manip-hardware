"""The ONE place hardware imports the sim's call-#3 pieces: the TSR compiler
gate (manip_sim.compile_tsr), the sim driver's stage list and retry budget
(scripts.emit_constraints), and -- lazily -- the stage-frame renderer
(scripts.render_stage_frames).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
The other sim pieces call #3 needs come through their own compats:
selection_compat.selected_points (the anchor points) and vlm_compat
(entry_line, PROMPT_DELTAS, TEMPERATURE, MAX_PARSE_RETRIES).

What emit_scene uses from here:
  compile_stage, check_pair_consistency,   the compile gate, exactly as the
  check_stage_seam, stage_accounting,      sim's emit_constraints.py runs it
  CompileError, ALIGN_TOL_RAD
  bound_stages, COMPILE_RETRIES,           the sim driver's stage list (one call
  point_token                              per bound role with an active
                                           object), retry budget, token lookup
  stage_frames_renderer()                  (compute_stage_frames, render_role,
                                           scene_extent, file): imports mujoco,
                                           so loaded AT CALL TIME, never at import

Import note: sim's scripts/ has no __init__.py, so `scripts.emit_constraints`
resolves as a NAMESPACE package off /opt/manip-sim -- any other `scripts/`
dir earlier on PYTHONPATH shadows it silently (same caveat as render_compat).

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""
import importlib

try:
    import manip_sim.compile_tsr as _ct
    import scripts.emit_constraints as _ec
    from manip_sim.compile_tsr import (ALIGN_TOL_RAD, CompileError, check_pair_consistency,
                                       check_stage_seam, compile_stage, stage_accounting)
    from scripts.emit_constraints import COMPILE_RETRIES, bound_stages
    from scripts.emit_constraints import _point_token as point_token
except ImportError as e:
    raise ImportError(
        "emit_compat: cannot import manip_sim.compile_tsr / scripts.emit_constraints. The sim "
        "checkout must be bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro), and sim's scripts/ is a NAMESPACE "
        "package, so any other scripts/ dir earlier on PYTHONPATH shadows it silently -- "
        "check inside the container:\n"
        "    ls /opt/manip-sim/scripts/emit_constraints.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'") from e

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

RENDER_MODULE = "scripts.render_stage_frames"
SOURCES = (_ct.__file__, _ec.__file__)

__all__ = ["ALIGN_TOL_RAD", "COMPILE_RETRIES", "CompileError", "bound_stages",
           "check_pair_consistency", "check_stage_seam", "compile_stage", "point_token",
           "stage_accounting", "stage_frames_renderer", "provenance", "SOURCES"]


def stage_frames_renderer():
    """(compute_stage_frames, render_role, scene_extent, module file). Imports
    mujoco (via scripts.render_candidates), so only the framed arm calls it."""
    m = importlib.import_module(RENDER_MODULE)
    return m.compute_stage_frames, m.render_role, m.scene_extent, m.__file__


def provenance(render: bool = False):
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files imported here (+ the renderer when the framed arm ran)."""
    src = SOURCES + ((stage_frames_renderer()[3],) if render else ())
    return _provenance(src)
