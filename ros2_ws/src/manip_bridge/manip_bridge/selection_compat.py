"""The ONE place hardware imports the sim's call-#2 resolution
(manip_sim.selection) and the symbol loader it resolves against
(manip_sim.frames.load_symbols).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
frames.load_symbols rides here rather than in a file of its own because its
only hardware caller is resolve_selection's input (render_compat pairs
ground_parts with proposal.load_obj the same way).

What select_scene uses, all GL-free (numpy/scipy):
  load_pool, vlm_subset, menu_from_pool   the call-#2 menu, rebuilt from the
                                          SAME candidates.json the marked
                                          views were drawn from
  resolve_selection, ResolutionError      the typed gate: a licit mark that
                                          cannot anchor a frame fails here
  selection_to_json, load_selections,     the role-keyed selections artifact,
  selection_objects                       written and read back in sim format
  load_symbols                            frames.json -> Symbols
"""

try:
    import manip_sim.frames as _frames
    import manip_sim.selection as _sel
    from manip_sim.frames import load_symbols
    from manip_sim.selection import (ResolutionError, load_pool, load_selections,
                                     menu_from_pool, resolve_selection, selection_objects,
                                     selection_to_json, vlm_subset)
except ImportError as e:
    raise ImportError(
        "selection_compat: cannot import manip_sim.selection / manip_sim.frames. The sim "
        "checkout must be bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro) -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/selection.py\n"
        "    echo $PYTHONPATH") from e

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

SOURCES = (_sel.__file__, _frames.__file__)

__all__ = ["ResolutionError", "load_pool", "load_selections", "load_symbols", "menu_from_pool",
           "resolve_selection", "selection_objects", "selection_to_json", "vlm_subset",
           "provenance", "SOURCES"]


def provenance():
    return _provenance(SOURCES)
