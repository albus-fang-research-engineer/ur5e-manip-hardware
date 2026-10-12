"""The ONE place hardware imports the sim's stage-plan artifact reader
(scripts/plan_stages.py: load_bindings).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

Why it exists now: hardware's call-#1 driver (outputs/runs/plan_scene.py)
writes its artifact in plan_stages' layout so that calls #2/#3 can later
consume the same files through load_bindings -- the reader that refuses an
artifact whose gate blocks it. Until role binding exists on hardware, every
hardware artifact carries a `roles[deferred]` blocking check, and the test
that pins the shared-layout claim reads the artifact through THIS reader,
not by its shape. Calls #2/#3 on hardware import load_bindings from here.

scripts/ has no __init__.py, so `scripts.plan_stages` resolves as a
NAMESPACE package off /opt/manip-sim -- any other scripts/ dir earlier on
PYTHONPATH shadows it silently (same caveat as render_compat). Importing
it pulls manip_sim.selection/scene/vlm (numpy/scipy, no GL) and reads
tasks/pour_tea.json from the sim checkout as its default task spec.
"""

try:
    import scripts.plan_stages as _ps
    from scripts.plan_stages import Bindings, load_bindings
except ImportError as e:
    raise ImportError(
        "stageplan_compat: cannot import scripts.plan_stages. The sim checkout must be "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH, and sim's scripts/ is a NAMESPACE "
        "package, so any other scripts/ dir earlier on PYTHONPATH shadows it silently -- "
        "check inside the container:\n"
        "    ls /opt/manip-sim/scripts/plan_stages.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'") from e

from manip_bridge.sim_provenance import provenance as _provenance

SOURCES = (_ps.__file__,)

__all__ = ["Bindings", "load_bindings", "provenance", "SOURCES"]


def provenance():
    return _provenance(SOURCES)
