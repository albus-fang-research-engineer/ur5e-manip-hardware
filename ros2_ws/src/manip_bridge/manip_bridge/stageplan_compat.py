"""The ONE place hardware imports the sim's stage-plan module
(scripts/plan_stages.py): its artifact reader (load_bindings) and the
verification pieces plan_scene's --grounding pass reuses (verify,
prune_parts, blocking, TaskSpec, and the pool loader plan_stages itself
uses, selection.load_pool).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

Why it exists: hardware's call-#1 driver (outputs/runs/plan_scene.py)
writes its artifact in plan_stages' layout so that calls #2/#3 consume the
same files through load_bindings -- the reader that refuses an artifact
whose gate blocks it. Before grounding the artifact carries a blocking
`roles[deferred]`; plan_scene --grounding prunes the plan, verifies it and
binds roles STRUCTURALLY (no task contract on hardware: active-only stage =
grasp), and the tests read the result through THIS reader, not by shape.

STRUCTURAL_SPEC / load_structural: hardware has no task contract, so it
verifies with an empty TaskSpec (identity/grounds/menu checks run, no
coverage/role/ordering) and must READ with it too -- load_bindings'
default spec is sim's tasks/pour_tea.json, whose `after` ordering would
otherwise be applied to any hardware role that shares a pour-tea role
name. Hardware never calls load_bindings without a spec; it calls
load_structural.

scripts/ has no __init__.py, so `scripts.plan_stages` resolves as a
NAMESPACE package off /opt/manip-sim -- any other scripts/ dir earlier on
PYTHONPATH shadows it silently (same caveat as render_compat). Importing
it pulls manip_sim.selection/scene/vlm (numpy/scipy, no GL) and reads
tasks/pour_tea.json from the sim checkout as its default task spec.
"""

try:
    import scripts.plan_stages as _ps
    from scripts.plan_stages import (Bindings, TaskSpec, blocking, load_bindings, load_pool,
                                     prune_parts, verify)
except ImportError as e:
    raise ImportError(
        "stageplan_compat: cannot import scripts.plan_stages. The sim checkout must be "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH, and sim's scripts/ is a NAMESPACE "
        "package, so any other scripts/ dir earlier on PYTHONPATH shadows it silently -- "
        "check inside the container:\n"
        "    ls /opt/manip-sim/scripts/plan_stages.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'") from e

from manip_bridge.sim_provenance import provenance as _provenance

import manip_sim.selection as _sel  # noqa: E402  (already imported by plan_stages)

SOURCES = (_ps.__file__, _sel.__file__)

STRUCTURAL_SPEC = TaskSpec(name="structural (hardware: no task contract)",
                           required_tags={}, roles={}, ordering=())


def load_structural(path):
    """load_bindings with the hardware spec, never sim's pour-tea default."""
    return load_bindings(path, STRUCTURAL_SPEC)


__all__ = ["Bindings", "TaskSpec", "STRUCTURAL_SPEC", "blocking", "load_bindings",
           "load_structural", "load_pool", "prune_parts", "verify", "provenance", "SOURCES"]


def provenance():
    return _provenance(SOURCES)
