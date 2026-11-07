"""The ONE place hardware imports the sim's stage-plan module
(scripts/plan_stages.py): its artifact reader (load_bindings) and the
verification and binding pieces plan_scene's --grounding pass reuses
(verify, bind_roles, prune_parts, blocking, TaskSpec/load_task_spec, and
the pool loader plan_stages itself uses, selection.load_pool).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

Why it exists: hardware's call-#1 driver (outputs/runs/plan_scene.py)
writes its artifact in plan_stages' layout so that calls #2/#3 consume the
same files through load_bindings -- the reader that refuses an artifact
whose gate blocks it. Before grounding the artifact carries a blocking
`roles[deferred]`; plan_scene --grounding prunes the plan, then verifies it
and binds roles against a SIM TASK CONTRACT (tasks/<name>.json, default
DEFAULT_TASK_SPEC = tasks/pour_tea.json) with sim's own verify/bind_roles,
exactly as plan_stages' run_one does.

load_plan: the artifact records the contract FILE it was bound with
(hardware.task_spec_file); load_plan reads it back with that contract, so
the reader applies the same `after`/optional rules the binding did.
Hardware calls load_plan, never bare load_bindings (whose fallback is the
pour-tea default whatever the artifact was bound with).

scripts/ has no __init__.py, so `scripts.plan_stages` resolves as a
NAMESPACE package off /opt/manip-sim -- any other scripts/ dir earlier on
PYTHONPATH shadows it silently (same caveat as render_compat). Importing
it pulls manip_sim.selection/scene/vlm (numpy/scipy, no GL) and reads
tasks/pour_tea.json from the sim checkout as its default task spec.
"""

try:
    import scripts.plan_stages as _ps
    from scripts.plan_stages import (DEFAULT_TASK_SPEC, Bindings, TaskSpec, bind_roles, blocking,
                                     load_bindings, load_pool, load_task_spec, prune_parts,
                                     verify)
except ImportError as e:
    raise ImportError(
        "stageplan_compat: cannot import scripts.plan_stages. The sim checkout must be "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH, and sim's scripts/ is a NAMESPACE "
        "package, so any other scripts/ dir earlier on PYTHONPATH shadows it silently -- "
        "check inside the container:\n"
        "    ls /opt/manip-sim/scripts/plan_stages.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'") from e

import json
from pathlib import Path

from manip_bridge.sim_provenance import provenance as _provenance

import manip_sim.selection as _sel  # noqa: E402  (already imported by plan_stages)

SOURCES = (_ps.__file__, _sel.__file__)


def load_plan(path):
    """load_bindings with the contract plan_scene --grounding bound the
    artifact with (hardware.task_spec_file). Refuses, as the reader does
    (SystemExit), an artifact that records none."""
    doc = json.loads(Path(path).read_text())
    f = (doc.get("hardware") or {}).get("task_spec_file")
    if not f:
        load_bindings(path)          # a blocked gate (roles[deferred]) is the reader's refusal
        raise SystemExit(f"[stageplan-compat] {path} records no task contract "
                         "(hardware.task_spec_file) -- re-run plan_scene.py --replay ... "
                         "--grounding to bind it")
    return load_bindings(path, load_task_spec(f))


__all__ = ["Bindings", "TaskSpec", "DEFAULT_TASK_SPEC", "bind_roles", "blocking",
           "load_bindings", "load_plan", "load_pool", "load_task_spec", "prune_parts",
           "verify", "provenance", "SOURCES"]


def provenance():
    return _provenance(SOURCES)
