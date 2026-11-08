"""The ONE place hardware imports the sim's VLM client (manip_sim.vlm).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).
When the shared modules move to a library, this file's import line changes
and nothing else does.

What hardware uses: call #1 (Client.plan_stages) with a mark-addressed
vocabulary (Vocabulary.from_marks over the run's mark set), and the typed
plan it returns. The client's transport is a plain callable
payload -> raw text, which is what makes record/replay possible without
touching sim code. The model is MANIP_VLM_MODEL (default
claude-sonnet-4-6), read when manip_sim.vlm is imported; the key is
ANTHROPIC_API_KEY. vlm.py is stdlib + numpy, so this import is GL-free.

Call #3 (emit_scene) also uses the emission-side pieces: entry_line (the
factual entry line), PROMPT_DELTAS / TEMPERATURE (recorded in the artifact)
and MAX_PARSE_RETRIES (the call budget it prints).

plan_stages_hw.py --method vlm reads call #3's artifact back with
emission_from_json (not load_emissions, which refuses an artifact whose
compile gate failed: the planner records that stage's failure instead) and
types its failures with FAILURE_TYPES, the names call #5 will consume.

Hardware-owned touchpoints (part_phrases.py) reuse the client's ask loop --
transport, bounded parse-rejection retry, CallLog -- with their own prompt
and parser through `ask()` below, so the one access to the client's private
`_ask` lives here and nowhere else.

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""

try:
    import manip_sim.vlm as _vlm
    from manip_sim.vlm import (FAILURE_TYPES, MAX_PARSE_RETRIES, MODEL, PROMPT_DELTAS,
                               TEMPERATURE, CallLog, Client, ObjectRef, ParseRejection, StagePlan,
                               StageSpec, VLMError, Vocabulary, emission_from_json, entry_line)
    from manip_sim.vlm import _urllib_transport as live_transport   # what plan_stages records
    from manip_sim.vlm import _JSON_ONLY as JSON_ONLY, _load_json as load_json
    from manip_sim.vlm import _text as text_block, image_block
except ImportError as e:
    raise ImportError(
        "vlm_compat: cannot import manip_sim.vlm. The sim checkout must be bind-mounted at "
        "/opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro). A missing host path makes Docker "
        "mount an EMPTY dir without any error -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/vlm.py\n"
        "    echo $PYTHONPATH") from e

from manip_bridge.sim_provenance import provenance as _provenance

SOURCES = (_vlm.__file__,)

__all__ = ["FAILURE_TYPES", "MAX_PARSE_RETRIES", "MODEL", "PROMPT_DELTAS", "TEMPERATURE",
           "emission_from_json", "entry_line",
           "CallLog", "Client", "ObjectRef", "ParseRejection", "StagePlan", "StageSpec",
           "VLMError", "Vocabulary", "live_transport", "provenance", "SOURCES",
           "JSON_ONLY", "load_json", "text_block", "image_block", "ask"]


def ask(client, touchpoint: str, system: str, messages: list, parse):
    """Run one hardware-owned touchpoint through the client's ask loop:
    `parse(raw)` raises ParseRejection to have the reason fed back to the
    model (bounded retry, as for calls #1-#3); budget exhausted or a
    transport failure raises VLMError. Every attempt lands in client.logs."""
    return client._ask(touchpoint, system, messages, parse)


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim file imported here; goes into every call-#1 artifact."""
    return _provenance(SOURCES)
