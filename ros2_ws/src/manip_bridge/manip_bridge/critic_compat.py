"""The ONE place hardware imports the sim's critic (VLM call #4, critic v2):
the edit grammar / parser / applier / loop guards (manip_sim.critic), the
critic v2 prompt and parser (manip_sim.vlm build_critic_prompt_v2,
parse_critic_v2) and the preview renderer (scripts.render_preview).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

Why not vlm_compat: these names exist only on sim branches that carry the
critic (critic-repair, until it merges). vlm_compat is imported by every VLM
driver, so a missing name there would break calls #1-#3 on a sim checkout
without the critic. This module is imported only by plan_stages_hw.py
--critic preview (critic_scene.py), so an older sim fails only that flag,
with the message below.

What critic_scene uses:
  SpecState, LoopLedger, apply_edits,      the editable specification, the
  EditRefusal, ApplyResult, Verdict,       round budget + no-reverse rule,
  StagePreviewInput, build_preview_report, the edit applier and the preview
  report_json, row_text, authored_slots,   report call #4 reads
  CRITIC_ACTIONS
  point_offsets (critic._point_offsets)    a point's horizontal distance and
                                           height from another, along an up
                                           axis: the report's measure, reused
                                           for the logged spout-to-rim gap
  build_critic_prompt_v2, parse_critic_v2  the prompt (dry run writes it) and
                                           parser (Client.critique_preview_v2
                                           runs both)
  render_preview, from_preview_inputs,     the preview renderer. The module
  GraspInput, gripper_from_urdf,           imports MuJoCo only inside its
  GripperModel (a mesh in the TCP frame;   render functions, so this import
  critic_scene loads the exported 2F-85    stays GL-free; render_preview needs
  into it)                                 GL when it is CALLED.
  contact_text, contact_bodies             the renderer's contact wording
  (render_preview._bodies)                 ("teapot collides with mug and
                                           table"), so captions match images

Import note: sim's scripts/ has no __init__.py, so `scripts.render_preview`
resolves as a NAMESPACE package off /opt/manip-sim (same caveat as
emit_compat).
"""

try:
    import manip_sim.critic as _critic
    import scripts.render_preview as _rp
    from manip_sim.critic import (CRITIC_ACTIONS, ApplyResult, EditRefusal, LoopLedger,
                                  SpecState, StagePreviewInput, Verdict, apply_edits,
                                  authored_slots, build_preview_report, report_json, row_text)
    from manip_sim.critic import _point_offsets as point_offsets
    from manip_sim.vlm import build_critic_prompt_v2, parse_critic_v2
    from scripts.render_preview import (GraspInput, GripperModel, contact_text,
                                        from_preview_inputs, gripper_from_urdf, render_preview)
    from scripts.render_preview import _bodies as contact_bodies
except ImportError as e:
    raise ImportError(
        "critic_compat: cannot import manip_sim.critic / scripts.render_preview. --critic "
        "preview needs a sim checkout that carries the critic (branch critic-repair or later) "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro) -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/critic.py /opt/manip-sim/scripts/render_preview.py\n"
        "    echo $PYTHONPATH") from e

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

SOURCES = (_critic.__file__, _rp.__file__)
GRIPPER_RGBA = getattr(_rp, "GRIPPER_RGBA", (0.25, 0.25, 0.28, 1.0))   # caption color legend

__all__ = ["CRITIC_ACTIONS", "ApplyResult", "EditRefusal", "LoopLedger", "SpecState",
           "StagePreviewInput", "Verdict", "apply_edits", "authored_slots",
           "build_preview_report", "report_json", "row_text",
           "build_critic_prompt_v2", "parse_critic_v2",
           "GraspInput", "GripperModel", "contact_text", "contact_bodies", "point_offsets",
           "from_preview_inputs", "gripper_from_urdf", "render_preview",
           "GRIPPER_RGBA", "provenance", "SOURCES"]


def provenance():
    """{root, commit, dirty, dirty_sources, files: {relpath: sha256}} for the
    sim files imported here; recorded in every critic round."""
    return _provenance(SOURCES)
