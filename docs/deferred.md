# Deferred items (hardware repo)

Known issues and improvements deliberately left for after the MVP. None
changes which plan runs or how the robot moves. Each entry says where it is,
what goes wrong, and what would make it worth doing.

Found by /code-review (medium) of the vlm-port commits ahead of
origin/vlm-port, 2026-11-10. "Verified" means the code was read and the
failure path confirmed; the rest are unverified review findings.

## H1. Paid-call log undercounts HTTP retries (verified)

`ros2_ws/src/manip_bridge/manip_bridge/vlm_record.py`, `RecordingTransport.__call__`:
`sent` counts one per transport call, but the sim's `live_transport` retries
429/5xx up to `HTTP_RETRIES` (3) times inside that call. `vlm_calls.log`
`calls=` can therefore undercount requests. 429/529/5xx are usually not
billed, so the cost error is small. Fix: count attempts in the transport (or
expose its retry count) and log both. Trigger: cost accounting for the paper.

## H2. Critic re-emit writes "world.-z" for any negative world axis (verified)

`outputs/runs/critic_scene.py`, `_dir_token`: a world axis with sign "-"
always becomes `world.-z`. Only reachable if an emission uses `world.x` or
`world.y` with a negative sign; none does today. Fix: keep the axis letter,
plus a round-trip test through `parse_emission`.

## H3. Stale critic/emit response files leak into replays (unverified)

`outputs/runs/critic_scene.py`, `save()`: `responses/critic_NN.txt` and
`emit_NN.txt` from an earlier run in the same `--out` are not cleared, and a
replay reads every file in sorted order. Workaround: a fresh `--out` per run
(current practice). Fix: clear the responses dir on a live run.

## H4. Empty-goal failures classified by error-message text (unverified)

`outputs/runs/plan_stages_hw.py`, `_EMPTY_MARKERS`: rewording the sim
compiler's messages would silently route call #5's empty-goal-intersection
repair to a call #3 retry. Fix: a typed code on `CompileError`. Trigger: any
change to the sim compiler's messages.

## H5. Critic preview duplicates the planner's compile (unverified)

`outputs/runs/critic_scene.py`, `compile_chain` re-implements
`plan_stages_hw.compile_at_entry`. They match today; they can drift. Fix:
one shared function. Trigger: any change to either compile path.

## H6. Root-owned files under outputs/runs/.vlm_calls (unverified)

`scripts/vlm_run.sh`: in container mode the per-run calls file is written as
root and the host-side `rm -f` fails silently. Clutter only.

## H7. critic_cases can report a stale critic.json (unverified)

`outputs/runs/critic_cases.py`, `case_record`: reads `<case_out>/critic/critic.json`
without checking this run wrote it, so a case that fails before the critic
reports the previous run's verdict. Test harness only.

## H8. Planner reports untimeable plans as planned

`outputs/runs/plan_stages_hw.py`: `excess_metrics` records a TOPP-RA refusal
(`dense: {"refused": ...}`) but the attempt still counts as planned, so the
executor is the first to refuse it. Proposed: count it as a failed attempt
and try the next seed. Deferred: 0 of 10 sweep10 seeds hit it at
`DENSIFY_STEP` 0.005. Trigger: a refusal during trials. If refusals come
back, prefer smoothing/shortcutting the CBiRRT path over a smaller step.
