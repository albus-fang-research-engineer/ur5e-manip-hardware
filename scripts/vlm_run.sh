#!/usr/bin/env bash
# Run one of the VLM drivers in /data/runs (plan_scene.py, select_scene.py,
# ...) inside Ros2Bridge with the Anthropic key from the repo's .env.
#
#   scripts/vlm_run.sh plan_scene.py --run /data/runs/<stamp> --task "pour tea into the mug"
#   MANIP_VLM_MODEL=claude-sonnet-4-6 scripts/vlm_run.sh plan_scene.py ...   # one-off override
#
# .env (gitignored) holds, one line each:
#   ANTHROPIC_API_KEY=sk-ant-...
#   MANIP_VLM_MODEL=claude-opus-5        # the model every live call uses
#   MANIP_VLM_TEMPERATURE=none           # optional: "none" omits the key (Opus 5
#                                        # rejects it); unset = the sim's 0.0
#
# The key is passed to docker by NAME (-e ANTHROPIC_API_KEY), never on the
# command line, so it is not printed and does not show in `ps`. The model is
# required: no silent fallback to the drivers' built-in default.
#
# Every invocation that is not a --replay (replays make no API call) is
# appended to outputs/vlm_calls.log: UTC time, model, script, arguments,
# exit code -- the running count of paid calls.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO/.env"
LOG="$REPO/outputs/vlm_calls.log"
CONTAINER="${VLM_CONTAINER:-Ros2Bridge}"

[[ $# -ge 1 ]] || { echo "usage: $0 <driver.py> [args...]" >&2; exit 64; }
SCRIPT="$1"; shift
[[ "$SCRIPT" =~ ^[A-Za-z0-9_]+\.py$ ]] || { echo "vlm_run: '$SCRIPT' is not a driver name in /data/runs" >&2; exit 64; }

env_value() {   # last KEY=value line in .env, without echoing anything else
  [[ -f "$ENV_FILE" ]] || return 0
  grep -E "^$1=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'  || true
}

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  ANTHROPIC_API_KEY="$(env_value ANTHROPIC_API_KEY)"
fi
if [[ -z "${MANIP_VLM_MODEL:-}" ]]; then
  MANIP_VLM_MODEL="$(env_value MANIP_VLM_MODEL)"
fi
[[ -n "$ANTHROPIC_API_KEY" ]] || { echo "vlm_run: no ANTHROPIC_API_KEY in $ENV_FILE" >&2; exit 78; }
[[ -n "$MANIP_VLM_MODEL" ]] || { echo "vlm_run: no MANIP_VLM_MODEL in $ENV_FILE (set it; no silent default)" >&2; exit 78; }
if [[ -z "${MANIP_VLM_TEMPERATURE:-}" ]]; then
  MANIP_VLM_TEMPERATURE="$(env_value MANIP_VLM_TEMPERATURE)"
fi
export ANTHROPIC_API_KEY MANIP_VLM_MODEL
temp_env=()
if [[ -n "$MANIP_VLM_TEMPERATURE" ]]; then
  export MANIP_VLM_TEMPERATURE
  temp_env=(-e MANIP_VLM_TEMPERATURE)
fi

live=1
for a in "$@"; do [[ "$a" == "--replay" || "$a" == --replay=* ]] && live=0; done

echo "[vlm_run] $SCRIPT model=$MANIP_VLM_MODEL temperature=${MANIP_VLM_TEMPERATURE:-0.0 (sim default)} $([[ $live == 1 ]] && echo LIVE || echo replay)" >&2
set +e
docker exec -e ANTHROPIC_API_KEY -e MANIP_VLM_MODEL "${temp_env[@]}" "$CONTAINER" bash -lc \
  'export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH; script="$1"; shift; exec python3 "/data/runs/$script" "$@"' \
  _ "$SCRIPT" "$@"
code=$?
set -e

if [[ $live == 1 ]]; then
  mkdir -p "$(dirname "$LOG")"
  printf '%s\t%s\ttemp=%s\t%s\t%s\texit=%s\n' "$(date -u +%FT%TZ)" "$MANIP_VLM_MODEL" \
    "${MANIP_VLM_TEMPERATURE:-0.0}" "$SCRIPT" "$*" "$code" >> "$LOG"
fi
exit $code
