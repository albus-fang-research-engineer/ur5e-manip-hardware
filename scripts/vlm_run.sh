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
# Host mode (no container): VLM_HOST_SIM=<sim checkout> runs the driver from
# outputs/runs/ with the host's manip-tsr Python, that sim checkout on
# PYTHONPATH and SIM_DIR, and the key in the child's environment only.
# For code not yet merged into the branch Ros2Bridge mounts, e.g.
#   VLM_HOST_SIM=~/ur5e-manip-sim-critic scripts/vlm_run.sh critic_cases.py run --run <host run dir> --live
#
# Every invocation is appended to outputs/vlm_calls.log: UTC time, model,
# script, arguments, exit code and calls=N, the API calls the driver itself
# counted (0 for a replay or dry run; "unknown" if it died before reporting).
# Sum the calls= column for the running count of paid calls.
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

# The driver counts its own API calls (vlm_record.RecordingTransport) and
# writes them at exit to $VLM_CALLS_FILE, a file under the run mount; this
# wrapper reads it back and logs the count.
CALLS_DIR="$REPO/outputs/runs/.vlm_calls"         # /data/runs/.vlm_calls in the container
mkdir -p "$CALLS_DIR"
CALLS_NAME="$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
rm -f "$CALLS_DIR/$CALLS_NAME"

echo "[vlm_run] $SCRIPT model=$MANIP_VLM_MODEL temperature=${MANIP_VLM_TEMPERATURE:-0.0 (sim default)}; the driver counts the API calls it makes" >&2
set +e
if [[ -n "${VLM_HOST_SIM:-}" ]]; then
  HOST_PY="${VLM_HOST_PYTHON:-$HOME/miniconda3/envs/manip-tsr/bin/python}"
  SIM="$(cd "$VLM_HOST_SIM" && pwd)"
  [[ -f "$SIM/manip_sim/vlm.py" ]] || { echo "vlm_run: $SIM is not a sim checkout" >&2; exit 64; }
  where="host, sim $SIM"
  echo "[vlm_run] host mode: $HOST_PY, sim $SIM" >&2
  ( cd "$REPO" && SIM_DIR="$SIM" VLM_CALLS_FILE="$CALLS_DIR/$CALLS_NAME" \
      PYTHONPATH="$REPO/../manip-tsr:$SIM:$REPO/../manip-cbirrt:$REPO/ros2_ws/src/manip_bridge" \
      "$HOST_PY" "outputs/runs/$SCRIPT" "$@" )
  code=$?
else
  where="$CONTAINER"
  docker exec -e ANTHROPIC_API_KEY -e MANIP_VLM_MODEL "${temp_env[@]}" \
    -e VLM_CALLS_FILE="/data/runs/.vlm_calls/$CALLS_NAME" "$CONTAINER" bash -lc \
    'export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH; script="$1"; shift; exec python3 "/data/runs/$script" "$@"' \
    _ "$SCRIPT" "$@"
  code=$?
fi
set -e

calls="unknown"                                   # the driver died before writing it
if [[ -f "$CALLS_DIR/$CALLS_NAME" ]]; then
  calls="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["calls"])' \
           "$CALLS_DIR/$CALLS_NAME" 2>/dev/null || echo unknown)"
  rm -f "$CALLS_DIR/$CALLS_NAME" 2>/dev/null || true
fi
echo "[vlm_run] $SCRIPT made $calls API call(s); exit $code" >&2
mkdir -p "$(dirname "$LOG")"
printf '%s\t%s\ttemp=%s\t%s\t%s\texit=%s\tcalls=%s\tin=%s\n' "$(date -u +%FT%TZ)" "$MANIP_VLM_MODEL" \
  "${MANIP_VLM_TEMPERATURE:-0.0}" "$SCRIPT" "$*" "$code" "$calls" "$where" >> "$LOG"
exit $code
