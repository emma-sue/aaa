#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/srsc_lite_v12
LEGACY_LAUNCHER_PID="${1:-14592}"
LOG="$ROOT/artifacts/logs/legacy_stage_a_lock_guard.log"

cd "$ROOT"
mkdir -p "$ROOT/artifacts/logs"

if [ ! -r "/proc/$LEGACY_LAUNCHER_PID/cmdline" ]; then
  echo "LEGACY_LOCK_GUARD_REFUSED utc=$(date -u +%FT%TZ) pid=$LEGACY_LAUNCHER_PID reason=missing_process" \
    | tee -a "$LOG" >&2
  exit 2
fi

launcher_cmd=$(tr '\0' ' ' < "/proc/$LEGACY_LAUNCHER_PID/cmdline")
if [[ "$launcher_cmd" != *"scripts/launch_aio3_stage_a_4x4090.sh"* ]]; then
  echo "LEGACY_LOCK_GUARD_REFUSED utc=$(date -u +%FT%TZ) pid=$LEGACY_LAUNCHER_PID reason=unexpected_cmdline cmd=$launcher_cmd" \
    | tee -a "$LOG" >&2
  exit 3
fi

exec 9>"$ROOT/.srsc_gpu_pipeline.lock"
if ! flock -n 9; then
  echo "LEGACY_LOCK_GUARD_REFUSED utc=$(date -u +%FT%TZ) pid=$LEGACY_LAUNCHER_PID reason=lock_already_held" \
    | tee -a "$LOG" >&2
  exit 4
fi

echo "LEGACY_LOCK_GUARD_ACQUIRED utc=$(date -u +%FT%TZ) pid=$LEGACY_LAUNCHER_PID cmd=$launcher_cmd" \
  | tee -a "$LOG"

# The old launcher already contains the verified exec handoff.  Release this
# compatibility lock as soon as that exact parent either exits or execs into
# launch_when_data_ready.sh, so the downstream orchestrator can acquire the
# same global lock itself.  The guard never signals or otherwise mutates the
# live training process.
while [ -r "/proc/$LEGACY_LAUNCHER_PID/cmdline" ]; do
  launcher_cmd=$(tr '\0' ' ' < "/proc/$LEGACY_LAUNCHER_PID/cmdline")
  if [[ "$launcher_cmd" != *"scripts/launch_aio3_stage_a_4x4090.sh"* ]]; then
    break
  fi
  sleep 5
done

echo "LEGACY_LOCK_GUARD_RELEASED utc=$(date -u +%FT%TZ) pid=$LEGACY_LAUNCHER_PID final_cmd=${launcher_cmd:-exited}" \
  | tee -a "$LOG"
flock -u 9
