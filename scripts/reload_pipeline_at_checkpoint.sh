#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/srsc_lite_v12
log=artifacts/logs/pipeline_reload.log
checkpoint=artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt
target_step="${1:-18000}"

if ! [[ "$target_step" =~ ^[0-9]+$ ]] || (( target_step <= 0 )); then
  echo "target step must be a positive integer" >&2
  exit 2
fi

exec >>"$log" 2>&1
echo "RELOAD_WATCH_START utc=$(date -u +%FT%TZ) target_step=$target_step"

last_mtime=-1
while true; do
  current_mtime=$(stat -c %Y "$checkpoint" 2>/dev/null || echo -1)
  if [[ "$current_mtime" == "$last_mtime" ]]; then
    sleep 15
    continue
  fi
  last_mtime="$current_mtime"
  step=$(python - "$checkpoint" <<'PY'
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(-1)
else:
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(d.get("step", -1) if d.get("batch_in_epoch") is not None else -1)
    except Exception:
        print(-1)
PY
)
  if [[ "$step" =~ ^[0-9]+$ ]] && (( step >= target_step )); then
    echo "SAFE_CHECKPOINT utc=$(date -u +%FT%TZ) step=$step"
    break
  fi
  sleep 15
done

train_pid=$(pgrep -fo 'python scripts/train.py --config /root/autodl-tmp/srsc_lite_v12/configs/protocol_aio3.yaml --stage a' || true)
if [[ -z "$train_pid" ]]; then
  echo "ABORT no matching Stage-A process"
  exit 2
fi
kill -TERM "$train_pid"
echo "TERM_SENT train_pid=$train_pid"

for _ in $(seq 1 60); do
  if ! kill -0 "$train_pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$train_pid" 2>/dev/null; then
  echo "ABORT train process did not terminate; no forced kill issued"
  exit 3
fi

lock_released=0
for _ in $(seq 1 120); do
  if flock -n .orchestrate_aio3.lock -c true; then
    lock_released=1
    break
  fi
  sleep 1
done
if (( lock_released == 0 )); then
  echo "ABORT orchestrator kernel lock did not release"
  exit 4
fi

tmux kill-session -t srsc_pipeline 2>/dev/null || true
tmux new-session -d -s srsc_pipeline \
  "cd /root/autodl-tmp/srsc_lite_v12 && export OMP_NUM_THREADS=1 && bash scripts/launch_when_data_ready.sh 2>&1 | tee -a artifacts/logs/pipeline.log"
echo "PIPELINE_RELOADED utc=$(date -u +%FT%TZ) resume_step=$step"
