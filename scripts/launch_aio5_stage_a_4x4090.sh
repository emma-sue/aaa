#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/srsc_lite_v12
cd "$ROOT"

# Serialize all SRSC jobs that claim the physical GPUs.  The orchestrator uses
# the same lock and releases it before this launcher is invoked.
exec 9>"$ROOT/.srsc_gpu_pipeline.lock"
if ! flock -n 9; then
  echo "ERROR: SRSC GPU pipeline lock is already held" >&2
  exit 4
fi

CONFIG=configs/protocol_aio5.yaml
RESUME=artifacts/checkpoints/aio5_stage_a_coarse_seed1415926/last.pt
RUN_NAME=aio5_stage_a_coarse_seed1415926
WORLD_SIZE=4
PER_GPU_BATCH=30
ACCUMULATION=1
WORKERS_PER_RANK=6
MASTER_ADDR=127.0.0.1
MASTER_PORT=29659
LOG_DIR=artifacts/logs/ddp_stage_a_aio5
FINAL_STEP=427440
mkdir -p "$LOG_DIR"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$gpu_count" -lt "$WORLD_SIZE" ]; then
  echo "ERROR: need $WORLD_SIZE GPUs, found $gpu_count" >&2
  exit 2
fi

if [ -s "$RESUME" ]; then
  start_args=(--resume "$RESUME")
  start_mode=resume
else
  start_args=(--fresh)
  start_mode=fresh
fi

echo "DDP_STAGE_A_AIO5_START $(date -u +%FT%TZ) world=$WORLD_SIZE per_gpu_batch=$PER_GPU_BATCH global_batch=$((WORLD_SIZE * PER_GPU_BATCH * ACCUMULATION)) mode=$start_mode" | tee -a "$LOG_DIR/launcher.log"

pids=()
for rank in 0 1 2 3; do
  setsid env \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" WORLD_SIZE="$WORLD_SIZE" \
    RANK="$rank" LOCAL_RANK="$rank" OMP_NUM_THREADS=1 \
    python scripts/exec_unblocked.py python scripts/train_stage_a_ddp.py \
      --config "$CONFIG" \
      "${start_args[@]}" \
      --run-name "$RUN_NAME" \
      --per-gpu-batch "$PER_GPU_BATCH" \
      --accumulation "$ACCUMULATION" \
      --workers-per-rank "$WORKERS_PER_RANK" \
      --enforce-config-effective-batch \
      --require-training-origin fresh \
      >> "$LOG_DIR/rank${rank}.log" 2>&1 &
  pids+=("$!")
done

terminate_children() {
  local pid
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  local deadline=$((SECONDS + 45))
  local alive
  while [ "$SECONDS" -lt "$deadline" ]; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    if [ "$alive" -eq 0 ]; then
      return
    fi
    sleep 1
  done
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
}
trap 'terminate_children; exit 130' INT TERM

remaining=${#pids[@]}
while [ "$remaining" -gt 0 ]; do
  if wait -n; then
    remaining=$((remaining - 1))
  else
    status=$?
    echo "ERROR: a DDP rank failed with status $status; terminating peers" | tee -a "$LOG_DIR/launcher.log" >&2
    terminate_children
    wait || true
    exit "$status"
  fi
done

python scripts/verify_stage_a_checkpoint.py \
  --checkpoint "$RESUME" \
  --config "$CONFIG" \
  --minimum-step "$FINAL_STEP" \
  --expected-run-name "$RUN_NAME" \
  --expected-epoch 240 \
  --expected-step "$FINAL_STEP" \
  --expected-world-size "$WORLD_SIZE" \
  --expected-global-effective-batch 120 \
  --expected-per-gpu-batch "$PER_GPU_BATCH" \
  --expected-accumulation "$ACCUMULATION" \
  --expected-workers-per-rank "$WORKERS_PER_RANK" \
  --expected-backend nccl \
  --expected-training-origin fresh \
  --require-validation-complete \
  --poll-seconds 1 \
  --timeout-hours 0.1 \
  --output artifacts/checkpoints/aio5_stage_a_coarse_seed1415926/final_integrity.json \
  >> "$LOG_DIR/launcher.log" 2>&1

echo "DDP_STAGE_A_AIO5_COMPLETE $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"
flock -u 9
