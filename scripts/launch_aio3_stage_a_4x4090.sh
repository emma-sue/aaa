#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/srsc_lite_v12
cd "$ROOT"

# The currently running AIO-3 job was started before the global flock was
# added to this launcher, so that legacy parent does not own FD 9.  Refuse a
# second launch based on the actual trainer process as well as the advisory
# lock.  The process check comes first; the flock still closes the race
# between two otherwise simultaneous fresh launches.
mapfile -t active_stage_a_pids < <(
  ps -eo pid=,args= | awk \
    '$0 ~ /python .*scripts\/train_stage_a_ddp\.py/ {print $1}'
)
if [ "${#active_stage_a_pids[@]}" -gt 0 ]; then
  echo "ERROR: AIO-3 Stage-A trainer already active; refusing duplicate launch: ${active_stage_a_pids[*]}" >&2
  exit 5
fi

exec 9>"$ROOT/.srsc_gpu_pipeline.lock"
if ! flock -n 9; then
  echo "ERROR: SRSC GPU pipeline lock is already held" >&2
  exit 4
fi

CONFIG=configs/protocol_aio3.yaml
RESUME=artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt
RUN_NAME=aio3_stage_a_coarse_seed1415926
WORLD_SIZE=4
PER_GPU_BATCH=30
ACCUMULATION=1
WORKERS_PER_RANK=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=29658
LOG_DIR=artifacts/logs/ddp_stage_a
mkdir -p "$LOG_DIR"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$gpu_count" -lt "$WORLD_SIZE" ]; then
  echo "ERROR: need $WORLD_SIZE GPUs, found $gpu_count" >&2
  exit 2
fi
if [ ! -s "$RESUME" ]; then
  echo "ERROR: missing resume checkpoint $RESUME" >&2
  exit 3
fi

echo "DDP_STAGE_A_START $(date -u +%FT%TZ) world=$WORLD_SIZE per_gpu_batch=$PER_GPU_BATCH global_batch=$((WORLD_SIZE * PER_GPU_BATCH * ACCUMULATION)) resume=$RESUME"

pids=()
for rank in 0 1 2 3; do
  setsid env \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" WORLD_SIZE="$WORLD_SIZE" \
    RANK="$rank" LOCAL_RANK="$rank" OMP_NUM_THREADS=1 \
    python scripts/exec_unblocked.py python scripts/train_stage_a_ddp.py \
      --config "$CONFIG" \
      --resume "$RESUME" \
      --run-name "$RUN_NAME" \
      --per-gpu-batch "$PER_GPU_BATCH" \
      --accumulation "$ACCUMULATION" \
      --workers-per-rank "$WORKERS_PER_RANK" \
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
    echo "ERROR: a DDP rank failed with status $status; terminating peers" >&2
    terminate_children
    wait || true
    exit "$status"
  fi
done

echo "DDP_STAGE_A_COMPLETE $(date -u +%FT%TZ)"
echo "CONTINUE_ORCHESTRATED_PIPELINE $(date -u +%FT%TZ)"
flock -u 9
exec bash scripts/launch_when_data_ready.sh
