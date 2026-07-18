#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/srsc_lite_v12

while tmux has-session -t srsc_ots 2>/dev/null; do
  sleep 30
done

grep -q 'OTS_MATERIALIZED' artifacts/logs/prepare_ots_kaggle.log
test -f artifacts/manifests/ots_materialized.json
python scripts/prepare_data.py --protocol aio3 --build
python scripts/prepare_data.py --protocol aio5 --build

# The live AIO-3 run migrated at epoch 55 from one-GPU batch 64 to the
# registered 4x30x1 runtime.  A clean rank exit is not enough: require the
# exact final epoch/step, four-rank runtime metadata, final locked validation,
# and retained top-3 transaction before any scientific arm can start.
python scripts/verify_stage_a_checkpoint.py \
  --checkpoint artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt \
  --config configs/protocol_aio3.yaml \
  --minimum-step 330500 \
  --expected-run-name aio3_stage_a_coarse_seed1415926 \
  --expected-epoch 240 \
  --expected-step 330500 \
  --expected-world-size 4 \
  --expected-global-effective-batch 120 \
  --expected-per-gpu-batch 30 \
  --expected-accumulation 1 \
  --expected-workers-per-rank 8 \
  --expected-backend nccl \
  --require-validation-complete \
  --poll-seconds 1 \
  --timeout-hours 0.1 \
  --output artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/final_integrity.json

# Stage-B comparison arms are independent single-GPU experiments.
# Explicitly expose four slots to the orchestrator so it can run one unchanged
# arm per GPU after Stage-A; this does not alter any arm's scientific budget.
export SRSC_PARALLEL_GPUS=0,1,2,3
export SRSC_TRAIN_WORKERS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Establish AIO-3 information value and learnability first.  The frozen prompt
# authorizes AIO-5 after this formal Stage-B gate, not after AIO-3 Stage-C or
# after seeing AIO-3 official-test results.
python scripts/orchestrate.py --protocol aio3 --pilot-steps 1000 \
  --stop-after-stage-b

# AIO-5 Stage-A has the same registered global batch (120), optimizer-step
# count and LR schedule under 4x30x1 DDP as under the original 15x8 setup.
# Train it independently from scratch before the AIO-5 scientific ladder;
# never warm-start it from AIO-3.
bash scripts/launch_aio5_stage_a_4x4090.sh

# Run and freeze AIO-5 Stage-B before either protocol is allowed to read an
# official test.  This prevents overlapping AIO-3 test tasks from becoming an
# implicit tuning signal for AIO-5.
aio5_stage_b_rc=0
if python scripts/orchestrate.py --protocol aio5 --pilot-steps 1000 \
  --stop-after-stage-b; then
  aio5_stage_b_go=1
else
  aio5_stage_b_rc=$?
  aio5_stage_b_go=0
  echo "AIO5_STAGE_B_NOT_AUTHORIZED rc=$aio5_stage_b_rc $(date -u +%FT%TZ)" \
    | tee -a artifacts/logs/pipeline_sequence.log
fi

# Resume AIO-3 from its idempotent Stage-B artifacts.  A Stage-C NO-GO remains
# a real nonzero scientific result, but it must not erase an independently
# authorized AIO-5 path.
aio3_full_rc=0
python scripts/orchestrate.py --protocol aio3 --pilot-steps 1000 \
  || aio3_full_rc=$?

aio5_full_rc=0
if [ "$aio5_stage_b_go" -eq 1 ]; then
  python scripts/orchestrate.py --protocol aio5 --pilot-steps 1000 \
    || aio5_full_rc=$?
else
  aio5_full_rc=$aio5_stage_b_rc
fi

if [ "$aio3_full_rc" -ne 0 ] || [ "$aio5_full_rc" -ne 0 ]; then
  echo "PIPELINE_TERMINAL_NONZERO aio3=$aio3_full_rc aio5=$aio5_full_rc $(date -u +%FT%TZ)" \
    | tee -a artifacts/logs/pipeline_sequence.log >&2
  if [ "$aio3_full_rc" -ne 0 ]; then
    exit "$aio3_full_rc"
  fi
  exit "$aio5_full_rc"
fi

echo "AIO3_AIO5_PIPELINE_COMPLETE $(date -u +%FT%TZ)" \
  | tee -a artifacts/logs/pipeline_sequence.log
