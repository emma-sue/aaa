#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/srsc_lite_v12
LOG="$ROOT/artifacts/logs/watchdog.log"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-1800}"
CONTRACT_EVERY="${WATCHDOG_CONTRACT_EVERY:-12}"
iteration=0

cd "$ROOT" || exit 1
mkdir -p "$ROOT/artifacts/logs"

while true; do
  iteration=$((iteration + 1))
  {
    echo "WATCHDOG_BEGIN utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ') iteration=$iteration"
    echo "TMUX"
    tmux ls 2>&1 || true
    echo "GPU"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader 2>&1 || true
    echo "TRAIN_PROCESSES"
    ps -eo pid,etimes,%cpu,%mem,stat,cmd | \
      grep -E 'scripts/(orchestrate|train|train_stage_a_ddp|train_baseline_hybrid|train_baseline_hybrid_ddp|cache_stage_a_outputs|eval_locked|compute_coordinate_stats)\.py' | \
      grep -v grep || true
    echo "FILESYSTEM"
    df -h /root/autodl-tmp 2>&1 || true
    echo "LATEST_TRAIN_RECORDS"
    find artifacts/logs -maxdepth 1 -name '*.csv' -type f -printf '%T@ %p\n' 2>/dev/null | \
      sort -n | tail -3 | cut -d' ' -f2- | while read -r file; do
        echo "FILE $file"
        tail -2 "$file" 2>&1 || true
      done
    echo "CHECKPOINTS"
    find artifacts/checkpoints -maxdepth 3 -type f \( -name '*.pt' -o -name '*.ckpt' \) \
      -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -10 || true
    echo "ERROR_SCAN"
    # Do not scan this watchdog's own accumulated output: matching text would
    # otherwise be copied back into watchdog.log on every iteration.  The
    # append-only pipeline.log also contains archived, explicitly invalidated
    # runs; active per-run/orchestrator logs are the authoritative health
    # sources and remain included here.
    find artifacts/logs -maxdepth 1 -type f -mmin -40 \
      ! -name 'watchdog.log' ! -name 'pipeline.log' -print0 2>/dev/null | \
      xargs -0 -r grep -Ein 'Traceback|RuntimeError|CUDA out of memory|(^|[^a-z])nan([^a-z]|$)' 2>/dev/null | \
      tail -30 || true
    # Preserve orchestration-level error detection, but only for the current
    # strict run.  pipeline.log is append-only and contains intentionally
    # invalidated earlier attempts before the most recent data-gate marker.
    pipeline_start=$(grep -nE 'ORCHESTRATOR (aio3|aio5) entered strict data gate' \
      artifacts/logs/pipeline.log 2>/dev/null | tail -1 | cut -d: -f1)
    if [[ -n "${pipeline_start:-}" ]]; then
      tail -n "+$pipeline_start" artifacts/logs/pipeline.log 2>/dev/null | \
        grep -Ein 'Traceback|RuntimeError|CUDA out of memory|(^|[^a-z])nan([^a-z]|$)' | \
        tail -30 || true
    fi
    if (( iteration == 1 || iteration % CONTRACT_EVERY == 0 )); then
      echo "CONTRACT_SHA256"
      sha256sum \
        /root/aaa/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md \
        /root/aaa/v1.4.md \
        /root/ResearchStudio/ideaspark_run/end-to-end-restoration-state-feedback/srsc_lite_v1_2_reassessment.md \
        /root/ResearchStudio/ResearchStudio-Idea/skills/idea_spark/SKILL.md \
        /root/.codex/skills/autosota/SKILL.md \
        reports/AUDIT.md reports/ARCHITECTURE.md reports/BASELINE_PARITY.md \
        reports/AUTOSOTA_STRATEGY_LIBRARY.md \
        reports/PRE_STAGE_B_RELOAD_REQUIRED.md \
        reports/PROTOCOL_CORRECTION_CENTER_CROP.md \
        reports/PROTOCOL_CORRECTION_AUGMENTATION.md \
        reports/PROTOCOL_AMENDMENT_AIO3_BATCH_MIGRATION.md \
        reports/CACHE_CONTRACT_REVISION_V1.md \
        src/data/aio_dataset.py \
        src/net/clean_restormer_aio.py src/net/srsc_lite.py src/net/restormer_blocks.py \
        src/net/srsc_coordinates.py \
        scripts/orchestrate.py scripts/train.py scripts/train_stage_a_ddp.py \
        scripts/launch_aio3_stage_a_4x4090.sh scripts/launch_aio5_stage_a_4x4090.sh \
        scripts/verify_promptir_baseline.py \
        scripts/eval_locked.py scripts/eval_local_composite.py \
        scripts/export_metrics_long.py \
        scripts/compute_coordinate_stats.py \
        scripts/cache_stage_a_outputs.py scripts/train_baseline_hybrid.py \
        scripts/train_baseline_hybrid_ddp.py \
        scripts/compare_paired.py scripts/compare_r2r.py \
        scripts/launch_when_data_ready.sh scripts/reload_pipeline_at_checkpoint.sh scripts/watchdog.sh \
        scripts/verify_stage_a_checkpoint.py \
        artifacts/manifests/aio3.json artifacts/manifests/locked_split_aio3.json \
        configs/protocol_aio3.yaml configs/protocol_aio5.yaml \
        configs/protocol_aio3_baseline_hybrid.yaml \
        configs/protocol_aio3_10_10.yaml \
        configs/stage_b_aio3.yaml configs/stage_b_aio5.yaml \
        configs/stage_b_aio3_10_10.yaml \
        configs/stage_c_aio3.yaml configs/stage_c_aio5.yaml 2>&1 || true
    fi
    echo "WATCHDOG_END utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ') iteration=$iteration"
  } >> "$LOG"
  sleep "$INTERVAL_SECONDS"
done
