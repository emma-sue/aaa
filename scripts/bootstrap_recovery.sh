#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_ROOT="/root/autodl-tmp/srsc_lite_v12"
PROMPTIR_COMMIT="106159ab809101f2e25b6714195cd6fa9a938d36"
R2R_COMMIT="bf387d56095aaf4edc0b685f8ea58cce5c64c2fc"
RESEARCHSTUDIO_COMMIT="61277686638adb87298a26cc7621cd7387723fb4"

if [[ "$ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "Exact resume requires clone path $EXPECTED_ROOT; current path is $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT/upstream" "$ROOT/artifacts/checkpoints" "$ROOT/artifacts/logs" \
  "$ROOT/artifacts/metrics" "$ROOT/artifacts/manifests" "$ROOT/artifacts/stats" \
  /root/aaa

if [[ -e "$ROOT/upstream/PromptIR" && ! -d "$ROOT/upstream/PromptIR/.git" ]]; then
  echo "Refusing to replace non-Git path $ROOT/upstream/PromptIR" >&2
  exit 3
fi
if [[ ! -d "$ROOT/upstream/PromptIR/.git" ]]; then
  git clone https://github.com/va1shn9v/PromptIR.git "$ROOT/upstream/PromptIR"
fi
git -C "$ROOT/upstream/PromptIR" fetch origin "$PROMPTIR_COMMIT"
git -C "$ROOT/upstream/PromptIR" checkout --detach "$PROMPTIR_COMMIT"
test "$(git -C "$ROOT/upstream/PromptIR" rev-parse HEAD)" = "$PROMPTIR_COMMIT"

if [[ -e /root/R2R && ! -d /root/R2R/.git ]]; then
  echo "Refusing to replace non-Git path /root/R2R" >&2
  exit 4
fi
if [[ ! -d /root/R2R/.git ]]; then
  git clone https://github.com/cscxwang/R2R.git /root/R2R
fi
git -C /root/R2R fetch origin "$R2R_COMMIT"
git -C /root/R2R checkout --detach "$R2R_COMMIT"
test "$(git -C /root/R2R rev-parse HEAD)" = "$R2R_COMMIT"

if [[ -e /root/ResearchStudio && ! -d /root/ResearchStudio/.git ]]; then
  echo "Refusing to replace non-Git path /root/ResearchStudio" >&2
  exit 5
fi
if [[ ! -d /root/ResearchStudio/.git ]]; then
  git clone https://github.com/microsoft/ResearchStudio.git /root/ResearchStudio
fi
git -C /root/ResearchStudio fetch origin "$RESEARCHSTUDIO_COMMIT"
git -C /root/ResearchStudio checkout --detach "$RESEARCHSTUDIO_COMMIT"
test "$(git -C /root/ResearchStudio rev-parse HEAD)" = "$RESEARCHSTUDIO_COMMIT"

mkdir -p /root/ResearchStudio/ideaspark_run/end-to-end-restoration-state-feedback
cp "$ROOT/vendor/researchstudio/srsc_lite_v1_2_reassessment.md" \
  /root/ResearchStudio/ideaspark_run/end-to-end-restoration-state-feedback/srsc_lite_v1_2_reassessment.md
cp "$ROOT/docs/contracts/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md" /root/aaa/
cp "$ROOT/docs/contracts/v1.4.md" /root/aaa/v1.4.md

python "$ROOT/scripts/verify_recovery_bundle.py" --root "$ROOT"
python "$ROOT/scripts/prepare_data.py" --protocol aio3 || true
python "$ROOT/scripts/prepare_data.py" --protocol aio5 || true

echo "Bootstrap complete. Restore datasets/checkpoint, require missing_entries=0, then run CPU tests before GPU resume."
