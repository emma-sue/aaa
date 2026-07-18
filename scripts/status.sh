#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/srsc_lite_v12
echo "UTC $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "TMUX"
tmux ls 2>&1 || true
echo "GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "FILESYSTEM"
df -h /root/autodl-tmp /dev/shm
echo "OTS"
ls -lh data_raw/downloads/srsc_ots_kaggle.zip data_raw/downloads/srsc_ots_kaggle.chunk \
  /dev/shm/srsc_ots_kaggle.zip /dev/shm/srsc_ots_kaggle.chunk 2>/dev/null || true
tail -5 artifacts/logs/prepare_ots_kaggle.log 2>/dev/null || true
echo "MANIFESTS"
python - <<'PY'
import json
from pathlib import Path
for name in ('aio3', 'aio5'):
    path = Path('artifacts/manifests') / f'{name}.json'
    if path.exists():
        data = json.loads(path.read_text())
        print(name, 'expected=', data['expected_entries'], 'missing=', data['missing_entries'])
PY
echo "ACTIVE TRAINING"
ps -eo pid,etime,stat,%cpu,%mem,cmd | \
  awk '/scripts\/orchestrate\.py/ || /scripts\/train_stage_a_ddp\.py/ || /scripts\/train_stage_a_capacity_hybrid_ddp\.py/ || /scripts\/train_baseline_hybrid_ddp\.py/ || (/scripts\/train\.py/ && $4 + 0 >= 10)' || true
latest_train_log=$(find artifacts/logs -maxdepth 3 -type f -name '*.log' \
  ! -name 'watchdog.log' ! -name 'pipeline.log' -printf '%T@ %p\n' 2>/dev/null | \
  sort -nr | awk 'NR==1 {$1=""; sub(/^ /, ""); print}')
if [[ -n "${latest_train_log:-}" ]]; then
  echo "LATEST_LOG $latest_train_log"
  tail -3 "$latest_train_log" 2>/dev/null || true
fi
echo "LOCKED-VAL BEST"
python - <<'PY'
import json
from pathlib import Path

files = sorted(
    Path('artifacts/metrics').glob('*_locked_val.jsonl'),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not files:
    print('none')
else:
    path = files[0]
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        print(path, 'empty')
    else:
        best = max(records, key=lambda item: item['macro_psnr'])
        latest = records[-1]
        print('file=', path)
        print('best=', json.dumps(best, sort_keys=True))
        print('latest=', json.dumps(latest, sort_keys=True))
PY
echo "DECISION"
if [[ -s reports/decision.json ]]; then
  python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('reports/decision.json').read_text())
keys = ('scientific_go', 'publication_go', 'selected_model', 'next_command')
print(json.dumps({key: d.get(key, 'INCOMPLETE') for key in keys}, sort_keys=True))
PY
else
  echo "INCOMPLETE"
fi
echo "CHECKPOINTS"
find artifacts/checkpoints -name '*.pt' -o -name '*.ckpt' 2>/dev/null | sort | tail -20
