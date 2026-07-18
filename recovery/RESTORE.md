# Checkpoint recovery procedure

This procedure restores code and experiment state. It does not redistribute
datasets or unlock the official test.

## 1. Clone at the frozen path

```bash
git clone https://github.com/emma-sue/aaa.git /root/autodl-tmp/srsc_lite_v12
cd /root/autodl-tmp/srsc_lite_v12
bash scripts/bootstrap_recovery.sh
```

Install the versions in `environment/requirements-resume.txt` using the
official PyTorch CUDA 12.1 wheel index, and install GitHub CLI before the
Release download. Credentials are deliberately not stored in this repository;
public Git data can be cloned without them.

Relocating the repository changes absolute config/data contracts. Do not resume
an old checkpoint from a different path and call it the same run.

## 2. Restore standard datasets

Mount the same WaterlooED, BSD400/CBSD68, RainTrainL/Rain100L, OTS/SOTS,
GoPro, and LOL-v1 assets at the paths recorded in `artifacts/manifests`.
Rebuild symlinks rather than copying the old absolute symlink tree:

```bash
python scripts/prepare_data.py --protocol aio3 --build
python scripts/prepare_data.py --protocol aio5 --build
```

Both audits must report `missing_entries=0`. Never substitute another dataset
or use official-test metrics to select a configuration.

## 3. Restore and verify a checkpoint

For the latest rolling continuation state:

```bash
python scripts/restore_checkpoint_asset.py \
  --repo emma-sue/aaa \
  --tag resume-aio3-stage-a \
  --asset last.pt \
  --destination artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt
```

For an immutable best model, use the tag and asset name in
`recovery/CHECKPOINTS.json` under `runs.<run-name>`. For example, later stages
can add `--run <exact-run-name>` to disambiguate a tag/asset pair. The restore
tool checks the Release SHA sidecar and metadata JSON, the Git-indexed
run/tag/asset/digest, the checkpoint payload contract, and the bound Git
commit/tree before deserializing the checkpoint on CPU. Entries marked
`index_only_not_published` are bounded pilot audit records and cannot be
restored from a Release.

For a checkpoint whose `code` completeness is `present`, preserve the bound
content commit from the current index, download first, and then detach to that
code snapshot before running the trainer:

```bash
SNAPSHOT=$(python -c 'import json; x=json.load(open("recovery/CHECKPOINTS.json")); print(x["runs"]["aio3_stage_a_coarse_seed1415926"]["resume_latest"]["git_snapshot_commit"])')
git checkout --detach "$SNAPSHOT"
```

The downloaded `.pt` is untracked and remains in place. A legacy AIO-3 row
marked `code: legacy_missing` can still provide a state-exact optimizer/RNG
continuation with the recorded config, split and distributed runtime. Its bound
Git commit is an audited recovery snapshot, not proof that the launch-time
Python source was byte-identical; do not describe that case as exact code
provenance.

## 4. Audit before GPU launch

```bash
CUDA_VISIBLE_DEVICES='' pytest -q
python scripts/prepare_data.py --protocol aio3
python scripts/verify_stage_a_checkpoint.py \
  --checkpoint artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt \
  --config configs/protocol_aio3.yaml \
  --expected-run-name aio3_stage_a_coarse_seed1415926 \
  --expected-world-size 4 \
  --expected-global-effective-batch 120 \
  --expected-per-gpu-batch 30 \
  --expected-accumulation 1 \
  --expected-workers-per-rank 8 \
  --expected-backend nccl
```

Do not start a second trainer if a live process or tmux pipeline already exists.

## 5. Resume

Use the registered launcher, which keeps four ranks, per-GPU batch 30, global
batch 120, LR schedule, RNG restoration, and exact run name:

```bash
bash scripts/launch_aio3_stage_a_4x4090.sh
```

After recovery, verify that the first new step is monotonic and that there is no
NaN/OOM/traceback. Record the new host, driver, GPUs, checkpoint SHA, and launch
time in the experiment log.
