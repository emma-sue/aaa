# Exact restore procedure

This procedure restores code and experiment state. It does not redistribute
datasets or unlock the official test.

## 1. Clone at the frozen path

```bash
git clone https://github.com/emma-sue/aaa.git /root/autodl-tmp/srsc_lite_v12
cd /root/autodl-tmp/srsc_lite_v12
bash scripts/bootstrap_recovery.sh
```

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

For the latest exact-resume state:

```bash
python scripts/restore_checkpoint_asset.py \
  --repo emma-sue/aaa \
  --tag resume-aio3-stage-a \
  --asset last.pt \
  --destination artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt
```

For an immutable best model, use the tag and asset name in
`recovery/CHECKPOINTS.json`. The restore tool checks the Release SHA sidecar,
then verifies the minimum PyTorch checkpoint state on CPU.

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

