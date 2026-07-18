# SRSC-Lite v1.2: reproducible All-in-One Image Restoration experiments

This repository is the public, continuously versioned research workspace for
SRSC-Lite v1.2. It contains the clean Restormer-AiO implementation, the
assessment-guided two-stage model, preregistered controls, training/evaluation
code, exact split/manifests, key logs, and recovery metadata.

## Scientific status

The implementation, data audit, and PromptIR protocol parity are established.
The AIO-3 Stage-A coarse model is still training. SRSC-Lite efficacy,
superiority over the matched residual-code control, AIO-3/AIO-5 publication
gates, and any SOTA claim are **not yet established**. Official-test data are
sealed until the registered selection gates are complete.

## What is and is not stored here

Tracked in Git:

- `src/`, `scripts/`, `configs/`, `tests/`, and `reports/`;
- exact train/locked-validation manifests and metric ledgers;
- selected key logs and immutable research contracts;
- environment, checkpoint hashes, and migration instructions.

Not tracked in Git:

- `.env`, tokens, SSH keys, or any credential;
- datasets, official-test images, absolute local symlink trees, or caches;
- `.pt/.ckpt` files. Large checkpoints are published separately as GitHub
  Release assets and are verified against `recovery/CHECKPOINTS.json`.

## Exact-resume quick start

Exact checkpoint resume intentionally uses the original absolute project path,
because the frozen configs and data/code contracts contain that path.

```bash
git clone https://github.com/emma-sue/aaa.git /root/autodl-tmp/srsc_lite_v12
cd /root/autodl-tmp/srsc_lite_v12
bash scripts/bootstrap_recovery.sh
```

After mounting the standard datasets and rebuilding the audited symlink tree:

```bash
python scripts/prepare_data.py --protocol aio3 --build
python scripts/prepare_data.py --protocol aio5 --build
```

Download and verify the rolling resume checkpoint (requires an authenticated
GitHub CLI for a private release; public assets also work):

```bash
python scripts/restore_checkpoint_asset.py \
  --repo emma-sue/aaa \
  --tag resume-aio3-stage-a \
  --asset last.pt \
  --destination artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt
```

Then run the validation commands printed by that script and resume through the
registered 4-GPU launcher. See [recovery/RESTORE.md](recovery/RESTORE.md) for
the complete fail-closed procedure.

## Upstream pins

- PromptIR: `va1shn9v/PromptIR@106159ab809101f2e25b6714195cd6fa9a938d36`
- R2R: `Wang et al. Retrieve-to-Restore` code snapshot
  `bf387d56095aaf4edc0b685f8ea58cce5c64c2fc`
- Microsoft ResearchStudio:
  `microsoft/ResearchStudio@61277686638adb87298a26cc7621cd7387723fb4`

The repository name is historical; the method and internal baseline names are
SRSC-Lite and Restormer-AiO, respectively.

