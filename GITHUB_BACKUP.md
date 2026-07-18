# GitHub backup and migration contract

## Storage model

Ordinary Git stores reviewable text artifacts. A 304 MB PyTorch checkpoint is
never committed directly because every replacement would permanently expand
repository history. Instead:

1. every new locked-validation top-1 checkpoint is an immutable Release asset;
2. `last.pt` is a rolling `resume-<protocol>-<stage>` Release asset, refreshed
   at locked validation, before shutdown, or at the configured hourly interval;
3. each asset has a SHA256 sidecar and metadata JSON;
4. `recovery/CHECKPOINTS.json` binds checkpoints to the config, split, metrics,
   code snapshot, distributed runtime, and Release tag;
5. the Release must succeed before the Git checkpoint index is updated.

The backup daemon refuses a public repository unless `--allow-public` is passed
explicitly. This repository was intentionally authorized for public release by
the project owner on 2026-07-18 UTC.

## Local mirror

The live training tree remains outside Git so snapshotting cannot disturb DDP.
The exporter builds an allowlisted mirror at:

```text
/root/autodl-tmp/srsc_lite_v12_github
```

Run one local snapshot:

```bash
python scripts/export_repro_snapshot.py \
  --source /root/autodl-tmp/srsc_lite_v12 \
  --destination /root/autodl-tmp/srsc_lite_v12_github
```

Once `gh auth status` succeeds, start the remote daemon:

```bash
tmux new-session -d -s srsc_github_backup \
  "cd /root/autodl-tmp/srsc_lite_v12 && \
   python scripts/checkpoint_backup_daemon.py \
     --repo emma-sue/aaa --allow-public \
     --source /root/autodl-tmp/srsc_lite_v12 \
     --mirror /root/autodl-tmp/srsc_lite_v12_github \
     --interval-seconds 900 --resume-interval-hours 1 \
     2>&1 | tee -a artifacts/logs/github_backup.log"
```

`--local-only` performs allowlisted snapshots and local commits without any
external write. It is safe to run before GitHub CLI authentication is ready.

## Fail-closed rules

- never copy a symlink, dataset, archive, checkpoint, `.env`, or credential;
- scan every text file for common token/private-key patterns before copying;
- reject any ordinary Git file larger than 95 MB;
- never infer that a checkpoint uploaded successfully from a command exit alone:
  compute and record SHA256 first;
- restore only to the exact project path for legacy AIO-3 continuation;
- do not treat a GitHub backup as an official-test authorization or a result.

