# GitHub backup and migration contract

## Storage model

Ordinary Git stores reviewable text artifacts. A 304 MB PyTorch checkpoint is
never committed directly because every replacement would permanently expand
repository history. Instead:

1. every controlled AIO-3/AIO-5 Stage-A, formal Stage-B/C, pretrain or finetune
   run publishes its current locked-validation best and retained top-3 as
   immutable Release assets;
2. a resumable `last.pt` is a rolling
   `resume-<protocol>-<stage>-<run>` Release asset, refreshed
   at locked validation, before shutdown, or at the configured hourly interval;
3. each asset has a SHA256 sidecar and metadata JSON;
4. `recovery/CHECKPOINTS.json` records the checkpoint's own config, split,
   code/data and distributed-runtime contracts and binds them to a real Git
   content commit/tree and Release tag;
5. the Release must succeed before the Git checkpoint index is updated.

Run discovery is fail-closed: Stage-A uses the explicit registered naming
contract; later runs must have an immutable `run_contract.json` whose run name,
protocol, stage, config, split and checkpoint-carried hash agree. Names marked
official, smoke, debug, invalid, tmp or test are excluded. Pilot runs are
bounded plumbing checks: their compact model hash and metadata may appear in
Git, but the daemon never creates a large pilot Release asset.

GitHub exposes one repository-global `Latest` Release. The daemon explicitly
marks every historical top-3 backfill and rolling resume Release as
`make_latest=false`, then promotes only the most recently written current or
formal scientific head. Thus uploading an old retained best cannot make it
look newer than the true current best.

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
- make every exported mirror an exact allowlisted closed set, remove stale
  files before commit, and reject any extra tracked path during verification;
- scan every text file for common token/private-key patterns before copying;
- reject any ordinary Git file larger than 95 MB;
- never infer that a checkpoint uploaded successfully from a command exit alone:
  compute and record SHA256 first, then verify the remote digest and size of the
  checkpoint, sidecar, and metadata assets;
- restore only to the exact project path for legacy AIO-3 continuation;
- publish no dataset bytes, `.env`, access token, private key or credential;
- do not treat a GitHub backup as an official-test authorization or a result.
