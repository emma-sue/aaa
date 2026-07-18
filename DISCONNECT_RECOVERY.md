# Network-disconnect recovery

Snapshot time: 2026-07-18 03:54 UTC

The four-GPU Stage-A run is detached in tmux and does not depend on the SSH connection.

- tmux session: `srsc_pipeline`
- watchdog: `srsc_watchdog`
- run: `aio3_stage_a_coarse_seed1415926`
- runtime: 4 x RTX 4090, per-GPU batch 30, global batch 120, DDP/NCCL
- last verified resumable checkpoint: epoch 60, step 124040, batch 0
- checkpoint: `artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt`
- checkpoint SHA256: `ab3aa8376e76b31072335269e25424a41b5c6df1e9ee878dc6936cc094179b98`
- integrity report: `artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/epoch60_ddp_integrity.json`
- integrity status: PASS
- locked-val top-1: `val_epoch060_step0124040.pt`, SHA256 `4755f270bd7eea7d6820fff60be5623d1fdb1d693d16cc1bea504e77b3607d15`
- locked-val five-setting mean: `33.8784359659 dB`

After reconnecting, inspect without restarting anything:

```bash
cd /root/autodl-tmp/srsc_lite_v12
bash scripts/status.sh
tmux attach -t srsc_pipeline
```

Detach from tmux with `Ctrl-b`, then `d`.

Only if all four DDP trainers and the `srsc_pipeline` tmux session are no
longer running, recreate the four-GPU Stage-A continuation.  It resumes the
atomic checkpoint and, after Stage-A completes, returns to the downstream
Oracle/Predicted/Stage-C and AIO-5 orchestrated continuation:

```bash
cd /root/autodl-tmp/srsc_lite_v12
tmux new-session -d -s srsc_pipeline \
  "cd /root/autodl-tmp/srsc_lite_v12 && \
   bash scripts/launch_aio3_stage_a_4x4090.sh \
   2>&1 | tee -a artifacts/logs/pipeline_ddp.log"
```

Do not run the recovery command while any `train_stage_a_ddp.py` rank or an
existing `srsc_pipeline` session is alive.  Never launch a trainer directly
as the long-horizon recovery path, because doing so would omit the post-Stage-A
scientific gates and the AIO-5 continuation.
