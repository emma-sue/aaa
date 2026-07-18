# Running Status

- Last independently audited: `2026-07-18 12:33 UTC`.
- AIO-3 Stage-A is active in tmux `srsc_pipeline` on 4×RTX 4090; `srsc_watchdog`, `srsc_stage_a_trend_chain`, and `srsc_runtime_accounting` are alive.
- Runtime remains fixed at per-GPU batch 30, world size 4, accumulation 1, global batch 120; all four GPUs are at 99–100% utilization and approximately 22.4–22.7 GiB used.
- Latest observed training point: zero-based epoch `135`, step `210700`; rank logs contain no NaN, Inf, OOM, NCCL, CUDA, traceback, or killed-process event.
- Latest committed locked validation: epoch `135`, step `210065`, five-setting mean `34.1019 dB` (below the retained best).
- Current locked-val top-1 remains epoch `125`, step `198595`: five-setting mean `34.1883311326 dB`, true three-task macro `35.3106504401 dB`.
- Top-1 per-setting PSNR: dehaze `36.526667`, denoise15 `35.261796`, denoise25 `32.780726`, denoise50 `29.472035`, derain `36.900432 dB`.
- Top-1 checkpoint: `artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/val_epoch125_step0198595.pt`, SHA256 `f16dd4974363b2b73adf4ed9fc797d3c9143193c89458c546282e87592459a8d`.
- Recent throughput is about `2.777 optimizer steps/s`; 119,800 steps remain, giving a non-binding ETA near `2026-07-19 00:31 UTC` (about 12.0 hours), subject to locked-validation and I/O overhead.
- The result is Stage-A locked validation only. It is not an SRSC gain, official-test result, or fair R2R comparison.
- Stage-B/Stage-C/AIO-5 remain gated and incomplete. Official test remains sealed.

Status command: `bash /root/autodl-tmp/srsc_lite_v12/scripts/status.sh`.
