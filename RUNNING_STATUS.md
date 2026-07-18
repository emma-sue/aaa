# Running Status

- Last independently audited: `2026-07-18 14:47 UTC`.
- AIO-3 Stage-A is active in tmux `srsc_pipeline` on 4×RTX 4090; `srsc_watchdog`, `srsc_stage_a_trend_chain`, and `srsc_runtime_accounting` are alive.
- Runtime remains fixed at per-GPU batch 30, world size 4, accumulation 1, global batch 120; all four GPUs remain at 99–100% utilization and approximately 22.4–22.7 GiB used.
- Latest observed training point: zero-based epoch `154`, step `232250`; no protocol change was made.
- Current locked-validation top-1 is epoch `145`, step `221535`, five-setting mean `34.1973872016 dB`.
- Top-1 per-setting PSNR: dehaze `36.565481`, denoise15 `35.301834`, denoise25 `32.817962`, denoise50 `29.508610`, derain `36.793048 dB`.
- Top-1 checkpoint: `artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/val_epoch145_step0221535.pt`; SHA256 `fc15d85e1dd8127ea84b9785ae6f993d352f07f0a3eefb1f103feb8f36c4cdc0`.
- Stage-A final target remains epoch `240`, step `330500`. This is locked validation only, not an SRSC gain, official-test result, or fair R2R comparison.
- Stage-B/Stage-C/AIO-5 remain gated and incomplete. Official test remains sealed.
- Current code tree CPU-only regression: `295 passed, 2 known scheduler warnings`.

Status command: `bash /root/autodl-tmp/srsc_lite_v12/scripts/status.sh`.
