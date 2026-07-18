# Stage-A Report

Status: **IN_PROGRESS**  
Latest committed locked validation: epoch `160`, step `238740`.  
Best five-setting mean PSNR: **34.1973872016 dB** at epoch `145`, step `221535`.  
Best three-task macro PSNR: **35.3004437544 dB**.  
Best checkpoint: `/root/autodl-tmp/srsc_lite_v12/artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/val_epoch145_step0221535.pt`  
Best checkpoint SHA256: `fc15d85e1dd8127ea84b9785ae6f993d352f07f0a3eefb1f103feb8f36c4cdc0`

## Best checkpoint settings

| Setting | PSNR (dB) |
|---|---:|
| dehaze | 36.565481 |
| denoise15 | 35.301834 |
| denoise25 | 32.817962 |
| denoise50 | 29.508610 |
| derain | 36.793048 |

This is the frozen coarse E+D1 locked-validation trajectory. It is not an official-test result and does not establish an SRSC or R2R gain; those claims remain gated on paired Stage-B/Stage-C experiments.
