# Stage-A Report

Status: **IN_PROGRESS**  
Latest committed locked validation: epoch `140`, step `215800`.  
Best five-setting mean PSNR: **34.1883311326 dB** at epoch `125`, step `198595`.  
Best three-task macro PSNR: **35.3106504401 dB**.  
Best checkpoint: `/root/autodl-tmp/srsc_lite_v12/artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/val_epoch125_step0198595.pt`  
Best checkpoint SHA256: `f16dd4974363b2b73adf4ed9fc797d3c9143193c89458c546282e87592459a8d`

## Best checkpoint settings

| Setting | PSNR (dB) |
|---|---:|
| dehaze | 36.526667 |
| denoise15 | 35.261796 |
| denoise25 | 32.780726 |
| denoise50 | 29.472035 |
| derain | 36.900432 |

This is the frozen coarse E+D1 locked-validation trajectory. It is not an official-test result and does not establish an SRSC or R2R gain; those claims remain gated on paired Stage-B/Stage-C experiments.
