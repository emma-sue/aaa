# SRSC-Lite v1.4 — Current Audit

Audit snapshot: 2026-07-13 UTC. This report describes verified implementation/protocol state; it does not claim method efficacy.

## 1. Immutable upstream and naming

- Official repository: `https://github.com/va1shn9v/PromptIR`.
- Checked-out commit: `106159ab809101f2e25b6714195cd6fa9a938d36`.
- Remote and HEAD were re-read from the local immutable clone; upstream working tree is clean.
- Reused generic components: MDTA, GDFN, LayerNorm, PixelShuffle/PixelUnshuffle, overlapping patch embedding, and ordinary encoder/decoder infrastructure.
- Physically removed from the new model: PromptGenBlock, prompt banks/weights, prompt1/2/3, prompt concatenation, noise_level*, reduce_noise_level*, reduce_noise_channel_*, chnl_reduce*, and prompt/label-conditioned paths.
- MoCE-IR expert/routing mechanisms are not present.

The resulting internal baseline is named **Restormer-AiO**, not PromptIR. PromptIR is retained as an immutable published external baseline. R2R is a separate external comparator and contributes no retrieval bank or prompt mechanism to SRSC-Lite.

## 2. Environment and measured capacity

- Active GPU: NVIDIA GeForce RTX 3090, 24GB.
- PyTorch: 2.3.0+cu121; torchvision: 0.18.0+cu121.
- Training precision: BF16 for every internal method/control.
- SRSC-Lite: 29,938,604 parameters, including the always-instantiated 81-to-8 O14 ceiling adapter.
- Clean Restormer-AiO: 25,437,220 parameters.
- Parameter-matched Restormer-AiO (dim52): 29,805,484 parameters; parameter difference from SRSC-Lite is 0.445%.
- At 256x256, fvcore counted operations are 102.553G (clean), 119.931G (matched), and 116.468G (SRSC-Lite). These are reported as fvcore operations/FLOPs, not relabeled as MACs.
- Median BF16 latency at 256x256 is 38.02ms (clean), 40.39ms (matched), and 39.76ms (SRSC-Lite).
- Current AIO-3 Stage-A crop128/micro-batch16 measured CUDA peak is about 11.1GiB.
- The full SRSC Stage-C crop224/micro-batch4 smoke path measured about 18.8GiB and completed finite forward/backward/optimizer execution on the 24GB card.

Raw profile: `artifacts/stats/model_profile_256.json`.

## 3. Locked training protocol

The public R2R 3D/5D source was selected as the competitive training-budget reference; PromptIR remains the code/data mother repository.

| Protocol | Phase | Epochs | Effective batch | Crop | LR | Workers |
|---|---|---:|---:|---:|---:|---:|
| AIO-3 | pretrain | 240 | 64 | 128 | 2e-4 | 32 |
| AIO-3 | finetune | 30 | 16 | 224 | 1e-6 | 32 |
| AIO-5 | pretrain | 240 | 120 | 128 | 2e-4 | 24 |
| AIO-5 | finetune | 30 | 32 | 224 | 1e-6 | 32 |

All four phases use Adam beta=(0.9,0.999), weight decay 0, and global-norm gradient clipping 0.01. Pretraining uses 15-epoch linear warm-up from 1e-7 followed by the R2R `max_epochs=epochs+30` cosine definition. Finetuning uses cosine `T_max=30, eta_min=1e-8`.

Single-GPU gradient accumulation preserves the registered effective batches. Additional 1,000-step atomic recovery checkpoints and train-only locked validation do not alter optimizer steps or LR. Official test remains unavailable to configuration/checkpoint selection.

Execution caveat: R2R public Lightning training does not explicitly request BF16, whereas the current single-24GB protocol uniformly uses BF16. Internal comparisons remain precision matched. R2R Table 1/2 comparisons are therefore external reported-result comparisons, not claims of bitwise R2R reproduction.

## 4. Data integrity gate

Strict current manifests:

- AIO-3: `expected_entries=149814`, `missing_entries=0`.
- AIO-5: `expected_entries=154990`, `missing_entries=0`.
- Project data symlinks were checked with zero broken links.

Verified standard assets include WaterlooED, BSD400, CBSD68, RainTrainL, official Rain100L, official SOTS outdoor, GoPro, LOL-v1, and OTS.

- LOL-v1: 485 train pairs and 15 eval pairs; `/root/autodl-tmp/.autodl/lowlight/LOLdataset.zip` passes archive integrity testing.
- GoPro: 2,103 train pairs and 1,111 test pairs.
- OTS source: exact-match public mirror `brunobelloni/outdoor-training-set-ots-reside`.
- OTS archive length: 11,877,920,850 bytes; verified MD5: `d87ca5941186701c42255459e177eb76`.
- Materialized OTS subset: all 72,135 registered hazy files and 2,061 unique clean GT images.

OTS preparation is complete; prior notes saying it was still downloading/materializing are historical and must not be used as current state. ITS, RESIDE-6K, Rain13K, and MIOIR are not silently substituted into the locked PromptIR/R2R protocol.

The training augmentation was re-audited against the public source. PromptIR/R2R samples modes 1--7 and excludes identity; the local loader now matches those seven rotation/vertical-flip compositions exactly. An earlier 600-step warmup-only run that included identity with probability 1/8 is archived and scientifically invalid. Evidence: `reports/PROTOCOL_CORRECTION_AUGMENTATION.md`.

## 5. Official PromptIR checkpoint parity

- Official checkpoint SHA256: `b77ef1a099b756c5f59a32e86f4b75616f467258e997e46fe5acf57855a912c3`.
- It strict-loads into the unmodified official PromptIR class.
- After correcting the local loader to the exact centered `crop_img(..., base=16)` used by both public PromptIR and R2R, reproduced results are: SOTS 30.5947, Rain100L 37.4396, BSD68 sigma15 33.9771, sigma25 31.3122, sigma50 28.0459.
- Four settings are within 0.04dB of the R2R-cited PromptIR row. Rain100L is 1.07dB higher; input/GT arrays were proven pixel-identical after official crop, so outputs are not calibrated to force table agreement.

Detailed evidence: `reports/BASELINE_PARITY.md` and `artifacts/metrics/promptir_official_aio3.*`.

## 6. Implementation checks

Verified invariants include:

- 9 descriptor channels x 3x3 unfold = one consistent 81-D coordinate space.
- Fixed row-orthogonal projections have shapes P=6x81 and Pr=8x81.
- Signed progress, transverse residual, magnitude, and direction are computed in the same local space.
- S1-S4 are generated/consumed natively at H, H/2, H/4, and H/8.
- Every feedback interface is exactly 8 channels.
- Zero-initialized SRSC-Mod is identity within tolerance.
- GT target builder is absent from deployable `forward(x)`.
- State supervision/evidence is detached from E/D1 as registered.
- Odd-sized input padding/crop-back and checkpoint roundtrip are tested.
- No prompt/MoE/task-label mechanism exists in `src/net`.

The orchestrator startup suite after the centered-crop and exact augmentation-support corrections passed `57 passed in 19.23s` (`artifacts/logs/orchestrate_aio3_pytest.log`). A subsequent CPU-only public-scheduler equivalence test was added and the latest complete suite passed `58 passed in 19.13s`. The suite includes exact center-crop examples, all seven public non-identity augmentation modes and their sampling range, every epoch of the public R2R pretraining LR closed form, odd-size behavior, coordinate geometry, feedback controls, gradient routing, checkpoint round-trip, prompt/MoE removal, pipeline guards, and report-contract checks. The scheduler test does not change the running model or optimization path.

Historical pre-Stage-B reload evidence at step 19,000 is retained only as provenance. A later protocol audit found that local `_crop_to_base` removed all remainder pixels from the bottom/right, while both official PromptIR and R2R split them around the image center. The affected Stage-A run was stopped at the complete epoch-9/step-19,359 boundary and archived under `artifacts/invalid_protocol/top_left_crop_20260713`; its coarse checkpoints and locked validation are explicitly invalid for scientific use. The corrected source passes `49` tests and official PromptIR parity was rerun before fresh Stage-A training.

## 7. Runtime and recovery audit

- The first formal AIO-3 Stage-A attempt is archived as invalid due to the top-left-versus-center crop mismatch. The replacement run with the same registered name started from random initialization after corrected parity and a new 34-resource startup contract; its first valid record is epoch0/step50 at LR `1e-7`.
- A read-only monitor runs in `srsc_watchdog`.
- An external interruption was recovered from the complete epoch-7/step-15,057 checkpoint.
- Replayed losses through step15,250 were bitwise identical to pre-interruption records.
- The first new post-recovery atomic checkpoint is step16,000, SHA256 `f70495f30f9a0bb9a3240c501666013e1ac742de520ff729c15ad2c3429e0195`.
- That checkpoint fully loads on CPU and contains model, 341 Adam states, scheduler, config/split hashes, args, and Torch/CUDA/NumPy/Python RNG states. It has no prohibited model-state key.
- Orchestrator locking now uses kernel flock, so dead processes do not leave permanent false locks while live duplicate orchestrators remain rejected.

## 8. Scientific status and hard limits

Only the corrected replacement Stage-A coarse training is in progress. No SRSC efficacy result exists yet. The archived epoch-5 locked validation belongs to the invalid preprocessing attempt and must never be compared with R2R or reused for checkpoint selection.

Not yet established:

- Oracle signed-progress information value;
- independent direction contribution;
- predicted-state learnability;
- superiority over matched 8-D residual code;
- joint Stage-C restoration gain;
- AIO-3/AIO-5 publication gate or R2R +0.30dB target.

AutoSOTA remains setup-only because the architecture has no formal positive Stage-B/Stage-C trend yet and no real API key is configured. It cannot define the novelty, access official-test feedback, merge task-specific checkpoints, or rescue a scientifically failed representation.

Current allowed statement:

```text
Implementation and baseline/data parity are established; SRSC-Lite efficacy is not yet established.
```
