# Official PromptIR Checkpoint Parity

## Immutable inputs

- Upstream commit: `106159ab809101f2e25b6714195cd6fa9a938d36`
- Official checkpoint SHA256: `b77ef1a099b756c5f59a32e86f4b75616f467258e997e46fe5acf57855a912c3`
- Evaluator: `scripts/verify_promptir_baseline.py`
- Per-image output: `artifacts/metrics/promptir_official_aio3.csv`
- Summary: `artifacts/metrics/promptir_official_aio3.json`

The checkpoint is strict-loaded into the unmodified official `PromptIR` class. Images are RGB, clamped to `[0,1]` for metrics, center-cropped to a multiple of 16 exactly as in official `crop_img`, and Gaussian noise uses the official one-time NumPy seed and σ15→25→50 order.

## Results

| Setting | This execution | R2R Table 1 cites for PromptIR | Delta |
|---|---:|---:|---:|
| SOTS | 30.5947 | 30.58 | +0.0147 |
| Rain100L | 37.4396 | 36.37 | +1.0696 |
| BSD68 σ15 | 33.9771 | 33.98 | -0.0029 |
| BSD68 σ25 | 31.3122 | 31.31 | +0.0022 |
| BSD68 σ50 | 28.0459 | 28.06 | -0.0141 |

Four of five settings reproduce the cited values within 0.02 dB. Rain100L is higher by 1.07 dB. This is not caused by a local dataset substitution: the newly downloaded official PromptIR Rain100L archive was compared with the evaluation files, and after official base-16 crop all 100 input and all 100 GT arrays are pixel-identical. The model class, checkpoint, centered preprocessing, and metric path are official/audited.

Decision: `PASS_CODE_AND_CHECKPOINT_PARITY_WITH_REPORTED_RAIN100L_DISCREPANCY`. Do not alter Rain100L or calibrate outputs to force agreement with a copied table value. Report both the immutable checkpoint execution and the cited table number.

## Baseline roles and locked training protocol

The following names are not interchangeable:

- **PromptIR** is the immutable published external baseline and the source of the data/evaluation infrastructure plus generic Restormer blocks.
- **Restormer-AiO** is the clean internal baseline after physically deleting PromptIR's prompt generators, prompt banks, prompt concatenation, noise-level reductions, and all label/prompt-conditioned paths.
- **SRSC-Lite** is the proposed two-stage end-to-end model built on the clean Restormer-AiO components.
- **R2R** is a separate external reported comparator. Its public 3D/5D training budgets are used to make the current full runs competitive; no R2R retrieval bank or prompt mechanism enters SRSC-Lite.

Source-code audit of `/root/R2R/options/options_3D.py`, `options_5D.py`, `train_3D.py`, and `train_5D.py` gives:

| Protocol | Phase | Epochs | Global batch | Crop | LR | Optimizer | Warm-up | Gradient clip |
|---|---|---:|---:|---:|---:|---|---:|---:|
| AIO-3 | pretrain | 240 | 64 | 128 | 2e-4 | Adam (.9/.999) | 15 epochs from 1e-7 | 0.01 norm |
| AIO-3 | finetune | 30 | 16 | 224 | 1e-6 | Adam (.9/.999) | none | 0.01 norm |
| AIO-5 | pretrain | 240 | 120 | 128 | 2e-4 | Adam (.9/.999) | 15 epochs from 1e-7 | 0.01 norm |
| AIO-5 | finetune | 30 | 32 | 224 | 1e-6 | Adam (.9/.999) | none | 0.01 norm |

The current four locked configs match these values. Pretraining uses the public R2R `max_epochs=epochs+30` cosine definition; finetuning uses cosine `T_max=30, eta_min=1e-8`. Additional 1,000-step recovery checkpoints and train-only `locked_val` evaluation do not alter optimizer steps or learning rates.

Execution caveat: the single-24GB implementation uses BF16 and gradient accumulation while preserving the registered effective batches. Every internal method/control uses the same precision and accumulation. Therefore internal causal comparisons are capacity/protocol matched, but comparisons to R2R Table 1/2 remain comparisons against externally reported results, not a claim of bitwise R2R reproduction.
