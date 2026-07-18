# Protocol Correction: Official Center Crop

Status: **CORRECTED; PRIOR STAGE-A ATTEMPT INVALIDATED AND ARCHIVED**.

## Defect

The local `_crop_to_base` previously returned `image[..., :h2, :w2]`, dropping all remainder pixels at the bottom and right. Both immutable public sources use a centered crop:

```python
image[crop_h // 2 : h - crop_h + crop_h // 2,
      crop_w // 2 : w - crop_w + crop_w // 2]
```

Verified sources:

- `upstream/PromptIR/utils/image_utils.py`
- `/root/R2R/utils/image_utils.py`

## Scope

- BSD68: 68/68 images are nonmultiples of 16, but remainder 1 makes center and former top-left crop coincide.
- Rain100L: 100/100 input and target images have remainder 1, so the two crops coincide.
- SOTS: 500/500 inputs and 492 clean targets are nonmultiples of 16, commonly with nonzero center offsets; results and training patch support differ.
- GoPro: all 1,111 pairs are already multiples of 16.
- LOL-v1: 15/15 pairs have width remainder 8 and require a four-pixel horizontal center offset.

The defect therefore changes AIO-3 dehaze training/evaluation and future AIO-5 low-light training/evaluation. It cannot be treated as an evaluation-only correction.

## Containment

- The running trainer was terminated only after a complete epoch boundary checkpoint existed: epoch 9, batch 0, step 19,359.
- The entire affected run, logs, metrics, top-3 index, and checkpoints were moved without deletion to `artifacts/invalid_protocol/top_left_crop_20260713/`.
- These artifacts are diagnostic-only and cannot initialize, select, or support any scientific claim.
- No Stage-B work had started.

## Correction evidence

- `_crop_to_base` now implements the exact PromptIR/R2R centered indices.
- Unit tests cover `(29,38)`, the SOTS-like `(413,550)`, and LOL-like `(400,600)` sizes, including an explicit non-top-left assertion.
- Full suite: `49 passed in 19.05s`.
- Official PromptIR checkpoint was rerun over the complete AIO-3 test sets after correction:
  - SOTS: 30.5946808 / 0.9778579
  - Rain100L: 37.4395734 / 0.9786196
  - BSD68 sigma15: 33.9770679 / 0.9332820
  - BSD68 sigma25: 31.3121810 / 0.8885848
  - BSD68 sigma50: 28.0459036 / 0.7969571
- SOTS is now within +0.0147 dB of the PromptIR number cited by R2R Table 1.

Only a fresh from-scratch Stage-A run under this corrected contract is eligible to continue.
