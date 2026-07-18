# Protocol Correction — PromptIR/R2R Augmentation Support

Timestamp: 2026-07-13 UTC

## Finding

The public PromptIR and R2R loaders call `random.randint(1, 7)` and then apply
the corresponding `data_augmentation` mode.  Identity mode 0 is deliberately
excluded.  The first centered-crop-corrected local Stage-A launch instead used
three independent Bernoulli flip/transpose decisions.  That implementation
sampled all eight dihedral transforms uniformly and therefore included identity
with probability 1/8.

This difference changes the training augmentation distribution.  It does not
affect official-test parity, but it violates the registered requirement to use
the public R2R training protocol exactly enough for a credible external-table
comparison.

## Disposition

- The affected run was stopped at epoch 0, step 600, before its first scheduled
  checkpoint.
- Its logs and startup manifests are preserved under
  `artifacts/invalid_protocol/identity_inclusive_augmentation_20260713/`.
- Those losses are non-scientific diagnostics and must never be mixed with the
  corrected run or used for model selection.
- `_augment` now samples exactly one integer from 1 through 7 and implements
  the public vertical-flip/rotation compositions on CHW tensors.
- Seven mode-by-mode tests and one sampling-range test were added.
- Directed dataset tests passed `13 passed`; the complete corrected suite
  passed `57 passed in 24.39s` before restart.

## Scope Audit

The same source comparison confirmed the remaining data schedule:

- denoising lists repeated three times for each sigma 15/25/50;
- derain repeated 120 times for AIO-3 and 80 times for AIO-5;
- dehaze list used once;
- AIO-5 deblur repeated 30 times and low-light repeated 60 times;
- paired random crop and center-to-base crop semantics match;
- Gaussian noise uses NumPy standard normal, clipping, and uint8 conversion.

The local denoising implementation generates i.i.d. Gaussian noise before the
random crop/augmentation, whereas the public code generates it after.  For
spatially i.i.d. isotropic Gaussian noise, cropping and a dihedral permutation
commute in distribution.  Pixel quantization, sigma, clipping, and target
pairing are unchanged; this is not treated as a distributional protocol error.

