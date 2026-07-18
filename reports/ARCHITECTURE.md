# SRSC-Lite v1.4 — Audited Architecture

## 1. Method boundary

SRSC-Lite is a conventional end-to-end, fixed-depth all-in-one restoration network. Its deployable forward accepts only one degraded RGB image `x` and returns one restored RGB image `y2`.

```text
single x -> shared encoder -> coarse decoder -> deterministic assessor
         -> correction decoder -> y2
```

It is **not** an AgenticIR-style tool scheduler. There is no VLM/LLM, external restoration tool, task label, prompt bank, mixture of experts, recurrence, dynamic stopping, or test-time GT branch. The fixed two-stage computation is `K=2`; D1 and D2 do not share weights.

Code entry points:

- generic Restormer blocks: `src/net/restormer_blocks.py`;
- clean single-stage baseline: `src/net/clean_restormer_aio.py`;
- SRSC model: `src/net/srsc_lite.py`;
- training-only coordinate targets: `src/net/srsc_coordinates.py`.

## 2. What was retained and deleted from PromptIR

PromptIR is used as the audited repository/data/evaluation mother code. The new model retains only generic components: OverlapPatchEmbed, MDTA, GDFN, LayerNorm, PixelUnshuffle downsampling, PixelShuffle upsampling, and an encoder-decoder layout.

The clean internal baseline and SRSC-Lite physically exclude:

- `PromptGenBlock`, `prompt1/2/3`, prompt parameters/banks/weights;
- prompt concatenation and prompt-conditioned decoder paths;
- `noise_level1/2/3`, `reduce_noise_level*`, `reduce_noise_channel_*`, and `chnl_reduce*`;
- task/degradation labels and noise-level inputs;
- MoCE-IR experts, sparse/top-k routing, complexity bias, frequency gates, and load balancing.

`tests/test_prompt_removal.py` scans the new model sources for the prohibited mechanisms. The audited step-16,000 checkpoint contains 610 model tensors and no prompt/noise-level/channel-reduction/MoE/expert/router state key.

Once prompts are deleted, the internal baseline is named **Restormer-AiO**, not PromptIR. PromptIR remains a separate published external baseline.

## 3. Deployable forward topology

Inputs are reflect-padded to a multiple of 8 and cropped back exactly at the output.

```text
x: Bx3xHxW
|
| Shared Restormer encoder E (executed once)
|-- F1: Bx48 xH   xW      [4 blocks, 1 head]
|-- F2: Bx96 xH/2 xW/2   [6 blocks, 2 heads]
|-- F3: Bx192xH/4 xW/4   [6 blocks, 4 heads]
`-- F4: Bx384xH/8 xW/8   [8 blocks, 8 heads]
     |
     | D1: ordinary coarse decoder
     |  F4 up 384->192 + F3 -> fuse 384->192 -> 2 blocks
     |  up 192->96 + F2    -> fuse 192->96  -> 2 blocks
     |  up 96->48 + F1     -> fuse 96->48   -> 2 blocks
     |  3x3 RGB head -> delta0
     `-- y1 = x + delta0
          |
          |-- shallow y1 pyramid G
          |   G1=24@H, G2=48@H/2, G3=96@H/4, G4=192@H/8
          |
          |-- deterministic state assessor A
          |   evidence = [stopgrad(x), stopgrad(y1), stopgrad(y1-x)]
          |   + compressed stopgrad(F1..F4)
          |   -> S1,S2,S3,S4, each exactly 8 channels at its native scale
          |
          `-- D2: residual correction decoder
              H/8: [F4(384),G4(192)] -> 1x1 576->384
                   -> SRSC-Mod4(S4)
              H/4: up 384->192 + F3(192)+G3(96) -> 1x1 480->192
                   -> SRSC-Mod3(S3) -> 4 blocks
              H/2: up 192->96 + F2(96)+G2(48) -> 1x1 240->96
                   -> SRSC-Mod2(S2) -> 4 blocks
              H:   up 96->48 + F1(48)+G1(24) -> 1x1 120->48
                   -> SRSC-Mod1(S1) -> 4 blocks -> 2 refinement blocks
              3x3 RGB head -> delta1
              y2 = y1 + delta1
```

D2 predicts only a residual correction; it does not reconstruct an unrelated full image. The shared encoder is not executed a second time. The block allocation is D1 `2+2+2=6`, D2 `4+4+4+2=14`, preserving 20 total decoder Transformer blocks, equal to the clean single-stage decoder budget.

## 4. State assessor and modulation

The assessor is a small CNN, not a library of small restoration models. Its image-evidence stem is `9->32->32`; successive stride-2 evidence widths are `32,48,64,96`. At each scale, a 1x1-compressed frozen encoder feature is fused with image evidence, followed by two depthwise-separable residual blocks and one linear 8-channel head. Signed channels have no ReLU.

At each D2 scale, SRSC-Mod computes:

```math
h=Conv(GELU(Conv(S))),
\quad \gamma=0.1\tanh(Conv_\gamma(h)),
\quad \beta=Conv_\beta(h),
\quad F'= (1+\gamma)F+\beta.
```

The final gamma/beta layers are zero-initialized, so the initial modulator is exactly the identity. Modulation occurs after skip/y1 fusion and before the Restormer blocks; it is not inserted into the encoder or multiplied directly onto RGB output.

## 5. Training-only SRSC coordinates

GT is used only to construct supervision during training. It never enters `model.eval(); model(x)`.

For each native scale, images are first area-downsampled and then described by:

```math
\psi(I)=[I,\;0.5\,Sobel_x(I),\;0.5\,Sobel_y(I)].
```

A reflect-padded 3x3 unfold gives an 81-D local patch vector:

```math
v^*=unfold_3(\psi(gt)-\psi(x)),
\qquad v_1=unfold_3(\psi(y_1)-\psi(x)).
```

All inner products, norms, projection residuals, and direction projections remain in this same 81-D space:

```math
\alpha=\frac{\langle v_1,v^*\rangle}{\|v^*\|^2+10^{-6}},
\quad p_{raw}=1-\alpha,
\quad e=v_1-\alpha v^*.
```

The bounded signed progress and transverse magnitude are:

```math
p=2\tanh(p_{raw}/2),
\qquad
m=2\tanh\left(\frac{\|e\|/(\|v^*\|+\epsilon)}{2}\right).
```

Direction first normalizes `e` in 81-D and then applies a fixed seeded row-orthogonal `P in R^(6x81)`. Train-only robust thresholds define edit validity `q` and direction validity `w_dir`. The actual feedback target is exactly:

```math
S_{eff}=[q p,\;q m,\;w_{dir}d_1,\ldots,w_{dir}d_6]\in\mathbb R^8.
```

The construction is a **target-relative local progress/transverse coordinate**, not a claim that physical degradations possess a unique decomposition.

The report-only PCA ablation is fitted once from the deterministic train-only statistics subset. Valid 81-D unit deviations are mean-centered; the top six covariance eigenvectors are sign-canonicalized, row-orthonormal, frozen, and serialized together with the 81-D training mean. O15/P15 replaces only the direction projection with `P_pca(u-mu)` while keeping p, m, gating, 8-channel width, assessor, D2, schedule, normalization, and direction-cosine supervision identical. PCA never replaces O7 as the main method and is not a GO criterion.

## 6. Fixed-width causal controls

Every feedback arm supplies four native-scale tensors of shape `Bx8xHi xWi` to the same D2 and SRSC-Mod interface. Missing coordinates are zero-filled rather than changing head width. The no-state arm still executes the assessor, adapter, modulators, y1 pyramid, and D2, then zeros the common interface; thus it is compute/capacity matched.

Key controls include:

- O/P3: magnitude-only error proxy;
- O/P4: unsigned U/D state;
- O/P5: signed progress only;
- O/P6: signed progress plus magnitude;
- O/P7: full SRSC-Lite `p+m+d`;
- O/P12: matched 8-D fixed projection of direct GT residual;
- O8/O9/O10/O11: absolute-sign, shuffled-direction, zero-direction, and random-noise negative controls.
- O13: fixed 8-D projection of the full 81-D transverse residual, diagnostic only;
- O14: full 81-D direct GT correction passed through the shared trainable 81-to-8 adapter instantiated in every arm, non-deployable ceiling only.

O9 uses a deterministic cyclic cross-sample derangement during multi-image training batches; per-image variable-size validation uses a deterministic half-image spatial displacement so `batch=1` cannot silently turn the control back into the true direction. O11 uses a scale-specific local fixed RNG and therefore neither changes across validation calls nor consumes the global training RNG.

Oracle feedback compares representation information value without an assessor. Predicted feedback uses one identical assessor architecture, evidence, parameter count, output width, schedule, and loss scale. Oracle and predicted results must not be mixed in one claim.

Formal GO requires independently retrained O8/O9/O10/O11 and, after the primary predicted ladder is promising, independently retrained P8/P9/P10/P11. Sign-absolute, shuffled/zero-direction, and random-noise controls must each lose at least the preregistered margin; a test-time-only corruption is not mechanism evidence.

O7/O8/O9/O10/O15 share the registered direction-cosine loss. For O9, `w_dir` undergoes the same cross-sample/spatial corruption as its direction target; therefore a control difference cannot be attributed to mismatched supervision. O15/P15 are trained only after the corresponding main ladder passes and are reported as fixed-random-versus-PCA ablations.

After the Oracle mechanism controls pass, O13 and O14 are independently trained to the same formal budget. They are stored in a separate bandwidth-diagnostic/ceiling table and never authorize scientific GO, enter the deployable main table, or substitute for the matched O12 comparison.

## 7. Gradient and stage semantics

- **Stage A:** train only E+D1; select the frozen coarse checkpoint by locked-validation macro PSNR (never by official test). Assessor, y1 pyramid, D2, and ceiling adapter are frozen. Coordinate statistics and every Stage-B arm initialize from this selected checkpoint rather than blindly using the final epoch.
- **Stage B Oracle:** strictly import only the trained E+D1 tensors from the selected Stage-A checkpoint, freeze them, and retain a freshly seeded D2; GT-derived 8-D coordinates train the identical D2 template. Assessor executes as a capacity/compute-matched freshly seeded dummy but does not supply D2.
- **Stage B Predicted:** strictly import only E+D1, freeze them, and train the freshly seeded identical assessor/D2 to predict/consume the selected 8-D representation. Arms sharing a seed have identical feedback-path initialization; the two preregistered repeat seeds genuinely change both initialization and data order rather than inheriting Stage-A's unused random D2.
- **Stage C:** joint finetuning; E+D1 use 0.1x main LR, while assessor/D2 use the main LR. State targets always use detached y1, and state loss cannot update E+D1. O0/P6/P7/P8/P9/P10/P11/P12 are independently fine-tuned from their corresponding Stage-B checkpoints. Publication GO requires the signed/direction/noise/residual-code ordering to survive joint training; official test is consumed only for frozen O0/P7 and the clean baselines.

Restoration uses full-RGB L1. Joint stages use `0.5 L1(y1,gt)+L1(y2,gt)` plus the registered state and clean-preservation terms. All representation controls share the base 8-channel SmoothL1 supervision; only SRSC direction adds the preregistered weighted cosine term.

## 8. Capacity and measured execution

Measured on one 256x256 RGB input; fvcore values are reported as counted operations, not relabeled MACs:

| Model | Parameters | fvcore operations | Median BF16 latency |
|---|---:|---:|---:|
| clean Restormer-AiO | 25,437,220 | 102.553G | 38.02 ms |
| parameter-matched Restormer-AiO, dim52 | 29,805,484 | 119.931G | 40.39 ms |
| SRSC-Lite | 29,938,604 | 116.468G | 39.76 ms |

SRSC-Lite differs from the parameter-matched baseline by 0.445% parameters and has fewer counted operations. Current AIO-3 Stage-A at crop128/micro-batch16 peaks near 11.1GiB allocated CUDA memory; the full Stage-C model at crop224/micro-batch4 was separately smoke-profiled near 18.8GiB on a 24GB card.

The main ownership allocation is D1/D2=`6/14` blocks. After the complete main Predicted Stage-B passes, a preregistered AIO-3 robustness run trains a second coarse model from scratch with D1 blocks `[3,3,4]` and D2 blocks `[3,3,3]+1` refinement, i.e. D1/D2=`10/10`. Total decoder blocks remain 20; deep, middle, and shallow/refinement totals remain `6/6/8`; an instantiated model has exactly the same 29,938,604 parameters. This branch performs no hyperparameter search and trains only P6, P7, and P12 under the same 30-epoch Stage-B protocol. General restoration-state superiority requires both `P7>P6` and `P7>P12`; failure is recorded as capacity sensitivity and vetoes publication GO even if the 6/14 main split is strong.

Coordinate statistics are cryptographically bound to both the locked split and the selected Stage-A checkpoint. A main 6/14 arm cannot silently consume 10/10 thresholds/PCA/normalization or statistics generated from a different coarse epoch.

After all standard configurations are frozen, a separate paired local-composite test adds deterministic sigma-25 noise inside a feathered ellipse on Rain100L degraded inputs, retaining the original clean target. This evaluates unseen global-rain plus local-noise composition and is excluded from AIO averages. Robustness GO requires P7 to exceed both matched O0 and the parameter-matched plain baseline by at least 0.10dB, with SSIM delta at least -0.0001 and a positive paired-bootstrap PSNR 95% lower bound.

These profiles establish capacity accounting, not efficacy. Efficacy requires the preregistered Oracle, Predicted, joint, negative-control, paired-statistical, and external robustness gates.
