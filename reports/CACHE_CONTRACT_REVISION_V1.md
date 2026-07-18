# Stage-A Output Cache Contract Revision V1

Status: **preregistered implementation correction before Stage-B**  
Scope: AIO-3 and AIO-5 SRSC-Lite experiments  
Scientific authority: this document narrows a storage instruction; it does not
change the selected Stage-A checkpoint, model, restoration objective, feedback
representation, Stage-B schedule, or any GO/NO-GO threshold.

## 1. Original instruction and required correction

The implementation prompt requested:

> Freeze the locked-validation-best Stage-A model and generate train/val/test
> `y1` caches with SHA256.

The phrase is well-defined for fixed locked validation, but a literal full
train/test materialization is either scientifically non-equivalent or violates
the existing evaluation lock:

- Training denoising is synthesized afresh on the full clean image, followed
  by a random paired crop and one of PromptIR/R2R augmentation modes 1--7.
  Consequently, a dataset index has a different observable input in each
  epoch. One cached tensor per index would repeat one degradation/crop for all
  Stage-B epochs and silently change the training distribution.
- Cropping a full-image `y1` is not equivalent to applying Encoder+D1 to the
  cropped input because padding, boundaries and the receptive field differ.
- D2 consumes the four Encoder feature maps in addition to `y1`. A `y1` cache
  alone cannot remove the frozen Encoder forward.
- Official-test access is governed by the frozen candidate manifest, protocol
  file lock and one-shot consumption ledger in `scripts/eval_locked.py`.
  Precomputing Stage-A outputs through a separate script would bypass that
  transaction and create avoidable test exposure.
- Stage-C is allowed to update Encoder+D1 at 0.1x the main learning rate.
  Therefore an earlier Stage-A test cache is not the final joint model's `y1`
  and cannot be substituted into final evaluation.

The formal equivalent correction is therefore:

1. materialize only fixed `locked_val` Stage-A outputs;
2. keep train Encoder+D1 frozen and online during Stage-B;
3. keep official test deferred to the existing frozen one-shot evaluator;
4. bind all three decisions to an auditable manifest.

## 2. Capacity calculation

The values below use the actual virtual dataset sizes and 128x128 training
patches. They exclude filesystem and serialization overhead.

| Protocol | Train observations/epoch | One `y1` epoch FP32 | One `y1` epoch BF16 | 30 epochs FP32 | 30 epochs BF16 | 30-epoch F1--F4 FP16 |
|---|---:|---:|---:|---:|---:|---:|
| AIO-3 | 137,669 | 25.21 GiB | 12.60 GiB | 756.24 GiB | 378.12 GiB | 11.08 TiB |
| AIO-5 | 213,779 | 39.14 GiB | 19.57 GiB | 1,174.32 GiB | 587.16 GiB | 17.20 TiB |

At the time of preregistration the data volume had about 155 GiB free. Even a
quantized full training cache would not fit, and quantization would no longer
be lossless or numerically identical to online feedback construction.

The fixed output sizes are tractable:

| Protocol | Locked observations | Lossless FP32 `y1` | Official observations | Stage-A FP32 `y1` (deferred, not built) |
|---|---:|---:|---:|---:|
| AIO-3 | 534 | about 1.276 GiB | 804 | about 1.939 GiB |
| AIO-5 | 396 | about 1.115 GiB | 1,794 | about 13.189 GiB |

These estimates use center-cropped pixel dimensions and three FP32 channels.
The AIO-5 locked set contains the five published-table settings only
(`denoise25`, derain, dehaze, deblur and low-light); the training distribution
still contains denoising sigmas 15/25/50.

## 3. Locked-validation cache contract

`scripts/cache_stage_a_outputs.py` is the only cache producer in this
revision. It has no split selector and imports only `build_locked_val`.
`official_test` is rejected by an explicit scope guard and by the output-path
guard.

A cache is valid only when all of the following match:

- protocol and `locked_val` scope;
- config path and SHA256;
- locked split path and SHA256;
- every source list path and its declared/current SHA256;
- selected Stage-A checkpoint path, SHA256, epoch, step and locked-validation
  selection evidence;
- dataset, model, feedback, trainer and cache-producer code SHA256 values;
- forward definition: pad to 8, selected frozen Encoder+D1, CUDA BF16
  autocast, no output clamp, lossless FP32 storage.

Each observation records:

- stable `(task, name)` identity and identity SHA256;
- exact synthesized/paired input tensor SHA256;
- exact GT tensor SHA256;
- output shape and FP32 tensor SHA256;
- relative `.npy` shard and file SHA256.

The cache aggregate, JSON manifest and manifest sidecar each have independent
SHA256 checks. Unexpected, missing, duplicated or path-traversing shards are
rejected.

### Transaction and replay rule

Creation occurs in a sibling staging directory:

1. first full pass writes lossless FP32 shards;
2. a newly constructed locked dataset is traversed again;
3. every input and GT hash is checked;
4. Encoder+D1 is rerun for every observation;
5. every new prediction must be exactly array-equal to its staged shard;
6. only then are the complete manifest and digest written;
7. the staged cache is checked once more on CPU and atomically renamed.

A failure removes the staging directory and leaves no complete cache.
Rerunning against an existing cache performs a full CPU contract, source-input,
GT and shard verification and returns idempotently without initializing CUDA.
An explicit flag can request another full online replay.

## 4. Train contract replacing tensor materialization

Formal Stage-B continues to run selected, frozen Encoder+D1 online under
`torch.no_grad()`. All feedback arms retain identical data, initialization,
schedule and frozen-coarse computation contracts. This preserves the dynamic
noise/crop/augmentation distribution and avoids a representation-specific
cache path.

The existing Stage-B deterministic run contract remains authoritative. A
future stateless sample-addressing enhancement may be preregistered separately,
but it is not smuggled into this cache correction and must not be introduced
after observing any Stage-B metric.

The optional train-only samples used to compute coordinate thresholds and
normalization remain online and are cryptographically bound to the selected
Stage-A checkpoint, locked split and statistics artifact. They are not called
a full train cache.

## 5. Official-test policy

This cache tool must never:

- import or call `build_test_sets`;
- accept `official_test` as a split;
- write an `official_test` cache directory;
- produce PSNR, SSIM or any other selection signal;
- unlock, reserve or modify an official-test ledger.

Official inputs remain untouched until the orchestrator freezes the final
candidate manifest and calls `scripts/eval_locked.py` with the explicit unlock
and manifest arguments. Final evaluation runs each frozen candidate online.
No Stage-A test cache is consumed by Stage-B, Stage-C or publication claims.

If a post-hoc Stage-A diagnostic cache is ever required, it must be integrated
inside the already-reserved one-shot official transaction and its SHA256 must
become part of that terminal record. That is outside Revision V1.

## 6. Interpretation boundary

This revision does not claim that caching improves restoration quality or
training speed. It supplies reproducibility evidence for the only fixed split
where a reusable Stage-A output is both well-defined and safe. Scientific
Stage-B conclusions remain based on online frozen Encoder+D1 computation, and
official-test conclusions remain based solely on the existing one-shot
evaluator.
