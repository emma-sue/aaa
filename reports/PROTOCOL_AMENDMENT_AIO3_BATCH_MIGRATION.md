# Protocol Amendment — AIO-3 Stage-A Batch Migration

Status: **FROZEN BEFORE STAGE-B; NO CLAIM OF AN UNCHANGED BATCH-64 OR FULL-BATCH-120 RUN**.

## What happened

The eligible AIO-3 Stage-A run began under the public single-GPU effective
batch 64 schedule and was migrated, after the completed epoch-55 locked
validation, to four-GPU DDP in order to reduce wall time.  The migration kept
the same model, Adam state, epoch-addressed learning-rate schedule, dataset,
augmentation distribution, checkpoint, and RNG state.  Its registered runtime
from epoch 56 onward is `4 GPUs × 30 samples/GPU × accumulation 1 = 120`.

The exact optimization budget is therefore:

| Segment | Completed epochs | Effective batch | Optimizer steps/epoch | Samples/epoch |
|---|---:|---:|---:|---:|
| Original segment | 1–55 | 64 | 2,151 | 137,664 |
| DDP segment | 56–240 | 120 | 1,147 | 137,640 |
| Total | 240 | hybrid | **330,500** | **33,034,920** |

These figures are part of the checkpoint-integrity contract.  The Stage-A
handoff must verify epoch 240, step 330,500, world size 4, global effective
batch 120 in the final runtime metadata, and a completed final locked
validation transaction.

## Scientific scope

- Every Oracle, Predicted, no-state, residual-code, and negative-control arm
  initializes from the same selected Stage-A checkpoint.  Their Stage-B
  causal comparisons therefore remain paired and do not confound the batch
  migration with feedback representation.
- The run must not be described as an exact reproduction of either an
  all-240-epoch batch-64 run or an all-240-epoch batch-120 run.
- A clean Restormer baseline trained for 240 epochs entirely at batch 120 has
  nearly the same sample exposure (33,033,600) but only 275,280 optimizer
  updates.  It is a schedule-sensitivity baseline, not the sole strict
  publication control.
- The primary clean and parameter-matched internal baselines must reproduce
  the same 55-epoch batch-64 plus 185-epoch batch-120 schedule with one
  continuous optimizer and epoch-wise LR trajectory.  They must finish at
  step 330,500 and record the same per-epoch raw-sample exposure budget.

## Four-GPU exact-update baseline execution

The publication controls execute the frozen global updates on four GPUs using
gradient accumulation, because a rank-local batch of 30 is safe for the
shallower live Stage-A model but is not a safe assumption for the full clean
and parameter-matched Restormer controls:

| Epochs (zero-based) | Per-rank micro-batch | Accumulation | World size | Global effective batch |
|---|---:|---:|---:|---:|
| 0–54 | 8 | 2 | 4 | 64 |
| 55–239 | 10 | 3 | 4 | 120 |

All non-final micro-batches of an optimizer update use `DDP.no_sync()`; the
loss is divided by the fixed accumulation count, gradients synchronize on the
final micro-batch, clipping is applied once, and Adam steps once.  The
authoritative checkpoint cursor is therefore the completed optimizer-update
index, never a half-finished micro-batch.  Both clean and parameter-matched
arms use the same canonical raw-index matrix and must finish with identical
240-epoch update-digest records.

This exactly matches raw-sample membership per optimizer update, total
optimizer steps, total sample exposure, objective, continuous Adam state and
the epoch-addressed learning-rate schedule.  It does not claim bitwise equality
to the historical live Stage-A stochastic trajectory: independently seeded
DDP workers and floating-point reduction order can change crop, noise,
augmentation and arithmetic details.  That limitation is symmetric across
the two baseline arms and is part of their immutable run contracts.

The exact-hybrid baselines can support a fair internal publication comparison.
The external R2R table remains a separately reported reference because model,
training implementation, and stochastic trajectory are not identical.

## What is not recoverable

The historical four-rank workers did not persist every crop, Gaussian-noise,
and augmentation draw.  A later baseline can match manifest, epoch count,
sample exposure, optimizer-update count, effective-batch schedule, LR
trajectory, and validation protocol, but cannot honestly claim a bitwise
identical stochastic trajectory.  This limitation is permanent and must be
reported rather than reconstructed after seeing results.
