# Pre-Stage-B Reload Requirement

Status: **SATISFIED; STORAGE-LIFECYCLE PATCH LOADED AT STEP 44,000; HISTORICAL STEP-19,000 RUN REMAINS SUPERSEDED**.

The step-19,000 reload proved the Stage-B source corrections were loadable, but the associated Stage-A checkpoint is no longer scientifically eligible. The replacement orchestrator imported the corrected centered preprocessing at startup, passed 49 tests, and launched Stage-A from random initialization without `--resume`; only this replacement run may feed coordinate statistics or Stage-B.

Reason: Stage-B preflight found and fixed two negative-control defects after the active process started:

1. O9 formerly used `randperm(batch)`, which becomes identity for per-image locked validation (`batch=1`). It now uses deterministic cross-sample cyclic derangement in training and deterministic spatial displacement for batch-one validation.
2. O11 formerly consumed global RNG and changed between validation calls. It now uses a scale-local fixed generator.
3. The required fixed-random-versus-train-only-PCA direction ablation was added as O15/P15. PCA fits only valid train-label unit deviations, stores its centered 81-D mean and frozen row-orthonormal 6x81 basis, and never participates in GO.
4. Stage-A-to-Stage-B initialization now explicitly selects the locked-validation-best coarse checkpoint rather than the last epoch. The preregistered 10/10 capacity ownership branch was added with independent Stage-A training, coordinate statistics, and only P6/P7/P12 formal comparisons after the main Stage-B GO.
5. Formal Stage-B initialization now strictly imports only the actually trained `encoder.*` and `d1.*` tensors. Assessor/D2 remain freshly initialized from each registered seed, so same-seed arms are paired while the three repeat seeds are genuinely independent. Stage-C and baseline fine-tuning continue to use full strict checkpoint loading.

The formal source also now requires independently retrained O8/O9/O10/O11 controls for both Oracle and conditionally for Predicted ladders, with explicit sign-absolute, direction, and random-noise margins. O7/O8/O9/O10/O15 share the direction-cosine supervision; O9 applies the same corruption to its direction weight.

Evidence before future reload:

- `python -m py_compile` passed for trainer/orchestrator/tests.
- Current coordinate/training/pipeline targeted suite before the capacity addition: 22 passed; the post-addition training+pipeline suite is 18 passed and both 6/14 and 10/10 instantiated models have exactly 29,938,604 parameters.
- Stage-B has not started, so no existing scientific result is invalidated.

This marker is retained as provenance. The step-18,000 startup contract/full pytest proved the earlier corrections active, but it predates the final additions listed below and is not sufficient authorization for Stage-B.

After the successful step-18,000 reload, additional prompt-compliance audit added O13/O14 formal diagnostics, non-dehaze per-task guards, joint P6/P7/negative/residual controls, the frozen local-composite robustness gate, and SHA binding between coordinate statistics, locked split, and selected Stage-A checkpoint. These changes do not affect Stage-A but had to be loaded before Stage-B. The `srsc_reload` watcher repeated the same atomic procedure at step 19,000; the fresh full test and resumed-step evidence below satisfy that requirement.

Final completion evidence:

- The trainer atomically wrote step 19,000 before the watcher sent TERM; checkpoint SHA256 is `880fb13d7469be7c2866d889ab81349783a9b9f1f900c2e83b17bb183dd213b3`.
- CPU deserialization verified epoch 8, `batch_in_epoch=7168`, step 19,000, scheduler epoch 8, LR `1.1432857142857143e-4`, 341 Adam states, all four RNG states, and locked config/split provenance.
- The old trainer was terminated only after `SAFE_CHECKPOINT`; the kernel flock was released and `srsc_pipeline` relaunched at `2026-07-13T08:06:14Z` with orchestrator/trainer PIDs 31521/31614.
- The fresh parent rebuilt the strict AIO-3 manifest (`149,814` expected, zero missing), emitted a 31-resource startup contract, and ran the complete suite: `42 passed in 18.64s`.
- Exact-resume replay completed and the new process produced step 19,050 with loss `0.0197403021157` at the unchanged LR. GPU utilization returned to 100%, proving the reloaded trainer crossed the checkpoint rather than merely starting.
- No further Stage-B source change or orchestrator reload is pending. Stage-A continues uninterrupted; Stage-B remains gated on the locked-validation-best Stage-A checkpoint and the preregistered scientific tests.

## 2026-07-13 storage-lifecycle reload (completed at step 44,000)

After the scientific Stage-B implementation had been frozen, a disk-capacity
audit found that the eight conditional Stage-C arms used the correct top-3
selection but did not invoke the already-audited post-completion compactor.
`scripts/orchestrate.py` now checks `formal_complete(...)`, atomically retains
the locked-validation-selected model plus provenance, and deletes only
redundant optimizer/RNG snapshots after each Stage-C arm is complete.  This
does not alter the model, initialization, gradients, losses, training budget,
validation, selected checkpoint, or official-test lock.

The active Python orchestrator predated this operational-only source change.
To ensure the corrected lifecycle was loaded before Stage-B/Stage-C, tmux
session `srsc_storage_reload` waited for trainer checkpoint step44,000. That
point was deliberately after the epoch20/step43,020 locked validation, so the
reload did not interrupt or suppress the registered validation event. The
atomic checkpoint, child termination, kernel-lock release, complete startup
tests, and exact optimizer/scheduler/RNG resume are now all evidenced below.

Pre-registration verification: complete suite `70 passed, 2 warnings in
19.08s`; the warnings are the known isolated public-R2R scheduler-equivalence
test warnings and are not emitted by the actual training path.

Completion evidence for the storage-lifecycle reload:

- Epoch20 locked validation completed first at step43,020 and its exact
  checkpoint/integrity report were safely written before any termination.
- `SAFE_CHECKPOINT` recorded step44,000 at `2026-07-14T00:00:39Z`; only then
  was old trainer PID68125 sent TERM. The kernel `flock` released normally and
  `PIPELINE_RELOADED` was recorded at `2026-07-14T00:00:44Z`.
- The reloaded orchestrator rebuilt the strict AIO-3 manifest with149,814
  expected references and zero missing, then ran the complete current suite:
  `70 passed, 2 warnings in 19.16s`.
- New orchestrator/trainer PIDs139580/139617 launched with explicit
  `--resume .../last.pt`. Independent CPU integrity verification of that file
  reports epoch20, step44,000, batch-in-epoch3920, 303907623 bytes, SHA256
  `2e09f9fa6db756ed28e84ab73f25cac04067701699b5ab311bdbbdb1f59515a9`,
  with model, Adam, scheduler, four RNG families, config/split provenance and
  all learning-rate checks passing.
- The new trainer produced step44,050 at the unchanged LR
  `0.00019981033287370443`; GPU utilization returned to100%. This proves exact
  continuation rather than a fresh run. No pre-Stage-B reload is now pending.

Replacement center-crop startup evidence:

- Startup contract `contract_review_aio3_startup.json` covers 34 resources, including the data loader, official parity evaluator, and protocol-correction report.
- Strict AIO-3 rebuild reported 149,814 expected references and zero missing.
- Fresh complete suite: `49 passed in 19.15s`.
- Trainer command contains no `--resume`; the first record is epoch 0, step 50, LR `1e-7`, loss `0.2219290733`.
- GPU returned to 100% utilization with a measured 11.0943GiB Stage-A peak.

Operational action completed: tmux session `srsc_reload` waited for the trainer-written atomic checkpoint at optimizer step 18,000. It loaded the checkpoint on CPU before sending TERM, waited for the persistent lock file's kernel `flock` to become acquirable (file existence was not treated as ownership), and relaunched `srsc_pipeline`.

Completion evidence:

- `SAFE_CHECKPOINT` recorded step 18,000 at `2026-07-13T07:40:18Z`; old trainer PID 12851 was terminated only afterward.
- The kernel lock released and `srsc_pipeline` relaunched at `2026-07-13T07:40:24Z`; new orchestrator/trainer PIDs are 23841/23897.
- Fresh `contract_review_aio3_startup.json` was emitted at 07:40:31Z; strict data rebuild reported 149,814 expected entries and zero missing.
- Fresh full test log reports `38 passed in 18.25s` before the new trainer started.
- The step-18,000 checkpoint SHA256 is `ce10b01b36d03ec0a34d1bcbc7a0a2526a7c3640390913f7308fb7e4cfda8c94`; CPU load verified epoch8, batch3168, scheduler epoch8, LR `1.1432857142857143e-4`, 341 Adam states, all four RNG states, and locked config/split hashes.
- New code resumed to step 18,050 with loss `0.0254132039845`; GPU returned to 100% utilization. Stage-B will therefore execute the corrected process.
