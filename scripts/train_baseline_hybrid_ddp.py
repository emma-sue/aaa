#!/usr/bin/env python3
"""Four-GPU exact-update AIO-3 hybrid baseline trainer.

This entry point trains one clean baseline arm per invocation.  Run the
``baseline`` invocation to completion and then run ``baseline_matched`` to
obtain the requested sequential controls.  It never imports or mutates the
live SRSC Stage-A process.

The canonical flat raw-index schedule comes from ``train_baseline_hybrid``:

* epochs 0..54: 2,151 updates x 64 raw samples;
* epochs 55..239: 1,147 updates x 120 raw samples.

For DDP, every canonical update is reshaped and split into four contiguous
rank-local effective slices (16 or 30 samples).  Those slices are executed by
the fixed safe profiles ``8 x 2`` and ``10 x 3`` (micro-batch x gradient
accumulation).  All non-final micros run inside ``DDP.no_sync``; the loss is
divided by the accumulation factor and Adam steps only after a complete
canonical update.  Thus every optimizer update consumes exactly the
registered global raw-sample set while Adam and the epoch scheduler remain
continuous across epoch 55.

Checkpoints are committed only after complete optimizer updates.  The
authoritative resume cursor is ``update_in_epoch``; the derived micro cursor
is stored and checked only as evidence.  Resume enumerates skipped micros to
replay worker RNG boundaries before continuing at the next full update.

The claim is deliberately about raw dataset identities and update budgets.
Independent DDP workers do not reproduce the historical single-process crop,
noise and augmentation RNG stream bit-for-bit; that limitation is frozen in
the run contract and applies equally to both baseline arms.

Sequential invocation (only after the four GPUs are free) is intentionally
external and crash-resumable::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 \
      scripts/train_baseline_hybrid_ddp.py \
      --config configs/protocol_aio3_baseline_hybrid.yaml --stage baseline \
      --run-name aio3_baseline_hybrid_ddp_seed1415926 --fresh
    CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 \
      scripts/train_baseline_hybrid_ddp.py \
      --config configs/protocol_aio3_baseline_hybrid.yaml --stage baseline_matched \
      --run-name aio3_baseline_matched_hybrid_ddp_seed1415926 --fresh
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_baseline_hybrid as reference  # noqa: E402
from scripts.runtime_accounting import (  # noqa: E402
    atomic_write_runtime_sidecar,
    read_runtime_sidecar,
    runtime_snapshot,
    start_runtime_accounting,
)
from scripts.train import (  # noqa: E402
    acquire_run_lock,
    atomic_write_locked_rows,
    build_model,
    configure_trainable,
    optimizer_groups,
    r2r_pretrain_epoch_ratio,
    reconcile_training_csv,
    restoration_l1,
    sha256_file,
    upsert_validation_record,
    validate_locked,
)
from src.data.aio_dataset import AIOTrainDataset, build_locked_val  # noqa: E402


WORLD_SIZE = 4
TRAINING_ORIGIN = "fresh_hybrid_baseline_ddp_exact_update_safe_micro_v1"
DIGEST_SCHEMA = 2
PARTITION_ALGORITHM = "canonical_update_contiguous_rank_slices_v1"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PROCESS_GROUP_TIMEOUT_SECONDS = 2 * 60 * 60
DATA_MANIFEST_PATH = ROOT / "artifacts/manifests/aio3.json"


@dataclass(frozen=True)
class SafeMicroProfile:
    start_epoch: int
    end_epoch: int
    micro_batch: int
    accumulation: int
    local_effective_batch: int
    global_effective_batch: int


SAFE_MICRO_PROFILES = (
    SafeMicroProfile(0, 55, 8, 2, 16, 64),
    SafeMicroProfile(55, 240, 10, 3, 30, 120),
)


def micro_profile_for_epoch(epoch: int) -> SafeMicroProfile:
    for profile in SAFE_MICRO_PROFILES:
        if profile.start_epoch <= epoch < profile.end_epoch:
            phase = reference.phase_for_epoch(epoch)
            if (
                profile.micro_batch * profile.accumulation
                != profile.local_effective_batch
                or profile.local_effective_batch * WORLD_SIZE
                != profile.global_effective_batch
                or profile.global_effective_batch != phase.effective_batch
            ):
                raise RuntimeError("safe microbatch profile no longer matches canonical phase")
            return profile
    raise ValueError(f"epoch {epoch} is outside the fixed safe microbatch schedule")


def micro_profile_payload(profile: SafeMicroProfile) -> dict:
    """Return the immutable execution contract for one safe micro profile."""
    return {
        "start_epoch": int(profile.start_epoch),
        "end_epoch": int(profile.end_epoch),
        "micro_batch": int(profile.micro_batch),
        "accumulation": int(profile.accumulation),
        "local_effective_batch": int(profile.local_effective_batch),
        "global_effective_batch": int(profile.global_effective_batch),
        "sync_policy": "DDP.no_sync on all non-final microbatches",
    }


class FixedRankIndexSampler(Sampler[int]):
    """Yield a prepartitioned rank-local epoch sequence without reshuffling."""

    def __init__(self, indices: torch.Tensor):
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise TypeError("rank indices must be a one-dimensional int64 tensor")
        self.indices = indices.contiguous()

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.indices)

    def __len__(self) -> int:
        return int(self.indices.numel())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=reference.ALLOWED_STAGES)
    parser.add_argument("--run-name", required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--fresh", action="store_true")
    start.add_argument("--resume")
    parser.add_argument("--workers-per-rank", type=int, default=8)
    return parser.parse_args(argv)


def canonical_epoch_update_matrix(
    dataset_size: int, seed: int, epoch: int,
) -> torch.Tensor:
    """Return the authoritative ``[update, global-sample]`` raw-index matrix."""
    phase = reference.phase_for_epoch(epoch)
    flat = reference.epoch_indices(dataset_size, seed, epoch)
    expected = phase.steps_per_epoch * phase.effective_batch
    if flat.numel() != expected:
        raise RuntimeError(
            f"canonical flat schedule size drift at epoch {epoch}: "
            f"{flat.numel()} != {expected}"
        )
    matrix = flat.view(phase.steps_per_epoch, phase.effective_batch).contiguous()
    if matrix.unique().numel() != matrix.numel():
        raise RuntimeError(f"canonical epoch {epoch} repeats a raw sample identity")
    return matrix


def rank_epoch_matrix(
    dataset_size: int, seed: int, epoch: int, rank: int,
) -> torch.Tensor:
    if not 0 <= rank < WORLD_SIZE:
        raise ValueError(f"rank must be in 0..{WORLD_SIZE - 1}")
    phase = reference.phase_for_epoch(epoch)
    if phase.effective_batch % WORLD_SIZE:
        raise RuntimeError("global effective batch is not divisible by four ranks")
    local_batch = phase.effective_batch // WORLD_SIZE
    canonical = canonical_epoch_update_matrix(dataset_size, seed, epoch)
    return canonical[
        :, rank * local_batch : (rank + 1) * local_batch
    ].contiguous()


def rank_epoch_indices(
    dataset_size: int, seed: int, epoch: int, rank: int,
) -> torch.Tensor:
    return rank_epoch_matrix(dataset_size, seed, epoch, rank).reshape(-1).contiguous()


def reassemble_rank_matrices(rank_matrices: list[torch.Tensor]) -> torch.Tensor:
    if len(rank_matrices) != WORLD_SIZE:
        raise ValueError("exactly four rank matrices are required")
    if not rank_matrices or any(matrix.ndim != 2 for matrix in rank_matrices):
        raise ValueError("rank matrices must be two-dimensional")
    shape = rank_matrices[0].shape
    if any(matrix.shape != shape for matrix in rank_matrices):
        raise ValueError("rank matrix shapes differ")
    return torch.cat(rank_matrices, dim=1).contiguous()


def int64_tensor_bytes(indices: torch.Tensor) -> bytes:
    return (
        indices.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
        .tobytes(order="C")
    )


def update_digest_root(update_matrix: torch.Tensor) -> str:
    """Hash every ordered update independently, then hash the ordered leaves."""
    if update_matrix.dtype != torch.int64 or update_matrix.ndim != 2:
        raise TypeError("update digest requires a two-dimensional int64 matrix")
    root = hashlib.sha256()
    for update in update_matrix:
        leaf = hashlib.sha256(int64_tensor_bytes(update)).digest()
        root.update(leaf)
    return root.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def epoch_digest_record(dataset_size: int, seed: int, epoch: int) -> dict:
    phase = reference.phase_for_epoch(epoch)
    profile = micro_profile_for_epoch(epoch)
    canonical = canonical_epoch_update_matrix(dataset_size, seed, epoch)
    local_batch = profile.local_effective_batch
    ranks = [
        canonical[:, rank * local_batch : (rank + 1) * local_batch].contiguous()
        for rank in range(WORLD_SIZE)
    ]
    if not torch.equal(reassemble_rank_matrices(ranks), canonical):
        raise RuntimeError(f"rank partition does not reconstruct epoch {epoch}")
    record = {
        "schema": DIGEST_SCHEMA,
        "epoch": int(epoch),
        "steps": int(phase.steps_per_epoch),
        "global_batch": int(profile.global_effective_batch),
        "local_effective_batch": int(profile.local_effective_batch),
        "micro_batch": int(profile.micro_batch),
        "accumulation": int(profile.accumulation),
        "flat_indices_sha256": hashlib.sha256(
            int64_tensor_bytes(canonical.reshape(-1))
        ).hexdigest(),
        "per_update_digest_root": update_digest_root(canonical),
        "rank_indices_sha256": [
            hashlib.sha256(int64_tensor_bytes(matrix.reshape(-1))).hexdigest()
            for matrix in ranks
        ],
        "partition_algorithm": PARTITION_ALGORITHM,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def schedule_digest(records: Mapping[str, dict]) -> str:
    ordered = [records[str(epoch)] for epoch in sorted(map(int, records))]
    return canonical_json_sha256({
        "schema": DIGEST_SCHEMA,
        "partition_algorithm": PARTITION_ALGORITHM,
        "epochs": ordered,
    })


def expected_digest_keys(epoch: int, update_in_epoch: int) -> list[str]:
    stop = epoch + (1 if update_in_epoch else 0)
    return [str(value) for value in range(stop)]


def verify_epoch_digest_records(
    records: Mapping[str, dict],
    *,
    seed: int,
    epoch: int,
    update_in_epoch: int,
    dataset_size: int = reference.EXPECTED_DATASET_SIZE,
) -> None:
    expected_keys = expected_digest_keys(epoch, update_in_epoch)
    if sorted(records, key=int) != expected_keys:
        raise RuntimeError(
            "epoch digest coverage mismatch: "
            f"actual={sorted(records, key=int)} expected={expected_keys}"
        )
    for key in expected_keys:
        expected = epoch_digest_record(dataset_size, seed, int(key))
        if records[key] != expected:
            raise RuntimeError(f"epoch/update digest mismatch at epoch {key}")
    if schedule_digest(records) != canonical_json_sha256({
        "schema": DIGEST_SCHEMA,
        "partition_algorithm": PARTITION_ALGORITHM,
        "epochs": [records[key] for key in expected_keys],
    }):
        raise RuntimeError("schedule digest construction is inconsistent")


def expected_progress(epoch: int, update_in_epoch: int) -> tuple[int, int]:
    steps, samples = reference.budget_before_epoch(epoch)
    if update_in_epoch == 0:
        return steps, samples
    if epoch >= reference.EXPECTED_PHASES[-1].end_epoch:
        raise ValueError("final epoch cannot have a nonzero update cursor")
    phase = reference.phase_for_epoch(epoch)
    if not 0 <= update_in_epoch <= phase.steps_per_epoch:
        raise ValueError("update cursor is outside the active epoch")
    return (
        steps + update_in_epoch,
        samples + update_in_epoch * phase.effective_batch,
    )


def expected_microbatch_cursor(
    epoch: int,
    update_in_epoch: int,
    total_epochs: int = reference.EXPECTED_PHASES[-1].end_epoch,
) -> int:
    """Derive the evidence-only micro cursor from the authoritative update cursor."""
    if epoch == total_epochs:
        if update_in_epoch != 0:
            raise ValueError("final epoch cannot have a nonzero update cursor")
        return 0
    if not 0 <= epoch < total_epochs:
        raise ValueError("epoch is outside the fixed safe microbatch schedule")
    phase = reference.phase_for_epoch(epoch)
    if not 0 <= update_in_epoch <= phase.steps_per_epoch:
        raise ValueError("update cursor is outside the active epoch")
    return update_in_epoch * micro_profile_for_epoch(epoch).accumulation


def stable_args(args: argparse.Namespace, workers_per_rank: int) -> dict:
    return {
        "config": str(Path(args.config).resolve()),
        "stage": args.stage,
        "run_name": args.run_name,
        "workers_per_rank": int(workers_per_rank),
        "world_size": WORLD_SIZE,
    }


def validate_execution_environment() -> None:
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "exact-update DDP hybrid baseline requires "
            f"CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG!r}; "
            f"actual={actual!r}. Set it before starting Python/CUDA."
        )


def environment_contract_payload() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "high",
        "process_group_backend": "nccl",
        "process_group_timeout_seconds": PROCESS_GROUP_TIMEOUT_SECONDS,
    }


def data_contract_payload(cfg: Mapping[str, object]) -> dict:
    """Validate and bind the concrete AIO-3 list/data manifests.

    The split manifest contains the authoritative hashes of the three source
    lists.  The materialization manifest independently proves that every
    registered entry existed when the data protocol was frozen.  Full image
    bytes are not re-hashed here; that claim boundary is explicit in the
    returned payload.
    """
    split_path = Path(str(cfg["split_manifest"]))
    split = json.loads(split_path.read_text())
    if split.get("protocol") != "aio3":
        raise RuntimeError("hybrid baseline split manifest is not AIO-3")
    list_hashes = split.get("list_sha256")
    if not isinstance(list_hashes, dict) or not list_hashes:
        raise RuntimeError("AIO-3 split manifest lacks source-list hashes")
    list_root = Path(str(cfg["list_root"]))
    for relative, expected in sorted(list_hashes.items()):
        path = list_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"AIO-3 source-list hash drift: {path}")

    if not DATA_MANIFEST_PATH.is_file():
        raise FileNotFoundError(DATA_MANIFEST_PATH)
    materialized = json.loads(DATA_MANIFEST_PATH.read_text())
    if (
        materialized.get("protocol") != "aio3"
        or int(materialized.get("missing_entries", -1)) != 0
        or Path(str(materialized.get("data_root", ""))).resolve()
        != Path(str(cfg["data_root"])).resolve()
        or materialized.get("list_sha256") != list_hashes
    ):
        raise RuntimeError("AIO-3 materialization manifest no longer matches the split/data contract")
    return {
        "split_manifest_path": str(split_path.resolve()),
        "split_manifest_sha256": sha256_file(split_path),
        "materialization_manifest_path": str(DATA_MANIFEST_PATH.resolve()),
        "materialization_manifest_sha256": sha256_file(DATA_MANIFEST_PATH),
        "list_root": str(list_root.resolve()),
        "list_sha256": dict(sorted(list_hashes.items())),
        "data_root": str(Path(str(cfg["data_root"])).resolve()),
        "expected_entries": int(materialized["expected_entries"]),
        "missing_entries": 0,
        "content_hash_claim": (
            "source-list bytes and materialization-manifest bytes are bound; "
            "individual image-file bytes are not re-hashed by this trainer"
        ),
    }


def run_contract_payload(
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
) -> dict:
    code_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/train_baseline_hybrid.py",
        ROOT / "scripts/train.py",
        ROOT / "scripts/runtime_accounting.py",
        ROOT / "src/net/clean_restormer_aio.py",
        ROOT / "src/net/srsc_lite.py",
        ROOT / "src/net/restormer_blocks.py",
        ROOT / "src/data/aio_dataset.py",
    )
    return {
        "schema": 2,
        "purpose": "aio3_four_gpu_exact_raw_update_hybrid_baseline",
        "args": stable_args(args, workers_per_rank),
        "config_sha256": sha256_file(args.config),
        "split_manifest_sha256": sha256_file(cfg["split_manifest"]),
        "reference_schedule": reference.expected_schedule_payload(),
        "reference_schedule_sha256": reference.schedule_sha256(cfg["seed"]),
        "partition_algorithm": PARTITION_ALGORITHM,
        "world_size": WORLD_SIZE,
        "checkpoint_cursor_authority": "optimizer_update",
        "safe_micro_profiles": [
            micro_profile_payload(profile) for profile in SAFE_MICRO_PROFILES
        ],
        "runtime_phases": [
            {
                **micro_profile_payload(
                    micro_profile_for_epoch(phase.start_epoch)
                ),
                "steps_per_epoch": phase.steps_per_epoch,
                "samples_per_epoch": phase.samples_per_epoch,
            }
            for phase in reference.EXPECTED_PHASES
        ],
        "training_origin": TRAINING_ORIGIN,
        "environment": environment_contract_payload(),
        "data_contract": data_contract_payload(cfg),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
        },
        "equivalence_claim": (
            "each optimizer update has the exact canonical raw-index set; total "
            "steps/samples, continuous Adam, clipping, objective and epoch LR are matched; "
            "safe microbatch accumulation changes execution only, not update membership"
        ),
        "stochastic_claim_boundary": (
            "DDP worker noise/crop/augmentation draws and floating reduction order "
            "are not claimed bitwise-identical to the historical single-process phase"
        ),
    }


def ensure_run_contract_rank0(
    run_dir: Path,
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
) -> str:
    contract = run_contract_payload(cfg, args, workers_per_rank)
    path = run_dir / "run_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("immutable DDP hybrid run contract mismatch")
    else:
        if any(run_dir.glob("*.pt")):
            raise RuntimeError("DDP hybrid checkpoint exists without run contract")
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    return sha256_file(path)


def rank_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _hash_update(digest: "hashlib._Hash", token: str, value: object) -> None:
    digest.update(token.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps({
            "dtype": str(tensor.dtype), "shape": list(tensor.shape)
        }, sort_keys=True, separators=(",", ":")).encode()
        digest.update(header)
        digest.update(b"\0")
        digest.update(memoryview(tensor.numpy()).cast("B"))
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, f"{token}/{key}", value[key])
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _hash_update(digest, f"{token}/{index}", item)
    else:
        digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    digest.update(b"\xff")


def full_state_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, "root", value)
    return digest.hexdigest()


def _probe_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            samples = []
            total = 0.0
            absolute = 0.0
        else:
            locations = sorted({0, flat.numel() // 2, flat.numel() - 1})
            samples = [float(flat[index].float().item()) for index in locations]
            total = float(flat.double().sum().item())
            absolute = float(flat.double().abs().sum().item())
        return {
            "kind": "tensor_probe",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "samples": samples,
            "sum": total,
            "abs_sum": absolute,
        }
    if isinstance(value, Mapping):
        return {
            str(key): _probe_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_probe_value(item) for item in value]
    return value


def probe_state_sha256(value: object) -> str:
    """Cheap all-rank divergence probe between full validation hashes.

    Tensor contents are represented by deterministic samples and reductions;
    all non-tensor optimizer/scheduler metadata is retained exactly.
    """
    return canonical_json_sha256(_probe_value(value))


def collect_ddp_integrity(
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    full: bool,
) -> dict:
    model_state = raw_model.state_dict()
    optimizer_state = optimizer.state_dict()
    local = {
        "level": "full_sha256" if full else "probe_sha256",
        "model": (
            full_state_sha256(model_state) if full else probe_state_sha256(model_state)
        ),
        "optimizer": (
            full_state_sha256(optimizer_state)
            if full else probe_state_sha256(optimizer_state)
        ),
        "scheduler": full_state_sha256(scheduler.state_dict()),
    }
    gathered: list[dict | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    if any(record != gathered[0] for record in gathered):
        raise RuntimeError(f"DDP model/optimizer/scheduler divergence: {gathered}")
    return {
        **local,
        "world_size": dist.get_world_size(),
        "all_ranks_identical": True,
    }


def validate_serialized_integrity(payload: Mapping[str, object]) -> None:
    """Bind a passing cross-rank record to the tensors actually serialized."""
    integrity = payload.get("ddp_integrity")
    if not isinstance(integrity, Mapping):
        raise RuntimeError("resume checkpoint lacks DDP integrity metadata")
    if integrity.get("all_ranks_identical") is not True:
        raise RuntimeError("resume checkpoint lacks passing DDP integrity")
    if int(integrity.get("world_size", -1)) != WORLD_SIZE:
        raise RuntimeError("resume checkpoint DDP integrity world-size mismatch")
    level = integrity.get("level")
    if level == "full_sha256":
        model_hash = full_state_sha256(payload.get("model"))
        optimizer_hash = full_state_sha256(payload.get("optimizer"))
    elif level == "probe_sha256":
        model_hash = probe_state_sha256(payload.get("model"))
        optimizer_hash = probe_state_sha256(payload.get("optimizer"))
    else:
        raise RuntimeError(f"unsupported DDP integrity level: {level!r}")
    scheduler_hash = full_state_sha256(payload.get("scheduler"))
    if model_hash != integrity.get("model"):
        raise RuntimeError("serialized model no longer matches DDP integrity")
    if optimizer_hash != integrity.get("optimizer"):
        raise RuntimeError("serialized Adam state no longer matches DDP integrity")
    if scheduler_hash != integrity.get("scheduler"):
        raise RuntimeError("serialized scheduler no longer matches DDP integrity")


def atomic_checkpoint_ddp(
    path: Path,
    *,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    update_in_epoch: int,
    step: int,
    samples_seen: int,
    epoch_digests: dict[str, dict],
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
    run_contract_sha256: str,
    rank: int,
    validation_pending: str | None = None,
    integrity: dict | None = None,
) -> None:
    expected_step, expected_samples = expected_progress(epoch, update_in_epoch)
    expected_microbatch = expected_microbatch_cursor(
        epoch, update_in_epoch, int(cfg["epochs"])
    )
    if (step, samples_seen) != (expected_step, expected_samples):
        raise RuntimeError(
            "DDP hybrid checkpoint budget drift: "
            f"actual={(step, samples_seen)} expected={(expected_step, expected_samples)}"
        )
    expected_keys = expected_digest_keys(epoch, update_in_epoch)
    if sorted(epoch_digests, key=int) != expected_keys:
        raise RuntimeError("checkpoint epoch digest coverage drift")
    if integrity is None:
        integrity = collect_ddp_integrity(
            raw_model, optimizer, scheduler, full=False
        )
    local_rng = rank_rng()
    rng_by_rank: list[dict | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rng_by_rank, local_rng)
    if rank == 0:
        accounting = runtime_snapshot()
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "update_in_epoch": int(update_in_epoch),
            "microbatch_in_epoch": int(expected_microbatch),
            # Compatibility alias now means the derived micro-batch cursor.
            "batch_in_epoch": int(expected_microbatch),
            "cursor_authority": "optimizer_update",
            "checkpoint_boundary": "after_complete_optimizer_update",
            "step": int(step),
            "samples_seen": int(samples_seen),
            "validation_pending": validation_pending,
            "validation_transaction_schema": 1,
            "training_origin": TRAINING_ORIGIN,
            "config": cfg,
            "config_sha256": sha256_file(args.config),
            "split_manifest_sha256": sha256_file(cfg["split_manifest"]),
            "args": stable_args(args, workers_per_rank),
            "reference_schedule": reference.expected_schedule_payload(),
            "reference_schedule_sha256": reference.schedule_sha256(cfg["seed"]),
            "partition_algorithm": PARTITION_ALGORITHM,
            "run_contract_sha256": run_contract_sha256,
            "epoch_update_digests": dict(epoch_digests),
            "schedule_digest": schedule_digest(epoch_digests),
            "active_phase": (
                micro_profile_payload(micro_profile_for_epoch(epoch))
                if epoch < int(cfg["epochs"]) else None
            ),
            "distributed_runtime": {
                "world_size": WORLD_SIZE,
                "workers_per_rank": int(workers_per_rank),
                "backend": dist.get_backend(),
            },
            "ddp_integrity": integrity,
            "runtime_accounting": accounting,
            "rng_by_rank": rng_by_rank,
        }
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        atomic_write_runtime_sidecar(
            path.parent / "runtime_accounting.json", accounting
        )
    dist.barrier()


def validate_resume_payload(
    payload: dict,
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
    run_contract_sha256: str,
    *,
    verify_digests: bool = True,
) -> None:
    if payload.get("training_origin") != TRAINING_ORIGIN:
        raise RuntimeError("resume checkpoint is not an exact-update DDP hybrid baseline")
    if payload.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("resume config hash mismatch")
    if payload.get("config") != cfg:
        raise RuntimeError("resume effective config mismatch")
    if payload.get("split_manifest_sha256") != sha256_file(cfg["split_manifest"]):
        raise RuntimeError("resume split-manifest mismatch")
    if payload.get("args") != stable_args(args, workers_per_rank):
        raise RuntimeError("resume DDP hybrid argument mismatch")
    if payload.get("reference_schedule") != reference.expected_schedule_payload():
        raise RuntimeError("resume reference schedule mismatch")
    if payload.get("reference_schedule_sha256") != reference.schedule_sha256(cfg["seed"]):
        raise RuntimeError("resume reference schedule hash mismatch")
    if payload.get("partition_algorithm") != PARTITION_ALGORITHM:
        raise RuntimeError("resume rank partition algorithm mismatch")
    runtime = payload.get("distributed_runtime", {})
    if (
        int(runtime.get("world_size", -1)) != WORLD_SIZE
        or int(runtime.get("workers_per_rank", -1)) != int(workers_per_rank)
        or runtime.get("backend") != "nccl"
    ):
        raise RuntimeError("resume distributed runtime mismatch")
    if payload.get("run_contract_sha256") != run_contract_sha256:
        raise RuntimeError("resume run-contract hash mismatch")
    if int(payload.get("validation_transaction_schema", -1)) != 1:
        raise RuntimeError("resume validation transaction schema mismatch")
    epoch = int(payload.get("epoch", -1))
    update_in_epoch = int(payload.get("update_in_epoch", -1))
    pending = payload.get("validation_pending")
    if pending not in {None, "epoch"}:
        raise RuntimeError("resume validation_pending value is invalid")
    if pending == "epoch" and update_in_epoch != 0:
        raise RuntimeError("pending validation is not at an epoch boundary")
    expected_step, expected_samples = expected_progress(epoch, update_in_epoch)
    expected_microbatch = expected_microbatch_cursor(
        epoch, update_in_epoch, int(cfg["epochs"])
    )
    if payload.get("cursor_authority") != "optimizer_update":
        raise RuntimeError("resume cursor authority is not optimizer_update")
    if payload.get("checkpoint_boundary") != "after_complete_optimizer_update":
        raise RuntimeError("resume checkpoint is not at a complete optimizer update")
    if int(payload.get("microbatch_in_epoch", -2)) != expected_microbatch:
        raise RuntimeError("resume derived microbatch cursor mismatch")
    if int(payload.get("batch_in_epoch", -2)) != expected_microbatch:
        raise RuntimeError("resume compatibility microbatch cursor mismatch")
    expected_active_phase = (
        micro_profile_payload(micro_profile_for_epoch(epoch))
        if epoch < int(cfg["epochs"]) else None
    )
    if payload.get("active_phase") != expected_active_phase:
        raise RuntimeError("resume safe microbatch profile mismatch")
    if int(payload.get("step", -1)) != expected_step:
        raise RuntimeError("resume optimizer-step budget mismatch")
    if int(payload.get("samples_seen", -1)) != expected_samples:
        raise RuntimeError("resume raw-sample budget mismatch")
    if int(payload.get("scheduler", {}).get("last_epoch", -1)) != epoch:
        raise RuntimeError("resume epoch scheduler position mismatch")
    rng_by_rank = payload.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != WORLD_SIZE:
        raise RuntimeError("resume rank RNG width mismatch")
    validate_serialized_integrity(payload)
    records = payload.get("epoch_update_digests")
    if not isinstance(records, dict):
        raise RuntimeError("resume checkpoint lacks epoch/update digests")
    if payload.get("schedule_digest") != schedule_digest(records):
        raise RuntimeError("resume schedule aggregate digest mismatch")
    if verify_digests:
        verify_epoch_digest_records(
            records,
            seed=int(cfg["seed"]),
            epoch=epoch,
            update_in_epoch=update_in_epoch,
        )


def update_top3_ddp(
    *,
    run_dir: Path,
    score: float,
    epoch: int,
    step: int,
    samples_seen: int,
    epoch_digests: dict[str, dict],
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
    run_contract_sha256: str,
    rank: int,
    integrity: dict,
) -> None:
    checkpoint = run_dir / f"val_epoch{epoch:03d}_step{step:07d}.pt"
    atomic_checkpoint_ddp(
        checkpoint,
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        update_in_epoch=0,
        step=step,
        samples_seen=samples_seen,
        epoch_digests=epoch_digests,
        cfg=cfg,
        args=args,
        workers_per_rank=workers_per_rank,
        run_contract_sha256=run_contract_sha256,
        rank=rank,
        integrity=integrity,
    )
    if rank == 0:
        index_path = run_dir / "top3.json"
        records = json.loads(index_path.read_text()) if index_path.is_file() else []
        records = [
            row for row in records
            if (int(row["epoch"]), int(row["step"])) != (epoch, step)
        ]
        records.append({
            "score": float(score),
            "epoch": int(epoch),
            "step": int(step),
            "checkpoint": checkpoint.name,
        })
        records.sort(key=lambda row: row["score"], reverse=True)
        retained, stale = records[:3], records[3:]
        temporary = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(retained, indent=2) + "\n")
        os.replace(temporary, index_path)
        retained_names = {row["checkpoint"] for row in retained}
        for row in stale:
            if row["checkpoint"] not in retained_names:
                (run_dir / row["checkpoint"]).unlink(missing_ok=True)
    dist.barrier()


def commit_pending_validation_ddp(
    *,
    raw_model: nn.Module,
    locked_val,
    run_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
    run_contract_sha256: str,
    epoch: int,
    step: int,
    samples_seen: int,
    epoch_digests: dict[str, dict],
    rank: int,
    device: torch.device,
) -> None:
    score = torch.zeros((), dtype=torch.float64, device=device)
    if rank == 0:
        if locked_val is None:
            raise RuntimeError("rank zero cannot validate without locked_val")
        summary, rows = validate_locked(
            raw_model,
            locked_val,
            args.stage,
            None,
            "O0",
            None,
            protocol="aio3",
            return_rows=True,
        )
        summary.update({"epoch": int(epoch), "step": int(step)})
        paired_path = (
            ROOT / "artifacts/metrics/locked_rows" / args.run_name
            / f"epoch{epoch:03d}_step{step:07d}.csv"
        )
        summary["paired_rows_path"] = str(paired_path.resolve())
        summary["paired_rows_sha256"] = atomic_write_locked_rows(
            paired_path, rows
        )
        metrics = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        upsert_validation_record(metrics, summary)
        print("LOCKED_VAL " + json.dumps(summary), flush=True)
        score.fill_(float(summary["macro_psnr"]))
    dist.broadcast(score, src=0)
    integrity = collect_ddp_integrity(
        raw_model, optimizer, scheduler, full=True
    )
    update_top3_ddp(
        run_dir=run_dir,
        score=float(score.item()),
        epoch=epoch,
        step=step,
        samples_seen=samples_seen,
        epoch_digests=epoch_digests,
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        args=args,
        workers_per_rank=workers_per_rank,
        run_contract_sha256=run_contract_sha256,
        rank=rank,
        integrity=integrity,
    )
    atomic_checkpoint_ddp(
        run_dir / "last.pt",
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        update_in_epoch=0,
        step=step,
        samples_seen=samples_seen,
        epoch_digests=epoch_digests,
        cfg=cfg,
        args=args,
        workers_per_rank=workers_per_rank,
        run_contract_sha256=run_contract_sha256,
        rank=rank,
        validation_pending=None,
        integrity=integrity,
    )


def build_rank_epoch_loader(
    dataset,
    *,
    cfg: dict,
    epoch: int,
    rank: int,
    workers_per_rank: int,
) -> tuple[DataLoader, reference.HybridPhase, SafeMicroProfile, dict]:
    phase = reference.phase_for_epoch(epoch)
    profile = micro_profile_for_epoch(epoch)
    digest_record = epoch_digest_record(len(dataset), int(cfg["seed"]), epoch)
    indices = rank_epoch_indices(len(dataset), int(cfg["seed"]), epoch, rank)
    generator = torch.Generator().manual_seed(
        int(cfg["seed"]) + epoch * WORLD_SIZE + rank
    )
    loader = DataLoader(
        dataset,
        batch_size=profile.micro_batch,
        sampler=FixedRankIndexSampler(indices),
        num_workers=workers_per_rank,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=generator,
    )
    expected_micros = phase.steps_per_epoch * profile.accumulation
    if len(loader) != expected_micros:
        raise RuntimeError(
            f"rank epoch loader length drift: {len(loader)} != {expected_micros}"
        )
    return loader, phase, profile, digest_record


def _broadcast_object(value, rank: int):
    carrier = [value if rank == 0 else None]
    dist.broadcast_object_list(carrier, src=0)
    return carrier[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = reference.load_and_validate_config(args.config)
    validate_execution_environment()
    data_contract_payload(cfg)
    workers_per_rank = int(args.workers_per_rank)
    if workers_per_rank <= 0:
        raise ValueError("workers-per-rank must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP hybrid baseline training")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"exact-update hybrid baseline requires world_size={WORLD_SIZE}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS)
    )
    device = torch.device("cuda", local_rank)
    lock_fd: int | None = None
    log_file = None
    try:
        seed = int(cfg["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        run_dir = ROOT / "artifacts/checkpoints" / args.run_name
        log_path = ROOT / "artifacts/logs" / f"{args.run_name}.csv"
        metric_path = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
        if args.fresh:
            reference.assert_fresh_outputs_absent(run_dir, log_path, metric_path)
        # Every rank must finish the read-only absence check before rank zero
        # creates the run directory; otherwise a scheduling race can make a
        # slower peer mistake the new contract/lock for stale output.
        dist.barrier()
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = acquire_run_lock(run_dir, args.run_name)
            contract_sha = ensure_run_contract_rank0(
                run_dir, cfg, args, workers_per_rank
            )
        else:
            contract_sha = None
        contract_sha = _broadcast_object(contract_sha, rank)
        dist.barrier()

        dataset = AIOTrainDataset(
            cfg["data_root"],
            cfg["list_root"],
            "aio3",
            cfg["crop_size"],
            strict=True,
            split_manifest=cfg["split_manifest"],
            split=cfg.get("train_split", "train"),
        )
        if len(dataset) != reference.EXPECTED_DATASET_SIZE:
            raise RuntimeError(
                f"formal AIO-3 train size drift: {len(dataset)} "
                f"!= {reference.EXPECTED_DATASET_SIZE}"
            )
        locked_val = (
            build_locked_val(
                cfg["data_root"], cfg["list_root"], "aio3", cfg["split_manifest"]
            )
            if rank == 0 else None
        )

        raw_model = build_model(cfg, args.stage).to(device)
        configure_trainable(raw_model, args.stage)
        groups = optimizer_groups(raw_model, args.stage, cfg["lr"])
        parameters = [parameter for group in groups for parameter in group["params"]]
        optimizer = torch.optim.Adam(
            groups, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: r2r_pretrain_epoch_ratio(
                epoch,
                cfg["lr"],
                cfg["warmup_epochs"],
                cfg["scheduler_max_epochs"],
                cfg["warmup_start_lr"],
                cfg.get("pretrain_eta_min", 0.0),
            ),
        )

        start_epoch = start_update = global_step = samples_seen = 0
        epoch_digests: dict[str, dict] = {}
        pending_validation = None
        prior_accounting = None
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume_payload(
                payload,
                cfg,
                args,
                workers_per_rank,
                contract_sha,
                verify_digests=True,
            )
            raw_model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload["epoch"])
            start_update = int(payload["update_in_epoch"])
            global_step = int(payload["step"])
            samples_seen = int(payload["samples_seen"])
            epoch_digests = dict(payload["epoch_update_digests"])
            pending_validation = payload.get("validation_pending")
            prior_accounting = payload.get("runtime_accounting")
            saved_rng = payload["rng_by_rank"][rank]
            torch.set_rng_state(saved_rng["torch"])
            torch.cuda.set_rng_state(saved_rng["cuda"], device=device)
            np.random.set_state(saved_rng["numpy"])
            random.setstate(saved_rng["python"])
            if rank == 0:
                reconcile_training_csv(log_path, global_step)

        ddp = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            sidecar = run_dir / "runtime_accounting.json"
            if prior_accounting is None and sidecar.is_file():
                prior_accounting = read_runtime_sidecar(sidecar)
            start_runtime_accounting(
                gpu_count=WORLD_SIZE,
                run_name=args.run_name,
                protocol="aio3",
                stage=args.stage,
                prior=prior_accounting,
            )
            print(json.dumps({
                "event": "DDP_RESUME" if args.resume else "DDP_FRESH",
                "stage": args.stage,
                "world_size": WORLD_SIZE,
                "start_epoch": start_epoch,
                "start_update": start_update,
                "start_step": global_step,
                "samples_seen": samples_seen,
            }), flush=True)

        if pending_validation is not None:
            if pending_validation != "epoch" or start_update != 0:
                raise RuntimeError("invalid pending DDP hybrid validation transaction")
            commit_pending_validation_ddp(
                raw_model=raw_model,
                locked_val=locked_val,
                run_dir=run_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                args=args,
                workers_per_rank=workers_per_rank,
                run_contract_sha256=contract_sha,
                epoch=start_epoch,
                step=global_step,
                samples_seen=samples_seen,
                epoch_digests=epoch_digests,
                rank=rank,
                device=device,
            )

        writer = None
        if rank == 0:
            header_needed = not log_path.is_file()
            log_file = log_path.open("a", newline="")
            writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "time", "epoch", "step", "samples_seen", "loss", "rest",
                    "lr", "peak_gb", "global_batch", "local_effective_batch",
                    "micro_batch", "accumulation", "sync_policy",
                    "epoch_digest", "update_digest_root",
                ],
            )
            if header_needed:
                writer.writeheader()
                log_file.flush()

        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, int(cfg["epochs"])):
            loader, phase, profile, digest_record = build_rank_epoch_loader(
                dataset,
                cfg=cfg,
                epoch=epoch,
                rank=rank,
                workers_per_rank=workers_per_rank,
            )
            key = str(epoch)
            if key in epoch_digests and epoch_digests[key] != digest_record:
                raise RuntimeError(f"epoch {epoch} raw-update digest changed on resume")
            epoch_digests[key] = digest_record
            ddp.train()

            rest_accumulator: torch.Tensor | None = None
            for micro_index, batch in enumerate(loader):
                update_index = micro_index // profile.accumulation
                micro_in_update = micro_index % profile.accumulation
                # Workers are recreated with an epoch/rank-addressed seed.
                # Enumerating skipped micros replays their RNG boundaries; the
                # checkpoint cursor itself remains an optimizer-update cursor.
                if epoch == start_epoch and update_index < start_update:
                    continue
                if micro_in_update == 0:
                    if rest_accumulator is not None:
                        raise RuntimeError("previous optimizer update left partial loss state")
                    rest_accumulator = torch.zeros(
                        (), dtype=torch.float32, device=device
                    )
                elif rest_accumulator is None:
                    raise RuntimeError("resume entered the middle of an optimizer update")
                degraded = batch["degraded"].to(device, non_blocking=True)
                clean = batch["clean"].to(device, non_blocking=True)
                final_micro = micro_in_update == profile.accumulation - 1
                sync_context = nullcontext() if final_micro else ddp.no_sync()
                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = ddp(degraded)
                        rest = restoration_l1(prediction, clean)
                        loss = rest / profile.accumulation
                    loss.backward()
                rest_accumulator.add_(
                    rest.detach().float() / profile.accumulation
                )
                if not final_micro:
                    continue
                torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                samples_seen += phase.effective_batch

                if global_step % 50 == 0:
                    global_rest = rest_accumulator.clone()
                    dist.all_reduce(global_rest, op=dist.ReduceOp.SUM)
                    global_rest /= WORLD_SIZE
                    if rank == 0 and writer is not None:
                        row = {
                            "time": time.time(),
                            "epoch": epoch,
                            "step": global_step,
                            "samples_seen": samples_seen,
                            "loss": float(global_rest),
                            "rest": float(global_rest),
                            "lr": optimizer.param_groups[0]["lr"],
                            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                            "global_batch": profile.global_effective_batch,
                            "local_effective_batch": profile.local_effective_batch,
                            "micro_batch": profile.micro_batch,
                            "accumulation": profile.accumulation,
                            "sync_policy": "no_sync_non_final_micro",
                            "epoch_digest": digest_record["flat_indices_sha256"],
                            "update_digest_root": digest_record["per_update_digest_root"],
                        }
                        writer.writerow(row)
                        log_file.flush()
                        print(json.dumps(row), flush=True)

                if global_step % int(cfg["save_every_steps"]) == 0:
                    atomic_checkpoint_ddp(
                        run_dir / "last.pt",
                        raw_model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        update_in_epoch=update_index + 1,
                        step=global_step,
                        samples_seen=samples_seen,
                        epoch_digests=epoch_digests,
                        cfg=cfg,
                        args=args,
                        workers_per_rank=workers_per_rank,
                        run_contract_sha256=contract_sha,
                        rank=rank,
                    )
                rest_accumulator = None

            if rest_accumulator is not None:
                raise RuntimeError("epoch ended with an incomplete optimizer update")

            completed_epoch = epoch + 1
            scheduler.step()
            expected_step, expected_samples = reference.budget_before_epoch(completed_epoch)
            if (global_step, samples_seen) != (expected_step, expected_samples):
                raise RuntimeError(
                    f"epoch {completed_epoch} budget drift: "
                    f"actual={(global_step, samples_seen)} "
                    f"expected={(expected_step, expected_samples)}"
                )
            should_validate = (
                completed_epoch % int(cfg["validate_every_epochs"]) == 0
                or completed_epoch == int(cfg["epochs"])
            )
            atomic_checkpoint_ddp(
                run_dir / "last.pt",
                raw_model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=completed_epoch,
                update_in_epoch=0,
                step=global_step,
                samples_seen=samples_seen,
                epoch_digests=epoch_digests,
                cfg=cfg,
                args=args,
                workers_per_rank=workers_per_rank,
                run_contract_sha256=contract_sha,
                rank=rank,
                validation_pending="epoch" if should_validate else None,
            )
            start_update = 0
            if should_validate:
                commit_pending_validation_ddp(
                    raw_model=raw_model,
                    locked_val=locked_val,
                    run_dir=run_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    args=args,
                    workers_per_rank=workers_per_rank,
                    run_contract_sha256=contract_sha,
                    epoch=completed_epoch,
                    step=global_step,
                    samples_seen=samples_seen,
                    epoch_digests=epoch_digests,
                    rank=rank,
                    device=device,
                )
            dist.barrier()

        if global_step != reference.EXPECTED_TOTAL_STEPS:
            raise RuntimeError("final optimizer-step budget mismatch")
        if samples_seen != reference.EXPECTED_TOTAL_SAMPLES:
            raise RuntimeError("final raw-sample budget mismatch")
        if int(scheduler.last_epoch) != int(cfg["epochs"]):
            raise RuntimeError("final scheduler epoch mismatch")
        verify_epoch_digest_records(
            epoch_digests,
            seed=int(cfg["seed"]),
            epoch=int(cfg["epochs"]),
            update_in_epoch=0,
        )
        if rank == 0:
            print(json.dumps({
                "status": "complete",
                "stage": args.stage,
                "epoch": int(cfg["epochs"]),
                "step": global_step,
                "samples_seen": samples_seen,
                "final_schedule_digest": schedule_digest(epoch_digests),
                "checkpoint": str(run_dir / "last.pt"),
            }), flush=True)
        return 0
    finally:
        if log_file is not None:
            log_file.close()
        if lock_fd is not None:
            os.close(lock_fd)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
