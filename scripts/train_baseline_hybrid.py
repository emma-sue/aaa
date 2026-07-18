#!/usr/bin/env python3
"""Optimization-budget-matched AIO-3 clean Restormer baselines.

The live SRSC Stage-A run used global batch 64 for epochs 0..54 and global
batch 120 for epochs 55..239.  This trainer reproduces that *single continuous
optimization schedule* for the clean and parameter-matched Restormer controls.
It intentionally remains a single-GPU entry point so the two controls can run
independently through the existing one-arm-per-GPU scheduler.

The raw training indices are reconstructed at optimizer-batch granularity.
This matches the observable sample/update budget, but does not claim bitwise
identity with the old single-process/four-rank DataLoader worker RNG streams.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    seed_all,
    sha256_file,
    upsert_validation_record,
    validate_locked,
)
from src.data import AIOTrainDataset, build_locked_val  # noqa: E402


ALLOWED_STAGES = ("baseline", "baseline_matched")
EXPECTED_DATASET_SIZE = 137_669
EXPECTED_TOTAL_STEPS = 330_500
EXPECTED_TOTAL_SAMPLES = 33_034_920
SCHEDULE_SCHEMA = 1
INDEX_ALGORITHM = "exact_aio3_main_v1"


@dataclass(frozen=True)
class HybridPhase:
    start_epoch: int
    end_epoch: int
    micro_batch: int
    accumulation: int
    effective_batch: int
    steps_per_epoch: int
    samples_per_epoch: int
    sampler: str


EXPECTED_PHASES = (
    HybridPhase(
        0, 55, 16, 4, 64, 2_151, 137_664,
        "legacy_single_gpu_random_sampler",
    ),
    HybridPhase(
        55, 240, 15, 8, 120, 1_147, 137_640,
        "reconstructed_four_rank_distributed_sampler",
    ),
)


class FixedIndexSampler(Sampler[int]):
    """Yield one already-audited epoch index sequence without reshuffling."""

    def __init__(self, indices: torch.Tensor):
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise TypeError("fixed indices must be a one-dimensional int64 tensor")
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.indices)

    def __len__(self) -> int:
        return int(self.indices.numel())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=ALLOWED_STAGES)
    parser.add_argument("--run-name", required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--fresh", action="store_true")
    start.add_argument("--resume")
    parser.add_argument(
        "--workers-override", type=int,
        help="runtime-only DataLoader worker count, frozen in the run contract",
    )
    return parser.parse_args(argv)


def _phase_dict(phase: HybridPhase) -> dict:
    return {
        "start_epoch": phase.start_epoch,
        "end_epoch": phase.end_epoch,
        "micro_batch": phase.micro_batch,
        "accumulation": phase.accumulation,
        "effective_batch": phase.effective_batch,
        "steps_per_epoch": phase.steps_per_epoch,
        "samples_per_epoch": phase.samples_per_epoch,
        "sampler": phase.sampler,
    }


def expected_schedule_payload() -> dict:
    return {
        "schema": SCHEDULE_SCHEMA,
        "index_algorithm": INDEX_ALGORITHM,
        "dataset_size": EXPECTED_DATASET_SIZE,
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "expected_total_samples": EXPECTED_TOTAL_SAMPLES,
        "phases": [_phase_dict(phase) for phase in EXPECTED_PHASES],
    }


def schedule_sha256(seed: int, schedule: dict | None = None) -> str:
    payload = {
        "seed": int(seed),
        "schedule": schedule if schedule is not None else expected_schedule_payload(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("protocol") != "aio3":
        raise ValueError("hybrid baseline trainer supports only protocol=aio3")
    if int(cfg.get("epochs", -1)) != 240:
        raise ValueError("hybrid baseline trainer requires exactly 240 epochs")
    if cfg.get("precision") != "bf16":
        raise ValueError("hybrid baseline trainer requires the audited bf16 path")
    if cfg.get("optimizer", "adam").lower() != "adam":
        raise ValueError("hybrid baseline trainer requires Adam")
    if cfg.get("hybrid_schedule") != expected_schedule_payload():
        raise ValueError("hybrid schedule differs from the preregistered AIO-3 schedule")
    if int(cfg.get("validate_every_epochs", -1)) != 5:
        raise ValueError("hybrid baseline validation must run every five epochs")
    if int(cfg.get("save_every_steps", -1)) <= 0:
        raise ValueError("save_every_steps must be positive")
    return cfg


def phase_for_epoch(epoch: int) -> HybridPhase:
    for phase in EXPECTED_PHASES:
        if phase.start_epoch <= epoch < phase.end_epoch:
            return phase
    raise ValueError(f"epoch {epoch} is outside the registered 0..239 schedule")


def budget_before_epoch(epoch: int) -> tuple[int, int]:
    """Return optimizer steps and samples consumed before zero-based epoch."""
    if not 0 <= epoch <= 240:
        raise ValueError("completed epoch must be in 0..240")
    steps = samples = 0
    for phase in EXPECTED_PHASES:
        completed = max(0, min(epoch, phase.end_epoch) - phase.start_epoch)
        steps += completed * phase.steps_per_epoch
        samples += completed * phase.samples_per_epoch
    return steps, samples


def expected_progress(epoch: int, batch_in_epoch: int) -> tuple[int, int]:
    """Expected counters for an epoch-boundary or optimizer-boundary save."""
    steps, samples = budget_before_epoch(epoch)
    if batch_in_epoch == 0:
        return steps, samples
    if epoch >= 240:
        raise ValueError("the final epoch cannot have a nonzero batch cursor")
    phase = phase_for_epoch(epoch)
    if batch_in_epoch < 0 or batch_in_epoch > phase.steps_per_epoch * phase.accumulation:
        raise ValueError("batch cursor is outside the active epoch")
    if batch_in_epoch % phase.accumulation:
        raise ValueError("checkpoint cursor is not an optimizer-step boundary")
    updates = batch_in_epoch // phase.accumulation
    return steps + updates, samples + updates * phase.effective_batch


def _legacy_single_gpu_indices(dataset_size: int, seed: int, epoch: int) -> torch.Tensor:
    """Reproduce the old RandomSampler order, including DataLoader base seed.

    In PyTorch 2.3 the DataLoader consumes one int64 draw from its generator
    before RandomSampler asks the same generator for ``randperm``.  This detail
    is covered by a CPU test and is part of the registered index algorithm.
    """
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    permutation = torch.randperm(dataset_size, generator=generator)
    return permutation[: EXPECTED_PHASES[0].samples_per_epoch].contiguous()


def _reconstructed_four_rank_indices(
    dataset_size: int, seed: int, epoch: int,
) -> torch.Tensor:
    """Reassemble the exact raw sample sets of each four-rank DDP update."""
    world_size = 4
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    permutation = torch.randperm(dataset_size, generator=generator)
    total_size = dataset_size - dataset_size % world_size
    permutation = permutation[:total_size]
    phase = EXPECTED_PHASES[1]
    local_per_update = phase.effective_batch // world_size
    local_used = phase.steps_per_epoch * local_per_update
    ranks = [
        permutation[rank:total_size:world_size][:local_used]
        .view(phase.steps_per_epoch, local_per_update)
        for rank in range(world_size)
    ]
    # [step, rank, local-position], then flatten into single-GPU micro-batches.
    return torch.stack(ranks, dim=1).reshape(-1).contiguous()


def epoch_indices(dataset_size: int, seed: int, epoch: int) -> torch.Tensor:
    if dataset_size != EXPECTED_DATASET_SIZE:
        raise ValueError(
            f"AIO-3 train size drift: actual={dataset_size} "
            f"expected={EXPECTED_DATASET_SIZE}"
        )
    phase = phase_for_epoch(epoch)
    if phase is EXPECTED_PHASES[0]:
        indices = _legacy_single_gpu_indices(dataset_size, seed, epoch)
    else:
        indices = _reconstructed_four_rank_indices(dataset_size, seed, epoch)
    if indices.numel() != phase.samples_per_epoch:
        raise RuntimeError("epoch index count does not match registered sample budget")
    return indices


def index_digest(indices: torch.Tensor) -> str:
    little_endian = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def expected_digest_keys(epoch: int, batch_in_epoch: int) -> list[str]:
    stop = epoch + (1 if batch_in_epoch else 0)
    return [str(value) for value in range(stop)]


def verify_epoch_digests(
    digests: dict[str, str], seed: int, epoch: int, batch_in_epoch: int,
) -> None:
    expected_keys = expected_digest_keys(epoch, batch_in_epoch)
    if sorted(digests, key=int) != expected_keys:
        raise RuntimeError(
            f"checkpoint epoch-index digest coverage mismatch: "
            f"actual={sorted(digests, key=int)} expected={expected_keys}"
        )
    for key in expected_keys:
        actual = index_digest(epoch_indices(EXPECTED_DATASET_SIZE, seed, int(key)))
        if digests[key] != actual:
            raise RuntimeError(f"epoch-index digest mismatch at epoch {key}")


def stable_args(args: argparse.Namespace, workers: int) -> dict:
    return {
        "config": str(Path(args.config).resolve()),
        "stage": args.stage,
        "run_name": args.run_name,
        "workers": int(workers),
    }


def run_contract_payload(
    cfg: dict, args: argparse.Namespace, workers: int,
) -> dict:
    config_path = Path(args.config).resolve()
    split_path = Path(cfg["split_manifest"]).resolve()
    code_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/train.py",
        ROOT / "src/net/clean_restormer_aio.py",
        ROOT / "src/data/aio_dataset.py",
    )
    return {
        "schema": 1,
        "purpose": "aio3_exact_hybrid_optimization_budget_baseline",
        "args": stable_args(args, workers),
        "config_sha256": sha256_file(config_path),
        "split_manifest_sha256": sha256_file(split_path),
        "schedule": expected_schedule_payload(),
        "schedule_sha256": schedule_sha256(cfg["seed"]),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
        },
        "training_origin": "fresh_hybrid_baseline",
        "stochastic_claim_boundary": (
            "raw sample identities per optimizer update are matched; "
            "worker augmentation RNG and floating reduction order are not bitwise matched"
        ),
    }


def ensure_run_contract(
    run_dir: Path, cfg: dict, args: argparse.Namespace, workers: int,
) -> str:
    contract = run_contract_payload(cfg, args, workers)
    path = run_dir / "run_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("immutable hybrid baseline run contract mismatch")
    else:
        if any(run_dir.glob("*.pt")):
            raise RuntimeError("checkpoint exists without a hybrid run contract")
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    return sha256_file(path)


def assert_fresh_outputs_absent(
    run_dir: Path, log_path: Path, metric_path: Path,
) -> None:
    conflicts: list[Path] = []
    if run_dir.exists():
        conflicts.extend(path for path in run_dir.iterdir() if path.name != ".DS_Store")
    conflicts.extend(path for path in (log_path, metric_path) if path.exists())
    if conflicts:
        raise RuntimeError(
            "fresh hybrid baseline refuses to overwrite artifacts: "
            + ", ".join(str(path) for path in sorted(conflicts, key=str)[:8])
        )


def atomic_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    samples_seen: int,
    epoch_index_digests: dict[str, str],
    cfg: dict,
    args: argparse.Namespace,
    workers: int,
    run_contract_sha256: str,
    validation_pending: str | None = None,
) -> None:
    expected_step, expected_samples = expected_progress(epoch, batch_in_epoch)
    if (step, samples_seen) != (expected_step, expected_samples):
        raise RuntimeError(
            "hybrid checkpoint counters drifted: "
            f"actual={(step, samples_seen)} expected={(expected_step, expected_samples)}"
        )
    accounting = runtime_snapshot()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "step": int(step),
        "samples_seen": int(samples_seen),
        "validation_pending": validation_pending,
        "validation_transaction_schema": 1,
        "config": cfg,
        "config_sha256": sha256_file(args.config),
        "split_manifest_sha256": sha256_file(cfg["split_manifest"]),
        "args": stable_args(args, workers),
        "training_origin": "fresh_hybrid_baseline",
        "hybrid_schedule": expected_schedule_payload(),
        "hybrid_schedule_sha256": schedule_sha256(cfg["seed"]),
        "run_contract_sha256": run_contract_sha256,
        "epoch_index_digests": dict(epoch_index_digests),
        "active_phase": (
            _phase_dict(phase_for_epoch(epoch)) if epoch < cfg["epochs"] else None
        ),
        "runtime_accounting": accounting,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    atomic_write_runtime_sidecar(path.parent / "runtime_accounting.json", accounting)


def validate_resume_payload(
    payload: dict,
    cfg: dict,
    args: argparse.Namespace,
    workers: int,
    run_contract_sha256: str,
    *,
    verify_digests: bool = True,
) -> None:
    if payload.get("training_origin") != "fresh_hybrid_baseline":
        raise RuntimeError("resume checkpoint is not a fresh hybrid baseline")
    if payload.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("resume config hash mismatch")
    if payload.get("config") != cfg:
        raise RuntimeError("resume effective config mismatch")
    if payload.get("split_manifest_sha256") != sha256_file(cfg["split_manifest"]):
        raise RuntimeError("resume split-manifest mismatch")
    if payload.get("args") != stable_args(args, workers):
        raise RuntimeError("resume hybrid baseline argument mismatch")
    if payload.get("hybrid_schedule") != expected_schedule_payload():
        raise RuntimeError("resume hybrid schedule payload mismatch")
    if payload.get("hybrid_schedule_sha256") != schedule_sha256(cfg["seed"]):
        raise RuntimeError("resume hybrid schedule hash mismatch")
    if payload.get("run_contract_sha256") != run_contract_sha256:
        raise RuntimeError("resume run-contract hash mismatch")
    epoch = int(payload.get("epoch", -1))
    batch_in_epoch = int(payload.get("batch_in_epoch", -1))
    expected_step, expected_samples = expected_progress(epoch, batch_in_epoch)
    if int(payload.get("step", -1)) != expected_step:
        raise RuntimeError("resume optimizer-step budget mismatch")
    if int(payload.get("samples_seen", -1)) != expected_samples:
        raise RuntimeError("resume sample budget mismatch")
    if int(payload.get("scheduler", {}).get("last_epoch", -1)) != epoch:
        raise RuntimeError("resume epoch scheduler position mismatch")
    digests = payload.get("epoch_index_digests")
    if not isinstance(digests, dict):
        raise RuntimeError("resume checkpoint lacks epoch-index digests")
    if verify_digests:
        verify_epoch_digests(digests, int(cfg["seed"]), epoch, batch_in_epoch)


def update_top3(
    *,
    run_dir: Path,
    score: float,
    epoch: int,
    step: int,
    model,
    optimizer,
    scheduler,
    samples_seen: int,
    epoch_index_digests: dict[str, str],
    cfg: dict,
    args: argparse.Namespace,
    workers: int,
    run_contract_sha256: str,
) -> None:
    checkpoint = run_dir / f"val_epoch{epoch:03d}_step{step:07d}.pt"
    atomic_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        batch_in_epoch=0,
        step=step,
        samples_seen=samples_seen,
        epoch_index_digests=epoch_index_digests,
        cfg=cfg,
        args=args,
        workers=workers,
        run_contract_sha256=run_contract_sha256,
    )
    index_path = run_dir / "top3.json"
    records = json.loads(index_path.read_text()) if index_path.is_file() else []
    records = [
        row for row in records
        if (int(row["epoch"]), int(row["step"])) != (epoch, step)
    ]
    records.append({
        "score": float(score), "epoch": epoch, "step": step,
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


def commit_pending_validation(
    *,
    model,
    locked_val,
    run_dir: Path,
    optimizer,
    scheduler,
    cfg: dict,
    args: argparse.Namespace,
    workers: int,
    run_contract_sha256: str,
    epoch: int,
    step: int,
    samples_seen: int,
    epoch_index_digests: dict[str, str],
) -> dict:
    summary, paired_rows = validate_locked(
        model, locked_val, args.stage, None, "O0", None,
        protocol="aio3", return_rows=True,
    )
    summary.update({"epoch": int(epoch), "step": int(step)})
    paired_path = (
        ROOT / "artifacts/metrics/locked_rows" / args.run_name
        / f"epoch{epoch:03d}_step{step:07d}.csv"
    )
    summary["paired_rows_path"] = str(paired_path.resolve())
    summary["paired_rows_sha256"] = atomic_write_locked_rows(
        paired_path, paired_rows
    )
    metric_path = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    upsert_validation_record(metric_path, summary)
    print("LOCKED_VAL " + json.dumps(summary), flush=True)
    update_top3(
        run_dir=run_dir,
        score=summary["macro_psnr"],
        epoch=epoch,
        step=step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        samples_seen=samples_seen,
        epoch_index_digests=epoch_index_digests,
        cfg=cfg,
        args=args,
        workers=workers,
        run_contract_sha256=run_contract_sha256,
    )
    atomic_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        batch_in_epoch=0,
        step=step,
        samples_seen=samples_seen,
        epoch_index_digests=epoch_index_digests,
        cfg=cfg,
        args=args,
        workers=workers,
        run_contract_sha256=run_contract_sha256,
        validation_pending=None,
    )
    return summary


def build_epoch_loader(
    dataset,
    *,
    cfg: dict,
    epoch: int,
    workers: int,
) -> tuple[DataLoader, HybridPhase, str]:
    phase = phase_for_epoch(epoch)
    indices = epoch_indices(len(dataset), int(cfg["seed"]), epoch)
    digest = index_digest(indices)
    worker_generator = torch.Generator().manual_seed(int(cfg["seed"]) + epoch)
    loader = DataLoader(
        dataset,
        batch_size=phase.micro_batch,
        sampler=FixedIndexSampler(indices),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=worker_generator,
    )
    expected_micro_batches = phase.steps_per_epoch * phase.accumulation
    if len(loader) != expected_micro_batches:
        raise RuntimeError(
            f"hybrid epoch loader length drift: actual={len(loader)} "
            f"expected={expected_micro_batches}"
        )
    return loader, phase, digest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_and_validate_config(args.config)
    workers = int(args.workers_override or cfg["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hybrid baseline training")

    seed_all(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    run_dir = ROOT / "artifacts/checkpoints" / args.run_name
    log_path = ROOT / "artifacts/logs" / f"{args.run_name}.csv"
    metric_path = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
    if args.fresh:
        assert_fresh_outputs_absent(run_dir, log_path, metric_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_run_lock(run_dir, args.run_name)

    try:
        contract_sha = ensure_run_contract(run_dir, cfg, args, workers)
        dataset = AIOTrainDataset(
            cfg["data_root"], cfg["list_root"], "aio3", cfg["crop_size"],
            strict=True, split_manifest=cfg["split_manifest"],
            split=cfg.get("train_split", "train"),
        )
        if len(dataset) != EXPECTED_DATASET_SIZE:
            raise RuntimeError(
                f"formal AIO-3 train size drift: {len(dataset)} != {EXPECTED_DATASET_SIZE}"
            )
        locked_val = build_locked_val(
            cfg["data_root"], cfg["list_root"], "aio3", cfg["split_manifest"]
        )

        model = build_model(cfg, args.stage).cuda()
        configure_trainable(model, args.stage)
        groups = optimizer_groups(model, args.stage, cfg["lr"])
        params = [parameter for group in groups for parameter in group["params"]]
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

        start_epoch = start_batch = global_step = samples_seen = 0
        epoch_digests: dict[str, str] = {}
        pending_validation = None
        prior_accounting = None
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume_payload(
                payload, cfg, args, workers, contract_sha, verify_digests=True
            )
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload["epoch"])
            start_batch = int(payload["batch_in_epoch"])
            global_step = int(payload["step"])
            samples_seen = int(payload["samples_seen"])
            epoch_digests = dict(payload["epoch_index_digests"])
            pending_validation = payload.get("validation_pending")
            prior_accounting = payload.get("runtime_accounting")
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
            np.random.set_state(payload["rng"]["numpy"])
            random.setstate(payload["rng"]["python"])
            reconcile_training_csv(log_path, global_step)

        sidecar = run_dir / "runtime_accounting.json"
        if prior_accounting is None and sidecar.is_file():
            prior_accounting = read_runtime_sidecar(sidecar)
        start_runtime_accounting(
            gpu_count=1,
            run_name=args.run_name,
            protocol="aio3",
            stage=args.stage,
            prior=prior_accounting,
        )

        if pending_validation is not None:
            if pending_validation != "epoch" or start_batch != 0:
                raise RuntimeError("invalid pending hybrid validation transaction")
            commit_pending_validation(
                model=model,
                locked_val=locked_val,
                run_dir=run_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                args=args,
                workers=workers,
                run_contract_sha256=contract_sha,
                epoch=start_epoch,
                step=global_step,
                samples_seen=samples_seen,
                epoch_index_digests=epoch_digests,
            )

        header_needed = not log_path.is_file()
        with log_path.open("a", newline="") as log_file:
            writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "time", "epoch", "step", "samples_seen", "loss", "rest",
                    "lr", "peak_gb", "effective_batch", "index_digest",
                ],
            )
            if header_needed:
                writer.writeheader()
                log_file.flush()
            optimizer.zero_grad(set_to_none=True)

            for epoch in range(start_epoch, int(cfg["epochs"])):
                loader, phase, digest = build_epoch_loader(
                    dataset, cfg=cfg, epoch=epoch, workers=workers
                )
                key = str(epoch)
                if key in epoch_digests and epoch_digests[key] != digest:
                    raise RuntimeError(f"epoch {epoch} index digest changed on resume")
                epoch_digests[key] = digest
                model.train()

                for batch_index, batch in enumerate(loader):
                    # Enumerating skipped batches deliberately replays worker RNG
                    # before a mid-epoch resume continues.
                    if epoch == start_epoch and batch_index < start_batch:
                        continue
                    degraded = batch["degraded"].cuda(non_blocking=True)
                    clean = batch["clean"].cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(degraded)
                        rest = restoration_l1(prediction, clean)
                        loss = rest / phase.accumulation
                    loss.backward()
                    if (batch_index + 1) % phase.accumulation:
                        continue
                    torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    samples_seen += phase.effective_batch

                    if global_step % 50 == 0:
                        row = {
                            "time": __import__("time").time(),
                            "epoch": epoch,
                            "step": global_step,
                            "samples_seen": samples_seen,
                            "loss": float(loss.detach() * phase.accumulation),
                            "rest": float(rest.detach()),
                            "lr": optimizer.param_groups[0]["lr"],
                            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                            "effective_batch": phase.effective_batch,
                            "index_digest": digest,
                        }
                        writer.writerow(row)
                        log_file.flush()
                        print(json.dumps(row), flush=True)

                    if global_step % int(cfg["save_every_steps"]) == 0:
                        atomic_checkpoint(
                            run_dir / "last.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            batch_in_epoch=batch_index + 1,
                            step=global_step,
                            samples_seen=samples_seen,
                            epoch_index_digests=epoch_digests,
                            cfg=cfg,
                            args=args,
                            workers=workers,
                            run_contract_sha256=contract_sha,
                        )

                completed_epoch = epoch + 1
                scheduler.step()
                expected_step, expected_samples = budget_before_epoch(completed_epoch)
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
                atomic_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=completed_epoch,
                    batch_in_epoch=0,
                    step=global_step,
                    samples_seen=samples_seen,
                    epoch_index_digests=epoch_digests,
                    cfg=cfg,
                    args=args,
                    workers=workers,
                    run_contract_sha256=contract_sha,
                    validation_pending="epoch" if should_validate else None,
                )
                start_batch = 0
                if should_validate:
                    commit_pending_validation(
                        model=model,
                        locked_val=locked_val,
                        run_dir=run_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        args=args,
                        workers=workers,
                        run_contract_sha256=contract_sha,
                        epoch=completed_epoch,
                        step=global_step,
                        samples_seen=samples_seen,
                        epoch_index_digests=epoch_digests,
                    )

        if global_step != EXPECTED_TOTAL_STEPS:
            raise RuntimeError(
                f"final optimizer budget mismatch: {global_step} != {EXPECTED_TOTAL_STEPS}"
            )
        if samples_seen != EXPECTED_TOTAL_SAMPLES:
            raise RuntimeError(
                f"final sample budget mismatch: {samples_seen} != {EXPECTED_TOTAL_SAMPLES}"
            )
        if int(scheduler.last_epoch) != 240:
            raise RuntimeError(f"final scheduler epoch mismatch: {scheduler.last_epoch}")
        if sorted(epoch_digests, key=int) != [str(epoch) for epoch in range(240)]:
            raise RuntimeError("final checkpoint does not cover all 240 epoch index digests")
        print(json.dumps({
            "status": "complete",
            "run": args.run_name,
            "stage": args.stage,
            "epoch": 240,
            "step": global_step,
            "samples_seen": samples_seen,
            "checkpoint": str(run_dir / "last.pt"),
        }))
        return 0
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
