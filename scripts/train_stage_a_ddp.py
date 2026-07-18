#!/usr/bin/env python3
"""Four-GPU Stage-A training with strict fresh/resume provenance.

This entry point is deliberately separate from ``train.py``: it can either
resume the audited AIO-3 checkpoint or start the independently initialized
AIO-5 Stage-A run.  It trains only Encoder+D1 and writes checkpoints that
remain compatible with the original orchestrator.  All ranks execute
identical optimizer steps; only rank zero writes logs and validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGE_A_CODE_FILES = (
    "scripts/train_stage_a_ddp.py",
    "scripts/train.py",
    "src/data/aio_dataset.py",
    "src/net/srsc_lite.py",
    "src/net/clean_restormer_aio.py",
    "src/net/restormer_blocks.py",
    "src/losses/objectives.py",
)

from scripts.train import (  # noqa: E402
    atomic_write_locked_rows,
    build_model,
    configure_trainable,
    optimizer_groups,
    reconcile_training_csv,
    r2r_pretrain_epoch_ratio,
    restoration_l1,
    upsert_validation_record,
    validate_locked,
)
from scripts.runtime_accounting import (  # noqa: E402
    atomic_write_runtime_sidecar,
    read_runtime_sidecar,
    runtime_snapshot,
    start_runtime_accounting,
)
from src.data import (  # noqa: E402
    AIOTrainDataset,
    build_locked_val,
    validate_split_list_binding,
)


class StageACoarseForward(nn.Module):
    """Expose exactly the Stage-A trainable graph through DDP.forward."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model.encoder
        self.d1 = model.d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        delta0, _ = self.d1(features)
        return x + delta0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_contract(cfg: dict) -> dict:
    """Bind Stage-A to the preregistered lists and materialization manifest."""
    split_payload = validate_split_list_binding(
        cfg["list_root"], cfg["split_manifest"], cfg["protocol"]
    )
    manifest_path = ROOT / "artifacts/manifests" / f"{cfg['protocol']}.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid data materialization manifest: {manifest_path}") from error
    list_sha256 = split_payload["list_sha256"]
    if (
        manifest.get("protocol") != cfg["protocol"]
        or int(manifest.get("missing_entries", -1)) != 0
        or int(manifest.get("expected_entries", 0)) <= 0
        or manifest.get("list_sha256") != list_sha256
    ):
        raise RuntimeError(
            "data materialization manifest is not bound to the locked split: "
            f"{manifest_path}"
        )
    return {
        "protocol": cfg["protocol"],
        "materialization_manifest": str(manifest_path.resolve()),
        "materialization_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(Path(cfg["split_manifest"]).resolve()),
        "split_manifest_sha256": sha256_file(cfg["split_manifest"]),
        "list_sha256": list_sha256,
    }


def resolve_code_contract() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in STAGE_A_CODE_FILES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--resume")
    start.add_argument("--fresh", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--per-gpu-batch", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers-per-rank", type=int, default=8)
    parser.add_argument(
        "--enforce-config-effective-batch",
        action="store_true",
        help="require world*per-GPU*accumulation to equal config effective_batch",
    )
    parser.add_argument(
        "--require-training-origin",
        help="reject resume unless the checkpoint has this immutable origin",
    )
    parser.add_argument("--max-new-steps", type=int, default=0)
    parser.add_argument("--smoke-no-save", action="store_true")
    # Preserve the checkpoint provenance contract used by the verifier.
    parser.set_defaults(stage="a", feedback="O7", seed_override=None)
    return parser.parse_args()


def rank_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": [torch.cuda.get_rng_state()],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def atomic_ddp_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    cfg: dict,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    validation_pending: str | None = None,
) -> None:
    local_rng = rank_rng()
    gathered: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_rng)
    if rank == 0:
        accounting = runtime_snapshot()
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "step": step,
            "validation_pending": validation_pending,
            "validation_transaction_schema": 1,
            "training_origin": getattr(args, "training_origin", "unknown"),
            "config": cfg,
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256(
                Path(cfg["split_manifest"]).read_bytes()
            ).hexdigest(),
            "data_contract": args.data_contract,
            "code_contract": args.code_contract,
            "args": {
                "config": args.config,
                "stage": "a",
                "feedback": "O7",
                "run_name": args.run_name,
                "resume": args.resume,
                "fresh": args.fresh,
                "seed_override": None,
                "workers_per_rank": args.workers_per_rank,
                "enforce_config_effective_batch": args.enforce_config_effective_batch,
                "require_training_origin": args.require_training_origin,
            },
            "rng": gathered[0],
            "rng_by_rank": gathered,
            "distributed_runtime": {
                "world_size": world_size,
                "per_gpu_batch": args.per_gpu_batch,
                "accumulation": args.accumulation,
                "workers_per_rank": args.workers_per_rank,
                "global_effective_batch": (
                    world_size * args.per_gpu_batch * args.accumulation
                ),
                "backend": dist.get_backend(),
            },
            "runtime_accounting": accounting,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        atomic_write_runtime_sidecar(
            path.parent / "runtime_accounting.json", accounting
        )
    dist.barrier()


def expected_distributed_runtime(
    args: argparse.Namespace, world_size: int
) -> dict[str, int]:
    return {
        "world_size": int(world_size),
        "per_gpu_batch": int(args.per_gpu_batch),
        "accumulation": int(args.accumulation),
        "global_effective_batch": int(
            world_size * args.per_gpu_batch * args.accumulation
        ),
        "workers_per_rank": int(args.workers_per_rank),
    }


def validate_distributed_resume_runtime(
    payload: dict, args: argparse.Namespace, world_size: int
) -> None:
    """Reject silent DDP budget changes after the first distributed save.

    A legacy single-GPU epoch-boundary checkpoint has no distributed runtime
    and is accepted only so the audited AIO-3 migration remains resumable.
    Mid-epoch legacy migration is never accepted.
    """
    expected = expected_distributed_runtime(args, world_size)
    saved = payload.get("distributed_runtime")
    batch_in_epoch = int(payload.get("batch_in_epoch", -1))
    if saved is None:
        if batch_in_epoch != 0:
            raise RuntimeError(
                "mid-epoch DDP resume requires a distributed_runtime contract"
            )
        return
    required = ("world_size", "per_gpu_batch", "accumulation", "global_effective_batch")
    mismatches = {
        key: (saved.get(key), expected[key])
        for key in required
        if int(saved.get(key, -1)) != expected[key]
    }
    if "workers_per_rank" in saved and int(saved["workers_per_rank"]) != expected["workers_per_rank"]:
        mismatches["workers_per_rank"] = (
            saved["workers_per_rank"], expected["workers_per_rank"]
        )
    if mismatches:
        raise RuntimeError(
            "DDP resume requires the identical distributed runtime: "
            f"mismatches={mismatches}"
        )


def assert_fresh_run_is_empty(run_dir: Path, log_path: Path, metric_path: Path) -> None:
    """Never overwrite partial or completed scientific artifacts on fresh start."""
    conflicts = []
    if run_dir.exists():
        conflicts.extend(path for path in run_dir.iterdir() if path.name != ".DS_Store")
    for path in (log_path, metric_path):
        if path.exists():
            conflicts.append(path)
    if conflicts:
        rendered = ", ".join(str(path) for path in sorted(conflicts, key=str)[:8])
        raise RuntimeError(
            "fresh Stage-A run refuses to overwrite existing artifacts: " + rendered
        )


def enforce_training_origin(actual: str, required: str | None) -> None:
    if required and actual != required:
        raise RuntimeError(
            "checkpoint training-origin mismatch: "
            f"actual={actual!r} required={required!r}"
        )


def legacy_final_checkpoint_needs_validation_replay(payload: dict, epochs: int) -> bool:
    """Repair a pre-transaction final checkpoint after an epoch-edge crash."""
    return (
        int(payload.get("epoch", -1)) >= int(epochs)
        and int(payload.get("batch_in_epoch", -1)) == 0
        and payload.get("validation_transaction_schema") != 1
    )


def update_top3_ddp(
    run_dir: Path,
    score: float,
    epoch: int,
    step: int,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> None:
    checkpoint = run_dir / f"val_epoch{epoch:03d}_step{step:07d}.pt"
    atomic_ddp_checkpoint(
        checkpoint,
        raw_model,
        optimizer,
        scheduler,
        epoch,
        0,
        step,
        cfg,
        args,
        rank,
        world_size,
    )
    if rank == 0:
        index_path = run_dir / "top3.json"
        records = json.loads(index_path.read_text()) if index_path.exists() else []
        records = [
            row for row in records
            if (int(row["epoch"]), int(row["step"])) != (int(epoch), int(step))
        ]
        records.append(
            {"score": score, "epoch": epoch, "step": step, "checkpoint": checkpoint.name}
        )
        records.sort(key=lambda item: item["score"], reverse=True)
        retained = records[:3]
        stale = records[3:]
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
    rank: int,
    world_size: int,
    device: torch.device,
    epoch: int,
    step: int,
) -> None:
    """Idempotently commit metric, top3 and the cleared pending marker."""
    score_tensor = torch.zeros((), dtype=torch.float64, device=device)
    if rank == 0:
        if locked_val is None:
            raise RuntimeError("rank zero cannot recover validation without locked_val")
        summary, paired_rows = validate_locked(
            raw_model,
            locked_val,
            "a",
            None,
            "O7",
            None,
            protocol=cfg["protocol"],
            return_rows=True,
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
        val_path = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
        val_path.parent.mkdir(parents=True, exist_ok=True)
        upsert_validation_record(val_path, summary)
        print("LOCKED_VAL " + json.dumps(summary), flush=True)
        score_tensor.fill_(summary["macro_psnr"])
    dist.broadcast(score_tensor, src=0)
    update_top3_ddp(
        run_dir,
        float(score_tensor.item()),
        epoch,
        step,
        raw_model,
        optimizer,
        scheduler,
        cfg,
        args,
        rank,
        world_size,
    )
    atomic_ddp_checkpoint(
        run_dir / "last.pt",
        raw_model,
        optimizer,
        scheduler,
        epoch,
        0,
        step,
        cfg,
        args,
        rank,
        world_size,
        validation_pending=None,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2:
        raise RuntimeError("use torchrun with at least two processes")
    if args.per_gpu_batch <= 0 or args.accumulation <= 0:
        raise ValueError("batch and accumulation must be positive")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.data_contract = resolve_data_contract(cfg)
    args.code_contract = resolve_code_contract()
    global_batch = world_size * args.per_gpu_batch * args.accumulation
    if args.enforce_config_effective_batch and global_batch != int(cfg["effective_batch"]):
        raise RuntimeError(
            "distributed global batch does not match the registered config: "
            f"runtime={global_batch} config={cfg['effective_batch']}"
        )

    run_dir = ROOT / "artifacts/checkpoints" / args.run_name
    log_path = ROOT / "artifacts/logs" / f"{args.run_name}.csv"
    metric_path = ROOT / "artifacts/metrics" / f"{args.run_name}_locked_val.jsonl"
    if args.fresh:
        # Every rank performs the same read-only check so a rank-zero refusal
        # cannot leave its peers blocked in a collective.
        assert_fresh_run_is_empty(run_dir, log_path, metric_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    # Rank-specific data randomness, followed by DDP's parameter broadcast.
    seed = int(cfg["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    dataset = AIOTrainDataset(
        cfg["data_root"],
        cfg["list_root"],
        cfg["protocol"],
        cfg["crop_size"],
        strict=True,
        split_manifest=cfg["split_manifest"],
        split=cfg.get("train_split", "train"),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch,
        sampler=sampler,
        num_workers=args.workers_per_rank,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=generator,
    )
    usable_batches = (len(loader) // args.accumulation) * args.accumulation
    steps_per_epoch = usable_batches // args.accumulation
    if steps_per_epoch == 0:
        raise RuntimeError("no complete distributed optimizer step")

    raw_model = build_model(cfg, "a").to(device)
    configure_trainable(raw_model, "a")
    groups = optimizer_groups(raw_model, "a", cfg["lr"])
    params = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.Adam(
        groups, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: r2r_pretrain_epoch_ratio(
            epoch,
            cfg["lr"],
            cfg["warmup_epochs"],
            cfg.get("scheduler_max_epochs", cfg["epochs"] + 30),
            cfg.get("warmup_start_lr", 1e-7),
            cfg.get("pretrain_eta_min", 0.0),
        ),
    )

    start_epoch = start_batch = global_step = 0
    resume_validation_pending = None
    prior_runtime_accounting = None
    args.training_origin = "fresh"
    if args.fresh:
        enforce_training_origin(args.training_origin, args.require_training_origin)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_origin = payload.get("training_origin")
        args.training_origin = (
            saved_origin
            if isinstance(saved_origin, str) and saved_origin
            else "legacy_aio3_epoch_boundary_migration"
        )
        enforce_training_origin(args.training_origin, args.require_training_origin)
        checkpoint_batch = int(payload.get("batch_in_epoch", -1))
        validate_distributed_resume_runtime(payload, args, world_size)
        saved_args = payload.get("args", {})
        if saved_args.get("stage") != "a":
            raise RuntimeError("resume checkpoint is not Stage-A")
        if saved_args.get("run_name") != args.run_name:
            raise RuntimeError(
                "resume run-name mismatch: "
                f"saved={saved_args.get('run_name')!r} current={args.run_name!r}"
            )
        expected_config = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
        if payload.get("config_sha256") != expected_config:
            raise RuntimeError("resume config hash mismatch")
        if payload.get("config") != cfg:
            raise RuntimeError("resume effective config mismatch")
        expected_split = hashlib.sha256(Path(cfg["split_manifest"]).read_bytes()).hexdigest()
        if payload.get("split_manifest_sha256") != expected_split:
            raise RuntimeError("resume split hash mismatch")
        saved_data_contract = payload.get("data_contract")
        if saved_data_contract is None:
            if cfg["protocol"] != "aio3":
                raise RuntimeError("resume checkpoint has no frozen data contract")
        elif saved_data_contract != args.data_contract:
            raise RuntimeError("resume data contract mismatch")
        saved_code_contract = payload.get("code_contract")
        if saved_code_contract is None:
            if cfg["protocol"] != "aio3":
                raise RuntimeError("resume checkpoint has no frozen code contract")
        elif saved_code_contract != args.code_contract:
            raise RuntimeError("resume code contract mismatch")
        raw_model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        start_batch = checkpoint_batch
        global_step = int(payload["step"])
        resume_validation_pending = payload.get("validation_pending")
        prior_runtime_accounting = payload.get("runtime_accounting")
        if legacy_final_checkpoint_needs_validation_replay(payload, cfg["epochs"]):
            # The live AIO-3 process predates durable pending markers.  If it
            # ever crashes after its final last.pt save but before validation,
            # replay the idempotent transaction once on the next resume.
            resume_validation_pending = "epoch"
        rng_by_rank = payload.get("rng_by_rank")
        if rng_by_rank is not None:
            if len(rng_by_rank) != world_size:
                raise RuntimeError("checkpoint rank-RNG width does not match world size")
            saved_rng = rng_by_rank[rank]
            torch.set_rng_state(saved_rng["torch"])
            torch.cuda.set_rng_state(saved_rng["cuda"][0], device=device)
            np.random.set_state(saved_rng["numpy"])
            random.setstate(saved_rng["python"])
        if rank == 0:
            reconcile_training_csv(log_path, global_step)

    sidecar = run_dir / "runtime_accounting.json"
    if prior_runtime_accounting is None and sidecar.is_file():
        prior_runtime_accounting = read_runtime_sidecar(sidecar)

    start_runtime_accounting(
        gpu_count=world_size,
        run_name=args.run_name,
        protocol=cfg["protocol"],
        stage="a",
        prior=prior_runtime_accounting,
    )

    wrapper = StageACoarseForward(raw_model)
    ddp = DDP(wrapper, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "DDP_RESUME" if args.resume else "DDP_FRESH",
                    "world_size": world_size,
                    "per_gpu_batch": args.per_gpu_batch,
                    "accumulation": args.accumulation,
                    "global_effective_batch": global_batch,
                    "steps_per_epoch": steps_per_epoch,
                    "start_epoch": start_epoch,
                    "start_batch": start_batch,
                    "start_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            ),
            flush=True,
        )

    locked_val = (
        build_locked_val(
            cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["split_manifest"]
        )
        if rank == 0 and not args.smoke_no_save
        else None
    )
    if resume_validation_pending is not None:
        if resume_validation_pending != "epoch":
            raise RuntimeError(
                "unsupported DDP pending validation kind: "
                f"{resume_validation_pending!r}"
            )
        commit_pending_validation_ddp(
            raw_model=raw_model,
            locked_val=locked_val,
            run_dir=run_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            epoch=start_epoch,
            step=global_step,
        )
    log_file = log_path.open("a", newline="") if rank == 0 and not args.smoke_no_save else None
    writer = (
        csv.DictWriter(
            log_file,
            fieldnames=["time", "epoch", "step", "loss", "rest", "state", "clean", "lr", "peak_gb"],
        )
        if log_file is not None
        else None
    )
    if writer is not None and log_path.stat().st_size == 0:
        writer.writeheader()
        log_file.flush()
    initial_step = global_step
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, int(cfg["epochs"])):
            sampler.set_epoch(epoch)
            generator.manual_seed(seed + epoch * world_size + rank)
            ddp.train()
            for batch_index, batch in enumerate(loader):
                if batch_index >= usable_batches:
                    break
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                x = batch["degraded"].to(device, non_blocking=True)
                gt = batch["clean"].to(device, non_blocking=True)
                sync_step = (batch_index + 1) % args.accumulation == 0
                context = ddp.no_sync() if not sync_step else torch.enable_grad()
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = ddp(x)
                        rest = restoration_l1(prediction, gt)
                        loss = rest / args.accumulation
                    loss.backward()
                if not sync_step:
                    continue
                torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and writer is not None and global_step % 50 == 0:
                    row = {
                        "time": __import__("time").time(),
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach() * args.accumulation),
                        "rest": float(rest.detach()),
                        "state": 0.0,
                        "clean": 0.0,
                        "lr": optimizer.param_groups[0]["lr"],
                        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    writer.writerow(row)
                    log_file.flush()
                    print(json.dumps(row), flush=True)
                if not args.smoke_no_save and global_step % cfg["save_every_steps"] == 0:
                    atomic_ddp_checkpoint(
                        run_dir / "last.pt", raw_model, optimizer, scheduler,
                        epoch, batch_index + 1, global_step, cfg, args, rank, world_size,
                    )
                if args.max_new_steps and global_step - initial_step >= args.max_new_steps:
                    if rank == 0:
                        print(json.dumps({"status": "smoke_complete", "step": global_step}), flush=True)
                    return 0

            completed_epoch = epoch + 1
            scheduler.step()
            should_validate = (
                completed_epoch % cfg["validate_every_epochs"] == 0
                or completed_epoch == cfg["epochs"]
            )
            if not args.smoke_no_save:
                atomic_ddp_checkpoint(
                    run_dir / "last.pt", raw_model, optimizer, scheduler,
                    completed_epoch, 0, global_step, cfg, args, rank, world_size,
                    validation_pending=("epoch" if should_validate else None),
                )
            start_batch = 0
            if should_validate and not args.smoke_no_save:
                commit_pending_validation_ddp(
                    raw_model=raw_model,
                    locked_val=locked_val,
                    run_dir=run_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    epoch=completed_epoch,
                    step=global_step,
                )
            dist.barrier()
    finally:
        if log_file is not None:
            log_file.close()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
