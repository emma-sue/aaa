from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml
import pytest

from scripts import train_baseline_hybrid_ddp as hybrid_ddp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_orchestrator():
    path = PROJECT_ROOT / "scripts/orchestrate.py"
    spec = importlib.util.spec_from_file_location(
        "srsc_hybrid_orchestrator_integration", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_paired_rows(path: Path, offset: float = 0.0) -> tuple[str, dict]:
    task_spec = {
        "dehaze": (205, 31.0 + offset, 0.91),
        "denoise15": (103, 32.0 + offset, 0.92),
        "denoise25": (103, 33.0 + offset, 0.93),
        "denoise50": (103, 34.0 + offset, 0.94),
        "derain": (20, 35.0 + offset, 0.95),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "name", "psnr", "ssim"]
        )
        writer.writeheader()
        for task, (count, psnr, ssim) in task_spec.items():
            for index in range(count):
                writer.writerow({
                    "task": task,
                    "name": f"{task}-{index:03d}",
                    "psnr": psnr,
                    "ssim": ssim,
                })
    task_psnr = {task: values[1] for task, values in task_spec.items()}
    task_ssim = {task: values[2] for task, values in task_spec.items()}
    summary = {
        **task_psnr,
        "macro_psnr": sum(task_psnr.values()) / len(task_psnr),
        "setting_ssim": task_ssim,
        "five_setting_mean_ssim": sum(task_ssim.values()) / len(task_ssim),
    }
    return hashlib.sha256(path.read_bytes()).hexdigest(), summary


def _completed_hybrid_fixture(tmp_path: Path, monkeypatch):
    module = _load_orchestrator()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)

    config = PROJECT_ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"
    cfg = yaml.safe_load(config.read_text())
    split = Path(cfg["split_manifest"])

    run_name = "aio3_baseline_hybrid_ddp_pretrain_s1415926"
    run_dir = tmp_path / "artifacts/checkpoints" / run_name
    run_dir.mkdir(parents=True)
    expected_args = argparse.Namespace(
        config=str(config.resolve()), stage="baseline", run_name=run_name
    )
    contract_sha = hybrid_ddp.ensure_run_contract_rank0(
        run_dir, cfg, expected_args, 8
    )

    records = {str(epoch): {"epoch": epoch} for epoch in range(240)}
    model = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    final_lr = cfg["lr"] * hybrid_ddp.r2r_pretrain_epoch_ratio(
        240,
        cfg["lr"],
        cfg["warmup_epochs"],
        cfg["scheduler_max_epochs"],
        cfg["warmup_start_lr"],
        cfg["pretrain_eta_min"],
    )
    optimizer = {"state": {}, "param_groups": [{
        "lr": final_lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "initial_lr": cfg["lr"],
        "params": [0],
    }]}
    scheduler = {
        "base_lrs": [cfg["lr"]],
        "last_epoch": 240,
        "_step_count": 241,
        "_last_lr": [final_lr],
    }
    integrity = {
        "level": "full_sha256",
        "model": hybrid_ddp.full_state_sha256(model),
        "optimizer": hybrid_ddp.full_state_sha256(optimizer),
        "scheduler": hybrid_ddp.full_state_sha256(scheduler),
        "world_size": 4,
        "all_ranks_identical": True,
    }
    payload = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": 240,
        "update_in_epoch": 0,
        "microbatch_in_epoch": 0,
        "batch_in_epoch": 0,
        "cursor_authority": "optimizer_update",
        "checkpoint_boundary": "after_complete_optimizer_update",
        "step": hybrid_ddp.reference.EXPECTED_TOTAL_STEPS,
        "samples_seen": hybrid_ddp.reference.EXPECTED_TOTAL_SAMPLES,
        "validation_pending": None,
        "validation_transaction_schema": 1,
        "training_origin": hybrid_ddp.TRAINING_ORIGIN,
        "config": cfg,
        "config_sha256": hybrid_ddp.sha256_file(config),
        "split_manifest_sha256": hybrid_ddp.sha256_file(split),
        "args": hybrid_ddp.stable_args(expected_args, 8),
        "reference_schedule": hybrid_ddp.reference.expected_schedule_payload(),
        "reference_schedule_sha256": hybrid_ddp.reference.schedule_sha256(
            cfg["seed"]
        ),
        "partition_algorithm": hybrid_ddp.PARTITION_ALGORITHM,
        "run_contract_sha256": contract_sha,
        "epoch_update_digests": records,
        "schedule_digest": hybrid_ddp.schedule_digest(records),
        "active_phase": None,
        "distributed_runtime": {
            "world_size": 4,
            "workers_per_rank": 8,
            "backend": "nccl",
        },
        "ddp_integrity": integrity,
        "rng_by_rank": [{"rank": rank} for rank in range(4)],
    }
    checkpoint_name = "val_epoch240_step0330500.pt"
    torch.save(payload, run_dir / "last.pt")
    torch.save(payload, run_dir / checkpoint_name)

    metrics = []
    for epoch in range(5, 241, 5):
        step, _ = hybrid_ddp.reference.budget_before_epoch(epoch)
        paired = (
            tmp_path / "artifacts/metrics/locked_rows" / run_name
            / f"epoch{epoch:03d}_step{step:07d}.csv"
        )
        paired_sha, summary = _write_paired_rows(paired, epoch / 100_000.0)
        metrics.append({
            **summary,
            "epoch": epoch,
            "step": step,
            "paired_rows_path": str(paired.resolve()),
            "paired_rows_sha256": paired_sha,
        })
    metric_path = tmp_path / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text("".join(json.dumps(row) + "\n" for row in metrics))
    (run_dir / "top3.json").write_text(json.dumps([{
        "score": metrics[-1]["macro_psnr"],
        "epoch": 240,
        "step": hybrid_ddp.reference.EXPECTED_TOTAL_STEPS,
        "checkpoint": checkpoint_name,
    }]) + "\n")

    calls = []

    def verify_records(actual, *, seed, epoch, update_in_epoch, dataset_size=None):
        calls.append((seed, epoch, update_in_epoch, dataset_size))
        assert list(actual) == [str(value) for value in range(240)]

    monkeypatch.setattr(hybrid_ddp, "verify_epoch_digest_records", verify_records)
    monkeypatch.setattr(
        hybrid_ddp,
        "build_model",
        lambda _cfg, _stage: torch.nn.Linear(3, 2, bias=False),
    )
    return module, config, run_name, run_dir, calls


def test_hybrid_completion_accepts_only_full_uncompressed_transaction(
    tmp_path, monkeypatch
):
    module, config, run_name, run_dir, calls = _completed_hybrid_fixture(
        tmp_path, monkeypatch
    )
    kwargs = {
        "config": config,
        "stage": "baseline",
        "workers_per_rank": 8,
    }
    assert module.hybrid_ddp_complete(run_name, **kwargs)
    assert calls == [(1415926, 240, 0, None)]

    (run_dir / "formal_complete.json").write_text("{}\n")
    assert not module.hybrid_ddp_complete(run_name, **kwargs)
    (run_dir / "formal_complete.json").unlink()

    top3 = json.loads((run_dir / "top3.json").read_text())
    checkpoint = run_dir / top3[0]["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["epoch_update_digests"] = copy.deepcopy(
        payload["epoch_update_digests"]
    )
    payload["epoch_update_digests"]["0"] = {"epoch": 0, "drift": True}
    payload["schedule_digest"] = hybrid_ddp.schedule_digest(
        payload["epoch_update_digests"]
    )
    torch.save(payload, checkpoint)
    assert not module.hybrid_ddp_complete(run_name, **kwargs)


def test_clean_and_matched_terminal_update_sets_must_be_identical(
    tmp_path, monkeypatch
):
    module = _load_orchestrator()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    run_names = [
        "aio3_baseline_hybrid_ddp_pretrain_s1415926",
        "aio3_baseline_matched_hybrid_ddp_pretrain_s1415926",
    ]
    records = {str(epoch): {"epoch": epoch} for epoch in range(240)}
    common = {
        "reference_schedule": hybrid_ddp.reference.expected_schedule_payload(),
        "reference_schedule_sha256": hybrid_ddp.reference.schedule_sha256(1415926),
        "partition_algorithm": hybrid_ddp.PARTITION_ALGORITHM,
        "epoch_update_digests": records,
        "schedule_digest": hybrid_ddp.schedule_digest(records),
    }
    for run_name in run_names:
        run_dir = tmp_path / "artifacts/checkpoints" / run_name
        run_dir.mkdir(parents=True)
        torch.save(common, run_dir / "last.pt")
    module.assert_matching_hybrid_ddp_update_digests(run_names)

    matched = tmp_path / "artifacts/checkpoints" / run_names[1] / "last.pt"
    drift = copy.deepcopy(common)
    drift["epoch_update_digests"]["55"] = {"epoch": 55, "drift": True}
    drift["schedule_digest"] = hybrid_ddp.schedule_digest(
        drift["epoch_update_digests"]
    )
    torch.save(drift, matched)
    with pytest.raises(RuntimeError, match="paired-update drift"):
        module.assert_matching_hybrid_ddp_update_digests(run_names)
