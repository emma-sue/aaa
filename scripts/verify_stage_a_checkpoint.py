#!/usr/bin/env python3
"""Wait for and integrity-check a resumable Stage-A checkpoint.

This is a read-only training observer: it never imports the model, changes the
checkpoint, or touches the optimizer process.  The report is written atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/srsc_lite_v12")
DEFAULT_CHECKPOINT = ROOT / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/last.pt"
DEFAULT_CONFIG = ROOT / "configs/protocol_aio3.yaml"
EXPECTED_VALIDATION_TASKS = {
    "aio3": ("dehaze", "denoise15", "denoise25", "denoise50", "derain"),
    "aio5": ("dehaze", "denoise25", "derain", "deblur", "lowlight"),
}
EXPECTED_LOCKED_COUNTS = {
    "aio3": {"dehaze": 205, "denoise15": 103, "denoise25": 103,
             "denoise50": 103, "derain": 20},
    "aio5": {"dehaze": 205, "denoise25": 103, "derain": 20,
             "deblur": 20, "lowlight": 48},
}
STAGE_A_CODE_FILES = (
    "scripts/train_stage_a_ddp.py",
    "scripts/train.py",
    "src/data/aio_dataset.py",
    "src/net/srsc_lite.py",
    "src/net/clean_restormer_aio.py",
    "src/net/restormer_blocks.py",
    "src/losses/objectives.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--minimum-step", type=int, default=1000)
    parser.add_argument(
        "--expected-run-name", default="aio3_stage_a_coarse_seed1415926"
    )
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--expected-global-effective-batch", type=int)
    parser.add_argument("--expected-per-gpu-batch", type=int)
    parser.add_argument("--expected-accumulation", type=int)
    parser.add_argument("--expected-workers-per-rank", type=int)
    parser.add_argument("--expected-backend")
    parser.add_argument("--expected-training-origin")
    parser.add_argument("--require-validation-complete", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926/step1000_integrity.json",
    )
    return parser.parse_args()


def validation_artifact_integrity(
    run_name: str,
    expected_epoch: int,
    expected_step: int,
    expected_config_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    expected_protocol: str | None = None,
    expected_validation_interval: int | None = None,
    expected_runtime: dict[str, object] | None = None,
    expected_training_origin: str | None = None,
    expected_data_contract: dict | None = None,
    expected_code_contract: dict | None = None,
    require_paired_rows: bool = False,
) -> tuple[dict[str, bool], dict]:
    metric_path = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    index_path = run_dir / "top3.json"
    metric_rows = []
    top3 = []
    parse_ok = True
    try:
        if metric_path.is_file():
            metric_rows = [
                json.loads(line)
                for line in metric_path.read_text().splitlines()
                if line.strip()
            ]
        if index_path.is_file():
            top3 = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError, TypeError):
        parse_ok = False
    metric_by_key = {}
    metric_parse_valid = parse_ok and isinstance(metric_rows, list)
    for row in metric_rows:
        try:
            key = (int(row["epoch"]), int(row["step"]))
            score = float(row["macro_psnr"])
            if key in metric_by_key or not math.isfinite(score):
                metric_parse_valid = False
            metric_by_key[key] = row
        except (KeyError, TypeError, ValueError, OverflowError):
            metric_parse_valid = False
    final_key = (int(expected_epoch), int(expected_step))
    exact_metric = final_key in metric_by_key

    validation_boundaries_complete = True
    expected_epochs = None
    if expected_validation_interval is not None:
        if expected_validation_interval <= 0:
            raise ValueError("expected validation interval must be positive")
        expected_epochs = list(
            range(expected_validation_interval, expected_epoch + 1,
                  expected_validation_interval)
        )
        validation_boundaries_complete = (
            sorted(key[0] for key in metric_by_key) == expected_epochs
            and len(metric_by_key) == len(expected_epochs)
        )

    metric_task_schema_valid = True
    if expected_protocol is not None:
        expected_tasks = set(EXPECTED_VALIDATION_TASKS[expected_protocol])
        for row in metric_rows:
            try:
                if any(not math.isfinite(float(row[task])) for task in expected_tasks):
                    metric_task_schema_valid = False
                setting_ssim = row["setting_ssim"]
                if set(setting_ssim) != expected_tasks or any(
                    not math.isfinite(float(setting_ssim[task]))
                    for task in expected_tasks
                ):
                    metric_task_schema_valid = False
                if not math.isfinite(float(row["five_setting_mean_ssim"])):
                    metric_task_schema_valid = False
            except (KeyError, TypeError, ValueError, OverflowError):
                metric_task_schema_valid = False

    paired_rows_valid = True
    paired_errors = []
    if require_paired_rows:
        expected_counts = EXPECTED_LOCKED_COUNTS[expected_protocol]
        for row in metric_rows:
            try:
                path = Path(row["paired_rows_path"])
                expected_sha = row["paired_rows_sha256"]
                if not path.is_file() or sha256(path) != expected_sha:
                    raise RuntimeError("missing or hash-mismatched paired rows")
                with path.open(newline="") as handle:
                    records = list(csv.DictReader(handle))
                counts = {task: 0 for task in expected_counts}
                keys = set()
                for record in records:
                    task = record["task"]
                    key = (task, record["name"])
                    if task not in counts or key in keys:
                        raise RuntimeError("unexpected or duplicate paired row")
                    keys.add(key)
                    counts[task] += 1
                    if not all(
                        math.isfinite(float(record[field]))
                        for field in ("psnr", "ssim")
                    ):
                        raise RuntimeError("non-finite paired metric")
                if counts != expected_counts:
                    raise RuntimeError(
                        f"paired task counts mismatch: {counts} != {expected_counts}"
                    )
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                paired_errors.append({
                    "epoch": row.get("epoch"), "step": row.get("step"),
                    "error": repr(error),
                })
        paired_rows_valid = not paired_errors and bool(metric_rows)
    top3_keys = [
        (int(row.get("epoch", -1)), int(row.get("step", -1)))
        for row in top3
        if isinstance(row, dict)
    ]
    top3_names = [
        row.get("checkpoint") for row in top3 if isinstance(row, dict)
    ]
    top3_shape = isinstance(top3, list) and 1 <= len(top3) <= 3
    top3_unique = (
        top3_shape
        and len(top3_keys) == len(top3)
        and len(top3_names) == len(top3)
        and len(top3_keys) == len(set(top3_keys))
        and len(top3_names) == len(set(top3_names))
    )
    top3_checkpoints_exist = top3_unique and all(
        isinstance(name, str) and (run_dir / name).is_file() for name in top3_names
    )
    top3_scores_valid = top3_unique
    top3_selection_valid = top3_unique and bool(metric_by_key)
    top3_score_errors = []
    top3_scores = []
    if top3_unique:
        for row in top3:
            key = (int(row["epoch"]), int(row["step"]))
            try:
                score = float(row["score"])
                metric_score = float(metric_by_key[key]["macro_psnr"])
                if (
                    not math.isfinite(score)
                    or not math.isclose(score, metric_score, rel_tol=0.0, abs_tol=1e-12)
                ):
                    raise RuntimeError("top3 score does not match locked metric")
                top3_scores.append(score)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                top3_score_errors.append({"key": key, "error": repr(error)})
        top3_scores_valid = (
            not top3_score_errors
            and all(
                top3_scores[index] >= top3_scores[index + 1]
                for index in range(len(top3_scores) - 1)
            )
        )
        if top3_scores_valid:
            selected_keys = set(top3_keys)
            outside = [
                float(row["macro_psnr"])
                for key, row in metric_by_key.items() if key not in selected_keys
            ]
            global_best = max(float(row["macro_psnr"]) for row in metric_rows)
            top3_selection_valid = (
                len(top3) == min(3, len(metric_rows))
                and math.isclose(top3_scores[0], global_best, rel_tol=0.0, abs_tol=1e-12)
                and (not outside or top3_scores[-1] >= max(outside) - 1e-12)
            )
        else:
            top3_selection_valid = False
    top3_payloads_valid = top3_checkpoints_exist
    top3_payload_errors = []
    if top3_checkpoints_exist:
        for row in top3:
            path = run_dir / row["checkpoint"]
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload_checks = {
                    "run_name": payload.get("args", {}).get("run_name") == run_name,
                    "stage": payload.get("args", {}).get("stage") == "a",
                    "epoch": int(payload.get("epoch", -1)) == int(row["epoch"]),
                    "step": int(payload.get("step", -1)) == int(row["step"]),
                    "epoch_boundary": int(payload.get("batch_in_epoch", -1)) == 0,
                    "validation_pending": payload.get("validation_pending") is None,
                }
                if expected_config_sha256 is not None:
                    payload_checks["config_sha256"] = (
                        payload.get("config_sha256") == expected_config_sha256
                    )
                if expected_split_sha256 is not None:
                    payload_checks["split_sha256"] = (
                        payload.get("split_manifest_sha256") == expected_split_sha256
                    )
                runtime = payload.get("distributed_runtime") or {}
                legacy_aio3_prefix = (
                    expected_protocol == "aio3" and int(row["epoch"]) <= 55
                )
                if expected_runtime is not None and not legacy_aio3_prefix:
                    for key, expected in expected_runtime.items():
                        payload_checks[f"runtime_{key}"] = runtime.get(key) == expected
                    payload_checks["rank_rng_width"] = (
                        len(payload.get("rng_by_rank", []))
                        == int(expected_runtime["world_size"])
                    )
                if expected_training_origin is not None:
                    payload_checks["training_origin"] = (
                        payload.get("training_origin") == expected_training_origin
                    )
                if expected_data_contract is not None:
                    payload_checks["data_contract"] = (
                        payload.get("data_contract") == expected_data_contract
                    )
                if expected_code_contract is not None:
                    payload_checks["code_contract"] = (
                        payload.get("code_contract") == expected_code_contract
                    )
                failed = sorted(key for key, value in payload_checks.items() if not value)
                if failed:
                    top3_payload_errors.append(
                        {"checkpoint": row["checkpoint"], "failed": failed}
                    )
            except (EOFError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
                top3_payload_errors.append(
                    {"checkpoint": row["checkpoint"], "error": repr(error)}
                )
            finally:
                if "payload" in locals():
                    del payload
        top3_payloads_valid = not top3_payload_errors
    checks = {
        "validation_artifacts_parse": parse_ok,
        "validation_metric_records_valid": metric_parse_valid,
        "final_locked_val_record_exists": exact_metric,
        "validation_boundaries_complete": validation_boundaries_complete,
        "validation_task_schema_valid": metric_task_schema_valid,
        "paired_rows_valid": paired_rows_valid,
        "top3_shape_valid": top3_shape,
        "top3_unique": top3_unique,
        "top3_scores_valid": top3_scores_valid,
        "top3_selection_valid": top3_selection_valid,
        "top3_checkpoints_exist": top3_checkpoints_exist,
        "top3_payload_provenance_valid": top3_payloads_valid,
    }
    details = {
        "metric_path": str(metric_path),
        "metric_records": len(metric_rows),
        "expected_validation_epochs": expected_epochs,
        "expected_metric_key": [expected_epoch, expected_step],
        "top3_path": str(index_path),
        "top3_records": len(top3) if isinstance(top3, list) else 0,
        "top3_checkpoints": top3_names,
        "top3_payload_errors": top3_payload_errors,
        "top3_score_errors": top3_score_errors,
        "paired_row_errors": paired_errors,
    }
    return checks, details


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_contract_integrity(payload: dict) -> tuple[dict[str, bool], dict]:
    """Verify split, current lists and materialization are one frozen dataset."""
    cfg = payload["config"]
    protocol = cfg["protocol"]
    split_path = Path(cfg["split_manifest"])
    list_root = Path(cfg["list_root"])
    materialization_path = ROOT / "artifacts/manifests" / f"{protocol}.json"
    try:
        split = json.loads(split_path.read_text())
        materialization = json.loads(materialization_path.read_text())
        current_lists = {
            str(path.relative_to(list_root)): sha256(path)
            for path in sorted(list_root.rglob("*.txt"))
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return {"data_contract_readable": False}, {"error": repr(error)}
    frozen_lists = split.get("list_sha256")
    expected_contract = {
        "protocol": protocol,
        "materialization_manifest": str(materialization_path.resolve()),
        "materialization_manifest_sha256": sha256(materialization_path),
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": sha256(split_path),
        "list_sha256": frozen_lists,
    }
    checkpoint_contract = payload.get("data_contract")
    registered_legacy_aio3 = (
        protocol == "aio3"
        and checkpoint_contract is None
        and (
            payload.get("training_origin") is None
            or str(payload.get("training_origin", "")).startswith("legacy_aio3")
        )
    )
    checks = {
        "data_contract_readable": True,
        "split_protocol_matches": split.get("protocol") == protocol,
        "materialization_protocol_matches": (
            materialization.get("protocol") == protocol
        ),
        "materialization_complete": (
            int(materialization.get("missing_entries", -1)) == 0
            and int(materialization.get("expected_entries", 0)) > 0
        ),
        "current_lists_match_locked_split": (
            isinstance(frozen_lists, dict)
            and bool(frozen_lists)
            and current_lists == frozen_lists
        ),
        "materialization_lists_match_locked_split": (
            materialization.get("list_sha256") == frozen_lists
        ),
        "checkpoint_data_contract_matches": (
            checkpoint_contract == expected_contract or registered_legacy_aio3
        ),
    }
    details = {
        "protocol": protocol,
        "split_manifest": str(split_path),
        "materialization_manifest": str(materialization_path),
        "list_file_count": len(current_lists),
        "checkpoint_contract_present": checkpoint_contract is not None,
        "registered_legacy_aio3": registered_legacy_aio3,
    }
    return checks, details


def code_contract_integrity(payload: dict) -> tuple[dict[str, bool], dict]:
    protocol = payload["config"]["protocol"]
    current = {relative: sha256(ROOT / relative) for relative in STAGE_A_CODE_FILES}
    checkpoint = payload.get("code_contract")
    registered_legacy_aio3 = protocol == "aio3" and checkpoint is None
    checks = {
        "checkpoint_code_contract_matches": (
            checkpoint == current or registered_legacy_aio3
        )
    }
    return checks, {
        "file_count": len(current),
        "checkpoint_contract_present": checkpoint is not None,
        "registered_legacy_aio3": registered_legacy_aio3,
    }


def load_if_ready(path: Path, minimum_step: int) -> tuple[dict, int, str] | None:
    """Load and fingerprint one immutable inode snapshot of ``path``.

    The trainer publishes checkpoints with ``os.replace``.  Keeping the file
    descriptor open while both deserializing and hashing guarantees that the
    payload, byte count, and digest refer to the same generation even if a
    newer ``last.pt`` is atomically published meanwhile.
    """
    if not path.is_file() or path.with_suffix(path.suffix + ".tmp").exists():
        return None
    try:
        with path.open("rb") as handle:
            checkpoint_bytes = os.fstat(handle.fileno()).st_size
            payload = torch.load(handle, map_location="cpu", weights_only=False)
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except (EOFError, OSError, RuntimeError):
        # A save is atomic, but tolerate unusual network/filesystem visibility.
        return None
    if int(payload.get("step", -1)) < minimum_step:
        return None
    return payload, checkpoint_bytes, digest.hexdigest()


def expected_r2r_lr(config: dict, epoch: int) -> float:
    """Return the public R2R epoch-wise warm-up/cosine learning rate."""
    peak = float(config["lr"])
    warmup_epochs = int(config["warmup_epochs"])
    warmup_start = float(config.get("warmup_start_lr", 1e-7))
    schedule_epochs = int(config.get("scheduler_max_epochs", int(config["epochs"]) + 30))
    eta_min = float(config.get("pretrain_eta_min", 0.0))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return warmup_start + epoch * (peak - warmup_start) / max(warmup_epochs - 1, 1)
    progress = (epoch - warmup_epochs) / max(schedule_epochs - warmup_epochs, 1)
    return eta_min + 0.5 * (peak - eta_min) * (1.0 + math.cos(math.pi * progress))


def learning_rate_integrity(payload: dict) -> tuple[dict[str, bool], dict]:
    """Cross-check optimizer, scheduler state, checkpoint epoch, and R2R LR."""
    epoch = int(payload["epoch"])
    optimizer_lrs = [float(group["lr"]) for group in payload["optimizer"].get("param_groups", [])]
    scheduler = payload["scheduler"]
    scheduler_lrs = [float(value) for value in scheduler.get("_last_lr", [])]
    expected = expected_r2r_lr(payload["config"], epoch)
    tolerance = max(1e-15, abs(expected) * 1e-12)
    same_width = bool(optimizer_lrs) and len(optimizer_lrs) == len(scheduler_lrs)
    optimizer_matches_scheduler = same_width and all(
        abs(actual - recorded) <= tolerance
        for actual, recorded in zip(optimizer_lrs, scheduler_lrs)
    )
    optimizer_matches_expected = bool(optimizer_lrs) and all(
        abs(actual - expected) <= tolerance for actual in optimizer_lrs
    )
    checks = {
        "scheduler_epoch_matches_checkpoint": int(scheduler.get("last_epoch", -1)) == epoch,
        "optimizer_lr_matches_scheduler": optimizer_matches_scheduler,
        "optimizer_lr_matches_r2r_closed_form": optimizer_matches_expected,
    }
    details = {
        "checkpoint_epoch": epoch,
        "scheduler_last_epoch": int(scheduler.get("last_epoch", -1)),
        "optimizer_lrs": optimizer_lrs,
        "scheduler_last_lrs": scheduler_lrs,
        "expected_r2r_lr": expected,
        "absolute_tolerance": tolerance,
    }
    return checks, details


def main() -> int:
    args = parse_args()
    if args.minimum_step <= 0 or args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("minimum-step, poll-seconds, and timeout-hours must be positive")
    deadline = time.monotonic() + args.timeout_hours * 3600
    snapshot = None
    while snapshot is None:
        snapshot = load_if_ready(args.checkpoint, args.minimum_step)
        if snapshot is not None:
            break
        if time.monotonic() >= deadline:
            print(json.dumps({"status": "timeout", "minimum_step": args.minimum_step}), flush=True)
            return 3
        time.sleep(args.poll_seconds)

    payload, checkpoint_bytes, checkpoint_sha256 = snapshot

    required = {"model", "optimizer", "scheduler", "epoch", "batch_in_epoch", "step", "config", "rng"}
    missing = sorted(required.difference(payload))
    config_sha = sha256(args.config)
    split_path = Path(payload["config"]["split_manifest"])
    split_sha = sha256(split_path)
    rng = payload["rng"]
    lr_checks, lr_details = learning_rate_integrity(payload)
    data_checks, data_details = data_contract_integrity(payload)
    code_checks, code_details = code_contract_integrity(payload)
    expected_epoch = args.expected_epoch
    expected_step = args.expected_step
    runtime = payload.get("distributed_runtime") or {}
    checks = {
        "required_keys_present": not missing,
        "model_nonempty": bool(payload["model"]),
        "optimizer_nonempty": bool(payload["optimizer"]),
        "scheduler_nonempty": bool(payload["scheduler"]),
        "rng_complete": all(key in rng for key in ("torch", "cuda", "numpy", "python")),
        "config_sha256_matches": payload.get("config_sha256") == config_sha,
        "split_manifest_sha256_matches": payload.get("split_manifest_sha256") == split_sha,
        "stage_is_a": payload.get("args", {}).get("stage") == "a",
        "feedback_is_O7": payload.get("args", {}).get("feedback") == "O7",
        "run_name_matches": payload.get("args", {}).get("run_name") == args.expected_run_name,
        "minimum_step_reached": int(payload["step"]) >= args.minimum_step,
        **data_checks,
        **code_checks,
        **lr_checks,
    }
    if expected_epoch is not None:
        checks["expected_epoch_matches"] = int(payload["epoch"]) == expected_epoch
    if expected_step is not None:
        checks["expected_step_matches"] = int(payload["step"]) == expected_step
    if args.expected_world_size is not None:
        checks["expected_world_size_matches"] = (
            int(runtime.get("world_size", -1)) == args.expected_world_size
        )
        checks["rank_rng_width_matches"] = (
            len(payload.get("rng_by_rank", [])) == args.expected_world_size
        )
    if args.expected_global_effective_batch is not None:
        checks["expected_global_effective_batch_matches"] = (
            int(runtime.get("global_effective_batch", -1))
            == args.expected_global_effective_batch
        )
    if args.expected_per_gpu_batch is not None:
        checks["expected_per_gpu_batch_matches"] = (
            int(runtime.get("per_gpu_batch", -1)) == args.expected_per_gpu_batch
        )
    if args.expected_accumulation is not None:
        checks["expected_accumulation_matches"] = (
            int(runtime.get("accumulation", -1)) == args.expected_accumulation
        )
    if args.expected_workers_per_rank is not None:
        checks["expected_workers_per_rank_matches"] = (
            int(runtime.get("workers_per_rank", -1))
            == args.expected_workers_per_rank
        )
    if args.expected_backend is not None:
        checks["expected_backend_matches"] = (
            runtime.get("backend") == args.expected_backend
        )
    if args.expected_training_origin is not None:
        checks["expected_training_origin_matches"] = (
            payload.get("training_origin") == args.expected_training_origin
        )
    validation_details = None
    if args.require_validation_complete:
        checks["validation_pending_clear"] = payload.get("validation_pending") is None
        checks["epoch_boundary_checkpoint"] = int(payload["batch_in_epoch"]) == 0
        expected_runtime = {
            key: value for key, value in {
                "world_size": args.expected_world_size,
                "per_gpu_batch": args.expected_per_gpu_batch,
                "accumulation": args.expected_accumulation,
                "workers_per_rank": args.expected_workers_per_rank,
                "global_effective_batch": args.expected_global_effective_batch,
                "backend": args.expected_backend,
            }.items() if value is not None
        }
        protocol = payload["config"]["protocol"]
        artifact_checks, validation_details = validation_artifact_integrity(
            args.expected_run_name,
            int(payload["epoch"] if expected_epoch is None else expected_epoch),
            int(payload["step"] if expected_step is None else expected_step),
            expected_config_sha256=config_sha,
            expected_split_sha256=split_sha,
            expected_protocol=protocol,
            expected_validation_interval=int(
                payload["config"].get("validate_every_epochs", 5)
            ),
            expected_runtime=expected_runtime or None,
            expected_training_origin=args.expected_training_origin,
            expected_data_contract=payload.get("data_contract"),
            expected_code_contract=payload.get("code_contract"),
            require_paired_rows=(protocol == "aio5"),
        )
        checks.update(artifact_checks)
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": int(payload["epoch"]),
        "batch_in_epoch": int(payload["batch_in_epoch"]),
        "step": int(payload["step"]),
        "missing_keys": missing,
        "checks": checks,
        "learning_rate": lr_details,
        "data_contract": data_details,
        "code_contract": code_details,
        "distributed_runtime": runtime,
        "training_origin": payload.get("training_origin"),
        "validation_artifacts": validation_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
