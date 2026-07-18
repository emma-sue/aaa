#!/usr/bin/env python3
"""Strict post-Stage-A locked-validation candidate reassessment.

This entry point is intentionally independent from training and from the
official-test evaluator.  It replays the *current* locked-validation evaluator
over exactly the three immutable ``top3.json`` entries plus terminal
``last.pt`` (deduplicated by epoch/step after model-state equality is proven).

The historical PSNR ledger and top-3 index are read-only evidence.  New
per-image PSNR/SSIM rows, summaries, and the selection attestation are committed
under a separate transaction directory.  A stable staging directory makes the
operation resumable without ever editing or backfilling the historical ledger.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    EXPECTED_VALIDATION_TASKS,
    build_model,
    validate_locked,
)
from src.data import build_locked_val  # noqa: E402


SCHEMA = "srsc.stage_a.locked_candidate_reassessment.v1"
ATTESTATION_SCHEMA = "srsc.stage_a.locked_selection_attestation.v1"
LEGACY_RUNTIME_ATTESTATION_SCHEMA = (
    "srsc.legacy_stage_a.live_runtime_attestation.v1"
)
LEGACY_SCIENTIFIC_LIMIT = "STATE_EXACT_LEGACY_CODE_DATA_UNPROVEN"
HISTORY_ABS_TOLERANCE = 1e-5
SUMMARY_ABS_TOLERANCE = 1e-12
FORBIDDEN_RUN_TOKENS = ("official", "test", "smoke", "debug", "invalid", "tmp")
EVALUATOR_CLOSURE_FILES = (
    "scripts/reassess_stage_a_candidates.py",
    "scripts/train.py",
    "src/data/__init__.py",
    "src/data/aio_dataset.py",
    "src/net/__init__.py",
    "src/net/clean_restormer_aio.py",
    "src/net/feedback_controls.py",
    "src/net/restormer_blocks.py",
    "src/net/srsc_coordinates.py",
    "src/net/srsc_lite.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reassess completed Stage-A top3 + terminal last on locked_val only"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--runtime-attestation",
        required=True,
        type=Path,
        help=(
            "required read-only live-runtime evidence for the legacy AIO-3 "
            "lineage; never treated as launch-time code/data proof"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "optional transaction destination below "
            "artifacts/metrics/stage_a_reassessment"
        ),
    )
    return parser.parse_args()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n")


def read_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be an existing non-symlink file: {path}")
    return path


def resolve_under(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    allowed = parent.resolve()
    if not is_relative_to(resolved, allowed):
        raise PermissionError(f"{label} is outside the allowed root: {resolved}")
    return resolved


def validate_run_name(run_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("run name contains unsafe characters")
    lowered = run_name.casefold()
    if any(token in lowered for token in FORBIDDEN_RUN_TOKENS):
        raise PermissionError(
            "Stage-A reassessment run name targets a forbidden/non-scientific scope"
        )


def validate_locked_only_paths(
    *, config_path: Path, output_dir: Path, cfg: dict, protocol: str
) -> None:
    """Make an official-test data or output path unreachable from this CLI."""
    if cfg.get("official_test_locked") is not True:
        raise PermissionError("reassessment requires official_test_locked=true")
    if protocol not in EXPECTED_VALIDATION_TASKS:
        raise ValueError(f"unsupported locked-validation protocol: {protocol!r}")
    expected_config_root = (ROOT / "configs").resolve()
    if not is_relative_to(config_path.resolve(), expected_config_root):
        raise PermissionError("config must be under the project configs directory")
    expected_output_root = (
        ROOT / "artifacts/metrics/stage_a_reassessment"
    ).resolve()
    if not is_relative_to(output_dir.resolve(), expected_output_root):
        raise PermissionError("output must stay inside stage_a_reassessment")
    data_root = Path(cfg.get("data_root", "")).resolve()
    if data_root != (ROOT / "data").resolve():
        raise PermissionError("reassessment accepts only the registered project data root")
    list_root = Path(cfg.get("list_root", "")).resolve()
    if not is_relative_to(list_root, ROOT.resolve()):
        raise PermissionError("registered list root is outside the project")
    split = Path(cfg.get("split_manifest", "")).resolve()
    expected_split = (
        ROOT / "artifacts/manifests" / f"locked_split_{protocol}.json"
    ).resolve()
    if split != expected_split:
        raise PermissionError("only the preregistered locked-validation split is accepted")
    for path in (config_path.resolve(), output_dir.resolve(), split):
        if any("official" in part.casefold() for part in path.parts):
            raise PermissionError("official-test paths are forbidden")


def acquire_nonblocking_lock(path: Path, conflict_message: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(conflict_message) from error
    return descriptor


def argument_value(tokens: list[str], flag: str) -> str | None:
    prefixed = [token for token in tokens if token.startswith(flag + "=")]
    if len(prefixed) > 1 or (prefixed and flag in tokens):
        return None
    if prefixed:
        return prefixed[0].split("=", 1)[1]
    if tokens.count(flag) != 1:
        return None
    index = tokens.index(flag)
    return tokens[index + 1] if index + 1 < len(tokens) else None


def assert_no_active_trainer(run_name: str, proc_root: Path = Path("/proc")) -> None:
    trainer_names = {
        "train.py",
        "train_stage_a_ddp.py",
        "train_stage_a_capacity_hybrid_ddp.py",
    }
    active: list[int] = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            tokens = [
                part.decode("utf-8", errors="strict")
                for part in (process / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
            continue
        if not any(Path(token).name in trainer_names for token in tokens):
            continue
        if argument_value(tokens, "--run-name") == run_name:
            active.append(int(process.name))
    if active:
        raise RuntimeError(
            f"Stage-A trainer is still active for {run_name}: pids={sorted(active)}"
        )


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint model state is absent or empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_snapshot(path: Path) -> tuple[dict, dict]:
    """Load and hash one open checkpoint generation, including model state."""
    require_regular_file(path, "checkpoint")
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        payload = torch.load(handle, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint payload must be a mapping: {path}")
        state_sha = model_state_sha256(payload.get("model"))
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return payload, {
        "path": str(path.resolve()),
        "bytes": int(size),
        "sha256": digest.hexdigest(),
        "model_state_sha256": state_sha,
    }


def evaluator_closure() -> dict:
    files = {}
    for relative in EVALUATOR_CLOSURE_FILES:
        path = require_regular_file(ROOT / relative, "evaluator closure file")
        files[relative] = sha256_file(path)
    environment = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "numpy": str(np.__version__),
        "scikit_image": importlib.metadata.version("scikit-image"),
        "torchvision": importlib.metadata.version("torchvision"),
        "pillow": importlib.metadata.version("pillow"),
        "visible_cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    closure_payload = {"files": files, "environment": environment}
    return {
        **closure_payload,
        "sha256": sha256_json(closure_payload),
    }


def current_dataset_list_sha256(list_root: Path) -> dict[str, str]:
    if list_root.is_symlink() or not list_root.is_dir():
        raise RuntimeError("registered list root must be a non-symlink directory")
    result: dict[str, str] = {}
    for path in sorted(list_root.rglob("*.txt")):
        require_regular_file(path, "dataset list")
        relative = path.relative_to(list_root).as_posix()
        result[relative] = sha256_file(path)
    if not result:
        raise RuntimeError("registered dataset-list set is empty")
    return result


def validate_dataset_binding(cfg: dict, split_path: Path, protocol: str) -> dict:
    split_payload = read_json_object(split_path, "locked split manifest")
    frozen = split_payload.get("list_sha256")
    if (
        split_payload.get("protocol") != protocol
        or not isinstance(frozen, dict)
        or not frozen
        or any(
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not is_sha256(digest)
            for relative, digest in frozen.items()
        )
    ):
        raise RuntimeError("locked split has no valid frozen list_sha256 set")
    list_root = Path(cfg["list_root"]).resolve()
    actual = current_dataset_list_sha256(list_root)
    if actual != frozen:
        raise RuntimeError("current dataset-list files differ from the locked split")
    list_payload = {
        "root": str(list_root),
        "relative_paths": sorted(actual),
        "count": len(actual),
        "per_file_sha256": actual,
    }
    list_payload["canonical_sha256"] = sha256_json({
        "relative_paths": list_payload["relative_paths"],
        "per_file_sha256": actual,
    })

    materialization_path = require_regular_file(
        ROOT / "artifacts/manifests" / f"{protocol}.json",
        "data materialization manifest",
    )
    materialization = read_json_object(
        materialization_path, "data materialization manifest"
    )
    if (
        materialization.get("protocol") != protocol
        or int(materialization.get("missing_entries", -1)) != 0
        or int(materialization.get("expected_entries", 0)) <= 0
        or materialization.get("list_sha256") != frozen
    ):
        raise RuntimeError(
            "data materialization manifest is not complete/bound to the locked lists"
        )
    return {
        "dataset_lists": list_payload,
        "materialization": {
            "path": str(materialization_path.resolve()),
            "sha256": sha256_file(materialization_path),
            "protocol": protocol,
            "missing_entries": 0,
            "expected_entries": int(materialization["expected_entries"]),
            "list_set_canonical_sha256": list_payload["canonical_sha256"],
        },
    }


def read_historical_ledger(path: Path) -> tuple[bytes, list[dict]]:
    raw = require_regular_file(path, "historical locked-validation ledger").read_bytes()
    rows: list[dict] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid historical ledger JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"historical ledger line {line_number} is not an object")
        row = dict(row)
        row["_historical_line_number"] = line_number
        rows.append(row)
    if not rows:
        raise RuntimeError("historical locked-validation ledger is empty")
    return raw, rows


def validate_runtime_attestation(path: Path, run_name: str, protocol: str) -> dict:
    path = require_regular_file(
        resolve_under(path, ROOT / "artifacts/manifests", "runtime attestation"),
        "runtime attestation",
    )
    if any("official" in part.casefold() for part in path.parts):
        raise PermissionError("official-test paths are forbidden")
    payload = read_json_object(path, "runtime attestation")
    if protocol != "aio3":
        raise RuntimeError(
            "the required legacy runtime-attestation path is registered only for AIO-3"
        )
    if (
        payload.get("schema") != LEGACY_RUNTIME_ATTESTATION_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("run_name") != run_name
    ):
        raise RuntimeError("legacy runtime attestation schema/status/run mismatch")
    observed = payload.get("observed_runtime")
    expected = {
        "world_size": 4,
        "per_gpu_batch": 30,
        "accumulation": 1,
        "global_effective_batch": 120,
        "backend": "nccl",
        "workers_per_rank": 8,
        "workers_evidence": "live_process_cmdline",
    }
    if observed != expected:
        raise RuntimeError(
            f"legacy runtime attestation observed-runtime mismatch: {observed!r}"
        )
    scientific_limit = payload.get("scientific_limit")
    if scientific_limit != LEGACY_SCIENTIFIC_LIMIT:
        raise RuntimeError("legacy runtime attestation lost its scientific limit")
    owners = payload.get("cuda_owners")
    if not isinstance(owners, list) or len(owners) != 4:
        raise RuntimeError("legacy runtime attestation requires four CUDA owners")
    ranks: set[int] = set()
    local_ranks: set[int] = set()
    pids: set[int] = set()
    owner_uuids: set[str] = set()
    for owner in owners:
        if not isinstance(owner, dict):
            raise RuntimeError("legacy CUDA owner must be an object")
        try:
            pid = int(owner["pid"])
            start_ticks = int(owner["start_ticks_since_boot"])
            used_memory = int(owner["used_memory_mib"])
            gpu_uuid = owner["gpu_uuid"]
            cmdline = owner["cmdline"]
            environment = owner["environment"]
            rank = int(environment["RANK"])
            local_rank = int(environment["LOCAL_RANK"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("legacy CUDA owner structure is invalid") from error
        if (
            pid <= 0
            or start_ticks <= 0
            or used_memory <= 0
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            or not isinstance(cmdline, list)
            or not all(isinstance(token, str) for token in cmdline)
            or not isinstance(environment, dict)
            or not any(Path(token).name == "train_stage_a_ddp.py" for token in cmdline)
        ):
            raise RuntimeError("legacy CUDA owner values are invalid")
        expected_arguments = {
            "--config": "configs/protocol_aio3.yaml",
            "--run-name": run_name,
            "--per-gpu-batch": str(observed["per_gpu_batch"]),
            "--accumulation": str(observed["accumulation"]),
            "--workers-per-rank": str(observed["workers_per_rank"]),
        }
        if any(argument_value(cmdline, flag) != value for flag, value in expected_arguments.items()):
            raise RuntimeError("legacy CUDA owner cmdline differs from observed runtime")
        if environment.get("WORLD_SIZE") != str(observed["world_size"]):
            raise RuntimeError("legacy CUDA owner WORLD_SIZE mismatch")
        ranks.add(rank)
        local_ranks.add(local_rank)
        pids.add(pid)
        owner_uuids.add(gpu_uuid)
    if (
        ranks != {0, 1, 2, 3}
        or local_ranks != {0, 1, 2, 3}
        or len(pids) != 4
        or len(owner_uuids) != 4
    ):
        raise RuntimeError("legacy CUDA owner rank/PID/GPU uniqueness mismatch")

    inventory = payload.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise RuntimeError("legacy runtime attestation requires four inventory GPUs")
    try:
        inventory_indices = {int(row["index"]) for row in inventory}
        inventory_uuids = {row["uuid"] for row in inventory}
        inventory_valid = all(
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and row["name"]
            and int(row.get("memory_total_mib", 0)) > 0
            for row in inventory
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("legacy GPU inventory structure is invalid") from error
    if (
        not inventory_valid
        or inventory_indices != {0, 1, 2, 3}
        or inventory_uuids != owner_uuids
    ):
        raise RuntimeError("legacy GPU inventory does not match CUDA owners")

    checkpoint = payload.get("checkpoint_observation")
    embedded_runtime = {
        key: observed[key]
        for key in (
            "world_size", "per_gpu_batch", "accumulation",
            "global_effective_batch", "backend",
        )
    }
    if not isinstance(checkpoint, dict):
        raise RuntimeError("legacy checkpoint observation is absent")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (
        not is_sha256(checkpoint.get("sha256"))
        or not is_sha256(checkpoint.get("config_sha256"))
        or not is_sha256(checkpoint.get("split_manifest_sha256"))
        or int(checkpoint.get("bytes", 0)) <= 0
        or int(checkpoint.get("step", 0)) <= 0
        or int(checkpoint.get("epoch", -1)) < 0
        or int(checkpoint.get("rank_rng_width", 0)) != 4
        or checkpoint.get("embedded_runtime") != embedded_runtime
        or "workers_per_rank" in checkpoint.get("embedded_runtime", {})
        or checkpoint.get("embedded_workers_per_rank") is not None
        or checkpoint.get("embedded_code_contract") is not None
        or checkpoint.get("embedded_data_contract") is not None
        or checkpoint_path.name != "last.pt"
        or run_name not in checkpoint_path.parts
        or any("official" in part.casefold() for part in checkpoint_path.parts)
    ):
        raise RuntimeError("legacy checkpoint observation structure is invalid")
    registered_config = ROOT / "configs/protocol_aio3.yaml"
    registered_split = ROOT / "artifacts/manifests/locked_split_aio3.json"
    if (
        checkpoint["config_sha256"] != sha256_file(registered_config)
        or checkpoint["split_manifest_sha256"] != sha256_file(registered_split)
    ):
        raise RuntimeError("legacy checkpoint observation config/split SHA mismatch")

    launcher = payload.get("launcher")
    if not isinstance(launcher, dict):
        raise RuntimeError("legacy launcher evidence is absent")
    launcher_cmdline = launcher.get("cmdline")
    if (
        int(launcher.get("pid", 0)) <= 0
        or int(launcher.get("start_ticks_since_boot", 0)) <= 0
        or not isinstance(launcher_cmdline, list)
        or not any(
            Path(token).name == "launch_aio3_stage_a_4x4090.sh"
            for token in launcher_cmdline if isinstance(token, str)
        )
    ):
        raise RuntimeError("legacy launcher evidence structure is invalid")
    boot_id = payload.get("boot_id")
    claim = payload.get("claim")
    evidence_files = payload.get("evidence_files")
    if (
        not isinstance(boot_id, str)
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id) is None
        or not isinstance(claim, str)
        or "does not backfill" not in claim
        or "launch-time code/data" not in claim
        or not isinstance(evidence_files, dict)
        or not evidence_files
    ):
        raise RuntimeError("legacy boot/claim/evidence structure is invalid")
    for relative, evidence in evidence_files.items():
        if (
            not isinstance(relative, str)
            or any("official" in part.casefold() for part in Path(relative).parts)
            or not isinstance(evidence, dict)
            or int(evidence.get("bytes_at_capture", 0)) <= 0
            or not is_sha256(evidence.get("sha256_at_capture"))
        ):
            raise RuntimeError("legacy evidence-file record is invalid")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "status": payload["status"],
        "run_name": payload["run_name"],
        "observed_runtime": observed,
        "scientific_limit": scientific_limit,
        "evidence_scope": (
            "externally_observed_runtime_only_not_launch_time_code_or_data_proof"
        ),
    }


def historical_rows_by_key(rows: list[dict], task_order: tuple[str, ...]) -> dict:
    indexed: dict[tuple[int, int], dict] = {}
    required = (*task_order, "macro_psnr", "epoch", "step")
    for row in rows:
        try:
            key = (int(row["epoch"]), int(row["step"]))
            values = [float(row[field]) for field in required[:-2]]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("historical ledger lacks required PSNR fields") from error
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"historical ledger contains non-finite PSNR: {key}")
        if key in indexed:
            raise RuntimeError(f"historical ledger has duplicate epoch/step: {key}")
        indexed[key] = row
    return indexed


def validate_checkpoint_payload(
    *, payload: dict, source: dict, cfg: dict, config_sha: str,
    split_sha: str, run_name: str, expected_key: tuple[int, int]
) -> dict:
    try:
        actual_key = (int(payload["epoch"]), int(payload["step"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("checkpoint lacks an integer epoch/step") from error
    if actual_key != expected_key:
        raise RuntimeError(
            f"checkpoint/index epoch-step mismatch: {actual_key} != {expected_key}"
        )
    args = payload.get("args")
    if not isinstance(args, dict) or args.get("stage") != "a":
        raise RuntimeError("candidate checkpoint is not Stage-A")
    if args.get("run_name") != run_name:
        raise RuntimeError("candidate checkpoint run-name mismatch")
    if payload.get("config") != cfg:
        raise RuntimeError("candidate embedded config differs from current config")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("candidate config SHA256 mismatch")
    if payload.get("split_manifest_sha256") != split_sha:
        raise RuntimeError("candidate split-manifest SHA256 mismatch")
    return {
        **source,
        "epoch": actual_key[0],
        "step": actual_key[1],
        "checkpoint_config_sha256": payload["config_sha256"],
        "checkpoint_split_manifest_sha256": payload["split_manifest_sha256"],
        "embedded_code_contract": payload.get("code_contract"),
        "embedded_data_contract": payload.get("data_contract"),
    }


def safe_checkpoint_path(run_dir: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RuntimeError(f"unsafe checkpoint name in top3 index: {name!r}")
    path = run_dir / name
    if path.resolve().parent != run_dir.resolve():
        raise RuntimeError("top3 checkpoint escapes the run directory")
    return require_regular_file(path, "top3 checkpoint")


def discover_candidates(
    *, run_dir: Path, top3_path: Path, ledger_rows: list[dict], cfg: dict,
    config_sha: str, split_sha: str, run_name: str
) -> list[dict]:
    try:
        top3 = json.loads(require_regular_file(top3_path, "top3 index").read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError("top3 index is invalid JSON") from error
    if not isinstance(top3, list) or len(top3) != 3:
        raise RuntimeError("completed Stage-A requires exactly three top3 entries")
    task_order = EXPECTED_VALIDATION_TASKS[cfg["protocol"]]
    historical = historical_rows_by_key(ledger_rows, task_order)
    seen_keys: set[tuple[int, int]] = set()
    seen_names: set[str] = set()
    previous_score = math.inf
    sources: list[tuple[str, Path, tuple[int, int], float | None]] = []
    for rank, row in enumerate(top3, start=1):
        if not isinstance(row, dict):
            raise RuntimeError("top3 entries must be JSON objects")
        try:
            key = (int(row["epoch"]), int(row["step"]))
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("top3 entry has invalid score/epoch/step") from error
        name = row.get("checkpoint")
        path = safe_checkpoint_path(run_dir, name)
        if key in seen_keys or name in seen_names:
            raise RuntimeError("top3 entries are not unique")
        if not math.isfinite(score) or score > previous_score + SUMMARY_ABS_TOLERANCE:
            raise RuntimeError("top3 scores are non-finite or not descending")
        previous_score = score
        seen_keys.add(key)
        seen_names.add(name)
        history = historical.get(key)
        if history is None:
            raise RuntimeError(f"top3 entry has no historical ledger record: {key}")
        if abs(score - float(history["macro_psnr"])) > SUMMARY_ABS_TOLERANCE:
            raise RuntimeError("top3 score differs from the historical ledger")
        sources.append((f"top3_rank_{rank}", path, key, score))

    outside_scores = [
        float(row["macro_psnr"])
        for key, row in historical.items()
        if key not in seen_keys
    ]
    if outside_scores and max(outside_scores) > previous_score + SUMMARY_ABS_TOLERANCE:
        raise RuntimeError("top3 index does not contain the historical top three scores")
    if float(top3[0]["score"]) < max(
        float(row["macro_psnr"]) for row in historical.values()
    ) - SUMMARY_ABS_TOLERANCE:
        raise RuntimeError("top3[0] is not the global historical PSNR winner")

    last_path = require_regular_file(run_dir / "last.pt", "terminal last checkpoint")
    last_payload, last_snapshot = checkpoint_snapshot(last_path)
    terminal_key = (int(last_payload.get("epoch", -1)), int(last_payload.get("step", -1)))
    if terminal_key[0] != int(cfg["epochs"]):
        raise RuntimeError(
            f"Stage-A is not terminal: last epoch={terminal_key[0]} expected={cfg['epochs']}"
        )
    if int(last_payload.get("batch_in_epoch", -1)) != 0:
        raise RuntimeError("terminal last checkpoint is not at an epoch boundary")
    if last_payload.get("validation_pending") is not None:
        raise RuntimeError("terminal Stage-A validation transaction is still pending")
    if terminal_key not in historical:
        raise RuntimeError("terminal last checkpoint has no historical validation record")
    last_source = validate_checkpoint_payload(
        payload=last_payload,
        source=last_snapshot,
        cfg=cfg,
        config_sha=config_sha,
        split_sha=split_sha,
        run_name=run_name,
        expected_key=terminal_key,
    )
    del last_payload

    candidates_by_key: dict[tuple[int, int], dict] = {}
    for role, path, key, top3_score in sources:
        payload, snapshot = checkpoint_snapshot(path)
        source = validate_checkpoint_payload(
            payload=payload,
            source=snapshot,
            cfg=cfg,
            config_sha=config_sha,
            split_sha=split_sha,
            run_name=run_name,
            expected_key=key,
        )
        del payload
        candidate = {
            "candidate_id": f"epoch{key[0]:03d}_step{key[1]:07d}",
            "epoch": key[0],
            "step": key[1],
            "historical_line_number": int(
                historical[key]["_historical_line_number"]
            ),
            "historical_psnr": {
                task: float(historical[key][task]) for task in task_order
            } | {"macro_psnr": float(historical[key]["macro_psnr"])},
            "top3_score": top3_score,
            "sources": [{"role": role, **source}],
            "primary_source": {"role": role, **source},
        }
        candidates_by_key[key] = candidate

    if terminal_key in candidates_by_key:
        candidate = candidates_by_key[terminal_key]
        if (
            candidate["primary_source"]["model_state_sha256"]
            != last_source["model_state_sha256"]
        ):
            raise RuntimeError(
                "terminal last and same-step top3 checkpoint have different model states"
            )
        candidate["sources"].append({"role": "terminal_last", **last_source})
    else:
        history = historical[terminal_key]
        candidates_by_key[terminal_key] = {
            "candidate_id": (
                f"epoch{terminal_key[0]:03d}_step{terminal_key[1]:07d}"
            ),
            "epoch": terminal_key[0],
            "step": terminal_key[1],
            "historical_line_number": int(history["_historical_line_number"]),
            "historical_psnr": {
                task: float(history[task]) for task in task_order
            } | {"macro_psnr": float(history["macro_psnr"])},
            "top3_score": None,
            "sources": [{"role": "terminal_last", **last_source}],
            "primary_source": {"role": "terminal_last", **last_source},
        }

    ordered = list(candidates_by_key.values())
    if len(ordered) not in {3, 4}:
        raise RuntimeError("top3 + terminal-last dedup produced an invalid candidate count")
    return ordered


def expected_locked_identities(dataset, task_order: tuple[str, ...]) -> tuple[set, dict]:
    identities: list[tuple[str, str]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        task = item.get("task")
        name = item.get("name")
        if not isinstance(task, str) or not isinstance(name, str) or not name:
            raise RuntimeError("locked-validation dataset lacks stable task/name identity")
        identities.append((task, name))
    if not identities or len(identities) != len(set(identities)):
        raise RuntimeError("locked-validation identities are empty or duplicated")
    counts = Counter(task for task, _ in identities)
    if set(counts) != set(task_order) or any(counts[task] <= 0 for task in task_order):
        raise RuntimeError(
            f"locked-validation task counts are invalid: {dict(sorted(counts.items()))}"
        )
    return set(identities), {task: int(counts[task]) for task in task_order}


def validate_rows_and_recompute(
    rows: list[dict], expected_identities: set[tuple[str, str]],
    task_order: tuple[str, ...]
) -> tuple[list[dict], dict]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("locked evaluator returned no paired rows")
    normalized: list[dict] = []
    identities = []
    by_task_psnr: dict[str, list[float]] = {task: [] for task in task_order}
    by_task_ssim: dict[str, list[float]] = {task: [] for task in task_order}
    for row in rows:
        try:
            task = row["task"]
            name = row["name"]
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("paired evaluator row has invalid fields") from error
        if task not in by_task_psnr or not isinstance(name, str) or not name:
            raise RuntimeError("paired evaluator row has an unknown task/name")
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise RuntimeError("paired evaluator row contains non-finite metrics")
        identities.append((task, name))
        by_task_psnr[task].append(psnr)
        by_task_ssim[task].append(ssim)
        normalized.append({"task": task, "name": name, "psnr": psnr, "ssim": ssim})
    if len(identities) != len(set(identities)):
        raise RuntimeError("paired evaluator rows contain duplicate task/name identities")
    if set(identities) != expected_identities:
        missing = sorted(expected_identities - set(identities))[:8]
        extra = sorted(set(identities) - expected_identities)[:8]
        raise RuntimeError(f"paired row identity mismatch: missing={missing} extra={extra}")
    setting_psnr = {
        task: float(np.asarray(by_task_psnr[task], dtype=np.float64).mean())
        for task in task_order
    }
    setting_ssim = {
        task: float(np.asarray(by_task_ssim[task], dtype=np.float64).mean())
        for task in task_order
    }
    summary = {
        **setting_psnr,
        "macro_psnr": float(np.mean(list(setting_psnr.values()))),
        "setting_ssim": setting_ssim,
        "five_setting_mean_ssim": float(np.mean(list(setting_ssim.values()))),
    }
    return normalized, summary


def compare_summaries(actual: dict, recomputed: dict, task_order: tuple[str, ...]) -> None:
    for field in (*task_order, "macro_psnr", "five_setting_mean_ssim"):
        try:
            delta = abs(float(actual[field]) - float(recomputed[field]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"locked evaluator summary lacks {field}") from error
        if not math.isfinite(delta) or delta > SUMMARY_ABS_TOLERANCE:
            raise RuntimeError(f"locked evaluator/row recomputation mismatch for {field}")
    actual_ssim = actual.get("setting_ssim")
    if not isinstance(actual_ssim, dict):
        raise RuntimeError("locked evaluator summary lacks setting_ssim")
    for task in task_order:
        delta = abs(float(actual_ssim[task]) - recomputed["setting_ssim"][task])
        if not math.isfinite(delta) or delta > SUMMARY_ABS_TOLERANCE:
            raise RuntimeError(f"locked evaluator/row SSIM mismatch for {task}")


def compare_history(
    reassessed: dict, historical: dict, task_order: tuple[str, ...]
) -> dict:
    deltas = {
        field: abs(float(reassessed[field]) - float(historical[field]))
        for field in (*task_order, "macro_psnr")
    }
    if any(not math.isfinite(delta) for delta in deltas.values()):
        raise RuntimeError("historical PSNR comparison is non-finite")
    if max(deltas.values()) > HISTORY_ABS_TOLERANCE:
        raise RuntimeError(
            "reassessed PSNR differs from the immutable historical ledger by "
            f"more than {HISTORY_ABS_TOLERANCE}: {deltas}"
        )
    return {
        "absolute_psnr_differences": deltas,
        "maximum_absolute_difference": max(deltas.values()),
        "required_abs_tolerance": HISTORY_ABS_TOLERANCE,
        "status": "PASS",
    }


def atomic_write_paired_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["task", "name", "psnr", "ssim"]
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def read_paired_csv(path: Path) -> list[dict]:
    require_regular_file(path, "paired reassessment CSV")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["task", "name", "psnr", "ssim"]:
            raise RuntimeError("paired reassessment CSV header mismatch")
        return list(reader)


def load_model_strict(candidate: dict, cfg: dict) -> torch.nn.Module:
    source = candidate["primary_source"]
    path = Path(source["path"])
    payload, observed = checkpoint_snapshot(path)
    for field in ("bytes", "sha256", "model_state_sha256"):
        if observed[field] != source[field]:
            raise RuntimeError(f"candidate checkpoint drift before evaluation: {field}")
    model = build_model(cfg, "a")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    return model


def move_model_to_cuda(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the audited locked evaluator")
    return model.cuda().eval()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def candidate_evidence_bindings(candidate: dict, binding: dict) -> dict:
    return {
        "checkpoint_sha256": candidate["primary_source"]["sha256"],
        "checkpoint_model_state_sha256": candidate["primary_source"][
            "model_state_sha256"
        ],
        "config_sha256": binding["config"]["sha256"],
        "split_manifest_sha256": binding["split_manifest"]["sha256"],
        "top3_sha256": binding["top3"]["sha256"],
        "historical_ledger_sha256": binding["historical_ledger"]["sha256"],
        "evaluator_closure_sha256": binding["evaluator_closure"]["sha256"],
        "runtime_attestation_sha256": binding["runtime_attestation"]["sha256"],
        "materialization_manifest_sha256": binding["materialization"]["sha256"],
        "dataset_list_set_canonical_sha256": binding["dataset_lists"][
            "canonical_sha256"
        ],
        "dataset_list_count": binding["dataset_lists"]["count"],
        "runtime_scientific_limit": binding["runtime_attestation"][
            "scientific_limit"
        ],
    }


def verify_candidate_artifacts(
    *, staging: Path, candidate: dict, binding: dict, binding_sha: str,
    expected_identities: set[tuple[str, str]], task_order: tuple[str, ...]
) -> dict | None:
    summary_path = staging / "summaries" / f"{candidate['candidate_id']}.json"
    csv_path = staging / "paired" / f"{candidate['candidate_id']}.csv"
    if not summary_path.exists():
        return None
    summary = read_json_object(summary_path, "candidate reassessment summary")
    if (
        summary.get("schema") != SCHEMA
        or summary.get("status") != "PASS"
        or summary.get("candidate_id") != candidate["candidate_id"]
        or summary.get("input_binding_sha256") != binding_sha
        or summary.get("source_checkpoint") != candidate["primary_source"]
        or summary.get("evidence_bindings") != candidate_evidence_bindings(
            candidate, binding
        )
    ):
        raise RuntimeError("existing candidate summary binding drift")
    if summary.get("paired_rows", {}).get("sha256") != sha256_file(csv_path):
        raise RuntimeError("existing candidate paired CSV SHA256 mismatch")
    csv_rows = read_paired_csv(csv_path)
    _, recomputed = validate_rows_and_recompute(
        csv_rows, expected_identities, task_order
    )
    compare_summaries(summary["reevaluation"], recomputed, task_order)
    compare_history(recomputed, candidate["historical_psnr"], task_order)
    if int(summary["paired_rows"]["count"]) != len(csv_rows):
        raise RuntimeError("existing candidate paired-row count mismatch")
    return summary


def evaluate_candidate(
    *, candidate: dict, cfg: dict, dataset, staging: Path, binding: dict,
    binding_sha: str, expected_identities: set[tuple[str, str]],
    expected_counts: dict[str, int], task_order: tuple[str, ...]
) -> dict:
    existing = verify_candidate_artifacts(
        staging=staging,
        candidate=candidate,
        binding=binding,
        binding_sha=binding_sha,
        expected_identities=expected_identities,
        task_order=task_order,
    )
    if existing is not None:
        return existing
    model = load_model_strict(candidate, cfg)
    try:
        model = move_model_to_cuda(model)
        summary, rows = validate_locked(
            model=model,
            dataset=dataset,
            stage="a",
            builder=None,
            feedback="O7",
            feedback_stats=None,
            protocol=cfg["protocol"],
            return_rows=True,
        )
    finally:
        del model
        clear_cuda_cache()
    normalized_rows, recomputed = validate_rows_and_recompute(
        rows, expected_identities, task_order
    )
    compare_summaries(summary, recomputed, task_order)
    historical_comparison = compare_history(
        recomputed, candidate["historical_psnr"], task_order
    )
    csv_path = staging / "paired" / f"{candidate['candidate_id']}.csv"
    csv_sha = atomic_write_paired_csv(csv_path, normalized_rows)
    # Prove the committed bytes, rather than only the in-memory rows, reproduce
    # the same summary and task counts.
    committed_rows = read_paired_csv(csv_path)
    _, committed_summary = validate_rows_and_recompute(
        committed_rows, expected_identities, task_order
    )
    compare_summaries(recomputed, committed_summary, task_order)
    actual_counts = Counter(row["task"] for row in committed_rows)
    if {task: actual_counts[task] for task in task_order} != expected_counts:
        raise RuntimeError("committed paired CSV task counts do not match locked_val")
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "locked_val",
        "official_test_accessed": False,
        "candidate_id": candidate["candidate_id"],
        "epoch": candidate["epoch"],
        "step": candidate["step"],
        "source_checkpoint": candidate["primary_source"],
        "source_aliases": candidate["sources"],
        "input_binding_sha256": binding_sha,
        "evidence_bindings": candidate_evidence_bindings(candidate, binding),
        "historical_psnr_evidence": {
            "ledger_line_number": candidate["historical_line_number"],
            "values": candidate["historical_psnr"],
            "note": "copied evidence only; the historical ledger was not edited",
        },
        "reevaluation": committed_summary,
        "historical_comparison": historical_comparison,
        "task_counts": expected_counts,
        "paired_rows": {
            "path": str((Path("paired") / csv_path.name).as_posix()),
            "sha256": csv_sha,
            "count": len(committed_rows),
        },
    }
    atomic_write_json(
        staging / "summaries" / f"{candidate['candidate_id']}.json", result
    )
    return result


def input_paths_still_match(binding: dict) -> None:
    for key in (
        "config",
        "split_manifest",
        "top3",
        "historical_ledger",
        "runtime_attestation",
        "materialization",
    ):
        record = binding[key]
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"immutable reassessment input drifted: {key}")
    actual_lists = current_dataset_list_sha256(
        Path(binding["dataset_lists"]["root"])
    )
    expected_lists = binding["dataset_lists"]
    actual_canonical_sha = sha256_json({
        "relative_paths": sorted(actual_lists),
        "per_file_sha256": actual_lists,
    })
    if (
        actual_lists != expected_lists["per_file_sha256"]
        or sorted(actual_lists) != expected_lists["relative_paths"]
        or len(actual_lists) != int(expected_lists["count"])
        or actual_canonical_sha != expected_lists["canonical_sha256"]
    ):
        raise RuntimeError("immutable dataset-list set drifted")
    for candidate in binding["candidates"]:
        for source in candidate["sources"]:
            if sha256_file(source["path"]) != source["sha256"]:
                raise RuntimeError(
                    f"immutable candidate checkpoint drifted: {source['path']}"
                )


def build_selection_attestation(
    *, results: list[dict], candidates: list[dict], binding: dict,
    binding_sha: str, task_order: tuple[str, ...]
) -> dict:
    ranked = sorted(
        results,
        key=lambda item: (-float(item["reevaluation"]["macro_psnr"]), item["candidate_id"]),
    )
    if len(ranked) >= 2 and abs(
        float(ranked[0]["reevaluation"]["macro_psnr"])
        - float(ranked[1]["reevaluation"]["macro_psnr"])
    ) <= SUMMARY_ABS_TOLERANCE:
        raise RuntimeError("reassessed candidate selection has an unresolved PSNR tie")
    selected = ranked[0]
    top3_rank_one = next(
        candidate for candidate in candidates
        if candidate["primary_source"]["role"] == "top3_rank_1"
    )
    if selected["candidate_id"] != top3_rank_one["candidate_id"]:
        raise RuntimeError(
            "reassessed locked-val winner differs from immutable historical top3[0]"
        )
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS",
        "scope": "locked_val",
        "official_test_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "macro_psnr",
        "selection_rule": (
            "highest five-setting locked-validation mean PSNR among exactly "
            "top3 plus terminal last, after epoch/step/model-state deduplication"
        ),
        "task_order": list(task_order),
        "input_binding_sha256": binding_sha,
        "evaluator_closure_sha256": binding["evaluator_closure"]["sha256"],
        "runtime_attestation": binding["runtime_attestation"],
        "selected": {
            "candidate_id": selected["candidate_id"],
            "epoch": selected["epoch"],
            "step": selected["step"],
            "source_checkpoint": selected["source_checkpoint"],
            "macro_psnr": selected["reevaluation"]["macro_psnr"],
            "five_setting_mean_ssim": selected["reevaluation"][
                "five_setting_mean_ssim"
            ],
        },
        "ranked_candidates": [
            {
                "rank": rank,
                "candidate_id": result["candidate_id"],
                "epoch": result["epoch"],
                "step": result["step"],
                "checkpoint_sha256": result["source_checkpoint"]["sha256"],
                "macro_psnr": result["reevaluation"]["macro_psnr"],
                "five_setting_mean_ssim": result["reevaluation"][
                    "five_setting_mean_ssim"
                ],
            }
            for rank, result in enumerate(ranked, start=1)
        ],
        "claim_limit": (
            "This selects a Stage-A coarse checkpoint on locked_val only; it "
            "is not an official-test result or evidence of SRSC efficacy."
        ),
    }


def verify_completed_output(output_dir: Path, binding: dict, binding_sha: str) -> dict:
    manifest = read_json_object(output_dir / "manifest.json", "reassessment manifest")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("input_binding_sha256") != binding_sha
        or manifest.get("input_binding") != binding
    ):
        raise RuntimeError("completed reassessment manifest binding drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("completed reassessment has no artifact index")
    for relative, expected_sha in artifacts.items():
        path = resolve_under(output_dir / relative, output_dir, "reassessment artifact")
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"completed reassessment artifact drift: {relative}")
    attestation = read_json_object(
        output_dir / "selection_attestation.json", "selection attestation"
    )
    if (
        attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("input_binding_sha256") != binding_sha
        or attestation.get("official_test_accessed") is not False
    ):
        raise RuntimeError("completed selection attestation is invalid")
    return manifest


def verify_completed_candidate_set(
    *, output_dir: Path, candidates: list[dict], binding: dict,
    binding_sha: str, expected_identities: set[tuple[str, str]],
    task_order: tuple[str, ...]
) -> list[dict]:
    results = []
    for candidate in candidates:
        result = verify_candidate_artifacts(
            staging=output_dir,
            candidate=candidate,
            binding=binding,
            binding_sha=binding_sha,
            expected_identities=expected_identities,
            task_order=task_order,
        )
        if result is None:
            raise RuntimeError(
                f"completed reassessment lacks candidate {candidate['candidate_id']}"
            )
        results.append(result)
    expected = build_selection_attestation(
        results=results,
        candidates=candidates,
        binding=binding,
        binding_sha=binding_sha,
        task_order=task_order,
    )
    stored = read_json_object(
        output_dir / "selection_attestation.json", "selection attestation"
    )
    for key in (
        "selection_metric",
        "selection_rule",
        "task_order",
        "input_binding_sha256",
        "evaluator_closure_sha256",
        "runtime_attestation",
        "selected",
        "ranked_candidates",
        "claim_limit",
    ):
        if stored.get(key) != expected.get(key):
            raise RuntimeError(f"completed selection attestation drift: {key}")
    return results


def run_reassessment(
    *, config_path: Path, run_name: str, runtime_attestation: Path,
    output_dir: Path | None = None
) -> dict:
    validate_run_name(run_name)
    config_path = require_regular_file(
        resolve_under(config_path, ROOT / "configs", "config"), "config"
    )
    try:
        cfg = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as error:
        raise RuntimeError(f"invalid configuration YAML: {config_path}") from error
    if not isinstance(cfg, dict):
        raise RuntimeError("configuration must be a mapping")
    protocol = cfg.get("protocol")
    default_output = (
        ROOT / "artifacts/metrics/stage_a_reassessment" / run_name
    )
    output_dir = output_dir or default_output
    validate_locked_only_paths(
        config_path=config_path,
        output_dir=output_dir,
        cfg=cfg,
        protocol=protocol,
    )
    output_dir = resolve_under(
        output_dir,
        ROOT / "artifacts/metrics/stage_a_reassessment",
        "output directory",
    )
    run_dir = resolve_under(
        ROOT / "artifacts/checkpoints" / run_name,
        ROOT / "artifacts/checkpoints",
        "checkpoint run directory",
    )
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise RuntimeError(f"checkpoint run directory is invalid: {run_dir}")
    top3_path = require_regular_file(run_dir / "top3.json", "top3 index")
    ledger_path = require_regular_file(
        ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl",
        "historical locked-validation ledger",
    )
    split_path = require_regular_file(
        Path(cfg["split_manifest"]), "locked split manifest"
    )
    dataset_binding = validate_dataset_binding(cfg, split_path, protocol)
    runtime_evidence = validate_runtime_attestation(
        runtime_attestation, run_name, protocol
    )

    gpu_lock_fd = acquire_nonblocking_lock(
        ROOT / ".srsc_gpu_pipeline.lock",
        "the shared SRSC GPU pipeline is already active",
    )
    transaction_lock = output_dir.with_name(output_dir.name + ".lock")
    try:
        lock_fd = acquire_nonblocking_lock(
            transaction_lock, "another Stage-A reassessment transaction is active"
        )
    except Exception:
        os.close(gpu_lock_fd)
        raise
    try:
        assert_no_active_trainer(run_name)
        # A trainer from a future entry point may honor this lock even if the
        # legacy DDP trainer did not.  The /proc check above remains mandatory.
        run_lock_fd = acquire_nonblocking_lock(
            run_dir / ".train.lock", "the Stage-A run lock is still held"
        )
        try:
            config_sha = sha256_file(config_path)
            split_sha = sha256_file(split_path)
            ledger_raw, ledger_rows = read_historical_ledger(ledger_path)
            candidates = discover_candidates(
                run_dir=run_dir,
                top3_path=top3_path,
                ledger_rows=ledger_rows,
                cfg=cfg,
                config_sha=config_sha,
                split_sha=split_sha,
                run_name=run_name,
            )
            closure = evaluator_closure()
            binding = {
                "schema": SCHEMA,
                "run_name": run_name,
                "protocol": protocol,
                "scope": "locked_val",
                "official_test_accessed": False,
                "config": {
                    "path": str(config_path),
                    "sha256": config_sha,
                    "embedded_effective_config": cfg,
                },
                "split_manifest": {
                    "path": str(split_path.resolve()),
                    "sha256": split_sha,
                },
                "dataset_lists": dataset_binding["dataset_lists"],
                "materialization": dataset_binding["materialization"],
                "top3": {
                    "path": str(top3_path.resolve()),
                    "sha256": sha256_file(top3_path),
                },
                "historical_ledger": {
                    "path": str(ledger_path.resolve()),
                    "sha256": sha256_bytes(ledger_raw),
                    "bytes": len(ledger_raw),
                    "record_count": len(ledger_rows),
                    "preservation_rule": "read_only_no_backfill_no_rewrite",
                },
                "evaluator_closure": closure,
                "runtime_attestation": runtime_evidence,
                "candidates": candidates,
            }
            binding_sha = sha256_json(binding)
            staging = output_dir.with_name(f".{output_dir.name}.staging")
            dataset = build_locked_val(
                cfg["data_root"], cfg["list_root"], protocol, cfg["split_manifest"]
            )
            task_order = EXPECTED_VALIDATION_TASKS[protocol]
            expected_identities, expected_counts = expected_locked_identities(
                dataset, task_order
            )
            if output_dir.exists():
                if staging.exists():
                    raise RuntimeError("both completed and staging reassessment exist")
                manifest = verify_completed_output(output_dir, binding, binding_sha)
                completed_results = verify_completed_candidate_set(
                    output_dir=output_dir,
                    candidates=candidates,
                    binding=binding,
                    binding_sha=binding_sha,
                    expected_identities=expected_identities,
                    task_order=task_order,
                )
                if len(completed_results) != manifest["candidate_count_after_dedup"]:
                    raise RuntimeError("completed manifest candidate-count drift")
                if manifest.get("task_counts") != expected_counts:
                    raise RuntimeError("completed manifest locked task-count drift")
                input_paths_still_match(binding)
                if ledger_path.read_bytes() != ledger_raw:
                    raise RuntimeError("historical ledger bytes changed during verification")
                return {**manifest, "idempotent_reuse": True}

            transaction = {
                "schema": SCHEMA,
                "status": "INPUTS_BOUND",
                "input_binding_sha256": binding_sha,
                "input_binding": binding,
            }
            if staging.exists():
                if not staging.is_dir() or staging.is_symlink():
                    raise RuntimeError("reassessment staging path is invalid")
                existing = read_json_object(
                    staging / "transaction.json", "reassessment transaction"
                )
                if existing != transaction:
                    raise RuntimeError("staging reassessment input binding drift")
            else:
                staging.mkdir(parents=True)
                atomic_write_json(staging / "transaction.json", transaction)

            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
            results = [
                evaluate_candidate(
                    candidate=candidate,
                    cfg=cfg,
                    dataset=dataset,
                    staging=staging,
                    binding=binding,
                    binding_sha=binding_sha,
                    expected_identities=expected_identities,
                    expected_counts=expected_counts,
                    task_order=task_order,
                )
                for candidate in candidates
            ]
            attestation = build_selection_attestation(
                results=results,
                candidates=candidates,
                binding=binding,
                binding_sha=binding_sha,
                task_order=task_order,
            )
            atomic_write_json(staging / "selection_attestation.json", attestation)
            artifacts: dict[str, str] = {
                "selection_attestation.json": sha256_file(
                    staging / "selection_attestation.json"
                ),
                "transaction.json": sha256_file(staging / "transaction.json"),
            }
            for result in results:
                summary_rel = f"summaries/{result['candidate_id']}.json"
                csv_rel = f"paired/{result['candidate_id']}.csv"
                artifacts[summary_rel] = sha256_file(staging / summary_rel)
                artifacts[csv_rel] = sha256_file(staging / csv_rel)
            manifest = {
                "schema": SCHEMA,
                "status": "PASS",
                "scope": "locked_val",
                "official_test_accessed": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_name": run_name,
                "protocol": protocol,
                "input_binding_sha256": binding_sha,
                "input_binding": binding,
                "candidate_count_after_dedup": len(candidates),
                "task_counts": expected_counts,
                "historical_ledger_preserved_verbatim": True,
                "selected_candidate_id": attestation["selected"]["candidate_id"],
                "artifacts": artifacts,
                "claim_limit": attestation["claim_limit"],
            }
            input_paths_still_match(binding)
            if ledger_path.read_bytes() != ledger_raw:
                raise RuntimeError("historical ledger bytes changed during reassessment")
            atomic_write_json(staging / "manifest.json", manifest)
            fsync_directory(staging)
            input_paths_still_match(binding)
            if ledger_path.read_bytes() != ledger_raw:
                raise RuntimeError("historical ledger bytes changed before commit")
            os.replace(staging, output_dir)
            fsync_directory(output_dir.parent)
            verified = verify_completed_output(output_dir, binding, binding_sha)
            verify_completed_candidate_set(
                output_dir=output_dir,
                candidates=candidates,
                binding=binding,
                binding_sha=binding_sha,
                expected_identities=expected_identities,
                task_order=task_order,
            )
            return {**verified, "idempotent_reuse": False}
        finally:
            os.close(run_lock_fd)
    finally:
        os.close(lock_fd)
        os.close(gpu_lock_fd)


def main() -> int:
    args = parse_args()
    manifest = run_reassessment(
        config_path=args.config,
        run_name=args.run_name,
        runtime_attestation=args.runtime_attestation,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": manifest["status"],
        "scope": manifest["scope"],
        "run_name": manifest["run_name"],
        "selected_candidate_id": manifest["selected_candidate_id"],
        "output": str(
            args.output_dir
            or ROOT / "artifacts/metrics/stage_a_reassessment" / args.run_name
        ),
        "idempotent_reuse": manifest["idempotent_reuse"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
