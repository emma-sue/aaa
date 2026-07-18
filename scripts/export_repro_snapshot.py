#!/usr/bin/env python3
"""Build a fail-closed, text-only Git mirror of the live SRSC workspace."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".md",
    ".txt", ".log", ".toml", ".cfg", ".ini", ".lock",
}
MAX_GIT_FILE_BYTES = 95 * 1024 * 1024
TOP_LEVEL_FILES = {
    ".gitignore", "LICENSE.md", "README.md", "GITHUB_BACKUP.md", "RUNNING_STATUS.md",
    "STOP_REASON.md", "DISCONNECT_RECOVERY.md",
}
TREE_ROOTS = ("src", "scripts", "configs", "tests", "reports", "environment", "recovery")
ARTIFACT_ROOTS = (
    "artifacts/manifests", "artifacts/metrics", "artifacts/stats", "artifacts/reference",
)
KEY_LOGS = {
    "pipeline.log", "pipeline_ddp.log", "aio3_stage_a_coarse_seed1415926.csv",
    "aio3_stage_a_coarse_seed1415926.log", "stage_a_trend_chain.log",
    "runtime_accounting_aio3_stage_a.log", "orchestrate_aio3_pytest.log",
    "promptir_official_parity.log", "promptir_official_parity_center_crop.log",
    "github_backup.log",
}
DOC_CONTRACTS = {
    Path("/root/aaa/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md"):
        Path("docs/contracts/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md"),
    Path("/root/aaa/v1.4.md"): Path("docs/contracts/v1.4.md"),
    Path("/root/aaa/ResearchStudio_SEC_SRSC_DOGC_交叉终审_Codex.md"):
        Path("docs/contracts/ResearchStudio_SEC_SRSC_DOGC_交叉终审_Codex.md"),
    Path("/root/aaa/ResearchStudio_DOGC_Codex_最终方案.md"):
        Path("docs/contracts/ResearchStudio_DOGC_Codex_最终方案.md"),
    Path("/root/ResearchStudio/ideaspark_run/end-to-end-restoration-state-feedback/srsc_lite_v1_2_reassessment.md"):
        Path("vendor/researchstudio/srsc_lite_v1_2_reassessment.md"),
}
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(
        rb"(?im)^\s*(?:password|passwd|access[_-]?token|auth[_-]?token|api[_-]?key|secret)"
        rb"\s*=\s*[\"']?[A-Za-z0-9_./+=-]{16,}[\"']?\s*$"
    ),
)
FORBIDDEN_NAME_PARTS = (".env", "credential", "private_key", "id_rsa", "id_ed25519")
LOCAL_CONTROL_FILES = {
    ".backup_daemon.lock",
    ".checkpoint_hash_cache.json",
    ".local_backup_state.json",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def safe_source_file(path: Path) -> bytes:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink: {path}")
    if not path.is_file():
        raise RuntimeError(f"Not a regular file: {path}")
    lower = path.name.lower()
    if any(part in lower for part in FORBIDDEN_NAME_PARTS):
        raise RuntimeError(f"Forbidden credential-like filename: {path}")
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
        raise RuntimeError(f"Refusing non-text artifact: {path}")
    size = path.stat().st_size
    if size > MAX_GIT_FILE_BYTES:
        raise RuntimeError(f"File exceeds ordinary Git limit ({size} bytes): {path}")
    data = path.read_bytes()
    if b"\x00" in data:
        raise RuntimeError(f"Refusing binary-looking file: {path}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise RuntimeError(f"Potential secret detected; snapshot aborted: {path}")
    return data


def iter_tree(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and ".ruff_cache" not in p.parts
        and p.suffix.lower() in TEXT_SUFFIXES
    )


def copy_checked(source: Path, destination: Path) -> dict[str, object]:
    data = safe_source_file(source)
    atomic_write(destination, data)
    mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
    os.chmod(destination, mode)
    return {"sha256": sha256_bytes(data), "size": len(data)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_checkpoint_record(path: Path, cache: dict[str, object]) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Checkpoint missing or unsafe: {path}")
    before = path.stat()
    key = str(path.resolve())
    cached = cache.get(key, {}) if isinstance(cache.get(key), dict) else {}
    if cached.get("size") == before.st_size and cached.get("mtime_ns") == before.st_mtime_ns:
        digest = str(cached["sha256"])
    else:
        digest = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Checkpoint changed while hashing; retry later: {path}")
    cached_contract = cached.get("checkpoint_contract")
    contract_cache_current = (
        isinstance(cached_contract, dict)
        and "resumable_training_state" in cached_contract
        and "split_manifest_path" in cached_contract
    )
    if cached.get("sha256") == digest and contract_cache_current:
        checkpoint_contract = cached_contract
    else:
        checkpoint_contract = checkpoint_payload_contract(path, after)
    cache[key] = {
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
        "checkpoint_contract": checkpoint_contract,
    }
    return {
        "path": str(path.relative_to(path.parents[3])),
        "absolute_source_path": str(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
        "checkpoint_contract": copy.deepcopy(checkpoint_contract),
        "embedded_checkpoint_contract": copy.deepcopy(checkpoint_contract),
    }


def _json_contract(value: object, field: str) -> object:
    """Accept only deterministic JSON metadata from a checkpoint payload."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Checkpoint {field} is not JSON-serializable") from error


def checkpoint_payload_contract(path: Path, expected_stat: os.stat_result) -> dict[str, object]:
    """Read provenance without retaining tensor storage in the backup index."""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as error:
        raise RuntimeError(f"Cannot read checkpoint provenance: {path}") from error
    after = path.stat()
    if (
        after.st_size != expected_stat.st_size
        or after.st_mtime_ns != expected_stat.st_mtime_ns
    ):
        raise RuntimeError(f"Checkpoint changed while reading provenance: {path}")
    runtime = payload.get("runtime_contract")
    distributed = payload.get("distributed_runtime")
    args = payload.get("args") or {}
    effective_config = payload.get("config") or {}
    if not isinstance(args, dict):
        raise RuntimeError(f"Checkpoint args are not a mapping: {path}")
    if not isinstance(effective_config, dict):
        raise RuntimeError(f"Checkpoint effective config is not a mapping: {path}")
    run_contract_sha256 = (
        payload.get("run_contract_sha256")
        or args.get("run_contract_sha256")
        or (runtime or {}).get("run_contract_sha256")
    )
    contract = {
        "schema_version": 1,
        "epoch": int(payload.get("epoch", -1)),
        "step": int(payload.get("step", -1)),
        "batch_in_epoch": int(payload.get("batch_in_epoch", -1)),
        "config_sha256": payload.get("config_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "training_origin": payload.get("training_origin"),
        "distributed_runtime": _json_contract(distributed, "distributed_runtime"),
        "runtime_contract": _json_contract(runtime, "runtime_contract"),
        "data_contract": _json_contract(payload.get("data_contract"), "data_contract"),
        "code_contract": _json_contract(payload.get("code_contract"), "code_contract"),
        "run_name": args.get("run_name"),
        "stage": args.get("stage"),
        "feedback": args.get("feedback"),
        "config_path": args.get("config"),
        "split_manifest_path": effective_config.get("split_manifest"),
        "run_contract_sha256": run_contract_sha256,
        "checkpoint_kind": payload.get("checkpoint_kind", "resumable_training_state"),
        "model_state_present": "model" in payload,
        "optimizer_state_present": "optimizer" in payload,
        "scheduler_state_present": "scheduler" in payload,
        "rng_state_present": "rng" in payload or "rng_by_rank" in payload,
    }
    contract["resumable_training_state"] = all(
        contract[key] for key in (
            "model_state_present", "optimizer_state_present",
            "scheduler_state_present", "rng_state_present",
        )
    )
    contract["completeness"] = {
        "runtime": "present" if runtime is not None or distributed is not None else "missing",
        "data": "present" if contract["data_contract"] is not None else "legacy_missing",
        "code": "present" if contract["code_contract"] is not None else "legacy_missing",
    }
    return contract


def purge_unexpected_destination_files(destination: Path, expected: set[str]) -> None:
    """Make the mirror an exact closed set instead of an additive copy tree."""
    preserve = expected | LOCAL_CONTROL_FILES | {"recovery/SNAPSHOT_MANIFEST.json"}
    candidates = sorted(
        destination.rglob("*"), key=lambda item: len(item.relative_to(destination).parts),
        reverse=True,
    )
    for path in candidates:
        relative = path.relative_to(destination)
        if relative.parts and relative.parts[0] == ".git":
            continue
        rendered = relative.as_posix()
        if path.is_symlink():
            path.unlink()
        elif path.is_file() and rendered not in preserve:
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


LEGACY_STAGE_A = re.compile(r"^aio(?P<count>[35])_stage_a_coarse(?:_10_10)?_seed\d+$")
SCIENTIFIC_NAME_TOKENS = ("_formal_", "_stage_c_", "_pretrain_", "_finetune_")
FORBIDDEN_RUN_TOKENS = ("official", "smoke", "debug", "invalid", "tmp", "test")
CONTROLLED_CHECKPOINT_METADATA = {
    "run_contract.json", "top3.json", "formal_complete.json", "pilot_complete.json",
}


def load_json_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON mapping: {path}")
    return payload


def stable_slug(value: str, limit: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"
    if len(slug) <= limit:
        return slug
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{slug[:limit - 11]}-{suffix}"


def run_contract_field(contract: dict[str, object], key: str) -> object:
    value = contract.get(key)
    args = contract.get("args")
    if value is None and isinstance(args, dict):
        value = args.get(key)
    return value


def controlled_run(run_dir: Path) -> tuple[str, dict[str, object] | None] | None:
    name = run_dir.name
    legacy = LEGACY_STAGE_A.fullmatch(name)
    if legacy:
        return "stage_a", None
    lowered = name.lower()
    unsafe_name = any(
        re.search(rf"(?:^|[_-]){re.escape(token)}(?:$|[_-])", lowered)
        for token in FORBIDDEN_RUN_TOKENS
    )
    if not lowered.startswith(("aio3_", "aio5_")) or unsafe_name:
        return None
    contract_path = run_dir / "run_contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        return None
    contract = load_json_mapping(contract_path)
    if run_contract_field(contract, "run_name") != name:
        raise RuntimeError(f"Run-contract name mismatch: {run_dir}")
    effective = contract.get("effective_config") or {}
    protocol = effective.get("protocol") if isinstance(effective, dict) else None
    protocol = protocol or name.split("_", 1)[0]
    if protocol not in {"aio3", "aio5"} or not name.startswith(protocol + "_"):
        raise RuntimeError(f"Run-contract protocol mismatch: {run_dir}")
    if "_pilot_" in lowered:
        if int(run_contract_field(contract, "max_steps") or 0) <= 0:
            raise RuntimeError(f"Pilot run has no bounded max_steps: {run_dir}")
        return "pilot", contract
    if any(token in lowered for token in SCIENTIFIC_NAME_TOKENS):
        return "formal", contract
    return None


def run_upload_state(upload_state: dict[str, object], run_name: str) -> dict[str, object]:
    runs = upload_state.get("runs")
    state = dict(runs.get(run_name, {})) if isinstance(runs, dict) else {}
    if run_name == "aio3_stage_a_coarse_seed1415926":
        for key in (
            "best_sha256", "best_sha256s", "best_uploaded_utc",
            "resume_sha256", "resume_upload_unix", "resume_uploaded_utc",
        ):
            if key in upload_state and key not in state:
                state[key] = upload_state[key]
    return state


def previous_run(previous_index: dict[str, object], run_name: str) -> dict[str, object]:
    runs = previous_index.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get(run_name), dict):
        record = dict(runs[run_name])
        if run_name == "aio3_stage_a_coarse_seed1415926":
            for key in ("resume_uploaded", "resume_latest", "resume_local_latest"):
                alias = previous_index.get(key)
                current = record.get(key)
                if (
                    isinstance(alias, dict)
                    and isinstance(alias.get("git_snapshot_commit"), str)
                    and isinstance(alias.get("git_snapshot_tree"), str)
                    and (
                        not isinstance(current, dict)
                        or not isinstance(current.get("git_snapshot_commit"), str)
                    )
                ):
                    record[key] = alias
        return record
    if run_name == "aio3_stage_a_coarse_seed1415926":
        return previous_index
    return {}


def relative_file_record(source: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Scientific contract path escapes project root: {path}") from error
    return {"path": relative, "sha256": file_sha256(resolved)}


def resolve_project_path(source: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else source / path


def enrich_from_run_contract(
    record: dict[str, object], run_contract: dict[str, object] | None,
    run_contract_sha256: str | None,
) -> None:
    if run_contract is None:
        return
    contract = record["checkpoint_contract"]
    if contract.get("run_contract_sha256") != run_contract_sha256:
        raise RuntimeError("Checkpoint does not bind its immutable run contract")
    code = run_contract.get("code_sha256")
    if contract.get("code_contract") is None and isinstance(code, dict) and code:
        contract["code_contract"] = code
        contract["completeness"]["code"] = "present_via_run_contract"
    effective = run_contract.get("effective_config") or {}
    run_data = run_contract.get("data_contract") or {}
    if not isinstance(effective, dict) or not isinstance(run_data, dict):
        raise RuntimeError("Run contract contains malformed config/data metadata")
    if contract.get("data_contract") is None:
        contract["data_contract"] = {
            "config_path": contract.get("config_path"),
            "config_sha256": contract.get("config_sha256"),
            "split_manifest_path": (
                effective.get("split_manifest")
                or run_data.get("split_manifest_path")
                or contract.get("split_manifest_path")
            ),
            "split_manifest_sha256": contract.get("split_manifest_sha256"),
            "coordinate_stats_path": effective.get("coordinate_stats"),
            "coordinate_stats_sha256": run_contract.get("coordinate_stats_sha256"),
            "source_init_path": run_contract.get("source_init_path"),
            "source_init_sha256": run_contract.get("source_init_sha256"),
        }
        contract["completeness"]["data"] = "present_via_run_contract"
    if contract.get("runtime_contract") is None:
        contract["runtime_contract"] = {
            "run_contract_sha256": run_contract_sha256,
            "deterministic_algorithms": run_contract.get("deterministic_algorithms"),
            "cublas_workspace_config": run_contract.get("cublas_workspace_config"),
            "stage_b_runtime_manifest_sha256": run_contract.get(
                "stage_b_runtime_manifest_sha256"
            ),
        }
        contract["completeness"]["runtime"] = "present_via_run_contract"


def record_release_identity(
    record: dict[str, object], *, protocol: str, stage: str, run_name: str,
    selection: str, state: dict[str, object], legacy_tags: bool,
) -> None:
    epoch = int(record["checkpoint_contract"].get("epoch", -1))
    step = int(record["checkpoint_contract"].get("step", -1))
    base = f"{protocol}-{stable_slug(stage)}-{stable_slug(run_name)}"
    if selection == "resume-last":
        tag = "resume-aio3-stage-a" if legacy_tags else f"resume-{base}"
        uploaded = state.get("resume_sha256") == record["sha256"]
    elif selection == "formal-best-model":
        tag = f"formal-{base}-e{epoch:04d}-s{step:07d}"
        uploaded = record["sha256"] in set(state.get("formal_sha256s", []))
    else:
        tag = (
            f"best-aio3-stage-a-e{epoch:04d}-s{step:07d}"
            if legacy_tags else f"best-{base}-e{epoch:04d}-s{step:07d}"
        )
        uploaded_best = set(state.get("best_sha256s", []))
        if state.get("best_sha256"):
            uploaded_best.add(str(state["best_sha256"]))
        uploaded = record["sha256"] in uploaded_best
    record.update({
        "protocol": protocol, "stage": stage, "run_name": run_name,
        "selection": selection, "epoch": epoch, "step": step,
        "release_tag": tag, "asset_name": Path(str(record["path"])).name,
        "release_state": "uploaded" if uploaded else "planned",
    })


def discover_run(
    source: Path, run_dir: Path, category: str,
    run_contract: dict[str, object] | None, cache: dict[str, object],
    state: dict[str, object], previous: dict[str, object],
) -> dict[str, object] | None:
    run_name = run_dir.name
    legacy_match = LEGACY_STAGE_A.fullmatch(run_name)
    protocol = (
        f"aio{legacy_match.group('count')}" if legacy_match
        else run_name.split("_", 1)[0]
    )
    stage = "a" if legacy_match else str(run_contract_field(run_contract or {}, "stage"))
    run_contract_path = run_dir / "run_contract.json"
    run_contract_sha = file_sha256(run_contract_path) if run_contract is not None else None
    top3_path = run_dir / "top3.json"
    top3_payload = (
        json.loads(top3_path.read_text())
        if category != "pilot" and top3_path.is_file() else []
    )
    if not isinstance(top3_payload, list):
        raise RuntimeError(f"Invalid top3 index: {top3_path}")
    top3: list[dict[str, object]] = []
    for row in top3_payload:
        checkpoint = run_dir / str(row.get("checkpoint"))
        if not checkpoint.is_file():
            continue
        record = stable_checkpoint_record(checkpoint, cache)
        enrich_from_run_contract(record, run_contract, run_contract_sha)
        record_release_identity(
            record, protocol=protocol, stage=stage, run_name=run_name,
            selection="top3", state=state, legacy_tags=bool(legacy_match),
        )
        record["score"] = row.get("score")
        validate_checkpoint_record(
            record, source, run_name, stage,
            expected_epoch=int(row["epoch"]), expected_step=int(row["step"]),
            run_contract=run_contract, run_contract_sha256=run_contract_sha,
        )
        top3.append(record)

    candidates = [*top3]
    last_path = run_dir / "last.pt"
    resume = None
    if category != "pilot" and last_path.is_file():
        candidate = stable_checkpoint_record(last_path, cache)
        enrich_from_run_contract(candidate, run_contract, run_contract_sha)
        validate_checkpoint_record(
            candidate, source, run_name, stage,
            run_contract=run_contract, run_contract_sha256=run_contract_sha,
        )
        if candidate["checkpoint_contract"]["resumable_training_state"]:
            record_release_identity(
                candidate, protocol=protocol, stage=stage, run_name=run_name,
                selection="resume-last", state=state, legacy_tags=bool(legacy_match),
            )
            resume = candidate
            candidates.append(candidate)

    formal_best = None
    formal_path = run_dir / "formal_best_model.pt"
    formal_marker = run_dir / "formal_complete.json"
    if formal_path.is_file() or formal_marker.is_file():
        if not formal_path.is_file() or not formal_marker.is_file():
            raise RuntimeError(f"Partial formal compaction: {run_dir}")
        marker = load_json_mapping(formal_marker)
        formal_best = stable_checkpoint_record(formal_path, cache)
        if marker.get("model") != formal_path.name or marker.get("model_sha256") != formal_best["sha256"]:
            raise RuntimeError(f"Formal marker/model mismatch: {run_dir}")
        enrich_from_run_contract(formal_best, run_contract, run_contract_sha)
        validate_checkpoint_record(
            formal_best, source, run_name, stage,
            run_contract=run_contract, run_contract_sha256=run_contract_sha,
        )
        record_release_identity(
            formal_best, protocol=protocol, stage=stage, run_name=run_name,
            selection="formal-best-model", state=state, legacy_tags=False,
        )
        formal_best["score"] = (marker.get("selected_locked_val") or {}).get("macro_psnr")
        formal_best["completion_marker"] = relative_file_record(source, formal_marker)
        candidates.append(formal_best)

    pilot_model = None
    pilot_path = run_dir / "pilot_model.pt"
    pilot_marker = run_dir / "pilot_complete.json"
    if category == "pilot" and (pilot_path.is_file() or pilot_marker.is_file()):
        if not pilot_path.is_file() or not pilot_marker.is_file():
            raise RuntimeError(f"Partial pilot compaction: {run_dir}")
        marker = load_json_mapping(pilot_marker)
        pilot_model = stable_checkpoint_record(pilot_path, cache)
        if marker.get("model") != pilot_path.name or marker.get("model_sha256") != pilot_model["sha256"]:
            raise RuntimeError(f"Pilot marker/model mismatch: {run_dir}")
        enrich_from_run_contract(pilot_model, run_contract, run_contract_sha)
        validate_checkpoint_record(
            pilot_model, source, run_name, stage,
            run_contract=run_contract, run_contract_sha256=run_contract_sha,
        )
        pilot_model.update({
            "protocol": protocol, "stage": stage, "run_name": run_name,
            "selection": "pilot-model-index-only", "asset_name": pilot_path.name,
            "release_tag": None, "release_state": "index_only_not_published",
        })
        pilot_model["completion_marker"] = relative_file_record(source, pilot_marker)
        candidates.append(pilot_model)
    if not candidates:
        return None

    first_contract = candidates[0]["checkpoint_contract"]
    config_path = resolve_project_path(source, first_contract.get("config_path"))
    if not config_path.is_file():
        expected_config = first_contract.get("config_sha256")
        search_roots = (source / "configs", source / "artifacts/manifests")
        matches = [
            path for root in search_roots for path in iter_tree(root)
            if path.suffix.lower() in {".yaml", ".yml"}
            and file_sha256(path) == expected_config
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely resolve run config: {run_dir}")
        config_path = matches[0]
    split_value = (
        (run_contract or {}).get("effective_config", {}).get("split_manifest")
        or first_contract.get("split_manifest_path")
        or (first_contract.get("data_contract") or {}).get("split_manifest_path")
        or (first_contract.get("data_contract") or {}).get("split_manifest")
    )
    split_path = resolve_project_path(source, split_value)
    if not split_path.is_file():
        expected_split = first_contract.get("split_manifest_sha256")
        split_matches = [
            path for path in iter_tree(source / "artifacts/manifests")
            if path.suffix.lower() == ".json" and file_sha256(path) == expected_split
        ]
        if len(split_matches) != 1:
            raise RuntimeError(f"Cannot uniquely resolve run split manifest: {run_dir}")
        split_path = split_matches[0]
    config_record = relative_file_record(source, config_path)
    split_record = relative_file_record(source, split_path)
    run_contract_record = (
        relative_file_record(source, run_contract_path)
        if run_contract is not None else None
    )
    top3_record = relative_file_record(source, top3_path) if top3_path.is_file() else None
    for record in candidates:
        if record["checkpoint_contract"].get("config_sha256") != config_record["sha256"]:
            raise RuntimeError(f"Run config/checkpoint mismatch: {run_dir}")
        if record["checkpoint_contract"].get("split_manifest_sha256") != split_record["sha256"]:
            raise RuntimeError(f"Run split/checkpoint mismatch: {run_dir}")
        record["config"] = config_record
        record["split_manifest"] = split_record
        record["run_contract"] = run_contract_record
        record["selection_index"] = top3_record

    uploaded_resume = None
    uploaded_resume_sha = state.get("resume_sha256")
    if uploaded_resume_sha:
        if resume is not None and resume.get("sha256") == uploaded_resume_sha:
            uploaded_resume = dict(resume)
        else:
            previous_candidates = [
                previous.get("resume_uploaded"), previous.get("resume_latest"),
                previous.get("resume_local_latest"),
            ]
            uploaded_resume = next(
                (dict(row) for row in previous_candidates if (
                    isinstance(row, dict)
                    and row.get("sha256") == uploaded_resume_sha
                    and isinstance(row.get("checkpoint_contract"), dict)
                    and isinstance(row.get("git_snapshot_commit"), str)
                    and isinstance(row.get("git_snapshot_tree"), str)
                )), None,
            )
        if uploaded_resume is not None:
            uploaded_resume["release_state"] = "uploaded"

    return {
        "run_name": run_name, "protocol": protocol, "stage": stage,
        "category": category, "publish_large_assets": category != "pilot",
        "run_contract": (
            {**run_contract_record, "payload": run_contract}
            if run_contract_record is not None else None
        ),
        "config": config_record, "split_manifest": split_record,
        "top3": top3, "current_best": top3[0] if top3 else None,
        "formal_best": formal_best, "pilot_model": pilot_model,
        "resume_latest": uploaded_resume or resume,
        "resume_uploaded": uploaded_resume, "resume_local_latest": resume,
    }


def checkpoint_index(source: Path, destination: Path) -> dict[str, object]:
    previous_index_path = destination / "recovery/CHECKPOINTS.json"
    try:
        previous_index = load_json_mapping(previous_index_path)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_index = {}
    cache_path = destination / ".checkpoint_hash_cache.json"
    cache = load_json_mapping(cache_path) if cache_path.exists() else {}
    state_path = destination / ".local_backup_state.json"
    upload_state = load_json_mapping(state_path) if state_path.exists() else {}
    checkpoint_root = source / "artifacts/checkpoints"
    runs: dict[str, dict[str, object]] = {}
    for run_dir in sorted(checkpoint_root.iterdir() if checkpoint_root.exists() else []):
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        classification = controlled_run(run_dir)
        if classification is None:
            continue
        category, run_contract = classification
        record = discover_run(
            source, run_dir, category, run_contract, cache,
            run_upload_state(upload_state, run_dir.name),
            previous_run(previous_index, run_dir.name),
        )
        if record is not None:
            runs[run_dir.name] = record
    atomic_write(cache_path, (json.dumps(cache, indent=2, sort_keys=True) + "\n").encode())
    legacy_name = "aio3_stage_a_coarse_seed1415926"
    legacy = runs.get(legacy_name, {})
    return {
        "schema_version": 2, "generated_utc": utc(), "source_root": str(source),
        "ordinary_git_contains_checkpoints": False, "runs": runs,
        # Backward-compatible aliases for the live AIO-3 Stage-A recovery path.
        "top3": legacy.get("top3", []),
        "current_best": legacy.get("current_best"),
        "resume_latest": legacy.get("resume_latest"),
        "resume_uploaded": legacy.get("resume_uploaded"),
        "resume_local_latest": legacy.get("resume_local_latest"),
        "config": legacy.get("config"),
        "split_manifest": legacy.get("split_manifest"),
    }


def validate_checkpoint_record(
    record: dict[str, object],
    source: Path,
    run_name: str,
    expected_stage: str,
    expected_epoch: int | None = None,
    expected_step: int | None = None,
    run_contract: dict[str, object] | None = None,
    run_contract_sha256: str | None = None,
) -> None:
    contract = record.get("checkpoint_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Checkpoint record has no payload contract")
    checks = {
        "run_name": contract.get("run_name") == run_name,
        "stage": contract.get("stage") == expected_stage,
    }
    if run_contract is not None:
        checks["run_contract_sha256"] = (
            contract.get("run_contract_sha256") == run_contract_sha256
        )
    if expected_epoch is not None:
        checks["epoch"] = int(contract.get("epoch", -1)) == expected_epoch
    if expected_step is not None:
        checks["step"] = int(contract.get("step", -1)) == expected_step
    code_contract = contract.get("code_contract")
    if isinstance(code_contract, dict):
        checks["code_contract_files"] = all(
            isinstance(relative, str)
            and isinstance(expected, str)
            and (source / relative).is_file()
            and file_sha256(source / relative) == expected
            for relative, expected in code_contract.items()
        )
    data_contract = contract.get("data_contract")
    if isinstance(data_contract, dict):
        for path_key, digest_key in (
            ("materialization_manifest", "materialization_manifest_sha256"),
            ("split_manifest", "split_manifest_sha256"),
            ("config_path", "config_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
            ("coordinate_stats_path", "coordinate_stats_sha256"),
            ("source_init_path", "source_init_sha256"),
        ):
            contract_path = data_contract.get(path_key)
            expected_digest = data_contract.get(digest_key)
            if expected_digest is not None and contract_path is not None:
                resolved_path = resolve_project_path(source, contract_path)
                checks[f"data_{path_key}"] = bool(
                    isinstance(contract_path, str)
                    and isinstance(expected_digest, str)
                    and resolved_path.is_file()
                    and file_sha256(resolved_path) == expected_digest
                )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Checkpoint payload/index provenance mismatch: {failed}")


def environment_record() -> dict[str, object]:
    record: dict[str, object] = {
        "generated_utc": utc(), "python": sys.version, "platform": platform.platform(),
    }
    try:
        import torch
        record.update({
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    except Exception as exc:  # pragma: no cover - diagnostic only
        record["torch_error"] = repr(exc)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        record["gpus"] = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except Exception as exc:  # pragma: no cover - diagnostic only
        record["nvidia_smi_error"] = repr(exc)
    return record


def build_snapshot(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, object]] = {}

    for name in sorted(TOP_LEVEL_FILES):
        path = source / name
        if path.exists():
            copied[name] = copy_checked(path, destination / name)

    for relative_root in (*TREE_ROOTS, *ARTIFACT_ROOTS):
        root = source / relative_root
        for path in iter_tree(root):
            relative = path.relative_to(source)
            if relative.parts[:2] == ("recovery", "SNAPSHOT_MANIFEST.json"):
                continue
            copied[str(relative)] = copy_checked(path, destination / relative)

    logs_root = source / "artifacts/logs"
    for name in sorted(KEY_LOGS):
        path = logs_root / name
        if path.exists():
            relative = Path("artifacts/logs") / name
            copied[str(relative)] = copy_checked(path, destination / relative)
    ddp_logs = logs_root / "ddp_stage_a"
    for path in iter_tree(ddp_logs):
        relative = path.relative_to(source)
        copied[str(relative)] = copy_checked(path, destination / relative)

    promptir_lists = source / "upstream/PromptIR/data_dir"
    for path in iter_tree(promptir_lists):
        relative = Path("vendor/promptir_data_dir") / path.relative_to(promptir_lists)
        copied[str(relative)] = copy_checked(path, destination / relative)

    for source_doc, relative in DOC_CONTRACTS.items():
        if source_doc.exists():
            copied[str(relative)] = copy_checked(source_doc, destination / relative)

    index = checkpoint_index(source, destination)
    # Controlled run metadata is text-only but lives beside large checkpoints.
    # Copy only the files explicitly bound by the generated scientific index;
    # never recurse through the checkpoint directory.
    bound_metadata: dict[str, dict[str, object]] = {}
    for run in index.get("runs", {}).values():
        if not isinstance(run, dict):
            continue
        metadata = [run.get("run_contract")]
        for key in ("top3",):
            for row in run.get(key, []):
                if isinstance(row, dict):
                    metadata.extend((row.get("selection_index"), row.get("completion_marker")))
        for key in ("current_best", "formal_best", "pilot_model"):
            row = run.get(key)
            if isinstance(row, dict):
                metadata.extend((row.get("selection_index"), row.get("completion_marker")))
        for record in metadata:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            relative = str(record["path"])
            if relative in bound_metadata and bound_metadata[relative] != record:
                raise RuntimeError(f"Conflicting scientific metadata binding: {relative}")
            bound_metadata[relative] = record
    for relative, expected in sorted(bound_metadata.items()):
        actual = copy_checked(source / relative, destination / relative)
        if actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Scientific metadata changed during snapshot: {relative}")
        copied[relative] = actual
    for run_name in sorted(index.get("runs", {})):
        for suffix in (".log", ".csv"):
            path = source / "artifacts/logs" / f"{run_name}{suffix}"
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(source).as_posix()
                copied[relative] = copy_checked(path, destination / relative)
    index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(destination / "recovery/CHECKPOINTS.json", index_bytes)
    copied["recovery/CHECKPOINTS.json"] = {
        "sha256": sha256_bytes(index_bytes), "size": len(index_bytes),
    }
    env_bytes = (json.dumps(environment_record(), indent=2, sort_keys=True) + "\n").encode()
    atomic_write(destination / "recovery/ENVIRONMENT.json", env_bytes)
    copied["recovery/ENVIRONMENT.json"] = {"sha256": sha256_bytes(env_bytes), "size": len(env_bytes)}

    manifest = {
        "schema_version": 1, "generated_utc": utc(), "source_root": str(source),
        "destination_root": str(destination.resolve()), "files": copied,
        "excluded": [
            "credentials", "datasets", "official-test images", "symlinks", "caches",
            "checkpoint binaries (controlled text metadata only)",
        ],
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(destination / "recovery/SNAPSHOT_MANIFEST.json", payload)
    purge_unexpected_destination_files(destination, set(copied))
    return {"files": len(copied), "bytes": sum(int(v["size"]) for v in copied.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12"))
    parser.add_argument("--destination", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12_github"))
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise SystemExit("Source and destination must be different")
    summary = build_snapshot(args.source, args.destination)
    print(json.dumps({"status": "PASS", **summary, "destination": str(args.destination)}, indent=2))


if __name__ == "__main__":
    main()
