#!/usr/bin/env python3
"""Build a fail-closed, text-only Git mirror of the live SRSC workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
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
    ".gitignore", "README.md", "GITHUB_BACKUP.md", "RUNNING_STATUS.md",
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
    Path("/root/ResearchStudio/ResearchStudio-Idea/skills/idea_spark/SKILL.md"):
        Path("vendor/researchstudio/idea_spark_SKILL.md"),
    Path("/root/.codex/skills/autosota/SKILL.md"):
        Path("vendor/autosota/SKILL.md"),
    Path("/root/R2R/utils/schedulers.py"): Path("vendor/r2r/utils/schedulers.py"),
    Path("/root/R2R/utils/image_utils.py"): Path("vendor/r2r/utils/image_utils.py"),
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
    cache[key] = {"size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}
    return {
        "path": str(path.relative_to(path.parents[3])),
        "absolute_source_path": str(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def checkpoint_index(source: Path, destination: Path) -> dict[str, object]:
    run_name = "aio3_stage_a_coarse_seed1415926"
    run_dir = source / "artifacts/checkpoints" / run_name
    top3_path = run_dir / "top3.json"
    top3 = json.loads(top3_path.read_text()) if top3_path.exists() else []
    cache_path = destination / ".checkpoint_hash_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    state_path = destination / ".local_backup_state.json"
    upload_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    records = []
    for row in top3:
        checkpoint = run_dir / str(row["checkpoint"])
        if not checkpoint.exists():
            continue
        record = stable_checkpoint_record(checkpoint, cache)
        record.update({
            "protocol": "aio3", "stage": "a", "run_name": run_name,
            "selection": "top3", "score": row.get("score"),
            "epoch": row.get("epoch"), "step": row.get("step"),
            "release_tag": f"best-aio3-stage-a-e{int(row['epoch']):04d}-s{int(row['step']):07d}",
            "asset_name": checkpoint.name,
            "release_state": (
                "uploaded" if upload_state.get("best_sha256") == record["sha256"]
                else "planned"
            ),
        })
        records.append(record)
    last_path = run_dir / "last.pt"
    resume = stable_checkpoint_record(last_path, cache) if last_path.exists() else None
    if resume is not None:
        resume.update({
            "protocol": "aio3", "stage": "a", "run_name": run_name,
            "selection": "resume-last", "release_tag": "resume-aio3-stage-a",
            "asset_name": "last.pt",
            "release_state": (
                "uploaded" if upload_state.get("resume_sha256") == resume["sha256"]
                else "planned"
            ),
        })
    atomic_write(cache_path, (json.dumps(cache, indent=2, sort_keys=True) + "\n").encode())
    return {
        "schema_version": 1,
        "generated_utc": utc(),
        "source_root": str(source),
        "ordinary_git_contains_checkpoints": False,
        "top3": records,
        "current_best": records[0] if records else None,
        "resume_latest": resume,
        "config": {
            "path": "configs/protocol_aio3.yaml",
            "sha256": file_sha256(source / "configs/protocol_aio3.yaml"),
        },
        "split_manifest": {
            "path": "artifacts/manifests/locked_split_aio3.json",
            "sha256": file_sha256(source / "artifacts/manifests/locked_split_aio3.json"),
        },
    }


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

    r2r_lists = Path("/root/R2R/data_dir")
    for path in iter_tree(r2r_lists):
        relative = Path("vendor/r2r/data_dir") / path.relative_to(r2r_lists)
        copied[str(relative)] = copy_checked(path, destination / relative)

    for source_doc, relative in DOC_CONTRACTS.items():
        if source_doc.exists():
            copied[str(relative)] = copy_checked(source_doc, destination / relative)

    index = checkpoint_index(source, destination)
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
        "excluded": ["credentials", "datasets", "official-test images", "symlinks", "caches", "checkpoints"],
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(destination / "recovery/SNAPSHOT_MANIFEST.json", payload)
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
