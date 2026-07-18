#!/usr/bin/env python3
"""Verify the Git snapshot and, optionally, a restored PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


LOCAL_CONTROL_FILES = {
    ".backup_daemon.lock",
    ".checkpoint_hash_cache.json",
    ".local_backup_state.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_manifest(root: Path) -> int:
    manifest_path = root / "recovery/SNAPSHOT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unsafe tracked file: {relative}")
        if path.stat().st_size != int(expected["size"]):
            raise RuntimeError(f"Size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {relative}")
        checked += 1
    expected_paths = set(manifest["files"]) | {"recovery/SNAPSHOT_MANIFEST.json"}
    actual_paths = set()
    if (root / ".git").is_dir():
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True, capture_output=True,
        ).stdout.split(b"\0")
        actual_paths = {item.decode() for item in tracked if item}
        for rendered in actual_paths:
            if (root / rendered).is_symlink():
                raise RuntimeError(f"Unexpected tracked symlink in recovery bundle: {rendered}")
    else:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            rendered = relative.as_posix()
            if path.is_symlink():
                raise RuntimeError(f"Unexpected symlink in recovery bundle: {rendered}")
            if path.is_file() and rendered not in LOCAL_CONTROL_FILES:
                actual_paths.add(rendered)
    extras = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if extras or missing:
        raise RuntimeError(
            f"Recovery bundle is not a closed set: extras={extras[:8]} missing={missing[:8]}"
        )
    return checked


def payload_contract(payload: dict) -> dict[str, object]:
    args = payload.get("args") or {}
    runtime = payload.get("runtime_contract")
    distributed = payload.get("distributed_runtime")
    contract = {
        "schema_version": 1,
        "epoch": int(payload.get("epoch", -1)),
        "step": int(payload.get("step", -1)),
        "batch_in_epoch": int(payload.get("batch_in_epoch", -1)),
        "config_sha256": payload.get("config_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "training_origin": payload.get("training_origin"),
        "distributed_runtime": distributed,
        "runtime_contract": runtime,
        "data_contract": payload.get("data_contract"),
        "code_contract": payload.get("code_contract"),
        "run_name": args.get("run_name"),
        "stage": args.get("stage"),
    }
    contract["completeness"] = {
        "runtime": "present" if runtime is not None or distributed is not None else "missing",
        "data": "present" if contract["data_contract"] is not None else "legacy_missing",
        "code": "present" if contract["code_contract"] is not None else "legacy_missing",
    }
    return contract


def verify_git_snapshot(root: Path, row: dict) -> None:
    commit = row.get("git_snapshot_commit")
    expected_tree = row.get("git_snapshot_tree")
    if not isinstance(commit, str) or not isinstance(expected_tree, str):
        raise RuntimeError("Checkpoint row is not bound to a Git snapshot")
    subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    actual_tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{commit}^{{tree}}"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if actual_tree != expected_tree:
        raise RuntimeError("Checkpoint Git snapshot tree does not match its index binding")
    index = json.loads((root / "recovery/CHECKPOINTS.json").read_text())
    contract = row.get("checkpoint_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Checkpoint Git binding has no payload contract")
    for index_key, contract_key in (
        ("config", "config_sha256"),
        ("split_manifest", "split_manifest_sha256"),
    ):
        metadata = index.get(index_key) or {}
        relative = metadata.get("path")
        expected = contract.get(contract_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError(f"Recovery index has no {index_key} binding")
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if sha256_bytes(blob) != expected:
            raise RuntimeError(f"Checkpoint {index_key} does not match its Git snapshot")
    code_contract = contract.get("code_contract")
    if isinstance(code_contract, dict):
        for relative, expected in code_contract.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError("Unsafe checkpoint code-contract path")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe checkpoint code-contract path")
            blob = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{relative}"],
                check=True, capture_output=True,
            ).stdout
            if sha256_bytes(blob) != expected:
                raise RuntimeError(f"Checkpoint code contract differs from Git: {relative}")


def verify_checkpoint(root: Path, checkpoint: Path) -> dict[str, object]:
    import torch

    index = json.loads((root / "recovery/CHECKPOINTS.json").read_text())
    candidates = [*index.get("top3", [])]
    for key in ("resume_uploaded", "resume_latest", "resume_local_latest"):
        if index.get(key):
            candidates.append(index[key])
    digest = sha256(checkpoint)
    matches = [row for row in candidates if row.get("sha256") == digest]
    if not matches:
        raise RuntimeError(f"Checkpoint SHA256 is absent from recovery index: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "rng", "epoch", "step", "config_sha256"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Checkpoint missing required resume state: {missing}")
    actual_contract = payload_contract(payload)
    for row in matches:
        verify_git_snapshot(root, row)
        if row.get("checkpoint_contract") != actual_contract:
            raise RuntimeError("Checkpoint payload contract differs from recovery index")
    completeness = actual_contract["completeness"]
    legacy_missing = any(
        completeness.get(field) == "legacy_missing" for field in ("code", "data")
    )
    return {
        "sha256": digest,
        "git_snapshot_commit": matches[0].get("git_snapshot_commit"),
        "completeness": completeness,
        "provenance_status": (
            "STATE_EXACT_LEGACY_CODE_DATA_UNPROVEN"
            if legacy_missing else "FULL_CHECKPOINT_AND_GIT_CONTRACT_BOUND"
        ),
        "warning": (
            "Legacy checkpoint has no embedded code/data hashes; the bound Git snapshot "
            "is an audited recovery snapshot, not proof of launch-time source identity."
            if legacy_missing else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    count = verify_manifest(args.root)
    checkpoint_details = verify_checkpoint(args.root, args.checkpoint) if args.checkpoint else None
    print(json.dumps({
        "status": "PASS", "manifest_files": count,
        "checkpoint": str(args.checkpoint or ""),
        "checkpoint_verification": checkpoint_details,
    }, indent=2))


if __name__ == "__main__":
    main()
