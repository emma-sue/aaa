#!/usr/bin/env python3
"""Verify the Git snapshot and, optionally, a restored PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return checked


def verify_checkpoint(root: Path, checkpoint: Path) -> None:
    import torch

    index = json.loads((root / "recovery/CHECKPOINTS.json").read_text())
    candidates = [*index.get("top3", [])]
    if index.get("resume_latest"):
        candidates.append(index["resume_latest"])
    digest = sha256(checkpoint)
    matches = [row for row in candidates if row.get("sha256") == digest]
    if not matches:
        raise RuntimeError(f"Checkpoint SHA256 is absent from recovery index: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "rng", "epoch", "step", "config_sha256"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Checkpoint missing required resume state: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    count = verify_manifest(args.root)
    if args.checkpoint:
        verify_checkpoint(args.root, args.checkpoint)
    print(json.dumps({"status": "PASS", "manifest_files": count, "checkpoint": str(args.checkpoint or "")}, indent=2))


if __name__ == "__main__":
    main()

