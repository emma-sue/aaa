#!/usr/bin/env python3
"""Download a GitHub Release checkpoint and verify its SHA sidecar/index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if shutil.which("gh") is None:
        raise SystemExit("GitHub CLI `gh` is required; install it and run `gh auth login` first")
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="srsc_release_") as temp:
        temp_path = Path(temp)
        sidecar_name = args.asset + ".sha256"
        subprocess.run([
            "gh", "release", "download", args.tag, "--repo", args.repo,
            "--pattern", args.asset, "--pattern", sidecar_name, "--dir", str(temp_path),
        ], check=True)
        downloaded = temp_path / args.asset
        sidecar = temp_path / sidecar_name
        expected = sidecar.read_text().strip().split()[0]
        actual = sha256(downloaded)
        if actual != expected:
            raise RuntimeError(f"Release SHA mismatch: expected {expected}, got {actual}")
        index = json.loads((args.root / "recovery/CHECKPOINTS.json").read_text())
        rows = [*index.get("top3", [])]
        if index.get("resume_latest"):
            rows.append(index["resume_latest"])
        indexed = {row.get("sha256") for row in rows if row.get("release_tag") == args.tag}
        if indexed and actual not in indexed:
            raise RuntimeError("Release checkpoint does not match the Git checkpoint index")
        temp_destination = destination.with_name(destination.name + ".download")
        shutil.copy2(downloaded, temp_destination)
        temp_destination.replace(destination)
    subprocess.run([
        "python", str(args.root / "scripts/verify_recovery_bundle.py"),
        "--root", str(args.root), "--checkpoint", str(destination),
    ], check=True)
    print(f"Restored and verified: {destination}")


if __name__ == "__main__":
    main()

