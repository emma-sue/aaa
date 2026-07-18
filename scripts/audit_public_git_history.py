#!/usr/bin/env python3
"""Offline, content-redacting audit of every object reachable in Git history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SECRET_PATTERNS = {
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "openai_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "generic_assignment": re.compile(
        rb"(?im)^\s*(?:password|passwd|access[_-]?token|auth[_-]?token|api[_-]?key|secret)"
        rb"\s*=\s*[\"']?[A-Za-z0-9_./+=-]{16,}[\"']?\s*$"
    ),
}
FORBIDDEN_NAMES = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\..*)?|.*credential.*|.*private_key.*|id_rsa.*|id_ed25519.*)$"
)
THIRD_PARTY_PREFIXES = (
    "vendor/autosota/", "vendor/researchstudio/idea_spark_SKILL.md",
    "vendor/r2r/",
)


def git(root: Path, *args: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=text,
    )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    listing = str(git(root, "rev-list", "--objects", "--all", text=True)).splitlines()
    objects: dict[str, set[str]] = {}
    for line in listing:
        object_id, separator, path = line.partition(" ")
        objects.setdefault(object_id, set())
        if separator:
            objects[object_id].add(path)

    all_named_paths = str(git(
        root, "log", "--all", "--format=", "--name-only", text=True,
    )).splitlines()

    findings: list[dict[str, object]] = []
    blob_count = 0
    scanned_bytes = 0
    max_blob = {"size": 0, "object": None, "paths": []}
    historical_paths: set[str] = {path for path in all_named_paths if path}
    for object_id, paths in objects.items():
        object_type = str(git(root, "cat-file", "-t", object_id, text=True)).strip()
        if object_type != "blob":
            continue
        blob_count += 1
        historical_paths.update(paths)
        payload = bytes(git(root, "cat-file", "blob", object_id))
        scanned_bytes += len(payload)
        if len(payload) > int(max_blob["size"]):
            max_blob = {"size": len(payload), "object": object_id, "paths": sorted(paths)}
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                findings.append({
                    "kind": "secret_pattern", "pattern": name,
                    "object": object_id, "paths": sorted(paths),
                })

    forbidden_paths = sorted(path for path in historical_paths if FORBIDDEN_NAMES.search(path))
    third_party_paths = sorted(
        path for path in historical_paths
        if any(path.startswith(prefix) for prefix in THIRD_PARTY_PREFIXES)
    )
    license_paths = sorted(
        path for path in historical_paths
        if Path(path).name.lower().startswith(("license", "copying", "notice"))
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all objects reachable from all local refs; blob content never printed",
        "commit_count": int(str(git(root, "rev-list", "--all", "--count", text=True)).strip()),
        "blob_count": blob_count,
        "scanned_bytes": scanned_bytes,
        "secret_findings": findings,
        "forbidden_credential_paths": forbidden_paths,
        "largest_blob": max_blob,
        "historical_third_party_paths_requiring_license_review": third_party_paths,
        "historical_license_notice_paths": license_paths,
        "history_rewrite_required": bool(third_party_paths),
        "rewrite_reason": (
            "Historical third-party source/skill files remain reachable although removed "
            "from HEAD; no co-located upstream license/notice closure is present in history."
            if third_party_paths else "none"
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
