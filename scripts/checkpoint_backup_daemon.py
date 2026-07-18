#!/usr/bin/env python3
"""Continuously snapshot Git state and publish checkpoint Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_git(mirror: Path, repo: str | None) -> None:
    if not (mirror / ".git").exists():
        run(["git", "init", "-b", "main"], cwd=mirror)
    email = subprocess.run(
        ["git", "config", "user.email"], cwd=mirror, text=True, capture_output=True,
    ).stdout.strip()
    if not email:
        run(["git", "config", "user.email", "codex-srsc-backup@users.noreply.github.com"], cwd=mirror)
    name = subprocess.run(
        ["git", "config", "user.name"], cwd=mirror, text=True, capture_output=True,
    ).stdout.strip()
    if not name:
        run(["git", "config", "user.name", "Codex SRSC Backup"], cwd=mirror)
    if repo:
        remote = f"https://github.com/{repo}.git"
        remotes = run(["git", "remote"], cwd=mirror, capture=True).stdout.split()
        if "origin" not in remotes:
            run(["git", "remote", "add", "origin", remote], cwd=mirror)
        elif run(["git", "remote", "get-url", "origin"], cwd=mirror, capture=True).stdout.strip() != remote:
            raise RuntimeError("Existing origin does not match requested GitHub repository")


def commit_snapshot(mirror: Path) -> str:
    run(["git", "add", "-A"], cwd=mirror)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=mirror).returncode != 0
    if changed:
        run(["git", "commit", "-m", f"backup: reproducible snapshot {utc()}"], cwd=mirror)
    return run(["git", "rev-parse", "HEAD"], cwd=mirror, capture=True).stdout.strip()


def ensure_gh_repo(repo: str, allow_public: bool) -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI `gh` is not installed")
    run(["gh", "auth", "status"])
    visibility = json.loads(run(
        ["gh", "repo", "view", repo, "--json", "visibility"], capture=True,
    ).stdout)["visibility"]
    if visibility.upper() != "PRIVATE" and not allow_public:
        raise RuntimeError(f"Refusing checkpoint publication to {visibility} repo without --allow-public")


def stage_asset(source: Path, staging: Path) -> Path:
    destination = staging / source.name
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def release_exists(repo: str, tag: str) -> bool:
    return subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def publish_release(repo: str, row: dict[str, object], immutable: bool) -> None:
    source = Path(str(row["absolute_source_path"]))
    expected = str(row["sha256"])
    tag = str(row["release_tag"])
    with tempfile.TemporaryDirectory(prefix="srsc_upload_") as temp:
        temp_path = Path(temp)
        asset = stage_asset(source, temp_path)
        actual = sha256(asset)
        if actual != expected:
            raise RuntimeError(f"Checkpoint changed after snapshot: {source}")
        sidecar = temp_path / (str(row["asset_name"]) + ".sha256")
        sidecar.write_text(f"{actual}  {row['asset_name']}\n")
        metadata = temp_path / (str(row["asset_name"]) + ".json")
        metadata.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        if immutable and release_exists(repo, tag):
            return
        if not release_exists(repo, tag):
            run([
                "gh", "release", "create", tag, "--repo", repo, "--target", "main",
                "--title", tag, "--notes", "SRSC checkpoint with SHA256-bound recovery metadata.",
            ])
        run([
            "gh", "release", "upload", tag, str(asset), str(sidecar), str(metadata),
            "--repo", repo, "--clobber",
        ])


def load_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict[str, object]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def iteration(args: argparse.Namespace) -> None:
    run([
        "python", str(args.source / "scripts/export_repro_snapshot.py"),
        "--source", str(args.source), "--destination", str(args.mirror),
    ])
    initialize_git(args.mirror, None if args.local_only else args.repo)
    commit_snapshot(args.mirror)
    if args.local_only:
        return
    ensure_gh_repo(args.repo, args.allow_public)
    run(["gh", "auth", "setup-git"])
    run(["git", "push", "-u", "origin", "main"], cwd=args.mirror)

    index = json.loads((args.mirror / "recovery/CHECKPOINTS.json").read_text())
    state_path = args.mirror / ".local_backup_state.json"
    state = load_state(state_path)
    uploaded = False
    best = index.get("current_best")
    if best and state.get("best_sha256") != best.get("sha256"):
        publish_release(args.repo, best, immutable=True)
        state["best_sha256"] = best["sha256"]
        state["best_uploaded_utc"] = utc()
        uploaded = True
    resume = index.get("resume_latest")
    due = time.time() - float(state.get("resume_upload_unix", 0)) >= args.resume_interval_hours * 3600
    if resume and due and state.get("resume_sha256") != resume.get("sha256"):
        publish_release(args.repo, resume, immutable=False)
        state["resume_sha256"] = resume["sha256"]
        state["resume_upload_unix"] = time.time()
        state["resume_uploaded_utc"] = utc()
        uploaded = True
    save_state(state_path, state)
    if uploaded:
        # Only after successful asset publication do we mark the Git index as
        # uploaded and push that state. The exporter reads the local state file.
        run([
            "python", str(args.source / "scripts/export_repro_snapshot.py"),
            "--source", str(args.source), "--destination", str(args.mirror),
        ])
        commit_snapshot(args.mirror)
        run(["git", "push", "origin", "main"], cwd=args.mirror)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="emma-sue/aaa")
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12"))
    parser.add_argument("--mirror", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12_github"))
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--resume-interval-hours", type=float, default=1.0)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--allow-public", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            iteration(args)
            print(f"[{utc()}] backup iteration PASS", flush=True)
        except Exception as exc:
            print(f"[{utc()}] backup iteration FAIL: {exc!r}", flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    main()
