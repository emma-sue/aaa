#!/usr/bin/env python3
"""Continuously snapshot Git state and publish checkpoint Release assets."""

from __future__ import annotations

import argparse
import fcntl
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


def run_network(
    command: list[str], cwd: Path | None = None, capture: bool = False, attempts: int = 5,
) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run(command, cwd=cwd, capture=capture)
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(15, 2 * attempt))
    assert last_error is not None
    raise last_error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def acquire_daemon_lock(mirror: Path):
    """Hold a process-lifetime lock so two daemons cannot clobber one Release."""
    mirror.mkdir(parents=True, exist_ok=True)
    handle = (mirror / ".backup_daemon.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"Another backup daemon owns {mirror}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={utc()}\n")
    handle.flush()
    return handle


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
        run(["git", "config", "http.version", "HTTP/1.1"], cwd=mirror)


def commit_snapshot(mirror: Path) -> str:
    run(["git", "add", "-A"], cwd=mirror)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=mirror).returncode != 0
    if changed:
        run(["git", "commit", "-m", f"backup: reproducible snapshot {utc()}"], cwd=mirror)
    return run(["git", "rev-parse", "HEAD"], cwd=mirror, capture=True).stdout.strip()


def bind_checkpoint_index_to_commit(mirror: Path, snapshot_commit: str) -> None:
    """Bind checkpoint metadata to the preceding immutable content commit.

    A commit cannot contain its own hash.  The exporter is therefore committed
    first, then this binding is recorded in a second commit.  The bound commit
    contains the exact code/config snapshot, while the following commit contains
    the recovery index that names it.
    """
    tree = run(
        ["git", "rev-parse", f"{snapshot_commit}^{{tree}}"], cwd=mirror, capture=True,
    ).stdout.strip()
    index_path = mirror / "recovery/CHECKPOINTS.json"
    index = json.loads(index_path.read_text())
    binding = {"commit": snapshot_commit, "tree": tree}
    index["git_snapshot"] = binding
    rows: list[dict[str, object]] = []
    rows.extend(row for row in index.get("top3", []) if isinstance(row, dict))
    for key in (
        "current_best", "resume_latest", "resume_uploaded", "resume_local_latest",
    ):
        row = index.get(key)
        if isinstance(row, dict):
            rows.append(row)
    for row in rows:
        if not isinstance(row.get("checkpoint_contract"), dict):
            raise RuntimeError("Checkpoint row has no payload contract")
        if (
            row.get("release_state") == "uploaded"
            and isinstance(row.get("git_snapshot_commit"), str)
            and isinstance(row.get("git_snapshot_tree"), str)
        ):
            # A rolling resume asset may refer to the previous generation while
            # live last.pt has already advanced. Preserve that generation's
            # original code binding rather than rebinding history to HEAD.
            continue
        row["git_snapshot_commit"] = snapshot_commit
        row["git_snapshot_tree"] = tree
    index_payload = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_bytes(index_payload)
    temporary.replace(index_path)

    manifest_path = mirror / "recovery/SNAPSHOT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["recovery/CHECKPOINTS.json"] = {
        "sha256": sha256_bytes(index_payload), "size": len(index_payload),
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_bytes(manifest_payload)
    temporary.replace(manifest_path)


def commit_bound_snapshot(mirror: Path) -> tuple[str, str]:
    snapshot_commit = commit_snapshot(mirror)
    bind_checkpoint_index_to_commit(mirror, snapshot_commit)
    binding_commit = commit_snapshot(mirror)
    return snapshot_commit, binding_commit


def ensure_gh_repo(repo: str, allow_public: bool) -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI `gh` is not installed")
    run(["gh", "auth", "status"])
    payload = json.loads(run_network(
        ["gh", "repo", "view", repo, "--json", "isPrivate"], capture=True,
    ).stdout)
    visibility = "PRIVATE" if payload["isPrivate"] else "PUBLIC"
    if visibility != "PRIVATE" and not allow_public:
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


def release_assets(repo: str, tag: str) -> dict[str, dict[str, object]]:
    if not release_exists(repo, tag):
        return {}
    payload = json.loads(run_network(
        ["gh", "api", f"repos/{repo}/releases/tags/{tag}"], capture=True,
    ).stdout)
    return {str(asset["name"]): asset for asset in payload.get("assets", [])}


def asset_matches(row: dict[str, object] | None, digest: str, size: int) -> bool:
    return bool(
        row
        and row.get("state") == "uploaded"
        and int(row.get("size", -1)) == int(size)
        and row.get("digest") == f"sha256:{digest}"
    )


def verify_remote_release_assets(
    repo: str, tag: str, expected: dict[str, tuple[str, int]],
) -> None:
    payload = json.loads(run_network(
        ["gh", "api", f"repos/{repo}/releases/tags/{tag}"], capture=True,
    ).stdout)
    assets = {str(asset["name"]): asset for asset in payload.get("assets", [])}
    mismatches = [
        name for name, (digest, size) in expected.items()
        if not asset_matches(assets.get(name), digest, size)
    ]
    if mismatches:
        raise RuntimeError(f"Release assets incomplete or digest-mismatched: {tag}/{mismatches}")


def publish_release(repo: str, row: dict[str, object], immutable: bool) -> None:
    source = Path(str(row["absolute_source_path"]))
    expected = str(row["sha256"])
    tag = str(row["release_tag"])
    with tempfile.TemporaryDirectory(prefix="srsc_upload_") as temp:
        temp_path = Path(temp)
        sidecar = temp_path / (str(row["asset_name"]) + ".sha256")
        sidecar.write_text(f"{expected}  {row['asset_name']}\n")
        metadata = temp_path / (str(row["asset_name"]) + ".json")
        metadata.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        sidecar_digest = sha256(sidecar)
        metadata_digest = sha256(metadata)
        expected_assets = {
            str(row["asset_name"]): (expected, int(row["size"])),
            sidecar.name: (sidecar_digest, sidecar.stat().st_size),
            metadata.name: (metadata_digest, metadata.stat().st_size),
        }
        existing = release_assets(repo, tag)
        if immutable and existing:
            main = existing.get(str(row["asset_name"]))
            if main and not asset_matches(main, expected, int(row["size"])):
                raise RuntimeError(f"Immutable checkpoint asset drift: {tag}/{row['asset_name']}")
            if all(
                asset_matches(existing.get(name), digest, size)
                for name, (digest, size) in expected_assets.items()
            ):
                return
            if main:
                run_network([
                    "gh", "release", "upload", tag, str(sidecar), str(metadata),
                    "--repo", repo, "--clobber",
                ], capture=True)
                verify_remote_release_assets(repo, tag, expected_assets)
                return
        asset = stage_asset(source, temp_path)
        actual = sha256(asset)
        if actual != expected:
            raise RuntimeError(f"Checkpoint changed after snapshot: {source}")
        if not release_exists(repo, tag):
            target = str(row.get("git_snapshot_commit") or "main")
            run_network([
                "gh", "release", "create", tag, "--repo", repo, "--target", target,
                "--title", tag, "--notes", "SRSC checkpoint with SHA256-bound recovery metadata.",
            ])
        run_network([
            "gh", "release", "upload", tag, str(asset), str(sidecar), str(metadata),
            "--repo", repo, "--clobber",
        ], capture=True)
        verify_remote_release_assets(repo, tag, expected_assets)


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
    commit_bound_snapshot(args.mirror)
    run([
        "python", str(args.source / "scripts/verify_recovery_bundle.py"),
        "--root", str(args.mirror),
    ])
    if args.local_only:
        return
    ensure_gh_repo(args.repo, args.allow_public)
    run(["gh", "auth", "setup-git"])
    run_network(["git", "push", "-u", "origin", "main"], cwd=args.mirror)

    index = json.loads((args.mirror / "recovery/CHECKPOINTS.json").read_text())
    state_path = args.mirror / ".local_backup_state.json"
    state = load_state(state_path)
    uploaded = False
    uploaded_best = set(state.get("best_sha256s", []))
    if state.get("best_sha256"):
        uploaded_best.add(str(state["best_sha256"]))
    # Protect the current top-1 first.
    current_best = index.get("current_best")
    if current_best and current_best.get("sha256") not in uploaded_best:
        publish_release(args.repo, current_best, immutable=True)
        uploaded_best.add(str(current_best["sha256"]))
        state["best_sha256"] = current_best["sha256"]
        state["best_sha256s"] = sorted(uploaded_best)
        state["best_uploaded_utc"] = utc()
        save_state(state_path, state)
        uploaded = True
    resume = index.get("resume_local_latest") or index.get("resume_latest")
    due = time.time() - float(state.get("resume_upload_unix", 0)) >= args.resume_interval_hours * 3600
    if resume and due and state.get("resume_sha256") != resume.get("sha256"):
        publish_release(args.repo, resume, immutable=False)
        state["resume_sha256"] = resume["sha256"]
        state["resume_upload_unix"] = time.time()
        state["resume_uploaded_utc"] = utc()
        save_state(state_path, state)
        uploaded = True
    # Once current best and rolling resume state are protected, archive the
    # bounded retained top-3 backlog. This recovers surviving historical bests
    # without delaying the checkpoint needed for exact continuation.
    for best in index.get("top3", []):
        if best.get("sha256") in uploaded_best:
            publish_release(args.repo, best, immutable=True)
            continue
        publish_release(args.repo, best, immutable=True)
        uploaded_best.add(str(best["sha256"]))
        state["best_sha256"] = best["sha256"]
        state["best_sha256s"] = sorted(uploaded_best)
        state["best_uploaded_utc"] = utc()
        save_state(state_path, state)
        uploaded = True
    save_state(state_path, state)
    if uploaded:
        # Only after successful asset publication do we mark the Git index as
        # uploaded and push that state. The exporter reads the local state file.
        run([
            "python", str(args.source / "scripts/export_repro_snapshot.py"),
            "--source", str(args.source), "--destination", str(args.mirror),
        ])
        commit_bound_snapshot(args.mirror)
        run([
            "python", str(args.source / "scripts/verify_recovery_bundle.py"),
            "--root", str(args.mirror),
        ])
        run_network(["git", "push", "origin", "main"], cwd=args.mirror)


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
    daemon_lock = acquire_daemon_lock(args.mirror)
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
    daemon_lock.close()


if __name__ == "__main__":
    main()
