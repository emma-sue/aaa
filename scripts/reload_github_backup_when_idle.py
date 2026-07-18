#!/usr/bin/env python3
"""One-shot hot reload of the GitHub daemon after its current archive transaction."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


def upload_process_active() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "^gh release upload "],
        text=True, capture_output=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def transaction_complete(mirror: Path, source: Path) -> bool:
    state_path = mirror / ".local_backup_state.json"
    if not state_path.exists() or upload_process_active():
        return False
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return False
    index_path = mirror / "recovery/CHECKPOINTS.json"
    if not index_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return False
    runs = index.get("runs")
    if not isinstance(runs, dict) or not runs:
        return False
    state_runs = state.get("runs") if isinstance(state.get("runs"), dict) else {}
    for run_name, run in runs.items():
        if not isinstance(run, dict) or not run.get("publish_large_assets", True):
            continue
        run_state = dict(state_runs.get(run_name, {}))
        if run_name == "aio3_stage_a_coarse_seed1415926":
            for key in ("best_sha256", "best_sha256s", "resume_sha256"):
                if key in state and key not in run_state:
                    run_state[key] = state[key]
        best = set(run_state.get("best_sha256s", []))
        if run_state.get("best_sha256"):
            best.add(str(run_state["best_sha256"]))
        formal = set(run_state.get("formal_sha256s", []))
        for row in run.get("top3", []):
            if isinstance(row, dict) and row.get("sha256") not in best:
                return False
        head = run.get("formal_best") or run.get("current_best")
        if isinstance(head, dict):
            uploaded = formal if head.get("selection") == "formal-best-model" else best
            if head.get("sha256") not in uploaded:
                return False
        resume = run.get("resume_local_latest") or run.get("resume_latest")
        if (
            isinstance(resume, dict)
            and (resume.get("checkpoint_contract") or {}).get("resumable_training_state")
            and not run_state.get("resume_sha256")
        ):
            return False
    return True


def wait_for_daemon_lock_release(mirror: Path, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    lock_path = mirror / ".backup_daemon.lock"
    while time.monotonic() < deadline:
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            time.sleep(0.5)
            continue
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        return
    raise RuntimeError("Timed out waiting for the prior backup daemon lock")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12"))
    parser.add_argument("--mirror", type=Path, default=Path("/root/autodl-tmp/srsc_lite_v12_github"))
    parser.add_argument("--session", default="srsc_github_backup")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    consecutive = 0
    while consecutive < 2:
        if transaction_complete(args.mirror, args.source):
            consecutive += 1
        else:
            consecutive = 0
        time.sleep(max(10, args.poll_seconds))
    subprocess.run(["tmux", "kill-session", "-t", args.session], check=True)
    wait_for_daemon_lock_release(args.mirror)
    command = (
        f"cd {args.source} && "
        "python scripts/checkpoint_backup_daemon.py --repo emma-sue/aaa --allow-public "
        f"--source {args.source} --mirror {args.mirror} "
        "--interval-seconds 900 --resume-interval-hours 1 "
        "2>&1 | tee -a artifacts/logs/github_backup.log"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session, command], check=True)


if __name__ == "__main__":
    main()
