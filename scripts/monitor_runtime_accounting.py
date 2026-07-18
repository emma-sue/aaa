#!/usr/bin/env python3
"""Bridge a legacy live trainer into the checkpoint runtime sidecar.

This is used only when the already-running process predates embedded runtime
accounting.  New trainer invocations read the sidecar and then maintain it
themselves, so this monitor exits as soon as the nominated legacy PID exits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_accounting import (
    atomic_write_runtime_sidecar,
    runtime_snapshot,
    start_runtime_accounting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--expected-command-token", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gpu-count", required=True, type=int)
    parser.add_argument("--prior-wall-seconds", required=True, type=float)
    parser.add_argument("--prior-gpu-seconds", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    return parser.parse_args()


def process_matches(pid: int, token: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
        return False
    return token in command


def main() -> int:
    args = parse_args()
    if args.pid <= 1 or args.interval_seconds <= 0 or args.interval_seconds > 60:
        raise ValueError("invalid PID or monitor interval")
    if not process_matches(args.pid, args.expected_command_token):
        raise RuntimeError("legacy trainer PID does not match the expected command")
    start_runtime_accounting(
        gpu_count=args.gpu_count,
        run_name=args.run_name,
        protocol=args.protocol,
        stage=args.stage,
        prior={
            "schema": 1,
            "run_name": args.run_name,
            "protocol": args.protocol,
            "stage": args.stage,
            "accumulated_wall_seconds": args.prior_wall_seconds,
            "accumulated_gpu_seconds": args.prior_gpu_seconds,
        },
    )
    while True:
        snapshot = runtime_snapshot()
        assert snapshot is not None
        snapshot.update({
            "accounting_origin": "LEGACY_LOG_ACTIVE_INTERVAL_ESTIMATE_PLUS_LIVE_PID",
            "legacy_active_gap_cap_seconds": 600.0,
            "legacy_pid": args.pid,
            "monitor_pid": os.getpid(),
        })
        atomic_write_runtime_sidecar(args.output, snapshot)
        if not process_matches(args.pid, args.expected_command_token):
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
