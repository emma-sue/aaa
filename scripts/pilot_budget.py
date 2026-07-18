#!/usr/bin/env python3
"""Conservative cumulative budget guard for non-authoritative Stage-B pilots."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from scripts.runtime_accounting import read_runtime_sidecar


MAX_PILOT_WALL_HOURS = 24.0
MAX_PILOT_GPU_HOURS = 20.0


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def collect_pilot_budget(root: Path, protocol: str) -> dict:
    """Sum durable pilot trainer counters across Oracle and Predicted arms.

    Summing per-arm wall time is conservative under parallel execution.  It
    cannot understate actual campaign wall time and the one-GPU pilot arms make
    accumulated GPU time exact up to each sidecar's accounting provenance.
    """
    checkpoint_root = Path(root) / "artifacts/checkpoints"
    runs = {}
    wall_seconds = 0.0
    gpu_seconds = 0.0
    if checkpoint_root.is_dir():
        for run_dir in sorted(checkpoint_root.glob(f"{protocol}_*_pilot_n*_s*")):
            sidecar = run_dir / "runtime_accounting.json"
            if not sidecar.is_file():
                continue
            payload = read_runtime_sidecar(sidecar)
            if payload.get("protocol") != protocol:
                raise RuntimeError(f"pilot sidecar protocol mismatch: {sidecar}")
            if payload.get("stage") not in {"b_oracle", "b_predicted"}:
                raise RuntimeError(f"non-Stage-B sidecar matched pilot family: {sidecar}")
            wall = float(payload["accumulated_wall_seconds"])
            gpu = float(payload["accumulated_gpu_seconds"])
            if not math.isfinite(wall + gpu) or min(wall, gpu) < 0:
                raise ValueError(f"invalid pilot accounting: {sidecar}")
            wall_seconds += wall
            gpu_seconds += gpu
            runs[run_dir.name] = {
                "wall_hours": wall / 3600.0,
                "gpu_hours": gpu / 3600.0,
                "origin": payload.get(
                    "accounting_origin", "EMBEDDED_TRAINER_MONOTONIC_COUNTER"
                ),
            }
    wall_hours = wall_seconds / 3600.0
    gpu_hours = gpu_seconds / 3600.0
    exceeded = (
        wall_hours > MAX_PILOT_WALL_HOURS
        or gpu_hours > MAX_PILOT_GPU_HOURS
    )
    return {
        "schema": "srsc.stage_b_pilot_budget.v1",
        "protocol": protocol,
        "status": "TIME_BUDGET_EXCEEDED" if exceeded else "WITHIN_BUDGET",
        "summed_trainer_wall_hours": wall_hours,
        "training_gpu_hours": gpu_hours,
        "max_summed_trainer_wall_hours": MAX_PILOT_WALL_HOURS,
        "max_training_gpu_hours": MAX_PILOT_GPU_HOURS,
        "wall_accounting_is_conservative_under_parallelism": True,
        "runs": runs,
    }


def persist_and_enforce_pilot_budget(root: Path, protocol: str) -> dict:
    usage = collect_pilot_budget(root, protocol)
    output = (
        Path(root) / "artifacts/manifests" / f"stage_b_pilot_budget_{protocol}.json"
    )
    _atomic_json(output, usage)
    if usage["status"] == "TIME_BUDGET_EXCEEDED":
        stop = Path(root) / "STOP_REASON.md"
        stop.write_text(
            "# TIME_BUDGET_EXCEEDED\n\n"
            f"Protocol: `{protocol}`  \n"
            f"Summed pilot trainer wall-hours: `{usage['summed_trainer_wall_hours']:.4f}`  \n"
            f"Pilot GPU-hours: `{usage['training_gpu_hours']:.4f}`  \n"
            "No formal Stage-B experiment may start from this invocation.\n"
        )
        raise RuntimeError("TIME_BUDGET_EXCEEDED: Stage-B pilot budget")
    return usage

