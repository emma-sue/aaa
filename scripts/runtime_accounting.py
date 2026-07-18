"""Crash-resumable training GPU-time accounting.

The tracker measures only time spent inside a trainer invocation.  Every
checkpoint stores the cumulative counters, and a small atomic sidecar mirrors
the latest snapshot so the orchestrator can aggregate runs without loading
large optimizer checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = 1
_ACTIVE: "RuntimeTracker | None" = None


class RuntimeTracker:
    def __init__(
        self,
        *,
        gpu_count: int,
        run_name: str,
        protocol: str,
        stage: str,
        prior: dict | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        unix_clock: Callable[[], float] = time.time,
    ) -> None:
        if int(gpu_count) <= 0:
            raise ValueError("gpu_count must be positive")
        self.gpu_count = int(gpu_count)
        self.run_name = str(run_name)
        self.protocol = str(protocol)
        self.stage = str(stage)
        self._monotonic_clock = monotonic_clock
        self._unix_clock = unix_clock
        self._started_monotonic = float(monotonic_clock())
        self._started_unix = float(unix_clock())
        prior = prior or {}
        if prior and int(prior.get("schema", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported runtime-accounting schema")
        for key, expected in (
            ("run_name", self.run_name),
            ("protocol", self.protocol),
            ("stage", self.stage),
        ):
            if key in prior and str(prior[key]) != expected:
                raise ValueError(
                    f"runtime-accounting {key} mismatch: "
                    f"saved={prior[key]!r} current={expected!r}"
                )
        self._prior_wall = self._finite_nonnegative(
            prior.get("accumulated_wall_seconds", 0.0),
            "prior accumulated_wall_seconds",
        )
        self._prior_gpu = self._finite_nonnegative(
            prior.get("accumulated_gpu_seconds", 0.0),
            "prior accumulated_gpu_seconds",
        )

    @staticmethod
    def _finite_nonnegative(value, label: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
        return value

    def snapshot(self) -> dict:
        elapsed = self._finite_nonnegative(
            self._monotonic_clock() - self._started_monotonic,
            "current invocation elapsed time",
        )
        now = float(self._unix_clock())
        return {
            "schema": SCHEMA_VERSION,
            "scope": "TRAINING_ONLY",
            "run_name": self.run_name,
            "protocol": self.protocol,
            "stage": self.stage,
            "accumulated_wall_seconds": self._prior_wall + elapsed,
            "accumulated_gpu_seconds": self._prior_gpu + elapsed * self.gpu_count,
            "current_invocation_gpu_count": self.gpu_count,
            "current_invocation_started_unix": self._started_unix,
            "last_snapshot_unix": now,
        }


def start_runtime_accounting(
    *,
    gpu_count: int,
    run_name: str,
    protocol: str,
    stage: str,
    prior: dict | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    unix_clock: Callable[[], float] = time.time,
) -> RuntimeTracker:
    global _ACTIVE
    _ACTIVE = RuntimeTracker(
        gpu_count=gpu_count,
        run_name=run_name,
        protocol=protocol,
        stage=stage,
        prior=prior,
        monotonic_clock=monotonic_clock,
        unix_clock=unix_clock,
    )
    return _ACTIVE


def runtime_snapshot() -> dict | None:
    return None if _ACTIVE is None else _ACTIVE.snapshot()


def atomic_write_runtime_sidecar(path: Path, snapshot: dict | None) -> None:
    if snapshot is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)
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


def read_runtime_sidecar(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "schema", "scope", "run_name", "protocol", "stage",
        "accumulated_wall_seconds", "accumulated_gpu_seconds",
    }
    if not required <= payload.keys():
        raise ValueError(f"runtime sidecar missing fields: {sorted(required - payload.keys())}")
    if int(payload["schema"]) != SCHEMA_VERSION or payload["scope"] != "TRAINING_ONLY":
        raise ValueError("invalid runtime sidecar identity")
    for key in ("accumulated_wall_seconds", "accumulated_gpu_seconds"):
        value = float(payload[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid runtime sidecar value for {key}")
    return payload
