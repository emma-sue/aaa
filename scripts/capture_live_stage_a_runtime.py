#!/usr/bin/env python3
"""Capture externally observable provenance for the live legacy AIO-3 run.

The four-GPU process was launched before the trainer embedded
``workers_per_rank`` and code/data contracts in every checkpoint.  This tool
does not edit a checkpoint or claim to recover those missing fields.  It
records the live CUDA owners, their whitelisted runtime arguments/environment,
and one immutable checkpoint generation so a later post-hoc attestation can
honestly distinguish observed runtime evidence from checkpoint metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "aio3_stage_a_coarse_seed1415926"
DEFAULT_CHECKPOINT = ROOT / "artifacts/checkpoints" / DEFAULT_RUN / "last.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/manifests/aio3_live_runtime_attestation.json"
ALLOWED_ENVIRONMENT = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
)
EXPECTED_ARGUMENTS = {
    "--config": "configs/protocol_aio3.yaml",
    "--run-name": DEFAULT_RUN,
    "--per-gpu-batch": "30",
    "--accumulation": "1",
    "--workers-per-rank": "8",
}
EXPECTED_ENVIRONMENT = {
    "WORLD_SIZE": "4",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29658",
    "OMP_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--launcher-pid", type=int, default=14592)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_checkpoint_snapshot(path: Path) -> tuple[dict, int, str]:
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        payload = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return payload, size, digest.hexdigest()


def read_cmdline(proc_root: Path, pid: int) -> list[str]:
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return [part.decode("utf-8", errors="strict") for part in raw.split(b"\0") if part]


def read_environment(proc_root: Path, pid: int) -> dict[str, str]:
    raw = (proc_root / str(pid) / "environ").read_bytes()
    environment: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key_bytes, value_bytes = part.split(b"=", 1)
        key = key_bytes.decode("utf-8", errors="strict")
        if key in ALLOWED_ENVIRONMENT:
            environment[key] = value_bytes.decode("utf-8", errors="strict")
    return environment


def process_start_ticks(proc_root: Path, pid: int) -> int:
    # Field 2 may contain spaces and parentheses.  Everything after the final
    # ')' starts at proc-stat field 3; starttime is field 22, hence index 19.
    stat = (proc_root / str(pid) / "stat").read_text()
    suffix = stat[stat.rfind(")") + 2 :].split()
    return int(suffix[19])


def argument_value(tokens: list[str], flag: str) -> str:
    if tokens.count(flag) != 1:
        raise RuntimeError(f"trainer cmdline must contain {flag} exactly once")
    index = tokens.index(flag)
    if index + 1 >= len(tokens):
        raise RuntimeError(f"trainer cmdline has no value for {flag}")
    return tokens[index + 1]


def nvidia_csv(query: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def trainer_cuda_owners(proc_root: Path, run_name: str) -> list[dict]:
    rows = nvidia_csv("compute-apps=pid,gpu_uuid,process_name,used_memory")
    owners: list[dict] = []
    for pid_text, gpu_uuid, process_name, used_memory in rows:
        pid = int(pid_text)
        try:
            tokens = read_cmdline(proc_root, pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if "scripts/train_stage_a_ddp.py" not in tokens:
            continue
        if argument_value(tokens, "--run-name") != run_name:
            continue
        environment = read_environment(proc_root, pid)
        owners.append(
            {
                "pid": pid,
                "gpu_uuid": gpu_uuid,
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
                "cmdline": tokens,
                "environment": environment,
                "start_ticks_since_boot": process_start_ticks(proc_root, pid),
            }
        )
    return owners


def validate_owners(owners: list[dict], run_name: str) -> None:
    if len(owners) != 4:
        raise RuntimeError(f"expected exactly four CUDA trainer owners, found {len(owners)}")
    if len({owner["pid"] for owner in owners}) != 4:
        raise RuntimeError("CUDA trainer PIDs are not unique")
    if len({owner["gpu_uuid"] for owner in owners}) != 4:
        raise RuntimeError("CUDA trainer GPU UUIDs are not unique")
    expected_arguments = dict(EXPECTED_ARGUMENTS)
    expected_arguments["--run-name"] = run_name
    ranks = set()
    local_ranks = set()
    for owner in owners:
        tokens = owner["cmdline"]
        for flag, expected in expected_arguments.items():
            actual = argument_value(tokens, flag)
            if actual != expected:
                raise RuntimeError(f"trainer argument mismatch: {flag}={actual!r} != {expected!r}")
        environment = owner["environment"]
        for key, expected in EXPECTED_ENVIRONMENT.items():
            if environment.get(key) != expected:
                raise RuntimeError(
                    f"trainer environment mismatch: {key}={environment.get(key)!r} != {expected!r}"
                )
        ranks.add(int(environment["RANK"]))
        local_ranks.add(int(environment["LOCAL_RANK"]))
    if ranks != {0, 1, 2, 3} or local_ranks != {0, 1, 2, 3}:
        raise RuntimeError(f"rank coverage mismatch: ranks={ranks}, local_ranks={local_ranks}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    owners = trainer_cuda_owners(args.proc_root, args.run_name)
    validate_owners(owners, args.run_name)
    launcher_tokens = read_cmdline(args.proc_root, args.launcher_pid)
    if "scripts/launch_aio3_stage_a_4x4090.sh" not in launcher_tokens:
        raise RuntimeError("launcher PID does not own the registered AIO-3 launcher")

    payload, checkpoint_bytes, checkpoint_sha256 = immutable_checkpoint_snapshot(
        args.checkpoint
    )
    cfg = payload.get("config") or {}
    if cfg.get("protocol") != "aio3":
        raise RuntimeError("checkpoint is not the registered AIO-3 protocol")
    if payload.get("args", {}).get("run_name") != args.run_name:
        raise RuntimeError("checkpoint run name mismatch")
    runtime = payload.get("distributed_runtime") or {}
    expected_runtime = {
        "world_size": 4,
        "per_gpu_batch": 30,
        "accumulation": 1,
        "global_effective_batch": 120,
        "backend": "nccl",
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"legacy checkpoint runtime mismatch: {runtime!r}")
    config_path = ROOT / EXPECTED_ARGUMENTS["--config"]
    split_path = Path(cfg["split_manifest"])
    if payload.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("checkpoint/config SHA mismatch")
    if payload.get("split_manifest_sha256") != sha256_file(split_path):
        raise RuntimeError("checkpoint/split SHA mismatch")
    if len(payload.get("rng_by_rank") or []) != 4:
        raise RuntimeError("checkpoint does not contain four rank RNG states")

    inventory = [
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory_total),
        }
        for index, uuid, name, memory_total in nvidia_csv(
            "gpu=index,uuid,name,memory.total"
        )
    ]
    if len(inventory) != 4 or {row["uuid"] for row in inventory} != {
        owner["gpu_uuid"] for owner in owners
    }:
        raise RuntimeError("physical GPU inventory does not match CUDA owners")

    evidence_files = {}
    for relative in (
        "artifacts/logs/pipeline_ddp.log",
        "artifacts/logs/ddp_stage_a/rank0.log",
        "artifacts/logs/watchdog.log",
        "reports/PROTOCOL_AMENDMENT_AIO3_BATCH_MIGRATION.md",
    ):
        path = ROOT / relative
        evidence_files[relative] = {
            "bytes_at_capture": path.stat().st_size,
            "sha256_at_capture": sha256_file(path),
        }

    record = {
        "schema": "srsc.legacy_stage_a.live_runtime_attestation.v1",
        "status": "PASS",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_limit": "STATE_EXACT_LEGACY_CODE_DATA_UNPROVEN",
        "claim": (
            "Externally observed runtime evidence only; this does not backfill "
            "or attribute missing launch-time code/data fields to the checkpoint."
        ),
        "run_name": args.run_name,
        "launcher": {
            "pid": args.launcher_pid,
            "cmdline": launcher_tokens,
            "start_ticks_since_boot": process_start_ticks(
                args.proc_root, args.launcher_pid
            ),
        },
        "boot_id": (args.proc_root / "sys/kernel/random/boot_id").read_text().strip(),
        "cuda_owners": sorted(
            owners, key=lambda owner: int(owner["environment"]["RANK"])
        ),
        "gpu_inventory": sorted(inventory, key=lambda row: row["index"]),
        "observed_runtime": {
            **expected_runtime,
            "workers_per_rank": 8,
            "workers_evidence": "live_process_cmdline",
        },
        "checkpoint_observation": {
            "path": str(args.checkpoint.resolve()),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "epoch": int(payload["epoch"]),
            "batch_in_epoch": int(payload["batch_in_epoch"]),
            "step": int(payload["step"]),
            "config_sha256": payload["config_sha256"],
            "split_manifest_sha256": payload["split_manifest_sha256"],
            "rank_rng_width": len(payload["rng_by_rank"]),
            "embedded_runtime": runtime,
            "embedded_workers_per_rank": runtime.get("workers_per_rank"),
            "embedded_code_contract": payload.get("code_contract"),
            "embedded_data_contract": payload.get("data_contract"),
        },
        "evidence_files": evidence_files,
    }
    atomic_json(args.output, record)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "checkpoint_step": int(payload["step"]),
        "trainer_pids": [owner["pid"] for owner in record["cuda_owners"]],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
