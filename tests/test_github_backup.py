from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.export_repro_snapshot import build_snapshot, safe_source_file
from scripts.checkpoint_backup_daemon import (
    acquire_daemon_lock,
    bind_checkpoint_index_to_commit,
)
from scripts.verify_recovery_bundle import (
    verify_checkpoint,
    verify_git_snapshot,
    verify_manifest,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _minimal_source(root: Path) -> None:
    _write(root / "README.md", "# test\n")
    _write(
        root / ".gitignore",
        "*.pt\n.checkpoint_hash_cache.json\n.local_backup_state.json\n",
    )
    _write(root / "src/model.py", "VALUE = 1\n")
    _write(root / "configs/protocol_aio3.yaml", "protocol: aio3\n")
    _write(root / "artifacts/manifests/locked_split_aio3.json", "{}\n")


def _checkpoint(path: Path, epoch: int, marker: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = path.parents[3]
    config_sha256 = hashlib.sha256(
        (source / "configs/protocol_aio3.yaml").read_bytes()
    ).hexdigest()
    split_sha256 = hashlib.sha256(
        (source / "artifacts/manifests/locked_split_aio3.json").read_bytes()
    ).hexdigest()
    torch.save({
        "model": {"marker": torch.tensor(marker)},
        "optimizer": {}, "scheduler": {}, "rng": {},
        "epoch": epoch, "step": epoch * 2, "batch_in_epoch": 0,
        "config_sha256": config_sha256, "split_manifest_sha256": split_sha256,
        "distributed_runtime": {"world_size": 4},
        "runtime_contract": None, "data_contract": None, "code_contract": None,
        "training_origin": None,
        "args": {"run_name": "aio3_stage_a_coarse_seed1415926", "stage": "a"},
    }, path)


def test_secret_scan_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.txt"
    fake_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz012345"
    path.write_text("api_key=" + fake_secret + "\n")
    with pytest.raises(RuntimeError, match="Potential secret"):
        safe_source_file(path)


def test_snapshot_excludes_symlinks_and_checkpoints(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    _write(source / "src/real.py", "REAL = True\n")
    (source / "src/link.py").symlink_to(source / "src/real.py")
    checkpoint = source / "artifacts/checkpoints/run/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    build_snapshot(source, destination)
    assert (destination / "src/real.py").is_file()
    assert not (destination / "src/link.py").exists()
    assert not list(destination.rglob("*.pt"))


def test_checkpoint_index_binds_sha_without_tracking_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    run_dir = source / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
    run_dir.mkdir(parents=True)
    _checkpoint(run_dir / "best.pt", 5, marker=1)
    _checkpoint(run_dir / "last.pt", 6, marker=2)
    (run_dir / "top3.json").write_text(json.dumps([{
        "score": 1.0, "epoch": 5, "step": 10, "checkpoint": "best.pt",
    }]))
    build_snapshot(source, destination)
    index = json.loads((destination / "recovery/CHECKPOINTS.json").read_text())
    assert index["current_best"]["sha256"] == hashlib.sha256(
        (run_dir / "best.pt").read_bytes()
    ).hexdigest()
    assert index["current_best"]["release_state"] == "planned"
    assert index["current_best"]["checkpoint_contract"]["epoch"] == 5
    assert not list(destination.rglob("*.pt"))


def test_uploaded_resume_remains_recoverable_while_live_last_advances(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    run_dir = source / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
    run_dir.mkdir(parents=True)
    _checkpoint(run_dir / "last.pt", 1, marker=1)
    build_snapshot(source, destination)
    first_sha = hashlib.sha256((run_dir / "last.pt").read_bytes()).hexdigest()
    previous_index_path = destination / "recovery/CHECKPOINTS.json"
    previous_index = json.loads(previous_index_path.read_text())
    for key in ("resume_latest", "resume_local_latest"):
        previous_index[key]["git_snapshot_commit"] = "a" * 40
        previous_index[key]["git_snapshot_tree"] = "b" * 40
    previous_index_path.write_text(json.dumps(previous_index))
    (destination / ".local_backup_state.json").write_text(json.dumps({
        "resume_sha256": first_sha,
    }))
    _checkpoint(run_dir / "last.pt", 2, marker=2)
    build_snapshot(source, destination)
    index = json.loads((destination / "recovery/CHECKPOINTS.json").read_text())
    assert index["resume_uploaded"]["sha256"] == first_sha
    assert index["resume_latest"]["sha256"] == first_sha
    assert index["resume_latest"]["release_state"] == "uploaded"
    assert index["resume_local_latest"]["sha256"] == hashlib.sha256(
        (run_dir / "last.pt").read_bytes()
    ).hexdigest()
    assert index["resume_local_latest"]["release_state"] == "planned"


def test_snapshot_purges_files_outside_closed_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    build_snapshot(source, destination)
    stale = destination / "stale/arbitrary.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"must not survive")
    build_snapshot(source, destination)
    assert not stale.exists()
    verify_manifest(destination)


def test_manifest_rejects_unexpected_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    build_snapshot(source, destination)
    _write(destination / "unexpected.txt", "not in manifest\n")
    with pytest.raises(RuntimeError, match="not a closed set"):
        verify_manifest(destination)


def test_checkpoint_index_can_bind_real_git_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    run_dir = source / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
    _checkpoint(run_dir / "last.pt", 3, marker=3)
    build_snapshot(source, destination)
    subprocess.run(["git", "init", "-b", "main"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=destination, check=True)
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    subprocess.run(["git", "commit", "-m", "snapshot"], cwd=destination, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    bind_checkpoint_index_to_commit(destination, commit)
    index = json.loads((destination / "recovery/CHECKPOINTS.json").read_text())
    row = index["resume_latest"]
    assert row["git_snapshot_commit"] == commit
    verify_git_snapshot(destination, row)
    verify_manifest(destination)
    verify_checkpoint(destination, run_dir / "last.pt")


def test_checkpoint_contract_drift_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    run_dir = source / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
    _checkpoint(run_dir / "last.pt", 3, marker=3)
    build_snapshot(source, destination)
    subprocess.run(["git", "init", "-b", "main"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=destination, check=True)
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    subprocess.run(["git", "commit", "-m", "snapshot"], cwd=destination, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    bind_checkpoint_index_to_commit(destination, commit)
    index_path = destination / "recovery/CHECKPOINTS.json"
    index = json.loads(index_path.read_text())
    index["resume_latest"]["checkpoint_contract"]["step"] = -1
    index_path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="payload contract differs"):
        verify_checkpoint(destination, run_dir / "last.pt")


def test_backup_daemon_lock_rejects_second_writer(tmp_path: Path) -> None:
    first = acquire_daemon_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another backup daemon"):
            acquire_daemon_lock(tmp_path)
    finally:
        first.close()
    recovered = acquire_daemon_lock(tmp_path)
    recovered.close()


def test_bound_uploaded_resume_is_not_rebound_to_newer_code(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)
    run_dir = source / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
    _checkpoint(run_dir / "last.pt", 3, marker=3)
    build_snapshot(source, destination)
    subprocess.run(["git", "init", "-b", "main"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=destination, check=True)
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=destination, check=True)
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    bind_checkpoint_index_to_commit(destination, first)
    index_path = destination / "recovery/CHECKPOINTS.json"
    index = json.loads(index_path.read_text())
    index["resume_latest"]["release_state"] = "uploaded"
    index_path.write_text(json.dumps(index))
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    subprocess.run(["git", "commit", "-m", "binding"], cwd=destination, check=True)
    second = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    bind_checkpoint_index_to_commit(destination, second)
    rebound = json.loads(index_path.read_text())["resume_latest"]
    assert rebound["git_snapshot_commit"] == first
