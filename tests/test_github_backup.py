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
    latest_scientific_head,
    publish_release,
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


def _scientific_contract(
    source: Path, run_name: str, stage: str, *, protocol: str = "aio3",
    max_steps: int | None = None,
) -> tuple[Path, str]:
    config = source / f"configs/protocol_{protocol}.yaml"
    split = source / f"artifacts/manifests/locked_split_{protocol}.json"
    if not config.exists():
        _write(config, f"protocol: {protocol}\n")
    if not split.exists():
        _write(split, json.dumps({"protocol": protocol}) + "\n")
    code = source / "src/model.py"
    run_dir = source / "artifacts/checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": 1, "run_name": run_name, "stage": stage, "feedback": "O7",
        "max_steps": max_steps,
        "effective_config": {
            "protocol": protocol, "split_manifest": str(split.resolve()),
        },
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "code_sha256": {
            "src/model.py": hashlib.sha256(code.read_bytes()).hexdigest(),
        },
    }
    contract_path = run_dir / "run_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract_path, hashlib.sha256(contract_path.read_bytes()).hexdigest()


def _scientific_model_only(
    path: Path, *, run_name: str, stage: str, protocol: str,
    contract_sha: str, epoch: int, kind: str,
) -> str:
    source = path.parents[3]
    config = source / f"configs/protocol_{protocol}.yaml"
    split = source / f"artifacts/manifests/locked_split_{protocol}.json"
    torch.save({
        "model": {"marker": torch.tensor(epoch)},
        "epoch": epoch, "step": epoch * 10, "batch_in_epoch": 0,
        "config": {"protocol": protocol, "split_manifest": str(split.resolve())},
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "run_contract_sha256": contract_sha,
        "checkpoint_kind": kind,
        "args": {
            "run_name": run_name, "stage": stage, "feedback": "O7",
            "config": str(config.resolve()), "run_contract_sha256": contract_sha,
        },
    }, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    # A Git-backed recovery bundle is not valid between export and commit:
    # files may exist on disk but are not clone-recoverable until tracked.
    with pytest.raises(RuntimeError, match="not a closed set"):
        verify_manifest(destination)
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


def test_generalized_runs_publish_formal_but_keep_pilot_index_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "mirror"
    _minimal_source(source)

    formal_name = "aio5_stage_c_o7_s1415926"
    _, formal_contract_sha = _scientific_contract(
        source, formal_name, "c", protocol="aio5"
    )
    formal_dir = source / "artifacts/checkpoints" / formal_name
    formal_model = formal_dir / "formal_best_model.pt"
    formal_sha = _scientific_model_only(
        formal_model, run_name=formal_name, stage="c", protocol="aio5",
        contract_sha=formal_contract_sha, epoch=8,
        kind="formal_locked_val_best_model_only",
    )
    (formal_dir / "formal_complete.json").write_text(json.dumps({
        "run": formal_name, "model": formal_model.name,
        "model_sha256": formal_sha,
        "selected_locked_val": {"macro_psnr": 33.1},
        "run_contract_sha256": formal_contract_sha,
    }, indent=2) + "\n")

    pilot_name = "aio3_oracle_o7_pilot_n10_s1415926"
    _, pilot_contract_sha = _scientific_contract(
        source, pilot_name, "b_oracle", max_steps=10
    )
    pilot_dir = source / "artifacts/checkpoints" / pilot_name
    pilot_model = pilot_dir / "pilot_model.pt"
    pilot_sha = _scientific_model_only(
        pilot_model, run_name=pilot_name, stage="b_oracle", protocol="aio3",
        contract_sha=pilot_contract_sha, epoch=1, kind="completed_pilot_model_only",
    )
    (pilot_dir / "pilot_complete.json").write_text(json.dumps({
        "run": pilot_name, "model": pilot_model.name,
        "model_sha256": pilot_sha, "max_steps": 10,
        "selected_locked_val": {"macro_psnr": 30.0},
        "run_contract_sha256": pilot_contract_sha,
    }, indent=2) + "\n")

    smoke_name = "aio3_oracle_o7_smoke_s1415926"
    _scientific_contract(source, smoke_name, "b_oracle")
    official = source / "artifacts/checkpoints/official_promptir"
    official.mkdir(parents=True)
    _write(official / "run_contract.json", "{}\n")

    build_snapshot(source, destination)
    index = json.loads((destination / "recovery/CHECKPOINTS.json").read_text())
    assert set(index["runs"]) == {formal_name, pilot_name}
    formal = index["runs"][formal_name]
    assert formal["publish_large_assets"] is True
    assert formal["formal_best"]["release_state"] == "planned"
    assert formal["formal_best"]["release_tag"].startswith(
        "formal-aio5-c-aio5-stage-c-o7-s1415926-e0008-s0000080"
    )
    pilot = index["runs"][pilot_name]
    assert pilot["publish_large_assets"] is False
    assert pilot["pilot_model"]["release_tag"] is None
    assert pilot["pilot_model"]["release_state"] == "index_only_not_published"
    assert not list(destination.rglob("*.pt"))
    assert (destination / formal["run_contract"]["path"]).is_file()

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
    details = verify_checkpoint(destination, formal_model)
    assert details["provenance_status"] == "MODEL_ONLY_CHECKPOINT_AND_GIT_CONTRACT_BOUND"


def test_latest_head_ignores_older_backfill_and_newer_rolling_resume() -> None:
    top_old = {
        "sha256": "old", "release_tag": "best-e0115", "mtime_ns": 100,
        "step": 115, "selection": "top3",
    }
    top_current = {
        "sha256": "current", "release_tag": "best-e0145", "mtime_ns": 200,
        "step": 145, "selection": "top3",
    }
    resume = {
        "sha256": "resume", "release_tag": "resume-aio3-stage-a", "mtime_ns": 300,
        "step": 153, "selection": "resume-last",
    }
    index = {"runs": {"aio3_stage_a_coarse_seed1415926": {
        "publish_large_assets": True, "top3": [top_current, top_old],
        "current_best": top_current, "formal_best": None,
        "resume_local_latest": resume,
    }}}
    state = {"runs": {"aio3_stage_a_coarse_seed1415926": {
        "best_sha256s": ["old", "current"], "resume_sha256": "resume",
    }}}
    assert latest_scientific_head(index, state) is top_current


def test_immutable_backfill_explicitly_cannot_be_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    row = {
        "absolute_source_path": str(checkpoint), "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "release_tag": "best-old", "asset_name": "best.pt", "size": len(b"checkpoint"),
    }
    monkeypatch.setattr(
        "scripts.checkpoint_backup_daemon.release_assets",
        lambda _repo, _tag: {"already": {"state": "uploaded"}},
    )
    monkeypatch.setattr(
        "scripts.checkpoint_backup_daemon.asset_matches",
        lambda _row, _digest, _size: True,
    )
    latest_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "scripts.checkpoint_backup_daemon.set_release_latest",
        lambda _repo, tag, flag: latest_calls.append((tag, flag)),
    )
    publish_release("owner/repo", row, immutable=True, make_latest=False)
    assert latest_calls == [("best-old", False)]
