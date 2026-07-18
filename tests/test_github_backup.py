from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_repro_snapshot import build_snapshot, safe_source_file


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _minimal_source(root: Path) -> None:
    _write(root / "README.md", "# test\n")
    _write(root / ".gitignore", "*.pt\n")
    _write(root / "src/model.py", "VALUE = 1\n")
    _write(root / "configs/protocol_aio3.yaml", "protocol: aio3\n")
    _write(root / "artifacts/manifests/locked_split_aio3.json", "{}\n")


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
    payload = b"immutable-best"
    (run_dir / "best.pt").write_bytes(payload)
    (run_dir / "last.pt").write_bytes(b"resume")
    (run_dir / "top3.json").write_text(json.dumps([{
        "score": 1.0, "epoch": 5, "step": 10, "checkpoint": "best.pt",
    }]))
    build_snapshot(source, destination)
    index = json.loads((destination / "recovery/CHECKPOINTS.json").read_text())
    assert index["current_best"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert index["current_best"]["release_state"] == "planned"
    assert not list(destination.rglob("*.pt"))
