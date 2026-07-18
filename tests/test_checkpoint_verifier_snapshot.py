from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier_module():
    path = ROOT / "scripts" / "verify_stage_a_checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_snapshot_binds_payload_size_and_digest(tmp_path: Path):
    verifier = _load_verifier_module()
    checkpoint = tmp_path / "last.pt"
    torch.save({"step": 60_000, "weights": torch.arange(32)}, checkpoint)

    snapshot = verifier.load_if_ready(checkpoint, minimum_step=60_000)

    assert snapshot is not None
    payload, checkpoint_bytes, checkpoint_sha256 = snapshot
    assert payload["step"] == 60_000
    assert checkpoint_bytes == checkpoint.stat().st_size
    assert checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_checkpoint_snapshot_respects_minimum_step(tmp_path: Path):
    verifier = _load_verifier_module()
    checkpoint = tmp_path / "last.pt"
    torch.save({"step": 59_999}, checkpoint)

    assert verifier.load_if_ready(checkpoint, minimum_step=60_000) is None
