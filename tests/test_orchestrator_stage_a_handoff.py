from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_orchestrator():
    path = PROJECT_ROOT / "scripts/orchestrate.py"
    spec = importlib.util.spec_from_file_location(
        "srsc_orchestrator_stage_a_handoff", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("protocol", "expected_step", "expected_origin", "expected_workers"),
    [("aio3", 330_500, None, 8), ("aio5", 427_440, "fresh", 6)],
)
def test_protocol_handoff_invokes_exact_terminal_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    expected_step: int,
    expected_origin: str | None,
    expected_workers: int,
):
    module = _load_orchestrator()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    config = tmp_path / f"protocol_{protocol}.yaml"
    config.write_text(yaml.safe_dump({"protocol": protocol, "epochs": 240}))
    run_name = f"{protocol}_stage_a_coarse_seed1415926"
    checkpoint = tmp_path / "artifacts/checkpoints" / run_name / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable-stage-a-transaction")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    observed = {}

    def fake_run(command: list[str], log_name: str):
        observed["command"] = command
        observed["log_name"] = log_name
        assert command[command.index("--minimum-step") + 1] == "1"
        assert int(command[command.index("--expected-epoch") + 1]) == 240
        assert int(command[command.index("--expected-step") + 1]) == expected_step
        assert int(command[command.index("--expected-world-size") + 1]) == 4
        assert int(
            command[command.index("--expected-global-effective-batch") + 1]
        ) == 120
        assert int(command[command.index("--expected-per-gpu-batch") + 1]) == 30
        assert int(command[command.index("--expected-accumulation") + 1]) == 1
        assert int(
            command[command.index("--expected-workers-per-rank") + 1]
        ) == expected_workers
        assert command[command.index("--expected-backend") + 1] == "nccl"
        if expected_origin is None:
            assert "--expected-training-origin" not in command
        else:
            assert command[
                command.index("--expected-training-origin") + 1
            ] == expected_origin
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "status": "pass",
            "checkpoint_sha256": checkpoint_sha,
            "epoch": 240,
            "step": expected_step,
            "checks": {"all_registered_checks": True},
        }))

    monkeypatch.setattr(module, "run", fake_run)
    report = module.ensure_protocol_stage_a_handoff(
        protocol=protocol,
        config=config,
        run_name=run_name,
        checkpoint=checkpoint,
    )
    assert report["status"] == "pass"
    assert observed["log_name"] == f"{protocol}_stage_a_orchestrator_handoff.log"


def test_protocol_handoff_rejects_stale_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_orchestrator()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    config = tmp_path / "protocol_aio3.yaml"
    config.write_text(yaml.safe_dump({"protocol": "aio3", "epochs": 240}))
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"current-checkpoint")

    def fake_run(command: list[str], _log_name: str):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "status": "pass",
            "checkpoint_sha256": "0" * 64,
            "epoch": 240,
            "step": 330_500,
            "checks": {"all_registered_checks": True},
        }))

    monkeypatch.setattr(module, "run", fake_run)
    with pytest.raises(RuntimeError, match="incomplete or stale"):
        module.ensure_protocol_stage_a_handoff(
            protocol="aio3",
            config=config,
            run_name="aio3_stage_a_coarse_seed1415926",
            checkpoint=checkpoint,
        )


def test_locked_val_cache_accepts_producer_checksum_file_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_orchestrator()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    config = tmp_path / "protocol_aio3.yaml"
    config.write_text(yaml.safe_dump({"protocol": "aio3"}))
    checkpoint = tmp_path / "stage_a.pt"
    checkpoint.write_bytes(b"selected-best-stage-a")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cache_dir = (
        tmp_path / "artifacts/cache/stage_a_y1/aio3"
        / checkpoint_sha[:16] / "locked_val"
    )

    def fake_run(_command: list[str], _log_name: str):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "COMPLETE_TWO_PASS_VERIFIED",
            "scope": "locked_val",
            "official_test_forbidden": True,
            "item_count": 2,
            "aggregate_sha256": "a" * 64,
            "bindings": {"stage_a_checkpoint": {"sha256": checkpoint_sha}},
        }
        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (cache_dir / "manifest.sha256").write_text(
            f"{digest}  manifest.json\n", encoding="ascii"
        )

    monkeypatch.setattr(module, "run", fake_run)
    manifest_path = module.ensure_stage_a_locked_val_cache(
        config, checkpoint, "cache.log"
    )
    assert manifest_path == cache_dir / "manifest.json"
    evidence = json.loads(
        (tmp_path / "artifacts/manifests/aio3_stage_a_locked_val_cache.json")
        .read_text()
    )
    assert evidence["stage_a_checkpoint_sha256"] == checkpoint_sha
    assert evidence["item_count"] == 2


@pytest.mark.parametrize(
    "contents",
    ["f" * 64, "f" * 64 + "  wrong.json\n", "not-a-sha  manifest.json\n"],
)
def test_checksum_sidecar_parser_fails_closed(contents: str, tmp_path: Path):
    module = _load_orchestrator()
    sidecar = tmp_path / "manifest.sha256"
    sidecar.write_text(contents)
    with pytest.raises(ValueError, match="SHA256"):
        module._read_named_sha256_sidecar(sidecar, "manifest.json")
