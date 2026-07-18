import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_locked


def _write_shared_stage_b_marker(root: Path, *, authorized: bool = True) -> Path:
    directory = root / "artifacts/manifests"
    directory.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for protocol, filename in eval_locked.STAGE_B_TERMINAL_ATTESTATION_NAMES.items():
        path = directory / filename
        terminal = {
            "protocol": protocol,
            "stage": "STAGE_B_COMPLETE",
            "predicted_go": True,
            "scientific_go": "GO",
            "capacity_robustness_go": True if protocol == "aio3" else None,
            "decision_revision_sha256": "c" * 64,
        }
        terminal_sha = hashlib.sha256(json.dumps(
            terminal, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        payload = {
            "schema_version": eval_locked.SHARED_STAGE_B_MARKER_SCHEMA_VERSION,
            "status": "FROZEN",
            "protocol": protocol,
            "stage": "STAGE_B_COMPLETE",
            "predicted_go": True,
            "scientific_go": "GO",
            "stage_b_runtime_manifest_sha256": "a" * 64,
            "terminal_decision": terminal,
            "terminal_decision_sha256": terminal_sha,
            "decision_revision_sha256": "c" * 64,
            "capacity_robustness_go": True if protocol == "aio3" else None,
            "official_access_authorized": authorized,
        }
        eval_locked.atomic_write_json(path, payload)
        bindings[protocol] = {
            "path": str(path.resolve()),
            "sha256": eval_locked.sha256_file(path),
        }
    marker = directory / eval_locked.SHARED_STAGE_B_MARKER_NAME
    eval_locked.atomic_write_json(marker, {
        "schema_version": eval_locked.SHARED_STAGE_B_MARKER_SCHEMA_VERSION,
        "status": "FROZEN",
        "marker_path": str(marker.resolve()),
        "protocols": bindings,
        "official_access_authorized": authorized,
    })
    return marker


@pytest.fixture(autouse=True)
def _tiny_official_identity_contract(monkeypatch, request, tmp_path):
    monkeypatch.setattr(eval_locked, "ROOT", tmp_path)
    _write_shared_stage_b_marker(tmp_path)
    code_contract = {"tests/frozen_eval.py": "e" * 64}
    monkeypatch.setattr(
        eval_locked, "official_evaluation_code_hashes", lambda: code_contract
    )

    def validate_data(path, _cfg, expected_sha):
        if eval_locked.sha256_file(Path(path)) != expected_sha:
            raise PermissionError("official data manifest path/SHA256 drift")
        return json.loads(Path(path).read_text())

    if request.node.name != (
        "test_official_data_manifest_detects_content_drift_before_evaluation"
    ):
        monkeypatch.setattr(
            eval_locked, "validate_official_data_manifest", validate_data
        )


def _frozen_fixture(tmp_path: Path):
    config = tmp_path / "protocol.yaml"
    stats = tmp_path / "coordinate_stats.json"
    stats.write_text('{"frozen": true}\n')
    config.write_text(
        f"protocol: aio3\ncoordinate_stats: {stats}\ndata_root: {tmp_path / 'data'}\n"
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    output = tmp_path / "official.csv"
    manifest = tmp_path / "official_candidates.json"
    config_sha = eval_locked.sha256_file(config)
    run_contract = tmp_path / "run_contract.json"
    run_contract.write_text('{"frozen": true}\n')
    run_contract_sha = eval_locked.sha256_file(run_contract)
    data_manifest = tmp_path / "official_dataset_aio3.json"
    data_manifest.write_text('{"frozen": true}\n')
    candidate = {
        "candidate_id": "aio3-final-srsc",
        "model": "srsc",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": eval_locked.sha256_file(checkpoint),
        "run_contract_path": str(run_contract.resolve()),
        "run_contract_sha256": run_contract_sha,
        "checkpoint_contract": {
            "stage": "c",
            "feedback": "O7",
            "run_contract_sha256": run_contract_sha,
            "config_sha256": config_sha,
            "split_manifest_sha256": "a" * 64,
        },
        "output_paths": [str(output.resolve())],
    }
    payload = {
        "schema_version": eval_locked.OFFICIAL_MANIFEST_SCHEMA_VERSION,
        "status": "FROZEN",
        "protocol": "aio3",
        "config_path": str(config.resolve()),
        "config_sha256": config_sha,
        "coordinate_stats_path": str(stats.resolve()),
        "coordinate_stats_sha256": eval_locked.sha256_file(stats),
        "official_data_manifest_path": str(data_manifest.resolve()),
        "official_data_manifest_sha256": eval_locked.sha256_file(data_manifest),
        "evaluation_code_sha256": eval_locked.official_evaluation_code_hashes(),
        "candidates": [candidate],
    }
    eval_locked.atomic_write_json(manifest, payload)
    manifest_sha = eval_locked.sha256_file(manifest)
    return config, checkpoint, output, manifest, config_sha, manifest_sha, candidate


def test_official_eval_lock_fires_before_checkpoint_read(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "eval_locked.py"),
            "--config", str(root / "configs" / "protocol_aio3.yaml"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--model", "srsc",
            "--split", "official_test",
            "--output", str(tmp_path / "metrics.csv"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "official test is locked" in result.stderr


def test_unlocked_official_eval_requires_prefrozen_manifest_before_checkpoint_read(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "eval_locked.py"),
            "--config", str(root / "configs" / "protocol_aio3.yaml"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--model", "srsc",
            "--split", "official_test",
            "--unlock-official-test",
            "--output", str(tmp_path / "metrics.csv"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires --official-manifest" in result.stderr
    assert "missing.pt" not in result.stderr


def test_unlocked_direct_eval_cannot_bypass_shared_stage_b_marker(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "eval_locked.py"),
            "--config", str(tmp_path / "missing-config.yaml"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--model", "srsc",
            "--split", "official_test",
            "--unlock-official-test",
            "--official-manifest", str(tmp_path / "missing-manifest.json"),
            "--output", str(tmp_path / "metrics.csv"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "shared AIO3+AIO5 Stage-B terminal marker" in result.stderr
    assert "missing-config" not in result.stderr
    assert "missing.pt" not in result.stderr


def test_shared_stage_b_marker_rejects_bound_attestation_tamper(tmp_path: Path):
    marker = eval_locked.validate_shared_stage_b_terminal_marker(tmp_path)
    assert marker["official_access_authorized"] is True
    attestation = (
        tmp_path / "artifacts/manifests"
        / eval_locked.STAGE_B_TERMINAL_ATTESTATION_NAMES["aio3"]
    )
    attestation.write_text(attestation.read_text() + "\n")
    with pytest.raises(PermissionError, match="path/SHA256 drift"):
        eval_locked.validate_shared_stage_b_terminal_marker(tmp_path)


def test_manifest_selects_only_exact_frozen_model_checkpoint_and_output(tmp_path: Path):
    config, checkpoint, output, manifest, config_sha, _, candidate = _frozen_fixture(tmp_path)
    _, _, selected = eval_locked.validate_frozen_official_manifest(
        manifest,
        protocol="aio3",
        config_path=config,
        config_sha256=config_sha,
        model_kind="srsc",
        checkpoint_path=checkpoint,
        output_path=output,
    )
    assert selected == candidate

    with pytest.raises(PermissionError, match="not an allowed frozen"):
        eval_locked.validate_frozen_official_manifest(
            manifest,
            protocol="aio3",
            config_path=config,
            config_sha256=config_sha,
            model_kind="srsc",
            checkpoint_path=tmp_path / "new_checkpoint.pt",
            output_path=output,
        )
    with pytest.raises(PermissionError, match="not an allowed frozen"):
        eval_locked.validate_frozen_official_manifest(
            manifest,
            protocol="aio3",
            config_path=config,
            config_sha256=config_sha,
            model_kind="srsc",
            checkpoint_path=checkpoint,
            output_path=tmp_path / "new_output.csv",
        )


def test_checkpoint_internal_contract_must_match_frozen_candidate(tmp_path: Path):
    _, _, _, _, config_sha, _, candidate = _frozen_fixture(tmp_path)
    payload = {
        "config_sha256": config_sha,
        "split_manifest_sha256": "a" * 64,
        "args": {
            "stage": "c",
            "feedback": "O7",
            "run_contract_sha256": candidate["run_contract_sha256"],
        },
    }
    eval_locked.validate_checkpoint_contract(payload, candidate)
    payload["args"]["feedback"] = "O6"
    with pytest.raises(PermissionError, match="checkpoint-internal"):
        eval_locked.validate_checkpoint_contract(payload, candidate)


def test_official_data_manifest_detects_content_drift_before_evaluation(
    tmp_path: Path, monkeypatch,
):
    class Paired:
        def __init__(self, pair):
            self.pairs = [pair]

        def __len__(self):
            return 1

    class Denoise:
        def __init__(self, clean, sigma):
            self.clean_paths = [clean]
            self.sigma = sigma

        def __len__(self):
            return 1

    degraded = tmp_path / "input.png"
    clean = tmp_path / "clean.png"
    degraded.write_bytes(b"degraded")
    clean.write_bytes(b"clean")
    sets = {
        "dehaze": Paired((degraded, clean)),
        "derain": Paired((degraded, clean)),
        "denoise15": Denoise(clean, 15),
        "denoise25": Denoise(clean, 25),
        "denoise50": Denoise(clean, 50),
    }
    monkeypatch.setattr(eval_locked, "ROOT", tmp_path)
    monkeypatch.setattr(eval_locked, "build_test_sets", lambda *_args: sets)
    monkeypatch.setitem(
        eval_locked.EXPECTED_OFFICIAL_COUNTS,
        "aio3",
        {task: 1 for task in eval_locked.EXPECTED_TASKS["aio3"]},
    )
    cfg = {"protocol": "aio3", "data_root": str(tmp_path)}
    manifest = eval_locked.freeze_official_data_manifest(cfg)
    frozen_sha = eval_locked.sha256_file(manifest)
    eval_locked.validate_official_data_manifest(manifest, cfg, frozen_sha)
    clean.write_bytes(b"changed-clean-content")
    with pytest.raises(PermissionError, match="image bytes drifted"):
        eval_locked.validate_official_data_manifest(manifest, cfg, frozen_sha)


def test_manifest_binds_exact_config_sha_before_checkpoint_access(tmp_path: Path):
    config, checkpoint, output, manifest, config_sha, _, _ = _frozen_fixture(tmp_path)
    config.write_text("protocol: aio3\nseed: 2\n")
    changed_sha = eval_locked.sha256_file(config)
    assert changed_sha != config_sha
    with pytest.raises(PermissionError, match="config SHA256"):
        eval_locked.validate_frozen_official_manifest(
            manifest,
            protocol="aio3",
            config_path=config,
            config_sha256=changed_sha,
            model_kind="srsc",
            checkpoint_path=checkpoint,
            output_path=output,
        )


def test_protocol_flock_rejects_concurrent_official_process(tmp_path: Path):
    lock_path = tmp_path / "official_test_aio3.flock"
    with eval_locked.protocol_file_lock(lock_path):
        with pytest.raises(RuntimeError, match="already holds"):
            with eval_locked.protocol_file_lock(lock_path):
                pass


def test_consumption_ledger_makes_candidate_one_shot_and_rejects_manifest_drift(
    tmp_path: Path,
):
    config, _, output, manifest, config_sha, manifest_sha, candidate = _frozen_fixture(tmp_path)
    ledger = tmp_path / "official_test_aio3_consumption.json"
    common = dict(
        protocol="aio3",
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        config_sha256=config_sha,
        candidate=candidate,
        output_path=output,
    )
    eval_locked.reserve_official_candidate(ledger, **common)
    stored = json.loads(ledger.read_text())
    assert stored["consumptions"][0]["status"] == "STARTED"
    with pytest.raises(FileExistsError, match="already been consumed"):
        eval_locked.reserve_official_candidate(ledger, **common)

    eval_locked.finalize_official_candidate(
        ledger,
        candidate_id=candidate["candidate_id"],
        status="COMPLETE",
        details={"record_sha256": "0" * 64},
    )
    stored = json.loads(ledger.read_text())
    assert stored["consumptions"][0]["status"] == "COMPLETE"

    drifted_sha = hashlib.sha256(b"different frozen manifest").hexdigest()
    with pytest.raises(PermissionError, match="manifest/config drift"):
        eval_locked.assert_official_candidate_available(
            ledger,
            **{**common, "manifest_sha256": drifted_sha},
        )


def test_manifest_gated_artifact_reuse_requires_complete_ledger(
    tmp_path: Path, monkeypatch,
):
    config, checkpoint, output, manifest, config_sha, manifest_sha, candidate = (
        _frozen_fixture(tmp_path)
    )
    monkeypatch.setattr(eval_locked, "ROOT", tmp_path)
    _, ledger = eval_locked.official_control_paths("aio3")
    eval_locked.reserve_official_candidate(
        ledger,
        protocol="aio3",
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        config_sha256=config_sha,
        candidate=candidate,
        output_path=output,
    )
    rows = [{"task": "dehaze", "name": "one", "psnr": 30.0, "ssim": 0.9}]
    eval_locked.atomic_write_csv(output, rows)
    checkpoint_sha = eval_locked.sha256_file(checkpoint)
    meta = {
        "split": "official_test",
        "protocol": "aio3",
        "model": "srsc",
        "checkpoint_sha256": checkpoint_sha,
        "paper_comparable_full_image": True,
        "candidate_id": candidate["candidate_id"],
        "official_manifest": str(manifest.resolve()),
        "official_manifest_sha256": manifest_sha,
        "official_ledger": str(ledger.resolve()),
    }
    summary_path = output.with_suffix(".json")
    eval_locked.atomic_write_json(summary_path, {"_meta": meta})
    record = {
        **meta,
        "status": "COMPLETE",
        "rows": 1,
        "csv": str(output.resolve()),
        "csv_sha256": eval_locked.sha256_file(output),
        "summary": str(summary_path.resolve()),
        "summary_sha256": eval_locked.sha256_file(summary_path),
    }
    record_path = eval_locked.official_record_path("aio3", "srsc", checkpoint_sha)
    eval_locked.atomic_write_json(record_path, record)
    assert not eval_locked.official_artifacts_complete(
        "aio3", "srsc", checkpoint, output, official_manifest=manifest
    )

    eval_locked.finalize_official_candidate(
        ledger,
        candidate_id=candidate["candidate_id"],
        status="COMPLETE",
        details={
            "record": str(record_path.resolve()),
            "record_sha256": eval_locked.sha256_file(record_path),
            "csv_sha256": eval_locked.sha256_file(output),
            "summary_sha256": eval_locked.sha256_file(summary_path),
        },
    )
    assert eval_locked.official_artifacts_complete(
        "aio3", "srsc", checkpoint, output, official_manifest=manifest
    )


def test_locked_val_does_not_require_official_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "eval_locked.py"),
            "--config", str(root / "configs" / "protocol_aio3.yaml"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--model", "srsc",
            "--split", "locked_val",
            "--output", str(tmp_path / "metrics.csv"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires --official-manifest" not in result.stderr
    assert "missing.pt" in result.stderr
