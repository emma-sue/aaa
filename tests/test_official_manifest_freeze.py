import json
from pathlib import Path

import pytest
import torch

from scripts import eval_locked, orchestrate


def _candidate(candidate_id, model, checkpoint, output):
    return {
        "candidate_id": candidate_id,
        "model": model,
        "checkpoint_path": str(checkpoint),
        "output_path": str(output),
    }


def _frozen_inputs(tmp_path: Path, monkeypatch, checkpoint_specs):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    monkeypatch.setattr(eval_locked, "ROOT", tmp_path)
    stats = tmp_path / "coordinate_stats.json"
    stats.write_text('{"frozen": true}\n')
    config = tmp_path / "stage_c_aio3.yaml"
    config.write_text(
        f"protocol: aio3\ncoordinate_stats: {stats}\n"
        f"data_root: {tmp_path / 'data'}\nsplit_manifest: {tmp_path / 'split.json'}\n"
    )
    config_sha = eval_locked.sha256_file(config)
    data_manifest = tmp_path / "artifacts/manifests/official_dataset_aio3.json"
    data_manifest.parent.mkdir(parents=True)
    data_manifest.write_text('{"frozen": true}\n')
    code_contract = {"tests/eval.py": "e" * 64}
    monkeypatch.setattr(
        orchestrate, "freeze_official_data_manifest", lambda _cfg: data_manifest
    )
    monkeypatch.setattr(
        orchestrate, "official_evaluation_code_hashes", lambda: code_contract
    )
    monkeypatch.setattr(
        eval_locked, "official_evaluation_code_hashes", lambda: code_contract
    )
    monkeypatch.setattr(
        eval_locked,
        "validate_official_data_manifest",
        lambda path, _cfg, expected: json.loads(Path(path).read_text())
        if eval_locked.sha256_file(Path(path)) == expected
        else (_ for _ in ()).throw(PermissionError("data drift")),
    )
    run_contract = tmp_path / "checkpoints/run_contract.json"
    run_contract.parent.mkdir(parents=True)
    run_contract.write_text('{"frozen": true}\n')
    run_sha = eval_locked.sha256_file(run_contract)
    for checkpoint, stage, feedback, model_value in checkpoint_specs:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": {"weight": torch.tensor([float(model_value)])},
            "config_sha256": config_sha,
            "split_manifest_sha256": "a" * 64,
            "args": {
                "stage": stage,
                "feedback": feedback,
                "run_contract_sha256": run_sha,
            },
        }, checkpoint)
    return config


def test_orchestrator_freezes_manifest_compatible_with_eval_guard(tmp_path, monkeypatch):
    first = tmp_path / "checkpoints/o7.pt"
    second = tmp_path / "checkpoints/baseline.pt"
    config = _frozen_inputs(tmp_path, monkeypatch, [
        (first, "c", "O7", 1),
        (second, "baseline_ft", "O0", 2),
    ])
    first_output = tmp_path / "metrics/o7.csv"
    second_output = tmp_path / "metrics/baseline.csv"
    candidates = [
        _candidate("aio3-stage-c-o7", "srsc", first, first_output),
        _candidate("aio3-baseline", "baseline", second, second_output),
    ]

    manifest = orchestrate.freeze_official_candidate_manifest(
        "aio3", config, candidates
    )
    original = manifest.read_bytes()
    assert orchestrate.freeze_official_candidate_manifest(
        "aio3", config, candidates
    ).read_bytes() == original
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "FROZEN"
    assert len(payload["candidates"]) == 2

    _, _, selected = eval_locked.validate_frozen_official_manifest(
        manifest,
        protocol="aio3",
        config_path=config,
        config_sha256=eval_locked.sha256_file(config),
        model_kind="srsc",
        checkpoint_path=first,
        output_path=first_output,
    )
    assert selected["candidate_id"] == "aio3-stage-c-o7"


def test_frozen_manifest_rejects_candidate_or_checkpoint_drift(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/model.pt"
    config = _frozen_inputs(
        tmp_path, monkeypatch, [(checkpoint, "c", "O7", 1)]
    )
    output = tmp_path / "official.csv"
    candidates = [_candidate("aio3-o7", "srsc", checkpoint, output)]
    orchestrate.freeze_official_candidate_manifest("aio3", config, candidates)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["weight"] = torch.tensor([2.0])
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="manifest drift"):
        orchestrate.freeze_official_candidate_manifest("aio3", config, candidates)


def test_every_orchestrated_official_eval_passes_the_frozen_manifest():
    source = open(orchestrate.__file__).read()
    official_section = source[source.index('review_contract(f"{args.protocol}_before_official_test")'):]
    official_section = official_section[:official_section.index("# Additional OOD/local-composite")]
    assert "freeze_official_candidate_manifest" in official_section
    assert official_section.count('"--official-manifest", str(official_manifest)') == 2
    for feedback in ("O0", "O1", "O2", "O7"):
        assert f'"{feedback}"' in official_section
