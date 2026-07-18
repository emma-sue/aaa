import hashlib
import json
from pathlib import Path

import pytest

from scripts import orchestrate


REQUIRED_SCHEMA = {
    "promptir_parity",
    "stage_a",
    "oracle_sign",
    "oracle_direction",
    "predicted_srsc",
    "scientific_go",
    "publication_go",
    "residual_code_control",
    "selected_model",
    "per_task_deltas",
    "params",
    "macs",
    "gpu_hours",
    "blocking_issues",
    "next_command",
}


def test_persist_decision_is_schema_complete_synchronized_and_renders_reports(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    # Protocol-independent audited evidence must survive pipeline stage updates.
    (reports / "decision.json").write_text(json.dumps({
        "promptir_parity": "PASS",
        "params": {"srsc_lite": 123},
        "macs": {"status": "profiled"},
        "gpu_hours": None,
    }))
    protocol_path = reports / "decision_aio3.json"
    decision = {
        "protocol": "aio3",
        "stage": "ORACLE_FORMAL_TIER1",
        "stage_a": "PASS",
        "oracle_sign": "GO",
        "oracle_direction": "NO_GO",
        "scientific_go": "NO_GO",
        "publication_go": "NO_GO",
        "residual_code_control": "RESIDUAL_BETTER",
        "selected_model": "NO_GO",
        "oracle": {"O7": {"macro_psnr": 31.0}},
        "oracle_sign_delta": 0.04,
        "oracle_direction_delta": -0.01,
        "oracle_vs_magnitude_delta": 0.02,
        "oracle_vs_residual_code_delta": -0.03,
        "oracle_paired_ci_go": False,
        "next_command": "stop before predicted feedback and Stage-C",
    }

    orchestrate.persist_decision(decision, protocol_path)

    assert protocol_path.read_bytes() == (reports / "decision.json").read_bytes()
    payload = json.loads(protocol_path.read_text())
    assert REQUIRED_SCHEMA <= payload.keys()
    assert payload["promptir_parity"] == "PASS"
    assert payload["params"] == {"srsc_lite": 123}
    assert payload["oracle_direction"] == "NO_GO"
    assert payload["scientific_go"] == payload["publication_go"] == "NO_GO"
    oracle_report = (reports / "STAGE_B_ORACLE_REPORT.md").read_text()
    assert "signed p vs U/D" in oracle_report
    assert payload["decision_revision_sha256"] in oracle_report
    assert "Formal predicted feedback has not completed" in (
        reports / "STAGE_B_PREDICTED_REPORT.md"
    ).read_text()
    final = (reports / "FINAL_DECISION.md").read_text()
    assert "14. Scope" in final
    assert "RESIDUAL_BETTER" in final
    assert not list(reports.glob(".*.tmp.*"))


def test_atomic_write_failure_preserves_previous_destination(tmp_path, monkeypatch):
    destination = tmp_path / "decision.json"
    destination.write_text("previous\n")

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(orchestrate.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        orchestrate._atomic_write_text(destination, "new\n")

    assert destination.read_text() == "previous\n"
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_predicted_selection_and_residual_outcomes_are_consistent():
    predicted = {
        "O4": {"macro_psnr": 30.0},
        "O5": {"macro_psnr": 30.1},
        "O6": {"macro_psnr": 30.1},
        "O7": {"macro_psnr": 30.05},
        "O12": {"macro_psnr": 30.2},
    }
    assert orchestrate.residual_code_outcome(0.01) == "SRSC_BETTER"
    assert orchestrate.residual_code_outcome(-0.01) == "RESIDUAL_BETTER"
    assert orchestrate.residual_code_outcome(0.0) == "TIE"
    assert orchestrate.select_predicted_model(predicted, False) == "RESIDUAL_CODE"
    assert orchestrate.select_predicted_model(predicted, True) == "SRSC_LITE"


def _write_local_composite_artifacts(
    output: Path, checkpoint: Path, *, protocol="aio3", model="srsc"
):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("task,name,psnr,ssim\nrain_plus_local_noise25,a.png,30,0.9\n")
    output.with_suffix(".json").write_text(json.dumps({
        "rain_plus_local_noise25": {"psnr": 30.0, "ssim": 0.9, "n": 1},
        "_meta": {
            "protocol": protocol,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model": model,
            "generation_seed": 20260720,
            "included_in_standard_aio_average": False,
        },
    }))


def test_local_composite_cache_is_bound_to_checkpoint_and_protocol(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    output = tmp_path / "metrics" / "aio3_o1_local_composite.csv"
    _write_local_composite_artifacts(output, checkpoint)

    assert orchestrate.local_composite_artifacts_complete(
        "aio3", checkpoint, "srsc", output
    )
    assert not orchestrate.local_composite_artifacts_complete(
        "aio5", checkpoint, "srsc", output
    )
    checkpoint.write_bytes(b"checkpoint-v2")
    assert not orchestrate.local_composite_artifacts_complete(
        "aio3", checkpoint, "srsc", output
    )


def test_local_composite_cache_rejects_corrupted_or_duplicate_csv(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "metrics" / "aio3_o2_local_composite.csv"
    _write_local_composite_artifacts(output, checkpoint)
    output.write_text(
        "task,name,psnr,ssim\n"
        "rain_plus_local_noise25,a.png,30,0.9\n"
        "rain_plus_local_noise25,a.png,not-a-number,0.9\n"
    )
    assert not orchestrate.local_composite_artifacts_complete(
        "aio3", checkpoint, "srsc", output
    )


def test_local_composite_specs_cover_every_compared_joint_feedback():
    source = Path(orchestrate.__file__).read_text()
    specs = source[source.index("local_checkpoint_specs = {"):]
    specs = specs[:specs.index("for key, (checkpoint, model_kind)")]
    for feedback in ("O0", "O1", "O2", "O7"):
        assert f'"{feedback}": (joint_runs["{feedback}"]' in specs
