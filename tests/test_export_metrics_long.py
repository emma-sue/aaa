import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import export_metrics_long as exporter


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _locked_fixture(root: Path) -> Path:
    path = (
        root / "artifacts/metrics"
        / "aio3_stage_a_coarse_seed7_locked_val.jsonl"
    )
    path.parent.mkdir(parents=True)
    record = {
        "dehaze": 30.0,
        "derain": 31.0,
        "denoise15": 32.0,
        "denoise25": 29.0,
        "denoise50": 26.0,
        "setting_ssim": {
            "dehaze": 0.90,
            "derain": 0.91,
            "denoise15": 0.92,
            "denoise25": 0.89,
            "denoise50": 0.86,
        },
        "macro_psnr": 29.6,
        "five_setting_mean_ssim": 0.896,
        "epoch": 5,
        "step": 100,
    }
    path.write_text(json.dumps(record) + "\n")
    return path


def _official_fixture(root: Path, *, terminal_status: str = "COMPLETE") -> dict:
    metrics = root / "artifacts/metrics"
    manifests = root / "artifacts/manifests"
    run_dir = root / "artifacts/checkpoints/aio3_stage_c_o7_s7"
    config = root / "configs/stage_c_aio3.yaml"
    for directory in (metrics, manifests, run_dir, config.parent):
        directory.mkdir(parents=True, exist_ok=True)
    config.write_text("protocol: aio3\nseed: 7\n")
    checkpoint = run_dir / "val_epoch030_step0001234.pt"
    checkpoint.write_bytes(b"frozen model artifact")
    checkpoint_sha = exporter.sha256_file(checkpoint)
    output = metrics / "aio3_stage_c_o7_official.csv"
    tasks = ("dehaze", "derain", "denoise15", "denoise25", "denoise50")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "name", "psnr", "ssim"])
        writer.writeheader()
        for index, task in enumerate(tasks):
            writer.writerow({
                "task": task,
                "name": f"{task}-one",
                "psnr": 30.0 + index,
                "ssim": 0.90 + index / 100,
            })
    candidate_id = "aio3-stage-c-o7"
    manifest = manifests / "official_candidates_aio3.json"
    config_sha = exporter.sha256_file(config)
    manifest_payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "protocol": "aio3",
        "config_path": str(config.resolve()),
        "config_sha256": config_sha,
        "candidates": [{
            "candidate_id": candidate_id,
            "model": "srsc",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "output_paths": [str(output.resolve())],
        }],
    }
    _write_json(manifest, manifest_payload)
    manifest_sha = exporter.sha256_file(manifest)
    ledger = manifests / "official_test_aio3_consumption.json"
    meta = {
        "split": "official_test",
        "protocol": "aio3",
        "model": "srsc",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "paper_comparable_full_image": True,
        "candidate_id": candidate_id,
        "official_manifest": str(manifest.resolve()),
        "official_manifest_sha256": manifest_sha,
        "official_ledger": str(ledger.resolve()),
    }
    summary = {
        task: {"psnr": 30.0 + index, "ssim": 0.90 + index / 100, "n": 1}
        for index, task in enumerate(tasks)
    }
    summary["aggregates"] = {
        "five_setting_mean": {"psnr": 32.0, "ssim": 0.92},
        "task_macro": {"psnr": 31.5, "ssim": 0.915},
        "denoise_task_mean": {"psnr": 33.0, "ssim": 0.93},
    }
    summary["_meta"] = meta
    summary_path = output.with_suffix(".json")
    _write_json(summary_path, summary)
    record = {
        **meta,
        "status": "COMPLETE",
        "rows": 5,
        "csv": str(output.resolve()),
        "csv_sha256": exporter.sha256_file(output),
        "summary": str(summary_path.resolve()),
        "summary_sha256": exporter.sha256_file(summary_path),
    }
    record_path = manifests / f"official_test_aio3_srsc_{checkpoint_sha[:16]}.json"
    _write_json(record_path, record)
    entry = {
        "candidate_id": candidate_id,
        "model": "srsc",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "output": str(output.resolve()),
        "status": terminal_status,
        "record": str(record_path.resolve()),
        "record_sha256": exporter.sha256_file(record_path),
        "csv_sha256": exporter.sha256_file(output),
        "summary_sha256": exporter.sha256_file(summary_path),
    }
    _write_json(ledger, {
        "schema_version": 1,
        "protocol": "aio3",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "consumptions": [entry],
    })
    return {
        "output": output,
        "summary": summary_path,
        "manifest": manifest,
        "ledger": ledger,
        "record": record_path,
    }


def test_locked_val_is_normalized_with_provenance(tmp_path):
    source = _locked_fixture(tmp_path)
    rows = exporter.collect_rows(tmp_path)
    assert len(rows) == 12
    assert {row["metric"] for row in rows} == {"psnr", "ssim"}
    assert {row["scope"] for row in rows} == {"locked_val"}
    assert {row["protocol"] for row in rows} == {"aio3"}
    assert {row["stage"] for row in rows} == {"a"}
    assert {row["model_kind"] for row in rows} == {"srsc"}
    assert {row["seed"] for row in rows} == {7}
    assert {row["source_sha256"] for row in rows} == {
        hashlib.sha256(source.read_bytes()).hexdigest()
    }
    assert {row["manifest_path"] for row in rows} == {""}
    macro = [
        row for row in rows
        if row["task"] == "five_setting_mean" and row["metric"] == "psnr"
    ]
    assert len(macro) == 1 and macro[0]["value"] == pytest.approx(29.6)


def test_unledgered_official_named_files_are_never_collected(tmp_path):
    metrics = tmp_path / "artifacts/metrics"
    metrics.mkdir(parents=True)
    (metrics / "aio3_fake_official.csv").write_text("task,name,psnr,ssim\n")
    _write_json(metrics / "aio3_fake_official.json", {
        "_meta": {"split": "official_test", "protocol": "aio3"}
    })
    assert exporter.collect_rows(tmp_path) == []


def test_complete_manifest_and_ledger_admit_official_summary(tmp_path):
    evidence = _official_fixture(tmp_path)
    rows = exporter.collect_rows(tmp_path)
    assert len(rows) == 16
    assert {row["scope"] for row in rows} == {"official_test"}
    assert {row["run_name"] for row in rows} == {"aio3_stage_c_o7_s7"}
    assert {row["stage"] for row in rows} == {"c"}
    assert {row["feedback"] for row in rows} == {"O7"}
    assert {row["seed"] for row in rows} == {7}
    assert {row["epoch"] for row in rows} == {30}
    assert {row["step"] for row in rows} == {1234}
    assert {row["manifest_path"] for row in rows} == {
        str(evidence["manifest"].resolve())
    }
    assert {row["ledger_path"] for row in rows} == {
        str(evidence["ledger"].resolve())
    }


def test_started_or_failed_official_consumption_is_not_publishable(tmp_path):
    _official_fixture(tmp_path, terminal_status="STARTED")
    assert exporter.collect_rows(tmp_path) == []


def test_complete_official_transaction_hash_drift_fails_closed(tmp_path):
    evidence = _official_fixture(tmp_path)
    evidence["summary"].write_text(evidence["summary"].read_text() + "\n")
    with pytest.raises(RuntimeError, match="artifact SHA256 mismatch"):
        exporter.collect_rows(tmp_path)


def test_atomic_export_is_stable_and_leaves_no_temporary_file(tmp_path):
    _locked_fixture(tmp_path)
    output, first_rows = exporter.export_metrics(tmp_path)
    first_bytes = output.read_bytes()
    output, second_rows = exporter.export_metrics(tmp_path)
    assert first_rows == second_rows
    assert output.read_bytes() == first_bytes
    assert not list(output.parent.glob("metrics_long.csv.tmp.*"))
    with output.open(newline="") as handle:
        restored = list(csv.DictReader(handle))
    assert len(restored) == len(first_rows)
    assert tuple(restored[0]) == exporter.FIELDS


def test_export_output_cannot_escape_artifacts_metrics(tmp_path):
    (tmp_path / "artifacts/metrics").mkdir(parents=True)
    with pytest.raises(ValueError, match="must remain"):
        exporter.export_metrics(tmp_path, tmp_path / "elsewhere.csv")


def test_summary_rejects_missing_task_and_nonfinite_metric():
    incomplete = {
        "dehaze": 30.0,
        "derain": 31.0,
        "denoise15": 32.0,
        "denoise25": 29.0,
    }
    with pytest.raises(RuntimeError, match="lacks required tasks"):
        exporter.summary_measurements(incomplete, "aio3")
    incomplete["denoise50"] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        exporter.summary_measurements(incomplete, "aio3")


def test_exporter_has_no_official_data_or_evaluator_entrypoint_import():
    source = Path(exporter.__file__).read_text()
    assert "build_test_sets" not in source
    assert "evaluate_rows" not in source
    assert "unlock_official" not in source
    assert "torch.load" not in source
