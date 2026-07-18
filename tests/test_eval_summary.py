import csv
import json

import pytest

from scripts import eval_locked as eval_module
from scripts.eval_locked import atomic_write_csv, atomic_write_json, summarize_rows


def _rows(protocol="aio3"):
    tasks = (
        ("dehaze", 30.0, 0.90),
        ("derain", 33.0, 0.93),
        ("denoise15", 36.0, 0.96),
        ("denoise25", 33.0, 0.93),
        ("denoise50", 30.0, 0.90),
    ) if protocol == "aio3" else (
        ("dehaze", 30.0, 0.90),
        ("derain", 31.0, 0.91),
        ("denoise25", 32.0, 0.92),
        ("deblur", 33.0, 0.93),
        ("lowlight", 34.0, 0.94),
    )
    return [
        {"task": task, "name": f"{task}_one", "psnr": psnr, "ssim": ssim}
        for task, psnr, ssim in tasks
    ]


def test_aio3_summary_has_explicit_setting_and_task_macros():
    summary = summarize_rows(_rows("aio3"), "aio3")
    assert summary["macro"]["psnr"] == pytest.approx(32.4)
    assert summary["aggregates"]["five_setting_mean"]["psnr"] == pytest.approx(32.4)
    assert summary["aggregates"]["denoise_task_mean"]["psnr"] == pytest.approx(33.0)
    assert summary["aggregates"]["task_macro"]["psnr"] == pytest.approx(32.0)
    assert summary["aggregates"]["legacy_macro_semantics"] == "alias_of_five_setting_mean"


def test_aio5_setting_and_task_macros_are_identical():
    summary = summarize_rows(_rows("aio5"), "aio5")
    assert summary["aggregates"]["five_setting_mean"] == summary["aggregates"]["task_macro"]


def test_summary_rejects_duplicate_or_incomplete_rows():
    rows = _rows("aio3")
    with pytest.raises(RuntimeError, match="duplicate"):
        summarize_rows(rows + [dict(rows[0])], "aio3")
    with pytest.raises(RuntimeError, match="task set mismatch"):
        summarize_rows(rows[:-1], "aio3")


def test_metric_artifacts_are_atomically_replaced(tmp_path):
    rows = _rows("aio3")
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"
    atomic_write_csv(csv_path, rows)
    atomic_write_json(json_path, {"status": "COMPLETE"})
    with csv_path.open(newline="") as handle:
        restored = list(csv.DictReader(handle))
    assert len(restored) == 5
    assert json.loads(json_path.read_text()) == {"status": "COMPLETE"}
    assert not list(tmp_path.glob("*.tmp.*"))


def test_legacy_official_transaction_is_never_reused_without_manifest_gate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(eval_module, "ROOT", tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    output = tmp_path / "official.csv"
    rows = [{"task": "dehaze", "name": "one", "psnr": 30.0, "ssim": 0.9}]
    eval_module.atomic_write_csv(output, rows)
    checkpoint_sha = eval_module.sha256_file(checkpoint)
    summary = {
        "_meta": {
            "split": "official_test",
            "protocol": "aio3",
            "model": "srsc",
            "checkpoint_sha256": checkpoint_sha,
            "paper_comparable_full_image": True,
        }
    }
    summary_path = output.with_suffix(".json")
    eval_module.atomic_write_json(summary_path, summary)
    record = dict(summary["_meta"])
    record.update({
        "status": "COMPLETE",
        "rows": 1,
        "csv": str(output.resolve()),
        "csv_sha256": eval_module.sha256_file(output),
        "summary": str(summary_path.resolve()),
        "summary_sha256": eval_module.sha256_file(summary_path),
    })
    record_path = eval_module.official_record_path("aio3", "srsc", checkpoint_sha)
    eval_module.atomic_write_json(record_path, record)
    missing_manifest = tmp_path / "official_candidates_aio3.json"
    assert not eval_module.official_artifacts_complete(
        "aio3", "srsc", checkpoint, output,
        official_manifest=missing_manifest,
    )
    output.write_text("corrupted\n")
    assert not eval_module.official_artifacts_complete(
        "aio3", "srsc", checkpoint, output,
        official_manifest=missing_manifest,
    )
