import csv
from pathlib import Path

import pytest

from scripts.compare_locked_paired import compare_locked, read_metrics, read_psnr


def write_rows(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "name", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_locked_paired_reports_mean_median_wins_worst_and_bootstrap(tmp_path):
    baseline = tmp_path / "baseline.csv"
    method = tmp_path / "method.csv"
    rows = [
        {
            "task": task,
            "name": f"{task}_{index}",
            "psnr": 30.0 + index,
            "ssim": 0.90 + index * 0.001,
        }
        for task in ("dehaze", "derain", "denoise15", "denoise25", "denoise50")
        for index in range(4)
    ]
    write_rows(baseline, rows)
    improved = [
        dict(
            row,
            psnr=float(row["psnr"]) + 0.2,
            ssim=float(row["ssim"]) + 0.002,
        )
        for row in rows
    ]
    write_rows(method, improved)
    result = compare_locked(baseline, method)
    assert result["five_setting_mean_psnr_delta"] == pytest.approx(0.2)
    assert result["degradation_task_macro_psnr_delta"] == pytest.approx(0.2)
    assert result["all_images_psnr_delta"]["median"] == pytest.approx(0.2)
    assert result["all_images_psnr_delta"]["win_rate"] == 1.0
    assert result["all_images_psnr_delta"]["worst_10pct_mean"] == pytest.approx(0.2)
    assert result["five_setting_mean_ssim_delta"] == pytest.approx(0.002)
    assert result["degradation_task_macro_ssim_delta"] == pytest.approx(0.002)
    assert result["tasks_ssim"]["dehaze"]["mean"] == pytest.approx(0.002)
    assert result["all_images_ssim_delta"]["median"] == pytest.approx(0.002)
    assert result["all_images_ssim_delta"]["win_rate"] == 1.0
    assert result["five_setting_ssim_bootstrap_95ci"] == pytest.approx(
        [0.002, 0.002]
    )
    assert result["degradation_task_ssim_bootstrap_95ci"] == pytest.approx(
        [0.002, 0.002]
    )
    assert result["bootstrap_draws"] == 10_000
    assert result["five_setting_bootstrap_95ci"][0] > 0.0


def test_locked_paired_rejects_duplicate_or_mismatched_keys(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    write_rows(duplicate, [
        {"task": "dehaze", "name": "x", "psnr": 30.0, "ssim": 0.9},
        {"task": "dehaze", "name": "x", "psnr": 31.0, "ssim": 0.91},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        read_psnr(duplicate)

    baseline = tmp_path / "baseline.csv"
    method = tmp_path / "method.csv"
    write_rows(baseline, [
        {"task": "dehaze", "name": "x", "psnr": 30.0, "ssim": 0.9}
    ])
    write_rows(method, [
        {"task": "dehaze", "name": "y", "psnr": 31.0, "ssim": 0.91}
    ])
    with pytest.raises(ValueError, match="keys differ"):
        compare_locked(baseline, method)


def test_locked_reader_requires_finite_joint_psnr_ssim(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("task,name,psnr\ndehaze,x,30\n")
    with pytest.raises(ValueError, match="invalid or empty"):
        read_metrics(missing)

    nonfinite = tmp_path / "nonfinite.csv"
    write_rows(nonfinite, [
        {"task": "dehaze", "name": "x", "psnr": 30.0, "ssim": "nan"}
    ])
    with pytest.raises(ValueError, match="non-finite"):
        read_metrics(nonfinite)
    with pytest.raises(ValueError, match="non-finite"):
        read_psnr(nonfinite)
