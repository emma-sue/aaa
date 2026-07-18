import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_paired import degradation_task_macro_ci, read, stratified_macro_ci


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def test_read_rejects_duplicate_paired_keys(tmp_path):
    path = tmp_path / "duplicate.csv"
    row = {"task": "dehaze", "name": "one", "psnr": 30.0, "ssim": 0.9}
    _write(path, [row, row])
    with pytest.raises(ValueError, match="duplicate paired keys"):
        read(path)


def test_stratified_macro_bootstrap_is_deterministic_and_task_balanced():
    tasks = [np.array([1.0, 1.0]), np.array([3.0])]
    first = stratified_macro_ci(tasks, np.random.RandomState(7))
    second = stratified_macro_ci(tasks, np.random.RandomState(7))
    assert first == second == pytest.approx([2.0, 2.0])


def test_degradation_macro_groups_three_denoise_severities_as_one_task():
    values = {
        "dehaze": np.array([3.0]),
        "derain": np.array([3.0]),
        "denoise15": np.array([0.0]),
        "denoise25": np.array([0.0]),
        "denoise50": np.array([0.0]),
    }
    assert degradation_task_macro_ci(values, np.random.RandomState(1)) == pytest.approx([2.0, 2.0])


def test_cli_reports_stratified_macro_ci_and_positive_gate(tmp_path):
    baseline = tmp_path / "baseline.csv"
    method = tmp_path / "method.csv"
    output = tmp_path / "paired.json"
    base_rows = []
    method_rows = []
    for task in ("dehaze", "derain"):
        for index in range(3):
            base_rows.append({"task": task, "name": str(index), "psnr": 30.0, "ssim": 0.90})
            method_rows.append({"task": task, "name": str(index), "psnr": 30.4, "ssim": 0.901})
    _write(baseline, base_rows)
    _write(method, method_rows)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/compare_paired.py"),
        "--baseline", str(baseline), "--method", str(method),
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["macro_task_psnr_delta"] == pytest.approx(0.4)
    assert result["macro_task_psnr_bootstrap_95ci"] == pytest.approx([0.4, 0.4])
    assert result["publication_thresholds"]["macro_task_psnr_ci_lower_gt_0"]
    assert result["publication_go_internal"]
