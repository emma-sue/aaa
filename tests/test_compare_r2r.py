import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reference/r2r_cvpr2026_tables.json"


@pytest.mark.parametrize(
    "protocol,table_key",
    [("aio3", "table1_aio3"), ("aio5", "table2_aio5")],
)
def test_r2r_reference_and_plus_point_three_targets(tmp_path, protocol, table_key):
    payload = json.loads(REFERENCE.read_text())
    table = payload[table_key]
    method = {
        task: {"psnr": values["psnr"] + 0.30, "ssim": values["ssim"] + 0.001}
        for task, values in table.items()
        if task != "reported_average"
    }
    method_path = tmp_path / "method.json"
    output_path = tmp_path / "comparison.json"
    method_path.write_text(json.dumps(method))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_r2r.py"),
            "--protocol",
            protocol,
            "--method-summary",
            str(method_path),
            "--reference",
            str(REFERENCE),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(output_path.read_text())
    assert result["metric_protocol"] == "full RGB images"
    assert result["reference_pdf_sha256"] == payload["source_sha256"]
    assert result["user_target_met"]
    assert result["tasks_psnr_ge_plus_0.30"] == 5
    for item in result["deltas"].values():
        assert item["psnr"] == pytest.approx(0.30)
        assert item["psnr_ge_plus_0.30"]


def test_r2r_reference_rejects_pdf_hash_drift(tmp_path):
    payload = copy.deepcopy(json.loads(REFERENCE.read_text()))
    payload["source_sha256"] = "0" * 64
    reference = tmp_path / "bad_reference.json"
    reference.write_text(json.dumps(payload))
    method = {
        task: {"psnr": values["psnr"], "ssim": values["ssim"]}
        for task, values in payload["table1_aio3"].items()
        if task != "reported_average"
    }
    method_path = tmp_path / "method.json"
    method_path.write_text(json.dumps(method))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_r2r.py"),
            "--protocol",
            "aio3",
            "--method-summary",
            str(method_path),
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "out.json"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "SHA256 drift" in completed.stderr
