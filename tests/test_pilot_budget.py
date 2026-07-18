import json

import pytest

from scripts import pilot_budget
from scripts.runtime_accounting import atomic_write_runtime_sidecar


def _sidecar(root, name, stage, wall_hours, gpu_hours):
    path = root / "artifacts/checkpoints" / name / "runtime_accounting.json"
    atomic_write_runtime_sidecar(path, {
        "schema": 1,
        "scope": "TRAINING_ONLY",
        "run_name": name,
        "protocol": "aio3",
        "stage": stage,
        "accumulated_wall_seconds": wall_hours * 3600.0,
        "accumulated_gpu_seconds": gpu_hours * 3600.0,
    })


def test_pilot_budget_sums_only_registered_oracle_and_predicted_pilots(tmp_path):
    _sidecar(tmp_path, "aio3_oracle_o7_pilot_n1000_s1", "b_oracle", 1.0, 1.0)
    _sidecar(tmp_path, "aio3_predicted_o7_pilot_n1000_s1", "b_predicted", 2.0, 2.0)
    _sidecar(tmp_path, "aio3_oracle_o7_formal_s1", "b_oracle", 99.0, 99.0)
    usage = pilot_budget.persist_and_enforce_pilot_budget(tmp_path, "aio3")
    assert usage["status"] == "WITHIN_BUDGET"
    assert usage["summed_trainer_wall_hours"] == pytest.approx(3.0)
    assert usage["training_gpu_hours"] == pytest.approx(3.0)
    persisted = json.loads(
        (tmp_path / "artifacts/manifests/stage_b_pilot_budget_aio3.json").read_text()
    )
    assert persisted == usage


def test_pilot_budget_fails_closed_and_writes_stop_reason(tmp_path):
    _sidecar(tmp_path, "aio3_oracle_o7_pilot_n1000_s1", "b_oracle", 10.1, 10.1)
    _sidecar(tmp_path, "aio3_predicted_o7_pilot_n1000_s1", "b_predicted", 10.1, 10.1)
    with pytest.raises(RuntimeError, match="TIME_BUDGET_EXCEEDED"):
        pilot_budget.persist_and_enforce_pilot_budget(tmp_path, "aio3")
    assert "TIME_BUDGET_EXCEEDED" in (tmp_path / "STOP_REASON.md").read_text()
