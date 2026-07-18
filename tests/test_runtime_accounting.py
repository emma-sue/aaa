import json
from pathlib import Path

import pytest

from scripts import orchestrate
from scripts.runtime_accounting import (
    RuntimeTracker,
    atomic_write_runtime_sidecar,
    read_runtime_sidecar,
)


class MutableClock:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def test_runtime_tracker_resumes_without_double_counting_and_counts_all_gpus():
    monotonic = MutableClock(100.0)
    unix = MutableClock(1_000.0)
    tracker = RuntimeTracker(
        gpu_count=4,
        run_name="aio3_stage_a",
        protocol="aio3",
        stage="a",
        prior={
            "schema": 1,
            "accumulated_wall_seconds": 10.0,
            "accumulated_gpu_seconds": 20.0,
        },
        monotonic_clock=monotonic,
        unix_clock=unix,
    )
    monotonic.value += 5.0
    unix.value += 5.0
    first = tracker.snapshot()
    assert first["accumulated_wall_seconds"] == pytest.approx(15.0)
    assert first["accumulated_gpu_seconds"] == pytest.approx(40.0)

    monotonic.value += 3.0
    unix.value += 3.0
    second = tracker.snapshot()
    assert second["accumulated_wall_seconds"] == pytest.approx(18.0)
    assert second["accumulated_gpu_seconds"] == pytest.approx(52.0)


def test_runtime_sidecar_is_atomic_validated_and_aggregated(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    checkpoint_root = tmp_path / "artifacts/checkpoints"
    expected = {"run_a": 3600.0, "run_b": 7200.0}
    for run_name, gpu_seconds in expected.items():
        payload = {
            "schema": 1,
            "scope": "TRAINING_ONLY",
            "run_name": run_name,
            "protocol": "aio3",
            "stage": "b_predicted",
            "accumulated_wall_seconds": gpu_seconds,
            "accumulated_gpu_seconds": gpu_seconds,
            "current_invocation_gpu_count": 1,
            "current_invocation_started_unix": 1.0,
            "last_snapshot_unix": 2.0,
        }
        path = checkpoint_root / run_name / "runtime_accounting.json"
        atomic_write_runtime_sidecar(path, payload)
        assert read_runtime_sidecar(path) == payload

    total, by_run, metadata = orchestrate.collect_training_gpu_hours()
    assert total == pytest.approx(3.0)
    assert by_run == {"run_a": pytest.approx(1.0), "run_b": pytest.approx(2.0)}
    assert metadata == {
        "run_a": {
            "scope": "TRAINING_ONLY",
            "origin": "EMBEDDED_TRAINER_MONOTONIC_COUNTER",
            "is_estimate": False,
            "last_snapshot_unix": 2.0,
        },
        "run_b": {
            "scope": "TRAINING_ONLY",
            "origin": "EMBEDDED_TRAINER_MONOTONIC_COUNTER",
            "is_estimate": False,
            "last_snapshot_unix": 2.0,
        },
    }
    assert not list(tmp_path.rglob(".runtime_accounting.json.tmp.*"))


def test_runtime_sidecar_rejects_negative_or_mismatched_run(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    path = tmp_path / "artifacts/checkpoints/run_a/runtime_accounting.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema": 1,
        "scope": "TRAINING_ONLY",
        "run_name": "wrong_name",
        "protocol": "aio3",
        "stage": "c",
        "accumulated_wall_seconds": 1.0,
        "accumulated_gpu_seconds": 1.0,
    }))
    with pytest.raises(RuntimeError, match="run mismatch"):
        orchestrate.collect_training_gpu_hours()

    payload = json.loads(path.read_text())
    payload["run_name"] = "run_a"
    payload["accumulated_gpu_seconds"] = -1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="accumulated_gpu_seconds"):
        orchestrate.collect_training_gpu_hours()


def test_both_trainers_persist_runtime_accounting():
    root = Path(__file__).resolve().parents[1]
    single = (root / "scripts/train.py").read_text()
    distributed = (root / "scripts/train_stage_a_ddp.py").read_text()
    for source in (single, distributed):
        assert '"runtime_accounting": accounting' in source
        assert "atomic_write_runtime_sidecar" in source


def test_single_gpu_trainer_reads_sidecar_outside_resume_only_branch():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/train.py").read_text()
    resume_start = source.index("    if args.resume:")
    accounting_start = source.index("    start_runtime_accounting(", resume_start)
    block = source[resume_start:accounting_start]
    sidecar_line = block.index('    sidecar = run_dir / "runtime_accounting.json"')
    assert block[sidecar_line:].startswith(
        '    sidecar = run_dir / "runtime_accounting.json"'
    )
