from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import stage_b_runtime as runtime
from scripts import preflight_stage_b_runtime as preflight
from scripts import orchestrate


def _write_templates(root: Path, protocol: str) -> tuple[Path, Path, Path]:
    split = root / "split.json"
    split.write_text('{"locked": true}\n')
    stats = root / "stats.json"
    stats.write_text('{"tau_v": 1}\n')
    stage_a = root / "stage_a.pt"
    stage_a.write_bytes(b"selected-stage-a")
    configs = root / "configs"
    configs.mkdir(parents=True)
    common = {
        "protocol": protocol,
        "seed": 7,
        "split_manifest": str(split),
        "coordinate_stats": str(stats),
        "crop_size": 128,
        "micro_batch": 1,
        "accumulation": runtime.EXPECTED_EFFECTIVE_BATCH[protocol],
        "effective_batch": runtime.EXPECTED_EFFECTIVE_BATCH[protocol],
        "workers": 3,
        "epochs": 30,
        "precision": "bf16",
        "model": {"d1_blocks": [2, 2, 2], "d2_blocks": [4, 4, 4]},
    }
    (configs / f"stage_b_{protocol}.yaml").write_text(
        yaml.safe_dump(common, sort_keys=False)
    )
    if protocol == "aio3":
        capacity = dict(common)
        capacity["coordinate_stats"] = str(root / "future_capacity_stats.json")
        capacity["model"] = {
            "d1_blocks": [3, 3, 4], "d2_blocks": [3, 3, 3]
        }
        (configs / "stage_b_aio3_10_10.yaml").write_text(
            yaml.safe_dump(capacity, sort_keys=False)
        )
    for relative in runtime.PREFLIGHT_CODE_RELATIVE_PATHS:
        code = root / relative
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text(f"# immutable mock code: {relative}\n")
    return stage_a, stats, split


def _attempt(protocol: str, pair: tuple[int, int], passed: bool) -> dict:
    probes = [
        {"probe_id": probe_id, "passed": passed, "peak_allocated_bytes": 1}
        for probe_id in runtime.required_probe_ids(protocol)
    ]
    return {
        "micro_batch": pair[0],
        "accumulation": pair[1],
        "effective_batch": pair[0] * pair[1],
        "all_pass": passed,
        "probes": probes,
    }


def _worker_payload(protocol: str, selected_index: int = 1) -> dict:
    candidates = runtime.STAGE_B_RUNTIME_CANDIDATES[protocol]
    attempts = [
        _attempt(protocol, pair, index == selected_index)
        for index, pair in enumerate(candidates[: selected_index + 1])
    ]
    return {
        "schema": runtime.WORKER_RESULT_SCHEMA,
        "protocol": protocol,
        "candidate_order": [list(pair) for pair in candidates],
        "selected_candidate": list(candidates[selected_index]),
        "selected": {
            "micro_batch": candidates[selected_index][0],
            "accumulation": candidates[selected_index][1],
            "effective_batch": (
                candidates[selected_index][0] * candidates[selected_index][1]
            ),
        },
        "all_pass": True,
        "fatal_error": None,
        "scope": "MEMORY_ONLY",
        "scientific_authority": "NONE",
        "official_test_accessed": False,
        "quality_metrics_computed": False,
        "input_bindings": {"mock": True},
        "inputs_unchanged_through_preflight": True,
        "attempts": attempts,
        "hardware": {"name": "mock-gpu"},
    }


def _runner_for(root: Path, payload: dict, calls: list[list[str]]):
    def runner(command: list[str], _log_name: str) -> None:
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        protocol = command[command.index("--protocol") + 1]
        stage_a = Path(command[command.index("--stage-a-checkpoint") + 1])
        main_template = Path(command[command.index("--main-template") + 1])
        main_cfg = yaml.safe_load(main_template.read_text())
        effective_payload = json.loads(json.dumps(payload))
        effective_payload["input_bindings"] = runtime.preflight_input_bindings(
            root,
            protocol,
            stage_a_checkpoint=stage_a,
            coordinate_stats=Path(main_cfg["coordinate_stats"]),
        )
        effective_payload["inputs_unchanged_through_preflight"] = True
        output.write_text(json.dumps(effective_payload, indent=2) + "\n")

    return runner


def test_first_all_pass_is_candidate_order_only():
    assert runtime.first_all_pass(
        "aio3", {(16, 4): False, (8, 8): True, (4, 16): True}
    ) == (8, 8)
    with pytest.raises(RuntimeError, match="no preregistered"):
        runtime.first_all_pass("aio3", {(16, 4): False})


def test_embargo_detects_metric_contract_checkpoint_and_replay(tmp_path: Path):
    assert runtime.stage_b_artifact_evidence(tmp_path, "aio3") == ()
    metric = tmp_path / "artifacts/metrics/aio3_oracle_o7_pilot_locked_val.jsonl"
    metric.parent.mkdir(parents=True)
    metric.write_text("{}\n")
    run_dir = tmp_path / "artifacts/checkpoints/aio3_predicted_o7_formal_s7"
    run_dir.mkdir(parents=True)
    (run_dir / "run_contract.json").write_text("{}\n")
    replay = (
        tmp_path
        / "artifacts/manifests/replay_digests/aio3_capacity_10_10_o7"
    )
    replay.mkdir(parents=True)
    (replay / "epoch001.json").write_text("{}\n")
    evidence = runtime.stage_b_artifact_evidence(tmp_path, "aio3")
    assert str(metric.resolve()) in evidence
    assert str((run_dir / "run_contract.json").resolve()) in evidence
    assert str((replay / "epoch001.json").resolve()) in evidence
    with pytest.raises(RuntimeError, match="permanently embargoed"):
        runtime.assert_stage_b_artifact_embargo_clear(tmp_path, "aio3")


@pytest.mark.parametrize(
    ("protocol", "selected_index"), (("aio3", 1), ("aio5", 2))
)
def test_bundle_freezes_first_pass_and_reuses_read_only(
    tmp_path: Path, protocol: str, selected_index: int
):
    stage_a, stats, _ = _write_templates(tmp_path, protocol)
    payload = _worker_payload(protocol, selected_index)
    calls: list[list[str]] = []
    bundle = runtime.ensure_stage_b_runtime_bundle(
        tmp_path,
        protocol,
        stage_a_checkpoint=stage_a,
        coordinate_stats=stats,
        runner=_runner_for(tmp_path, payload, calls),
    )
    selected = runtime.STAGE_B_RUNTIME_CANDIDATES[protocol][selected_index]
    assert (bundle.micro_batch, bundle.accumulation) == selected
    assert len(calls) == 1
    main = yaml.safe_load(bundle.main_config.read_text())
    assert main["micro_batch"] == selected[0]
    assert main["accumulation"] == selected[1]
    assert main["workers"] == bundle.workers == 3
    assert runtime.runtime_identity_for_config(bundle.main_config) == {
        "stage_b_runtime_family": protocol,
        "stage_b_runtime_role": "main",
        "stage_b_runtime_manifest_path": str(bundle.manifest_path),
        "stage_b_runtime_manifest_sha256": bundle.manifest_sha256,
    }
    if protocol == "aio3":
        assert bundle.capacity_config is not None
        capacity = yaml.safe_load(bundle.capacity_config.read_text())
        assert capacity["micro_batch"] == main["micro_batch"]
        assert capacity["accumulation"] == main["accumulation"]
        assert capacity["workers"] == main["workers"]
        manifest = json.loads(bundle.manifest_path.read_text())
        assert manifest["capacity_preflight_scope"]["coarse_weights"] == (
            "RANDOM_STRUCTURE_ONLY"
        )
        assert not (tmp_path / "future_capacity_stats.json").exists()

    # Existing scientific output is allowed only because the complete frozen
    # bundle already exists; the runner must never be called again.
    metric = tmp_path / f"artifacts/metrics/{protocol}_oracle_o7_locked_val.jsonl"
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text("{}\n")
    reused = runtime.ensure_stage_b_runtime_bundle(
        tmp_path,
        protocol,
        stage_a_checkpoint=stage_a,
        coordinate_stats=stats,
        runner=lambda *_args: pytest.fail("valid bundle must not rerun preflight"),
    )
    assert reused == bundle


def test_missing_manifest_after_stage_b_artifact_cannot_select(tmp_path: Path):
    stage_a, stats, _ = _write_templates(tmp_path, "aio3")
    checkpoint = tmp_path / "artifacts/checkpoints/aio3_oracle_o7_pilot/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"started")
    with pytest.raises(RuntimeError, match="permanently embargoed"):
        runtime.ensure_stage_b_runtime_bundle(
            tmp_path,
            "aio3",
            stage_a_checkpoint=stage_a,
            coordinate_stats=stats,
            runner=lambda *_args: pytest.fail("embargo must precede worker"),
        )


def test_frozen_bundle_drift_fails_without_reselection(tmp_path: Path):
    stage_a, stats, _ = _write_templates(tmp_path, "aio3")
    calls: list[list[str]] = []
    bundle = runtime.ensure_stage_b_runtime_bundle(
        tmp_path,
        "aio3",
        stage_a_checkpoint=stage_a,
        coordinate_stats=stats,
        runner=_runner_for(tmp_path, _worker_payload("aio3"), calls),
    )
    bundle.main_config.write_text(bundle.main_config.read_text() + "# drift\n")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        runtime.ensure_stage_b_runtime_bundle(
            tmp_path,
            "aio3",
            stage_a_checkpoint=stage_a,
            coordinate_stats=stats,
            runner=lambda *_args: pytest.fail("drift must never trigger reselection"),
        )


def test_preflight_binding_rejects_stats_argument_not_used_by_template(tmp_path: Path):
    stage_a, _stats, _split = _write_templates(tmp_path, "aio5")
    other = tmp_path / "other_stats.json"
    other.write_text("{}\n")
    with pytest.raises(RuntimeError, match="does not match the main template"):
        runtime.preflight_input_bindings(
            tmp_path,
            "aio5",
            stage_a_checkpoint=stage_a,
            coordinate_stats=other,
        )


def test_template_drift_during_worker_cannot_be_frozen(tmp_path: Path):
    stage_a, stats, _split = _write_templates(tmp_path, "aio5")

    def drifting_runner(command: list[str], _log_name: str) -> None:
        template = tmp_path / "configs/stage_b_aio5.yaml"
        cfg = yaml.safe_load(template.read_text())
        cfg["workers"] += 1
        template.write_text(yaml.safe_dump(cfg, sort_keys=False))
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_worker_payload("aio5", 0)) + "\n")

    with pytest.raises(RuntimeError, match="inputs drifted"):
        runtime.ensure_stage_b_runtime_bundle(
            tmp_path,
            "aio5",
            stage_a_checkpoint=stage_a,
            coordinate_stats=stats,
            runner=drifting_runner,
        )


def test_run_contract_identity_rejects_missing_worker_evidence(tmp_path: Path):
    stage_a, stats, _ = _write_templates(tmp_path, "aio5")
    bundle = runtime.ensure_stage_b_runtime_bundle(
        tmp_path,
        "aio5",
        stage_a_checkpoint=stage_a,
        coordinate_stats=stats,
        runner=_runner_for(tmp_path, _worker_payload("aio5", 0), []),
    )
    bundle.worker_result.unlink()
    with pytest.raises(RuntimeError, match="worker evidence drift"):
        runtime.runtime_identity_for_config(bundle.main_config)


def test_worker_must_stop_at_first_pass_and_cover_every_probe():
    payload = _worker_payload("aio3", selected_index=1)
    payload["attempts"][0]["all_pass"] = True
    payload["attempts"][0]["probes"] = [
        {**probe, "passed": True} for probe in payload["attempts"][0]["probes"]
    ]
    with pytest.raises(RuntimeError, match="stop immediately"):
        runtime._validate_worker_result(payload, "aio3")
    payload = _worker_payload("aio3", selected_index=0)
    payload["attempts"][0]["probes"].pop()
    with pytest.raises(RuntimeError, match="probe coverage mismatch"):
        runtime._validate_worker_result(payload, "aio3")
    payload = _worker_payload("aio3", selected_index=0)
    payload["quality_metrics_computed"] = True
    with pytest.raises(RuntimeError, match="exceeded its no-metric"):
        runtime._validate_worker_result(payload, "aio3")


def test_frozen_runtime_forbids_ambient_workers():
    config = {"stage_b_runtime_manifest": "/frozen", "workers": 4}
    with pytest.raises(RuntimeError, match="SRSC_TRAIN_WORKERS is forbidden"):
        runtime.assert_no_runtime_worker_override(
            config, {"SRSC_TRAIN_WORKERS": "4"}
        )
    runtime.assert_no_runtime_worker_override(config, {})
    runtime.assert_no_runtime_worker_override({}, {"SRSC_TRAIN_WORKERS": "8"})


def test_stage_b_templates_and_launcher_own_one_worker_contract():
    root = Path(__file__).resolve().parents[1]
    templates = (
        root / "configs/stage_b_aio3.yaml",
        root / "configs/stage_b_aio5.yaml",
        root / "configs/stage_b_aio3_10_10.yaml",
    )
    assert {yaml.safe_load(path.read_text())["workers"] for path in templates} == {8}
    launcher = (root / "scripts/launch_when_data_ready.sh").read_text()
    assert "unset SRSC_TRAIN_WORKERS" in launcher
    assert "export SRSC_TRAIN_WORKERS" not in launcher
    assert "export SRSC_PARALLEL_GPUS=0,1,2,3" in launcher


def test_real_preflight_driver_output_matches_parent_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage_a, _stats, _split = _write_templates(tmp_path, "aio3")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(
        preflight,
        "largest_locked_val_shape",
        lambda _config: {
            "height": 128,
            "width": 128,
            "area": 128 * 128,
            "task": "mock",
            "name": "mock",
            "index": 0,
            "clean_path": "/mock",
        },
    )
    observed_workers = []

    def fake_worker(**kwargs):
        observed_workers.append(kwargs)
        return {
                "schema": preflight.SCHEMA,
                "status": "PASS",
                "train_step": {
                    "memory": {
                        "peak_allocated_bytes": 10, "headroom_pass": True
                    }
                },
                "native_val": {
                    "memory": {
                        "peak_allocated_bytes": 20, "headroom_pass": True
                    }
                },
            "template_role": kwargs["role"],
            "stage": kwargs["stage"],
            "feedback": kwargs["feedback"],
        }

    monkeypatch.setattr(preflight, "_run_worker", fake_worker)
    monkeypatch.setattr(
        preflight, "_validate_worker_for_probe", lambda *_args, **_kwargs: "PASS"
    )
    output = tmp_path / "driver-output.json"
    args = SimpleNamespace(
        root=str(tmp_path),
        protocol="aio3",
        stage_a_checkpoint=str(stage_a),
        main_template=str(tmp_path / "configs/stage_b_aio3.yaml"),
        capacity_template=str(tmp_path / "configs/stage_b_aio3_10_10.yaml"),
        capacity_stage_a_checkpoint=None,
        candidates_json=json.dumps(
            [list(pair) for pair in runtime.STAGE_B_RUNTIME_CANDIDATES["aio3"]]
        ),
        output=str(output),
        probe_height=128,
        probe_width=128,
    )
    payload, passed = preflight.execute_driver(args)
    assert passed is True
    assert output.is_file()
    assert all(
        "passed" in probe and "pass" not in probe
        for probe in payload["attempts"][0]["probes"]
    )
    selected, attempts = runtime._validate_worker_result(payload, "aio3")
    assert selected == runtime.STAGE_B_RUNTIME_CANDIDATES["aio3"][0]
    assert len(attempts) == 1
    capacity_calls = [
        call for call in observed_workers if call["role"] == "capacity_10_10"
    ]
    assert capacity_calls
    assert all(call["checkpoint"] is None for call in capacity_calls)
    assert all(call["allow_random"] is True for call in capacity_calls)
    assert all(
        call["coordinate_stats_override"] == _stats.resolve()
        for call in capacity_calls
    )


def test_orchestrator_cache_identity_and_command_use_frozen_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        runtime.REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG,
    )
    stage_a, stats, _split = _write_templates(tmp_path, "aio3")
    calls: list[list[str]] = []
    bundle = runtime.ensure_stage_b_runtime_bundle(
        tmp_path,
        "aio3",
        stage_a_checkpoint=stage_a,
        coordinate_stats=stats,
        runner=_runner_for(tmp_path, _worker_payload("aio3", 0), calls),
    )
    expectation = orchestrate._run_cache_expectation(
        run_name="aio3_oracle_o7_pilot_n1000_s7",
        config=bundle.main_config,
        stage="b_oracle",
        feedback="O7",
        init=stage_a,
        max_steps=1000,
        seed_override=None,
        epochs=None,
    )
    assert expectation is not None
    contract = expectation["contract"]
    assert contract["stage_b_runtime_manifest_sha256"] == bundle.manifest_sha256
    assert contract["stage_b_runtime_family"] == "aio3"
    assert contract["stage_b_runtime_role"] == "main"
    assert contract["workers_override"] is None
    assert (
        contract["cublas_workspace_config"]
        == runtime.REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG
    )

    monkeypatch.delenv("SRSC_TRAIN_WORKERS", raising=False)
    command = orchestrate.train_command(
        bundle.main_config,
        "b_oracle",
        "aio3_oracle_o7_pilot_n1000_s7",
        "O7",
        stage_a,
        max_steps=1000,
    )
    assert "--workers-override" not in command
    monkeypatch.setenv("SRSC_TRAIN_WORKERS", "3")
    with pytest.raises(RuntimeError, match="SRSC_TRAIN_WORKERS is forbidden"):
        orchestrate.train_command(
            bundle.main_config,
            "b_oracle",
            "aio3_oracle_o7_pilot_n1000_s7",
            "O7",
            stage_a,
            max_steps=1000,
        )
    monkeypatch.delenv("SRSC_TRAIN_WORKERS")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        orchestrate.train_command(
            bundle.main_config,
            "b_oracle",
            "aio3_oracle_o7_pilot_n1000_s7",
            "O7",
            stage_a,
            max_steps=1000,
        )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        runtime.assert_stage_b_cublas_environment(
            runtime.runtime_identity_for_config(bundle.main_config)
        )
