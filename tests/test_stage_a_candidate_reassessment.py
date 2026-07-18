from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from scripts import reassess_stage_a_candidates as reassess


RUN_NAME = "aio3_stage_a_coarse_seed1415926"
TASKS = reassess.EXPECTED_VALIDATION_TASKS["aio3"]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class TinyLockedDataset:
    def __init__(self):
        self.items = [
            {"task": task, "name": f"{task}:locked"} for task in TASKS
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _fixture_tree(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(reassess, "ROOT", tmp_path)
    monkeypatch.setattr(reassess, "assert_no_active_trainer", lambda _name: None)
    monkeypatch.setattr(
        reassess,
        "evaluator_closure",
        lambda: {
            "files": {"scripts/train.py": "e" * 64},
            "environment": {"torch": "test"},
            "sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(reassess, "build_model", lambda _cfg, _stage: TinyModel())
    monkeypatch.setattr(reassess, "move_model_to_cuda", lambda model: model.eval())
    monkeypatch.setattr(reassess, "clear_cuda_cache", lambda: None)
    monkeypatch.setattr(
        reassess, "build_locked_val", lambda *_args, **_kwargs: TinyLockedDataset()
    )

    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    list_root = tmp_path / "upstream/PromptIR/data_dir"
    list_root.mkdir(parents=True)
    for relative, content in (
        ("noisy/denoise.txt", "clean.png\n"),
        ("rainy/rainTrain.txt", "rain-1.png\n"),
        ("hazy/hazy_outside.txt", "1_1.png\n"),
    ):
        path = list_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    list_sha256 = {
        str(path.relative_to(list_root)): reassess.sha256_file(path)
        for path in sorted(list_root.rglob("*.txt"))
    }
    manifests = tmp_path / "artifacts/manifests"
    manifests.mkdir(parents=True)
    split = manifests / "locked_split_aio3.json"
    split.write_text(json.dumps({
        "protocol": "aio3",
        "locked_groups": [],
        "list_sha256": list_sha256,
    }, sort_keys=True) + "\n")
    materialization = manifests / "aio3.json"
    materialization.write_text(json.dumps({
        "protocol": "aio3",
        "missing_entries": 0,
        "expected_entries": 3,
        "list_sha256": list_sha256,
    }, sort_keys=True) + "\n")
    config = tmp_path / "configs/protocol_aio3.yaml"
    cfg = {
        "protocol": "aio3",
        "seed": 1415926,
        "data_root": str(tmp_path / "data"),
        "list_root": str(list_root),
        "split_manifest": str(split),
        "epochs": 240,
        "official_test_locked": True,
        "model": {"dim": 1},
    }
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    config_sha = reassess.sha256_file(config)
    split_sha = reassess.sha256_file(split)

    runtime = manifests / "aio3_live_runtime_attestation.json"
    owner_uuids = [f"GPU-{index:032x}" for index in range(4)]
    owner_cmdline = [
        "python",
        "scripts/train_stage_a_ddp.py",
        "--config",
        "configs/protocol_aio3.yaml",
        "--resume",
        f"artifacts/checkpoints/{RUN_NAME}/last.pt",
        "--run-name",
        RUN_NAME,
        "--per-gpu-batch",
        "30",
        "--accumulation",
        "1",
        "--workers-per-rank",
        "8",
    ]
    observed_runtime = {
        "world_size": 4,
        "per_gpu_batch": 30,
        "accumulation": 1,
        "global_effective_batch": 120,
        "backend": "nccl",
        "workers_per_rank": 8,
        "workers_evidence": "live_process_cmdline",
    }
    runtime.write_text(json.dumps({
        "schema": reassess.LEGACY_RUNTIME_ATTESTATION_SCHEMA,
        "status": "PASS",
        "run_name": RUN_NAME,
        "scientific_limit": reassess.LEGACY_SCIENTIFIC_LIMIT,
        "claim": (
            "Externally observed runtime evidence only; this does not backfill "
            "or attribute missing launch-time code/data fields to the checkpoint."
        ),
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "observed_runtime": observed_runtime,
        "cuda_owners": [
            {
                "pid": 100 + rank,
                "gpu_uuid": owner_uuids[rank],
                "process_name": "python",
                "used_memory_mib": 22000,
                "start_ticks_since_boot": 1000,
                "cmdline": owner_cmdline,
                "environment": {
                    "WORLD_SIZE": "4",
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                },
            }
            for rank in range(4)
        ],
        "gpu_inventory": [
            {
                "index": rank,
                "uuid": owner_uuids[rank],
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total_mib": 24564,
            }
            for rank in range(4)
        ],
        "checkpoint_observation": {
            "path": str(
                tmp_path / "artifacts/checkpoints" / RUN_NAME / "last.pt"
            ),
            "bytes": 100,
            "sha256": "a" * 64,
            "epoch": 158,
            "step": 237000,
            "rank_rng_width": 4,
            "config_sha256": config_sha,
            "split_manifest_sha256": split_sha,
            "embedded_runtime": {
                key: observed_runtime[key]
                for key in (
                    "world_size",
                    "per_gpu_batch",
                    "accumulation",
                    "global_effective_batch",
                    "backend",
                )
            },
            "embedded_workers_per_rank": None,
            "embedded_code_contract": None,
            "embedded_data_contract": None,
        },
        "launcher": {
            "pid": 90,
            "start_ticks_since_boot": 900,
            "cmdline": ["bash", "scripts/launch_aio3_stage_a_4x4090.sh"],
        },
        "evidence_files": {
            "artifacts/logs/pipeline_ddp.log": {
                "bytes_at_capture": 10,
                "sha256_at_capture": "b" * 64,
            },
        },
    }, sort_keys=True) + "\n")

    run_dir = tmp_path / "artifacts/checkpoints" / RUN_NAME
    run_dir.mkdir(parents=True)
    metrics_dir = tmp_path / "artifacts/metrics"
    metrics_dir.mkdir(parents=True)
    candidates = [
        (230, 2300, 34.0),
        (220, 2200, 33.0),
        (240, 2400, 32.0),
    ]
    top3 = []
    ledger_rows = []
    for epoch, step, score in candidates:
        state_model = TinyModel()
        state_model.weight.data.fill_(score)
        checkpoint_name = f"val_epoch{epoch:03d}_step{step:07d}.pt"
        payload = {
            "model": state_model.state_dict(),
            "epoch": epoch,
            "step": step,
            "batch_in_epoch": 0,
            "validation_pending": None,
            "config": cfg,
            "config_sha256": config_sha,
            "split_manifest_sha256": split_sha,
            "args": {"stage": "a", "run_name": RUN_NAME},
        }
        _atomic_checkpoint(run_dir / checkpoint_name, payload)
        top3.append({
            "score": score,
            "epoch": epoch,
            "step": step,
            "checkpoint": checkpoint_name,
        })
        row = {task: score for task in TASKS}
        row.update({"macro_psnr": score, "epoch": epoch, "step": step})
        ledger_rows.append(row)
        if epoch == 240:
            _atomic_checkpoint(run_dir / "last.pt", copy.deepcopy(payload))
    top3_path = run_dir / "top3.json"
    top3_path.write_text(json.dumps(top3, indent=2) + "\n")
    ledger = metrics_dir / f"{RUN_NAME}_locked_val.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in ledger_rows))
    output = metrics_dir / "stage_a_reassessment" / RUN_NAME
    return {
        "cfg": cfg,
        "config": config,
        "runtime": runtime,
        "split": split,
        "materialization": materialization,
        "list_root": list_root,
        "run_dir": run_dir,
        "top3": top3_path,
        "ledger": ledger,
        "output": output,
    }


def _mock_evaluator(call_scores: list[float] | None = None, fail_call: int | None = None):
    calls = []

    def evaluate(**kwargs):
        assert kwargs["stage"] == "a"
        assert kwargs["return_rows"] is True
        assert kwargs["protocol"] == "aio3"
        score = float(kwargs["model"].weight.item())
        calls.append(score)
        if fail_call is not None and len(calls) == fail_call:
            raise RuntimeError("injected reassessment interruption")
        if call_scores is not None:
            score = call_scores[len(calls) - 1]
        rows = [
            {"task": task, "name": f"{task}:locked", "psnr": score, "ssim": 0.9}
            for task in TASKS
        ]
        summary = {task: score for task in TASKS}
        summary.update({
            "macro_psnr": score,
            "setting_ssim": {task: 0.9 for task in TASKS},
            "five_setting_mean_ssim": 0.9,
        })
        return summary, rows

    return calls, evaluate


def _run(tree: dict):
    return reassess.run_reassessment(
        config_path=tree["config"],
        run_name=RUN_NAME,
        runtime_attestation=tree["runtime"],
        output_dir=tree["output"],
    )


def test_reassessment_is_real_deduplicated_read_only_and_idempotent(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    ledger_before = tree["ledger"].read_bytes()
    top3_before = tree["top3"].read_bytes()
    calls, evaluator = _mock_evaluator()
    monkeypatch.setattr(reassess, "validate_locked", evaluator)

    manifest = _run(tree)

    assert manifest["status"] == "PASS"
    assert manifest["candidate_count_after_dedup"] == 3
    assert manifest["selected_candidate_id"] == "epoch230_step0002300"
    assert calls == [34.0, 33.0, 32.0]
    assert tree["ledger"].read_bytes() == ledger_before
    assert tree["top3"].read_bytes() == top3_before
    assert not list(tree["output"].rglob("*.tmp.*"))

    attestation = json.loads(
        (tree["output"] / "selection_attestation.json").read_text()
    )
    assert attestation["runtime_attestation"]["sha256"] == reassess.sha256_file(
        tree["runtime"]
    )
    assert (
        attestation["runtime_attestation"]["scientific_limit"]
        == reassess.LEGACY_SCIENTIFIC_LIMIT
    )
    terminal = json.loads(
        (tree["output"] / "summaries/epoch240_step0002400.json").read_text()
    )
    assert {source["role"] for source in terminal["source_aliases"]} == {
        "top3_rank_3",
        "terminal_last",
    }
    assert "checkpoint" not in terminal["historical_psnr_evidence"]["values"]
    assert set(terminal["evidence_bindings"]) == {
        "checkpoint_sha256",
        "checkpoint_model_state_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "top3_sha256",
        "historical_ledger_sha256",
        "evaluator_closure_sha256",
        "runtime_attestation_sha256",
        "materialization_manifest_sha256",
        "dataset_list_set_canonical_sha256",
        "dataset_list_count",
        "runtime_scientific_limit",
    }

    monkeypatch.setattr(
        reassess,
        "validate_locked",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("idempotent replay evaluated a model")
        ),
    )
    reused = _run(tree)
    assert reused["idempotent_reuse"] is True
    assert tree["ledger"].read_bytes() == ledger_before


def test_interrupted_transaction_resumes_only_unfinished_candidates(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    first_calls, first = _mock_evaluator(fail_call=2)
    monkeypatch.setattr(reassess, "validate_locked", first)
    with pytest.raises(RuntimeError, match="injected reassessment interruption"):
        _run(tree)
    staging = tree["output"].with_name(f".{tree['output'].name}.staging")
    assert first_calls == [34.0, 33.0]
    assert (staging / "summaries/epoch230_step0002300.json").is_file()
    assert not tree["output"].exists()

    resumed_calls, resumed = _mock_evaluator()
    monkeypatch.setattr(reassess, "validate_locked", resumed)
    manifest = _run(tree)
    assert manifest["status"] == "PASS"
    assert resumed_calls == [33.0, 32.0]
    assert not staging.exists()


def test_psnr_history_drift_fails_without_rewriting_legacy_ledger(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    legacy = tree["ledger"].read_bytes()
    _, evaluator = _mock_evaluator(call_scores=[34.001])
    monkeypatch.setattr(reassess, "validate_locked", evaluator)

    with pytest.raises(RuntimeError, match="immutable historical ledger"):
        _run(tree)

    assert tree["ledger"].read_bytes() == legacy
    assert not tree["output"].exists()


def test_strict_model_load_rejects_checkpoint_state_mismatch(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    path = tree["run_dir"] / "val_epoch230_step0002300.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"] = {"unexpected": torch.ones(1)}
    torch.save(payload, path)
    _, evaluator = _mock_evaluator()
    monkeypatch.setattr(reassess, "validate_locked", evaluator)

    with pytest.raises(RuntimeError):
        _run(tree)


def test_runtime_attestation_is_required_and_cannot_overclaim(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    payload = json.loads(tree["runtime"].read_text())
    payload["scientific_limit"] = "FULL_CODE_DATA_PROVEN"
    tree["runtime"].write_text(json.dumps(payload) + "\n")
    _, evaluator = _mock_evaluator()
    monkeypatch.setattr(reassess, "validate_locked", evaluator)

    with pytest.raises(RuntimeError, match="scientific limit"):
        _run(tree)


def test_runtime_attestation_rejects_structural_forgery(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    original = json.loads(tree["runtime"].read_text())
    forged_payloads = []

    missing_owner = copy.deepcopy(original)
    missing_owner["cuda_owners"].pop()
    forged_payloads.append(missing_owner)

    duplicate_gpu = copy.deepcopy(original)
    duplicate_gpu["cuda_owners"][1]["gpu_uuid"] = duplicate_gpu["cuda_owners"][0][
        "gpu_uuid"
    ]
    forged_payloads.append(duplicate_gpu)

    wrong_workers = copy.deepcopy(original)
    tokens = wrong_workers["cuda_owners"][0]["cmdline"]
    tokens[tokens.index("--workers-per-rank") + 1] = "4"
    forged_payloads.append(wrong_workers)

    fake_checkpoint = copy.deepcopy(original)
    fake_checkpoint["checkpoint_observation"]["rank_rng_width"] = 1
    forged_payloads.append(fake_checkpoint)

    official_evidence = copy.deepcopy(original)
    official_evidence["evidence_files"] = {
        "artifacts/official_test/leak.log": {
            "bytes_at_capture": 1,
            "sha256_at_capture": "c" * 64,
        }
    }
    forged_payloads.append(official_evidence)

    for forged in forged_payloads:
        tree["runtime"].write_text(json.dumps(forged) + "\n")
        with pytest.raises((RuntimeError, PermissionError)):
            reassess.validate_runtime_attestation(
                tree["runtime"], RUN_NAME, "aio3"
            )


def test_dataset_list_and_materialization_drift_fail_without_image_scan(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    list_file = tree["list_root"] / "noisy/denoise.txt"
    original_list = list_file.read_bytes()
    list_file.write_bytes(original_list + b"tampered.png\n")
    with pytest.raises(RuntimeError, match="dataset-list files differ"):
        _run(tree)

    list_file.write_bytes(original_list)
    materialization = json.loads(tree["materialization"].read_text())
    materialization["missing_entries"] = 1
    tree["materialization"].write_text(json.dumps(materialization) + "\n")
    with pytest.raises(RuntimeError, match="materialization manifest"):
        _run(tree)


def test_shared_gpu_pipeline_lock_is_required_for_entire_reassessment(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    lock_path = tmp_path / ".srsc_gpu_pipeline.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="GPU pipeline is already active"):
            _run(tree)
    finally:
        os.close(descriptor)


def test_incomplete_terminal_checkpoint_and_official_paths_fail_closed(
    tmp_path: Path, monkeypatch
):
    tree = _fixture_tree(tmp_path, monkeypatch)
    last = torch.load(tree["run_dir"] / "last.pt", map_location="cpu", weights_only=False)
    last["epoch"] = 239
    torch.save(last, tree["run_dir"] / "last.pt")
    _, evaluator = _mock_evaluator()
    monkeypatch.setattr(reassess, "validate_locked", evaluator)
    with pytest.raises(RuntimeError, match="not terminal"):
        _run(tree)

    with pytest.raises(PermissionError):
        reassess.validate_run_name("aio3_official_test")
    with pytest.raises(PermissionError, match="output"):
        reassess.validate_locked_only_paths(
            config_path=tree["config"],
            output_dir=tmp_path / "outside",
            cfg=tree["cfg"],
            protocol="aio3",
        )


def test_csv_recomputation_rejects_identity_and_summary_drift():
    dataset = TinyLockedDataset()
    identities, _ = reassess.expected_locked_identities(dataset, TASKS)
    rows = [
        {"task": task, "name": f"{task}:locked", "psnr": 30.0, "ssim": 0.8}
        for task in TASKS
    ]
    _, summary = reassess.validate_rows_and_recompute(rows, identities, TASKS)
    drifted = copy.deepcopy(summary)
    drifted["macro_psnr"] += 1e-6
    with pytest.raises(RuntimeError, match="recomputation mismatch"):
        reassess.compare_summaries(drifted, summary, TASKS)
    rows[0]["name"] = "wrong"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        reassess.validate_rows_and_recompute(rows, identities, TASKS)


def test_active_trainer_detection_accepts_absolute_script_and_equals_argument(
    tmp_path: Path,
):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0/root/project/scripts/train_stage_a_ddp.py\0"
        + f"--run-name={RUN_NAME}\0".encode()
    )
    with pytest.raises(RuntimeError, match="trainer is still active"):
        reassess.assert_no_active_trainer(RUN_NAME, proc_root=tmp_path)
