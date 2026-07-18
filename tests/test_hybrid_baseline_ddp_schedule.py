from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import train_baseline_hybrid as reference
from scripts import train_baseline_hybrid_ddp as hybrid_ddp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"


@pytest.mark.parametrize(
    ("epoch", "global_batch", "local_batch", "micro_batch", "accumulation", "steps"),
    [(0, 64, 16, 8, 2, 2_151), (54, 64, 16, 8, 2, 2_151),
     (55, 120, 30, 10, 3, 1_147), (239, 120, 30, 10, 3, 1_147)],
)
def test_rank_partition_reassembles_every_canonical_update(
    epoch: int,
    global_batch: int,
    local_batch: int,
    micro_batch: int,
    accumulation: int,
    steps: int,
):
    seed = 1415926
    canonical = hybrid_ddp.canonical_epoch_update_matrix(
        reference.EXPECTED_DATASET_SIZE, seed, epoch
    )
    ranks = [
        hybrid_ddp.rank_epoch_matrix(
            reference.EXPECTED_DATASET_SIZE, seed, epoch, rank
        )
        for rank in range(4)
    ]
    assert canonical.shape == (steps, global_batch)
    assert all(matrix.shape == (steps, local_batch) for matrix in ranks)
    rank_micros = [
        matrix.view(steps, accumulation, micro_batch) for matrix in ranks
    ]
    assert all(
        micros.shape == (steps, accumulation, micro_batch)
        for micros in rank_micros
    )
    assert all(
        torch.equal(micros.reshape(steps, local_batch), matrix)
        for micros, matrix in zip(rank_micros, ranks, strict=True)
    )
    assert torch.equal(hybrid_ddp.reassemble_rank_matrices(ranks), canonical)
    assert torch.equal(
        canonical.reshape(-1),
        reference.epoch_indices(reference.EXPECTED_DATASET_SIZE, seed, epoch),
    )
    # The equality is per update, not merely an epoch-level multiset match.
    for update in (0, steps // 2, steps - 1):
        assert torch.equal(torch.cat([matrix[update] for matrix in ranks]), canonical[update])


def test_epoch_update_digest_binds_flat_order_updates_and_rank_slices():
    first = hybrid_ddp.epoch_digest_record(
        reference.EXPECTED_DATASET_SIZE, 1415926, 55
    )
    repeat = hybrid_ddp.epoch_digest_record(
        reference.EXPECTED_DATASET_SIZE, 1415926, 55
    )
    adjacent = hybrid_ddp.epoch_digest_record(
        reference.EXPECTED_DATASET_SIZE, 1415926, 56
    )
    assert first == repeat
    assert first["global_batch"] == 120
    assert first["local_effective_batch"] == 30
    assert first["micro_batch"] == 10
    assert first["accumulation"] == 3
    canonical_flat = reference.epoch_indices(
        reference.EXPECTED_DATASET_SIZE, 1415926, 55
    )
    assert first["flat_indices_sha256"] == hashlib.sha256(
        hybrid_ddp.int64_tensor_bytes(canonical_flat)
    ).hexdigest()
    assert first["per_update_digest_root"] == hybrid_ddp.update_digest_root(
        canonical_flat.view(1_147, 120)
    )
    assert len(first["rank_indices_sha256"]) == 4
    assert first["per_update_digest_root"] != adjacent["per_update_digest_root"]
    assert first["record_sha256"] != adjacent["record_sha256"]


def test_exact_ddp_budget_and_phase_boundary_counters():
    assert hybrid_ddp.expected_progress(0, 0) == (0, 0)
    assert hybrid_ddp.expected_progress(0, 1) == (1, 64)
    assert hybrid_ddp.expected_progress(54, 2_151) == (118_305, 7_571_520)
    assert hybrid_ddp.expected_progress(55, 0) == (118_305, 7_571_520)
    assert hybrid_ddp.expected_progress(55, 1) == (118_306, 7_571_640)
    assert hybrid_ddp.expected_progress(240, 0) == (
        reference.EXPECTED_TOTAL_STEPS,
        reference.EXPECTED_TOTAL_SAMPLES,
    )
    with pytest.raises(ValueError, match="outside the active epoch"):
        hybrid_ddp.expected_progress(55, 1_148)

    assert hybrid_ddp.expected_microbatch_cursor(0, 1) == 2
    assert hybrid_ddp.expected_microbatch_cursor(54, 2_151) == 4_302
    assert hybrid_ddp.expected_microbatch_cursor(55, 1) == 3
    assert hybrid_ddp.expected_microbatch_cursor(239, 1_147) == 3_441
    assert hybrid_ddp.expected_microbatch_cursor(240, 0) == 0


def _resume_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    split = tmp_path / "split.json"
    split.write_bytes(
        (ROOT / "artifacts/manifests/locked_split_aio3.json").read_bytes()
    )
    cfg["split_manifest"] = str(split)
    config = tmp_path / "hybrid.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = SimpleNamespace(
        config=str(config),
        stage="baseline",
        run_name="ddp-hybrid-test",
        fresh=False,
        resume=str(tmp_path / "last.pt"),
        workers_per_rank=8,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_sha = hybrid_ddp.ensure_run_contract_rank0(run_dir, cfg, args, 8)
    record = hybrid_ddp.epoch_digest_record(
        reference.EXPECTED_DATASET_SIZE, cfg["seed"], 0
    )
    records = {"0": record}
    model_state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    optimizer_state = {"state": {}, "param_groups": [{"lr": 2e-4, "params": [0]}]}
    scheduler_state = {"last_epoch": 1, "_last_lr": [2e-4]}
    integrity = {
        "level": "probe_sha256",
        "model": hybrid_ddp.probe_state_sha256(model_state),
        "optimizer": hybrid_ddp.probe_state_sha256(optimizer_state),
        "scheduler": hybrid_ddp.full_state_sha256(scheduler_state),
        "world_size": 4,
        "all_ranks_identical": True,
    }
    payload = {
        "training_origin": hybrid_ddp.TRAINING_ORIGIN,
        "config_sha256": hybrid_ddp.sha256_file(config),
        "config": cfg,
        "split_manifest_sha256": hybrid_ddp.sha256_file(split),
        "args": hybrid_ddp.stable_args(args, 8),
        "reference_schedule": reference.expected_schedule_payload(),
        "reference_schedule_sha256": reference.schedule_sha256(cfg["seed"]),
        "partition_algorithm": hybrid_ddp.PARTITION_ALGORITHM,
        "distributed_runtime": {
            "world_size": 4, "workers_per_rank": 8, "backend": "nccl",
        },
        "run_contract_sha256": contract_sha,
        "epoch": 1,
        "update_in_epoch": 0,
        "microbatch_in_epoch": 0,
        "batch_in_epoch": 0,
        "cursor_authority": "optimizer_update",
        "checkpoint_boundary": "after_complete_optimizer_update",
        "step": 2_151,
        "samples_seen": 137_664,
        "validation_pending": None,
        "validation_transaction_schema": 1,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "rng_by_rank": [{"rank": rank} for rank in range(4)],
        "ddp_integrity": integrity,
        "active_phase": hybrid_ddp.micro_profile_payload(
            hybrid_ddp.micro_profile_for_epoch(1)
        ),
        "epoch_update_digests": records,
        "schedule_digest": hybrid_ddp.schedule_digest(records),
    }
    return cfg, args, contract_sha, payload


def test_resume_contract_accepts_exact_transaction_and_rejects_digest_tampering(
    tmp_path: Path,
):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    hybrid_ddp.validate_resume_payload(payload, cfg, args, 8, contract_sha)

    broken = copy.deepcopy(payload)
    broken["epoch_update_digests"]["0"]["per_update_digest_root"] = "f" * 64
    broken["schedule_digest"] = hybrid_ddp.schedule_digest(
        broken["epoch_update_digests"]
    )
    with pytest.raises(RuntimeError, match="epoch/update digest mismatch"):
        hybrid_ddp.validate_resume_payload(broken, cfg, args, 8, contract_sha)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("training_origin", "legacy", "not an exact-update"),
        ("step", 2_150, "optimizer-step budget"),
        ("samples_seen", 137_600, "raw-sample budget"),
        ("batch_in_epoch", 1, "compatibility microbatch cursor"),
        ("microbatch_in_epoch", 1, "derived microbatch cursor"),
        ("cursor_authority", "microbatch", "cursor authority"),
    ],
)
def test_resume_rejects_origin_budget_and_cursor_drift(
    tmp_path: Path, field: str, replacement, message: str
):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    payload[field] = replacement
    with pytest.raises(RuntimeError, match=message):
        hybrid_ddp.validate_resume_payload(
            payload, cfg, args, 8, contract_sha, verify_digests=False
        )


def test_resume_rejects_world_worker_rng_and_integrity_drift(tmp_path: Path):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    payload["distributed_runtime"]["world_size"] = 2
    with pytest.raises(RuntimeError, match="distributed runtime"):
        hybrid_ddp.validate_resume_payload(
            payload, cfg, args, 8, contract_sha, verify_digests=False
        )

    cfg2, args2, contract2, payload2 = _resume_fixture(tmp_path / "second")
    payload2["rng_by_rank"] = payload2["rng_by_rank"][:3]
    with pytest.raises(RuntimeError, match="RNG width"):
        hybrid_ddp.validate_resume_payload(
            payload2, cfg2, args2, 8, contract2, verify_digests=False
        )

    cfg3, args3, contract3, payload3 = _resume_fixture(tmp_path / "third")
    payload3["model"]["weight"][0, 0] += 1
    with pytest.raises(RuntimeError, match="model no longer matches"):
        hybrid_ddp.validate_resume_payload(
            payload3, cfg3, args3, 8, contract3, verify_digests=False
        )


def test_contract_freezes_safe_micro_profiles_and_claim_boundary(tmp_path: Path):
    cfg, args, _, _ = _resume_fixture(tmp_path)
    contract = hybrid_ddp.run_contract_payload(cfg, args, 8)
    assert contract["world_size"] == 4
    assert contract["checkpoint_cursor_authority"] == "optimizer_update"
    assert [
        phase["local_effective_batch"] for phase in contract["runtime_phases"]
    ] == [16, 30]
    assert [phase["micro_batch"] for phase in contract["runtime_phases"]] == [8, 10]
    assert [phase["accumulation"] for phase in contract["runtime_phases"]] == [2, 3]
    assert all(
        phase["sync_policy"] == "DDP.no_sync on all non-final microbatches"
        for phase in contract["runtime_phases"]
    )
    assert "exact canonical raw-index set" in contract["equivalence_claim"]
    assert "not claimed bitwise-identical" in contract["stochastic_claim_boundary"]
    assert contract["environment"]["cublas_workspace_config"] == ":4096:8"
    assert contract["environment"]["process_group_timeout_seconds"] == 7_200
    assert contract["data_contract"]["missing_entries"] == 0
    assert set(contract["code_sha256"]) >= {
        "scripts/runtime_accounting.py",
        "src/net/clean_restormer_aio.py",
        "src/net/srsc_lite.py",
        "src/net/restormer_blocks.py",
        "src/data/aio_dataset.py",
    }


def test_execution_environment_fails_before_cuda_when_cublas_is_unlocked(
    monkeypatch,
):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        hybrid_ddp.validate_execution_environment()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="actual=':16:8'"):
        hybrid_ddp.validate_execution_environment()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    hybrid_ddp.validate_execution_environment()


def test_data_contract_detects_source_list_drift(tmp_path: Path):
    cfg = yaml.safe_load(CONFIG.read_text())
    split = json.loads(Path(cfg["split_manifest"]).read_text())
    list_root = tmp_path / "lists"
    for relative in split["list_sha256"]:
        source = Path(cfg["list_root"]) / relative
        target = list_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    cfg["list_root"] = str(list_root)
    exact = hybrid_ddp.data_contract_payload(cfg)
    assert exact["list_sha256"] == split["list_sha256"]

    first = list_root / sorted(split["list_sha256"])[0]
    first.write_bytes(first.read_bytes() + b"\nDRIFT\n")
    with pytest.raises(RuntimeError, match="source-list hash drift"):
        hybrid_ddp.data_contract_payload(cfg)


def test_phase_two_resume_cursor_is_update_authoritative(tmp_path: Path):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    payload["epoch"] = 55
    payload["update_in_epoch"] = 1
    payload["microbatch_in_epoch"] = 3
    payload["batch_in_epoch"] = 3
    payload["step"] = 118_306
    payload["samples_seen"] = 7_571_640
    payload["scheduler"]["last_epoch"] = 55
    payload["ddp_integrity"]["scheduler"] = hybrid_ddp.full_state_sha256(
        payload["scheduler"]
    )
    payload["active_phase"] = hybrid_ddp.micro_profile_payload(
        hybrid_ddp.micro_profile_for_epoch(55)
    )
    hybrid_ddp.validate_resume_payload(
        payload, cfg, args, 8, contract_sha, verify_digests=False
    )

    payload["microbatch_in_epoch"] = 1
    with pytest.raises(RuntimeError, match="derived microbatch cursor"):
        hybrid_ddp.validate_resume_payload(
            payload, cfg, args, 8, contract_sha, verify_digests=False
        )


def test_training_source_uses_no_sync_and_scaled_micro_loss():
    source = Path(hybrid_ddp.__file__).read_text()
    assert "ddp.no_sync()" in source
    assert "loss = rest / profile.accumulation" in source
    assert "update_index = micro_index // profile.accumulation" in source


@pytest.mark.parametrize(
    ("micro_batch", "accumulation", "global_batch"),
    [(8, 2, 64), (10, 3, 120)],
)
def test_four_rank_scaled_micro_gradients_equal_canonical_global_mean(
    micro_batch: int, accumulation: int, global_batch: int
):
    torch.manual_seed(20260718)
    template = torch.nn.Linear(5, 3)
    features = torch.randn(global_batch, 5)
    targets = torch.randn(global_batch, 3)

    canonical = copy.deepcopy(template)
    canonical_loss = torch.nn.functional.l1_loss(canonical(features), targets)
    canonical_loss.backward()

    local_effective = micro_batch * accumulation
    rank_gradients = []
    for rank in range(hybrid_ddp.WORLD_SIZE):
        local = copy.deepcopy(template)
        begin = rank * local_effective
        for offset in range(0, local_effective, micro_batch):
            micro_slice = slice(begin + offset, begin + offset + micro_batch)
            micro_loss = torch.nn.functional.l1_loss(
                local(features[micro_slice]), targets[micro_slice]
            ) / accumulation
            micro_loss.backward()
        rank_gradients.append([parameter.grad for parameter in local.parameters()])

    for parameter_index, canonical_parameter in enumerate(canonical.parameters()):
        ddp_averaged = torch.stack([
            gradients[parameter_index] for gradients in rank_gradients
        ]).mean(dim=0)
        assert torch.allclose(
            ddp_averaged, canonical_parameter.grad, atol=2e-7, rtol=2e-6
        )


def test_state_hashes_are_deterministic_and_detect_tensor_or_adam_drift():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    full = hybrid_ddp.full_state_sha256(state)
    probe = hybrid_ddp.probe_state_sha256(state)
    assert full == hybrid_ddp.full_state_sha256(state)
    assert probe == hybrid_ddp.probe_state_sha256(state)
    with torch.no_grad():
        model.weight.view(-1)[0] += 1
    changed = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    assert full != hybrid_ddp.full_state_sha256(changed)
    assert probe != hybrid_ddp.probe_state_sha256(changed)
    changed_metadata = copy.deepcopy(changed)
    changed_metadata["optimizer"]["param_groups"][0]["lr"] = 9e-3
    assert hybrid_ddp.probe_state_sha256(changed) != hybrid_ddp.probe_state_sha256(
        changed_metadata
    )


def test_collect_ddp_integrity_requires_all_four_rank_records_to_match(monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    monkeypatch.setattr(hybrid_ddp.dist, "get_world_size", lambda: 4)

    def matching(output, local):
        output[:] = [copy.deepcopy(local) for _ in range(4)]

    monkeypatch.setattr(hybrid_ddp.dist, "all_gather_object", matching)
    result = hybrid_ddp.collect_ddp_integrity(
        model, optimizer, scheduler, full=True
    )
    assert result["world_size"] == 4
    assert result["all_ranks_identical"] is True
    assert result["level"] == "full_sha256"

    def diverged(output, local):
        output[:] = [copy.deepcopy(local) for _ in range(4)]
        output[3]["model"] = "0" * 64

    monkeypatch.setattr(hybrid_ddp.dist, "all_gather_object", diverged)
    with pytest.raises(RuntimeError, match="DDP model/optimizer/scheduler divergence"):
        hybrid_ddp.collect_ddp_integrity(model, optimizer, scheduler, full=False)


def test_ddp_entry_rejects_nonbaseline_stage_before_cuda():
    with pytest.raises(SystemExit):
        hybrid_ddp.parse_args([
            "--config", str(CONFIG), "--stage", "a",
            "--run-name", "bad", "--fresh",
        ])


def test_schedule_digest_is_canonical_in_numeric_epoch_order():
    records = {
        "1": {"record_sha256": "b"},
        "0": {"record_sha256": "a"},
    }
    first = hybrid_ddp.schedule_digest(records)
    second = hybrid_ddp.schedule_digest(json.loads(json.dumps(records)))
    assert first == second
