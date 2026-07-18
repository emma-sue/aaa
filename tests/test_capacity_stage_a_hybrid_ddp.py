from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import orchestrate
from scripts import train_baseline_hybrid as reference
from scripts import train_baseline_hybrid_ddp as engine
from scripts import train_stage_a_capacity_hybrid_ddp as capacity
from scripts.train import build_model, configure_trainable, load_formal_init


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/protocol_aio3_10_10_hybrid.yaml"
BASELINE_CONFIG = ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"


def test_capacity_config_replays_primary_hybrid_budget_with_only_ownership_change():
    cfg = capacity.load_and_validate_config(CONFIG)
    primary = reference.load_and_validate_config(BASELINE_CONFIG)
    assert capacity._training_protocol(cfg) == capacity._training_protocol(primary)
    assert cfg["hybrid_schedule"]["expected_total_steps"] == 330_500
    assert cfg["hybrid_schedule"]["expected_total_samples"] == 33_034_920
    assert cfg["epochs"] == 240
    assert cfg["validate_every_epochs"] == 5
    assert list(range(5, cfg["epochs"] + 1, cfg["validate_every_epochs"]))[-1] == 240
    assert len(range(5, cfg["epochs"] + 1, cfg["validate_every_epochs"])) == 48
    assert sum(cfg["model"]["d1_blocks"]) == 10
    assert sum(cfg["model"]["d2_blocks"]) + cfg["model"]["d2_refinement"] == 10
    ownership = {"d1_blocks", "d2_blocks", "d2_refinement"}
    assert {
        key: value for key, value in cfg["model"].items() if key not in ownership
    } == {
        key: value
        for key, value in primary["model"].items()
        if key not in ownership
    }
    assert capacity.per_scale_total_blocks(cfg["model"]) == (6, 6, 8)
    assert capacity.per_scale_total_blocks(primary["model"]) == (6, 6, 8)
    assert cfg["coordinate_stats"].endswith("coordinate_stats_aio3_10_10.json")


def test_capacity_and_primary_have_exactly_equal_total_parameter_count():
    capacity_cfg = capacity.load_and_validate_config(CONFIG)
    primary_cfg = reference.load_and_validate_config(BASELINE_CONFIG)
    capacity_model = build_model(capacity_cfg, "a")
    primary_model = build_model(primary_cfg, "a")
    assert sum(parameter.numel() for parameter in capacity_model.parameters()) == (
        sum(parameter.numel() for parameter in primary_model.parameters())
    )


@pytest.mark.parametrize("epoch", [0, 54, 55, 239])
def test_capacity_uses_identical_canonical_update_membership(epoch: int):
    cfg = capacity.load_and_validate_config(CONFIG)
    canonical = engine.canonical_epoch_update_matrix(
        reference.EXPECTED_DATASET_SIZE, cfg["seed"], epoch
    )
    ranks = [
        engine.rank_epoch_matrix(
            reference.EXPECTED_DATASET_SIZE, cfg["seed"], epoch, rank
        )
        for rank in range(engine.WORLD_SIZE)
    ]
    assert torch.equal(engine.reassemble_rank_matrices(ranks), canonical)
    assert torch.equal(
        canonical.reshape(-1),
        reference.epoch_indices(reference.EXPECTED_DATASET_SIZE, cfg["seed"], epoch),
    )


def _tiny_srsc_config() -> dict:
    return {
        "model": {
            "dim": 8,
            "matched_dim": 8,
            "encoder_blocks": [1, 1, 1, 1],
            "d1_blocks": [1, 1, 1],
            "d2_blocks": [1, 1, 1],
            "d2_refinement": 1,
            "heads": [1, 2, 4, 8],
            "expansion": 2.0,
        }
    }


def test_capacity_adapter_trains_coarse_graph_but_checkpoint_state_is_prefix_free():
    cfg = _tiny_srsc_config()
    raw = build_model(cfg, "a")
    configure_trainable(raw, "a")
    adapter = capacity.CapacityStageACoarseForward(raw)
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        _, expected = raw._encode_coarse(image)
        actual = adapter(image)
    assert torch.equal(actual, expected)

    checkpoint_state = raw.state_dict()
    assert checkpoint_state
    assert not any(key.startswith("model.") for key in checkpoint_state)
    assert all(key.startswith("model.") for key in adapter.state_dict())

    strict_clone = build_model(cfg, "a")
    strict_clone.load_state_dict(checkpoint_state, strict=True)
    stage_b = build_model(cfg, "b_predicted")
    assert load_formal_init(
        stage_b, checkpoint_state, "b_predicted"
    ) == "coarse_only_fresh_seeded_feedback_path"
    for key, value in raw.encoder.state_dict().items():
        assert torch.equal(stage_b.encoder.state_dict()[key], value)
    for key, value in raw.d1.state_dict().items():
        assert torch.equal(stage_b.d1.state_dict()[key], value)


def test_capacity_contract_is_fresh_isolated_and_does_not_mutate_baseline_contract():
    cfg = capacity.load_and_validate_config(CONFIG)
    args = capacity.stable_engine_args(CONFIG, "capacity-contract", 8)
    before_origin = engine.TRAINING_ORIGIN
    payload = capacity.run_contract_payload(cfg, args, 8)
    assert engine.TRAINING_ORIGIN == before_origin
    assert payload["purpose"] == capacity.CAPACITY_PURPOSE
    assert payload["training_origin"] == capacity.CAPACITY_TRAINING_ORIGIN
    assert payload["reference_schedule"]["expected_total_steps"] == 330_500
    assert payload["reference_schedule"]["expected_total_samples"] == 33_034_920
    assert payload["world_size"] == 4
    assert payload["capacity_definition"]["only_registered_change"].endswith(
        "6/14 -> 10/10"
    )
    assert "scripts/train_stage_a_capacity_hybrid_ddp.py" in payload["code_sha256"]

    baseline_cfg = reference.load_and_validate_config(BASELINE_CONFIG)
    baseline_args = SimpleNamespace(
        config=str(BASELINE_CONFIG.resolve()),
        stage="baseline",
        run_name="baseline-contract",
    )
    baseline = engine.run_contract_payload(baseline_cfg, baseline_args, 8)
    assert baseline["purpose"] == "aio3_four_gpu_exact_raw_update_hybrid_baseline"
    assert baseline["training_origin"] == engine.TRAINING_ORIGIN


def test_capacity_entry_installs_process_local_hooks_and_restores_baseline(monkeypatch):
    original_allowed = reference.ALLOWED_STAGES
    original_loader = reference.load_and_validate_config
    original_contract = engine.run_contract_payload
    original_ddp = engine.DDP
    original_origin = engine.TRAINING_ORIGIN

    def fake_engine_main(argv):
        assert "--stage" in argv
        assert argv[argv.index("--stage") + 1] == "a"
        assert "a" in reference.ALLOWED_STAGES
        assert reference.load_and_validate_config is capacity.load_and_validate_config
        assert engine.run_contract_payload is capacity.run_contract_payload
        assert engine.DDP is capacity._capacity_ddp
        assert engine.TRAINING_ORIGIN == capacity.CAPACITY_TRAINING_ORIGIN
        return 17

    monkeypatch.setattr(engine, "main", fake_engine_main)
    result = capacity.main([
        "--config", str(CONFIG),
        "--run-name", "hook-test",
        "--workers-per-rank", "8",
        "--fresh",
    ])
    assert result == 17
    assert reference.ALLOWED_STAGES == original_allowed
    assert reference.load_and_validate_config is original_loader
    assert engine.run_contract_payload is original_contract
    assert engine.DDP is original_ddp
    assert engine.TRAINING_ORIGIN == original_origin


def _capacity_resume_fixture(tmp_path: Path):
    cfg = capacity.load_and_validate_config(CONFIG)
    args = capacity.stable_engine_args(CONFIG, "capacity-resume", 8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_sha = capacity.ensure_run_contract_rank0(run_dir, cfg, args, 8)
    records = {
        "0": engine.epoch_digest_record(
            reference.EXPECTED_DATASET_SIZE, cfg["seed"], 0
        )
    }
    model_state = {"encoder.weight": torch.arange(6).reshape(2, 3).float()}
    optimizer_state = {"state": {}, "param_groups": [{"lr": 2e-4, "params": [0]}]}
    scheduler_state = {"last_epoch": 1, "_last_lr": [2e-4]}
    integrity = {
        "level": "probe_sha256",
        "model": engine.probe_state_sha256(model_state),
        "optimizer": engine.probe_state_sha256(optimizer_state),
        "scheduler": engine.full_state_sha256(scheduler_state),
        "world_size": 4,
        "all_ranks_identical": True,
    }
    payload = {
        "training_origin": capacity.CAPACITY_TRAINING_ORIGIN,
        "config_sha256": engine.sha256_file(CONFIG),
        "config": cfg,
        "split_manifest_sha256": engine.sha256_file(cfg["split_manifest"]),
        "args": engine.stable_args(args, 8),
        "reference_schedule": reference.expected_schedule_payload(),
        "reference_schedule_sha256": reference.schedule_sha256(cfg["seed"]),
        "partition_algorithm": engine.PARTITION_ALGORITHM,
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
        "active_phase": engine.micro_profile_payload(engine.micro_profile_for_epoch(1)),
        "epoch_update_digests": records,
        "schedule_digest": engine.schedule_digest(records),
    }
    return cfg, args, contract_sha, payload


def test_capacity_resume_accepts_exact_fresh_transaction_and_rejects_drift(tmp_path):
    cfg, args, contract_sha, payload = _capacity_resume_fixture(tmp_path)
    capacity.validate_resume_payload(payload, cfg, args, 8, contract_sha)
    assert engine.TRAINING_ORIGIN != capacity.CAPACITY_TRAINING_ORIGIN

    wrong_origin = copy.deepcopy(payload)
    wrong_origin["training_origin"] = engine.TRAINING_ORIGIN
    with pytest.raises(RuntimeError, match="not an exact-update"):
        capacity.validate_resume_payload(
            wrong_origin, cfg, args, 8, contract_sha, verify_digests=False
        )
    wrong_budget = copy.deepcopy(payload)
    wrong_budget["samples_seen"] -= 64
    with pytest.raises(RuntimeError, match="raw-sample budget"):
        capacity.validate_resume_payload(
            wrong_budget, cfg, args, 8, contract_sha, verify_digests=False
        )


def test_orchestrator_capacity_stage_a_uses_dedicated_four_rank_transaction(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    command = orchestrate.capacity_hybrid_ddp_train_command(
        CONFIG, "aio3_stage_a_coarse_10_10_seed1415926", workers_per_rank=8
    )
    assert command[:5] == [
        orchestrate.sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
    ]
    assert "scripts/train_stage_a_capacity_hybrid_ddp.py" in command
    assert "--stage" not in command
    assert command[-1] == "--fresh"

    last = (
        tmp_path / "artifacts/checkpoints/aio3_stage_a_coarse_10_10_seed1415926/last.pt"
    )
    last.parent.mkdir(parents=True)
    last.write_bytes(b"resume-boundary")
    resumed = orchestrate.capacity_hybrid_ddp_train_command(
        CONFIG, "aio3_stage_a_coarse_10_10_seed1415926", workers_per_rank=8
    )
    assert resumed[-2:] == ["--resume", str(last)]


def test_capacity_orchestration_forbids_generic_stage_a_completion_and_config():
    source = Path(orchestrate.__file__).read_text()
    start = source.index("def run_aio3_capacity_robustness(")
    end = source.index("\ndef main():", start)
    capacity_flow = source[start:end]
    assert "protocol_aio3_10_10_hybrid.yaml" in capacity_flow
    assert "capacity_hybrid_ddp_train_command(" in capacity_flow
    assert capacity_flow.count("hybrid_ddp_complete(") >= 2
    assert "implementation=capacity_hybrid_ddp" in capacity_flow
    assert "checkpoint_complete(stage_a_last" not in capacity_flow
    assert 'train_command(config, "a"' not in capacity_flow
    contracted = {path.resolve() for path in orchestrate.CONTRACTS}
    assert (ROOT / "scripts/train_stage_a_capacity_hybrid_ddp.py").resolve() in contracted
    assert (ROOT / "configs/protocol_aio3_10_10_hybrid.yaml").resolve() in contracted
    watchdog = (ROOT / "scripts/watchdog.sh").read_text()
    assert "train_stage_a_capacity_hybrid_ddp" in watchdog
    assert "protocol_aio3_10_10_hybrid.yaml" in watchdog
    completion_start = source.index("def hybrid_ddp_complete(")
    completion_end = source.index("\ndef assert_matching_hybrid_ddp_update_digests", completion_start)
    completion = source[completion_start:completion_end]
    assert 'len(\n            selection["top3_records"]\n        ) != 3' in completion
    assert "EXPECTED_TOTAL_STEPS" in completion
    assert "EXPECTED_TOTAL_SAMPLES" in completion
    assert "implementation.validate_resume_payload(" in completion
