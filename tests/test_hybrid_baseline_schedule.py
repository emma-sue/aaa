import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from scripts import train_baseline_hybrid as hybrid


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"


def test_hybrid_config_and_exact_budget_boundaries():
    cfg = hybrid.load_and_validate_config(CONFIG)
    assert cfg["protocol"] == "aio3"
    assert hybrid.phase_for_epoch(0).effective_batch == 64
    assert hybrid.phase_for_epoch(54).effective_batch == 64
    assert hybrid.phase_for_epoch(55).effective_batch == 120
    assert hybrid.phase_for_epoch(239).effective_batch == 120
    assert hybrid.budget_before_epoch(0) == (0, 0)
    assert hybrid.budget_before_epoch(55) == (118_305, 7_571_520)
    assert hybrid.budget_before_epoch(56) == (119_452, 7_709_160)
    assert hybrid.budget_before_epoch(240) == (330_500, 33_034_920)
    assert hybrid.expected_progress(55, 8) == (118_306, 7_571_640)
    with pytest.raises(ValueError, match="optimizer-step boundary"):
        hybrid.expected_progress(55, 1)


def test_legacy_epoch_indices_match_pytorch23_random_sampler_order():
    dataset = TensorDataset(torch.arange(97))
    generator = torch.Generator()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, generator=generator
    )
    seed, epoch = 1234, 9
    generator.manual_seed(seed + epoch)
    observed = torch.tensor([int(batch[0].item()) for batch in loader])
    reconstructed = hybrid._legacy_single_gpu_indices(len(dataset), seed, epoch)
    assert torch.equal(observed, reconstructed)


def test_four_rank_reconstruction_matches_distributed_sampler_update_sets():
    size = hybrid.EXPECTED_DATASET_SIZE
    seed, epoch = 1415926, 55
    reconstructed = hybrid._reconstructed_four_rank_indices(size, seed, epoch)
    rank_batches = []
    for rank in range(4):
        sampler = DistributedSampler(
            range(size), num_replicas=4, rank=rank, shuffle=True,
            seed=seed, drop_last=True,
        )
        sampler.set_epoch(epoch)
        local = torch.tensor(list(sampler), dtype=torch.int64)[:34_410]
        rank_batches.append(local.view(1_147, 30))
    reference = torch.stack(rank_batches, dim=1).reshape(-1)
    assert torch.equal(reconstructed, reference)
    assert reconstructed.numel() == 137_640
    assert reconstructed.unique().numel() == reconstructed.numel()


def test_epoch_index_digest_is_deterministic_and_phase_sensitive():
    first = hybrid.epoch_indices(
        hybrid.EXPECTED_DATASET_SIZE, 1415926, 54
    )
    repeat = hybrid.epoch_indices(
        hybrid.EXPECTED_DATASET_SIZE, 1415926, 54
    )
    next_epoch = hybrid.epoch_indices(
        hybrid.EXPECTED_DATASET_SIZE, 1415926, 55
    )
    assert hybrid.index_digest(first) == hybrid.index_digest(repeat)
    assert hybrid.index_digest(first) != hybrid.index_digest(next_epoch)
    with pytest.raises(ValueError, match="train size drift"):
        hybrid.epoch_indices(hybrid.EXPECTED_DATASET_SIZE - 1, 1415926, 0)


def _resume_fixture(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    split = tmp_path / "split.json"
    split.write_text('{"protocol":"aio3"}\n')
    cfg["split_manifest"] = str(split)
    config = tmp_path / "hybrid.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = SimpleNamespace(
        config=str(config), stage="baseline_matched", run_name="hybrid-test",
        fresh=False, resume=str(tmp_path / "last.pt"), workers_override=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_sha = hybrid.ensure_run_contract(run_dir, cfg, args, workers=8)
    digest = hybrid.index_digest(
        hybrid.epoch_indices(hybrid.EXPECTED_DATASET_SIZE, cfg["seed"], 0)
    )
    payload = {
        "training_origin": "fresh_hybrid_baseline",
        "config_sha256": hybrid.sha256_file(config),
        "config": cfg,
        "split_manifest_sha256": hybrid.sha256_file(split),
        "args": hybrid.stable_args(args, 8),
        "hybrid_schedule": hybrid.expected_schedule_payload(),
        "hybrid_schedule_sha256": hybrid.schedule_sha256(cfg["seed"]),
        "run_contract_sha256": contract_sha,
        "epoch": 1,
        "batch_in_epoch": 0,
        "step": 2_151,
        "samples_seen": 137_664,
        "scheduler": {"last_epoch": 1},
        "epoch_index_digests": {"0": digest},
    }
    return cfg, args, contract_sha, payload


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("training_origin", "legacy", "not a fresh hybrid"),
        ("hybrid_schedule_sha256", "0" * 64, "schedule hash"),
        ("step", 2_150, "optimizer-step budget"),
        ("samples_seen", 137_600, "sample budget"),
    ],
)
def test_resume_contract_rejects_budget_and_schedule_drift(
    tmp_path, field, replacement, message
):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    broken = copy.deepcopy(payload)
    broken[field] = replacement
    with pytest.raises(RuntimeError, match=message):
        hybrid.validate_resume_payload(
            broken, cfg, args, 8, contract_sha, verify_digests=False
        )


def test_resume_contract_rejects_index_digest_tampering(tmp_path):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    hybrid.validate_resume_payload(payload, cfg, args, 8, contract_sha)
    payload["epoch_index_digests"]["0"] = "f" * 64
    with pytest.raises(RuntimeError, match="digest mismatch"):
        hybrid.validate_resume_payload(payload, cfg, args, 8, contract_sha)


def test_mid_epoch_resume_cursor_is_an_exact_optimizer_boundary(tmp_path):
    cfg, args, contract_sha, payload = _resume_fixture(tmp_path)
    payload.update({
        "epoch": 0,
        "batch_in_epoch": 4_000,
        "step": 1_000,
        "samples_seen": 64_000,
        "scheduler": {"last_epoch": 0},
    })
    hybrid.validate_resume_payload(payload, cfg, args, 8, contract_sha)
    payload["batch_in_epoch"] = 3_999
    with pytest.raises(ValueError, match="optimizer-step boundary"):
        hybrid.validate_resume_payload(
            payload, cfg, args, 8, contract_sha, verify_digests=False
        )


def test_hybrid_entry_rejects_other_stages_and_protocols(tmp_path):
    with pytest.raises(SystemExit):
        hybrid.parse_args([
            "--config", str(CONFIG), "--stage", "a", "--run-name", "bad",
            "--fresh",
        ])
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["protocol"] = "aio5"
    bad = tmp_path / "aio5.yaml"
    bad.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="only protocol=aio3"):
        hybrid.load_and_validate_config(bad)


def test_schedule_hash_and_contract_are_canonical(tmp_path):
    cfg, args, first_sha, _ = _resume_fixture(tmp_path)
    run_dir = tmp_path / "run"
    second_sha = hybrid.ensure_run_contract(run_dir, cfg, args, workers=8)
    assert first_sha == second_sha
    contract = json.loads((run_dir / "run_contract.json").read_text())
    assert contract["schedule_sha256"] == hybrid.schedule_sha256(cfg["seed"])
    assert contract["schedule"]["expected_total_steps"] == 330_500
    assert contract["schedule"]["expected_total_samples"] == 33_034_920
    assert "bitwise matched" in contract["stochastic_claim_boundary"]
