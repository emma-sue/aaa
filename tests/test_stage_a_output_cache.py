from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import cache_stage_a_outputs as cache


class TinyLockedDataset:
    def __init__(self, *, gt_offset: float = 0.0):
        self.items = [
            {
                "task": "denoise25",
                "name": "clean-a:25:one",
                "degraded": torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 20,
                "clean": torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 19
                + gt_offset,
            },
            {
                "task": "derain",
                "name": "rain-b:0:two",
                "degraded": torch.full((3, 3, 2), 0.25, dtype=torch.float32),
                "clean": torch.full((3, 3, 2), 0.75, dtype=torch.float32)
                + gt_offset,
            },
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in item.items()
        }


def bindings():
    return {
        "protocol": "aio3",
        "scope": "locked_val",
        "config": {"path": "/frozen/protocol.yaml", "sha256": "1" * 64},
        "split_manifest": {"path": "/frozen/split.json", "sha256": "2" * 64},
        "dataset_lists": {"noisy/denoise.txt": {"sha256": "3" * 64}},
        "stage_a_checkpoint": {"path": "/frozen/stage_a.pt", "sha256": "4" * 64},
        "model_code_sha256": {"src/net/srsc_lite.py": "5" * 64},
        "forward_contract": {"autocast": "cuda_bfloat16", "clamp": False},
    }


class CountingPredictor:
    def __init__(self, fail_at: int | None = None, drift_second_pass: bool = False):
        self.calls = 0
        self.fail_at = fail_at
        self.drift_second_pass = drift_second_pass

    def __call__(self, degraded: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("injected replay failure")
        output = degraded.float() * 0.5 + 0.125
        if self.drift_second_pass and self.calls > 2:
            output = output.clone()
            output.view(-1)[0] += 1e-4
        return output


def test_two_pass_atomic_cache_is_lossless_and_existing_run_is_idempotent(tmp_path: Path):
    target = tmp_path / "stage_a_cache"
    predictor = CountingPredictor()
    manifest, created = cache.ensure_locked_val_cache(
        target,
        bindings(),
        TinyLockedDataset,
        predictor,
    )
    assert created is True
    assert predictor.calls == 4  # two observations times two complete passes
    assert manifest["status"] == cache.CACHE_STATUS
    assert manifest["scope"] == "locked_val"
    assert manifest["official_test_forbidden"] is True
    assert manifest["item_count"] == 2
    assert (target / "manifest.json").is_file()
    assert (target / "manifest.sha256").is_file()

    for record in manifest["items"]:
        shard = cache.load_npy_strict(target / record["shard"])
        assert shard.dtype == np.dtype("<f4")
        assert cache.array_sha256(shard) == record["y1_sha256"]

    def must_not_run(_):
        raise AssertionError("existing offline verification initialized the model")

    second, created_again = cache.ensure_locked_val_cache(
        target,
        bindings(),
        TinyLockedDataset,
        must_not_run,
    )
    assert created_again is False
    assert second["aggregate_sha256"] == manifest["aggregate_sha256"]


def test_existing_cache_rejects_binding_input_gt_and_shard_drift(tmp_path: Path):
    target = tmp_path / "cache"
    cache.ensure_locked_val_cache(target, bindings(), TinyLockedDataset, CountingPredictor())

    changed = copy.deepcopy(bindings())
    changed["stage_a_checkpoint"]["sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="binding drift"):
        cache.ensure_locked_val_cache(target, changed, TinyLockedDataset, CountingPredictor())

    with pytest.raises(RuntimeError, match="gt_sha256 drift"):
        cache.ensure_locked_val_cache(
            target,
            bindings(),
            lambda: TinyLockedDataset(gt_offset=0.01),
            CountingPredictor(),
        )

    manifest = cache._read_manifest(target)
    shard = target / manifest["items"][0]["shard"]
    shard.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="shard SHA256 mismatch"):
        cache.ensure_locked_val_cache(
            target, bindings(), TinyLockedDataset, CountingPredictor()
        )


@pytest.mark.parametrize(
    "predictor",
    [
        CountingPredictor(fail_at=3),
        CountingPredictor(drift_second_pass=True),
    ],
)
def test_failed_second_pass_never_commits_a_partial_cache(
    tmp_path: Path, predictor: CountingPredictor
):
    target = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        cache.ensure_locked_val_cache(
            target, bindings(), TinyLockedDataset, predictor
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".cache.staging.*"))


def test_explicit_online_reverification_replays_every_existing_item(tmp_path: Path):
    target = tmp_path / "cache"
    cache.ensure_locked_val_cache(target, bindings(), TinyLockedDataset, CountingPredictor())
    replay = CountingPredictor()
    _, created = cache.ensure_locked_val_cache(
        target,
        bindings(),
        TinyLockedDataset,
        replay,
        online_reverify_existing=True,
    )
    assert created is False
    assert replay.calls == len(TinyLockedDataset())


def test_official_test_is_unreachable_and_output_path_is_rejected(tmp_path: Path):
    assert cache.validate_cache_scope("locked_val") == "locked_val"
    with pytest.raises(PermissionError, match="official_test is never reachable"):
        cache.validate_cache_scope("official_test")
    with pytest.raises(PermissionError, match="must not target official_test"):
        cache.validate_output_path(tmp_path / "official_test" / "stage_a_y1")
