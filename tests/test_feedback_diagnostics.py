import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.eval_feedback_diagnostics as diagnostics
import scripts.orchestrate as orchestrate


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_test_is_unconditionally_rejected():
    with pytest.raises(PermissionError, match="official_test is never unlockable"):
        diagnostics.validate_diagnostic_split("official_test")
    assert diagnostics.validate_diagnostic_split("locked_val") == "locked_val"
    assert (
        diagnostics.validate_diagnostic_split("train_diagnostic")
        == "train_diagnostic"
    )


def test_metric_math_and_zero_vector_policy():
    # Four 8-D spatial vectors. Their first channels are:
    # target=[1,1,0,1], prediction=[1,0,1,-1].
    target = torch.zeros(1, 8, 1, 4)
    prediction = torch.zeros_like(target)
    target[:, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    prediction[:, 0, 0] = torch.tensor([1.0, 0.0, 1.0, -1.0])
    accumulator = diagnostics.FeedbackDiagnosticsAccumulator(
        entropy_bins=16, entropy_range=2.0
    )
    accumulator.update(prediction, target)
    result = accumulator.finalize()

    assert result["vector_count"] == 4
    assert result["scalar_count"] == 32
    assert result["scalar_mae"] == pytest.approx(4.0 / 32.0)
    assert result["channel_mae"][0] == pytest.approx(1.0)
    cosine = result["cosine"]
    assert cosine["target_valid_count"] == 3
    assert cosine["target_zero_count"] == 1
    assert cosine["prediction_zero_count"] == 1
    assert cosine["both_nonzero_count"] == 2
    # Target-valid cosine values are 1, 0 (zero prediction), and -1.
    assert cosine["mean_over_target_valid_zero_prediction_is_zero"] == pytest.approx(0.0)
    assert cosine["mean_over_both_nonzero_diagnostic"] == pytest.approx(0.0)


def test_zero_code_has_zero_entropy_variance_and_undefined_cosine():
    zero = torch.zeros(2, 8, 3, 4)
    accumulator = diagnostics.FeedbackDiagnosticsAccumulator(
        entropy_bins=16, entropy_range=4.0
    )
    accumulator.update(zero, zero)
    result = accumulator.finalize()
    assert result["scalar_mae"] == 0.0
    assert result["cosine"]["target_valid_count"] == 0
    assert result["cosine"]["mean_over_target_valid_zero_prediction_is_zero"] is None
    for kind in ("prediction", "target", "error"):
        distribution = result["distribution"][kind]
        assert distribution["mean_channel_population_variance"] == 0.0
        assert distribution["mean_channel_marginal_entropy_bits"] == 0.0


def test_streaming_updates_equal_single_update():
    generator = torch.Generator().manual_seed(11)
    prediction = torch.randn(2, 8, 3, 4, generator=generator)
    target = torch.randn(2, 8, 3, 4, generator=generator)
    one = diagnostics.FeedbackDiagnosticsAccumulator()
    one.update(prediction, target)
    split = diagnostics.FeedbackDiagnosticsAccumulator()
    split.update(prediction[:1], target[:1])
    split.update(prediction[1:], target[1:])
    first = one.finalize()
    second = split.finalize()
    assert first["scalar_mae"] == pytest.approx(second["scalar_mae"])
    assert first["scalar_rmse"] == pytest.approx(second["scalar_rmse"])
    assert first["cosine"] == pytest.approx(second["cosine"])
    for kind in ("prediction", "target", "error"):
        for field in (
            "channel_mean",
            "channel_population_variance",
            "channel_marginal_entropy_bits",
        ):
            assert first["distribution"][kind][field] == pytest.approx(
                second["distribution"][kind][field]
            )


def test_scale_macro_is_equal_scale_not_vector_weighted():
    rows = {}
    for index, name in enumerate(diagnostics.SCALE_NAMES, 1):
        rows[name] = {
            "scalar_mae": float(index),
            "scalar_rmse": float(index + 1),
            "cosine": {
                "mean_over_target_valid_zero_prediction_is_zero": index / 10.0
            },
            "distribution": {
                key: {
                    "mean_channel_population_variance": float(index),
                    "mean_channel_marginal_entropy_bits": float(index + 2),
                }
                for key in ("prediction", "target")
            },
        }
    result = diagnostics.scale_macro_summary(rows)
    assert result["scalar_mae"] == pytest.approx(2.5)
    assert result["cosine_target_valid"] == pytest.approx(0.25)


def test_train_selection_deduplicates_and_is_deterministic():
    samples = [
        SimpleNamespace(task="denoise25", degraded=None, clean="a", sigma=25),
        SimpleNamespace(task="denoise25", degraded=None, clean="a", sigma=25),
        SimpleNamespace(task="denoise25", degraded=None, clean="b", sigma=25),
        SimpleNamespace(task="derain", degraded="r1", clean="g1", sigma=0),
        SimpleNamespace(task="derain", degraded="r2", clean="g2", sigma=0),
    ]
    dataset = SimpleNamespace(samples=samples)
    first = diagnostics.deterministic_train_indices(dataset, limit_per_task=1)
    second = diagnostics.deterministic_train_indices(dataset, limit_per_task=1)
    assert first == second
    indices, counts, digest = first
    assert len(indices) == 2
    assert counts == {"denoise25": 1, "derain": 1}
    assert len(digest) == 64


def test_provenance_binds_checkpoint_config_split_stats_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    config_path = tmp_path / "stage_b.yaml"
    split_path = tmp_path / "locked_split.json"
    stats_path = tmp_path / "stats.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    split_path.write_text('{"locked_groups": []}\n')
    split_sha = _sha(split_path)
    stats_path.write_text(
        json.dumps({"protocol": "aio3", "split_manifest_sha256": split_sha}) + "\n"
    )
    cfg = {
        "protocol": "aio3",
        "split_manifest": str(split_path),
        "coordinate_stats": str(stats_path),
    }
    config_path.write_text("protocol: aio3\n")
    checkpoint_path.write_bytes(b"formal-checkpoint-placeholder")
    config_sha = _sha(config_path)
    source_path = tmp_path / "scripts/train.py"
    source_path.parent.mkdir()
    source_path.write_text("# frozen training code\n")
    run_dir = tmp_path / "artifacts/checkpoints/p12"
    run_dir.mkdir(parents=True)
    contract = {
        "feedback": "O12",
        "stage": "b_predicted",
        "config_sha256": config_sha,
        "split_manifest_sha256": split_sha,
        "coordinate_stats_sha256": _sha(stats_path),
        "code_sha256": {"scripts/train.py": _sha(source_path)},
    }
    contract_path = run_dir / "run_contract.json"
    contract_path.write_text(json.dumps(contract) + "\n")
    payload = {
        "config": cfg,
        "config_sha256": config_sha,
        "split_manifest_sha256": split_sha,
        "args": {
            "stage": "b_predicted",
            "feedback": "O12",
            "run_name": "p12",
            "run_contract_sha256": _sha(contract_path),
        },
    }
    result = diagnostics.validate_checkpoint_provenance(
        payload=payload,
        cfg=cfg,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        feedback="O12",
    )
    assert result["feedback_interface_mode"] == "O12"
    assert result["feedback_supervision_mode"] == "O12"
    assert result["checkpoint_sha256"] == _sha(checkpoint_path)
    assert result["run_contract_sha256"] == _sha(contract_path)

    payload["split_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="split-manifest"):
        diagnostics.validate_checkpoint_provenance(
            payload=payload,
            cfg=cfg,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            feedback="O12",
        )


def test_atomic_json_has_no_partial_file(tmp_path):
    output = tmp_path / "diagnostic.json"
    diagnostics.atomic_write_json(output, {"status": "COMPLETE", "value": 1})
    diagnostics.atomic_write_json(output, {"status": "COMPLETE", "value": 2})
    assert json.loads(output.read_text())["value"] == 2
    assert not list(tmp_path.glob("*.tmp.*"))


def test_odd_resolution_crops_every_scale_to_fully_observed_native_cells():
    original = (127, 131)
    padded_shapes = ((128, 136), (64, 68), (32, 34), (16, 17))
    expected_shapes = ((127, 131), (63, 65), (31, 32), (15, 16))
    for scale_index, (padded, expected) in enumerate(
        zip(padded_shapes, expected_shapes)
    ):
        tensor = torch.ones(1, 8, *padded)
        tensor[..., expected[0] :, :] = -1
        tensor[..., :, expected[1] :] = -1
        cropped = diagnostics.crop_to_valid_native_region(
            tensor, original, scale_index
        )
        assert cropped.shape[-2:] == expected
        assert torch.all(cropped == 1)
        accumulator = diagnostics.FeedbackDiagnosticsAccumulator()
        accumulator.update(cropped, cropped)
        assert accumulator.finalize()["vector_count"] == expected[0] * expected[1]

    exact = torch.ones(1, 8, 16, 24)
    assert diagnostics.crop_to_valid_native_region(
        exact, (128, 192), 3
    ).shape == exact.shape
    with pytest.raises(ValueError, match="too small"):
        diagnostics.crop_to_valid_native_region(torch.ones(1, 8, 1, 1), (7, 7), 3)


def _complete_metric_block(vector_count=5):
    entropy = [1.0] * 8
    distribution = {
        "channel_mean": [0.0] * 8,
        "channel_population_variance": [1.0] * 8,
        "mean_channel_population_variance": 1.0,
        "channel_marginal_entropy_bits": entropy,
        "mean_channel_marginal_entropy_bits": 1.0,
        "mean_channel_normalized_entropy": 1.0 / 6.0,
    }
    return {
        "vector_count": vector_count,
        "scalar_count": vector_count * 8,
        "channel_mae": [0.25] * 8,
        "scalar_mae": 0.25,
        "channel_rmse": [0.5] * 8,
        "scalar_rmse": 0.5,
        "cosine": {
            "target_valid_count": vector_count,
            "target_zero_count": 0,
            "prediction_zero_count": 0,
            "both_nonzero_count": vector_count,
            "target_valid_fraction": 1.0,
            "prediction_zero_fraction": 0.0,
            "mean_over_target_valid_zero_prediction_is_zero": 0.5,
            "mean_over_both_nonzero_diagnostic": 0.5,
        },
        "distribution": {
            key: copy.deepcopy(distribution)
            for key in ("prediction", "target", "error")
        },
    }


def _write_complete_cache_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrate, "CODE_ROOT", tmp_path)
    relative_code = set(orchestrate.FEEDBACK_DIAGNOSTIC_CODE_RELATIVE_PATHS) | {
        "src/losses/objectives.py",
        "scripts/stage_b_runtime.py",
    }
    for relative in relative_code:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n")

    split = tmp_path / "artifacts/manifests/locked_split_aio3.json"
    stats = tmp_path / "artifacts/stats/coordinate_stats_aio3.json"
    split.parent.mkdir(parents=True)
    stats.parent.mkdir(parents=True)
    split.write_text('{"locked_groups": ["one"]}\n')
    split_sha = _sha(split)
    stats.write_text(json.dumps({
        "protocol": "aio3", "split_manifest_sha256": split_sha
    }) + "\n")

    config = tmp_path / "configs/stage_b_aio3.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({
        "protocol": "aio3",
        "split_manifest": str(split.resolve()),
        "coordinate_stats": str(stats.resolve()),
    }) + "\n")
    checkpoint = (
        tmp_path / "artifacts/checkpoints/aio3_predicted_o12_formal_s1"
        / "formal_best_model.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"formal predicted residual-code checkpoint")
    run_contract = checkpoint.parent / "run_contract.json"
    contract_payload = {
        "feedback": "O12",
        "stage": "b_predicted",
        "config_sha256": _sha(config),
        "split_manifest_sha256": split_sha,
        "coordinate_stats_sha256": _sha(stats),
        "code_sha256": orchestrate.current_train_code_hashes(),
    }
    run_contract.write_text(json.dumps(contract_payload) + "\n")

    scales = {
        name: _complete_metric_block(vector_count=5)
        for name in diagnostics.SCALE_NAMES
    }
    payload = {
        "schema": "srsc.predicted_feedback_diagnostics.v1",
        "status": "COMPLETE",
        "split": "locked_val",
        "protocol": "aio3",
        "feedback_interface_mode": "O12",
        "feedback_supervision_mode": "O12",
        "selection": {
            "policy": "all preregistered locked-validation images",
            "complete_split": True,
            "selection_sha256": "1" * 64,
        },
        "metric_parameters": {
            "channels": 8,
            "zero_epsilon": 1e-6,
            "entropy_bins": 64,
            "entropy_range": 8.0,
            "entropy_clipping": True,
        },
        "spatial_validity": {
            "policy": "test fixture",
            "scale_divisors": {"S1": 1, "S2": 2, "S3": 4, "S4": 8},
            "complete_source_block_required": True,
            "model_padding_included": False,
        },
        "image_count": 2,
        "image_count_per_task": {"dehaze": 1, "derain": 1},
        "per_scale": scales,
        "pooled_aggregate": _complete_metric_block(vector_count=20),
        "scale_macro": diagnostics.scale_macro_summary(scales),
        "provenance": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha(checkpoint),
            "config": str(config.resolve()),
            "config_sha256": _sha(config),
            "split_manifest": str(split.resolve()),
            "split_manifest_sha256": split_sha,
            "coordinate_stats": str(stats.resolve()),
            "coordinate_stats_sha256": _sha(stats),
            "run_contract": str(run_contract.resolve()),
            "run_contract_sha256": _sha(run_contract),
            "checkpoint_stage": "b_predicted",
            "feedback_interface_mode": "O12",
            "feedback_supervision_mode": "O12",
        },
        "code_sha256": orchestrate.feedback_diagnostic_code_hashes(),
        "runtime": {"elapsed_seconds": 1.0},
    }
    output = tmp_path / "artifacts/metrics/feedback_diagnostics/aio3_p12.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(payload) + "\n")
    return output, checkpoint, config, split, stats, run_contract, payload


def test_diagnostic_cache_rejects_fake_complete_and_every_provenance_drift(
    tmp_path, monkeypatch
):
    output, checkpoint, config, split, stats, run_contract, payload = (
        _write_complete_cache_fixture(tmp_path, monkeypatch)
    )

    def complete():
        return orchestrate.feedback_diagnostics_complete(
            output, checkpoint=checkpoint, config=config, feedback="O12"
        )

    assert complete()

    fake = copy.deepcopy(payload)
    fake["per_scale"]["S1"]["scalar_mae"] = float("nan")
    output.write_text(json.dumps(fake) + "\n")
    assert not complete()

    fake = copy.deepcopy(payload)
    fake["selection"]["complete_split"] = False
    output.write_text(json.dumps(fake) + "\n")
    assert not complete()

    fake = copy.deepcopy(payload)
    fake["code_sha256"]["scripts/eval_feedback_diagnostics.py"] = "0" * 64
    output.write_text(json.dumps(fake) + "\n")
    assert not complete()

    output.write_text(json.dumps(payload) + "\n")
    split.write_text('{"locked_groups": ["drift"]}\n')
    assert not complete()
    split.write_text('{"locked_groups": ["one"]}\n')
    assert complete()

    stats.write_text(json.dumps({
        "protocol": "aio5", "split_manifest_sha256": _sha(split)
    }) + "\n")
    assert not complete()
    stats.write_text(json.dumps({
        "protocol": "aio3", "split_manifest_sha256": _sha(split)
    }) + "\n")
    assert complete()

    contract = json.loads(run_contract.read_text())
    contract["feedback"] = "O7"
    run_contract.write_text(json.dumps(contract) + "\n")
    assert not complete()
