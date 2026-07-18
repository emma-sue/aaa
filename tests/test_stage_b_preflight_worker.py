from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from scripts import preflight_stage_b_runtime as preflight
from scripts import train


class TinyAssessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(9, 8, 1)
        self.calls = 0

    def forward(self, x, y1, _features):
        self.calls += 1
        state = self.head(torch.cat((x, y1, y1 - x), dim=1))
        return [state, state, state, state]


class TinyStageB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 3, 1)
        self.d1 = nn.Conv2d(3, 3, 1)
        self.assessor = TinyAssessor()
        self.y1_pyramid = nn.Conv2d(3, 3, 1)
        self.d2 = nn.Conv2d(11, 3, 1)
        self.oracle_ceiling_adapter = nn.Conv2d(81, 8, 1, bias=False)
        self.predicted_feedback_mode = None
        self.force_zero_state = False

    def _encode_coarse(self, x):
        feature = self.encoder(x)
        y1 = x + self.d1(feature)
        return (feature, feature, feature, feature), y1

    def _run_d2(self, _x, y1, _features, states):
        return y1 + self.d2(torch.cat((y1, states[0]), dim=1))

    def forward_details(self, x):
        features, y1 = self._encode_coarse(x)
        states = self.assessor(x, y1, features)
        y2 = self._run_d2(x, y1, features, states)
        return SimpleNamespace(y1=y1, y2=y2, states=states, features=features)


class TinyCoordinateBuilder:
    def __call__(self, x, y1, gt, sizes, requested=None):
        assert requested in ({"O7"}, {"O12"})
        outputs = []
        for height, width in sizes:
            yd = torch.nn.functional.interpolate(
                y1.detach(), size=(height, width), mode="area"
            )
            gd = torch.nn.functional.interpolate(
                gt.detach(), size=(height, width), mode="area"
            )
            error = (gd - yd).mean(dim=1, keepdim=True)
            direction = error.repeat(1, 6, 1, 1)
            state = torch.cat((error, error.abs(), direction), dim=1)
            outputs.append(
                SimpleNamespace(
                    state=state,
                    residual_code=error.repeat(1, 8, 1, 1),
                    q=torch.full_like(error, 0.5),
                    w_dir=torch.ones_like(error),
                )
            )
        return outputs


def _tiny_cfg():
    return {
        "lr": 2e-4,
        "optimizer": "adam",
        "gradient_clip": 0.01,
        "lambda_state": 0.1,
        "lambda_clean": 0.1,
    }


@pytest.mark.parametrize(
    ("stage", "feedback", "assessor_has_grad"),
    (
        ("b_oracle", "O7", False),
        ("b_oracle", "O12", False),
        ("b_predicted", "O7", True),
        ("b_predicted", "O12", True),
    ),
)
def test_three_update_worker_uses_real_shared_backward_and_adam(
    stage: str, feedback: str, assessor_has_grad: bool
):
    torch.manual_seed(17)
    model = TinyStageB()
    train.configure_trainable(model, stage)
    params, optimizer = train.build_optimizer(model, stage, _tiny_cfg())
    result = preflight.run_three_optimizer_updates(
        model=model,
        optimizer=optimizer,
        params=params,
        cfg=_tiny_cfg(),
        stage=stage,
        feedback=feedback,
        builder=TinyCoordinateBuilder(),
        feedback_stats=None,
        device=torch.device("cpu"),
        micro_batch=2,
        accumulation=2,
        crop_size=8,
        updates=3,
    )
    assert result["optimizer_updates"] == 3
    assert result["micro_steps"] == 6
    assert result["adam"]["all_observed_gradient_parameters_have_moments"]
    assert result["gradient_routing"]["frozen_encoder_d1_gradients"] == 0
    assert (
        result["gradient_routing"]["assessor_gradient_parameter_count"] > 0
    ) is assessor_has_grad
    assert model.assessor.calls == 6


def test_formal_training_calls_the_same_stage_b_backward_and_optimizer_helpers():
    source = inspect.getsource(train.main)
    assert "backward_stage_b_microbatch(" in source
    assert "commit_optimizer_update(" in source
    assert "compute_stage_b_terms(" not in source


def test_native_probe_is_tile_zero_shape_only_on_cpu():
    model = TinyStageB()
    observed = {}

    def prediction_fn(model, x, _gt, stage, _builder, feedback, _stats, tile, **_kwargs):
        observed.update(stage=stage, feedback=feedback, tile=tile)
        return x + 0.0 * next(model.parameters()).sum()

    result = preflight.run_native_shape_probe(
        model=model,
        stage="b_predicted",
        feedback="O7",
        builder=TinyCoordinateBuilder(),
        feedback_stats=None,
        device=torch.device("cpu"),
        height=16,
        width=24,
        prediction_fn=prediction_fn,
    )
    assert observed == {"stage": "b_predicted", "feedback": "O7", "tile": 0}
    assert result["input_shape"] == [1, 3, 16, 24]
    assert result["quality_metrics_computed"] is False


def test_worker_source_has_no_metric_official_or_checkpoint_entrypoint():
    source = Path(preflight.__file__).read_text()
    for forbidden_call in (
        "validate_locked(",
        "full_rgb_ssim(",
        "build_test_sets(",
        "commit_pending_validation(",
        "save_checkpoint(",
        "ensure_run_contract(",
    ):
        assert forbidden_call not in source


def test_driver_fails_fatally_on_non_oom_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stats = tmp_path / "stats.json"
    stats.write_text("{}\n")
    split = tmp_path / "split.json"
    split.write_text("{}\n")
    checkpoint = tmp_path / "stage_a.pt"
    checkpoint.write_bytes(b"memory-only-test")
    config = {
        "protocol": "aio5",
        "coordinate_stats": str(stats),
        "split_manifest": str(split),
        "effective_batch": 120,
    }
    template = tmp_path / "stage_b_aio5.yaml"
    template.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    calls = []

    def failed_worker(**kwargs):
        calls.append(kwargs)
        return {"schema": preflight.SCHEMA, "status": "ERROR", "error": "finite"}

    monkeypatch.setattr(preflight, "_run_worker", failed_worker)
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        root=str(tmp_path),
        protocol="aio5",
        stage_a_checkpoint=str(checkpoint),
        main_template=str(template),
        capacity_template=None,
        capacity_stage_a_checkpoint=None,
        candidates_json=json.dumps([[15, 8], [12, 10]]),
        output=str(output),
        probe_height=16,
        probe_width=24,
    )
    payload, passed = preflight.execute_driver(args)
    assert passed is False
    assert len(calls) == 1
    assert payload["fatal_error"]["status"] == "ERROR"
    assert payload["selected_candidate"] is None
    assert payload["all_pass"] is False


def test_native_shape_constants_are_explicit_height_width():
    assert preflight.DEFAULT_NATIVE_SHAPES == {
        "aio3": (736, 544),
        "aio5": (720, 1280),
    }
