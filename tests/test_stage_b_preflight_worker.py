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
from scripts import stage_b_runtime as runtime
from scripts import train


def _physical_gpu_inventory(*, used_mib: int = 64, name: str = "RTX 4090"):
    total_mib = 24 * 1024
    records = []
    for index in runtime.REQUIRED_PHYSICAL_GPU_INDICES:
        free_mib = total_mib - used_mib
        records.append(
            {
                "physical_index": index,
                "uuid": f"GPU-{index:032d}",
                "name": name,
                "total_memory_mib": total_mib,
                "free_memory_mib": free_mib,
                "total_memory_bytes": total_mib * 2**20,
                "free_memory_bytes": free_mib * 2**20,
                "startup_used_bytes": used_mib * 2**20,
                "startup_idle_pass": (
                    used_mib * 2**20 <= runtime.MAX_STARTUP_GPU_USED_BYTES
                ),
                "compute_mode": "Default",
            }
        )
    return {
        "schema": runtime.PHYSICAL_GPU_INVENTORY_SCHEMA,
        "source": "nvidia-smi_parent_no_cuda_context",
        "command": preflight._nvidia_smi_inventory_command(),
        "required_physical_indices": list(runtime.REQUIRED_PHYSICAL_GPU_INDICES),
        "max_startup_used_bytes": runtime.MAX_STARTUP_GPU_USED_BYTES,
        "gpu_count": 4,
        "homogeneous_name": name,
        "homogeneous_total_memory_bytes": total_mib * 2**20,
        "homogeneous_compute_mode": "Default",
        "all_startup_idle": all(
            record["startup_idle_pass"] for record in records
        ),
        "gpus": records,
    }


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
    monkeypatch.setattr(
        preflight, "_query_physical_gpu_inventory", _physical_gpu_inventory
    )
    calls = []

    def failed_worker(**kwargs):
        calls.append(kwargs)
        return {"schema": preflight.SCHEMA, "status": "ERROR", "error": "finite"}

    monkeypatch.setattr(preflight, "_run_worker", failed_worker)
    bindings = {
        "protocol": "aio5",
        "roles": {
            "main": {
                "template_path": str(template.resolve()),
                "template_sha256": "mock",
                "stage_a_checkpoint": str(checkpoint.resolve()),
                "stage_a_checkpoint_sha256": "mock",
                "init_policy": "COARSE_ONLY_FROM_SELECTED_STAGE_A",
                "coordinate_stats_path": str(stats.resolve()),
                "coordinate_stats_sha256": "mock",
                "coordinate_stats_origin": "TEMPLATE_BOUND",
            }
        },
        "split_manifest": {"path": str(split.resolve()), "sha256": "mock"},
        "code_sha256": {"mock": "mock"},
    }
    monkeypatch.setattr(
        preflight, "preflight_input_bindings", lambda *_args, **_kwargs: bindings
    )
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
    assert payload["fatal_error"]["status"] == "INVALID_WORKER_EVIDENCE"
    assert payload["selected_candidate"] is None
    assert payload["all_pass"] is False


def test_native_shape_constants_are_explicit_height_width():
    assert preflight.DEFAULT_NATIVE_SHAPES == {
        "aio3": (736, 544),
        "aio5": (720, 1280),
    }


def _nvidia_smi_rows(
    *,
    count: int = 4,
    used_mib: int = 64,
    heterogeneous_index: int | None = None,
) -> str:
    total_mib = 24 * 1024
    rows = []
    for index in range(count):
        name = "RTX 4090 Ti" if index == heterogeneous_index else "RTX 4090"
        rows.append(
            f"{index}, GPU-{index:032d}, {name}, {total_mib}, "
            f"{total_mib - used_mib}, Default"
        )
    return "\n".join(rows) + "\n"


def test_parent_inventory_uses_nvidia_smi_without_torch_cuda(monkeypatch):
    observed = {}

    def runner(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=_nvidia_smi_rows(),
            stderr="",
        )

    monkeypatch.setattr(
        preflight.torch.cuda,
        "is_available",
        lambda: pytest.fail("parent inventory must not initialize CUDA"),
    )
    inventory = preflight._query_physical_gpu_inventory(runner=runner)
    assert observed["command"] == preflight._nvidia_smi_inventory_command()
    assert "env" not in observed["kwargs"]
    assert [gpu["physical_index"] for gpu in inventory["gpus"]] == [0, 1, 2, 3]
    assert inventory["all_startup_idle"] is True
    runtime.validate_physical_gpu_inventory(inventory)


def test_parent_inventory_nvidia_smi_failure_is_fatal():
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=9, stdout="", stderr="driver unavailable")

    with pytest.raises(RuntimeError, match="returncode=9"):
        preflight._query_physical_gpu_inventory(runner=runner)


@pytest.mark.parametrize(
    ("stdout", "message"),
    (
        (_nvidia_smi_rows(count=3), "exactly four"),
        (_nvidia_smi_rows(heterogeneous_index=3), "share model"),
        (_nvidia_smi_rows(used_mib=1024), "significant startup occupancy"),
    ),
)
def test_parent_inventory_rejects_missing_heterogeneous_or_busy_cards(
    stdout: str, message: str
):
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match=message):
        preflight._query_physical_gpu_inventory(runner=runner)


def test_worker_subprocess_is_bound_to_one_physical_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inventory = _physical_gpu_inventory()
    observed = {}

    def runner(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(
            returncode=2,
            stdout=json.dumps(
                {
                    "schema": preflight.SCHEMA,
                    "status": "OOM",
                    "failed_phase": "train_step",
                    "error": "synthetic oom",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", runner)
    payload = preflight._run_worker(
        root=tmp_path,
        protocol="aio5",
        template=tmp_path / "stage_b.yaml",
        role="main",
        checkpoint=None,
        allow_random=False,
        coordinate_stats_override=None,
        stage="b_predicted",
        feedback="O12",
        micro_batch=15,
        accumulation=8,
        height=720,
        width=1280,
        physical_gpu_index=3,
        physical_gpu_inventory=inventory,
    )
    assert payload["status"] == "OOM"
    assert observed["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == inventory[
        "gpus"
    ][3]["uuid"]
    assert observed["kwargs"]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    command = observed["command"]
    assert command[command.index("--physical-gpu-index") + 1] == "3"
    embedded = json.loads(
        command[command.index("--physical-gpu-inventory-json") + 1]
    )
    assert embedded == inventory


def _memory_snapshot(*, free_gib: float = 3.0, reserved_gib: float = 20.0):
    total = 24 * 2**30
    reserved = int(reserved_gib * 2**30)
    free = int(free_gib * 2**30)
    headroom = min(total - reserved, free)
    return {
        "peak_allocated_bytes": reserved - 2**20,
        "peak_reserved_bytes": reserved,
        "free_bytes_after_probe": free,
        "total_bytes": total,
        "headroom_bytes": headroom,
        "required_headroom_bytes": preflight.MIN_HEADROOM_BYTES,
        "headroom_pass": headroom >= preflight.MIN_HEADROOM_BYTES,
    }


def _complete_probe_record(status: str = "PASS"):
    inventory = _physical_gpu_inventory()
    inventory_sha256 = runtime.canonical_json_sha256(inventory)
    gpu_record = inventory["gpus"][0]
    binding = {
        "protocol": "aio3",
        "roles": {
            "main": {
                "template_path": "/registered/stage_b.yaml",
                "template_sha256": "c" * 64,
                "stage_a_checkpoint": "/registered/stage_a.pt",
                "stage_a_checkpoint_sha256": "a" * 64,
                "init_policy": "COARSE_ONLY_FROM_SELECTED_STAGE_A",
                "coordinate_stats_path": "/registered/stats.json",
                "coordinate_stats_sha256": "s" * 64,
                "coordinate_stats_origin": "TEMPLATE_BOUND",
            }
        },
        "split_manifest": {
            "path": "/registered/split.json",
            "sha256": "m" * 64,
        },
        "code_sha256": {"scripts/train.py": "t" * 64},
    }
    train_evidence = {
        "optimizer_updates": 3,
        "micro_steps": 12,
        "expected_micro_steps": 12,
        "finite_losses_gradients_predictions": True,
        "gradient_routing": {
            "frozen_encoder_d1_gradients": 0,
            "d2_gradient_parameter_count": 10,
            "assessor_gradient_parameter_count": 0,
        },
        "adam": {
            "optimizer": "Adam",
            "state_parameter_count": 10,
            "state_tensor_bytes": 123456,
            "all_observed_gradient_parameters_have_moments": True,
        },
        "memory": _memory_snapshot(),
    }
    native_evidence = {
        "input_shape": [1, 3, 736, 544],
        "output_shape": [1, 3, 736, 544],
        "tile": 0,
        "finite_prediction": True,
        "quality_metrics_computed": False,
        "adam_state_retained_bytes": 123456,
        "memory": _memory_snapshot(),
    }
    worker = {
        "schema": preflight.SCHEMA,
        "status": status,
        "protocol": "aio3",
        "template_role": "main",
        "stage": "b_oracle",
        "feedback": "O7",
        "micro_batch": 16,
        "accumulation": 4,
        "effective_batch": 64,
        "init_scope": "coarse_only_fresh_seeded_feedback_path",
        "stage_a_checkpoint": "/registered/stage_a.pt",
        "stage_a_checkpoint_sha256": "a" * 64,
        "config": "/registered/stage_b.yaml",
        "config_sha256": "c" * 64,
        "coordinate_stats_path": "/registered/stats.json",
        "coordinate_stats_sha256": "s" * 64,
        "coordinate_stats_origin": "TEMPLATE_BOUND",
        "split_manifest_path": "/registered/split.json",
        "split_manifest_sha256": "m" * 64,
        "code_sha256": {"scripts/train.py": "t" * 64},
        "physical_gpu_index": 0,
        "physical_gpu_inventory": inventory,
        "physical_gpu_inventory_sha256": inventory_sha256,
        "gpu": {
            "physical_index": 0,
            "uuid": gpu_record["uuid"],
            "name": gpu_record["name"],
            "total_memory_bytes": gpu_record["total_memory_bytes"],
            "cuda_usable_total_memory_bytes": gpu_record["total_memory_bytes"],
            "compute_mode": "Default",
            "logical_cuda_index": 0,
            "visible_device_count": 1,
            "cuda_visible_devices": gpu_record["uuid"],
            "cuda_device_order": "PCI_BUS_ID",
            "compute_capability": [8, 9],
        },
        "software": {"torch": "2.3", "cuda_runtime": "12.1"},
        "train_step": train_evidence,
        "native_val": native_evidence,
        "optimizer_state_retained_for_native_val": True,
        "official_test_accessed": False,
        "quality_metrics_computed": False,
        "checkpoint_written": False,
        "run_contract_written": False,
    }
    return worker, binding


def test_headroom_uses_real_free_memory_not_only_own_peak():
    snapshot = _memory_snapshot(free_gib=1.0, reserved_gib=20.0)
    assert snapshot["headroom_bytes"] == 1 * 2**30
    assert preflight._validate_memory_snapshot(snapshot, "test") is False
    snapshot["headroom_bytes"] = 4 * 2**30
    snapshot["headroom_pass"] = True
    with pytest.raises(RuntimeError, match="real free memory"):
        preflight._validate_memory_snapshot(snapshot, "test")


def test_only_explicit_oom_or_derived_headroom_can_fallback():
    worker, bindings = _complete_probe_record()
    kwargs = dict(
        protocol="aio3",
        role="main",
        stage="b_oracle",
        feedback="O7",
        micro_batch=16,
        accumulation=4,
        height=736,
        width=544,
        input_bindings=bindings,
        physical_gpu_index=0,
        physical_gpu_inventory=worker["physical_gpu_inventory"],
    )
    assert preflight._validate_worker_for_probe(worker, **kwargs) == "PASS"

    incomplete = dict(worker)
    incomplete.pop("native_val")
    with pytest.raises(RuntimeError, match="native probe lacks"):
        preflight._validate_worker_for_probe(incomplete, **kwargs)

    false_headroom, _ = _complete_probe_record(status="HEADROOM_FAIL")
    false_headroom["native_val"]["memory"] = _memory_snapshot(free_gib=1.0)
    assert (
        preflight._validate_worker_for_probe(false_headroom, **kwargs)
        == "HEADROOM_FAIL"
    )

    arbitrary, _ = _complete_probe_record(status="ERROR")
    with pytest.raises(RuntimeError, match="non-memory worker failure"):
        preflight._validate_worker_for_probe(arbitrary, **kwargs)

    swapped = json.loads(json.dumps(worker))
    swapped["physical_gpu_index"] = 1
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        preflight._validate_worker_for_probe(swapped, **kwargs)

    changed_inventory = json.loads(json.dumps(worker))
    changed_inventory["physical_gpu_inventory"]["gpus"][0]["uuid"] = (
        "GPU-tampered"
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        preflight._validate_worker_for_probe(changed_inventory, **kwargs)
