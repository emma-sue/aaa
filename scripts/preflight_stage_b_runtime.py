#!/usr/bin/env python3
"""No-metric Stage-B CUDA memory preflight.

The internal worker executes the same BF16 loss/backward/clip/Adam path as
``scripts/train.py`` for three complete optimizer updates.  It then keeps the
materialized Adam moments resident while probing one native full-image shape
through the formal ``tiled_stage_prediction(tile=0)`` path.

This script never builds an official-test dataset, computes PSNR/SSIM, writes a
checkpoint, or creates a run contract.  The outer driver only aggregates
memory/runtime evidence into the explicitly requested JSON output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    backward_stage_b_microbatch,
    build_coordinate_builder,
    build_model,
    build_optimizer,
    commit_optimizer_update,
    configure_feedback_mode,
    configure_trainable,
    load_formal_init,
    seed_all,
    tiled_stage_prediction,
)
from src.data import build_locked_val  # noqa: E402
from src.data.aio_dataset import locked_sample_name  # noqa: E402


SCHEMA = "srsc.stage_b_memory_preflight.v1"
ALLOWED_STAGES = ("b_oracle", "b_predicted")
ALLOWED_FEEDBACK = ("O7", "O12")
DEFAULT_NATIVE_SHAPES = {
    "aio3": (736, 544),
    "aio5": (720, 1280),
}
MIN_HEADROOM_BYTES = int(1.5 * 2**30)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def largest_locked_val_shape(config: str | Path) -> dict:
    """Select the largest Train-heldout tensor by padded area, without metrics."""
    cfg = yaml.safe_load(Path(config).read_text())
    dataset = build_locked_val(
        cfg["data_root"],
        cfg["list_root"],
        cfg["protocol"],
        cfg["split_manifest"],
    )
    candidates = []
    for index, sample in enumerate(dataset.samples):
        with Image.open(sample.clean) as image:
            width, height = image.size
        # Formal LockedValidationDataset applies _crop_to_base(..., base=16).
        height = (height // 16) * 16
        width = (width // 16) * 16
        name = locked_sample_name(sample)
        candidates.append(
            (
                height * width,
                height,
                width,
                sample.task,
                name,
                index,
                str(sample.clean.resolve()),
            )
        )
    if not candidates:
        raise RuntimeError("locked validation contains no Train-heldout samples")
    # Largest area first; all remaining fields make ties deterministic.
    area, height, width, task, name, index, path = sorted(
        candidates,
        key=lambda row: (-row[0], row[3], row[4], row[5], row[6]),
    )[0]
    expected = DEFAULT_NATIVE_SHAPES[cfg["protocol"]]
    if (height, width) != expected:
        raise RuntimeError(
            "largest locked-val shape drift: "
            f"actual(H,W)={(height, width)} expected(H,W)={expected}"
        )
    return {
        "height": height,
        "width": width,
        "area": area,
        "task": task,
        "name": name,
        "index": index,
        "clean_path": path,
        "source": "Train-heldout_locked_val_metadata_after_crop_to_base16",
        "shape_semantics": "height_width",
        "quality_metrics_computed": False,
        "official_test_accessed": False,
    }


def _finite_scalar(value: torch.Tensor, label: str) -> float:
    detached = value.detach().float()
    if detached.numel() != 1 or not torch.isfinite(detached).all():
        raise FloatingPointError(f"non-finite/non-scalar {label}")
    return float(detached.item())


def _assert_tensor_finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value.detach().float()).all():
        raise FloatingPointError(f"non-finite tensor: {label}")


def _named_parameters(model) -> dict[int, tuple[str, torch.nn.Parameter]]:
    return {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}


def _validate_gradient_routing(
    stage: str,
    gradient_updates: dict[str, int],
) -> dict:
    observed = set(gradient_updates)
    frozen_leaks = sorted(
        name for name in observed if name.startswith(("encoder.", "d1."))
    )
    if frozen_leaks:
        raise RuntimeError(f"Stage-B frozen-path gradient leak: {frozen_leaks[:8]}")
    d2 = sorted(name for name in observed if name.startswith("d2."))
    if not d2:
        raise RuntimeError("Stage-B D2 received no gradient")
    assessor = sorted(name for name in observed if name.startswith("assessor."))
    if stage == "b_predicted" and not assessor:
        raise RuntimeError("predicted Stage-B assessor received no gradient")
    if stage == "b_oracle" and assessor:
        raise RuntimeError("Oracle Stage-B assessor unexpectedly received gradient")
    return {
        "frozen_encoder_d1_gradients": 0,
        "d2_gradient_parameter_count": len(d2),
        "assessor_gradient_parameter_count": len(assessor),
    }


def validate_adam_state(
    optimizer,
    named: dict[int, tuple[str, torch.nn.Parameter]],
    gradient_updates: dict[str, int],
) -> dict:
    """Verify lazy Adam moments for every parameter used by the real path."""
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("Stage-B memory preflight requires the registered Adam optimizer")
    state_names = []
    state_bytes = 0
    for parameter, state in optimizer.state.items():
        if id(parameter) not in named:
            raise RuntimeError("optimizer state contains an unknown parameter")
        name = named[id(parameter)][0]
        if name not in gradient_updates:
            raise RuntimeError(f"Adam state exists for a parameter without gradient: {name}")
        missing = {"step", "exp_avg", "exp_avg_sq"} - set(state)
        if missing:
            raise RuntimeError(f"Adam state missing {sorted(missing)} for {name}")
        expected_steps = gradient_updates[name]
        actual_steps = int(float(torch.as_tensor(state["step"]).detach().cpu()))
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"Adam step mismatch for {name}: actual={actual_steps} "
                f"expected={expected_steps}"
            )
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            if tensor.shape != parameter.shape:
                raise RuntimeError(f"Adam {key} shape mismatch for {name}")
            _assert_tensor_finite(tensor, f"Adam {key} {name}")
            state_bytes += tensor.numel() * tensor.element_size()
        state_names.append(name)
    missing_state = sorted(set(gradient_updates) - set(state_names))
    if missing_state:
        raise RuntimeError(f"parameters with gradients lack Adam state: {missing_state[:8]}")
    return {
        "optimizer": "Adam",
        "state_parameter_count": len(state_names),
        "state_tensor_bytes": state_bytes,
        "all_observed_gradient_parameters_have_moments": True,
    }


def run_three_optimizer_updates(
    *,
    model,
    optimizer,
    params,
    cfg: dict,
    stage: str,
    feedback: str,
    builder,
    feedback_stats: dict | None,
    device: torch.device,
    micro_batch: int,
    accumulation: int,
    crop_size: int = 128,
    updates: int = 3,
) -> dict:
    """Run formal Stage-B micro-steps; usable on CPU with a mock model in tests."""
    if stage not in ALLOWED_STAGES or feedback not in ALLOWED_FEEDBACK:
        raise ValueError(f"unsupported Stage-B probe: {stage}/{feedback}")
    if updates != 3:
        raise ValueError("the registered memory probe requires exactly three updates")
    if min(micro_batch, accumulation, crop_size) <= 0:
        raise ValueError("micro_batch, accumulation, and crop_size must be positive")

    model.train()
    model.encoder.eval()
    model.d1.eval()
    optimizer.zero_grad(set_to_none=True)
    named = _named_parameters(model)
    gradient_updates: dict[str, int] = {}
    micro_steps = 0
    for _update_index in range(updates):
        for _micro_index in range(accumulation):
            degraded = torch.rand(
                micro_batch, 3, crop_size, crop_size, device=device
            )
            clean = torch.rand(
                micro_batch, 3, crop_size, crop_size, device=device
            )
            terms, scaled_loss = backward_stage_b_microbatch(
                model,
                degraded,
                clean,
                stage=stage,
                builder=builder,
                feedback=feedback,
                feedback_stats=feedback_stats,
                cfg=cfg,
                accumulation=accumulation,
                amp_dtype=torch.bfloat16,
            )
            for label, value in (
                ("total", terms.total),
                ("rest", terms.rest),
                ("state", terms.state),
                ("clean", terms.clean),
                ("scaled_loss", scaled_loss),
            ):
                _finite_scalar(value, label)
            _assert_tensor_finite(terms.prediction, "training prediction")
            micro_steps += 1
            del degraded, clean, terms, scaled_loss

        current_gradient_names = set()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            _assert_tensor_finite(parameter.grad, f"gradient {name}")
            current_gradient_names.add(name)
            gradient_updates[name] = gradient_updates.get(name, 0) + 1
        if not current_gradient_names:
            raise RuntimeError("optimizer update has no gradients")
        grad_norm = commit_optimizer_update(
            params,
            optimizer,
            gradient_clip=cfg["gradient_clip"],
        )
        _finite_scalar(torch.as_tensor(grad_norm), "clipped gradient norm")

    routing = _validate_gradient_routing(stage, gradient_updates)
    adam = validate_adam_state(optimizer, named, gradient_updates)
    return {
        "optimizer_updates": updates,
        "micro_steps": micro_steps,
        "expected_micro_steps": updates * accumulation,
        "finite_losses_gradients_predictions": True,
        "gradient_routing": routing,
        "adam": adam,
    }


def run_native_shape_probe(
    *,
    model,
    stage: str,
    feedback: str,
    builder,
    feedback_stats: dict | None,
    device: torch.device,
    height: int,
    width: int,
    prediction_fn=tiled_stage_prediction,
) -> dict:
    """Probe the formal tile=0 inference path without quality metrics."""
    if min(height, width) <= 0 or height % 8 or width % 8:
        raise ValueError("native probe shape must be positive and divisible by 8")
    model.eval()
    model.encoder.eval()
    model.d1.eval()
    degraded = torch.rand(1, 3, height, width, device=device)
    clean = torch.rand(1, 3, height, width, device=device)
    device_type = device.type
    with torch.inference_mode(), torch.autocast(
        device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        prediction = prediction_fn(
            model,
            degraded,
            clean,
            stage,
            builder,
            feedback,
            feedback_stats,
            tile=0,
        )
    expected_shape = (1, 3, height, width)
    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"native probe output shape {tuple(prediction.shape)} != {expected_shape}"
        )
    _assert_tensor_finite(prediction, "native full-image prediction")
    del degraded, clean, prediction
    return {
        "input_shape": list(expected_shape),
        "output_shape": list(expected_shape),
        "tile": 0,
        "finite_prediction": True,
        "quality_metrics_computed": False,
    }


def _load_stage_a(
    model,
    cfg: dict,
    checkpoint: Path | None,
    *,
    allow_random_coarse: bool,
) -> str:
    if checkpoint is None:
        if not allow_random_coarse:
            raise ValueError("Stage-A checkpoint is required")
        return "memory_only_random_coarse"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in payload or "split_manifest_sha256" not in payload:
        raise RuntimeError("memory preflight requires a provenance-bearing Stage-A checkpoint")
    if payload["config"].get("protocol") != cfg["protocol"]:
        raise RuntimeError("Stage-A checkpoint protocol mismatch")
    expected_split = _sha256(cfg["split_manifest"])
    if payload["split_manifest_sha256"] != expected_split:
        raise RuntimeError("Stage-A checkpoint locked-split mismatch")
    scope = load_formal_init(model, payload.get("model", payload), "b_predicted")
    return scope


def _cuda_memory_snapshot() -> dict:
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    payload = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes_after_probe": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    payload["headroom_bytes"] = payload["total_bytes"] - payload["peak_reserved_bytes"]
    payload["required_headroom_bytes"] = MIN_HEADROOM_BYTES
    payload["headroom_pass"] = payload["headroom_bytes"] >= MIN_HEADROOM_BYTES
    return payload


def _software_versions() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _static_worker_metadata(args) -> dict:
    """Best-effort immutable provenance, including pre-forward failures."""
    config = Path(args.config).resolve()
    cfg = yaml.safe_load(config.read_text())
    stats = Path(
        args.coordinate_stats_override or cfg["coordinate_stats"]
    ).resolve()
    checkpoint = (
        Path(args.stage_a_checkpoint).resolve()
        if args.stage_a_checkpoint
        else None
    )
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "gpu": gpu,
        "software": _software_versions(),
        "config": str(config),
        "config_sha256": _sha256(config),
        "template_role": args.template_role,
        "init_scope": (
            "memory_only_random_coarse"
            if args.allow_random_coarse_memory_only and checkpoint is None
            else "coarse_only_formal_expected"
        ),
        "stage_a_checkpoint": str(checkpoint) if checkpoint else None,
        "stage_a_checkpoint_sha256": (
            _sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
        ),
        "coordinate_stats_path": str(stats),
        "coordinate_stats_sha256": _sha256(stats) if stats.is_file() else None,
        "coordinate_stats_origin": (
            "MAIN_PROTOCOL_VALUES_MEMORY_SHAPE_ONLY"
            if args.coordinate_stats_override
            else "TEMPLATE_BOUND"
        ),
        "official_test_accessed": False,
        "quality_metrics_computed": False,
    }


def _safe_static_worker_metadata(args) -> dict:
    try:
        return _static_worker_metadata(args)
    except Exception as error:
        return {
            "metadata_error": f"{type(error).__name__}: {error}",
            "software": _software_versions(),
            "official_test_accessed": False,
            "quality_metrics_computed": False,
        }


def _probe_provenance(worker: dict) -> dict:
    metadata = worker.get("worker_metadata", {})
    return {
        "gpu": worker.get("gpu", metadata.get("gpu")),
        "software": worker.get("software", metadata.get("software")),
        "config": worker.get("config", metadata.get("config")),
        "config_sha256": worker.get(
            "config_sha256", metadata.get("config_sha256")
        ),
        "template_role": worker.get(
            "template_role", metadata.get("template_role")
        ),
        "init_scope": worker.get("init_scope", metadata.get("init_scope")),
        "stage_a_checkpoint": worker.get(
            "stage_a_checkpoint", metadata.get("stage_a_checkpoint")
        ),
        "stage_a_checkpoint_sha256": worker.get(
            "stage_a_checkpoint_sha256",
            metadata.get("stage_a_checkpoint_sha256"),
        ),
        "coordinate_stats_path": worker.get(
            "coordinate_stats_path", metadata.get("coordinate_stats_path")
        ),
        "coordinate_stats_sha256": worker.get(
            "coordinate_stats_sha256",
            metadata.get("coordinate_stats_sha256"),
        ),
        "coordinate_stats_origin": worker.get(
            "coordinate_stats_origin",
            metadata.get("coordinate_stats_origin"),
        ),
        "official_test_accessed": worker.get(
            "official_test_accessed",
            metadata.get("official_test_accessed", False),
        ),
        "quality_metrics_computed": worker.get(
            "quality_metrics_computed",
            metadata.get("quality_metrics_computed", False),
        ),
    }


def execute_worker(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a Stage-B memory worker")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("protocol") != args.protocol:
        raise ValueError("worker protocol/config mismatch")
    if cfg.get("precision") != "bf16":
        raise ValueError("worker only supports the audited bf16 path")
    if int(cfg.get("crop_size", -1)) != 128:
        raise ValueError("worker requires the registered 128x128 training crop")
    if args.micro_batch * args.accumulation != int(cfg["effective_batch"]):
        raise ValueError("worker candidate does not preserve effective_batch")
    cfg = dict(cfg)
    cfg["micro_batch"] = args.micro_batch
    cfg["accumulation"] = args.accumulation
    coordinate_stats_original = str(cfg["coordinate_stats"])
    coordinate_stats_origin = "TEMPLATE_BOUND"
    if args.coordinate_stats_override:
        if not args.allow_random_coarse_memory_only:
            raise ValueError(
                "coordinate-stats override is restricted to random-coarse "
                "memory-shape probes"
            )
        override = Path(args.coordinate_stats_override).resolve()
        if not override.is_file():
            raise FileNotFoundError(override)
        cfg["coordinate_stats"] = str(override)
        coordinate_stats_origin = "MAIN_PROTOCOL_VALUES_MEMORY_SHAPE_ONLY"

    seed_all(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", 0)

    model = build_model(cfg, args.stage).to(device)
    configure_feedback_mode(model, args.stage, args.feedback)
    checkpoint = Path(args.stage_a_checkpoint) if args.stage_a_checkpoint else None
    init_scope = _load_stage_a(
        model,
        cfg,
        checkpoint,
        allow_random_coarse=args.allow_random_coarse_memory_only,
    )
    configure_trainable(model, args.stage)
    params, optimizer = build_optimizer(model, args.stage, cfg)
    builder, feedback_stats = build_coordinate_builder(
        cfg,
        allow_unlocked=False,
        expected_stage_a_checkpoint=(
            checkpoint if init_scope != "memory_only_random_coarse" else None
        ),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_result = run_three_optimizer_updates(
        model=model,
        optimizer=optimizer,
        params=params,
        cfg=cfg,
        stage=args.stage,
        feedback=args.feedback,
        builder=builder,
        feedback_stats=feedback_stats,
        device=device,
        micro_batch=args.micro_batch,
        accumulation=args.accumulation,
        crop_size=128,
        updates=3,
    )
    train_result["memory"] = _cuda_memory_snapshot()

    optimizer.zero_grad(set_to_none=True)
    # Keep model, optimizer and its now-materialized Adam moments resident.
    adam_bytes_before_native = train_result["adam"]["state_tensor_bytes"]
    torch.cuda.reset_peak_memory_stats()
    try:
        native_result = run_native_shape_probe(
            model=model,
            stage=args.stage,
            feedback=args.feedback,
            builder=builder,
            feedback_stats=feedback_stats,
            device=device,
            height=args.probe_height,
            width=args.probe_width,
        )
    except torch.cuda.OutOfMemoryError as error:
        return {
            "schema": SCHEMA,
            "status": "OOM",
            "failed_phase": "native_val",
            "error": f"{type(error).__name__}: {error}",
            "protocol": args.protocol,
            "template_role": args.template_role,
            "stage": args.stage,
            "feedback": args.feedback,
            "micro_batch": args.micro_batch,
            "accumulation": args.accumulation,
            "effective_batch": args.micro_batch * args.accumulation,
            "train_step": train_result,
            "worker_metadata": _safe_static_worker_metadata(args),
            "official_test_accessed": False,
            "quality_metrics_computed": False,
        }
    native_result["memory"] = _cuda_memory_snapshot()
    native_result["adam_state_retained_bytes"] = adam_bytes_before_native

    properties = torch.cuda.get_device_properties(0)
    headroom_pass = (
        train_result["memory"]["headroom_pass"]
        and native_result["memory"]["headroom_pass"]
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if headroom_pass else "HEADROOM_FAIL",
        "protocol": args.protocol,
        "template_role": args.template_role,
        "stage": args.stage,
        "feedback": args.feedback,
        "micro_batch": args.micro_batch,
        "accumulation": args.accumulation,
        "effective_batch": args.micro_batch * args.accumulation,
        "init_scope": init_scope,
        "stage_a_checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "stage_a_checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
        "config": str(Path(args.config).resolve()),
        "config_sha256": _sha256(args.config),
        "coordinate_stats_sha256": _sha256(cfg["coordinate_stats"]),
        "coordinate_stats_path": str(Path(cfg["coordinate_stats"]).resolve()),
        "coordinate_stats_override": bool(args.coordinate_stats_override),
        "coordinate_stats_original_path": coordinate_stats_original,
        "coordinate_stats_scope": (
            "memory_only_shape_surrogate"
            if args.coordinate_stats_override
            else "template_native_strict"
        ),
        "coordinate_stats_origin": coordinate_stats_origin,
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        },
        "software": _software_versions(),
        "train_step": train_result,
        "native_val": native_result,
        "optimizer_state_retained_for_native_val": True,
        "official_test_accessed": False,
        "quality_metrics_computed": False,
        "checkpoint_written": False,
        "run_contract_written": False,
    }


def _worker_parser(subparsers) -> None:
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--protocol", choices=tuple(DEFAULT_NATIVE_SHAPES), required=True)
    worker.add_argument("--config", required=True)
    worker.add_argument("--template-role", choices=("main", "capacity_10_10"), required=True)
    worker.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    worker.add_argument("--feedback", choices=ALLOWED_FEEDBACK, required=True)
    worker.add_argument("--stage-a-checkpoint")
    worker.add_argument("--allow-random-coarse-memory-only", action="store_true")
    worker.add_argument("--coordinate-stats-override")
    worker.add_argument("--micro-batch", type=int, required=True)
    worker.add_argument("--accumulation", type=int, required=True)
    worker.add_argument("--probe-height", type=int, required=True)
    worker.add_argument("--probe-width", type=int, required=True)


def _driver_parser(subparsers) -> None:
    driver = subparsers.add_parser("driver")
    driver.add_argument("--protocol", choices=tuple(DEFAULT_NATIVE_SHAPES), required=True)
    driver.add_argument("--root", required=True)
    driver.add_argument("--stage-a-checkpoint", required=True)
    driver.add_argument("--main-template", required=True)
    driver.add_argument("--capacity-template")
    driver.add_argument("--capacity-stage-a-checkpoint")
    driver.add_argument("--candidates-json", required=True)
    driver.add_argument("--output", required=True)
    driver.add_argument("--probe-height", type=int)
    driver.add_argument("--probe-width", type=int)


def parse_args(argv=None):
    # Accept the parent's flat CLI as well as an explicit ``driver`` token.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"driver", "worker"}:
        argv.insert(0, "driver")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _worker_parser(subparsers)
    _driver_parser(subparsers)
    return parser.parse_args(argv)


def _parse_worker_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            return payload
    raise RuntimeError("worker did not emit a schema-bearing JSON record")


def _probe_records(worker: dict, role: str, stage: str, feedback: str) -> list[dict]:
    status = worker.get("status", "ERROR")
    provenance = _probe_provenance(worker)
    train_evidence = worker.get("train_step")
    native_evidence = worker.get("native_val")
    train_memory = train_evidence.get("memory", {}) if isinstance(train_evidence, dict) else {}
    native_memory = native_evidence.get("memory", {}) if isinstance(native_evidence, dict) else {}
    train_pass = (
        isinstance(train_evidence, dict)
        and worker.get("failed_phase") != "train_step"
        and train_memory.get("headroom_pass") is True
    )
    native_pass = (
        isinstance(native_evidence, dict)
        and native_memory.get("headroom_pass") is True
        and worker.get("failed_phase") != "native_val"
    )
    return [
        {
            "probe_id": f"{role}:{stage}:{feedback}:train_step",
            "passed": train_pass,
            "status": "PASS" if train_pass else (
                "HEADROOM_FAIL"
                if isinstance(train_evidence, dict)
                and train_memory.get("headroom_pass") is False
                else status
            ),
            "evidence": train_evidence,
            "worker_status": status,
            "error": worker.get("error"),
            "worker_provenance": provenance,
        },
        {
            "probe_id": f"{role}:{stage}:{feedback}:native_val",
            "passed": native_pass,
            "status": "PASS" if native_pass else (
                "SKIPPED" if not train_pass else status
            ),
            "evidence": native_evidence,
            "worker_status": status,
            "error": worker.get("error"),
            "worker_provenance": provenance,
        },
    ]


def _run_worker(
    *,
    root: Path,
    protocol: str,
    template: Path,
    role: str,
    checkpoint: Path | None,
    allow_random: bool,
    coordinate_stats_override: Path | None,
    stage: str,
    feedback: str,
    micro_batch: int,
    accumulation: int,
    height: int,
    width: int,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--protocol",
        protocol,
        "--config",
        str(template),
        "--template-role",
        role,
        "--stage",
        stage,
        "--feedback",
        feedback,
        "--micro-batch",
        str(micro_batch),
        "--accumulation",
        str(accumulation),
        "--probe-height",
        str(height),
        "--probe-width",
        str(width),
    ]
    if checkpoint is not None:
        command += ["--stage-a-checkpoint", str(checkpoint)]
    if allow_random:
        command.append("--allow-random-coarse-memory-only")
    if coordinate_stats_override is not None:
        command += [
            "--coordinate-stats-override", str(coordinate_stats_override.resolve())
        ]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = _parse_worker_json(completed.stdout)
    except RuntimeError:
        payload = {
            "schema": SCHEMA,
            "status": "ERROR",
            "error": (completed.stderr or completed.stdout)[-4000:],
        }
    if completed.returncode != 0 and payload.get("status") not in {
        "OOM",
        "HEADROOM_FAIL",
    }:
        payload = {
            "schema": SCHEMA,
            "status": "ERROR",
            "error": (
                f"worker returncode={completed.returncode}; "
                + str(payload.get("error") or completed.stderr or completed.stdout)[-4000:]
            ),
        }
    payload["returncode"] = completed.returncode
    return payload


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def execute_driver(args) -> tuple[dict, bool]:
    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise ValueError(f"--root {root} does not own this worker at {ROOT}")
    candidates_raw = json.loads(args.candidates_json)
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates-json must be a non-empty list")
    candidates = []
    for pair in candidates_raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("every candidate must be [micro_batch, accumulation]")
        micro_batch, accumulation = map(int, pair)
        if min(micro_batch, accumulation) <= 0:
            raise ValueError("candidate values must be positive")
        candidates.append((micro_batch, accumulation))

    main = Path(args.main_template).resolve()
    main_cfg = yaml.safe_load(main.read_text())
    main_stats = Path(main_cfg["coordinate_stats"]).resolve()
    templates = [
        ("main", main, Path(args.stage_a_checkpoint).resolve(), False, None)
    ]
    if args.capacity_template:
        capacity_checkpoint = (
            Path(args.capacity_stage_a_checkpoint).resolve()
            if args.capacity_stage_a_checkpoint
            else None
        )
        templates.append(
            (
                "capacity_10_10",
                Path(args.capacity_template).resolve(),
                capacity_checkpoint,
                capacity_checkpoint is None,
                main_stats if capacity_checkpoint is None else None,
            )
        )
    if (args.probe_height is None) != (args.probe_width is None):
        raise ValueError("probe-height and probe-width must be supplied together")
    if args.probe_height is None:
        shape_evidence = largest_locked_val_shape(main)
        height = shape_evidence["height"]
        width = shape_evidence["width"]
    else:
        height = int(args.probe_height)
        width = int(args.probe_width)
        shape_evidence = {
            "height": height,
            "width": width,
            "source": "explicit_driver_override",
            "shape_semantics": "height_width",
            "quality_metrics_computed": False,
            "official_test_accessed": False,
        }
    attempts = []
    selected_candidate = None
    selected_runtime = None
    fatal_error = None
    started = time.monotonic()
    for micro_batch, accumulation in candidates:
        probes = []
        effective_batch = micro_batch * accumulation
        for (
            role, template, checkpoint, allow_random, coordinate_stats_override
        ) in templates:
            cfg = yaml.safe_load(template.read_text())
            if cfg.get("protocol") != args.protocol:
                raise ValueError(f"template protocol mismatch: {template}")
            if effective_batch != int(cfg["effective_batch"]):
                raise ValueError(
                    f"candidate {micro_batch}x{accumulation} does not preserve "
                    f"{template} effective_batch={cfg['effective_batch']}"
                )
            for stage in ALLOWED_STAGES:
                for feedback in ALLOWED_FEEDBACK:
                    worker = _run_worker(
                        root=root,
                        protocol=args.protocol,
                        template=template,
                        role=role,
                        checkpoint=checkpoint,
                        allow_random=allow_random,
                        coordinate_stats_override=coordinate_stats_override,
                        stage=stage,
                        feedback=feedback,
                        micro_batch=micro_batch,
                        accumulation=accumulation,
                        height=height,
                        width=width,
                    )
                    probes.extend(_probe_records(worker, role, stage, feedback))
                    status = worker.get("status", "ERROR")
                    if status not in {"PASS", "OOM", "HEADROOM_FAIL"}:
                        fatal_error = {
                            "template_role": role,
                            "stage": stage,
                            "feedback": feedback,
                            "status": status,
                            "returncode": worker.get("returncode"),
                            "error": worker.get("error"),
                        }
                        break
                if fatal_error is not None:
                    break
            if fatal_error is not None:
                break
        all_pass = all(probe["passed"] for probe in probes)
        attempt = {
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "effective_batch": effective_batch,
            "all_pass": all_pass,
            "probes": probes,
        }
        attempts.append(attempt)
        if fatal_error is not None:
            break
        if all_pass:
            selected_candidate = [micro_batch, accumulation]
            selected_runtime = {
                "micro_batch": micro_batch,
                "accumulation": accumulation,
                "effective_batch": effective_batch,
            }
            break

    payload = {
        "schema": SCHEMA,
        "protocol": args.protocol,
        "candidate_order": [[micro, accum] for micro, accum in candidates],
        "attempts": attempts,
        "selected_candidate": selected_candidate,
        "selected": selected_runtime,
        "all_pass": selected_candidate is not None and fatal_error is None,
        "fatal_error": fatal_error,
        "native_shape": [height, width],
        "native_shape_semantics": "height_width",
        "native_shape_evidence": shape_evidence,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "MEMORY_ONLY",
        "scientific_authority": "NONE",
        "official_test_accessed": False,
        "quality_metrics_computed": False,
    }
    _atomic_json(Path(args.output), payload)
    return payload, payload["all_pass"]


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        failed_phase = "train_step"
        try:
            payload = execute_worker(args)
        except torch.cuda.OutOfMemoryError as error:
            payload = {
                "schema": SCHEMA,
                "status": "OOM",
                "failed_phase": failed_phase,
                "error": f"{type(error).__name__}: {error}",
                "worker_metadata": _safe_static_worker_metadata(args),
                "official_test_accessed": False,
                "quality_metrics_computed": False,
            }
        except Exception as error:  # worker failure is serialized for the driver
            payload = {
                "schema": SCHEMA,
                "status": "ERROR",
                "failed_phase": failed_phase,
                "error": f"{type(error).__name__}: {error}",
                "worker_metadata": _safe_static_worker_metadata(args),
                "official_test_accessed": False,
                "quality_metrics_computed": False,
            }
        print(json.dumps(payload, sort_keys=True), flush=True)
        if payload["status"] != "PASS":
            raise SystemExit(2)
        return
    _payload, passed = execute_driver(args)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
