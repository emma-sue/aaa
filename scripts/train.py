#!/usr/bin/env python3
"""Resumable stage-aware trainer for Restormer-AiO/SRSC-Lite."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import AIOTrainDataset, build_locked_val
from src.losses import state_loss
from src.net import (
    DETERMINISTIC_FEEDBACK_MODES,
    PREDICTED_FEEDBACK_MODES,
    CleanRestormerAiO,
    SRSCCoordinateBuilder,
    SRSCLite,
    corrupt_direction_control,
    fixed_random_state_like,
    predicted_supervision_mode,
)
from src.net.restormer_blocks import pad_to_multiple
from scripts.runtime_accounting import (
    atomic_write_runtime_sidecar,
    read_runtime_sidecar,
    runtime_snapshot,
    start_runtime_accounting,
)
from scripts.stage_b_runtime import (
    assert_no_runtime_worker_override,
    assert_stage_b_cublas_environment,
    runtime_identity_for_config,
)


EXPECTED_VALIDATION_TASKS = {
    "aio3": ("dehaze", "derain", "denoise15", "denoise25", "denoise50"),
    "aio5": ("dehaze", "derain", "denoise25", "deblur", "lowlight"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=[
            "baseline", "baseline_matched", "baseline_ft", "baseline_matched_ft",
            "a", "b_oracle", "b_predicted", "c",
        ],
        required=True,
    )
    parser.add_argument(
        "--feedback", default="O7",
        choices=["O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13", "O14", "O15"],
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--workers-override",
        type=int,
        help="runtime-only DataLoader worker count; recorded in checkpoint args",
    )
    parser.add_argument(
        "--seed-override", type=int,
        help="explicit preregistered repeat seed; stored in checkpoint config and args",
    )
    parser.add_argument("--allow-incomplete-data", action="store_true", help="smoke only; never publication training")
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_manifest_binding(protocol: str) -> dict:
    """Bind the small prepared-data ledger without scanning training images."""
    path = ROOT / "artifacts/manifests" / f"{protocol}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    list_sha = payload.get("list_sha256")
    if (
        payload.get("protocol") != protocol
        or not isinstance(list_sha, dict)
        or not list_sha
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in list_sha.values()
        )
    ):
        raise RuntimeError(f"invalid prepared-data manifest binding: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "list_sha256": dict(sorted(list_sha.items())),
    }


def update_replay_digest(digest, batch: dict) -> int:
    """Bind the exact ordered raw sample identities consumed by one batch."""
    indices = batch.get("sample_index")
    keys = batch.get("sample_key")
    if indices is None or keys is None:
        raise RuntimeError("training batch lacks replay sample identity")
    index_values = indices.tolist() if torch.is_tensor(indices) else list(indices)
    key_values = list(keys)
    if len(index_values) != len(key_values):
        raise RuntimeError("replay sample index/key width mismatch")
    for index, key in zip(index_values, key_values):
        digest.update(f"{int(index)}:{key}\n".encode("ascii"))
    return len(index_values)


def commit_epoch_replay_digest(
    run_name: str,
    epoch: int,
    step: int,
    sample_count: int,
    digest_hex: str,
    cfg: dict,
) -> Path:
    """Atomically commit one complete epoch's ordered sample-identity digest."""
    directory = ROOT / "artifacts/manifests/replay_digests" / run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch{epoch:03d}.json"
    payload = {
        "schema": 1,
        "run_name": run_name,
        "epoch": int(epoch),
        "optimizer_step_end": int(step),
        "sample_count": int(sample_count),
        "ordered_sample_identity_sha256": digest_hex,
        "protocol": cfg["protocol"],
        "seed": int(cfg["seed"]),
        "split_manifest_sha256": sha256_file(cfg["split_manifest"]),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError(
                f"epoch replay digest drift for {run_name} epoch {epoch}"
            )
        return path
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(serialized)
    os.replace(temporary, path)
    return path


def validate_resume_contract(payload: dict, cfg: dict, args):
    expected_config = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    expected_split = hashlib.sha256(Path(cfg["split_manifest"]).read_bytes()).hexdigest()
    if payload.get("config_sha256") != expected_config:
        raise RuntimeError("resume config hash mismatch")
    if payload.get("split_manifest_sha256") != expected_split:
        raise RuntimeError("resume split-manifest hash mismatch")
    if payload.get("config") != cfg:
        raise RuntimeError("resume effective config/seed mismatch")
    saved = payload.get("args", {})
    for key in (
        "stage", "feedback", "run_name", "max_steps", "seed_override",
        "workers_override", "allow_incomplete_data",
    ):
        if saved.get(key) != getattr(args, key):
            raise RuntimeError(
                f"resume argument mismatch for {key}: saved={saved.get(key)!r} "
                f"current={getattr(args, key)!r}"
            )
    source_path = saved.get("source_init_path") or saved.get("init")
    source_sha = saved.get("source_init_sha256")
    if source_path and source_sha:
        if not Path(source_path).is_file():
            raise RuntimeError(f"resume source-init checkpoint missing: {source_path}")
        if sha256_file(source_path) != source_sha:
            raise RuntimeError("resume source-init checkpoint hash mismatch")
    # Preserve permanent Stage-A/source provenance in every later checkpoint;
    # a resume command intentionally supplies --resume instead of --init.
    args.init = saved.get("init")
    args.source_init_path = source_path
    args.source_init_sha256 = source_sha


def ensure_run_contract(run_dir: Path, cfg: dict, args):
    code_paths = (
        Path(__file__),
        ROOT / "scripts/stage_b_runtime.py",
        ROOT / "src/net/feedback_controls.py",
        ROOT / "src/net/srsc_lite.py",
        ROOT / "src/net/srsc_coordinates.py",
        ROOT / "src/data/aio_dataset.py",
        ROOT / "src/losses/objectives.py",
    )
    runtime_identity = runtime_identity_for_config(Path(args.config), cfg)
    if runtime_identity.get("stage_b_runtime_role") == "main":
        runtime_manifest = json.loads(
            Path(runtime_identity["stage_b_runtime_manifest_path"]).read_text()
        )
        bound_stage_a = runtime_manifest.get("bindings", {}).get(
            "stage_a_checkpoint", {}
        )
        if (
            args.source_init_path != bound_stage_a.get("path")
            or args.source_init_sha256 != bound_stage_a.get("sha256")
        ):
            raise RuntimeError(
                "main Stage-B source checkpoint differs from the frozen "
                "runtime preflight binding"
            )
    contract = {
        "schema": 1,
        "run_name": args.run_name,
        "stage": args.stage,
        "feedback": args.feedback,
        "max_steps": args.max_steps,
        "seed_override": args.seed_override,
        "workers_override": args.workers_override,
        "allow_incomplete_data": args.allow_incomplete_data,
        "effective_config": cfg,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(
            Path(cfg["split_manifest"]).read_bytes()
        ).hexdigest(),
        "data_manifest_binding": data_manifest_binding(cfg["protocol"]),
        "source_init_path": args.source_init_path,
        "source_init_sha256": args.source_init_sha256,
        "coordinate_stats_sha256": (
            sha256_file(cfg["coordinate_stats"])
            if args.stage in {"b_oracle", "b_predicted", "c"}
            and cfg.get("coordinate_stats")
            and Path(cfg["coordinate_stats"]).is_file()
            else None
        ),
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    contract.update(runtime_identity)
    path = run_dir / "run_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise RuntimeError(
                "immutable run contract mismatch; use a new run name or archive the old run"
            )
    else:
        preexisting = [candidate.name for candidate in run_dir.glob("*.pt")]
        if preexisting:
            raise RuntimeError(
                "run directory contains checkpoints but no run_contract.json: "
                f"{sorted(preexisting)}"
            )
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    args.run_contract_sha256 = sha256_file(path)
    return contract


def acquire_run_lock(run_dir: Path, run_name: str) -> int:
    """Prevent two trainers from writing the same scientific run.

    The persistent lock file is harmless; ownership is the kernel flock held
    by this open descriptor and is released automatically on process exit.
    This also protects against an orchestrator being killed while a child
    trainer survives and a replacement orchestrator attempts the same run.
    """
    path = run_dir / ".train.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.lseek(fd, 0, os.SEEK_SET)
        owner = os.read(fd, 128).decode(errors="replace").strip() or "unknown"
        os.close(fd)
        raise RuntimeError(
            f"training run is already active: {run_name} (lock pid={owner})"
        ) from exc
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    os.fsync(fd)
    return fd


def model_kwargs(cfg):
    m = cfg["model"]
    return dict(
        dim=m["dim"],
        encoder_blocks=tuple(m["encoder_blocks"]),
        d1_blocks=tuple(m["d1_blocks"]),
        d2_blocks=tuple(m["d2_blocks"]),
        d2_refinement=m["d2_refinement"],
        heads=tuple(m["heads"]),
        expansion=m["expansion"],
    )


def build_model(cfg, stage):
    if stage in {"baseline", "baseline_matched", "baseline_ft", "baseline_matched_ft"}:
        m = cfg["model"]
        return CleanRestormerAiO(
            dim=m["matched_dim"] if stage in {"baseline_matched", "baseline_matched_ft"} else m["dim"],
            encoder_blocks=tuple(m["encoder_blocks"]),
            heads=tuple(m["heads"]),
            expansion=m["expansion"],
        )
    return SRSCLite(**model_kwargs(cfg))


def load_formal_init(model, state: dict[str, torch.Tensor], stage: str) -> str:
    """Load exactly the state that has scientific authority for this stage.

    Stage-A trains only encoder+D1.  Copying its untouched random assessor/D2
    into every Stage-B repeat would silently make repeat seeds share the same
    initialization.  Stage-B therefore imports the complete coarse path and
    retains the freshly seeded feedback/correction path.  Joint and baseline
    fine-tuning checkpoints are fully authoritative and remain strict loads.
    """
    if stage not in {"b_oracle", "b_predicted"}:
        model.load_state_dict(state, strict=True)
        return "full"

    current = model.state_dict()
    prefixes = ("encoder.", "d1.")
    expected = {key for key in current if key.startswith(prefixes)}
    coarse = {key: value for key, value in state.items() if key.startswith(prefixes)}
    if set(coarse) != expected:
        missing = sorted(expected - set(coarse))
        unexpected = sorted(set(coarse) - expected)
        raise RuntimeError(
            f"formal Stage-B coarse init mismatch: missing={missing} unexpected={unexpected}"
        )
    mismatched = [
        key for key in expected if tuple(coarse[key].shape) != tuple(current[key].shape)
    ]
    if mismatched:
        raise RuntimeError(f"formal Stage-B coarse init shape mismatch: {sorted(mismatched)}")
    incompatible = model.load_state_dict(coarse, strict=False)
    allowed_missing = set(current) - expected
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "formal Stage-B partial-load contract failed: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    return "coarse_only_fresh_seeded_feedback_path"


def validate_coordinate_statistics(
    cfg: dict, payload: dict, expected_stage_a_checkpoint: str | Path | None = None
):
    if payload["protocol"] != cfg["protocol"]:
        raise ValueError("coordinate statistics protocol mismatch")
    expected_split = hashlib.sha256(Path(cfg["split_manifest"]).read_bytes()).hexdigest()
    if payload.get("split_manifest_sha256") != expected_split:
        raise RuntimeError("coordinate statistics locked-split mismatch")
    if expected_stage_a_checkpoint is not None:
        path = Path(expected_stage_a_checkpoint)
        actual_checkpoint = hashlib.sha256(path.read_bytes()).hexdigest()
        if payload.get("stage_a_checkpoint_sha256") != actual_checkpoint:
            raise RuntimeError("coordinate statistics Stage-A checkpoint mismatch")


def build_coordinate_builder(
    cfg, allow_unlocked=False, expected_stage_a_checkpoint: str | Path | None = None
):
    path = Path(cfg["coordinate_stats"])
    if not path.is_file():
        if allow_unlocked:
            print(f"WARNING smoke-only unlocked coordinate thresholds: {path}", flush=True)
            return SRSCCoordinateBuilder().cuda(), None
        raise FileNotFoundError(
            f"coordinate statistics are not locked: {path}; run "
            "scripts/compute_coordinate_stats.py from the frozen Stage-A checkpoint"
        )
    payload = json.loads(path.read_text())
    validate_coordinate_statistics(cfg, payload, expected_stage_a_checkpoint)
    return SRSCCoordinateBuilder(
        tau_v=payload["tau_v"],
        tau_e=payload["tau_e"],
        pca_projection=payload.get("pca_direction_matrix"),
        pca_mean=payload.get("pca_direction_mean"),
    ).cuda(), payload


def configure_trainable(model, stage):
    for p in model.parameters():
        p.requires_grad = True
    if stage == "a":
        for module in (model.assessor, model.y1_pyramid, model.d2, model.oracle_ceiling_adapter):
            for p in module.parameters():
                p.requires_grad = False
    elif stage in {"b_oracle", "b_predicted"}:
        for module in (model.encoder, model.d1):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        if stage == "b_oracle":
            for p in model.assessor.parameters():
                p.requires_grad = False


def feedback_from_coordinates(outputs, mode, ceiling_adapter=None):
    states = []
    for scale_index, out in enumerate(outputs):
        zero = torch.zeros_like(out.state)
        if mode == "O0":
            state = zero
        elif mode == "O1":
            state = out.y1_code
        elif mode == "O2":
            state = out.edit_code
        elif mode == "O3":
            state = zero.clone(); state[:, :1] = out.error_magnitude
        elif mode == "O4":
            state = zero.clone()
            state[:, :1] = F.relu(out.p_raw)
            state[:, 1:2] = out.m_raw + F.relu(-out.p_raw)
        elif mode == "O5":
            state = zero.clone(); state[:, :1] = out.state[:, :1]
        elif mode == "O6":
            state = zero.clone(); state[:, :2] = out.state[:, :2]
        elif mode == "O7":
            state = out.state
        elif mode == "O8":
            state = out.state.clone(); state[:, :1] = state[:, :1].abs()
        elif mode == "O9":
            state = out.state.clone()
            # A cyclic derangement cannot accidentally sample identity;
            # batch-one variable-size validation uses spatial displacement.
            state[:, 2:] = corrupt_direction_control(out.state[:, 2:])
        elif mode == "O10":
            state = out.state.clone(); state[:, 2:] = 0
        elif mode == "O11":
            # Fixed local RNG: equal-dimensional noise must be repeatable and
            # must not perturb the global training/validation RNG stream.
            state = fixed_random_state_like(out.state, scale_index)
        elif mode == "O12":
            state = out.residual_code
        elif mode == "O13":
            state = out.transverse_code
        elif mode == "O14":
            if ceiling_adapter is None:
                raise ValueError("O14 requires the shared trainable 81->8 ceiling adapter")
            state = ceiling_adapter(out.correction)
        elif mode == "O15":
            if out.state_pca is None:
                raise ValueError("O15 requires a train-only fitted PCA projection")
            state = out.state_pca
        else:
            raise ValueError(mode)
        states.append(state)
    return states


def direction_weights_from_coordinates(outputs, mode):
    weights = [out.w_dir for out in outputs]
    if mode == "O9":
        weights = [corrupt_direction_control(weight) for weight in weights]
    return weights


def direction_valid_masks(raw_targets, direction_weights):
    """Build the preregistered validity mask in unnormalized SRSC units."""
    if len(raw_targets) != len(direction_weights):
        raise ValueError("direction target/weight scale count mismatch")
    masks = []
    for target, weight in zip(raw_targets, direction_weights):
        raw_direction_norm = target[:, 2:8].float().norm(dim=1, keepdim=True)
        masks.append(
            ((raw_direction_norm >= 1e-6) & (weight.detach().float() >= 1e-3)).detach()
        )
    return masks


def normalize_feedback(states, mode, statistics):
    if mode == "O0" or statistics is None:
        return states
    if mode == "O11":
        # Pre-registered standard random noise is already unit-scaled.
        return states
    if mode == "O14":
        return states
    reference_mode = "O7" if mode in {"O8", "O9", "O10"} else mode
    spec = statistics["normalization"][reference_mode]
    normalized = []
    for state in states:
        center = state.new_tensor(spec["center"]).view(1, 8, 1, 1)
        scale = state.new_tensor(spec["scale"]).view(1, 8, 1, 1)
        normalized.append((state - center) / scale.clamp_min(1e-4))
    return normalized


def warmup_cosine(step, total, warmup, min_ratio):
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def r2r_pretrain_epoch_ratio(epoch, lr, warmup_epochs, schedule_epochs, warmup_start_lr, eta_min):
    """R2R's epoch-wise LinearWarmupCosineAnnealingLR in ratio form."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        value = warmup_start_lr + epoch * (lr - warmup_start_lr) / max(warmup_epochs - 1, 1)
    else:
        progress = (epoch - warmup_epochs) / max(schedule_epochs - warmup_epochs, 1)
        value = eta_min + 0.5 * (lr - eta_min) * (1.0 + math.cos(math.pi * progress))
    return value / lr


def restoration_l1(prediction, target):
    """PromptIR's registered full-RGB training objective."""
    return F.l1_loss(prediction, target)


def optimizer_groups(model, stage, lr):
    """Use the preregistered 0.1x E/D1 learning rate in joint Stage-C."""
    if stage != "c":
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    slow_ids = {id(p) for module in (model.encoder, model.d1) for p in module.parameters() if p.requires_grad}
    slow = [p for p in model.parameters() if p.requires_grad and id(p) in slow_ids]
    main = [p for p in model.parameters() if p.requires_grad and id(p) not in slow_ids]
    if not slow or not main:
        raise RuntimeError("Stage-C optimizer groups are empty")
    return [{"params": slow, "lr": lr * 0.1}, {"params": main, "lr": lr}]


def build_optimizer(model, stage: str, cfg: dict):
    """Build the optimizer used by both formal training and memory preflight.

    Adam creates its moment tensors lazily on the first ``step``.  Keeping this
    construction in one helper prevents a memory probe from underestimating the
    formal path by accidentally using different parameter groups or optimizer
    settings.
    """
    groups = optimizer_groups(model, stage, cfg["lr"])
    params = [parameter for group in groups for parameter in group["params"]]
    optimizer_name = cfg.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            groups, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            groups,
            lr=cfg["lr"],
            betas=(0.9, 0.999),
            weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    return params, optimizer


def configure_feedback_mode(model, stage: str, feedback: str) -> None:
    """Configure the common predicted interface without duplicating trainers."""
    if not isinstance(model, SRSCLite):
        return
    if stage == "b_predicted":
        if feedback not in PREDICTED_FEEDBACK_MODES:
            raise ValueError(
                f"feedback {feedback} is oracle-only and cannot enter {stage}"
            )
        model.predicted_feedback_mode = feedback
        # Retained only for compatibility with old O0 checkpoints/tests.  The
        # common interface itself already hard-zeros O0.
        model.force_zero_state = feedback == "O0"
    elif stage == "c":
        if feedback in DETERMINISTIC_FEEDBACK_MODES:
            model.predicted_feedback_mode = None
        elif feedback in PREDICTED_FEEDBACK_MODES:
            model.predicted_feedback_mode = feedback
            model.force_zero_state = feedback == "O0"
        else:
            raise ValueError(f"feedback {feedback} is not deployable in Stage-C")


@dataclass
class StageBTerms:
    """Unscaled formal Stage-B objective components for one micro-batch."""

    total: torch.Tensor
    rest: torch.Tensor
    state: torch.Tensor
    clean: torch.Tensor
    prediction: torch.Tensor


def compute_stage_b_terms(
    model,
    x: torch.Tensor,
    gt: torch.Tensor,
    *,
    stage: str,
    builder,
    feedback: str,
    feedback_stats: dict | None,
    cfg: dict,
) -> StageBTerms:
    """Execute the exact Oracle/Predicted Stage-B forward and objective.

    The caller owns autocast and backward.  Both formal training and the
    no-metric memory preflight call this function, so the probe cannot replace
    the expensive 81-D coordinate construction or state supervision with a
    synthetic approximation.
    """
    if stage not in {"b_oracle", "b_predicted"}:
        raise ValueError(f"Stage-B terms do not support stage={stage!r}")
    state_term = x.new_zeros(())
    clean_term = x.new_zeros(())
    if stage == "b_oracle":
        with torch.no_grad():
            features, y1 = model._encode_coarse(x)
            # Capacity/compute-matched dummy assessor path.  Its output is
            # deliberately discarded at the common eight-channel interface.
            _ = model.assessor(x, y1, features)
            coords = builder(
                x,
                y1,
                gt,
                [feature.shape[-2:] for feature in features],
                requested={feedback},
            )
        states = normalize_feedback(
            feedback_from_coordinates(
                coords, feedback, model.oracle_ceiling_adapter
            ),
            feedback,
            feedback_stats,
        )
        prediction = model._run_d2(x, y1, features, states)
        rest = restoration_l1(prediction, gt)
        clean_term = ((1.0 - coords[0].q) * (prediction - y1).abs()).mean()
    else:
        details = model.forward_details(x)
        prediction = details.y2
        target_mode = (
            feedback
            if feedback in DETERMINISTIC_FEEDBACK_MODES
            else predicted_supervision_mode(feedback)
        )
        coords = builder(
            x,
            details.y1,
            gt,
            [feature.shape[-2:] for feature in details.features],
            requested={target_mode},
        )
        raw_targets = feedback_from_coordinates(
            coords, target_mode, model.oracle_ceiling_adapter
        )
        targets = normalize_feedback(raw_targets, target_mode, feedback_stats)
        direction_weight = 0.1 if target_mode in {"O7", "O8", "O15"} else 0.0
        coordinate_direction_weights = (
            direction_weights_from_coordinates(coords, target_mode)
            if direction_weight
            else None
        )
        state_term, _ = state_loss(
            details.states,
            targets,
            direction_cosine_weight=direction_weight,
            direction_weights=coordinate_direction_weights,
            direction_valid_masks=(
                direction_valid_masks(raw_targets, coordinate_direction_weights)
                if direction_weight
                else None
            ),
        )
        rest = 0.5 * restoration_l1(details.y1, gt) + restoration_l1(
            prediction, gt
        )
        clean_term = (
            (1.0 - coords[0].q) * (prediction - details.y1).abs()
        ).mean()
    total = (
        rest
        + cfg.get("lambda_state", 0.1) * state_term
        + cfg.get("lambda_clean", 0.1) * clean_term
    )
    return StageBTerms(total, rest, state_term, clean_term, prediction)


def backward_stage_b_microbatch(
    model,
    x: torch.Tensor,
    gt: torch.Tensor,
    *,
    stage: str,
    builder,
    feedback: str,
    feedback_stats: dict | None,
    cfg: dict,
    accumulation: int,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[StageBTerms, torch.Tensor]:
    """Run the shared BF16 Stage-B forward, scaled loss, and backward."""
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    device_type = x.device.type
    with torch.autocast(
        device_type,
        dtype=amp_dtype,
        enabled=device_type == "cuda",
    ):
        terms = compute_stage_b_terms(
            model,
            x,
            gt,
            stage=stage,
            builder=builder,
            feedback=feedback,
            feedback_stats=feedback_stats,
            cfg=cfg,
        )
        scaled_loss = terms.total / accumulation
    scaled_loss.backward()
    return terms, scaled_loss


def commit_optimizer_update(
    params,
    optimizer,
    *,
    gradient_clip: float,
    scheduler=None,
    scheduler_unit: str | None = None,
) -> torch.Tensor:
    """Commit the exact formal clip/optimizer/optional step-scheduler path."""
    grad_norm = torch.nn.utils.clip_grad_norm_(params, gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler_unit == "step":
        if scheduler is None:
            raise ValueError("step scheduler unit requires a scheduler")
        scheduler.step()
    return grad_norm


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, batch_in_epoch, step, cfg, args,
    validation_pending: str | None = None,
):
    tmp = path.with_suffix(path.suffix + ".tmp")
    accounting = runtime_snapshot()
    run_contract_path = path.parent / "run_contract.json"
    if not run_contract_path.is_file():
        raise RuntimeError(
            f"refusing checkpoint without immutable run contract: {run_contract_path}"
        )
    run_contract_sha256 = sha256_file(run_contract_path)
    if getattr(args, "run_contract_sha256", None) != run_contract_sha256:
        raise RuntimeError("checkpoint/run-contract SHA256 carrier mismatch")
    run_contract = json.loads(run_contract_path.read_text())
    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    split_manifest_sha256 = hashlib.sha256(
        Path(cfg["split_manifest"]).read_bytes()
    ).hexdigest()
    code_contract = run_contract.get("code_sha256")
    if not isinstance(code_contract, dict) or not code_contract:
        raise RuntimeError("run contract has no checkpoint-bindable code closure")
    data_contract = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": config_sha256,
        "split_manifest_path": str(Path(cfg["split_manifest"]).resolve()),
        "split_manifest_sha256": split_manifest_sha256,
        "coordinate_stats_path": (
            str(Path(cfg["coordinate_stats"]).resolve())
            if cfg.get("coordinate_stats") else None
        ),
        "coordinate_stats_sha256": run_contract.get(
            "coordinate_stats_sha256"
        ),
        "source_init_path": run_contract.get("source_init_path"),
        "source_init_sha256": run_contract.get("source_init_sha256"),
    }
    runtime_contract = {
        "run_contract_path": str(run_contract_path.resolve()),
        "run_contract_sha256": run_contract_sha256,
        "deterministic_algorithms": run_contract.get(
            "deterministic_algorithms"
        ),
        "cublas_workspace_config": run_contract.get(
            "cublas_workspace_config"
        ),
        "stage_b_runtime_manifest_path": run_contract.get(
            "stage_b_runtime_manifest_path"
        ),
        "stage_b_runtime_manifest_sha256": run_contract.get(
            "stage_b_runtime_manifest_sha256"
        ),
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "step": step,
            "validation_pending": validation_pending,
            "config": cfg,
            "config_sha256": config_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "args": vars(args),
            "runtime_accounting": accounting,
            "runtime_contract": runtime_contract,
            "data_contract": data_contract,
            "code_contract": code_contract,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        tmp,
    )
    tmp.replace(path)
    atomic_write_runtime_sidecar(path.parent / "runtime_accounting.json", accounting)


def stage_prediction(model, x, gt, stage, builder, feedback, feedback_stats):
    if stage in {"baseline", "baseline_matched", "baseline_ft", "baseline_matched_ft"}:
        return model(x)
    if stage == "a":
        padded, original = pad_to_multiple(x, 8)
        features, y1 = model._encode_coarse(padded)
        return y1[..., : original[0], : original[1]]
    if stage == "b_oracle":
        features, y1 = model._encode_coarse(x)
        # Capacity/compute-matched dummy assessor path for every oracle arm.
        # Its output is intentionally discarded at the common interface.
        _ = model.assessor(x, y1, features)
        coords = builder(x, y1, gt, [f.shape[-2:] for f in features], requested={feedback})
        states = normalize_feedback(
            feedback_from_coordinates(coords, feedback, model.oracle_ceiling_adapter),
            feedback, feedback_stats,
        )
        return model._run_d2(x, y1, features, states)
    return model(x)


def tiled_stage_prediction(model, x, gt, stage, builder, feedback, feedback_stats, tile=0, overlap=32):
    _, _, h, w = x.shape
    if tile <= 0 or max(h, w) <= tile:
        return stage_prediction(model, x, gt, stage, builder, feedback, feedback_stats)
    tile = min(tile, h, w)
    stride = tile - overlap
    hs = list(range(0, max(h - tile, 0), stride)) + [max(h - tile, 0)]
    ws = list(range(0, max(w - tile, 0), stride)) + [max(w - tile, 0)]
    output = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for top in sorted(set(hs)):
        for left in sorted(set(ws)):
            xp = x[..., top : top + tile, left : left + tile]
            gp = gt[..., top : top + tile, left : left + tile]
            yp = stage_prediction(model, xp, gp, stage, builder, feedback, feedback_stats)
            output[..., top : top + yp.shape[-2], left : left + yp.shape[-1]] += yp
            weight[..., top : top + yp.shape[-2], left : left + yp.shape[-1]] += 1
    return output / weight.clamp_min(1)


def full_rgb_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the paper-protocol full-image RGB SSIM for one paired image."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"SSIM expects matching BCHW tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[0] != 1 or prediction.shape[1] != 3:
        raise ValueError(
            f"SSIM expects one RGB image, got {tuple(prediction.shape)}"
        )
    prediction_rgb = (
        prediction.detach().float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
    target_rgb = target.detach().float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    value = float(
        structural_similarity(
            target_rgb, prediction_rgb, channel_axis=2, data_range=1.0
        )
    )
    if not math.isfinite(value):
        raise FloatingPointError("locked-validation SSIM is non-finite")
    return value


def summarize_locked_metrics(
    per_task_psnr: dict[str, list[float]],
    per_task_ssim: dict[str, list[float]],
    ordered_tasks: tuple[str, ...],
) -> dict:
    """Keep legacy PSNR task keys while adding a separate SSIM namespace."""
    if set(per_task_psnr) != set(ordered_tasks) or set(per_task_ssim) != set(
        ordered_tasks
    ):
        raise RuntimeError("locked metric task sets are inconsistent")
    setting_psnr: dict[str, float] = {}
    setting_ssim: dict[str, float] = {}
    for task in ordered_tasks:
        psnr_values = np.asarray(per_task_psnr[task], dtype=np.float64)
        ssim_values = np.asarray(per_task_ssim[task], dtype=np.float64)
        if (
            psnr_values.size == 0
            or psnr_values.shape != ssim_values.shape
            or not np.isfinite(psnr_values).all()
            or not np.isfinite(ssim_values).all()
        ):
            raise RuntimeError(f"invalid locked metrics for setting {task}")
        setting_psnr[task] = float(psnr_values.mean())
        setting_ssim[task] = float(ssim_values.mean())
    summary: dict = dict(setting_psnr)
    # ``macro_psnr`` remains the historical five-setting PSNR alias consumed
    # by checkpoint selection and all existing scientific gates.
    summary["macro_psnr"] = float(np.mean(list(setting_psnr.values())))
    summary["setting_ssim"] = setting_ssim
    summary["five_setting_mean_ssim"] = float(np.mean(list(setting_ssim.values())))
    return summary


def validate_locked(
    model, dataset, stage, builder, feedback, feedback_stats,
    protocol: str | None = None, return_rows: bool = False,
):
    was_training = model.training
    model.eval()
    per_task_psnr: dict[str, list[float]] = {}
    per_task_ssim: dict[str, list[float]] = {}
    rows: list[dict] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(len(dataset)):
            item = dataset[index]
            x = item["degraded"].unsqueeze(0).cuda(non_blocking=True)
            gt = item["clean"].unsqueeze(0).cuda(non_blocking=True)
            prediction = tiled_stage_prediction(
                model, x, gt, stage, builder, feedback, feedback_stats
            ).float().clamp(0, 1)
            mse = (prediction - gt).square().mean().clamp_min(1e-12)
            psnr = float((-10.0 * torch.log10(mse)).item())
            ssim = full_rgb_ssim(prediction, gt)
            per_task_psnr.setdefault(item["task"], []).append(psnr)
            per_task_ssim.setdefault(item["task"], []).append(ssim)
            rows.append({
                "task": item["task"], "name": item["name"],
                "psnr": psnr, "ssim": ssim,
            })
    if protocol is not None:
        expected = set(EXPECTED_VALIDATION_TASKS[protocol])
        actual = set(per_task_psnr)
        if actual != expected:
            raise RuntimeError(
                f"locked validation task set mismatch: actual={sorted(actual)} "
                f"expected={sorted(expected)}"
            )
        ordered_tasks = EXPECTED_VALIDATION_TASKS[protocol]
    else:
        ordered_tasks = tuple(sorted(per_task_psnr))
    summary = summarize_locked_metrics(
        per_task_psnr, per_task_ssim, tuple(ordered_tasks)
    )
    if was_training:
        model.train()
        if stage in {"b_oracle", "b_predicted"}:
            model.encoder.eval(); model.d1.eval()
    return (summary, rows) if return_rows else summary


def atomic_write_locked_rows(path: Path, rows: list[dict]) -> str:
    if not rows:
        raise RuntimeError("locked validation produced no per-image rows")
    keys = [(row["task"], row["name"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("locked validation produced duplicate (task, name) rows")
    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in ("psnr", "ssim")):
            raise RuntimeError(f"locked validation produced non-finite metrics: {row}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "name", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def upsert_validation_record(path: Path, summary: dict):
    """Atomically store one unique (epoch, step) locked-val record."""
    records = []
    if path.is_file():
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    key = (int(summary["epoch"]), int(summary["step"]))
    records = [
        row for row in records
        if (int(row["epoch"]), int(row["step"])) != key
    ]
    records.append(summary)
    records.sort(key=lambda row: (int(row["epoch"]), int(row["step"])))
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
    os.replace(temporary, path)


def reconcile_training_csv(path: Path, checkpoint_step: int):
    """Atomically discard uncheckpointed/duplicate rows before resume."""
    if not path.is_file():
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return
        by_step = {}
        for row in reader:
            step = int(row["step"])
            if step <= checkpoint_step:
                by_step[step] = row
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(by_step):
            writer.writerow(by_step[step])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_top3(run_dir, score, epoch, step, model, optimizer, scheduler, cfg, args):
    index_path = run_dir / "top3.json"
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    checkpoint = run_dir / f"val_epoch{epoch:03d}_step{step:07d}.pt"
    save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, 0, step, cfg, args)
    records = [
        row for row in records
        if (int(row["epoch"]), int(row["step"])) != (int(epoch), int(step))
    ]
    records.append({"score": score, "epoch": epoch, "step": step, "checkpoint": checkpoint.name})
    records.sort(key=lambda item: item["score"], reverse=True)
    for stale in records[3:]:
        (run_dir / stale["checkpoint"]).unlink(missing_ok=True)
    records = records[:3]
    temporary = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    os.replace(temporary, index_path)


def commit_pending_validation(
    *, model, locked_val, stage, builder, feedback, feedback_stats,
    run_dir: Path, optimizer, scheduler, cfg, args,
    epoch: int, batch_in_epoch: int, step: int, kind: str,
):
    """Idempotently finish the checkpoint -> metric -> top3 transaction."""
    if kind not in {"epoch", "max_steps"}:
        raise ValueError(f"unknown validation transaction kind: {kind}")
    summary, paired_rows = validate_locked(
        model, locked_val, stage, builder, feedback, feedback_stats,
        protocol=cfg["protocol"],
        return_rows=True,
    )
    summary.update({"epoch": epoch, "step": step})
    if kind == "max_steps":
        summary["batch_in_epoch"] = batch_in_epoch
    paired_path = (
        ROOT / "artifacts/metrics/locked_rows" / args.run_name
        / f"epoch{epoch:03d}_step{step:07d}.csv"
    )
    summary["paired_rows_path"] = str(paired_path.resolve())
    summary["paired_rows_sha256"] = atomic_write_locked_rows(
        paired_path, paired_rows
    )
    val_path = ROOT / "artifacts" / "metrics" / f"{args.run_name}_locked_val.jsonl"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    upsert_validation_record(val_path, summary)
    print("LOCKED_VAL " + json.dumps(summary), flush=True)
    if kind == "epoch":
        update_top3(
            run_dir, summary["macro_psnr"], epoch, step,
            model, optimizer, scheduler, cfg, args,
        )
    # Clear the durable pending marker only after every required artifact has
    # been atomically committed.  A crash earlier simply replays this helper.
    save_checkpoint(
        run_dir / "last.pt", model, optimizer, scheduler,
        epoch, batch_in_epoch, step, cfg, args, validation_pending=None,
    )
    return summary


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed_override is not None:
        cfg["seed"] = args.seed_override
    if cfg["micro_batch"] * cfg["accumulation"] != cfg["effective_batch"]:
        raise ValueError("micro_batch * accumulation must equal effective_batch")
    if cfg.get("precision") != "bf16":
        raise ValueError("only the audited bf16 training path is enabled")
    if args.workers_override is not None and args.workers_override <= 0:
        raise ValueError("workers-override must be positive")
    runtime_identity = runtime_identity_for_config(Path(args.config), cfg)
    assert_no_runtime_worker_override(cfg)
    if runtime_identity and args.workers_override is not None:
        raise ValueError(
            "workers-override is forbidden for a frozen Stage-B runtime; "
            "the generated config owns the worker count"
        )
    assert_stage_b_cublas_environment(runtime_identity)
    seed_all(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    # Stage-B mechanism thresholds are as small as 0.02 dB.  Runtime kernel
    # autotuning must not select different algorithms across parallel arms or
    # resumes and masquerade as a feedback effect.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    run_dir = ROOT / "artifacts" / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock_fd = acquire_run_lock(run_dir, args.run_name)
    args.source_init_path = str(Path(args.init).resolve()) if args.init else None
    args.source_init_sha256 = sha256_file(args.init) if args.init else None
    log_path = ROOT / "artifacts" / "logs" / f"{args.run_name}.csv"
    dataset = AIOTrainDataset(
        cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["crop_size"],
        strict=not args.allow_incomplete_data,
        split_manifest=cfg.get("split_manifest"),
        split=cfg.get("train_split", "train"),
    )
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=cfg["micro_batch"],
        shuffle=True,
        num_workers=(args.workers_override or cfg["workers"]),
        pin_memory=True,
        drop_last=True,
        # Recreate workers each epoch so epoch-addressable seeds also hold
        # after process restart/resume.
        persistent_workers=False,
        generator=loader_generator,
    )
    model = build_model(cfg, args.stage).cuda()
    configure_feedback_mode(model, args.stage, args.feedback)
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        if args.allow_incomplete_data:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print("SMOKE_ONLY init missing", missing, "unexpected", unexpected)
        else:
            if "config" not in payload or "split_manifest_sha256" not in payload:
                raise RuntimeError("formal internal init requires checkpoint provenance metadata")
            if payload["config"].get("protocol") != cfg["protocol"]:
                raise RuntimeError("init checkpoint protocol mismatch")
            expected_split = hashlib.sha256(Path(cfg["split_manifest"]).read_bytes()).hexdigest()
            if payload["split_manifest_sha256"] != expected_split:
                raise RuntimeError("init checkpoint locked-split mismatch")
            init_scope = load_formal_init(model, state, args.stage)
            print(
                f"STRICT_INIT scope={init_scope} checkpoint={args.init} protocol={cfg['protocol']} "
                f"step={payload.get('step')} split_sha256={expected_split}",
                flush=True,
            )
    configure_trainable(model, args.stage)
    params, optimizer = build_optimizer(model, args.stage, cfg)
    # Discard at most accumulation-1 micro-batches per epoch.  Carrying a
    # partial accumulated gradient across an epoch boundary makes resume and
    # scheduler accounting non-reproducible.
    usable_batches = (len(loader) // cfg["accumulation"]) * cfg["accumulation"]
    if usable_batches == 0:
        raise RuntimeError("not enough micro-batches for one optimizer step")
    steps_per_epoch = usable_batches // cfg["accumulation"]
    total_steps = cfg["epochs"] * steps_per_epoch
    # Short Stage-B kill experiments need a complete, identical schedule in
    # their registered step budget.  Using the 240-epoch warmup here would
    # leave every arm effectively untrained.
    if args.max_steps and args.stage in {"b_oracle", "b_predicted"}:
        pilot_warmup = min(50, max(args.max_steps // 10, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: warmup_cosine(
                step, args.max_steps, pilot_warmup, cfg["min_lr"] / cfg["lr"]
            ),
        )
        scheduler_unit = "step"
    elif args.stage in {"c", "baseline_ft", "baseline_matched_ft"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"]
        )
        scheduler_unit = "epoch"
    else:
        schedule_epochs = cfg.get("scheduler_max_epochs", cfg["epochs"] + 30)
        warmup_start_lr = cfg.get("warmup_start_lr", 1e-7)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: r2r_pretrain_epoch_ratio(
                epoch,
                cfg["lr"],
                cfg["warmup_epochs"],
                schedule_epochs,
                warmup_start_lr,
                cfg.get("pretrain_eta_min", 0.0),
            ),
        )
        scheduler_unit = "epoch"
    start_epoch = start_batch = global_step = 0
    resume_validation_pending = None
    prior_runtime_accounting = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_contract(payload, cfg, args)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch, global_step = payload["epoch"], payload["step"]
        start_batch = payload.get("batch_in_epoch", 0)
        resume_validation_pending = payload.get("validation_pending")
        prior_runtime_accounting = payload.get("runtime_accounting")
        torch.set_rng_state(payload["rng"]["torch"])
        if "cuda" in payload["rng"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        reconcile_training_csv(log_path, global_step)
    # A compacted or crash-interrupted run may retain the durable sidecar even
    # when no resumable ``last.pt`` is present.  Never reset its cumulative
    # counters merely because this invocation was entered without --resume.
    sidecar = run_dir / "runtime_accounting.json"
    if prior_runtime_accounting is None and sidecar.is_file():
        prior_runtime_accounting = read_runtime_sidecar(sidecar)
    start_runtime_accounting(
        gpu_count=1,
        run_name=args.run_name,
        protocol=cfg["protocol"],
        stage=args.stage,
        prior=prior_runtime_accounting,
    )
    ensure_run_contract(run_dir, cfg, args)
    builder_payload = (
        build_coordinate_builder(
            cfg,
            args.allow_incomplete_data,
            expected_stage_a_checkpoint=(
                args.init if args.stage in {"b_oracle", "b_predicted"} else None
            ),
        )
        if args.stage in {"b_oracle", "b_predicted", "c"} else None
    )
    builder, feedback_stats = builder_payload if builder_payload is not None else (None, None)
    if (
        isinstance(model, SRSCLite)
        and args.stage == "c"
        and args.feedback in DETERMINISTIC_FEEDBACK_MODES
    ):
        if feedback_stats is None:
            raise RuntimeError("deterministic Stage-C feedback requires locked train statistics")
        model.configure_deterministic_feedback(args.feedback, feedback_stats)
    locked_val = None if args.allow_incomplete_data else build_locked_val(
        cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["split_manifest"]
    )
    if resume_validation_pending is not None:
        if locked_val is None:
            raise RuntimeError("cannot recover a pending validation without locked_val")
        commit_pending_validation(
            model=model,
            locked_val=locked_val,
            stage=args.stage,
            builder=builder,
            feedback=args.feedback,
            feedback_stats=feedback_stats,
            run_dir=run_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            args=args,
            epoch=start_epoch,
            batch_in_epoch=start_batch,
            step=global_step,
            kind=resume_validation_pending,
        )
        if args.max_steps and global_step >= args.max_steps:
            print(json.dumps({
                "status": "stopped_max_steps",
                "run": args.run_name,
                "step": global_step,
                "checkpoint": str(run_dir / "last.pt"),
                "recovered_pending_validation": True,
            }))
            os.close(run_lock_fd)
            return
    if args.max_steps and global_step >= args.max_steps:
        print(json.dumps({
            "status": "stopped_max_steps",
            "run": args.run_name,
            "step": global_step,
            "checkpoint": str(run_dir / "last.pt"),
            "already_complete": True,
        }))
        os.close(run_lock_fd)
        return
    amp_dtype = torch.bfloat16
    optimizer.zero_grad(set_to_none=True)
    header_needed = not log_path.exists()
    with log_path.open("a", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["time", "epoch", "step", "loss", "rest", "state", "clean", "lr", "peak_gb"],
        )
        if header_needed:
            writer.writeheader()
        stop = False
        for epoch in range(start_epoch, cfg["epochs"]):
            # Epoch-addressable sampler and worker seeds make a resumed epoch
            # replay the exact skipped batches before continuing.
            loader_generator.manual_seed(cfg["seed"] + epoch)
            model.train()
            replay_digest = hashlib.sha256()
            replay_sample_count = 0
            if args.stage in {"b_oracle", "b_predicted"}:
                model.encoder.eval(); model.d1.eval()
            for batch_index, batch in enumerate(loader):
                if batch_index >= usable_batches:
                    break
                replay_sample_count += update_replay_digest(replay_digest, batch)
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                x = batch["degraded"].cuda(non_blocking=True)
                gt = batch["clean"].cuda(non_blocking=True)
                if args.stage in {"b_oracle", "b_predicted"}:
                    terms, loss = backward_stage_b_microbatch(
                        model,
                        x,
                        gt,
                        stage=args.stage,
                        builder=builder,
                        feedback=args.feedback,
                        feedback_stats=feedback_stats,
                        cfg=cfg,
                        accumulation=cfg["accumulation"],
                        amp_dtype=amp_dtype,
                    )
                    rest = terms.rest
                    state_term = terms.state
                    clean_term = terms.clean
                else:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        state_term = x.new_zeros(())
                        clean_term = x.new_zeros(())
                        if args.stage in {
                            "baseline",
                            "baseline_matched",
                            "baseline_ft",
                            "baseline_matched_ft",
                        }:
                            prediction = model(x)
                            rest = restoration_l1(prediction, gt)
                        elif args.stage == "a":
                            features = model.encoder(x)
                            delta0, _ = model.d1(features)
                            rest = restoration_l1(x + delta0, gt)
                        else:
                            # Stage-C retains its joint gradient routing.  Its
                            # terms are intentionally not used by the frozen
                            # Stage-B memory preflight.
                            details = model.forward_details(x)
                            prediction = details.y2
                            target_mode = (
                                args.feedback
                                if args.feedback in DETERMINISTIC_FEEDBACK_MODES
                                else predicted_supervision_mode(args.feedback)
                            )
                            coords = builder(
                                x,
                                details.y1,
                                gt,
                                [feature.shape[-2:] for feature in details.features],
                                requested={target_mode},
                            )
                            raw_targets = feedback_from_coordinates(
                                coords,
                                target_mode,
                                model.oracle_ceiling_adapter,
                            )
                            targets = normalize_feedback(
                                raw_targets, target_mode, feedback_stats
                            )
                            direction_weight = (
                                0.1
                                if target_mode in {"O7", "O8", "O15"}
                                else 0.0
                            )
                            coordinate_direction_weights = (
                                direction_weights_from_coordinates(
                                    coords, target_mode
                                )
                                if direction_weight
                                else None
                            )
                            state_term, _ = state_loss(
                                details.states,
                                targets,
                                direction_cosine_weight=direction_weight,
                                direction_weights=coordinate_direction_weights,
                                direction_valid_masks=(
                                    direction_valid_masks(
                                        raw_targets,
                                        coordinate_direction_weights,
                                    )
                                    if direction_weight
                                    else None
                                ),
                            )
                            rest = 0.5 * restoration_l1(
                                details.y1, gt
                            ) + restoration_l1(prediction, gt)
                            clean_term = (
                                (1.0 - coords[0].q)
                                * (prediction - details.y1).abs()
                            ).mean()
                        loss = (
                            rest
                            + cfg.get("lambda_state", 0.1) * state_term
                            + cfg.get("lambda_clean", 0.1) * clean_term
                        ) / cfg["accumulation"]
                    loss.backward()
                if (batch_index + 1) % cfg["accumulation"] == 0:
                    commit_optimizer_update(
                        params,
                        optimizer,
                        gradient_clip=cfg["gradient_clip"],
                        scheduler=scheduler,
                        scheduler_unit=scheduler_unit,
                    )
                    global_step += 1
                    if global_step % 50 == 0:
                        row = {
                            "time": time.time(), "epoch": epoch, "step": global_step,
                            "loss": float(loss.detach() * cfg["accumulation"]), "rest": float(rest.detach()),
                            "state": float(state_term.detach()), "lr": optimizer.param_groups[0]["lr"],
                            "clean": float(clean_term.detach()),
                            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                        }
                        writer.writerow(row); log_file.flush(); print(json.dumps(row), flush=True)
                    if global_step % cfg["save_every_steps"] == 0:
                        save_checkpoint(
                            run_dir / "last.pt", model, optimizer, scheduler,
                            epoch, batch_index + 1, global_step, cfg, args,
                        )
                    if args.max_steps and global_step >= args.max_steps:
                        stop = True; break
            if stop:
                pending_kind = "max_steps" if locked_val is not None else None
                save_checkpoint(
                    run_dir / "last.pt", model, optimizer, scheduler,
                    epoch, batch_index + 1, global_step, cfg, args,
                    validation_pending=pending_kind,
                )
                if locked_val is not None:
                    commit_pending_validation(
                        model=model,
                        locked_val=locked_val,
                        stage=args.stage,
                        builder=builder,
                        feedback=args.feedback,
                        feedback_stats=feedback_stats,
                        run_dir=run_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        args=args,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        step=global_step,
                        kind="max_steps",
                    )
                break
            completed_epoch = epoch + 1
            commit_epoch_replay_digest(
                args.run_name,
                completed_epoch,
                global_step,
                replay_sample_count,
                replay_digest.hexdigest(),
                cfg,
            )
            if scheduler_unit == "epoch":
                scheduler.step()
            should_validate = locked_val is not None and (
                completed_epoch % cfg["validate_every_epochs"] == 0 or completed_epoch == cfg["epochs"]
            )
            save_checkpoint(
                run_dir / "last.pt", model, optimizer, scheduler,
                completed_epoch, 0, global_step, cfg, args,
                validation_pending=("epoch" if should_validate else None),
            )
            if should_validate:
                commit_pending_validation(
                    model=model,
                    locked_val=locked_val,
                    stage=args.stage,
                    builder=builder,
                    feedback=args.feedback,
                    feedback_stats=feedback_stats,
                    run_dir=run_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    args=args,
                    epoch=completed_epoch,
                    batch_in_epoch=0,
                    step=global_step,
                    kind="epoch",
                )
            start_batch = 0
    status = "stopped_max_steps" if stop else "complete"
    print(json.dumps({"status": status, "run": args.run_name, "step": global_step, "checkpoint": str(run_dir / 'last.pt')}))
    # Keep the descriptor live through every checkpoint/log flush above.  The
    # explicit close is mainly documentary; abnormal exits release it too.
    os.close(run_lock_fd)


if __name__ == "__main__":
    main()
