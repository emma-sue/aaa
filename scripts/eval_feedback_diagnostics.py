#!/usr/bin/env python3
"""Audit predicted feedback representations on non-test data only.

The primary use is the P12/O12 residual-code arm.  Every metric is computed in
the same train-statistics-normalized eight-channel space used to supervise the
assessor.  This script deliberately does not execute D2 or report restoration
quality; it answers only whether the predicted feedback represents its target.

Formal split policy
-------------------
``locked_val`` evaluates every preregistered held-out validation image.
``train_diagnostic`` evaluates a deterministic SHA256-ranked set of unique
training examples.  ``official_test`` is rejected unconditionally.

Metric definitions
------------------
At each scale, one sample is an eight-dimensional vector at one spatial
location.  MAE is the mean absolute scalar error over vectors and channels.
Cosine is the ordinary 8-D cosine.  A target vector is direction-valid when
its L2 norm is at least ``zero_epsilon``.  Valid targets with a zero prediction
receive cosine zero (rather than being discarded); target-zero vectors are
excluded from the primary cosine and counted separately.  Variance is the
population variance E[z^2]-E[z]^2 per channel.  Entropy is the marginal
base-2 Shannon entropy of each channel after clamping normalized values to a
fixed range and assigning them to fixed-width bins.  The reported aggregate
is both vector-count-weighted pooling and an equal-scale macro average; no
high-dimensional joint-entropy estimator is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    feedback_from_coordinates,
    model_kwargs,
    normalize_feedback,
)
from src.data import AIOTrainDataset, build_locked_val  # noqa: E402
from src.net import (  # noqa: E402
    PREDICTED_FEEDBACK_MODES,
    SRSCCoordinateBuilder,
    SRSCLite,
    predicted_supervision_mode,
)
from src.net.restormer_blocks import pad_to_multiple  # noqa: E402


ALLOWED_SPLITS = frozenset({"locked_val", "train_diagnostic"})
SCALE_NAMES = ("S1", "S2", "S3", "S4")
SCALE_DIVISORS = (1, 2, 4, 8)
CODE_PATHS = (
    Path(__file__),
    ROOT / "scripts/train.py",
    ROOT / "src/net/srsc_lite.py",
    ROOT / "src/net/srsc_coordinates.py",
    ROOT / "src/net/feedback_controls.py",
    ROOT / "src/data/aio_dataset.py",
)


def crop_to_valid_native_region(
    tensor: torch.Tensor,
    original_shape: tuple[int, int],
    scale_index: int,
) -> torch.Tensor:
    """Remove right/bottom model padding from one native-scale tensor.

    The network is padded to a multiple of eight before its encoder.  At a
    downsampled scale, only cells whose complete ``2**scale_index`` source
    block lies inside the original image are counted.  Because padding is
    applied only on the right and bottom, those cells form a top-left
    rectangle with floor-divided spatial dimensions.
    """
    if tensor.ndim != 4:
        raise ValueError(f"native-scale feedback must be BCHW, got {tuple(tensor.shape)}")
    if scale_index < 0 or scale_index >= len(SCALE_DIVISORS):
        raise ValueError(f"invalid native-scale index: {scale_index}")
    if len(original_shape) != 2:
        raise ValueError(f"original_shape must be (H,W), got {original_shape!r}")
    original_h, original_w = (int(value) for value in original_shape)
    divisor = SCALE_DIVISORS[scale_index]
    valid_h = original_h // divisor
    valid_w = original_w // divisor
    if valid_h <= 0 or valid_w <= 0:
        raise ValueError(
            f"original image {original_shape} is too small for native divisor {divisor}"
        )
    actual_h, actual_w = tensor.shape[-2:]
    if valid_h > actual_h or valid_w > actual_w:
        raise ValueError(
            "native feedback is smaller than its unpadded valid region: "
            f"valid={(valid_h, valid_w)} actual={(actual_h, actual_w)}"
        )
    return tensor[..., :valid_h, :valid_w]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("feedback diagnostics output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_diagnostic_split(split: str) -> str:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(
            f"feedback diagnostics permit only {sorted(ALLOWED_SPLITS)}; "
            f"split {split!r} is forbidden (official_test is never unlockable)"
        )
    return split


def deterministic_train_indices(
    dataset: AIOTrainDataset, limit_per_task: int
) -> tuple[list[int], dict[str, int], str]:
    """Select unique train items without using model outputs or metrics."""
    if limit_per_task <= 0:
        raise ValueError("limit_per_task must be positive")
    unique: dict[tuple[str, str, str, int], int] = {}
    for index, sample in enumerate(dataset.samples):
        key = (sample.task, str(sample.degraded), str(sample.clean), sample.sigma)
        unique.setdefault(key, index)
    ranked: dict[str, list[tuple[bytes, int]]] = defaultdict(list)
    for key, index in unique.items():
        token = hashlib.sha256((repr(key) + ":feedback-diagnostic-v1").encode()).digest()
        ranked[key[0]].append((token, index))
    selected: list[int] = []
    counts: dict[str, int] = {}
    for task, values in sorted(ranked.items()):
        values.sort()
        chosen = [index for _, index in values[:limit_per_task]]
        selected.extend(chosen)
        counts[task] = len(chosen)
    policy_digest = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode()
    ).hexdigest()
    return selected, counts, policy_digest


class FeedbackDiagnosticsAccumulator:
    """Streaming sufficient statistics for paired Bx8xHxW feedback tensors."""

    def __init__(
        self,
        *,
        channels: int = 8,
        zero_epsilon: float = 1e-6,
        entropy_bins: int = 64,
        entropy_range: float = 8.0,
    ):
        if channels <= 0:
            raise ValueError("channels must be positive")
        if zero_epsilon <= 0:
            raise ValueError("zero_epsilon must be positive")
        if entropy_bins < 2 or entropy_range <= 0:
            raise ValueError("entropy bins/range are invalid")
        self.channels = channels
        self.zero_epsilon = float(zero_epsilon)
        self.entropy_bins = int(entropy_bins)
        self.entropy_range = float(entropy_range)
        self.vector_count = 0
        self.abs_error_sum = torch.zeros(channels, dtype=torch.float64)
        self.square_error_sum = torch.zeros(channels, dtype=torch.float64)
        self.sums = {
            key: torch.zeros(channels, dtype=torch.float64)
            for key in ("prediction", "target", "error")
        }
        self.sumsq = {
            key: torch.zeros(channels, dtype=torch.float64)
            for key in ("prediction", "target", "error")
        }
        self.hist = {
            key: torch.zeros(channels, entropy_bins, dtype=torch.int64)
            for key in ("prediction", "target", "error")
        }
        self.target_valid_count = 0
        self.target_zero_count = 0
        self.prediction_zero_count = 0
        self.both_nonzero_count = 0
        self.cosine_target_valid_sum = 0.0
        self.cosine_both_nonzero_sum = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}"
            )
        if prediction.ndim != 4 or prediction.shape[1] != self.channels:
            raise ValueError(
                f"feedback must be Bx{self.channels}xHxW, got {tuple(prediction.shape)}"
            )
        # Move one image-scale statistic block to CPU in float64.  Metrics are
        # diagnostics, so they must not inherit BF16 accumulation error.
        pred = prediction.detach().float().permute(0, 2, 3, 1).reshape(-1, self.channels).cpu().double()
        tgt = target.detach().float().permute(0, 2, 3, 1).reshape(-1, self.channels).cpu().double()
        if not torch.isfinite(pred).all() or not torch.isfinite(tgt).all():
            raise FloatingPointError("non-finite predicted/target feedback encountered")
        error = pred - tgt
        count = int(pred.shape[0])
        self.vector_count += count
        self.abs_error_sum += error.abs().sum(0)
        self.square_error_sum += error.square().sum(0)
        for key, values in (("prediction", pred), ("target", tgt), ("error", error)):
            self.sums[key] += values.sum(0)
            self.sumsq[key] += values.square().sum(0)
            clipped = values.clamp(-self.entropy_range, self.entropy_range)
            for channel in range(self.channels):
                counts = torch.histc(
                    clipped[:, channel].float(),
                    bins=self.entropy_bins,
                    min=-self.entropy_range,
                    max=self.entropy_range,
                ).round().to(torch.int64)
                self.hist[key][channel] += counts

        target_norm = tgt.norm(dim=1)
        prediction_norm = pred.norm(dim=1)
        target_valid = target_norm >= self.zero_epsilon
        prediction_valid = prediction_norm >= self.zero_epsilon
        both_nonzero = target_valid & prediction_valid
        self.target_valid_count += int(target_valid.sum())
        self.target_zero_count += int((~target_valid).sum())
        self.prediction_zero_count += int((~prediction_valid).sum())
        self.both_nonzero_count += int(both_nonzero.sum())
        if target_valid.any():
            cosine = (pred * tgt).sum(1) / (
                prediction_norm * target_norm
            ).clamp_min(self.zero_epsilon**2)
            cosine = cosine.clamp(-1.0, 1.0)
            # A zero prediction against a valid target is an assessor failure,
            # not a reason to remove the sample.  Its cosine is defined as 0.
            cosine = torch.where(prediction_valid, cosine, torch.zeros_like(cosine))
            self.cosine_target_valid_sum += float(cosine[target_valid].sum())
            if both_nonzero.any():
                self.cosine_both_nonzero_sum += float(cosine[both_nonzero].sum())

    @staticmethod
    def _entropy_bits(counts: torch.Tensor) -> float:
        total = int(counts.sum())
        if total == 0:
            return float("nan")
        probability = counts.double() / total
        positive = probability > 0
        return float(-(probability[positive] * torch.log2(probability[positive])).sum())

    def finalize(self) -> dict:
        if self.vector_count == 0:
            raise RuntimeError("feedback diagnostic accumulator is empty")
        count = float(self.vector_count)
        mae = self.abs_error_sum / count
        rmse = torch.sqrt(self.square_error_sum / count)
        distributions: dict[str, dict] = {}
        for key in ("prediction", "target", "error"):
            mean = self.sums[key] / count
            variance = (self.sumsq[key] / count - mean.square()).clamp_min(0.0)
            entropy = [self._entropy_bits(self.hist[key][c]) for c in range(self.channels)]
            distributions[key] = {
                "channel_mean": mean.tolist(),
                "channel_population_variance": variance.tolist(),
                "mean_channel_population_variance": float(variance.mean()),
                "channel_marginal_entropy_bits": entropy,
                "mean_channel_marginal_entropy_bits": float(np.mean(entropy)),
                "mean_channel_normalized_entropy": float(
                    np.mean(entropy) / math.log2(self.entropy_bins)
                ),
            }
        return {
            "vector_count": self.vector_count,
            "scalar_count": self.vector_count * self.channels,
            "channel_mae": mae.tolist(),
            "scalar_mae": float(mae.mean()),
            "channel_rmse": rmse.tolist(),
            "scalar_rmse": float(rmse.mean()),
            "cosine": {
                "target_valid_count": self.target_valid_count,
                "target_zero_count": self.target_zero_count,
                "prediction_zero_count": self.prediction_zero_count,
                "both_nonzero_count": self.both_nonzero_count,
                "target_valid_fraction": self.target_valid_count / self.vector_count,
                "prediction_zero_fraction": self.prediction_zero_count / self.vector_count,
                "mean_over_target_valid_zero_prediction_is_zero": (
                    self.cosine_target_valid_sum / self.target_valid_count
                    if self.target_valid_count else None
                ),
                "mean_over_both_nonzero_diagnostic": (
                    self.cosine_both_nonzero_sum / self.both_nonzero_count
                    if self.both_nonzero_count else None
                ),
            },
            "distribution": distributions,
        }


def scale_macro_summary(per_scale: dict[str, dict]) -> dict:
    if set(per_scale) != set(SCALE_NAMES):
        raise ValueError("scale macro requires exactly S1..S4")

    def finite_mean(values: Iterable[float | None]) -> float | None:
        kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return float(np.mean(kept)) if kept else None

    return {
        "definition": "unweighted arithmetic mean of the four native-scale metrics",
        "scalar_mae": finite_mean(row["scalar_mae"] for row in per_scale.values()),
        "scalar_rmse": finite_mean(row["scalar_rmse"] for row in per_scale.values()),
        "cosine_target_valid": finite_mean(
            row["cosine"]["mean_over_target_valid_zero_prediction_is_zero"]
            for row in per_scale.values()
        ),
        "prediction_mean_channel_variance": finite_mean(
            row["distribution"]["prediction"]["mean_channel_population_variance"]
            for row in per_scale.values()
        ),
        "target_mean_channel_variance": finite_mean(
            row["distribution"]["target"]["mean_channel_population_variance"]
            for row in per_scale.values()
        ),
        "prediction_mean_channel_entropy_bits": finite_mean(
            row["distribution"]["prediction"]["mean_channel_marginal_entropy_bits"]
            for row in per_scale.values()
        ),
        "target_mean_channel_entropy_bits": finite_mean(
            row["distribution"]["target"]["mean_channel_marginal_entropy_bits"]
            for row in per_scale.values()
        ),
    }


def validate_checkpoint_provenance(
    *,
    payload: dict,
    cfg: dict,
    config_path: Path,
    checkpoint_path: Path,
    feedback: str,
) -> dict:
    if feedback not in PREDICTED_FEEDBACK_MODES:
        raise ValueError(f"{feedback} is not a predicted feedback arm")
    checkpoint_args = payload.get("args", {})
    stage = checkpoint_args.get("stage")
    if stage not in {"b_predicted", "c"}:
        raise ValueError(
            f"predicted feedback diagnostics require b_predicted/c checkpoint, got {stage!r}"
        )
    if checkpoint_args.get("feedback") != feedback:
        raise ValueError(
            f"checkpoint feedback {checkpoint_args.get('feedback')!r} != requested {feedback!r}"
        )
    config_sha256 = sha256_file(config_path)
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("checkpoint/config SHA256 mismatch")
    if payload.get("config") != cfg:
        raise RuntimeError("checkpoint effective config differs from supplied config")
    split_path = Path(cfg["split_manifest"])
    stats_path = Path(cfg["coordinate_stats"])
    for path, label in ((split_path, "split manifest"), (stats_path, "coordinate statistics")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")
    split_sha256 = sha256_file(split_path)
    if payload.get("split_manifest_sha256") != split_sha256:
        raise RuntimeError("checkpoint/split-manifest SHA256 mismatch")
    stats = json.loads(stats_path.read_text())
    if stats.get("protocol") != cfg.get("protocol"):
        raise RuntimeError("coordinate statistics protocol mismatch")
    if stats.get("split_manifest_sha256") != split_sha256:
        raise RuntimeError("coordinate statistics split-manifest mismatch")

    run_contract_path = (
        ROOT / "artifacts/checkpoints" / str(checkpoint_args.get("run_name"))
        / "run_contract.json"
    )
    if not run_contract_path.is_file():
        raise FileNotFoundError(
            f"formal predicted-feedback checkpoint has no run contract: {run_contract_path}"
        )
    run_contract_sha256 = sha256_file(run_contract_path)
    expected_contract = checkpoint_args.get("run_contract_sha256")
    if not expected_contract or expected_contract != run_contract_sha256:
        raise RuntimeError("checkpoint/run-contract SHA256 mismatch")
    contract = json.loads(run_contract_path.read_text())
    if contract.get("feedback") != feedback or contract.get("stage") != stage:
        raise RuntimeError("run contract feedback/stage mismatch")
    if contract.get("config_sha256") != config_sha256:
        raise RuntimeError("run contract config SHA256 mismatch")
    if contract.get("split_manifest_sha256") != split_sha256:
        raise RuntimeError("run contract split-manifest SHA256 mismatch")
    if contract.get("coordinate_stats_sha256") != sha256_file(stats_path):
        raise RuntimeError("run contract coordinate-statistics SHA256 mismatch")
    contract_code = contract.get("code_sha256")
    if not isinstance(contract_code, dict) or not contract_code:
        raise RuntimeError("run contract has no immutable training-code hashes")
    for relative_path, expected_sha256 in contract_code.items():
        source_path = ROOT / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
            raise RuntimeError(
                f"current diagnostic code differs from training contract: {relative_path}"
            )

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": split_sha256,
        "coordinate_stats": str(stats_path.resolve()),
        "coordinate_stats_sha256": sha256_file(stats_path),
        "run_contract": str(run_contract_path.resolve()),
        "run_contract_sha256": run_contract_sha256,
        "checkpoint_stage": stage,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_step": payload.get("step"),
        "feedback_interface_mode": feedback,
        "feedback_supervision_mode": predicted_supervision_mode(feedback),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feedback", required=True, choices=sorted(PREDICTED_FEEDBACK_MODES))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--limit-per-task", type=int, default=64)
    parser.add_argument("--zero-epsilon", type=float, default=1e-6)
    parser.add_argument("--entropy-bins", type=int, default=64)
    parser.add_argument("--entropy-range", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = validate_diagnostic_split(args.split)
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("config/checkpoint does not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("CPU diagnostics require --precision fp32")

    cfg = yaml.safe_load(args.config.read_text())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = validate_checkpoint_provenance(
        payload=payload,
        cfg=cfg,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        feedback=args.feedback,
    )
    stats = json.loads(Path(cfg["coordinate_stats"]).read_text())
    supervision_mode = provenance["feedback_supervision_mode"]

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if args.device == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(args.device)
    model = SRSCLite(**model_kwargs(cfg)).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    builder = SRSCCoordinateBuilder(
        tau_v=stats["tau_v"],
        tau_e=stats["tau_e"],
        pca_projection=stats.get("pca_direction_matrix"),
        pca_mean=stats.get("pca_direction_mean"),
    ).to(device).eval()

    if split == "locked_val":
        dataset = build_locked_val(
            cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["split_manifest"]
        )
        indices = list(range(len(dataset)))
        selection = {
            "policy": "all preregistered locked-validation images",
            "complete_split": True,
            "selection_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    else:
        dataset = AIOTrainDataset(
            cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["crop_size"],
            strict=True, split_manifest=cfg["split_manifest"], split="train",
        )
        indices, counts, selection_sha256 = deterministic_train_indices(
            dataset, args.limit_per_task
        )
        selection = {
            "policy": "SHA256-ranked unique train examples; sequential fixed RNG crops",
            "complete_split": False,
            "limit_per_task": args.limit_per_task,
            "selected_per_task": counts,
            "selection_sha256": selection_sha256,
        }

    accumulator_kwargs = dict(
        zero_epsilon=args.zero_epsilon,
        entropy_bins=args.entropy_bins,
        entropy_range=args.entropy_range,
    )
    per_scale_accumulators = {
        name: FeedbackDiagnosticsAccumulator(**accumulator_kwargs) for name in SCALE_NAMES
    }
    pooled = FeedbackDiagnosticsAccumulator(**accumulator_kwargs)
    task_counts: Counter[str] = Counter()
    start_time = time.time()
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        for ordinal, index in enumerate(indices, 1):
            item = dataset[index]
            x = item["degraded"].unsqueeze(0).to(device)
            gt = item["clean"].unsqueeze(0).to(device)
            x_pad, original = pad_to_multiple(x, 8)
            gt_pad, gt_original = pad_to_multiple(gt, 8)
            if original != gt_original or x_pad.shape != gt_pad.shape:
                raise RuntimeError("degraded/GT padding mismatch")
            features, y1 = model._encode_coarse(x_pad)
            predictions = model.assessor(x_pad, y1, features)
            coordinates = builder(
                x_pad,
                y1,
                gt_pad,
                [feature.shape[-2:] for feature in features],
                requested={supervision_mode},
            )
            raw_targets = feedback_from_coordinates(
                coordinates, supervision_mode, model.oracle_ceiling_adapter
            )
            targets = normalize_feedback(raw_targets, supervision_mode, stats)
            if len(predictions) != 4 or len(targets) != 4:
                raise RuntimeError("assessor/target did not produce S1..S4")
            for scale_index, (scale_name, prediction, target) in enumerate(zip(
                SCALE_NAMES, predictions, targets
            )):
                prediction = crop_to_valid_native_region(
                    prediction, original, scale_index
                )
                target = crop_to_valid_native_region(target, original, scale_index)
                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"cropped {scale_name} prediction/target mismatch: "
                        f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
                    )
                per_scale_accumulators[scale_name].update(prediction, target)
                pooled.update(prediction, target)
            task = str(item["task"])
            if split == "train_diagnostic" and isinstance(item["task"], int):
                inverse = {value: key for key, value in AIOTrainDataset.TASK_ID.items()}
                task = inverse[item["task"]]
            task_counts[task] += 1
            if ordinal % 25 == 0 or ordinal == len(indices):
                print(
                    f"FEEDBACK_DIAGNOSTIC {ordinal}/{len(indices)} "
                    f"split={split} feedback={args.feedback}",
                    flush=True,
                )

    per_scale = {
        name: accumulator.finalize()
        for name, accumulator in per_scale_accumulators.items()
    }
    result = {
        "schema": "srsc.predicted_feedback_diagnostics.v1",
        "status": "COMPLETE",
        "split": split,
        "protocol": cfg["protocol"],
        "feedback_interface_mode": args.feedback,
        "feedback_supervision_mode": supervision_mode,
        "representation_space": (
            "train-only robust-statistics-normalized 8-channel assessor target"
        ),
        "definitions": {
            "sample": "one 8-D feedback vector at one native-scale spatial location",
            "mae": "sum_{i,c}|prediction[i,c]-target[i,c]|/(N*8)",
            "rmse": (
                "reported per channel as sqrt(mean squared error); scalar_rmse is "
                "the arithmetic mean of channel RMSEs"
            ),
            "cosine": (
                "8-D dot/(L2 norms); target norm < zero_epsilon is excluded; "
                "prediction norm < zero_epsilon with valid target contributes cosine 0"
            ),
            "variance": "population variance E[z^2]-E[z]^2 independently per channel",
            "entropy": (
                "per-channel marginal Shannon entropy in bits after clamping normalized "
                "values to [-entropy_range,+entropy_range] and fixed-width binning; "
                "not joint differential entropy"
            ),
            "pooled_aggregate": "all native-scale vectors pooled with equal vector weight",
            "scale_macro": "unweighted arithmetic mean over S1,S2,S3,S4",
        },
        "metric_parameters": {
            "channels": 8,
            "zero_epsilon": args.zero_epsilon,
            "entropy_bins": args.entropy_bins,
            "entropy_range": args.entropy_range,
            "entropy_clipping": True,
        },
        "spatial_validity": {
            "policy": (
                "exclude right/bottom model padding; at native scale i retain only "
                "the top-left floor(H/2^i) x floor(W/2^i) cells"
            ),
            "scale_divisors": dict(zip(SCALE_NAMES, SCALE_DIVISORS)),
            "complete_source_block_required": True,
            "model_padding_included": False,
        },
        "selection": selection,
        "image_count": len(indices),
        "image_count_per_task": dict(sorted(task_counts.items())),
        "per_scale": per_scale,
        "pooled_aggregate": pooled.finalize(),
        "scale_macro": scale_macro_summary(per_scale),
        "provenance": provenance,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_PATHS
        },
        "runtime": {
            "device": str(device),
            "precision": args.precision,
            "torch_version": torch.__version__,
            "elapsed_seconds": time.time() - start_time,
        },
    }
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "feedback": args.feedback,
                "split": split,
                "scalar_mae": result["pooled_aggregate"]["scalar_mae"],
                "cosine": result["pooled_aggregate"]["cosine"][
                    "mean_over_target_valid_zero_prediction_is_zero"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
