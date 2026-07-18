#!/usr/bin/env python3
"""Lock train-only SRSC thresholds and feedback normalization statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import feedback_from_coordinates, model_kwargs  # noqa: E402
from src.data import AIOTrainDataset  # noqa: E402
from src.net import SRSCCoordinateBuilder, SRSCLite  # noqa: E402
from src.net.feedback_controls import isotropic_direction_normalization  # noqa: E402


def file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_indices(dataset, limit_per_task: int):
    unique = {}
    for index, sample in enumerate(dataset.samples):
        key = (sample.task, str(sample.degraded), str(sample.clean), sample.sigma)
        unique.setdefault(key, index)
    by_task = defaultdict(list)
    for key, index in unique.items():
        token = hashlib.sha256((str(key) + ":20260713").encode()).digest()
        by_task[key[0]].append((token, index))
    selected = []
    counts = {}
    for task, values in sorted(by_task.items()):
        values.sort()
        chosen = [index for _, index in values[:limit_per_task]]
        selected.extend(chosen)
        counts[task] = len(chosen)
    return selected, counts


def selected_sample_records(dataset, indices: list[int]) -> list[dict]:
    """Serialize the exact train-only identity set without opening image bytes."""
    records = []
    for index in indices:
        sample = dataset.samples[index]
        records.append({
            "index": int(index),
            "task": str(sample.task),
            "degraded": str(Path(sample.degraded).resolve()),
            "clean": str(Path(sample.clean).resolve()),
            "sigma": int(sample.sigma),
        })
    return records


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepared_data_manifest_binding(protocol: str) -> dict:
    path = ROOT / "artifacts/manifests" / f"{protocol}.json"
    payload = json.loads(path.read_text())
    list_sha = payload.get("list_sha256")
    if payload.get("protocol") != protocol or not isinstance(list_sha, dict) or not list_sha:
        raise RuntimeError(f"invalid prepared-data manifest: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "list_sha256": dict(sorted(list_sha.items())),
    }


def spatial_sample(tensor: torch.Tensor, maximum: int):
    # B,C,H,W -> C,N, deterministic evenly spaced positions across BHW.
    values = tensor.detach().float().permute(1, 0, 2, 3).reshape(tensor.shape[1], -1)
    if values.shape[1] <= maximum:
        return values.cpu()
    index = torch.linspace(0, values.shape[1] - 1, maximum, device=values.device).long()
    return values[:, index].cpu()


def robust(values: torch.Tensor):
    values = values.float()
    # Scale-only robust normalization preserves the exact zero that denotes
    # an invalid/gated state coordinate.
    scale = torch.quantile(values.abs(), 0.90)
    if not torch.isfinite(scale) or scale < 1e-4:
        return 0.0, 1.0
    return 0.0, float(scale)


def build_normalization_statistics(collected: dict) -> dict:
    """Convert sampled train-only coordinates into locked normalization.

    Kept as a pure function so the exact producer consumed by training can be
    regression-tested without loading a dataset or allocating a GPU model.
    """

    normalization = {}
    for mode, channel_samples in collected.items():
        if len(channel_samples) != 8 or any(not values for values in channel_samples):
            raise ValueError(f"{mode} normalization requires samples for 8 channels")
        pairs = [robust(torch.cat(values)) for values in channel_samples]
        centers, scales = isotropic_direction_normalization(
            mode,
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        normalization[mode] = {"center": centers, "scale": scales}
    return normalization


def fit_pca_projection(
    unit_deviation: torch.Tensor,
    vnorm: torch.Tensor,
    mraw: torch.Tensor,
    floor_v: float,
    direction_dim: int = 6,
):
    """Fit a deterministic, train-only centered PCA basis in 81-D."""
    samples = unit_deviation.double()
    valid = (vnorm > floor_v) & (mraw > 1e-4) & torch.isfinite(mraw)
    samples = samples[valid]
    if samples.shape[0] < direction_dim:
        raise RuntimeError("not enough valid unit-deviation samples for PCA")
    mean = samples.mean(0, keepdim=True)
    centered = samples - mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    projection = eigenvectors[:, -direction_dim:].T.contiguous()
    # Eigenvector signs are mathematically arbitrary. Canonicalize each row
    # so repeated fits serialize exactly the same signed basis.
    for row in range(projection.shape[0]):
        pivot = torch.argmax(projection[row].abs())
        if projection[row, pivot] < 0:
            projection[row].neg_()
    projection = projection.float()
    mean = mean.squeeze(0).float()
    if not torch.allclose(
        projection @ projection.T,
        torch.eye(direction_dim),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise RuntimeError("fitted PCA directions are not row-orthonormal")
    return projection, mean, int(samples.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-a-checkpoint", required=True, type=Path)
    parser.add_argument("--limit-per-task", type=int, default=256)
    parser.add_argument("--pixels-per-scale", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    dataset = AIOTrainDataset(
        cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["crop_size"], strict=True,
        split_manifest=cfg["split_manifest"], split="train",
    )
    indices, sample_counts = deterministic_indices(dataset, args.limit_per_task)
    sample_records = selected_sample_records(dataset, indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SRSCLite(**model_kwargs(cfg)).cuda().eval()
    checkpoint = torch.load(args.stage_a_checkpoint, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "split_manifest_sha256" not in checkpoint:
        raise RuntimeError("formal coordinate statistics require checkpoint provenance metadata")
    if checkpoint["config"].get("protocol") != cfg["protocol"]:
        raise RuntimeError("coordinate-statistics checkpoint protocol mismatch")
    expected_split_sha256 = file_sha256(Path(cfg["split_manifest"]))
    if checkpoint["split_manifest_sha256"] != expected_split_sha256:
        raise RuntimeError("coordinate-statistics checkpoint locked-split mismatch")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)

    provisional = SRSCCoordinateBuilder(tau_v=0.05, tau_e=0.05).cuda()
    vnorm_values, mraw_values, unit_deviation_values = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, batch in enumerate(loader, 1):
            x, gt = batch["degraded"].cuda(), batch["clean"].cuda()
            features, y1 = model._encode_coarse(x)
            outputs = provisional(
                x, y1, gt, [f.shape[-2:] for f in features], requested={"PCA_STATS"}
            )
            for output in outputs:
                vnorm_values.append(spatial_sample(output.vstar_norm, args.pixels_per_scale).flatten())
                mraw_values.append(spatial_sample(output.m_raw, args.pixels_per_scale).flatten())
                if output.unit_deviation is None:
                    raise RuntimeError("PCA_STATS did not return unit deviations")
                unit_deviation_values.append(
                    spatial_sample(output.unit_deviation, args.pixels_per_scale).transpose(0, 1)
                )
            if number % 25 == 0:
                print(f"THRESHOLD_PASS {number}/{len(loader)}", flush=True)
    vnorm = torch.cat(vnorm_values)
    floor_v = max(1e-4, 1e-3 * float(vnorm[vnorm > 0].median()))
    valid_v = vnorm[vnorm > floor_v]
    tau_v = max(1e-4, float(torch.quantile(valid_v, 0.10)))
    mraw = torch.cat(mraw_values)
    valid_m = mraw[(vnorm > floor_v) & (mraw > 1e-4) & torch.isfinite(mraw)]
    if valid_m.numel() == 0:
        raise RuntimeError("no valid positive transverse magnitudes for tau_e")
    tau_e = max(1e-4, float(torch.quantile(valid_m, 0.10)))

    unit_deviation = torch.cat(unit_deviation_values, dim=0).double()
    pca_projection, pca_mean, pca_observations = fit_pca_projection(
        unit_deviation,
        vnorm,
        mraw,
        floor_v,
        provisional.direction_dim,
    )

    builder = SRSCCoordinateBuilder(
        tau_v=tau_v,
        tau_e=tau_e,
        pca_projection=pca_projection,
        pca_mean=pca_mean,
    ).cuda()
    modes = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O12", "O13", "O15"]
    collected = {mode: [[] for _ in range(8)] for mode in modes}
    q_values, w_values = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, batch in enumerate(loader, 1):
            x, gt = batch["degraded"].cuda(), batch["clean"].cuda()
            features, y1 = model._encode_coarse(x)
            outputs = builder(x, y1, gt, [f.shape[-2:] for f in features])
            for output in outputs:
                q_values.append(spatial_sample(output.q, args.pixels_per_scale).flatten())
                w_values.append(spatial_sample(output.w_dir, args.pixels_per_scale).flatten())
            for mode in modes:
                for state in feedback_from_coordinates(outputs, mode):
                    sampled = spatial_sample(state, args.pixels_per_scale)
                    for channel in range(8):
                        collected[mode][channel].append(sampled[channel])
            if number % 25 == 0:
                print(f"NORMALIZATION_PASS {number}/{len(loader)}", flush=True)
    normalization = build_normalization_statistics(collected)

    out = Path(cfg["coordinate_stats"])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "protocol": cfg["protocol"],
        "seed": cfg["seed"],
        "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "producer_sha256": {
            "scripts/compute_coordinate_stats.py": file_sha256(Path(__file__)),
            "src/net/srsc_coordinates.py": file_sha256(
                ROOT / "src/net/srsc_coordinates.py"
            ),
        },
        "data_manifest_binding": prepared_data_manifest_binding(cfg["protocol"]),
        "projection_seed": 20260713,
        "residual_projection_seed": 20260714,
        "direction_projection_matrix": builder.P.detach().cpu().tolist(),
        "residual_projection_matrix": builder.Pr.detach().cpu().tolist(),
        "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
        "stage_a_checkpoint_sha256": file_sha256(args.stage_a_checkpoint),
        "split_manifest": str(Path(cfg["split_manifest"]).resolve()),
        "split_manifest_sha256": file_sha256(Path(cfg["split_manifest"])),
        "sample_policy": "SHA256-ranked unique train samples, fixed crop RNG",
        "sample_counts": sample_counts,
        "sample_identity_records": sample_records,
        "sample_identity_record_count": len(sample_records),
        "sample_identity_sha256": canonical_sha256(sample_records),
        "pixels_per_scale": args.pixels_per_scale,
        "vnorm_observations": int(vnorm.numel()),
        "vnorm_excluded_fraction": float((vnorm <= floor_v).float().mean()),
        "tau_v": tau_v,
        "mraw_observations": int(mraw.numel()),
        "tau_e": tau_e,
        "pca_fit_scope": "train-only valid unit transverse deviations; centered covariance eigendecomposition",
        "pca_observations": pca_observations,
        "pca_direction_matrix": pca_projection.tolist(),
        "pca_direction_mean": pca_mean.tolist(),
        "q_observations": int(torch.cat(q_values).numel()),
        "w_dir_observations": int(torch.cat(w_values).numel()),
        "q_quantiles": [float(x) for x in torch.quantile(torch.cat(q_values), torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))],
        "w_dir_quantiles": [float(x) for x in torch.quantile(torch.cat(w_values), torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))],
        "normalization": normalization,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
