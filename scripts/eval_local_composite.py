#!/usr/bin/env python3
"""Evaluate a preregistered paired local-composite OOD degradation.

The standard Rain100L degraded image receives deterministic sigma-25 noise in
a feathered ellipse.  The clean Rain100L target is unchanged.  Results are
reported separately and are never mixed into the AIO-3/AIO-5 main tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_locked import build_model, configure_srsc_inference, metrics
from src.data import build_test_sets


GENERATION_SEED = 20260720
NOISE_SIGMA = 25.0


def elliptical_mask(height: int, width: int, dtype=torch.float32) -> torch.Tensor:
    """Fixed feathered ellipse: one locally degraded region, no hard seam."""
    yy = torch.linspace(0.0, 1.0, height, dtype=dtype).view(height, 1)
    xx = torch.linspace(0.0, 1.0, width, dtype=dtype).view(1, width)
    radius = torch.sqrt(((xx - 0.35) / 0.32).square() + ((yy - 0.50) / 0.42).square())
    transition = ((radius - 0.70) / 0.30).clamp(0.0, 1.0)
    return (0.5 * (1.0 + torch.cos(torch.pi * transition))).view(1, height, width)


def make_local_composite(degraded: torch.Tensor, name: str) -> torch.Tensor:
    """Add deterministic local noise without consuming global RNG state."""
    digest = hashlib.sha256(f"{name}:{GENERATION_SEED}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(degraded.shape, generator=generator, dtype=torch.float32)
    mask = elliptical_mask(degraded.shape[-2], degraded.shape[-1], torch.float32)
    return (degraded.float() + mask * noise * (NOISE_SIGMA / 255.0)).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["baseline", "baseline_matched", "srsc"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = build_model(cfg, args.model).cuda().eval()
    if args.model == "srsc":
        configure_srsc_inference(model, payload, cfg)
    model.load_state_dict(payload.get("model", payload), strict=True)
    rain = build_test_sets(cfg["data_root"], cfg["protocol"])["derain"]

    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in rain:
            composite = make_local_composite(item["degraded"], item["name"])
            x = composite.unsqueeze(0).cuda(non_blocking=True)
            gt = item["clean"].unsqueeze(0).cuda(non_blocking=True)
            prediction = model(x).float()
            psnr, ssim = metrics(prediction, gt)
            rows.append({
                "task": "rain_plus_local_noise25",
                "name": item["name"],
                "psnr": psnr,
                "ssim": ssim,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rain_plus_local_noise25": {
            "psnr": float(np.mean([row["psnr"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
            "n": len(rows),
        },
        "_meta": {
            "protocol": cfg["protocol"],
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "model": args.model,
            "source": "Rain100L official paired test inputs/targets",
            "degradation": "existing rain plus feathered local elliptical Gaussian noise",
            "noise_sigma": NOISE_SIGMA,
            "generation_seed": GENERATION_SEED,
            "selection_authority": "NONE_CONFIG_FROZEN_REPORT_ONLY_ROBUSTNESS_GATE",
            "included_in_standard_aio_average": False,
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
