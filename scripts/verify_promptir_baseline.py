#!/usr/bin/env python3
"""Evaluate the immutable official PromptIR checkpoint under audited RGB metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_test_sets


def load_official_model(checkpoint: Path):
    source = ROOT / "upstream" / "PromptIR" / "net" / "model.py"
    spec = importlib.util.spec_from_file_location("official_promptir_model", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.PromptIR(decoder=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    state = {key.removeprefix("net."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def metric(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = (prediction - target).square().mean().item()
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    pred_np = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ssim = structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default=ROOT / "data", type=Path)
    parser.add_argument("--output", default=ROOT / "artifacts/metrics/promptir_official_aio3.csv", type=Path)
    parser.add_argument("--max-images-per-task", type=int, default=0, help="smoke only when nonzero")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = load_official_model(args.checkpoint).cuda().eval()
    datasets = build_test_sets(args.data_root, "aio3")
    rows = []
    with torch.inference_mode():
        for task, dataset in datasets.items():
            limit = len(dataset) if not args.max_images_per_task else min(len(dataset), args.max_images_per_task)
            for index in range(limit):
                item = dataset[index]
                degraded = item["degraded"].unsqueeze(0).cuda()
                target = item["clean"].unsqueeze(0).cuda()
                restored = model(degraded)
                psnr, ssim = metric(restored, target)
                row = {"task": task, "name": item["name"], "psnr": psnr, "ssim": ssim}
                rows.append(row)
                print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for task in datasets:
        subset = [x for x in rows if x["task"] == task]
        if subset:
            summary[task] = {
                "psnr": float(np.mean([x["psnr"] for x in subset])),
                "ssim": float(np.mean([x["ssim"] for x in subset])),
                "n": len(subset),
            }
    summary["five_setting_mean"] = {
        "psnr": float(np.mean([x["psnr"] for x in summary.values()])),
        "ssim": float(np.mean([x["ssim"] for x in summary.values()])),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
