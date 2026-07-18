#!/usr/bin/env python3
"""Profile immutable 256x256 inference cost for internal baselines."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import build_model


def profile(model, size: int, repeats: int):
    model = model.cuda().eval()
    x = torch.randn(1, 3, size, size, device="cuda")
    params = sum(p.numel() for p in model.parameters())
    flops = FlopCountAnalysis(model.cpu(), x.cpu()).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total()
    model = model.cuda(); x = x.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10): model(x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter(); model(x); torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "params": params,
        "flops": int(flops),
        "macs_convention": "fvcore counted operations; report as FLOPs, not relabeled MACs",
        "latency_ms_median": statistics.median(timings),
        "latency_ms_mean": statistics.mean(timings),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "input": [1, 3, size, size],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/protocol_aio3.yaml", type=Path)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    payload = {
        "clean_restormer_aio": profile(build_model(cfg, "baseline"), args.size, args.repeats),
        "clean_restormer_aio_matched": profile(build_model(cfg, "baseline_matched"), args.size, args.repeats),
        "srsc_lite": profile(build_model(cfg, "c"), args.size, args.repeats),
    }
    out = ROOT / "artifacts/stats/model_profile_256.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
