#!/usr/bin/env python3
"""Paired per-image comparison with fixed 10k bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task", "name", "psnr", "ssim"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid or empty metric CSV: {path}")
    keys = [(row["task"], row["name"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate paired keys in {path}")
    for row in rows:
        if not np.isfinite(float(row["psnr"])) or not np.isfinite(float(row["ssim"])):
            raise ValueError(f"non-finite metric in {path}: {row}")
    return dict(zip(keys, rows))


def summarize(values: np.ndarray, rng: np.random.RandomState):
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "win_rate": float((values > 0).mean()),
        "worst_10pct_mean": float(np.sort(values)[:max(1, len(values) // 10)].mean()),
        "bootstrap_95ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def stratified_macro_ci(task_values: list[np.ndarray], rng: np.random.RandomState):
    """Bootstrap images within each task, then macro-average task means."""
    task_draws = []
    for values in task_values:
        task_draws.append(
            rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
        )
    macro_draws = np.stack(task_draws, axis=1).mean(axis=1)
    return [
        float(np.percentile(macro_draws, 2.5)),
        float(np.percentile(macro_draws, 97.5)),
    ]


def degradation_task_macro_ci(
    task_values: dict[str, np.ndarray], rng: np.random.RandomState
):
    draws = {
        task: rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
        for task, values in task_values.items()
    }
    denoise = ("denoise15", "denoise25", "denoise50")
    if all(task in draws for task in denoise) and all(
        task in draws for task in ("dehaze", "derain")
    ):
        denoise_draw = np.stack([draws[task] for task in denoise], axis=1).mean(axis=1)
        macro = np.stack([draws["dehaze"], draws["derain"], denoise_draw], axis=1).mean(axis=1)
    else:
        macro = np.stack(list(draws.values()), axis=1).mean(axis=1)
    return [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, method = read(args.baseline), read(args.method)
    if baseline.keys() != method.keys():
        missing_method = sorted(baseline.keys() - method.keys())[:20]
        missing_baseline = sorted(method.keys() - baseline.keys())[:20]
        raise ValueError(f"paired keys differ: method_missing={missing_method} baseline_missing={missing_baseline}")
    rng = np.random.RandomState(20260713)
    tasks = sorted({key[0] for key in baseline})
    result = {"tasks": {}, "baseline": str(args.baseline), "method": str(args.method)}
    task_psnr_means, task_ssim_means = [], []
    task_psnr_values, task_ssim_values = [], []
    task_psnr_map, task_ssim_map = {}, {}
    all_psnr = []
    for task in tasks:
        keys = sorted(key for key in baseline if key[0] == task)
        psnr = np.array([float(method[k]["psnr"]) - float(baseline[k]["psnr"]) for k in keys])
        ssim = np.array([float(method[k]["ssim"]) - float(baseline[k]["ssim"]) for k in keys])
        result["tasks"][task] = {"psnr_delta": summarize(psnr, rng), "ssim_delta": summarize(ssim, rng)}
        task_psnr_means.append(float(psnr.mean()))
        task_ssim_means.append(float(ssim.mean()))
        task_psnr_values.append(psnr)
        task_ssim_values.append(ssim)
        task_psnr_map[task] = psnr
        task_ssim_map[task] = ssim
        all_psnr.extend(psnr.tolist())
    result["macro_task_psnr_delta"] = float(np.mean(task_psnr_means))
    result["macro_task_ssim_delta"] = float(np.mean(task_ssim_means))
    result["macro_task_psnr_bootstrap_95ci"] = stratified_macro_ci(
        task_psnr_values, np.random.RandomState(20260714)
    )
    result["macro_task_ssim_bootstrap_95ci"] = stratified_macro_ci(
        task_ssim_values, np.random.RandomState(20260715)
    )
    denoise = ("denoise15", "denoise25", "denoise50")
    if all(task in task_psnr_map for task in denoise) and all(
        task in task_psnr_map for task in ("dehaze", "derain")
    ):
        denoise_psnr = float(np.mean([task_psnr_map[task].mean() for task in denoise]))
        denoise_ssim = float(np.mean([task_ssim_map[task].mean() for task in denoise]))
        degradation_psnr = float(np.mean([
            task_psnr_map["dehaze"].mean(), task_psnr_map["derain"].mean(), denoise_psnr,
        ]))
        degradation_ssim = float(np.mean([
            task_ssim_map["dehaze"].mean(), task_ssim_map["derain"].mean(), denoise_ssim,
        ]))
    else:
        denoise_psnr = float(np.mean([
            values.mean() for task, values in task_psnr_map.items()
            if task.startswith("denoise")
        ])) if any(task.startswith("denoise") for task in task_psnr_map) else None
        denoise_ssim = float(np.mean([
            values.mean() for task, values in task_ssim_map.items()
            if task.startswith("denoise")
        ])) if any(task.startswith("denoise") for task in task_ssim_map) else None
        degradation_psnr = result["macro_task_psnr_delta"]
        degradation_ssim = result["macro_task_ssim_delta"]
    result["aggregates"] = {
        "five_setting_mean_psnr_delta": result["macro_task_psnr_delta"],
        "five_setting_mean_ssim_delta": result["macro_task_ssim_delta"],
        "degradation_task_macro_psnr_delta": degradation_psnr,
        "degradation_task_macro_ssim_delta": degradation_ssim,
        "denoise_task_mean_psnr_delta": denoise_psnr,
        "denoise_task_mean_ssim_delta": denoise_ssim,
        "legacy_macro_task_semantics": "setting_macro; AIO-3 denoise severities are separate settings",
        "degradation_task_psnr_bootstrap_95ci": degradation_task_macro_ci(
            task_psnr_map, np.random.RandomState(20260716)
        ),
        "degradation_task_ssim_bootstrap_95ci": degradation_task_macro_ci(
            task_ssim_map, np.random.RandomState(20260717)
        ),
    }
    result["all_images_psnr_delta"] = summarize(np.asarray(all_psnr), rng)
    result["publication_thresholds"] = {
        "macro_psnr_ge_0.30": result["macro_task_psnr_delta"] >= 0.30,
        "every_task_psnr_ge_0.10": all(x >= 0.10 for x in task_psnr_means),
        "every_task_ssim_ge_minus_0.0001": all(x >= -0.0001 for x in task_ssim_means),
        "macro_task_psnr_ci_lower_gt_0": result["macro_task_psnr_bootstrap_95ci"][0] > 0.0,
    }
    result["publication_go_internal"] = all(result["publication_thresholds"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
