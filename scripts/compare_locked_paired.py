#!/usr/bin/env python3
"""Paired PSNR/SSIM statistics for a frozen Stage-B locked validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.compare_paired import (
    degradation_task_macro_ci,
    stratified_macro_ci,
    summarize,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task", "name", "psnr", "ssim"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid or empty locked metric CSV: {path}")
    result = {}
    for row in rows:
        key = (row["task"], row["name"])
        if key in result:
            raise ValueError(f"duplicate paired key in {path}: {key}")
        values = {metric: float(row[metric]) for metric in ("psnr", "ssim")}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite PSNR/SSIM in {path}: {row}")
        result[key] = values
    return result


def read_psnr(path: Path) -> dict[tuple[str, str], float]:
    """Backward-compatible PSNR-only view of the now joint metric reader."""
    return {key: row["psnr"] for key, row in read_metrics(path).items()}


def compare_locked(baseline_path: Path, method_path: Path) -> dict:
    baseline = read_metrics(baseline_path)
    method = read_metrics(method_path)
    if baseline.keys() != method.keys():
        raise ValueError("locked paired keys differ between baseline and method")
    tasks = sorted({task for task, _ in baseline})
    task_psnr_values = {}
    task_ssim_values = {}
    task_psnr_summaries = {}
    task_ssim_summaries = {}
    all_psnr_values = []
    all_ssim_values = []
    psnr_rng = np.random.RandomState(20260713)
    ssim_rng = np.random.RandomState(20260718)
    for task in tasks:
        keys = sorted(key for key in baseline if key[0] == task)
        psnr_values = np.asarray([
            method[key]["psnr"] - baseline[key]["psnr"] for key in keys
        ])
        ssim_values = np.asarray([
            method[key]["ssim"] - baseline[key]["ssim"] for key in keys
        ])
        if not np.isfinite(psnr_values).all() or not np.isfinite(ssim_values).all():
            raise ValueError(f"non-finite paired delta for setting {task}")
        task_psnr_values[task] = psnr_values
        task_ssim_values[task] = ssim_values
        task_psnr_summaries[task] = summarize(psnr_values, psnr_rng)
        task_ssim_summaries[task] = summarize(ssim_values, ssim_rng)
        all_psnr_values.extend(psnr_values.tolist())
        all_ssim_values.extend(ssim_values.tolist())
    setting_psnr_mean = float(np.mean([
        values.mean() for values in task_psnr_values.values()
    ]))
    setting_ssim_mean = float(np.mean([
        values.mean() for values in task_ssim_values.values()
    ]))
    denoise = tuple(
        task for task in ("denoise15", "denoise25", "denoise50")
        if task in task_psnr_values
    )
    denoise_psnr_mean = (
        float(np.mean([task_psnr_values[task].mean() for task in denoise]))
        if denoise else None
    )
    denoise_ssim_mean = (
        float(np.mean([task_ssim_values[task].mean() for task in denoise]))
        if denoise else None
    )
    if len(denoise) == 3 and all(
        task in task_psnr_values for task in ("dehaze", "derain")
    ):
        degradation_psnr_macro = float(np.mean([
            task_psnr_values["dehaze"].mean(),
            task_psnr_values["derain"].mean(), denoise_psnr_mean,
        ]))
        degradation_ssim_macro = float(np.mean([
            task_ssim_values["dehaze"].mean(),
            task_ssim_values["derain"].mean(), denoise_ssim_mean,
        ]))
    else:
        degradation_psnr_macro = setting_psnr_mean
        degradation_ssim_macro = setting_ssim_mean
    return {
        "baseline": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "method": str(method_path.resolve()),
        "method_sha256": sha256_file(method_path),
        # Preserve every historical PSNR key consumed by the orchestrator.
        "tasks": task_psnr_summaries,
        "five_setting_mean_psnr_delta": setting_psnr_mean,
        "denoise_task_mean_psnr_delta": denoise_psnr_mean,
        "degradation_task_macro_psnr_delta": degradation_psnr_macro,
        "five_setting_bootstrap_95ci": stratified_macro_ci(
            list(task_psnr_values.values()), np.random.RandomState(20260714)
        ),
        "degradation_task_bootstrap_95ci": degradation_task_macro_ci(
            task_psnr_values, np.random.RandomState(20260715)
        ),
        "all_images_psnr_delta": summarize(
            np.asarray(all_psnr_values), np.random.RandomState(20260716)
        ),
        # SSIM is a parallel evidence channel and never changes PSNR gates.
        "tasks_ssim": task_ssim_summaries,
        "five_setting_mean_ssim_delta": setting_ssim_mean,
        "denoise_task_mean_ssim_delta": denoise_ssim_mean,
        "degradation_task_macro_ssim_delta": degradation_ssim_macro,
        "five_setting_ssim_bootstrap_95ci": stratified_macro_ci(
            list(task_ssim_values.values()), np.random.RandomState(20260719)
        ),
        "degradation_task_ssim_bootstrap_95ci": degradation_task_macro_ci(
            task_ssim_values, np.random.RandomState(20260720)
        ),
        "all_images_ssim_delta": summarize(
            np.asarray(all_ssim_values), np.random.RandomState(20260721)
        ),
        "bootstrap_draws": 10_000,
    }


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_locked(args.baseline, args.method)
    atomic_write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
