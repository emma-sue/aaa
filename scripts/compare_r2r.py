#!/usr/bin/env python3
"""Compare one frozen official-test summary against the transcribed R2R row."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_TASKS = {
    "aio3": ("dehaze", "derain", "denoise15", "denoise25", "denoise50"),
    "aio5": ("dehaze", "derain", "denoise25", "deblur", "lowlight"),
}
NUMERIC_EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference(payload: dict, protocol: str) -> tuple[dict, tuple[str, ...]]:
    if payload.get("metric_protocol") != "full RGB images":
        raise ValueError("R2R comparison is locked to full RGB images")
    source = Path(payload["source"])
    expected_sha = payload.get("source_sha256")
    if not source.is_file() or not expected_sha:
        raise FileNotFoundError("R2R source PDF or its frozen SHA256 is missing")
    actual_sha = sha256(source)
    if actual_sha != expected_sha:
        raise ValueError(f"R2R source PDF SHA256 drift: {actual_sha} != {expected_sha}")
    table_key = "table1_aio3" if protocol == "aio3" else "table2_aio5"
    reference = payload[table_key]
    tasks = EXPECTED_TASKS[protocol]
    if set(reference) != set(tasks) | {"reported_average"}:
        raise ValueError(f"unexpected {table_key} task set: {sorted(reference)}")
    mean_psnr = sum(float(reference[key]["psnr"]) for key in tasks) / len(tasks)
    mean_ssim = sum(float(reference[key]["ssim"]) for key in tasks) / len(tasks)
    reported = reference["reported_average"]
    if abs(mean_psnr - float(reported["psnr"])) > 0.011:
        raise ValueError("R2R reported PSNR average is inconsistent with its task rows")
    if abs(mean_ssim - float(reported["ssim"])) > 0.0011:
        raise ValueError("R2R reported SSIM average is inconsistent with its task rows")
    return reference, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aio3", "aio5"], required=True)
    parser.add_argument("--method-summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method_summary.read_text())
    reference_payload = json.loads(args.reference.read_text())
    reference, tasks = validate_reference(reference_payload, args.protocol)
    missing = [task for task in tasks if task not in method]
    if missing:
        raise KeyError(f"official summary lacks R2R-comparable tasks: {missing}")
    extras = sorted(set(method) - set(tasks) - {"macro", "aggregates", "metadata", "_meta"})
    deltas = {}
    for task in tasks:
        deltas[task] = {
            "psnr": float(method[task]["psnr"] - reference[task]["psnr"]),
            "ssim": float(method[task]["ssim"] - reference[task]["ssim"]),
            "method": {"psnr": method[task]["psnr"], "ssim": method[task]["ssim"]},
            "r2r": reference[task],
            "absolute_psnr_target_r2r_plus_0.30": float(reference[task]["psnr"] + 0.30),
            "psnr_ge_plus_0.30": bool(
                method[task]["psnr"] - reference[task]["psnr"] >= 0.30 - NUMERIC_EPS
            ),
            "ssim_positive": bool(method[task]["ssim"] - reference[task]["ssim"] > 0),
        }
    psnr_target_count = sum(item["psnr"] >= 0.30 - NUMERIC_EPS for item in deltas.values())
    ssim_positive_count = sum(item["ssim"] > 0 for item in deltas.values())
    result = {
        "protocol": args.protocol,
        "reference_title": reference_payload["title"],
        "reference_pdf_sha256": reference_payload["source_sha256"],
        "metric_protocol": reference_payload["metric_protocol"],
        "reference_table": "Table 1" if args.protocol == "aio3" else "Table 2",
        "tasks_in_locked_order": list(tasks),
        "ignored_non_r2r_fields_in_method_summary": extras,
        "deltas": deltas,
        "macro_psnr_delta_vs_r2r": sum(x["psnr"] for x in deltas.values()) / len(deltas),
        "macro_ssim_delta_vs_r2r": sum(x["ssim"] for x in deltas.values()) / len(deltas),
        "tasks_psnr_ge_plus_0.30": psnr_target_count,
        "tasks_ssim_positive": ssim_positive_count,
        "allowed_tasks_below_target": 1,
        "required_tasks_psnr_ge_plus_0.30": len(tasks) - 1,
        "required_tasks_ssim_positive": len(tasks) - 1,
        "user_target_met": psnr_target_count >= len(tasks) - 1 and ssim_positive_count >= len(tasks) - 1,
        "note": "Exact frozen full-RGB official protocol comparison. At most one task may miss +0.30 dB and positive SSIM; this user target is separate from the stricter preregistered internal publication gate."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
