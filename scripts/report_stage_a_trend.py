#!/usr/bin/env python3
"""Read-only, idempotent Stage-A locked-validation trend reporter.

This utility never imports the model or touches checkpoints.  It waits for a
requested validation epoch, compares it with the immediately preceding locked
validation record, then writes a machine-readable report and one audit block
to the user experiment log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/srsc_lite_v12")
DEFAULT_METRICS = ROOT / "artifacts/metrics/aio3_stage_a_coarse_seed1415926_locked_val.jsonl"
DEFAULT_LOG = Path("/root/aaa/v1.4.md")
SETTING_KEYS = ("dehaze", "denoise15", "denoise25", "denoise50", "derain")
TASKS = SETTING_KEYS + ("five_setting_mean_psnr", "three_task_macro_psnr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--user-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(enrich_aio3_aggregates(json.loads(line)))
    return sorted(rows, key=lambda row: (int(row["epoch"]), int(row["step"])))


def enrich_aio3_aggregates(row: dict) -> dict:
    """Name the legacy five-setting field and add the true task macro."""
    enriched = dict(row)
    missing = [key for key in SETTING_KEYS if key not in enriched]
    if missing:
        raise KeyError(f"AIO-3 locked-val row lacks settings: {missing}")
    setting_mean = sum(float(enriched[key]) for key in SETTING_KEYS) / len(SETTING_KEYS)
    legacy = float(enriched.get("macro_psnr", setting_mean))
    if abs(legacy - setting_mean) > 1e-9:
        raise ValueError(
            "legacy macro_psnr is not the expected five-setting mean: "
            f"legacy={legacy} recomputed={setting_mean}"
        )
    denoise_mean = sum(
        float(enriched[key]) for key in ("denoise15", "denoise25", "denoise50")
    ) / 3.0
    enriched["five_setting_mean_psnr"] = setting_mean
    enriched["denoise_task_mean_psnr"] = denoise_mean
    enriched["three_task_macro_psnr"] = (
        float(enriched["dehaze"]) + float(enriched["derain"]) + denoise_mean
    ) / 3.0
    enriched["macro_psnr_semantics"] = "legacy_alias_of_five_setting_mean_psnr"
    return enriched


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_already_logged(existing: str, label: str) -> bool:
    """Match only a dedicated marker or an actual legacy report heading.

    A narrative sentence may mention the label while describing a test.  Such
    text must not suppress the real result block.
    """
    marker = f"<!-- {label} -->"
    legacy_heading = re.compile(rf"^## [^\n]+ — {re.escape(label)}\s*$", re.MULTILINE)
    return marker in existing or legacy_heading.search(existing) is not None


def render_stage_a_report(rows: list[dict], target: dict) -> str:
    """Render the durable Stage-A report from committed locked-val rows only."""
    if not rows:
        raise ValueError("Stage-A report requires at least one locked-val row")
    best = max(rows, key=lambda row: float(row["five_setting_mean_psnr"]))
    status = "COMPLETE" if int(target["epoch"]) >= 240 else "IN_PROGRESS"
    best_checkpoint = (
        ROOT / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
        / f"val_epoch{int(best['epoch']):03d}_step{int(best['step']):07d}.pt"
    )
    lines = [
        "# Stage-A Report",
        "",
        f"Status: **{status}**  ",
        f"Latest committed locked validation: epoch `{int(target['epoch'])}`, "
        f"step `{int(target['step'])}`.  ",
        f"Best five-setting mean PSNR: **{float(best['five_setting_mean_psnr']):.10f} dB** "
        f"at epoch `{int(best['epoch'])}`, step `{int(best['step'])}`.  ",
        f"Best three-task macro PSNR: **{float(best['three_task_macro_psnr']):.10f} dB**.  ",
        f"Best checkpoint: `{best_checkpoint}`  ",
        f"Best checkpoint SHA256: `{sha256(best_checkpoint) or 'not_retained_in_top3'}`",
        "",
        "## Best checkpoint settings",
        "",
        "| Setting | PSNR (dB) |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {key} | {float(best[key]):.6f} |" for key in SETTING_KEYS
    )
    lines.extend([
        "",
        "This is the frozen coarse E+D1 locked-validation trajectory. It is not "
        "an official-test result and does not establish an SRSC or R2R gain; "
        "those claims remain gated on paired Stage-B/Stage-C experiments.",
        "",
    ])
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.target_epoch <= 0 or args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("target epoch, poll interval, and timeout must be positive")
    deadline = time.monotonic() + args.timeout_hours * 3600
    target = None
    rows: list[dict] = []
    while target is None:
        rows = load_rows(args.metrics)
        target = next((row for row in rows if int(row["epoch"]) == args.target_epoch), None)
        if target is not None:
            break
        if args.no_wait:
            print(json.dumps({"status": "not_ready", "target_epoch": args.target_epoch}))
            return 2
        if time.monotonic() >= deadline:
            print(json.dumps({"status": "timeout", "target_epoch": args.target_epoch}))
            return 3
        time.sleep(args.poll_seconds)

    previous_rows = [row for row in rows if int(row["epoch"]) < args.target_epoch]
    # Epoch 5 is the first preregistered locked-validation point.  It has no
    # legitimate earlier metric to compare against, so report absolute values
    # instead of inventing a delta or failing after hours of waiting.  Later
    # targets (for example epoch 10) retain the paired trend calculation.
    previous = previous_rows[-1] if previous_rows else None
    deltas = (
        {key: float(target[key]) - float(previous[key]) for key in TASKS}
        if previous is not None
        else None
    )
    checkpoint = (
        ROOT
        / "artifacts/checkpoints/aio3_stage_a_coarse_seed1415926"
        / f"val_epoch{args.target_epoch:03d}_step{int(target['step']):07d}.pt"
    )
    # The metric line is flushed immediately before update_top3 writes and
    # ranks the checkpoint.  A non-top3 target is deliberately removed in the
    # same update, so waiting for minutes cannot make it reappear and can stall
    # an otherwise valid trend chain.  Allow only the brief atomic-write
    # window; ``None`` then correctly means "metric retained, model not top3".
    checkpoint_deadline = time.monotonic() + 10
    while not checkpoint.is_file() and time.monotonic() < checkpoint_deadline:
        time.sleep(2)
    payload = {
        "status": "ready",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(args.metrics),
        "metrics_sha256": sha256(args.metrics),
        "previous": previous,
        "target": target,
        "delta": deltas,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "interpretation": "Stage-A coarse trend only; not an SRSC or R2R gain claim.",
    }
    out = ROOT / "artifacts/metrics" / f"stage_a_epoch{args.target_epoch:03d}_trend.json"
    temporary = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out)
    atomic_write_text(
        ROOT / "reports/STAGE_A_REPORT.md",
        render_stage_a_report(rows, target),
    )

    label = f"STAGE_A_TREND_EPOCH_{args.target_epoch:03d}"
    marker = f"<!-- {label} -->"
    existing = args.user_log.read_text() if args.user_log.is_file() else ""
    if not report_already_logged(existing, label):
        if previous is None:
            metric_text = "，".join(f"{key} {float(target[key]):.4f} dB" for key in TASKS)
            evidence_text = (
                f"- 首个locked validation：epoch {target['epoch']}，step {target['step']}。\n"
                f"- 逐项绝对值：{metric_text}。此前没有合法locked-val点，因此不报告趋势delta。\n"
            )
        else:
            delta_text = "，".join(f"{key} {value:+.4f} dB" for key, value in deltas.items())
            evidence_text = (
                f"- locked validation：epoch {previous['epoch']} → {target['epoch']}，"
                f"step {previous['step']} → {target['step']}。\n"
                f"- 逐项变化：{delta_text}。\n"
            )
        block = (
            f"\n{marker}\n## {payload['generated_utc'][:10]} — {label}\n\n"
            f"{evidence_text}"
            f"- 机器可读证据：`{out}`；checkpoint SHA256："
            f"`{payload['checkpoint_sha256'] or 'checkpoint_not_retained_in_top3'}`。\n"
            "- 该结果仅评价Stage-A粗修复收敛趋势，不代表SRSC相对R2R涨点；"
            "架构增益必须等待公平的Stage-B/Stage-C配对实验。\n"
        )
        with args.user_log.open("a") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
