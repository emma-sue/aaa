#!/usr/bin/env python3
"""Export existing metric artifacts into one provenance-rich long CSV.

This is a read-only *artifact* scanner followed by one atomic CSV write.  It
never imports a dataset builder, deserializes a checkpoint, evaluates a model,
or changes an official-test lock.  Official-test summaries are admitted only
when a frozen candidate manifest, COMPLETE consumption ledger, completion
record, CSV and JSON summary form one hash-bound transaction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "protocol",
    "scope",
    "run_name",
    "stage",
    "model_kind",
    "feedback",
    "seed",
    "epoch",
    "step",
    "task",
    "metric",
    "value",
    "n",
    "source_path",
    "source_sha256",
    "manifest_path",
    "manifest_sha256",
    "ledger_path",
    "ledger_sha256",
)
EXPECTED_TASKS = {
    "aio3": ("dehaze", "derain", "denoise15", "denoise25", "denoise50"),
    "aio5": ("dehaze", "derain", "denoise25", "deblur", "lowlight"),
}
OFFICIAL_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunMetadata:
    protocol: str
    run_name: str
    stage: str
    model_kind: str
    feedback: str
    seed: int | str
    epoch: int | str = ""
    step: int | str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--output",
        help="must remain under <root>/artifacts/metrics; defaults to metrics_long.csv",
    )
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bytes_with_sha256(path: str | Path) -> tuple[bytes, str]:
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RuntimeError(f"{label} is not a lowercase SHA256 digest")
    return value


def _json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _infer_protocol(*values: object) -> str:
    for value in values:
        if value in EXPECTED_TASKS:
            return str(value)
        match = re.search(r"(?:^|[/_-])(aio[35])(?:[/_-]|$)", str(value).lower())
        if match:
            return match.group(1)
    return ""


def _infer_feedback(run_name: str, explicit: object = None) -> str:
    if isinstance(explicit, str) and re.fullmatch(r"O\d+", explicit):
        return explicit
    match = re.search(r"(?:^|_)o(\d+)(?:_|$)", run_name.lower())
    return f"O{match.group(1)}" if match else ""


def _infer_seed(run_name: str, explicit: object = None) -> int | str:
    if isinstance(explicit, int):
        return explicit
    if isinstance(explicit, str) and explicit.isdigit():
        return int(explicit)
    for pattern in (r"seed(\d+)", r"(?:^|_)s(\d+)(?:_|$)"):
        match = re.search(pattern, run_name.lower())
        if match:
            return int(match.group(1))
    return ""


def _infer_stage(run_name: str, explicit: object = None) -> str:
    if isinstance(explicit, str) and explicit:
        return explicit
    lowered = run_name.lower()
    if "_stage_a_" in lowered or "_coarse_" in lowered:
        return "a"
    if "_oracle_" in lowered:
        return "b_oracle"
    if "_predicted_" in lowered:
        return "b_predicted"
    if "_stage_c_" in lowered:
        return "c"
    if "baseline" in lowered and "finetune" in lowered:
        return "baseline_ft"
    if "baseline" in lowered:
        return "baseline"
    return "unknown"


def _infer_model_kind(run_name: str, explicit: object = None) -> str:
    if explicit in {"srsc", "baseline", "baseline_matched"}:
        return str(explicit)
    lowered = run_name.lower()
    if "baseline_matched" in lowered:
        return "baseline_matched"
    if "baseline" in lowered:
        return "baseline"
    return "srsc"


def _checkpoint_epoch_step(checkpoint: str | Path | None) -> tuple[int | str, int | str]:
    if not checkpoint:
        return "", ""
    match = re.search(r"val_epoch(\d+)_step(\d+)\.pt$", Path(checkpoint).name)
    if not match:
        return "", ""
    return int(match.group(1)), int(match.group(2))


def _run_contract(root: Path, run_name: str) -> dict:
    path = root / "artifacts/checkpoints" / run_name / "run_contract.json"
    return _json_object(path, "run contract") if path.is_file() else {}


def infer_run_metadata(
    root: Path,
    *,
    run_name: str,
    protocol: object = None,
    model_kind: object = None,
    checkpoint: str | Path | None = None,
) -> RunMetadata:
    contract = _run_contract(root, run_name)
    stable_args = contract.get("args") if isinstance(contract.get("args"), dict) else {}
    effective = (
        contract.get("effective_config")
        if isinstance(contract.get("effective_config"), dict)
        else {}
    )
    explicit_stage = contract.get("stage", stable_args.get("stage"))
    explicit_feedback = contract.get("feedback", stable_args.get("feedback"))
    explicit_seed = effective.get("seed")
    inferred_protocol = _infer_protocol(
        protocol, effective.get("protocol"), run_name, checkpoint
    )
    epoch, step = _checkpoint_epoch_step(checkpoint)
    return RunMetadata(
        protocol=inferred_protocol,
        run_name=run_name,
        stage=_infer_stage(run_name, explicit_stage),
        model_kind=_infer_model_kind(run_name, model_kind),
        feedback=_infer_feedback(run_name, explicit_feedback),
        seed=_infer_seed(run_name, explicit_seed),
        epoch=epoch,
        step=step,
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is boolean, not a metric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} is not finite")
    return numeric


def summary_measurements(summary: dict, protocol: str) -> list[tuple[str, str, float, int | str]]:
    """Normalize train-ledger and eval summary schemas without double aliases."""
    if protocol not in EXPECTED_TASKS:
        raise RuntimeError(f"cannot normalize metrics for unknown protocol {protocol!r}")
    measurements: dict[tuple[str, str], tuple[float, int | str]] = {}

    def add(task: str, metric: str, value: object, n: object = "") -> None:
        key = (str(task), str(metric))
        numeric = _finite_number(value, f"{task}.{metric}")
        count: int | str = ""
        if n != "":
            count = int(n)
            if count <= 0:
                raise RuntimeError(f"{task}.{metric} has invalid n={count}")
        previous = measurements.get(key)
        if previous is not None and previous != (numeric, count):
            raise RuntimeError(f"conflicting duplicate metric {task}.{metric}")
        measurements[key] = (numeric, count)

    missing = []
    for task in EXPECTED_TASKS[protocol]:
        item = summary.get(task)
        if isinstance(item, dict):
            for metric in ("psnr", "ssim"):
                if metric in item:
                    add(task, metric, item[metric], item.get("n", ""))
        elif item is not None:
            add(task, "psnr", item)
        else:
            missing.append(task)
    if missing:
        raise RuntimeError(f"metric summary lacks required tasks: {missing}")

    setting_ssim = summary.get("setting_ssim")
    if isinstance(setting_ssim, dict):
        for task in EXPECTED_TASKS[protocol]:
            if task not in setting_ssim:
                raise RuntimeError(f"setting_ssim lacks task {task}")
            add(task, "ssim", setting_ssim[task])

    aggregates = summary.get("aggregates")
    if isinstance(aggregates, dict):
        for task, item in aggregates.items():
            if not isinstance(item, dict):
                continue
            for metric in ("psnr", "ssim"):
                if metric in item:
                    add(str(task), metric, item[metric])

    explicit_aggregates = {
        "five_setting_mean_psnr": ("five_setting_mean", "psnr"),
        "three_task_macro_psnr": ("task_macro", "psnr"),
        "denoise_task_mean_psnr": ("denoise_task_mean", "psnr"),
        "five_setting_mean_ssim": ("five_setting_mean", "ssim"),
        "three_task_macro_ssim": ("task_macro", "ssim"),
        "denoise_task_mean_ssim": ("denoise_task_mean", "ssim"),
    }
    for key, (task, metric) in explicit_aggregates.items():
        if key in summary:
            add(task, metric, summary[key])
    # Historical train ledgers used macro_psnr as a five-setting alias.
    if "five_setting_mean_psnr" not in summary and "macro_psnr" in summary:
        add("five_setting_mean", "psnr", summary["macro_psnr"])
    if "five_setting_mean_ssim" not in summary and "macro_ssim" in summary:
        add("five_setting_mean", "ssim", summary["macro_ssim"])

    return [
        (task, metric, value, n)
        for (task, metric), (value, n) in sorted(measurements.items())
    ]


def _row(
    *,
    metadata: RunMetadata,
    scope: str,
    epoch: int | str,
    step: int | str,
    task: str,
    metric: str,
    value: float,
    n: int | str,
    source: Path,
    source_sha256: str,
    manifest: Path | None = None,
    manifest_sha256: str = "",
    ledger: Path | None = None,
    ledger_sha256: str = "",
) -> dict:
    return {
        "protocol": metadata.protocol,
        "scope": scope,
        "run_name": metadata.run_name,
        "stage": metadata.stage,
        "model_kind": metadata.model_kind,
        "feedback": metadata.feedback,
        "seed": metadata.seed,
        "epoch": epoch,
        "step": step,
        "task": task,
        "metric": metric,
        "value": value,
        "n": n,
        "source_path": _resolved(source),
        "source_sha256": source_sha256,
        "manifest_path": _resolved(manifest) if manifest else "",
        "manifest_sha256": manifest_sha256,
        "ledger_path": _resolved(ledger) if ledger else "",
        "ledger_sha256": ledger_sha256,
    }


def collect_locked_jsonl(root: Path) -> list[dict]:
    metric_dir = root / "artifacts/metrics"
    rows: list[dict] = []
    for path in sorted(metric_dir.glob("**/*_locked_val.jsonl")):
        run_name = path.name[: -len("_locked_val.jsonl")]
        source_bytes, source_sha = read_bytes_with_sha256(path)
        records = []
        for line_number, line in enumerate(
            source_bytes.decode("utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL record {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"locked-val record is not an object: {path}:{line_number}")
            records.append(record)
        if not records:
            raise RuntimeError(f"locked-val ledger is empty: {path}")
        keys = [(record.get("epoch"), record.get("step")) for record in records]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate locked-val epoch/step in {path}")
        for record in records:
            protocol = _infer_protocol(record.get("protocol"), run_name)
            metadata = infer_run_metadata(root, run_name=run_name, protocol=protocol)
            epoch = int(record["epoch"])
            step = int(record["step"])
            for task, metric, value, n in summary_measurements(record, protocol):
                rows.append(_row(
                    metadata=metadata,
                    scope="locked_val",
                    epoch=epoch,
                    step=step,
                    task=task,
                    metric=metric,
                    value=value,
                    n=n,
                    source=path,
                    source_sha256=source_sha,
                ))
    return rows


def collect_locked_summary_json(root: Path) -> list[dict]:
    """Collect eval_locked summaries; ordinary trend/comparison JSON is ignored."""
    metric_dir = root / "artifacts/metrics"
    rows: list[dict] = []
    for path in sorted(metric_dir.glob("**/*.json")):
        source_bytes, source_sha = read_bytes_with_sha256(path)
        try:
            summary = json.loads(source_bytes)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid metric JSON: {path}") from error
        if not isinstance(summary, dict):
            continue
        meta = summary.get("_meta")
        if not isinstance(meta, dict) or meta.get("split") != "locked_val":
            continue
        checkpoint = meta.get("checkpoint")
        run_name = Path(checkpoint).parent.name if checkpoint else path.stem
        protocol = _infer_protocol(meta.get("protocol"), run_name)
        metadata = infer_run_metadata(
            root,
            run_name=run_name,
            protocol=protocol,
            model_kind=meta.get("model"),
            checkpoint=checkpoint,
        )
        for task, metric, value, n in summary_measurements(summary, protocol):
            rows.append(_row(
                metadata=metadata,
                scope="locked_val",
                epoch=metadata.epoch,
                step=metadata.step,
                task=task,
                metric=metric,
                value=value,
                n=n,
                source=path,
                source_sha256=source_sha,
            ))
    return rows


def _canonical_stored_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is missing")
    path = Path(value)
    if value != _resolved(path):
        raise RuntimeError(f"{label} is not absolute and canonical")
    return path


def validate_official_transaction(
    root: Path,
    ledger_path: Path,
    ledger: dict,
    entry: dict,
) -> tuple[dict, Path, Path, str, str]:
    """Validate existing evidence only; never open official images or a model."""
    protocol = ledger.get("protocol")
    if protocol not in EXPECTED_TASKS:
        raise RuntimeError(f"invalid official ledger protocol: {protocol!r}")
    if ledger.get("schema_version") != OFFICIAL_SCHEMA_VERSION:
        raise RuntimeError("unsupported official consumption ledger schema")
    manifest_path = _canonical_stored_path(ledger.get("manifest_path"), "manifest_path")
    if not _is_within(manifest_path, root / "artifacts/manifests") or not manifest_path.is_file():
        raise RuntimeError("official manifest is missing or outside artifacts/manifests")
    manifest_sha = sha256_file(manifest_path)
    if ledger.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("official ledger manifest SHA256 mismatch")
    config_path = _canonical_stored_path(
        _json_object(manifest_path, "official manifest").get("config_path"),
        "manifest config_path",
    )
    if not config_path.is_file() or ledger.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("official ledger config SHA256 mismatch")

    manifest = _json_object(manifest_path, "official manifest")
    if not all((
        manifest.get("schema_version") == OFFICIAL_SCHEMA_VERSION,
        manifest.get("status") == "FROZEN",
        manifest.get("protocol") == protocol,
        manifest.get("config_sha256") == ledger.get("config_sha256"),
    )):
        raise RuntimeError("official frozen manifest metadata mismatch")
    if not isinstance(entry, dict) or entry.get("status") != "COMPLETE":
        raise RuntimeError("official consumption is not COMPLETE")
    candidate_id = entry.get("candidate_id")
    candidates = [
        item for item in manifest.get("candidates", [])
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    if len(candidates) != 1:
        raise RuntimeError("official consumption does not select one frozen candidate")
    candidate = candidates[0]
    model_kind = entry.get("model")
    checkpoint = _canonical_stored_path(entry.get("checkpoint_path"), "checkpoint_path")
    output = _canonical_stored_path(entry.get("output"), "official output")
    record_path = _canonical_stored_path(entry.get("record"), "official record")
    summary_path = output.with_suffix(".json")
    if not all((
        checkpoint.is_file(),
        output.is_file(),
        summary_path.is_file(),
        record_path.is_file(),
        _is_within(checkpoint, root / "artifacts/checkpoints"),
        _is_within(output, root / "artifacts/metrics"),
        _is_within(record_path, root / "artifacts/manifests"),
    )):
        raise RuntimeError("official transaction artifact is missing or outside project artifacts")
    checkpoint_sha = _require_sha(entry.get("checkpoint_sha256"), "checkpoint_sha256")
    if not all((
        candidate.get("model") == model_kind,
        candidate.get("checkpoint_path") == _resolved(checkpoint),
        candidate.get("checkpoint_sha256") == checkpoint_sha,
        _resolved(output) in candidate.get("output_paths", []),
    )):
        raise RuntimeError("official ledger entry differs from frozen candidate")

    output_sha = sha256_file(output)
    summary_sha = sha256_file(summary_path)
    record_sha = sha256_file(record_path)
    if not all((
        entry.get("csv_sha256") == output_sha,
        entry.get("summary_sha256") == summary_sha,
        entry.get("record_sha256") == record_sha,
    )):
        raise RuntimeError("official consumption artifact SHA256 mismatch")
    record = _json_object(record_path, "official completion record")
    summary = _json_object(summary_path, "official summary")
    meta = summary.get("_meta")
    if not isinstance(meta, dict):
        raise RuntimeError("official summary lacks _meta")
    with output.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if not csv_rows:
        raise RuntimeError("official CSV has no rows")
    expected_common = (
        record.get("status") == "COMPLETE",
        record.get("protocol") == protocol,
        record.get("model") == model_kind,
        record.get("candidate_id") == candidate_id,
        record.get("official_manifest") == _resolved(manifest_path),
        record.get("official_manifest_sha256") == manifest_sha,
        record.get("official_ledger") == _resolved(ledger_path),
        record.get("checkpoint_path", record.get("checkpoint")) in {None, _resolved(checkpoint)},
        record.get("checkpoint_sha256") == checkpoint_sha,
        record.get("paper_comparable_full_image") is True,
        record.get("rows") == len(csv_rows),
        record.get("csv") == _resolved(output),
        record.get("csv_sha256") == output_sha,
        record.get("summary") == _resolved(summary_path),
        record.get("summary_sha256") == summary_sha,
        meta.get("split") == "official_test",
        meta.get("protocol") == protocol,
        meta.get("model") == model_kind,
        meta.get("candidate_id") == candidate_id,
        meta.get("official_manifest") == _resolved(manifest_path),
        meta.get("official_manifest_sha256") == manifest_sha,
        meta.get("official_ledger") == _resolved(ledger_path),
        meta.get("checkpoint") == _resolved(checkpoint),
        meta.get("checkpoint_sha256") == checkpoint_sha,
        meta.get("paper_comparable_full_image") is True,
    )
    if not all(expected_common):
        raise RuntimeError("official completion record/summary evidence mismatch")
    return summary, output, manifest_path, manifest_sha, model_kind


def collect_official(root: Path) -> list[dict]:
    manifest_dir = root / "artifacts/manifests"
    rows: list[dict] = []
    for ledger_path in sorted(manifest_dir.glob("official_test_*_consumption.json")):
        ledger = _json_object(ledger_path, "official consumption ledger")
        ledger_sha = sha256_file(ledger_path)
        consumptions = ledger.get("consumptions")
        if not isinstance(consumptions, list):
            raise RuntimeError(f"official ledger has invalid consumptions: {ledger_path}")
        candidate_ids = [item.get("candidate_id") for item in consumptions if isinstance(item, dict)]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(f"official ledger has duplicate candidate consumptions: {ledger_path}")
        # STARTED/FAILED are consumed but do not contain publishable metrics.
        for entry in consumptions:
            if not isinstance(entry, dict) or entry.get("status") != "COMPLETE":
                continue
            summary, source, manifest, manifest_sha, model_kind = (
                validate_official_transaction(root, ledger_path, ledger, entry)
            )
            checkpoint = entry["checkpoint_path"]
            run_name = Path(checkpoint).parent.name
            protocol = str(ledger["protocol"])
            metadata = infer_run_metadata(
                root,
                run_name=run_name,
                protocol=protocol,
                model_kind=model_kind,
                checkpoint=checkpoint,
            )
            source_sha = sha256_file(source.with_suffix(".json"))
            for task, metric, value, n in summary_measurements(summary, protocol):
                rows.append(_row(
                    metadata=metadata,
                    scope="official_test",
                    epoch=metadata.epoch,
                    step=metadata.step,
                    task=task,
                    metric=metric,
                    value=value,
                    n=n,
                    source=source.with_suffix(".json"),
                    source_sha256=source_sha,
                    manifest=manifest,
                    manifest_sha256=manifest_sha,
                    ledger=ledger_path,
                    ledger_sha256=ledger_sha,
                ))
    return rows


def collect_rows(root: Path) -> list[dict]:
    rows = (
        collect_locked_jsonl(root)
        + collect_locked_summary_json(root)
        + collect_official(root)
    )
    def numeric(value: object) -> int:
        return int(value) if value not in {None, ""} else -1

    rows.sort(key=lambda row: (
        str(row["protocol"]), str(row["scope"]), str(row["run_name"]),
        str(row["stage"]), str(row["model_kind"]), str(row["feedback"]),
        numeric(row["seed"]), numeric(row["epoch"]), numeric(row["step"]),
        str(row["task"]), str(row["metric"]), str(row["source_path"]),
    ))
    identities = [
        tuple(row[field] for field in (
            "protocol", "scope", "run_name", "epoch", "step", "task", "metric",
            "source_path",
        ))
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate normalized metric rows")
    return rows


def atomic_write_long_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def export_metrics(root: Path, output: Path | None = None) -> tuple[Path, list[dict]]:
    root = root.resolve()
    metric_dir = root / "artifacts/metrics"
    output = (output or metric_dir / "metrics_long.csv").resolve()
    if not _is_within(output, metric_dir):
        raise ValueError("metrics_long output must remain under artifacts/metrics")
    rows = collect_rows(root)
    atomic_write_long_csv(output, rows)
    return output, rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else None
    path, rows = export_metrics(root, output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["scope"]] = counts.get(row["scope"], 0) + 1
    print(json.dumps({
        "output": str(path),
        "rows": len(rows),
        "scope_rows": counts,
        "sha256": sha256_file(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
