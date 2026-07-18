#!/usr/bin/env python3
"""Resumable preregistered AIO pipeline through the scientific pilot gates."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from scripts import train_baseline_hybrid_ddp as hybrid_ddp
from scripts import train_stage_a_capacity_hybrid_ddp as capacity_hybrid_ddp
from scripts.eval_locked import (
    EXPECTED_TASKS,
    OFFICIAL_MANIFEST_SCHEMA_VERSION,
    freeze_official_data_manifest,
    official_artifacts_complete,
    official_evaluation_code_hashes,
)
from scripts.runtime_accounting import read_runtime_sidecar
from scripts.stage_b_runtime import (
    REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG,
    StageBRuntimeBundle,
    assert_no_runtime_worker_override,
    assert_stage_b_cublas_environment,
    ensure_stage_b_runtime_bundle,
    runtime_identity_for_config,
)

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[1]
EXEC_UNBLOCKED = Path(__file__).resolve().parent / "exec_unblocked.py"
USER_LOG = Path("/root/aaa/v1.4.md")
CONTRACTS = [
    Path("/root/aaa/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md"),
    Path("/root/ResearchStudio/ideaspark_run/end-to-end-restoration-state-feedback/srsc_lite_v1_2_reassessment.md"),
    ROOT / "reports/AUDIT.md",
    ROOT / "reports/ARCHITECTURE.md",
    ROOT / "reports/BASELINE_PARITY.md",
    ROOT / "reports/AUTOSOTA_STRATEGY_LIBRARY.md",
    ROOT / "reports/PRE_STAGE_B_RELOAD_REQUIRED.md",
    ROOT / "reports/PROTOCOL_CORRECTION_CENTER_CROP.md",
    ROOT / "reports/PROTOCOL_CORRECTION_AUGMENTATION.md",
    ROOT / "reports/PROTOCOL_AMENDMENT_AIO3_BATCH_MIGRATION.md",
    ROOT / "reports/CACHE_CONTRACT_REVISION_V1.md",
    Path("/root/ResearchStudio/ResearchStudio-Idea/skills/idea_spark/SKILL.md"),
    Path("/root/.codex/skills/autosota/SKILL.md"),
    ROOT / "src/net/clean_restormer_aio.py",
    ROOT / "src/net/restormer_blocks.py",
    ROOT / "src/net/feedback_controls.py",
    ROOT / "src/net/srsc_lite.py",
    ROOT / "src/net/srsc_coordinates.py",
    ROOT / "src/losses/objectives.py",
    ROOT / "src/data/aio_dataset.py",
    ROOT / "scripts/orchestrate.py",
    ROOT / "scripts/train.py",
    ROOT / "scripts/train_stage_a_ddp.py",
    ROOT / "scripts/runtime_accounting.py",
    ROOT / "scripts/stage_b_runtime.py",
    ROOT / "scripts/preflight_stage_b_runtime.py",
    ROOT / "scripts/monitor_runtime_accounting.py",
    ROOT / "scripts/exec_unblocked.py",
    ROOT / "scripts/compute_coordinate_stats.py",
    ROOT / "scripts/cache_stage_a_outputs.py",
    ROOT / "scripts/prepare_data.py",
    ROOT / "scripts/create_locked_split.py",
    ROOT / "scripts/train_baseline_hybrid.py",
    ROOT / "scripts/train_baseline_hybrid_ddp.py",
    ROOT / "scripts/train_stage_a_capacity_hybrid_ddp.py",
    ROOT / "scripts/verify_promptir_baseline.py",
    ROOT / "scripts/eval_locked.py",
    ROOT / "scripts/eval_local_composite.py",
    ROOT / "scripts/eval_feedback_diagnostics.py",
    ROOT / "scripts/export_metrics_long.py",
    ROOT / "scripts/compare_paired.py",
    ROOT / "scripts/compare_locked_paired.py",
    ROOT / "scripts/compare_r2r.py",
    ROOT / "scripts/launch_when_data_ready.sh",
    ROOT / "scripts/launch_aio3_stage_a_4x4090.sh",
    ROOT / "scripts/launch_aio5_stage_a_4x4090.sh",
    ROOT / "scripts/reload_pipeline_at_checkpoint.sh",
    ROOT / "scripts/watchdog.sh",
    ROOT / "scripts/verify_stage_a_checkpoint.py",
    ROOT / "artifacts/reference/r2r_cvpr2026_tables.json",
    ROOT / "artifacts/manifests/aio3.json",
    ROOT / "artifacts/manifests/aio5.json",
    ROOT / "artifacts/manifests/locked_split_aio3.json",
    ROOT / "artifacts/manifests/locked_split_aio5.json",
    ROOT / "configs/protocol_aio3.yaml",
    ROOT / "configs/protocol_aio3_baseline_b120.yaml",
    ROOT / "configs/protocol_aio3_baseline_hybrid.yaml",
    ROOT / "configs/protocol_aio3_10_10_hybrid.yaml",
    ROOT / "configs/protocol_aio5.yaml",
    ROOT / "configs/stage_b_aio3.yaml",
    ROOT / "configs/stage_b_aio5.yaml",
    ROOT / "configs/stage_c_aio3.yaml",
    ROOT / "configs/stage_c_aio5.yaml",
    ROOT / "configs/protocol_aio3_10_10.yaml",
    ROOT / "configs/stage_b_aio3_10_10.yaml",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def note(message: str):
    line = f"- `{utc()}` {message}"
    print(line, flush=True)
    with USER_LOG.open("a") as handle:
        handle.write(line + "\n")
    (ROOT / "RUNNING_STATUS.md").write_text(
        "# Running Status\n\n" + line + "\n\n"
        + "Status command: `bash /root/autodl-tmp/srsc_lite_v12/scripts/status.sh`.\n"
    )


DECISION_SCHEMA_DEFAULTS = {
    "promptir_parity": "INCOMPLETE",
    "stage_a": "INCOMPLETE",
    "oracle_sign": "INCOMPLETE",
    "oracle_direction": "INCOMPLETE",
    "predicted_srsc": "INCOMPLETE",
    "scientific_go": "INCOMPLETE",
    "publication_go": "INCOMPLETE",
    "residual_code_control": "INCOMPLETE",
    "selected_model": None,
    "per_task_deltas": {},
    "params": {},
    "macs": {},
    "gpu_hours": None,
    "blocking_issues": [],
    "next_command": "",
}


# These are the frozen terminal transactions produced by the protocol-specific
# four-GPU Stage-A launchers.  Keeping the exact update count here prevents a
# manually restarted orchestrator from accepting a merely epoch-shaped
# checkpoint and silently changing the scientific handoff.
STAGE_A_HANDOFF_CONTRACTS = {
    "aio3": {
        "expected_epoch": 240,
        "expected_step": 330_500,
        "expected_world_size": 4,
        "expected_global_effective_batch": 120,
        "expected_per_gpu_batch": 30,
        "expected_accumulation": 1,
        "expected_workers_per_rank": 8,
        "expected_backend": "nccl",
        # AIO-3 legitimately migrated from the registered one-GPU prefix at
        # epoch 55, so it has no single fresh-only origin assertion.
        "expected_training_origin": None,
    },
    "aio5": {
        "expected_epoch": 240,
        "expected_step": 427_440,
        "expected_world_size": 4,
        "expected_global_effective_batch": 120,
        "expected_per_gpu_batch": 30,
        "expected_accumulation": 1,
        "expected_workers_per_rank": 6,
        "expected_backend": "nccl",
        "expected_training_origin": "fresh",
    },
}


def collect_training_gpu_hours() -> tuple[
    float | None, dict[str, float], dict[str, dict]
]:
    """Aggregate crash-resumable counters without hiding estimate origin."""
    checkpoint_root = ROOT / "artifacts/checkpoints"
    by_run: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    if not checkpoint_root.is_dir():
        return None, by_run, metadata
    for path in sorted(checkpoint_root.glob("*/runtime_accounting.json")):
        payload = read_runtime_sidecar(path)
        run_name = path.parent.name
        if payload["run_name"] != run_name:
            raise RuntimeError(
                f"runtime-accounting run mismatch: directory={run_name!r} "
                f"payload={payload['run_name']!r}"
            )
        by_run[run_name] = float(payload["accumulated_gpu_seconds"]) / 3600.0
        origin = payload.get(
            "accounting_origin", "EMBEDDED_TRAINER_MONOTONIC_COUNTER"
        )
        metadata[run_name] = {
            "scope": payload["scope"],
            "origin": origin,
            "is_estimate": "ESTIMATE" in str(origin).upper(),
            "last_snapshot_unix": payload.get("last_snapshot_unix"),
        }
    if not by_run:
        return None, by_run, metadata
    return float(sum(by_run.values())), by_run, metadata


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one report artifact and fsync file and directory.

    Temporary files live beside the destination so ``os.replace`` cannot
    cross filesystems.  A failed write leaves the previously committed file
    intact and removes its temporary artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_named_sha256_sidecar(path: Path, expected_name: str) -> str:
    """Read the standard ``<digest>  <name>`` checksum-file format."""
    fields = path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != expected_name:
        raise ValueError(f"invalid SHA256 sidecar format: {path}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid SHA256 digest in sidecar: {path}")
    return digest


def _schema_complete_decision(decision: dict) -> dict:
    """Fill the frozen reporting schema without importing stale gate results."""
    canonical_path = ROOT / "reports/decision.json"
    try:
        prior = json.loads(canonical_path.read_text()) if canonical_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        prior = {}

    # These fields describe protocol-independent audited facts or accounting.
    # Gate and model-selection fields must never leak from a prior protocol.
    for key in (
        "promptir_parity", "params", "macs", "gpu_hours",
        "gpu_hours_by_run", "gpu_hours_metadata",
    ):
        if key not in decision and key in prior:
            decision[key] = prior[key]
    gpu_hours, gpu_hours_by_run, gpu_hours_metadata = collect_training_gpu_hours()
    if gpu_hours is not None:
        decision["gpu_hours"] = gpu_hours
        decision["gpu_hours_by_run"] = gpu_hours_by_run
        decision["gpu_hours_metadata"] = gpu_hours_metadata
    for key, default in DECISION_SCHEMA_DEFAULTS.items():
        if key not in decision:
            # JSON round-tripping gives each decision independent list/dict
            # defaults and also catches non-serializable default regressions.
            decision[key] = json.loads(json.dumps(default))
    decision["updated_at_utc"] = utc()
    return decision


def _fmt_metric(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def _render_oracle_report(decision: dict) -> str:
    stage = decision.get("stage", "NOT_STARTED")
    pilot_only = decision.get("pilot_authority") == "PIPELINE_ONLY_NO_SCIENTIFIC_GATE"
    lines = [
        "# Stage-B Oracle Report",
        "",
        f"Protocol: `{decision.get('protocol', 'unknown')}`  ",
        f"Decision revision: `{decision['decision_revision_sha256']}`  ",
        f"Pipeline stage: `{stage}`  ",
        f"Signed-progress gate: **{decision['oracle_sign']}**  ",
        f"Direction gate: **{decision['oracle_direction']}**  ",
        f"Overall scientific status: **{decision['scientific_go']}**",
        "",
    ]
    if pilot_only and "oracle" not in decision:
        lines.extend([
            "Pilot values, if present in `decision.json`, are plumbing-only and have no scientific authority.",
            "Formal locked-validation Oracle results have not yet completed.",
            "",
        ])
    if "oracle" in decision:
        lines.extend([
            "## Formal locked-validation evidence",
            "",
            "| Comparison | Macro PSNR delta (dB) |",
            "|---|---:|",
            f"| signed p vs U/D | {_fmt_metric(decision.get('oracle_sign_delta'))} |",
            f"| p+m+d vs p+m | {_fmt_metric(decision.get('oracle_direction_delta'))} |",
            f"| SRSC vs magnitude proxy | {_fmt_metric(decision.get('oracle_vs_magnitude_delta'))} |",
            f"| SRSC vs matched residual code | {_fmt_metric(decision.get('oracle_vs_residual_code_delta'))} |",
            "",
            f"Paired-CI gate: **{_fmt_metric(decision.get('oracle_paired_ci_go'))}**.  ",
            f"Matched residual-code outcome: **{decision['residual_code_control']}**.",
            "",
        ])
    if "oracle_controls" in decision:
        lines.extend([
            "## Independently retrained controls",
            "",
            f"- Signed-vs-absolute control delta: {_fmt_metric(decision.get('oracle_sign_abs_control_delta'))} dB.",
            f"- Direction negative-control delta: {_fmt_metric(decision.get('oracle_direction_control_delta'))} dB.",
            f"- Random-noise control delta: {_fmt_metric(decision.get('oracle_random_noise_control_delta'))} dB.",
            f"- Formal Oracle decision: **{'GO' if decision.get('oracle_go') else 'NO_GO'}**.",
            "",
        ])
    lines.extend([
        "Oracle results use GT-derived feedback only to test information value; they are not deployable results.",
        "Official-test feedback is not used for these decisions.",
        "",
    ])
    return "\n".join(lines)


def _render_predicted_report(decision: dict) -> str:
    lines = [
        "# Stage-B Predicted Report",
        "",
        f"Protocol: `{decision.get('protocol', 'unknown')}`  ",
        f"Decision revision: `{decision['decision_revision_sha256']}`  ",
        f"Pipeline stage: `{decision.get('stage', 'NOT_STARTED')}`  ",
        f"Predicted SRSC gate: **{decision['predicted_srsc']}**  ",
        f"Scientific gate: **{decision['scientific_go']}**",
        "",
    ]
    if "predicted" not in decision:
        lines.extend([
            "Formal predicted feedback has not completed or was not authorized by the Oracle gate.",
            "AutoSOTA and Stage-C remain unauthorized unless `scientific_go` is `GO`.",
            "",
        ])
    else:
        lines.extend([
            "## Formal locked-validation evidence",
            "",
            "| Comparison | Macro PSNR delta (dB) |",
            "|---|---:|",
            f"| P7 vs P6 | {_fmt_metric(decision.get('predicted_direction_delta'))} |",
            f"| P7 vs magnitude/U-D controls | {_fmt_metric(decision.get('predicted_vs_controls_delta'))} |",
            f"| P7 vs matched predicted residual code | {_fmt_metric(decision.get('predicted_vs_residual_delta'))} |",
            f"| P7 vs deterministic O1/O2 | {_fmt_metric(decision.get('predicted_vs_deterministic_o1_o2_delta'))} |",
            "",
            f"Oracle information-gain capture ratio: {_fmt_metric(decision.get('predicted_capture_ratio'))}.  ",
            f"Three-seed direction consistency: **{_fmt_metric(decision.get('predicted_three_seed_direction_consistent'))}**.  ",
            f"Matched residual-code outcome: **{decision['residual_code_control']}**.",
            "",
        ])
    return "\n".join(lines)


def _render_final_report(decision: dict) -> str:
    capacity = decision.get("capacity_robustness_go")
    capacity_text = "INCOMPLETE" if capacity is None else ("PASS" if capacity else "FAIL")
    denoise = decision.get(
        "joint_denoise_direction_guard",
        decision.get("predicted_denoise_direction_guard", "INCOMPLETE"),
    )
    not_dehaze = decision.get(
        "joint_direction_not_dehaze_only",
        decision.get("predicted_direction_not_dehaze_only", "INCOMPLETE"),
    )
    official_done = "official_outputs" in decision
    threshold_text = (
        decision["publication_go"] if official_done
        else "INCOMPLETE — official test remains frozen"
    )
    lines = [
        "# Final Decision",
        "",
        f"Protocol: `{decision.get('protocol', 'unknown')}`  ",
        f"Decision revision: `{decision['decision_revision_sha256']}`  ",
        f"Current pipeline stage: `{decision.get('stage', 'NOT_STARTED')}`  ",
        f"Scientific status: **{decision['scientific_go']}**  ",
        f"Publication status: **{decision['publication_go']}**  ",
        f"Selected model: **{decision['selected_model'] or 'INCOMPLETE'}**",
        "",
        "## Frozen questions",
        "",
        f"1. PromptIR/evaluation parity: **{decision['promptir_parity']}**.",
        "2. Prompt/MoE removal: governed by `AUDIT.md` and the executable removal tests; this decision writer does not infer it from metrics.",
        f"3. Two-stage capacity confound: **{capacity_text}**.",
        f"4. Signed progress versus unsigned U/D: **{decision['oracle_sign']}**.",
        f"5. Independent direction contribution: **{decision['oracle_direction']}**; predicted SRSC: **{decision['predicted_srsc']}**.",
        "6. Oracle GT bandwidth: Oracle arms are diagnostic only; O12/O13/O14 are excluded from deployable claims.",
        f"7. Predicted SRSC versus matched residual code: **{decision['residual_code_control']}**.",
        f"8. Denoise guardrails: **{_fmt_metric(denoise)}**.",
        f"9. Not driven only by dehaze: **{_fmt_metric(not_dehaze)}**.",
        f"10. Current selection: **{decision['selected_model'] or 'INCOMPLETE'}**.",
        f"11. Stage-C authorization: **{'YES' if decision['scientific_go'] == 'GO' else 'NO'}**.",
        f"12. Next real command/action: `{decision['next_command'] or 'not recorded'}`.",
        f"13. +0.30 dB average / +0.10 dB each-setting / SSIM guardrail: **{threshold_text}**.",
        f"14. Scope: scientific=`{decision['scientific_go']}`, publication=`{decision['publication_go']}`.",
        "",
        "## Accounting",
        "",
        f"- GPU-hours: {_fmt_metric(decision['gpu_hours'])}.",
        "- Accounting provenance: "
        + _fmt_metric(decision.get("gpu_hours_metadata", {}))
        + ".",
        f"- Blocking issues: {', '.join(map(str, decision['blocking_issues'])) or 'none recorded'}.",
        "",
        "Only evidence already committed to the synchronized decision JSON is summarized here.",
        "",
    ]
    return "\n".join(lines)


def persist_decision(decision: dict, decision_path: Path) -> None:
    """Commit synchronized protocol/global decisions and derived reports."""
    _schema_complete_decision(decision)
    decision.pop("decision_revision_sha256", None)
    revision_payload = json.dumps(
        decision, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    decision["decision_revision_sha256"] = hashlib.sha256(revision_payload).hexdigest()
    payload = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    oracle_report = _render_oracle_report(decision)
    predicted_report = _render_predicted_report(decision)
    final_report = _render_final_report(decision)

    # Every destination is independently crash-safe.  Derived reports embed
    # the revision hash; the canonical decision is replaced last and acts as
    # the bundle commit marker after the protocol-specific decision is ready.
    _atomic_write_text(ROOT / "reports/STAGE_B_ORACLE_REPORT.md", oracle_report)
    _atomic_write_text(ROOT / "reports/STAGE_B_PREDICTED_REPORT.md", predicted_report)
    _atomic_write_text(ROOT / "reports/FINAL_DECISION.md", final_report)
    _atomic_write_text(decision_path, payload)
    canonical = ROOT / "reports/decision.json"
    if decision_path.resolve() != canonical.resolve():
        _atomic_write_text(canonical, payload)


def local_composite_artifacts_complete(
    protocol: str, checkpoint: Path, model_kind: str, output: Path
) -> bool:
    """Validate that a cached local-composite result belongs to this model."""
    summary_path = output.with_suffix(".json")
    if not output.is_file() or not summary_path.is_file() or not checkpoint.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        meta = summary["_meta"]
        data_rows = [value for key, value in summary.items() if key != "_meta"]
        with output.open(newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        required_columns = {"task", "name", "psnr", "ssim"}
        metric_keys = [(row.get("task"), row.get("name")) for row in metric_rows]
        expected_rows = sum(int(row.get("n", 0)) for row in data_rows)
        return (
            output.stat().st_size > 0
            and bool(data_rows)
            and all(int(row.get("n", 0)) > 0 for row in data_rows)
            and len(metric_rows) == expected_rows
            and len(metric_keys) == len(set(metric_keys))
            and all(required_columns <= row.keys() for row in metric_rows)
            and all(
                math.isfinite(float(row[column]))
                for row in metric_rows for column in ("psnr", "ssim")
            )
            and meta.get("protocol") == protocol
            and meta.get("checkpoint") == str(checkpoint.resolve())
            and meta.get("checkpoint_sha256")
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            and meta.get("model") == model_kind
            and meta.get("generation_seed") == 20260720
            and meta.get("included_in_standard_aio_average") is False
        )
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        csv.Error, json.JSONDecodeError,
    ):
        return False


def freeze_official_candidate_manifest(
    protocol: str,
    config_path: Path,
    candidates: list[dict],
) -> Path:
    """Freeze the complete final candidate set before any official read.

    Re-entry is allowed only when the byte-independent JSON payload is exactly
    the same.  A different checkpoint, output path, model kind, or config must
    use a new preregistered protocol run rather than mutate this manifest.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not candidates:
        raise ValueError("official candidate manifest cannot be empty")
    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict) or cfg.get("protocol") != protocol:
        raise ValueError("official config protocol mismatch")
    coordinate_stats = Path(cfg.get("coordinate_stats", "")).resolve()
    if not coordinate_stats.is_file():
        raise FileNotFoundError(coordinate_stats)
    official_data_manifest = freeze_official_data_manifest(cfg)
    normalized = []
    candidate_ids = set()
    claimed_outputs = set()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"duplicate/empty official candidate id: {candidate_id!r}")
        candidate_ids.add(candidate_id)
        model_kind = str(candidate["model"])
        if model_kind not in {"srsc", "baseline", "baseline_matched"}:
            raise ValueError(f"unsupported official model kind: {model_kind}")
        checkpoint = Path(candidate["checkpoint_path"]).resolve()
        output = Path(candidate["output_path"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if output.suffix.lower() != ".csv" or str(output) in claimed_outputs:
            raise ValueError(f"invalid/duplicate official output: {output}")
        claimed_outputs.add(str(output))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_args = payload.get("args") or {}
        run_contract = checkpoint.parent / "run_contract.json"
        if not run_contract.is_file():
            raise RuntimeError(
                f"official checkpoint has no immutable run contract: {checkpoint}"
            )
        run_contract_sha = _sha256_file(run_contract)
        if checkpoint_args.get("run_contract_sha256") != run_contract_sha:
            raise RuntimeError(
                f"official checkpoint/run-contract binding mismatch: {checkpoint}"
            )
        checkpoint_contract = {
            "stage": checkpoint_args.get("stage"),
            "feedback": checkpoint_args.get("feedback"),
            "run_contract_sha256": run_contract_sha,
            "config_sha256": payload.get("config_sha256"),
            "split_manifest_sha256": payload.get("split_manifest_sha256"),
        }
        if (
            checkpoint_contract["config_sha256"] != _sha256_file(config_path)
            or not _diagnostic_sha256(
                checkpoint_contract["split_manifest_sha256"]
            )
        ):
            raise RuntimeError(
                f"official checkpoint config/split provenance mismatch: {checkpoint}"
            )
        normalized.append({
            "candidate_id": candidate_id,
            "model": model_kind,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "run_contract_path": str(run_contract.resolve()),
            "run_contract_sha256": run_contract_sha,
            "checkpoint_contract": checkpoint_contract,
            "output_paths": [str(output)],
        })
    payload = {
        "schema_version": OFFICIAL_MANIFEST_SCHEMA_VERSION,
        "status": "FROZEN",
        "protocol": protocol,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "coordinate_stats_path": str(coordinate_stats),
        "coordinate_stats_sha256": _sha256_file(coordinate_stats),
        "official_data_manifest_path": str(official_data_manifest.resolve()),
        "official_data_manifest_sha256": _sha256_file(official_data_manifest),
        "evaluation_code_sha256": official_evaluation_code_hashes(),
        "candidates": normalized,
    }
    manifest = ROOT / "artifacts/manifests" / f"official_candidates_{protocol}.json"
    if manifest.is_file():
        try:
            existing = json.loads(manifest.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"corrupt frozen official manifest: {manifest}") from error
        if existing != payload:
            raise RuntimeError(
                "frozen official candidate manifest drift; archive the protocol "
                "run and preregister a new manifest"
            )
    else:
        _atomic_write_text(
            manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return manifest


FEEDBACK_DIAGNOSTIC_CODE_RELATIVE_PATHS = (
    "scripts/eval_feedback_diagnostics.py",
    "scripts/train.py",
    "src/net/srsc_lite.py",
    "src/net/srsc_coordinates.py",
    "src/net/feedback_controls.py",
    "src/data/aio_dataset.py",
)


def feedback_diagnostic_code_hashes() -> dict[str, str]:
    """Return the exact implementation closure recorded by the diagnostic."""
    return {
        relative: _sha256_file(ROOT / relative)
        for relative in FEEDBACK_DIAGNOSTIC_CODE_RELATIVE_PATHS
    }


def _diagnostic_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _diagnostic_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _diagnostic_finite(value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a diagnostic number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"invalid diagnostic number: {value!r}")
    return number


def _diagnostic_count(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a diagnostic count")
    count = int(value)
    if count < 0 or float(value) != float(count):
        raise ValueError(f"invalid diagnostic count: {value!r}")
    return count


def _diagnostic_vector(value: object, *, minimum: float | None = None) -> list[float]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("diagnostic channel vector must contain exactly eight values")
    return [_diagnostic_finite(item, minimum=minimum) for item in value]


def _diagnostic_close(actual: object, expected: float) -> bool:
    try:
        return math.isclose(
            _diagnostic_finite(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_feedback_metric_block(row: dict, entropy_bins: int) -> int:
    if not isinstance(row, dict):
        raise ValueError("diagnostic metric block must be an object")
    vector_count = _diagnostic_count(row["vector_count"])
    scalar_count = _diagnostic_count(row["scalar_count"])
    if vector_count <= 0 or scalar_count != vector_count * 8:
        raise ValueError("diagnostic vector/scalar counts are inconsistent")
    channel_mae = _diagnostic_vector(row["channel_mae"], minimum=0.0)
    channel_rmse = _diagnostic_vector(row["channel_rmse"], minimum=0.0)
    if not _diagnostic_close(row["scalar_mae"], sum(channel_mae) / 8.0):
        raise ValueError("scalar MAE is inconsistent with channel MAE")
    if not _diagnostic_close(row["scalar_rmse"], sum(channel_rmse) / 8.0):
        raise ValueError("scalar RMSE is inconsistent with channel RMSE")

    cosine = row["cosine"]
    if not isinstance(cosine, dict):
        raise ValueError("diagnostic cosine block must be an object")
    target_valid = _diagnostic_count(cosine["target_valid_count"])
    target_zero = _diagnostic_count(cosine["target_zero_count"])
    prediction_zero = _diagnostic_count(cosine["prediction_zero_count"])
    both_nonzero = _diagnostic_count(cosine["both_nonzero_count"])
    if target_valid + target_zero != vector_count:
        raise ValueError("target-valid/zero counts do not cover all vectors")
    if prediction_zero > vector_count or both_nonzero > target_valid:
        raise ValueError("diagnostic cosine counts are inconsistent")
    if both_nonzero > vector_count - prediction_zero:
        raise ValueError("both-nonzero count exceeds nonzero predictions")
    if not _diagnostic_close(
        cosine["target_valid_fraction"], target_valid / vector_count
    ) or not _diagnostic_close(
        cosine["prediction_zero_fraction"], prediction_zero / vector_count
    ):
        raise ValueError("diagnostic cosine fractions are inconsistent")
    for key, count in (
        ("mean_over_target_valid_zero_prediction_is_zero", target_valid),
        ("mean_over_both_nonzero_diagnostic", both_nonzero),
    ):
        value = cosine[key]
        if count == 0:
            if value is not None:
                raise ValueError(f"{key} must be null when its denominator is zero")
        else:
            number = _diagnostic_finite(value)
            if number < -1.0 or number > 1.0:
                raise ValueError(f"{key} is outside the cosine range")

    distributions = row["distribution"]
    if not isinstance(distributions, dict) or set(distributions) != {
        "prediction", "target", "error"
    }:
        raise ValueError("diagnostic distributions must be prediction/target/error")
    maximum_entropy = math.log2(entropy_bins)
    for distribution in distributions.values():
        if not isinstance(distribution, dict):
            raise ValueError("diagnostic distribution must be an object")
        _diagnostic_vector(distribution["channel_mean"])
        variances = _diagnostic_vector(
            distribution["channel_population_variance"], minimum=0.0
        )
        entropies = _diagnostic_vector(
            distribution["channel_marginal_entropy_bits"], minimum=0.0
        )
        if any(value > maximum_entropy + 1e-9 for value in entropies):
            raise ValueError("marginal entropy exceeds the configured bin ceiling")
        if not _diagnostic_close(
            distribution["mean_channel_population_variance"], sum(variances) / 8.0
        ):
            raise ValueError("mean channel variance is inconsistent")
        mean_entropy = sum(entropies) / 8.0
        if not _diagnostic_close(
            distribution["mean_channel_marginal_entropy_bits"], mean_entropy
        ) or not _diagnostic_close(
            distribution["mean_channel_normalized_entropy"],
            mean_entropy / maximum_entropy,
        ):
            raise ValueError("mean/normalized channel entropy is inconsistent")
    return vector_count


def _validate_feedback_scale_macro(scale_macro: dict, scales: dict) -> None:
    if not isinstance(scale_macro, dict) or scale_macro.get("definition") != (
        "unweighted arithmetic mean of the four native-scale metrics"
    ):
        raise ValueError("invalid diagnostic scale-macro definition")

    def mean(path):
        values = []
        for row in scales.values():
            value = path(row)
            if value is not None:
                values.append(_diagnostic_finite(value))
        return sum(values) / len(values) if values else None

    expected = {
        "scalar_mae": mean(lambda row: row["scalar_mae"]),
        "scalar_rmse": mean(lambda row: row["scalar_rmse"]),
        "cosine_target_valid": mean(
            lambda row: row["cosine"][
                "mean_over_target_valid_zero_prediction_is_zero"
            ]
        ),
        "prediction_mean_channel_variance": mean(
            lambda row: row["distribution"]["prediction"][
                "mean_channel_population_variance"
            ]
        ),
        "target_mean_channel_variance": mean(
            lambda row: row["distribution"]["target"][
                "mean_channel_population_variance"
            ]
        ),
        "prediction_mean_channel_entropy_bits": mean(
            lambda row: row["distribution"]["prediction"][
                "mean_channel_marginal_entropy_bits"
            ]
        ),
        "target_mean_channel_entropy_bits": mean(
            lambda row: row["distribution"]["target"][
                "mean_channel_marginal_entropy_bits"
            ]
        ),
    }
    if set(scale_macro) != {"definition", *expected.keys()}:
        raise ValueError("diagnostic scale-macro fields are incomplete or unexpected")
    for key, value in expected.items():
        if value is None:
            if scale_macro[key] is not None:
                raise ValueError(f"scale-macro {key} must be null")
        elif not _diagnostic_close(scale_macro[key], value):
            raise ValueError(f"scale-macro {key} is inconsistent")


def feedback_diagnostics_complete(
    output: Path,
    *,
    checkpoint: Path,
    config: Path,
    feedback: str,
) -> bool:
    if not output.is_file() or not checkpoint.is_file() or not config.is_file():
        return False
    try:
        payload = json.loads(output.read_text())
        cfg = yaml.safe_load(config.read_text())
        if not isinstance(cfg, dict):
            return False
        provenance = payload["provenance"]
        pooled = payload["pooled_aggregate"]
        scales = payload["per_scale"]
        selection = payload["selection"]
        metric_parameters = payload["metric_parameters"]
        spatial_validity = payload["spatial_validity"]
        if not (
            payload.get("schema") == "srsc.predicted_feedback_diagnostics.v1"
            and payload.get("status") == "COMPLETE"
            and payload.get("split") == "locked_val"
            and payload.get("protocol") == cfg.get("protocol")
            and payload.get("feedback_interface_mode") == feedback
            and payload.get("feedback_supervision_mode")
            == ("O7" if feedback in {"O9", "O10", "O11"} else feedback)
            and selection.get("complete_split") is True
            and selection.get("policy") == "all preregistered locked-validation images"
            and _diagnostic_sha256(selection.get("selection_sha256"))
            and metric_parameters.get("channels") == 8
            and metric_parameters.get("entropy_clipping") is True
            and spatial_validity.get("scale_divisors")
            == {"S1": 1, "S2": 2, "S3": 4, "S4": 8}
            and spatial_validity.get("complete_source_block_required") is True
            and spatial_validity.get("model_padding_included") is False
            and set(scales) == {"S1", "S2", "S3", "S4"}
            and provenance.get("checkpoint") == str(checkpoint.resolve())
            and provenance.get("checkpoint_sha256") == _sha256_file(checkpoint)
            and provenance.get("config") == str(config.resolve())
            and provenance.get("config_sha256") == _sha256_file(config)
            and provenance.get("feedback_interface_mode") == feedback
            and provenance.get("feedback_supervision_mode")
            == payload.get("feedback_supervision_mode")
            and provenance.get("checkpoint_stage") in {"b_predicted", "c"}
        ):
            return False

        zero_epsilon = _diagnostic_finite(
            metric_parameters["zero_epsilon"], minimum=0.0
        )
        entropy_bins = _diagnostic_count(metric_parameters["entropy_bins"])
        entropy_range = _diagnostic_finite(
            metric_parameters["entropy_range"], minimum=0.0
        )
        if zero_epsilon <= 0.0 or entropy_bins < 2 or entropy_range <= 0.0:
            return False
        scale_vector_count = sum(
            _validate_feedback_metric_block(row, entropy_bins)
            for row in scales.values()
        )
        if _validate_feedback_metric_block(pooled, entropy_bins) != scale_vector_count:
            return False
        _validate_feedback_scale_macro(payload["scale_macro"], scales)

        image_count = _diagnostic_count(payload["image_count"])
        task_counts = payload["image_count_per_task"]
        if (
            image_count <= 0
            or not isinstance(task_counts, dict)
            or not task_counts
            or sum(_diagnostic_count(value) for value in task_counts.values())
            != image_count
        ):
            return False

        split_path = _diagnostic_config_path(cfg["split_manifest"])
        stats_path = _diagnostic_config_path(cfg["coordinate_stats"])
        run_contract_path = checkpoint.resolve().parent / "run_contract.json"
        for path in (split_path, stats_path, run_contract_path):
            if not path.is_file():
                return False
        split_sha256 = _sha256_file(split_path)
        stats_sha256 = _sha256_file(stats_path)
        run_contract_sha256 = _sha256_file(run_contract_path)
        if not (
            provenance.get("split_manifest") == str(split_path)
            and provenance.get("split_manifest_sha256") == split_sha256
            and provenance.get("coordinate_stats") == str(stats_path)
            and provenance.get("coordinate_stats_sha256") == stats_sha256
            and provenance.get("run_contract") == str(run_contract_path)
            and provenance.get("run_contract_sha256") == run_contract_sha256
        ):
            return False

        stats = json.loads(stats_path.read_text())
        run_contract = json.loads(run_contract_path.read_text())
        expected_training_code = current_train_code_hashes()
        if not (
            stats.get("protocol") == cfg.get("protocol")
            and stats.get("split_manifest_sha256") == split_sha256
            and run_contract.get("feedback") == feedback
            and run_contract.get("stage") == provenance.get("checkpoint_stage")
            and run_contract.get("config_sha256") == _sha256_file(config)
            and run_contract.get("split_manifest_sha256") == split_sha256
            and run_contract.get("coordinate_stats_sha256") == stats_sha256
            and run_contract.get("code_sha256") == expected_training_code
            and payload.get("code_sha256") == feedback_diagnostic_code_hashes()
        ):
            return False

        runtime = payload["runtime"]
        if not isinstance(runtime, dict):
            return False
        _diagnostic_finite(runtime["elapsed_seconds"], minimum=0.0)
        return True
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        json.JSONDecodeError, yaml.YAMLError,
    ):
        return False


def metric_task_keys(protocol: str, metric: dict) -> list[str]:
    """Return only the protocol-defined finite numeric task fields.

    Metric records also contain transaction provenance strings.  A task
    allowlist prevents those fields from entering arithmetic in scientific
    gate calculations.
    """
    if protocol not in EXPECTED_TASKS:
        raise ValueError(f"unsupported metric protocol: {protocol}")
    expected = list(EXPECTED_TASKS[protocol])
    missing = [key for key in expected if key not in metric]
    if missing:
        raise KeyError(f"metric record missing protocol tasks: {missing}")
    for key in expected:
        value = metric[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"task metric {key} must be numeric, got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"task metric {key} must be finite")
    return expected


def paired_task_median_guard(protocol: str, comparison: dict) -> bool:
    """Implement the frozen Oracle O7-vs-O2/O3 per-task median rule."""
    if protocol not in EXPECTED_TASKS:
        raise ValueError(f"unsupported paired protocol: {protocol}")
    tasks = comparison.get("tasks")
    if not isinstance(tasks, dict):
        raise TypeError("paired comparison requires task summaries")
    expected = EXPECTED_TASKS[protocol]
    if set(tasks) != set(expected):
        raise KeyError(
            "paired comparison task mismatch: "
            f"actual={sorted(tasks)} expected={sorted(expected)}"
        )
    medians = []
    for task in expected:
        value = tasks[task].get("median")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"paired median for {task} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"paired median for {task} must be finite")
        medians.append(value)
    return all(value >= 0.0 for value in medians)


def residual_code_outcome(delta: float | None) -> str:
    if delta is None:
        return "INCOMPLETE"
    if delta > 0:
        return "SRSC_BETTER"
    if delta < 0:
        return "RESIDUAL_BETTER"
    return "TIE"


def select_predicted_model(predicted: dict, predicted_go: bool) -> str:
    if predicted_go:
        return "SRSC_LITE"
    scores = {
        "UD_V1": predicted["O4"]["macro_psnr"],
        "SIGNED_PROGRESS_ONLY": predicted["O5"]["macro_psnr"],
        "RESIDUAL_CODE": predicted["O12"]["macro_psnr"],
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= predicted["O7"]["macro_psnr"] else "NO_GO"


def contract_hashes(
    extra_contracts: tuple[Path, ...] | list[Path] = (),
) -> dict[str, str]:
    hashes = {}
    for path in (*CONTRACTS, *tuple(extra_contracts)):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"contract missing: {path}")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def review_contract(
    phase: str,
    *,
    extra_contracts: tuple[Path, ...] | list[Path] = (),
):
    hashes = contract_hashes(extra_contracts)
    prompt = CONTRACTS[0].read_text()
    required = ["fixed K=2", "no recurrence", "no MoE", "SCIENTIFIC_GO", "PUBLICATION_GO"]
    missing = [token for token in required if token not in prompt]
    if missing:
        raise RuntimeError(f"implementation contract lost required tokens: {missing}")
    out = ROOT / "artifacts/manifests" / f"contract_review_{phase}.json"
    protocol = phase.split("_", 1)[0] if phase.startswith(("aio3_", "aio5_")) else None
    canonical = (
        ROOT / "artifacts/manifests" / f"contract_review_{protocol}_before_stage_b.json"
        if protocol else None
    )
    if phase.endswith("_before_stage_b") and out.is_file():
        frozen = json.loads(out.read_text())
        if frozen.get("sha256") != hashes:
            raise RuntimeError(
                "Stage-B canonical contract drift; archive all affected arms and "
                "create a new preregistered run family"
            )
        note(f"CONTRACT_REVIEW phase={phase} matches frozen canonical `{out}`")
        return
    if canonical is not None and canonical.is_file() and phase != f"{protocol}_startup":
        frozen = json.loads(canonical.read_text())
        if frozen.get("sha256") != hashes:
            raise RuntimeError(
                f"pipeline contract drift since `{canonical}`; refusing mixed-code gates"
            )
    temporary = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps({"phase": phase, "time": utc(), "sha256": hashes}, indent=2) + "\n"
    )
    os.replace(temporary, out)
    note(f"CONTRACT_REVIEW phase={phase} manifest=`{out}`")


def current_train_code_hashes() -> dict[str, str]:
    paths = (
        CODE_ROOT / "scripts/train.py",
        CODE_ROOT / "scripts/stage_b_runtime.py",
        CODE_ROOT / "src/net/feedback_controls.py",
        CODE_ROOT / "src/net/srsc_lite.py",
        CODE_ROOT / "src/net/srsc_coordinates.py",
        CODE_ROOT / "src/data/aio_dataset.py",
        CODE_ROOT / "src/losses/objectives.py",
    )
    return {
        str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _run_cache_expectation(
    *,
    run_name: str,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    max_steps: int,
    seed_override: int | None,
    epochs: int | None,
) -> dict | None:
    """Build the exact immutable contract expected from a cached trainer run.

    The trainer stores the configuration path in checkpoint arguments, while
    ``run_contract.json`` stores its bytes and effective YAML.  Cache reuse must
    validate both carriers: matching code alone is not scientific provenance.
    Missing source/statistics files deliberately produce a cache miss so the
    normal training path can raise its more specific prerequisite error.
    """
    try:
        config = Path(config).resolve()
        if not config.is_file():
            return None
        effective_config = yaml.safe_load(config.read_text())
        if not isinstance(effective_config, dict):
            return None
        if seed_override is not None:
            effective_config["seed"] = seed_override
        if epochs is not None and int(effective_config.get("epochs", -1)) != int(epochs):
            return None

        split = Path(effective_config["split_manifest"]).resolve()
        if not split.is_file():
            return None
        source = Path(init).resolve() if init is not None else None
        if source is not None and not source.is_file():
            return None
        coordinate_stats = None
        if stage in {"b_oracle", "b_predicted", "c"}:
            stats_value = effective_config.get("coordinate_stats")
            if not stats_value:
                return None
            stats = Path(stats_value).resolve()
            if not stats.is_file():
                return None
            coordinate_stats = _sha256_file(stats)

        runtime_identity = runtime_identity_for_config(config, effective_config)
        assert_no_runtime_worker_override(effective_config)
        workers_override = (
            None if runtime_identity else registered_train_workers()
        )
        contract = {
            "schema": 1,
            "run_name": run_name,
            "stage": stage,
            "feedback": feedback,
            "max_steps": int(max_steps),
            "seed_override": seed_override,
            "workers_override": workers_override,
            "allow_incomplete_data": False,
            "effective_config": effective_config,
            "config_sha256": _sha256_file(config),
            "split_manifest_sha256": _sha256_file(split),
            "source_init_path": str(source) if source is not None else None,
            "source_init_sha256": (
                _sha256_file(source) if source is not None else None
            ),
            "coordinate_stats_sha256": coordinate_stats,
            "code_sha256": current_train_code_hashes(),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        }
        contract.update(runtime_identity)
        return {
            "config_path": str(config),
            "contract": contract,
        }
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def run_contract_matches_current(
    run_dir: Path,
    *,
    run_name: str,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    max_steps: int = 0,
    seed_override: int | None = None,
    epochs: int | None = None,
    checkpoint_args: dict | None = None,
    _expectation: dict | None = None,
) -> bool:
    """Require the contract and its checkpoint argument carrier to agree.

    ``checkpoint_args`` is mandatory for a cache hit because it is the only
    immutable artifact that records the actual configuration path passed to the
    trainer.  This prevents a same-bytes file at a different preregistered path
    from silently reusing an arm.
    """
    path = Path(run_dir) / "run_contract.json"
    expected = _expectation or _run_cache_expectation(
        run_name=run_name, config=config, stage=stage, feedback=feedback,
        init=init, max_steps=max_steps, seed_override=seed_override,
        epochs=epochs,
    )
    if expected is None or not path.is_file() or not isinstance(checkpoint_args, dict):
        return False
    try:
        payload = json.loads(path.read_text())
        contract = expected["contract"]
        # A cache hit requires exact equality with every scientific/runtime
        # field written by train.py; extra or missing fields are drift too.
        if payload != contract:
            return False
        contract_sha = _sha256_file(path)
        saved_config = checkpoint_args.get("config")
        if not saved_config:
            return False
        saved_config = str(Path(saved_config).resolve())
        expected_args = {
            "stage": stage,
            "feedback": feedback,
            "run_name": run_name,
            "max_steps": int(max_steps),
            "seed_override": seed_override,
            "workers_override": contract["workers_override"],
            "allow_incomplete_data": False,
            "source_init_path": contract["source_init_path"],
            "source_init_sha256": contract["source_init_sha256"],
            "run_contract_sha256": contract_sha,
        }
        return (
            saved_config == expected["config_path"]
            and all(checkpoint_args.get(key) == value for key, value in expected_args.items())
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _checkpoint_matches_run_cache(
    payload: dict,
    run_dir: Path,
    *,
    run_name: str,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    max_steps: int,
    seed_override: int | None,
    epochs: int | None,
    expectation: dict | None = None,
) -> bool:
    expected = expectation or _run_cache_expectation(
        run_name=run_name, config=config, stage=stage, feedback=feedback,
        init=init, max_steps=max_steps, seed_override=seed_override,
        epochs=epochs,
    )
    if expected is None or not isinstance(payload, dict):
        return False
    contract = expected["contract"]
    model = payload.get("model")
    return (
        isinstance(model, dict)
        and bool(model)
        and payload.get("config") == contract["effective_config"]
        and payload.get("config_sha256") == contract["config_sha256"]
        and payload.get("split_manifest_sha256")
        == contract["split_manifest_sha256"]
        and run_contract_matches_current(
            run_dir,
            run_name=run_name,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            max_steps=max_steps,
            seed_override=seed_override,
            epochs=epochs,
            checkpoint_args=payload.get("args"),
            _expectation=expected,
        )
    )


def run(command: list[str], log_name: str):
    log = ROOT / "artifacts/logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    note(f"START `{' '.join(command)}`; log `{log}`")
    with log.open("a") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        note(f"FAIL code={code}: `{' '.join(command)}`")
        raise subprocess.CalledProcessError(code, command)
    note(f"DONE `{' '.join(command)}`")


def refresh_metrics_long(protocol: str, boundary: str) -> None:
    """Atomically refresh the read-only long-form metric ledger."""
    run(
        [sys.executable, "scripts/export_metrics_long.py"],
        f"{protocol}_metrics_long_{boundary}.log",
    )


def ensure_stage_a_locked_val_cache(
    config: Path, stage_a: Path, log_name: str,
) -> Path:
    """Create or verify the only authorized fixed Stage-A output cache."""
    cfg = yaml.safe_load(config.read_text())
    checkpoint_sha = _sha256_file(stage_a)
    cache_dir = (
        ROOT / "artifacts/cache/stage_a_y1" / cfg["protocol"]
        / checkpoint_sha[:16] / "locked_val"
    )
    run(
        [
            sys.executable,
            "scripts/cache_stage_a_outputs.py",
            "--config", str(config),
            "--stage-a-checkpoint", str(stage_a),
        ],
        log_name,
    )
    manifest_path = cache_dir / "manifest.json"
    sidecar = cache_dir / "manifest.sha256"
    try:
        manifest = json.loads(manifest_path.read_text())
        sidecar_sha = _read_named_sha256_sidecar(sidecar, "manifest.json")
        binding = manifest["bindings"]["stage_a_checkpoint"]
        valid = (
            manifest.get("status") == "COMPLETE_TWO_PASS_VERIFIED"
            and manifest.get("scope") == "locked_val"
            and manifest.get("official_test_forbidden") is True
            and int(manifest.get("item_count", 0)) > 0
            and binding.get("sha256") == checkpoint_sha
            and sidecar_sha == _sha256_file(manifest_path)
            and isinstance(manifest.get("aggregate_sha256"), str)
            and len(manifest["aggregate_sha256"]) == 64
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        raise RuntimeError(f"Stage-A locked_val cache failed closed: {cache_dir}")
    evidence = ROOT / "artifacts/manifests" / (
        f"{cfg['protocol']}_stage_a_locked_val_cache.json"
        if "10_10" not in config.stem else
        f"{cfg['protocol']}_10_10_stage_a_locked_val_cache.json"
    )
    _atomic_write_text(
        evidence,
        json.dumps({
            "protocol": cfg["protocol"],
            "scope": "locked_val",
            "cache_dir": str(cache_dir),
            "manifest_sha256": _sha256_file(manifest_path),
            "aggregate_sha256": manifest["aggregate_sha256"],
            "stage_a_checkpoint": str(stage_a.resolve()),
            "stage_a_checkpoint_sha256": checkpoint_sha,
            "item_count": manifest["item_count"],
            "verified_utc": utc(),
        }, indent=2, sort_keys=True) + "\n",
    )
    note(
        f"STAGE_A_CACHE VERIFIED protocol={cfg['protocol']} "
        f"items={manifest['item_count']} manifest={manifest_path}"
    )
    return manifest_path


def coordinate_stats_complete(config: Path, stage_a: Path) -> bool:
    """Fail-closed provenance and shape check for frozen SRSC statistics."""
    try:
        cfg = yaml.safe_load(config.read_text())
        stats = Path(cfg["coordinate_stats"])
        payload = json.loads(stats.read_text())
        split_sha = _sha256_file(Path(cfg["split_manifest"]))
        checkpoint_sha = _sha256_file(stage_a)
        modes = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O12", "O13", "O15")
        normalization = payload["normalization"]
        if not (
            payload.get("protocol") == cfg["protocol"]
            and int(payload.get("seed", -1)) == int(cfg["seed"])
            and payload.get("stage_a_checkpoint_sha256") == checkpoint_sha
            and payload.get("split_manifest_sha256") == split_sha
            and math.isfinite(float(payload["tau_v"]))
            and math.isfinite(float(payload["tau_e"]))
            and float(payload["tau_v"]) >= 1e-4
            and float(payload["tau_e"]) >= 1e-4
            and len(payload["pca_direction_matrix"]) == 6
            and all(len(row) == 81 for row in payload["pca_direction_matrix"])
            and len(payload["pca_direction_mean"]) == 81
            and set(normalization) == set(modes)
        ):
            return False
        for mode in modes:
            center = normalization[mode]["center"]
            scale = normalization[mode]["scale"]
            if not (
                len(center) == len(scale) == 8
                and all(math.isfinite(float(value)) for value in center)
                and all(math.isfinite(float(value)) and float(value) > 0 for value in scale)
            ):
                return False
        return True
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        json.JSONDecodeError, yaml.YAMLError,
    ):
        return False


def ensure_coordinate_stats(config: Path, stage_a: Path, log_name: str) -> Path:
    cfg = yaml.safe_load(config.read_text())
    stats = Path(cfg["coordinate_stats"])
    if stats.is_file() and not coordinate_stats_complete(config, stage_a):
        raise RuntimeError(
            f"existing coordinate statistics drift from selected Stage-A: {stats}; "
            "archive them and rerun under a new preregistered artifact family"
        )
    if not stats.is_file():
        run(
            [
                sys.executable,
                "scripts/compute_coordinate_stats.py",
                "--config", str(config),
                "--stage-a-checkpoint", str(stage_a),
            ],
            log_name,
        )
    if not coordinate_stats_complete(config, stage_a):
        raise RuntimeError(f"coordinate-statistics transaction incomplete: {stats}")
    note(
        f"COORDINATE_STATS VERIFIED protocol={cfg['protocol']} path={stats} "
        f"sha256={_sha256_file(stats)}"
    )
    return stats


def configured_parallel_gpus() -> list[str]:
    """Return explicitly authorized GPUs for independent experiment arms.

    An empty environment variable retains the original sequential behavior.
    Each child sees exactly one physical GPU as cuda:0, so train.py, its
    optimizer, batch size, scheduler, and checkpoint format are unchanged.
    """
    raw = os.environ.get("SRSC_PARALLEL_GPUS", "").strip()
    if not raw:
        return []
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if any(not item.isdigit() for item in tokens):
        raise ValueError(f"invalid SRSC_PARALLEL_GPUS={raw!r}")
    # Canonicalize so aliases such as 0 and 00 cannot bypass duplicate checks.
    gpu_ids = [str(int(item)) for item in tokens]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"duplicate SRSC_PARALLEL_GPUS entries: {gpu_ids}")
    visible_count = torch.cuda.device_count()
    if visible_count and any(int(item) >= visible_count for item in gpu_ids):
        raise ValueError(
            f"SRSC_PARALLEL_GPUS outside visible range 0..{visible_count - 1}: {gpu_ids}"
        )
    return gpu_ids


def registered_train_workers() -> int | None:
    raw = os.environ.get("SRSC_TRAIN_WORKERS", "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("SRSC_TRAIN_WORKERS must be positive")
    return value


def run_independent_arms(jobs: list[tuple[list[str], str]]):
    """Run scientifically independent arms concurrently, one GPU per arm.

    This is wall-clock parallelism only.  It never shards a model or changes
    an arm's registered data, batch, number of steps, initialization, or LR.
    A failed child terminates its siblings so a partial tier cannot silently
    flow into a scientific gate.
    """
    if not jobs:
        return
    gpu_ids = configured_parallel_gpus()
    if not gpu_ids:
        for command, log_name in jobs:
            run(command, log_name)
        return

    pending = list(jobs)
    active: dict[str, tuple[subprocess.Popen, object, list[str], str]] = {}
    completed = 0
    note(
        f"PARALLEL_ARMS start jobs={len(jobs)} gpu_ids={','.join(gpu_ids)}; "
        "one unchanged single-GPU training process per arm"
    )
    previous_handlers = {}

    def _raise_on_shutdown(signum, _frame):
        raise RuntimeError(f"orchestrator received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_on_shutdown)
    try:
        while pending or active:
            for gpu_id in gpu_ids:
                if gpu_id in active or not pending:
                    continue
                command, log_name = pending.pop(0)
                command = list(command)
                log = ROOT / "artifacts/logs" / log_name
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                # The host exports OMP_NUM_THREADS=64.  Four simultaneous
                # arms must not each inherit all 64 CPU threads in addition
                # to their DataLoader workers.
                environment["OMP_NUM_THREADS"] = "4"
                environment["MKL_NUM_THREADS"] = "4"
                environment["OPENBLAS_NUM_THREADS"] = "4"
                environment["NUMEXPR_NUM_THREADS"] = "4"
                note(
                    f"START_GPU{gpu_id} `{' '.join(command)}`; log `{log}`"
                )
                blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
                try:
                    try:
                        process = subprocess.Popen(
                            [sys.executable, str(EXEC_UNBLOCKED), *command],
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=environment,
                            start_new_session=True,
                        )
                    except BaseException:
                        handle.close()
                        raise
                    active[gpu_id] = (process, handle, command, log_name)
                finally:
                    # Register the child before unblocking so a pending signal
                    # cannot create an untracked trainer.
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

            finished = []
            for gpu_id, (process, handle, command, log_name) in active.items():
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                if code:
                    note(f"FAIL_GPU{gpu_id} code={code}: `{' '.join(command)}`")
                    raise subprocess.CalledProcessError(code, command)
                completed += 1
                note(
                    f"DONE_GPU{gpu_id} ({completed}/{len(jobs)}) "
                    f"`{' '.join(command)}`"
                )
                finished.append(gpu_id)
            for gpu_id in finished:
                del active[gpu_id]
            if active and not finished:
                time.sleep(2)
    except BaseException:
        # Do not allow a second shutdown signal to interrupt best-effort group
        # cleanup and leave sibling trainers or DataLoader workers orphaned.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        for process, handle, _, _ in active.values():
            # Even if the Python trainer leader just exited, forked DataLoader
            # workers may still occupy its process group briefly.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, handle, _, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if not handle.closed:
                handle.close()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    note(f"PARALLEL_ARMS complete jobs={len(jobs)}")


def assert_matching_replay_digests(run_names: list[str]) -> None:
    """Prove paired arms consumed the same ordered raw sample identities."""
    if len(run_names) < 2:
        return
    root = ROOT / "artifacts/manifests/replay_digests"
    canonical = None
    canonical_name = None
    for run_name in run_names:
        directory = root / run_name
        files = sorted(directory.glob("epoch*.json")) if directory.is_dir() else []
        if not files:
            raise RuntimeError(f"missing formal replay digests for {run_name}")
        rows = []
        for path in files:
            payload = json.loads(path.read_text())
            rows.append({
                key: payload[key]
                for key in (
                    "epoch", "optimizer_step_end", "sample_count",
                    "ordered_sample_identity_sha256", "protocol", "seed",
                    "split_manifest_sha256",
                )
            })
        if canonical is None:
            canonical, canonical_name = rows, run_name
        elif rows != canonical:
            raise RuntimeError(
                f"paired-arm replay drift: {run_name} differs from {canonical_name}"
            )
    note(
        f"REPLAY_DIGEST_MATCH arms={','.join(run_names)} epochs={len(canonical or [])}"
    )


def checkpoint_complete(path: Path, epochs: int):
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload.get("epoch", 0) >= epochs
        and payload.get("batch_in_epoch", 0) == 0
        and payload.get("validation_pending") is None
    )


def ensure_protocol_stage_a_handoff(
    *, protocol: str, config: Path, run_name: str, checkpoint: Path,
) -> dict:
    """Require the exact registered Stage-A terminal transaction.

    The outer launchers already run the same verifier.  Repeating it at the
    orchestrator boundary is intentional: the orchestrator is resumable and
    may be invoked directly after a disconnect.  A successful rank exit or an
    ``epoch == 240`` payload is not sufficient evidence of the expected update
    budget, final locked validation, top-3 transaction, or DDP provenance.
    """
    if protocol not in STAGE_A_HANDOFF_CONTRACTS:
        raise ValueError(f"no frozen Stage-A handoff contract for {protocol!r}")
    contract = STAGE_A_HANDOFF_CONTRACTS[protocol]
    cfg = yaml.safe_load(config.read_text())
    if cfg.get("protocol") != protocol:
        raise RuntimeError(
            f"Stage-A config protocol mismatch: expected={protocol!r} "
            f"actual={cfg.get('protocol')!r}"
        )
    if int(cfg.get("epochs", -1)) != int(contract["expected_epoch"]):
        raise RuntimeError(
            "Stage-A epoch contract drift: "
            f"config={cfg.get('epochs')!r} "
            f"registered={contract['expected_epoch']}"
        )
    if not checkpoint.is_file():
        raise RuntimeError(f"missing Stage-A handoff checkpoint: {checkpoint}")

    output = checkpoint.parent / "orchestrator_handoff_integrity.json"
    command = [
        sys.executable,
        "scripts/verify_stage_a_checkpoint.py",
        "--checkpoint", str(checkpoint.resolve()),
        "--config", str(config.resolve()),
        # The expected-step assertion below is authoritative.  A low minimum
        # makes malformed terminal payloads fail once rather than repeatedly
        # loading a 300 MiB checkpoint while polling for an impossible update.
        "--minimum-step", "1",
        "--expected-run-name", run_name,
        "--expected-epoch", str(contract["expected_epoch"]),
        "--expected-step", str(contract["expected_step"]),
        "--expected-world-size", str(contract["expected_world_size"]),
        "--expected-global-effective-batch",
        str(contract["expected_global_effective_batch"]),
        "--expected-per-gpu-batch", str(contract["expected_per_gpu_batch"]),
        "--expected-accumulation", str(contract["expected_accumulation"]),
        "--expected-workers-per-rank",
        str(contract["expected_workers_per_rank"]),
        "--expected-backend", str(contract["expected_backend"]),
        "--require-validation-complete",
        "--poll-seconds", "1",
        "--timeout-hours", "0.01",
        "--output", str(output.resolve()),
    ]
    expected_origin = contract["expected_training_origin"]
    if expected_origin is not None:
        command.extend(["--expected-training-origin", str(expected_origin)])
    run(command, f"{protocol}_stage_a_orchestrator_handoff.log")

    try:
        report = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Stage-A handoff verifier did not publish a valid report: {output}"
        ) from error
    checkpoint_sha256 = _sha256_file(checkpoint)
    if (
        report.get("status") != "pass"
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or int(report.get("epoch", -1)) != int(contract["expected_epoch"])
        or int(report.get("step", -1)) != int(contract["expected_step"])
        or not all((report.get("checks") or {}).values())
    ):
        raise RuntimeError(
            f"Stage-A handoff attestation is incomplete or stale: {output}"
        )
    note(
        "STAGE_A_HANDOFF PASS "
        f"protocol={protocol} epoch={report['epoch']} step={report['step']} "
        f"sha256={checkpoint_sha256}"
    )
    return report


def pilot_complete(
    run_name: str,
    max_steps: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None = None,
) -> bool:
    """Require one exact-budget metric and a provenance-complete checkpoint."""
    metric = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    contract = run_dir / "run_contract.json"
    target = run_dir / "pilot_model.pt"
    marker = run_dir / "pilot_complete.json"
    expectation = _run_cache_expectation(
        run_name=run_name, config=config, stage=stage, feedback=feedback,
        init=init, max_steps=max_steps, seed_override=seed_override, epochs=None,
    )
    try:
        if expectation is None or not metric.is_file() or not contract.is_file():
            return False
        rows = [
            json.loads(line) for line in metric.read_text().splitlines()
            if line.strip()
        ]
        selected_rows = [
            row for row in rows if int(row.get("step", -1)) == int(max_steps)
        ]
        if len(selected_rows) != 1:
            return False
        selected = selected_rows[0]
        if target.is_file() != marker.is_file():
            return False
        candidate = target if target.is_file() else run_dir / "last.pt"
        if not candidate.is_file():
            return False
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if not _checkpoint_matches_run_cache(
            payload,
            run_dir,
            run_name=run_name,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            max_steps=max_steps,
            seed_override=seed_override,
            epochs=None,
            expectation=expectation,
        ):
            return False
        if (
            int(payload.get("step", -1)) != int(max_steps)
            or payload.get("validation_pending") is not None
        ):
            return False
        if not target.is_file():
            return True

        marker_payload = json.loads(marker.read_text())
        contract_sha = _sha256_file(contract)
        source_sha = payload.get("source_checkpoint_sha256")
        return (
            payload.get("checkpoint_kind") == "completed_pilot_model_only"
            and payload.get("selected_locked_val") == selected
            and isinstance(source_sha, str)
            and len(source_sha) == 64
            and marker_payload.get("run") == run_name
            and marker_payload.get("max_steps") == int(max_steps)
            and marker_payload.get("model") == target.name
            and marker_payload.get("model_sha256") == _sha256_file(target)
            and marker_payload.get("selected_locked_val") == selected
            and marker_payload.get("config_sha256") == payload.get("config_sha256")
            and marker_payload.get("split_manifest_sha256")
            == payload.get("split_manifest_sha256")
            and marker_payload.get("run_contract_sha256") == contract_sha
            and marker_payload.get("source_checkpoint_sha256") == source_sha
        )
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        json.JSONDecodeError, yaml.YAMLError,
    ):
        return False


def last_metric(run_name: str):
    path = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1])


def best_metric(run_name: str):
    """Return the locked-val-selected record, never the official test."""
    path = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"empty locked validation file: {path}")
    return max(records, key=lambda item: item["macro_psnr"])


def verified_paired_rows(metric: dict) -> Path:
    path_value = metric.get("paired_rows_path")
    expected_sha = metric.get("paired_rows_sha256")
    if not path_value or not expected_sha:
        raise RuntimeError("selected locked-val metric lacks paired-row provenance")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(f"paired locked-val CSV hash drift: {path}")
    return path


def locked_paired_comparison(label: str, baseline: dict, method: dict) -> dict:
    baseline_path = verified_paired_rows(baseline)
    method_path = verified_paired_rows(method)
    output = ROOT / "artifacts/metrics/locked_comparisons" / f"{label}.json"
    existing = json.loads(output.read_text()) if output.is_file() else None
    expected_baseline = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    expected_method = hashlib.sha256(method_path.read_bytes()).hexdigest()
    if not (
        existing
        and existing.get("baseline_sha256") == expected_baseline
        and existing.get("method_sha256") == expected_method
        and existing.get("bootstrap_draws") == 10_000
    ):
        run([
            sys.executable,
            "scripts/compare_locked_paired.py",
            "--baseline", str(baseline_path),
            "--method", str(method_path),
            "--output", str(output),
        ], f"locked_paired_{label}.log")
    result = json.loads(output.read_text())
    if (
        result.get("baseline_sha256") != expected_baseline
        or result.get("method_sha256") != expected_method
    ):
        raise RuntimeError(f"locked paired comparison provenance mismatch: {output}")
    return result


def best_checkpoint(run_name: str) -> Path:
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    compact = run_dir / "formal_best_model.pt"
    if compact.is_file():
        return compact
    index = run_dir / "top3.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    records = json.loads(index.read_text())
    if not records:
        raise RuntimeError(f"empty top3 index: {index}")
    path = run_dir / records[0]["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _formal_selection(run_name: str, epochs: int) -> dict | None:
    """Validate that top3[0] is exactly the locked-validation-selected row."""
    metrics = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
    top3 = ROOT / "artifacts/checkpoints" / run_name / "top3.json"
    try:
        if not metrics.is_file() or not top3.is_file():
            return None
        rows = [
            json.loads(line) for line in metrics.read_text().splitlines()
            if line.strip()
        ]
        records = json.loads(top3.read_text())
        if not rows or not isinstance(records, list) or not 1 <= len(records) <= 3:
            return None
        metric_by_boundary = {}
        for row in rows:
            boundary = (int(row["epoch"]), int(row["step"]))
            score = float(row["macro_psnr"])
            if boundary in metric_by_boundary or not math.isfinite(score):
                return None
            metric_by_boundary[boundary] = row
        if not any(epoch == int(epochs) for epoch, _ in metric_by_boundary):
            return None

        seen_checkpoints = set()
        prior_score = math.inf
        for record in records:
            score = float(record["score"])
            boundary = (int(record["epoch"]), int(record["step"]))
            checkpoint = str(record["checkpoint"])
            if (
                not math.isfinite(score)
                or score > prior_score
                or checkpoint in seen_checkpoints
                or Path(checkpoint).name != checkpoint
                or boundary not in metric_by_boundary
                or score != float(metric_by_boundary[boundary]["macro_psnr"])
            ):
                return None
            seen_checkpoints.add(checkpoint)
            prior_score = score

        selected = max(rows, key=lambda item: float(item["macro_psnr"]))
        selected_record = records[0]
        if (
            float(selected_record["score"]) != float(selected["macro_psnr"])
            or int(selected_record["epoch"]) != int(selected["epoch"])
            or int(selected_record["step"]) != int(selected["step"])
        ):
            return None
        return {
            "selected_metric": selected,
            "selected_top3_record": selected_record,
            "top3_records": records,
            "metric_rows": rows,
        }
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        json.JSONDecodeError,
    ):
        return None


def hybrid_ddp_workers_per_rank() -> int:
    """Return the immutable worker count used by the four-rank hybrid trainer.

    ``SRSC_TRAIN_WORKERS`` is interpreted per process for every trainer in this
    orchestrator.  The hybrid trainer's audited default is eight workers per
    rank, rather than the 32-worker aggregate recorded in its YAML file.
    """
    workers = registered_train_workers()
    return 8 if workers is None else workers


def baseline_pretrain_run_name(protocol: str, kind: str, seed: int) -> str:
    if kind not in {"baseline", "baseline_matched"}:
        raise ValueError(f"unsupported baseline kind: {kind!r}")
    if protocol == "aio3":
        return f"aio3_{kind}_hybrid_ddp_pretrain_s{seed}"
    if protocol == "aio5":
        return f"aio5_{kind}_pretrain_s{seed}"
    raise ValueError(f"unsupported baseline protocol: {protocol!r}")


def _hybrid_ddp_expectation(
    run_name: str,
    *,
    config: Path,
    stage: str,
    workers_per_rank: int,
    implementation=hybrid_ddp,
) -> tuple[dict, argparse.Namespace, dict]:
    """Build exactly the run contract the dedicated DDP trainer must write."""
    allowed = getattr(
        implementation,
        "ALLOWED_STAGES",
        implementation.reference.ALLOWED_STAGES,
    )
    if stage not in allowed:
        raise ValueError(f"unsupported hybrid DDP stage: {stage!r}")
    config = Path(config).resolve()
    loader = getattr(
        implementation,
        "load_and_validate_config",
        implementation.reference.load_and_validate_config,
    )
    cfg = loader(config)
    args = argparse.Namespace(
        config=str(config),
        stage=stage,
        run_name=run_name,
    )
    contract = implementation.run_contract_payload(
        cfg, args, workers_per_rank
    )
    return cfg, args, contract


def _verify_hybrid_aio3_locked_metric(run_name: str, metric: dict) -> None:
    """Cross-check one hybrid summary against all 534 preregistered images."""
    epoch = int(metric["epoch"])
    step = int(metric["step"])
    expected_path = (
        ROOT / "artifacts/metrics/locked_rows" / run_name
        / f"epoch{epoch:03d}_step{step:07d}.csv"
    ).resolve()
    path = verified_paired_rows(metric).resolve()
    if path != expected_path:
        raise RuntimeError(
            f"hybrid paired rows use the wrong run/boundary path: {path}"
        )
    expected_counts = {
        "dehaze": 205,
        "denoise15": 103,
        "denoise25": 103,
        "denoise50": 103,
        "derain": 20,
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["task", "name", "psnr", "ssim"]:
            raise RuntimeError(f"hybrid paired-row schema drift: {path}")
        rows = list(reader)
    if len(rows) != sum(expected_counts.values()):
        raise RuntimeError(f"hybrid paired-row image count drift: {path}")
    keys = [(row["task"], row["name"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate hybrid paired-row identity: {path}")
    by_task: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        psnr = float(row["psnr"])
        ssim = float(row["ssim"])
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise RuntimeError(f"non-finite hybrid paired-row metric: {path}")
        values = by_task.setdefault(row["task"], {"psnr": [], "ssim": []})
        values["psnr"].append(psnr)
        values["ssim"].append(ssim)
    if {task: len(values["psnr"]) for task, values in by_task.items()} != expected_counts:
        raise RuntimeError(f"hybrid paired-row task count drift: {path}")
    setting_ssim = metric.get("setting_ssim")
    if not isinstance(setting_ssim, dict) or set(setting_ssim) != set(expected_counts):
        raise RuntimeError("hybrid locked summary lacks complete SSIM settings")
    task_psnr = {}
    task_ssim = {}
    for task, values in by_task.items():
        task_psnr[task] = statistics.fmean(values["psnr"])
        task_ssim[task] = statistics.fmean(values["ssim"])
        if not math.isclose(
            task_psnr[task], float(metric[task]), rel_tol=0.0, abs_tol=1e-10
        ) or not math.isclose(
            task_ssim[task], float(setting_ssim[task]),
            rel_tol=0.0, abs_tol=1e-10,
        ):
            raise RuntimeError(f"hybrid summary/paired-row mismatch for {task}")
    if not math.isclose(
        statistics.fmean(task_psnr.values()),
        float(metric["macro_psnr"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ) or not math.isclose(
        statistics.fmean(task_ssim.values()),
        float(metric["five_setting_mean_ssim"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("hybrid macro summary/paired-row mismatch")


def _hybrid_ddp_full_integrity(payload: dict) -> bool:
    integrity = payload.get("ddp_integrity")
    return (
        isinstance(integrity, dict)
        and integrity.get("level") == "full_sha256"
        and int(integrity.get("world_size", -1)) == hybrid_ddp.WORLD_SIZE
        and integrity.get("all_ranks_identical") is True
    )


def _hybrid_optimizer_scheduler_valid(
    payload: dict, cfg: dict, epoch: int
) -> bool:
    """Bind Adam and the epoch-wise R2R LR to the registered boundary."""
    try:
        optimizer = payload["optimizer"]
        scheduler = payload["scheduler"]
        groups = optimizer["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1:
            return False
        group = groups[0]
        expected_lr = float(cfg["lr"]) * hybrid_ddp.r2r_pretrain_epoch_ratio(
            epoch,
            float(cfg["lr"]),
            int(cfg["warmup_epochs"]),
            int(cfg["scheduler_max_epochs"]),
            float(cfg["warmup_start_lr"]),
            float(cfg.get("pretrain_eta_min", 0.0)),
        )
        return (
            tuple(group["betas"]) == (0.9, 0.999)
            and math.isclose(float(group["eps"]), 1e-8, rel_tol=0.0, abs_tol=0.0)
            and math.isclose(
                float(group["weight_decay"]), 0.0, rel_tol=0.0, abs_tol=0.0
            )
            and group.get("amsgrad") is False
            and group.get("maximize") is False
            and group.get("capturable") is False
            and group.get("differentiable") is False
            and group.get("fused") is None
            and bool(group.get("params"))
            and math.isclose(
                float(group["initial_lr"]),
                float(cfg["lr"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(group["lr"]), expected_lr, rel_tol=0.0, abs_tol=1e-15
            )
            and int(scheduler["last_epoch"]) == epoch
            and int(scheduler["_step_count"]) == epoch + 1
            and len(scheduler["base_lrs"]) == 1
            and math.isclose(
                float(scheduler["base_lrs"][0]),
                float(cfg["lr"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and len(scheduler["_last_lr"]) == 1
            and math.isclose(
                float(scheduler["_last_lr"][0]),
                expected_lr,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _state_signature(state: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    if not isinstance(state, dict) or not state:
        raise TypeError("hybrid checkpoint model state is empty")
    signature = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("hybrid checkpoint model state is malformed")
        signature[key] = (tuple(value.shape), str(value.dtype))
    return signature


def hybrid_ddp_complete(
    run_name: str,
    *,
    config: Path,
    stage: str,
    workers_per_rank: int,
    implementation=hybrid_ddp,
) -> bool:
    """Fail-closed completion predicate for an uncompressed AIO-3 hybrid arm.

    This intentionally does not call ``formal_complete``.  The DDP trainer has
    a different cursor, schedule-digest, rank-integrity, and argument contract;
    accepting it through the generic cache predicate would erase precisely the
    evidence needed for the clean/matched optimization-budget comparison.
    """
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    contract_path = run_dir / "run_contract.json"
    last_path = run_dir / "last.pt"
    try:
        cfg, expected_args, expected_contract = _hybrid_ddp_expectation(
            run_name,
            config=config,
            stage=stage,
            workers_per_rank=workers_per_rank,
            implementation=implementation,
        )
        # Hybrid checkpoints are deliberately retained as last + top3.  A
        # generic compaction marker is evidence of a mixed cache family.
        if any(
            (run_dir / name).exists()
            for name in ("formal_complete.json", "formal_best_model.pt")
        ):
            return False
        if not contract_path.is_file() or not last_path.is_file():
            return False
        actual_contract = json.loads(contract_path.read_text())
        if actual_contract != expected_contract:
            return False
        contract_sha = _sha256_file(contract_path)

        epochs = int(cfg["epochs"])
        selection = _formal_selection(run_name, epochs)
        if selection is None:
            return False
        if implementation is capacity_hybrid_ddp and len(
            selection["top3_records"]
        ) != 3:
            # Forty-eight locked-validation boundaries must deterministically
            # leave a complete top-3 selection, not a partially committed
            # index that merely happens to contain top3[0].
            return False
        expected_validation_epochs = list(range(5, epochs + 1, 5))
        if sorted(int(row["epoch"]) for row in selection["metric_rows"]) != (
            expected_validation_epochs
        ):
            return False
        for row in selection["metric_rows"]:
            row_epoch = int(row["epoch"])
            expected_step, _ = hybrid_ddp.reference.budget_before_epoch(
                row_epoch
            )
            if int(row["step"]) != expected_step:
                return False
            _verify_hybrid_aio3_locked_metric(run_name, row)
        final_rows = [
            row for row in selection["metric_rows"]
            if int(row.get("epoch", -1)) == epochs
        ]
        if (
            len(final_rows) != 1
            or int(final_rows[0].get("step", -1))
            != hybrid_ddp.reference.EXPECTED_TOTAL_STEPS
        ):
            return False
        # Both the selected model and the terminal epoch need immutable
        # per-image evidence, even when they happen to be the same row.
        verified_paired_rows(selection["selected_metric"])
        verified_paired_rows(final_rows[0])

        last_payload = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        implementation.validate_resume_payload(
            last_payload,
            cfg,
            expected_args,
            workers_per_rank,
            contract_sha,
            verify_digests=True,
        )
        last_records = last_payload.get("epoch_update_digests")
        if not isinstance(last_records, dict):
            return False
        if (
            int(last_payload.get("epoch", -1)) != epochs
            or int(last_payload.get("update_in_epoch", -1)) != 0
            or int(last_payload.get("microbatch_in_epoch", -1)) != 0
            or int(last_payload.get("batch_in_epoch", -1)) != 0
            or int(last_payload.get("step", -1))
            != hybrid_ddp.reference.EXPECTED_TOTAL_STEPS
            or int(last_payload.get("samples_seen", -1))
            != hybrid_ddp.reference.EXPECTED_TOTAL_SAMPLES
            or last_payload.get("validation_pending") is not None
            or int(last_payload.get("validation_transaction_schema", -1)) != 1
            or last_payload.get("active_phase") is not None
            or not _hybrid_ddp_full_integrity(last_payload)
            or not _hybrid_optimizer_scheduler_valid(
                last_payload, cfg, epochs
            )
            or sorted(last_records, key=int)
            != [str(epoch) for epoch in range(epochs)]
            or last_payload.get("schedule_digest")
            != hybrid_ddp.schedule_digest(last_records)
        ):
            return False
        model = implementation.build_model(cfg, stage)
        model.load_state_dict(last_payload["model"], strict=True)
        model_signature = _state_signature(last_payload["model"])
        del model

        expected_checkpoint_names = {"last.pt"}
        for record in selection["top3_records"]:
            checkpoint = run_dir / str(record["checkpoint"])
            if not checkpoint.is_file():
                return False
            expected_checkpoint_names.add(checkpoint.name)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            implementation.validate_resume_payload(
                payload,
                cfg,
                expected_args,
                workers_per_rank,
                contract_sha,
                verify_digests=False,
            )
            checkpoint_epoch = int(record["epoch"])
            expected_keys = hybrid_ddp.expected_digest_keys(
                checkpoint_epoch, 0
            )
            expected_prefix = {
                key: last_records[key] for key in expected_keys
            }
            if (
                int(payload.get("epoch", -1)) != checkpoint_epoch
                or int(payload.get("step", -1)) != int(record["step"])
                or int(payload.get("update_in_epoch", -1)) != 0
                or payload.get("validation_pending") is not None
                or int(payload.get("validation_transaction_schema", -1)) != 1
                or not _hybrid_ddp_full_integrity(payload)
                or not _hybrid_optimizer_scheduler_valid(
                    payload, cfg, checkpoint_epoch
                )
                or _state_signature(payload.get("model")) != model_signature
                or payload.get("epoch_update_digests") != expected_prefix
                or payload.get("schedule_digest")
                != hybrid_ddp.schedule_digest(expected_prefix)
            ):
                return False

        actual_checkpoint_names = {
            path.name for path in run_dir.glob("*.pt") if path.is_file()
        }
        return actual_checkpoint_names == expected_checkpoint_names
    # A completion predicate must treat corrupt or partially committed
    # checkpoints as a cache miss; the subsequent trainer invocation will
    # surface the precise resume error without accepting any stale evidence.
    except Exception:
        return False


def assert_matching_hybrid_ddp_update_digests(run_names: list[str]) -> None:
    """Require clean and parameter-matched arms to share all 240 updates."""
    if len(run_names) < 2:
        raise ValueError("hybrid DDP comparison requires at least two arms")
    canonical = None
    canonical_name = None
    for run_name in run_names:
        last = ROOT / "artifacts/checkpoints" / run_name / "last.pt"
        if not last.is_file():
            raise RuntimeError(f"missing hybrid DDP terminal checkpoint: {last}")
        payload = torch.load(last, map_location="cpu", weights_only=False)
        records = payload.get("epoch_update_digests")
        if (
            not isinstance(records, dict)
            or sorted(records, key=int) != [str(epoch) for epoch in range(240)]
            or payload.get("schedule_digest") != hybrid_ddp.schedule_digest(records)
        ):
            raise RuntimeError(f"invalid hybrid DDP update digest set: {run_name}")
        identity = {
            "reference_schedule": payload.get("reference_schedule"),
            "reference_schedule_sha256": payload.get("reference_schedule_sha256"),
            "partition_algorithm": payload.get("partition_algorithm"),
            "epoch_update_digests": records,
            "schedule_digest": payload.get("schedule_digest"),
        }
        if canonical is None:
            canonical = identity
            canonical_name = run_name
        elif identity != canonical:
            raise RuntimeError(
                f"hybrid DDP paired-update drift: {run_name} differs from "
                f"{canonical_name}"
            )
    note(
        f"HYBRID_DDP_UPDATE_DIGEST_MATCH arms={','.join(run_names)} epochs=240 "
        f"schedule_digest={canonical['schedule_digest']}"
    )


def formal_complete(
    run_name: str,
    epochs: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None = None,
) -> bool:
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    marker = run_dir / "formal_complete.json"
    compact = run_dir / "formal_best_model.pt"
    selection = _formal_selection(run_name, epochs)
    expectation = _run_cache_expectation(
        run_name=run_name, config=config, stage=stage, feedback=feedback,
        init=init, max_steps=0, seed_override=seed_override, epochs=epochs,
    )
    try:
        if (
            expectation is None
            or selection is None
            or compact.is_file() != marker.is_file()
        ):
            return False
        if compact.is_file():
            marker_payload = json.loads(marker.read_text())
            model_payload = torch.load(compact, map_location="cpu", weights_only=False)
            contract = run_dir / "run_contract.json"
            if not contract.is_file() or not _checkpoint_matches_run_cache(
                model_payload,
                run_dir,
                run_name=run_name,
                config=config,
                stage=stage,
                feedback=feedback,
                init=init,
                max_steps=0,
                seed_override=seed_override,
                epochs=epochs,
                expectation=expectation,
            ):
                return False
            contract_sha = _sha256_file(contract)
            selected = selection["selected_metric"]
            selected_record = selection["selected_top3_record"]
            source_sha = model_payload.get("source_checkpoint_sha256")
            return (
                marker_payload.get("run") == run_name
                and marker_payload.get("completed_epochs") == int(epochs)
                and marker_payload.get("model") == compact.name
                and marker_payload.get("model_sha256") == _sha256_file(compact)
                and marker_payload.get("config_sha256")
                == model_payload.get("config_sha256")
                and marker_payload.get("split_manifest_sha256")
                == model_payload.get("split_manifest_sha256")
                and marker_payload.get("run_contract_sha256") == contract_sha
                and model_payload.get("run_contract_sha256") == contract_sha
                and marker_payload.get("selected_locked_val") == selected
                and model_payload.get("selected_locked_val") == selected
                and marker_payload.get("selected_top3_record") == selected_record
                and model_payload.get("selected_top3_record") == selected_record
                and model_payload.get("source_checkpoint")
                == selected_record["checkpoint"]
                and marker_payload.get("source_checkpoint")
                == selected_record["checkpoint"]
                and isinstance(source_sha, str)
                and len(source_sha) == 64
                and marker_payload.get("source_checkpoint_sha256") == source_sha
                and model_payload.get("checkpoint_kind")
                == "formal_locked_val_best_model_only"
                and int(model_payload.get("epoch", -1))
                == int(selected_record["epoch"])
                and int(model_payload.get("step", -1))
                == int(selected_record["step"])
                and model_payload.get("validation_pending") is None
            )

        last = run_dir / "last.pt"
        if not last.is_file():
            return False
        last_payload = torch.load(last, map_location="cpu", weights_only=False)
        if (
            int(last_payload.get("epoch", -1)) != int(epochs)
            or int(last_payload.get("batch_in_epoch", -1)) != 0
            or last_payload.get("validation_pending") is not None
            or not any(
                int(row["epoch"]) == int(epochs)
                and int(row["step"]) == int(last_payload.get("step", -1))
                for row in selection["metric_rows"]
            )
            or not _checkpoint_matches_run_cache(
                last_payload,
                run_dir,
                run_name=run_name,
                config=config,
                stage=stage,
                feedback=feedback,
                init=init,
                max_steps=0,
                seed_override=seed_override,
                epochs=epochs,
                expectation=expectation,
            )
        ):
            return False
        for record in selection["top3_records"]:
            checkpoint = run_dir / record["checkpoint"]
            if not checkpoint.is_file():
                return False
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                int(payload.get("epoch", -1)) != int(record["epoch"])
                or int(payload.get("step", -1)) != int(record["step"])
                or payload.get("validation_pending") is not None
                or not _checkpoint_matches_run_cache(
                    payload,
                    run_dir,
                    run_name=run_name,
                    config=config,
                    stage=stage,
                    feedback=feedback,
                    init=init,
                    max_steps=0,
                    seed_override=seed_override,
                    epochs=epochs,
                    expectation=expectation,
                )
            ):
                return False
        return True
    except (
        OSError, KeyError, TypeError, ValueError, OverflowError,
        json.JSONDecodeError, yaml.YAMLError,
    ):
        return False


def compact_completed_formal(
    run_name: str,
    epochs: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None = None,
):
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    lock_fd = acquire_process_lock(run_dir / ".train.lock")
    try:
        return _compact_completed_formal_locked(
            run_name,
            epochs,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            seed_override=seed_override,
        )
    finally:
        os.close(lock_fd)


def _compact_completed_formal_locked(
    run_name: str,
    epochs: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None,
):
    """Keep the locked-val-selected model/provenance, discard redundant Adam states."""
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    target = run_dir / "formal_best_model.pt"
    marker = run_dir / "formal_complete.json"
    if target.is_file() != marker.is_file():
        raise RuntimeError(f"partial formal compaction transaction: {run_name}")
    if target.is_file() and marker.is_file():
        if not formal_complete(
            run_name,
            epochs,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            seed_override=seed_override,
        ):
            raise RuntimeError(f"cached formal compaction contract mismatch: {run_name}")
        return
    if not formal_complete(
        run_name,
        epochs,
        config=config,
        stage=stage,
        feedback=feedback,
        init=init,
        seed_override=seed_override,
    ):
        raise RuntimeError(f"refusing to compact invalid formal run: {run_name}")
    source = best_checkpoint(run_name)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    selection = _formal_selection(run_name, epochs)
    if selection is None:
        raise RuntimeError(f"formal selection transaction is invalid: {run_name}")
    selected = selection["selected_metric"]
    selected_record = selection["selected_top3_record"]
    source_sha = _sha256_file(source)
    compact = {
        "model": payload["model"],
        "epoch": payload.get("epoch"),
        "batch_in_epoch": payload.get("batch_in_epoch"),
        "step": payload.get("step"),
        "validation_pending": payload.get("validation_pending"),
        "config": payload.get("config"),
        "config_sha256": payload.get("config_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "args": payload.get("args"),
        "run_contract_sha256": payload.get("args", {}).get("run_contract_sha256"),
        "selected_locked_val": selected,
        "selected_top3_record": selected_record,
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": source_sha,
        "checkpoint_kind": "formal_locked_val_best_model_only",
    }
    temporary = target.with_suffix(".pt.tmp")
    torch.save(compact, temporary)
    temporary.replace(target)
    marker_payload = {
        "run": run_name,
        "completed_epochs": epochs,
        "selected_locked_val": selected,
        "selected_top3_record": selected_record,
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": source_sha,
        "model": target.name,
        "model_sha256": _sha256_file(target),
        "config_sha256": compact.get("config_sha256"),
        "split_manifest_sha256": compact.get("split_manifest_sha256"),
        "run_contract_sha256": compact.get("run_contract_sha256"),
        "time": utc(),
    }
    _atomic_write_text(marker, json.dumps(marker_payload, indent=2) + "\n")
    if not formal_complete(
        run_name,
        epochs,
        config=config,
        stage=stage,
        feedback=feedback,
        init=init,
        seed_override=seed_override,
    ):
        raise RuntimeError(f"new formal compact transaction failed validation: {run_name}")
    for path in run_dir.glob("*.pt"):
        if path != target:
            path.unlink()
    if not formal_complete(
        run_name,
        epochs,
        config=config,
        stage=stage,
        feedback=feedback,
        init=init,
        seed_override=seed_override,
    ):
        raise RuntimeError(f"formal compact transaction drifted after cleanup: {run_name}")
    note(f"COMPACT completed formal `{run_name}` to locked-val best model `{target}` bytes={target.stat().st_size}")


def compact_completed_pilot(
    run_name: str,
    max_steps: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None = None,
):
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    lock_fd = acquire_process_lock(run_dir / ".train.lock")
    try:
        return _compact_completed_pilot_locked(
            run_name,
            max_steps,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            seed_override=seed_override,
        )
    finally:
        os.close(lock_fd)


def _compact_completed_pilot_locked(
    run_name: str,
    max_steps: int,
    *,
    config: Path,
    stage: str,
    feedback: str,
    init: Path | None,
    seed_override: int | None,
):
    """Retain reproducible pilot weights without permanent AdamW state.

    A completed pilot already has a locked-validation record and is never
    resumed by this orchestrator.  Keeping optimizer moments for every arm
    would consume several GiB after OTS is materialized.  Write the deployable
    model and provenance atomically first, then remove only the resumable
    checkpoint.  Stage-A and formal runs are deliberately not compacted.
    """
    run_dir = ROOT / "artifacts/checkpoints" / run_name
    source = run_dir / "last.pt"
    target = run_dir / "pilot_model.pt"
    marker = run_dir / "pilot_complete.json"
    if target.is_file() != marker.is_file():
        raise RuntimeError(f"partial pilot compaction transaction: {run_name}")
    if target.is_file():
        if not pilot_complete(
            run_name,
            max_steps,
            config=config,
            stage=stage,
            feedback=feedback,
            init=init,
            seed_override=seed_override,
        ):
            raise RuntimeError(f"cached pilot compaction contract mismatch: {run_name}")
        source.unlink(missing_ok=True)
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    if not pilot_complete(
        run_name,
        max_steps,
        config=config,
        stage=stage,
        feedback=feedback,
        init=init,
        seed_override=seed_override,
    ):
        raise RuntimeError(f"refusing to compact invalid pilot run: {run_name}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    selected = last_metric(run_name)
    source_sha = _sha256_file(source)
    compact = {
        "model": payload["model"],
        "epoch": payload.get("epoch"),
        "batch_in_epoch": payload.get("batch_in_epoch"),
        "step": payload.get("step"),
        "validation_pending": payload.get("validation_pending"),
        "config": payload.get("config"),
        "config_sha256": payload.get("config_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "args": payload.get("args"),
        "run_contract_sha256": payload.get("args", {}).get("run_contract_sha256"),
        "selected_locked_val": selected,
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": source_sha,
        "checkpoint_kind": "completed_pilot_model_only",
    }
    temporary = target.with_suffix(".pt.tmp")
    torch.save(compact, temporary)
    temporary.replace(target)
    marker_payload = {
        "run": run_name,
        "max_steps": int(max_steps),
        "selected_locked_val": selected,
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": source_sha,
        "model": target.name,
        "model_sha256": _sha256_file(target),
        "config_sha256": compact.get("config_sha256"),
        "split_manifest_sha256": compact.get("split_manifest_sha256"),
        "run_contract_sha256": compact.get("run_contract_sha256"),
        "time": utc(),
    }
    _atomic_write_text(marker, json.dumps(marker_payload, indent=2) + "\n")
    if not pilot_complete(
        run_name,
        max_steps,
        config=config,
        stage=stage,
        feedback=feedback,
        init=init,
        seed_override=seed_override,
    ):
        raise RuntimeError(f"new pilot compact transaction failed validation: {run_name}")
    source.unlink()
    note(
        f"COMPACT completed pilot `{run_name}` to model-only `{target}` "
        f"bytes={target.stat().st_size}"
    )


def train_command(
    config: Path, stage: str, run_name: str, feedback: str,
    init: Path | None, max_steps: int = 0, seed_override: int | None = None,
):
    effective_config = yaml.safe_load(Path(config).read_text())
    if not isinstance(effective_config, dict):
        raise RuntimeError(f"training config is not a mapping: {config}")
    runtime_identity = runtime_identity_for_config(config, effective_config)
    assert_no_runtime_worker_override(effective_config)
    assert_stage_b_cublas_environment(runtime_identity)
    command = [
        sys.executable, "scripts/train.py", "--config", str(config), "--stage", stage,
        "--feedback", feedback, "--run-name", run_name,
    ]
    last = ROOT / "artifacts/checkpoints" / run_name / "last.pt"
    if last.is_file():
        command += ["--resume", str(last)]
    elif init is not None:
        command += ["--init", str(init)]
    if max_steps:
        command += ["--max-steps", str(max_steps)]
    if seed_override is not None:
        command += ["--seed-override", str(seed_override)]
    # Frozen Stage-B configs own their worker count.  Ambient overrides are
    # forbidden above and therefore cannot change pilot/formal/capacity runs
    # or make a later resume acquire a different run-contract identity.
    workers = None if runtime_identity else registered_train_workers()
    if workers is not None:
        command += ["--workers-override", str(workers)]
    return command


def hybrid_ddp_train_command(
    config: Path,
    stage: str,
    run_name: str,
    *,
    workers_per_rank: int,
) -> list[str]:
    """Build one all-four-GPU exact-update hybrid baseline invocation."""
    if stage not in hybrid_ddp.reference.ALLOWED_STAGES:
        raise ValueError(f"unsupported hybrid DDP stage: {stage!r}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={hybrid_ddp.WORLD_SIZE}",
        "scripts/train_baseline_hybrid_ddp.py",
        "--config",
        str(Path(config).resolve()),
        "--stage",
        stage,
        "--run-name",
        run_name,
        "--workers-per-rank",
        str(workers_per_rank),
    ]
    last = ROOT / "artifacts/checkpoints" / run_name / "last.pt"
    if last.is_file():
        command += ["--resume", str(last)]
    else:
        command.append("--fresh")
    return command


def capacity_hybrid_ddp_train_command(
    config: Path,
    run_name: str,
    *,
    workers_per_rank: int,
) -> list[str]:
    """Build the isolated all-four-GPU exact-update 10/10 Stage-A command."""
    # Validate before scheduling any GPU process.  This proves the training
    # protocol matches the primary hybrid schedule and only ownership differs.
    capacity_hybrid_ddp.load_and_validate_config(config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={hybrid_ddp.WORLD_SIZE}",
        "scripts/train_stage_a_capacity_hybrid_ddp.py",
        "--config",
        str(Path(config).resolve()),
        "--run-name",
        run_name,
        "--workers-per-rank",
        str(workers_per_rank),
    ]
    last = ROOT / "artifacts/checkpoints" / run_name / "last.pt"
    if last.is_file():
        command += ["--resume", str(last)]
    else:
        command.append("--fresh")
    return command


def acquire_process_lock(path: Path) -> int:
    """Acquire a crash-safe, non-blocking process lock.

    The lock file is intentionally persistent.  Kernel ``flock`` ownership is
    tied to the open file description, so SIGKILL, tmux loss, or an instance
    interruption releases the lock without requiring a clean-up unlink.  This
    avoids treating a dead PID written by a previous run as a live lock.
    """
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.lseek(fd, 0, os.SEEK_SET)
        owner = os.read(fd, 128).decode(errors="replace").strip() or "unknown"
        os.close(fd)
        raise RuntimeError(f"orchestrator lock is held: {path} (pid={owner})") from exc
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    os.fsync(fd)
    return fd


def run_aio3_capacity_robustness(
    decision: dict,
    decision_path: Path,
    runtime_bundle: StageBRuntimeBundle,
) -> bool:
    """Run the preregistered 10/10 ownership split after the main Stage-B GO.

    The total 20-block decoder budget remains unchanged: the main split owns
    6/14 D1/D2 blocks, while this control owns 10/10.  It is trained
    from scratch and evaluates only the two required orderings, without a new
    hyperparameter search or feedback ladder.
    """
    if (
        runtime_bundle.protocol != "aio3"
        or runtime_bundle.capacity_config is None
        or runtime_bundle.capacity_config_sha256 is None
    ):
        raise RuntimeError("10/10 capacity control requires the frozen AIO3 bundle")
    capacity_identity = runtime_identity_for_config(
        runtime_bundle.capacity_config
    )
    if (
        capacity_identity.get("stage_b_runtime_manifest_sha256")
        != runtime_bundle.manifest_sha256
        or _sha256_file(runtime_bundle.capacity_config)
        != runtime_bundle.capacity_config_sha256
    ):
        raise RuntimeError("AIO3 10/10 runtime bundle drift; batch reselection forbidden")
    # One immutable config owns 10/10 Stage-A, its cache and coordinate
    # statistics.  Its non-model training protocol exactly matches the main
    # AIO-3 hybrid schedule; the only architectural change is D1/D2 ownership.
    config = ROOT / "configs/protocol_aio3_10_10_hybrid.yaml"
    stage_b_config = runtime_bundle.capacity_config
    cfg = capacity_hybrid_ddp.load_and_validate_config(config)
    stage_b_cfg = yaml.safe_load(stage_b_config.read_text())
    if stage_b_cfg.get("model") != cfg.get("model"):
        raise RuntimeError(
            "10/10 Stage-A and frozen Stage-B model definitions differ"
        )
    stage_a_name = f"aio3_stage_a_coarse_10_10_seed{cfg['seed']}"
    stage_a_last = ROOT / "artifacts/checkpoints" / stage_a_name / "last.pt"
    workers_per_rank = hybrid_ddp_workers_per_rank()
    if not hybrid_ddp_complete(
        stage_a_name,
        config=config,
        stage="a",
        workers_per_rank=workers_per_rank,
        implementation=capacity_hybrid_ddp,
    ):
        run(
            capacity_hybrid_ddp_train_command(
                config,
                stage_a_name,
                workers_per_rank=workers_per_rank,
            ),
            f"{stage_a_name}.log",
        )
    if not hybrid_ddp_complete(
        stage_a_name,
        config=config,
        stage="a",
        workers_per_rank=workers_per_rank,
        implementation=capacity_hybrid_ddp,
    ):
        raise RuntimeError("10/10 capacity Stage-A hybrid transaction incomplete")
    stage_a = best_checkpoint(stage_a_name)
    note(f"SELECT 10/10 Stage-A locked-val checkpoint `{stage_a}`")
    ensure_stage_a_locked_val_cache(
        config, stage_a, "aio3_stage_a_10_10_locked_val_cache.log"
    )
    ensure_coordinate_stats(config, stage_a, "aio3_coordinate_stats_10_10.log")
    review_contract(
        "aio3_before_capacity_10_10_stage_b",
        extra_contracts=runtime_bundle.contract_paths,
    )

    metrics = {}
    capacity_jobs = []
    for feedback in ("O6", "O7", "O12"):
        name = f"aio3_capacity_10_10_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
        if not formal_complete(
            name, stage_b_cfg["epochs"], config=stage_b_config,
            stage="b_predicted", feedback=feedback, init=stage_a,
        ):
            capacity_jobs.append(
                (
                    train_command(stage_b_config, "b_predicted", name, feedback, stage_a),
                    f"{name}.log",
                )
            )
    run_independent_arms(capacity_jobs)
    assert_matching_replay_digests([
        f"aio3_capacity_10_10_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
        for feedback in ("O6", "O7", "O12")
    ])
    for feedback in ("O6", "O7", "O12"):
        name = f"aio3_capacity_10_10_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
        metrics["P" + feedback[1:]] = best_metric(name)
        compact_completed_formal(
            name, stage_b_cfg["epochs"], config=stage_b_config,
            stage="b_predicted", feedback=feedback, init=stage_a,
        )

    direction_delta = metrics["P7"]["macro_psnr"] - metrics["P6"]["macro_psnr"]
    residual_delta = metrics["P7"]["macro_psnr"] - metrics["P12"]["macro_psnr"]
    passed = direction_delta > 0.0 and residual_delta > 0.0
    decision.update({
        "capacity_split_10_10": metrics,
        "capacity_split_10_10_direction_delta": direction_delta,
        "capacity_split_10_10_residual_delta": residual_delta,
        "capacity_robustness_go": passed,
        "capacity_robustness_rule": "P7>P6 and P7>P12; no hyperparameter search",
        "capacity_robustness_interpretation": (
            "main restoration-state ordering reproduced under 10/10"
            if passed else
            "main result is sensitive to the 6/14 coarse/correction ownership split"
        ),
    })
    persist_decision(decision, decision_path)
    note(
        f"CAPACITY 10/10 {'GO' if passed else 'SENSITIVE'}: "
        f"P7-P6={direction_delta:.4f}, P7-P12={residual_delta:.4f}"
    )
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aio3", "aio5"], default="aio3")
    parser.add_argument("--pilot-steps", type=int, default=1000)
    parser.add_argument(
        "--stop-after-stage-b",
        action="store_true",
        help=(
            "Return successfully immediately after the formal Oracle and predicted "
            "scientific gates pass, before Stage-C and official-test access."
        ),
    )
    args = parser.parse_args()
    config = ROOT / "configs" / f"protocol_{args.protocol}.yaml"
    cfg = yaml.safe_load(config.read_text())
    stage_c_config = ROOT / "configs" / f"stage_c_{args.protocol}.yaml"
    stage_c_cfg = yaml.safe_load(stage_c_config.read_text())

    gpu_lock_fd = acquire_process_lock(ROOT / ".srsc_gpu_pipeline.lock")
    lock = ROOT / f".orchestrate_{args.protocol}.lock"
    lock_fd = acquire_process_lock(lock)
    try:
        note(f"ORCHESTRATOR {args.protocol} entered strict data gate")
        review_contract(f"{args.protocol}_startup")
        run([sys.executable, "scripts/prepare_data.py", "--protocol", args.protocol, "--build"], f"orchestrate_{args.protocol}_manifest.log")
        run([sys.executable, "-m", "pytest", "-q"], f"orchestrate_{args.protocol}_pytest.log")
        refresh_metrics_long(args.protocol, "startup")

        stage_a_name = f"{args.protocol}_stage_a_coarse_seed{cfg['seed']}"
        stage_a_last = ROOT / "artifacts/checkpoints" / stage_a_name / "last.pt"
        ensure_protocol_stage_a_handoff(
            protocol=args.protocol,
            config=config,
            run_name=stage_a_name,
            checkpoint=stage_a_last,
        )
        note(f"REUSE completed Stage-A `{stage_a_last}`")
        stage_a = best_checkpoint(stage_a_name)
        note(f"SELECT Stage-A locked-val checkpoint `{stage_a}` for statistics and Stage-B")
        ensure_stage_a_locked_val_cache(
            config, stage_a, f"{args.protocol}_stage_a_locked_val_cache.log"
        )
        coordinate_stats = ensure_coordinate_stats(
            config, stage_a, f"{args.protocol}_coordinate_stats.log"
        )
        runtime_bundle = ensure_stage_b_runtime_bundle(
            ROOT,
            args.protocol,
            stage_a_checkpoint=stage_a,
            coordinate_stats=coordinate_stats,
            runner=run,
        )
        stage_b_config = runtime_bundle.main_config
        stage_b_cfg = yaml.safe_load(stage_b_config.read_text())
        note(
            "STAGE_B_RUNTIME FROZEN "
            f"protocol={args.protocol} micro_batch={runtime_bundle.micro_batch} "
            f"accumulation={runtime_bundle.accumulation} "
            f"effective_batch={runtime_bundle.effective_batch} "
            f"workers={runtime_bundle.workers} "
            f"manifest_sha256={runtime_bundle.manifest_sha256}"
        )

        review_contract(
            f"{args.protocol}_before_stage_b",
            extra_contracts=runtime_bundle.contract_paths,
        )

        # Pilot is a plumbing/optimization sanity check only.  Its metrics are
        # recorded but must never authorize SCIENTIFIC_GO or NO_GO.
        pilot_oracle = {}
        tier1 = ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O12")
        pilot_jobs = []
        for feedback in tier1:
            name = (
                f"{args.protocol}_oracle_{feedback.lower()}_pilot"
                f"_n{args.pilot_steps}_s{cfg['seed']}"
            )
            if not pilot_complete(
                name, args.pilot_steps, config=stage_b_config, stage="b_oracle",
                feedback=feedback, init=stage_a,
            ):
                pilot_jobs.append(
                    (
                        train_command(
                            stage_b_config, "b_oracle", name, feedback,
                            stage_a, args.pilot_steps
                        ),
                        f"{name}.log",
                    )
                )
        run_independent_arms(pilot_jobs)
        for feedback in tier1:
            name = (
                f"{args.protocol}_oracle_{feedback.lower()}_pilot"
                f"_n{args.pilot_steps}_s{cfg['seed']}"
            )
            if not pilot_complete(
                name, args.pilot_steps, config=stage_b_config, stage="b_oracle",
                feedback=feedback, init=stage_a,
            ):
                raise RuntimeError(f"pilot transaction incomplete after training: {name}")
            pilot_oracle[feedback] = last_metric(name)
            compact_completed_pilot(
                name, args.pilot_steps, config=stage_b_config, stage="b_oracle",
                feedback=feedback, init=stage_a,
            )
        decision = {
            "protocol": args.protocol,
            "stage": "ORACLE_PILOT_COMPLETE",
            "stage_a": "PASS",
            "pilot_steps": args.pilot_steps,
            "pilot_oracle": pilot_oracle,
            "pilot_authority": "PIPELINE_ONLY_NO_SCIENTIFIC_GATE",
            "stage_b_runtime": {
                "manifest": str(runtime_bundle.manifest_path),
                "manifest_sha256": runtime_bundle.manifest_sha256,
                "config": str(runtime_bundle.main_config),
                "config_sha256": runtime_bundle.main_config_sha256,
                "micro_batch": runtime_bundle.micro_batch,
                "accumulation": runtime_bundle.accumulation,
                "effective_batch": runtime_bundle.effective_batch,
                "workers": runtime_bundle.workers,
            },
            "oracle_sign": "INCOMPLETE",
            "oracle_direction": "INCOMPLETE",
            "predicted_srsc": "INCOMPLETE",
            "scientific_go": "INCOMPLETE",
            "publication_go": "INCOMPLETE",
            "residual_code_control": "INCOMPLETE",
            "selected_model": None,
            "blocking_issues": ["formal Oracle tier-1 has not completed"],
            "next_command": "run formal Oracle tier-1 on locked validation",
        }
        decision_path = ROOT / "reports" / f"decision_{args.protocol}.json"
        persist_decision(decision, decision_path)

        # Formal oracle ladder: every arm trains to the same preregistered
        # epoch budget and is selected only on locked_val.
        oracle = {}
        formal_oracle_jobs = []
        for feedback in tier1:
            name = f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            if not formal_complete(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            ):
                formal_oracle_jobs.append(
                    (
                        train_command(stage_b_config, "b_oracle", name, feedback, stage_a),
                        f"{name}.log",
                    )
                )
        run_independent_arms(formal_oracle_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            for feedback in tier1
        ])
        for feedback in tier1:
            name = f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            oracle[feedback] = best_metric(name)
            compact_completed_formal(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            )

        sign_delta = oracle["O5"]["macro_psnr"] - oracle["O4"]["macro_psnr"]
        direction_delta = oracle["O7"]["macro_psnr"] - oracle["O6"]["macro_psnr"]
        magnitude_delta = oracle["O7"]["macro_psnr"] - oracle["O3"]["macro_psnr"]
        edit_delta = oracle["O7"]["macro_psnr"] - oracle["O2"]["macro_psnr"]
        oracle_residual_delta = oracle["O7"]["macro_psnr"] - oracle["O12"]["macro_psnr"]
        oracle_paired = {
            "O5_vs_O4": locked_paired_comparison(
                f"{args.protocol}_oracle_o5_vs_o4", oracle["O4"], oracle["O5"]
            ),
            "O7_vs_O6": locked_paired_comparison(
                f"{args.protocol}_oracle_o7_vs_o6", oracle["O6"], oracle["O7"]
            ),
            "O7_vs_O2": locked_paired_comparison(
                f"{args.protocol}_oracle_o7_vs_o2", oracle["O2"], oracle["O7"]
            ),
            "O7_vs_O3": locked_paired_comparison(
                f"{args.protocol}_oracle_o7_vs_o3", oracle["O3"], oracle["O7"]
            ),
            "O7_vs_O12": locked_paired_comparison(
                f"{args.protocol}_oracle_o7_vs_o12", oracle["O12"], oracle["O7"]
            ),
        }
        oracle_paired_ci_all_positive = all(
            payload["five_setting_bootstrap_95ci"][0] > 0.0
            for payload in oracle_paired.values()
        )
        denoise_keys = [key for key in ("denoise15", "denoise25", "denoise50") if key in oracle["O5"]]
        sign_guard = all(oracle["O5"][key] - oracle["O4"][key] >= -0.01 for key in denoise_keys)
        sign_wins = sum(oracle["O5"][key] - oracle["O4"][key] >= 0.02 for key in denoise_keys)
        task_keys = metric_task_keys(args.protocol, oracle["O7"])
        oracle_direction_task_deltas = {
            key: oracle["O7"][key] - oracle["O6"][key] for key in task_keys
        }
        oracle_non_dehaze_direction = [
            delta for key, delta in oracle_direction_task_deltas.items() if key != "dehaze"
        ]
        oracle_direction_not_dehaze_only = bool(oracle_non_dehaze_direction) and (
            statistics.median(oracle_non_dehaze_direction) >= 0.0
            and max(oracle_non_dehaze_direction) >= 0.02
        )
        oracle_denoise_direction = [
            delta for key, delta in oracle_direction_task_deltas.items()
            if key.startswith("denoise")
        ]
        oracle_denoise_direction_guard = (
            not oracle_denoise_direction
            or all(delta >= -0.01 for delta in oracle_denoise_direction)
        )
        feasible_median_guard = all(
            paired_task_median_guard(args.protocol, oracle_paired[key])
            for key in ("O7_vs_O2", "O7_vs_O3")
        )
        provisional_oracle_go = (
            sign_delta >= 0.03 and direction_delta >= 0.02 and magnitude_delta > 0
            and edit_delta > 0
            and sign_guard and sign_wins >= min(2, len(denoise_keys))
            and feasible_median_guard
            and oracle_direction_not_dehaze_only and oracle_denoise_direction_guard
        )
        oracle_sign_tier1_go = (
            sign_delta >= 0.03 and sign_guard
            and sign_wins >= min(2, len(denoise_keys))
        )
        oracle_direction_tier1_go = (
            direction_delta >= 0.02 and oracle_direction_not_dehaze_only
            and oracle_denoise_direction_guard
        )
        decision.update({
            "stage": "ORACLE_FORMAL_TIER1",
            "oracle": oracle,
            "oracle_sign_delta": sign_delta,
            "oracle_direction_delta": direction_delta,
            "oracle_direction_task_deltas": oracle_direction_task_deltas,
            "oracle_direction_not_dehaze_only": oracle_direction_not_dehaze_only,
            "oracle_denoise_direction_guard": oracle_denoise_direction_guard,
            "oracle_vs_magnitude_delta": magnitude_delta,
            "oracle_vs_first_edit_delta": edit_delta,
            "oracle_o2_o3_per_task_median_guard": feasible_median_guard,
            "oracle_vs_residual_code_delta": oracle_residual_delta,
            "oracle_paired_locked": oracle_paired,
            "oracle_paired_ci_all_positive_report_only": oracle_paired_ci_all_positive,
            "oracle_residual_code_authority": (
                "REQUIRED_FAIR_DIAGNOSTIC_NOT_AN_ORACLE_TIER1_GO_CRITERION"
            ),
            "oracle_tier1_go": provisional_oracle_go,
            "oracle_sign": (
                "INCOMPLETE" if provisional_oracle_go
                else ("GO" if oracle_sign_tier1_go else "NO_GO")
            ),
            "oracle_direction": (
                "INCOMPLETE" if provisional_oracle_go
                else ("GO" if oracle_direction_tier1_go else "NO_GO")
            ),
            "residual_code_control": residual_code_outcome(oracle_residual_delta),
            "per_task_deltas": {"oracle_direction": oracle_direction_task_deltas},
            "scientific_go": "INCOMPLETE" if provisional_oracle_go else "NO_GO",
            "publication_go": "INCOMPLETE" if provisional_oracle_go else "NO_GO",
            "selected_model": None if provisional_oracle_go else "NO_GO",
            "blocking_issues": (
                ["independently retrained Oracle controls have not completed"]
                if provisional_oracle_go else
                ["formal Oracle tier-1 failed the preregistered information-value gate"]
            ),
            "next_command": (
                "run independently retrained Oracle controls"
                if provisional_oracle_go else
                "stop before predicted feedback and Stage-C"
            ),
        })
        persist_decision(decision, decision_path)
        if not provisional_oracle_go:
            note(f"ORACLE FORMAL TIER1 NO-GO: sign={sign_delta:.4f}, direction={direction_delta:.4f}, vsO3={magnitude_delta:.4f}")
            refresh_metrics_long(args.protocol, "oracle_tier1_no_go")
            return 2

        # Mechanism controls are retrained from the same Stage-A checkpoint.
        controls = {}
        control_jobs = []
        for feedback in ("O8", "O9", "O10", "O11"):
            name = f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            if not formal_complete(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            ):
                control_jobs.append(
                    (
                        train_command(stage_b_config, "b_oracle", name, feedback, stage_a),
                        f"{name}.log",
                    )
                )
        run_independent_arms(control_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            for feedback in ("O8", "O9", "O10", "O11")
        ])
        for feedback in ("O8", "O9", "O10", "O11"):
            name = f"{args.protocol}_oracle_{feedback.lower()}_formal_s{cfg['seed']}"
            controls[feedback] = best_metric(name)
            compact_completed_formal(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            )
        direction_control_delta = oracle["O7"]["macro_psnr"] - max(
            controls["O9"]["macro_psnr"], controls["O10"]["macro_psnr"]
        )
        sign_abs_control_delta = oracle["O7"]["macro_psnr"] - controls["O8"]["macro_psnr"]
        random_noise_control_delta = oracle["O7"]["macro_psnr"] - controls["O11"]["macro_psnr"]
        oracle_go = (
            sign_abs_control_delta >= 0.02
            and direction_control_delta >= 0.02
            and random_noise_control_delta >= 0.02
        )
        oracle_control_paired = {
            feedback: locked_paired_comparison(
                f"{args.protocol}_oracle_o7_vs_{feedback.lower()}",
                controls[feedback], oracle["O7"],
            )
            for feedback in ("O8", "O9", "O10", "O11")
        }
        oracle_control_paired_ci_go = all(
            payload["five_setting_bootstrap_95ci"][0] > 0.0
            for payload in oracle_control_paired.values()
        )
        oracle_go = oracle_go and oracle_control_paired_ci_go
        oracle_sign_go = (
            oracle_sign_tier1_go and sign_abs_control_delta >= 0.02
            and oracle_control_paired["O8"]["five_setting_bootstrap_95ci"][0] > 0.0
        )
        oracle_direction_go = (
            oracle_direction_tier1_go and direction_control_delta >= 0.02
            and random_noise_control_delta >= 0.02
            and all(
                oracle_control_paired[key]["five_setting_bootstrap_95ci"][0] > 0.0
                for key in ("O9", "O10", "O11")
            )
        )
        decision.update({
            "stage": "ORACLE_FORMAL_COMPLETE",
            "oracle_controls": controls,
            "oracle_sign_abs_control_delta": sign_abs_control_delta,
            "oracle_direction_control_delta": direction_control_delta,
            "oracle_random_noise_control_delta": random_noise_control_delta,
            "oracle_control_paired_locked": oracle_control_paired,
            "oracle_control_paired_ci_go": oracle_control_paired_ci_go,
            "oracle_go": oracle_go,
            "oracle_sign": "GO" if oracle_sign_go else "NO_GO",
            "oracle_direction": "GO" if oracle_direction_go else "NO_GO",
            "scientific_go": "INCOMPLETE" if oracle_go else "NO_GO",
            "publication_go": "INCOMPLETE" if oracle_go else "NO_GO",
            "selected_model": None if oracle_go else "NO_GO",
            "blocking_issues": (
                ["formal predicted ladder has not completed"] if oracle_go else
                ["formal Oracle negative controls failed the preregistered gate"]
            ),
            "next_command": (
                "run formal predicted feedback ladder" if oracle_go else
                "stop before predicted feedback and Stage-C"
            ),
        })
        persist_decision(decision, decision_path)
        if not oracle_go:
            note(
                "ORACLE FORMAL CONTROL NO-GO: "
                f"sign_abs={sign_abs_control_delta:.4f}, "
                f"direction={direction_control_delta:.4f}, "
                f"random_noise={random_noise_control_delta:.4f}"
            )
            refresh_metrics_long(args.protocol, "oracle_controls_no_go")
            return 2

        # Train-only PCA projection is a frozen ablation, never a main-method
        # replacement or a GO criterion.
        oracle_pca_name = f"{args.protocol}_oracle_o15_pca_formal_s{cfg['seed']}"
        if not formal_complete(
            oracle_pca_name, stage_b_cfg["epochs"], config=stage_b_config,
            stage="b_oracle", feedback="O15", init=stage_a,
        ):
            run(
                train_command(stage_b_config, "b_oracle", oracle_pca_name, "O15", stage_a),
                f"{oracle_pca_name}.log",
            )
        oracle_pca = best_metric(oracle_pca_name)
        compact_completed_formal(
            oracle_pca_name, stage_b_cfg["epochs"], config=stage_b_config,
            stage="b_oracle", feedback="O15", init=stage_a,
        )
        decision.update({
            "oracle_pca_ablation": oracle_pca,
            "oracle_fixed_random_vs_pca_delta": (
                oracle["O7"]["macro_psnr"] - oracle_pca["macro_psnr"]
            ),
            "pca_ablation_authority": "REPORT_ONLY_NOT_A_GO_CRITERION",
        })
        persist_decision(decision, decision_path)

        # Tier-2 bandwidth diagnostics required by the frozen prompt.  O13 is
        # the fixed 8-D projection of the full transverse residual; O14 is the
        # 81-D direct correction ceiling passed through the capacity-matched
        # trainable adapter instantiated in every arm.  Neither can authorize
        # a deployable-method claim or enter the main fair-comparison table.
        oracle_bandwidth_diagnostics = {}
        bandwidth_specs = (
            ("O13", "full_e_projected_diagnostic"),
            ("O14", "direct_gt_correction_ceiling"),
        )
        bandwidth_jobs = []
        for feedback, label in bandwidth_specs:
            name = f"{args.protocol}_oracle_{feedback.lower()}_{label}_formal_s{cfg['seed']}"
            if not formal_complete(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            ):
                bandwidth_jobs.append(
                    (
                        train_command(stage_b_config, "b_oracle", name, feedback, stage_a),
                        f"{name}.log",
                    )
                )
        run_independent_arms(bandwidth_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_oracle_{feedback.lower()}_{label}_formal_s{cfg['seed']}"
            for feedback, label in bandwidth_specs
        ])
        for feedback, label in bandwidth_specs:
            name = f"{args.protocol}_oracle_{feedback.lower()}_{label}_formal_s{cfg['seed']}"
            oracle_bandwidth_diagnostics[feedback] = best_metric(name)
            compact_completed_formal(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_oracle", feedback=feedback, init=stage_a,
            )
        decision.update({
            "oracle_bandwidth_diagnostics": oracle_bandwidth_diagnostics,
            "oracle_bandwidth_diagnostics_authority": (
                "DIAGNOSTIC_OR_CEILING_ONLY_NOT_DEPLOYABLE_NOT_A_GO_CRITERION"
            ),
        })
        persist_decision(decision, decision_path)

        # Predicted pilot remains non-authoritative, exactly like oracle pilot.
        pilot_predicted = {}
        predicted_primary = ("O0", "O3", "O4", "O5", "O6", "O7", "O12")
        predicted_pilot_jobs = []
        for feedback in predicted_primary:
            name = (
                f"{args.protocol}_predicted_{feedback.lower()}_pilot"
                f"_n{args.pilot_steps}_s{cfg['seed']}"
            )
            if not pilot_complete(
                name, args.pilot_steps, config=stage_b_config,
                stage="b_predicted",
                feedback=feedback, init=stage_a,
            ):
                predicted_pilot_jobs.append(
                    (
                        train_command(
                            stage_b_config, "b_predicted", name, feedback,
                            stage_a, args.pilot_steps
                        ),
                        f"{name}.log",
                    )
                )
        run_independent_arms(predicted_pilot_jobs)
        for feedback in predicted_primary:
            name = (
                f"{args.protocol}_predicted_{feedback.lower()}_pilot"
                f"_n{args.pilot_steps}_s{cfg['seed']}"
            )
            if not pilot_complete(
                name, args.pilot_steps, config=stage_b_config,
                stage="b_predicted",
                feedback=feedback, init=stage_a,
            ):
                raise RuntimeError(f"pilot transaction incomplete after training: {name}")
            pilot_predicted[feedback] = last_metric(name)
            compact_completed_pilot(
                name, args.pilot_steps, config=stage_b_config,
                stage="b_predicted",
                feedback=feedback, init=stage_a,
            )
        predicted = {}
        predicted_formal_jobs = []
        for feedback in predicted_primary:
            name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
            if not formal_complete(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_predicted", feedback=feedback, init=stage_a,
            ):
                predicted_formal_jobs.append(
                    (
                        train_command(stage_b_config, "b_predicted", name, feedback, stage_a),
                        f"{name}.log",
                    )
                )
        run_independent_arms(predicted_formal_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
            for feedback in predicted_primary
        ])
        for feedback in predicted_primary:
            name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
            predicted[feedback] = best_metric(name)
            compact_completed_formal(
                name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_predicted", feedback=feedback, init=stage_a,
            )
        p12_name = f"{args.protocol}_predicted_o12_formal_s{cfg['seed']}"
        p12_checkpoint = best_checkpoint(p12_name)
        p12_diagnostic_output = (
            ROOT / "artifacts/metrics/feedback_diagnostics"
            / f"{args.protocol}_p12_locked_val.json"
        )
        if not feedback_diagnostics_complete(
            p12_diagnostic_output,
            checkpoint=p12_checkpoint,
            config=stage_b_config,
            feedback="O12",
        ):
            run([
                sys.executable,
                "scripts/eval_feedback_diagnostics.py",
                "--config", str(stage_b_config),
                "--checkpoint", str(p12_checkpoint),
                "--feedback", "O12",
                "--split", "locked_val",
                "--output", str(p12_diagnostic_output),
            ], f"{args.protocol}_p12_feedback_diagnostics.log")
        if not feedback_diagnostics_complete(
            p12_diagnostic_output,
            checkpoint=p12_checkpoint,
            config=stage_b_config,
            feedback="O12",
        ):
            raise RuntimeError("P12 feedback diagnostics transaction is incomplete")
        predicted_residual_code_diagnostics = json.loads(
            p12_diagnostic_output.read_text()
        )
        decision.update({
            "predicted_residual_code_diagnostics": {
                "artifact": str(p12_diagnostic_output.resolve()),
                "artifact_sha256": _sha256_file(p12_diagnostic_output),
                "pooled_aggregate": predicted_residual_code_diagnostics[
                    "pooled_aggregate"
                ],
                "scale_macro": predicted_residual_code_diagnostics["scale_macro"],
            }
        })
        persist_decision(decision, decision_path)
        p_direction = predicted["O7"]["macro_psnr"] - predicted["O6"]["macro_psnr"]
        p_controls = predicted["O7"]["macro_psnr"] - max(predicted["O3"]["macro_psnr"], predicted["O4"]["macro_psnr"])
        p_residual = predicted["O7"]["macro_psnr"] - predicted["O12"]["macro_psnr"]
        p_feasible = predicted["O7"]["macro_psnr"] - max(
            oracle["O1"]["macro_psnr"], oracle["O2"]["macro_psnr"]
        )
        predicted_paired = {
            "P7_vs_P6": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_p6", predicted["O6"], predicted["O7"]
            ),
            "P7_vs_P3": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_p3", predicted["O3"], predicted["O7"]
            ),
            "P7_vs_P4": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_p4", predicted["O4"], predicted["O7"]
            ),
            "P7_vs_P12": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_p12", predicted["O12"], predicted["O7"]
            ),
            "P7_vs_O1": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_o1", oracle["O1"], predicted["O7"]
            ),
            "P7_vs_O2": locked_paired_comparison(
                f"{args.protocol}_predicted_p7_vs_o2", oracle["O2"], predicted["O7"]
            ),
        }
        predicted_paired_ci_go = all(
            payload["five_setting_bootstrap_95ci"][0] > 0.0
            for payload in predicted_paired.values()
        )
        oracle_gain = oracle["O7"]["macro_psnr"] - max(
            oracle["O2"]["macro_psnr"], oracle["O3"]["macro_psnr"],
            oracle["O4"]["macro_psnr"], oracle["O12"]["macro_psnr"]
        )
        predicted_gain = predicted["O7"]["macro_psnr"] - max(
            oracle["O1"]["macro_psnr"], oracle["O2"]["macro_psnr"],
            predicted["O3"]["macro_psnr"], predicted["O4"]["macro_psnr"],
            predicted["O12"]["macro_psnr"]
        )
        capture_ratio = predicted_gain / oracle_gain if oracle_gain > 0 else None
        predicted_task_keys = metric_task_keys(args.protocol, predicted["O7"])
        predicted_direction_task_deltas = {
            key: predicted["O7"][key] - predicted["O6"][key]
            for key in predicted_task_keys
        }
        predicted_residual_task_deltas = {
            key: predicted["O7"][key] - predicted["O12"][key]
            for key in predicted_task_keys
        }
        predicted_feasible_task_deltas = {
            key: predicted["O7"][key] - max(oracle["O1"][key], oracle["O2"][key])
            for key in predicted_task_keys
        }
        predicted_non_dehaze_direction = [
            delta for key, delta in predicted_direction_task_deltas.items() if key != "dehaze"
        ]
        predicted_direction_not_dehaze_only = bool(predicted_non_dehaze_direction) and (
            statistics.median(predicted_non_dehaze_direction) >= 0.0
            and max(predicted_non_dehaze_direction) >= 0.02
        )
        predicted_denoise_direction = [
            delta for key, delta in predicted_direction_task_deltas.items()
            if key.startswith("denoise")
        ]
        predicted_denoise_guard = (
            len(predicted_denoise_direction) == (3 if args.protocol == "aio3" else 1)
            and all(delta >= -0.01 for delta in predicted_denoise_direction)
        )
        predicted_residual_non_dehaze = [
            delta for key, delta in predicted_residual_task_deltas.items()
            if key != "dehaze"
        ]
        predicted_residual_not_dehaze_only = bool(predicted_residual_non_dehaze) and (
            statistics.median(predicted_residual_non_dehaze) >= 0.0
            and max(predicted_residual_non_dehaze) >= 0.02
        )
        predicted_residual_denoise = [
            delta for key, delta in predicted_residual_task_deltas.items()
            if key.startswith("denoise")
        ]
        predicted_residual_denoise_guard = (
            len(predicted_residual_denoise) == (3 if args.protocol == "aio3" else 1)
            and all(delta >= -0.01 for delta in predicted_residual_denoise)
        )
        predicted_feasible_non_dehaze = [
            delta for key, delta in predicted_feasible_task_deltas.items()
            if key != "dehaze"
        ]
        predicted_feasible_not_dehaze_only = bool(predicted_feasible_non_dehaze) and (
            statistics.median(predicted_feasible_non_dehaze) >= 0.0
            and max(predicted_feasible_non_dehaze) >= 0.02
        )
        predicted_feasible_denoise = [
            delta for key, delta in predicted_feasible_task_deltas.items()
            if key.startswith("denoise")
        ]
        predicted_feasible_denoise_guard = (
            len(predicted_feasible_denoise) == (3 if args.protocol == "aio3" else 1)
            and all(delta >= -0.01 for delta in predicted_feasible_denoise)
        )
        primary_predicted_go = (
            p_direction > 0 and p_controls > 0 and p_residual >= 0.02
            and p_feasible >= 0.02
            and capture_ratio is not None and capture_ratio >= 0.40
            and predicted_direction_not_dehaze_only and predicted_denoise_guard
            and predicted_residual_not_dehaze_only and predicted_residual_denoise_guard
            and predicted_feasible_not_dehaze_only and predicted_feasible_denoise_guard
            and predicted_paired_ci_go
        )
        predicted_controls = {}
        predicted_sign_abs_delta = None
        predicted_direction_control_delta = None
        predicted_random_noise_delta = None
        predicted_go = primary_predicted_go
        if primary_predicted_go:
            # Negative controls are independently retrained; test-time-only
            # sign/direction corruption is not accepted as mechanism evidence.
            predicted_control_jobs = []
            for feedback in ("O8", "O9", "O10", "O11"):
                name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
                if not formal_complete(
                    name, stage_b_cfg["epochs"], config=stage_b_config,
                    stage="b_predicted", feedback=feedback, init=stage_a,
                ):
                    predicted_control_jobs.append(
                        (
                            train_command(
                                stage_b_config, "b_predicted", name, feedback, stage_a
                            ),
                            f"{name}.log",
                        )
                    )
            run_independent_arms(predicted_control_jobs)
            assert_matching_replay_digests([
                f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
                for feedback in ("O8", "O9", "O10", "O11")
            ])
            for feedback in ("O8", "O9", "O10", "O11"):
                name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{cfg['seed']}"
                predicted_controls[feedback] = best_metric(name)
                compact_completed_formal(
                    name, stage_b_cfg["epochs"], config=stage_b_config,
                    stage="b_predicted", feedback=feedback, init=stage_a,
                )
            predicted_sign_abs_delta = (
                predicted["O7"]["macro_psnr"] - predicted_controls["O8"]["macro_psnr"]
            )
            predicted_direction_control_delta = predicted["O7"]["macro_psnr"] - max(
                predicted_controls["O9"]["macro_psnr"],
                predicted_controls["O10"]["macro_psnr"],
            )
            predicted_random_noise_delta = (
                predicted["O7"]["macro_psnr"] - predicted_controls["O11"]["macro_psnr"]
            )
            predicted_go = (
                predicted_sign_abs_delta >= 0.02
                and predicted_direction_control_delta >= 0.02
                and predicted_random_noise_delta >= 0.02
            )
            predicted_control_paired = {
                feedback: locked_paired_comparison(
                    f"{args.protocol}_predicted_p7_vs_{feedback.lower()}",
                    predicted_controls[feedback], predicted["O7"],
                )
                for feedback in ("O8", "O9", "O10", "O11")
            }
            predicted_control_paired_ci_go = all(
                payload["five_setting_bootstrap_95ci"][0] > 0.0
                for payload in predicted_control_paired.values()
            )
            predicted_go = predicted_go and predicted_control_paired_ci_go
        else:
            predicted_control_paired = {}
            predicted_control_paired_ci_go = False
        predicted_pca = None
        predicted_fixed_random_vs_pca_delta = None
        if predicted_go:
            predicted_pca_name = f"{args.protocol}_predicted_o15_pca_formal_s{cfg['seed']}"
            if not formal_complete(
                predicted_pca_name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_predicted", feedback="O15", init=stage_a,
            ):
                run(
                    train_command(
                        stage_b_config, "b_predicted", predicted_pca_name, "O15", stage_a
                    ),
                    f"{predicted_pca_name}.log",
                )
            predicted_pca = best_metric(predicted_pca_name)
            compact_completed_formal(
                predicted_pca_name, stage_b_cfg["epochs"], config=stage_b_config,
                stage="b_predicted", feedback="O15", init=stage_a,
            )
            predicted_fixed_random_vs_pca_delta = (
                predicted["O7"]["macro_psnr"] - predicted_pca["macro_psnr"]
            )
        repeat_metrics = {str(cfg["seed"]): {
            "P6": predicted["O6"], "P7": predicted["O7"], "P12": predicted["O12"]
        }}
        repeat_direction_consistent = predicted_go
        if predicted_go:
            repeat_jobs = []
            for repeat_seed in (2718281, 1618033):
                for feedback in ("O6", "O7", "O12"):
                    name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{repeat_seed}"
                    if not formal_complete(
                        name, stage_b_cfg["epochs"], config=stage_b_config,
                        stage="b_predicted", feedback=feedback, init=stage_a,
                        seed_override=repeat_seed,
                    ):
                        repeat_jobs.append(
                            (
                                train_command(
                                    stage_b_config, "b_predicted", name, feedback, stage_a,
                                    seed_override=repeat_seed,
                                ),
                                f"{name}.log",
                            )
                        )
            run_independent_arms(repeat_jobs)
            for repeat_seed in (2718281, 1618033):
                assert_matching_replay_digests([
                    f"{args.protocol}_predicted_{feedback.lower()}_formal_s{repeat_seed}"
                    for feedback in ("O6", "O7", "O12")
                ])
            for repeat_seed in (2718281, 1618033):
                seed_metrics = {}
                for feedback in ("O6", "O7", "O12"):
                    name = f"{args.protocol}_predicted_{feedback.lower()}_formal_s{repeat_seed}"
                    seed_metrics["P" + feedback[1:]] = best_metric(name)
                    compact_completed_formal(
                        name, stage_b_cfg["epochs"], config=stage_b_config,
                        stage="b_predicted", feedback=feedback, init=stage_a,
                        seed_override=repeat_seed,
                    )
                repeat_metrics[str(repeat_seed)] = seed_metrics
                repeat_direction_consistent = repeat_direction_consistent and (
                    seed_metrics["P7"]["macro_psnr"] > seed_metrics["P6"]["macro_psnr"]
                )
        predicted_go = predicted_go and repeat_direction_consistent
        decision.update({
            "stage": "PREDICTED_FORMAL",
            "pilot_predicted": pilot_predicted,
            "predicted": predicted,
            "predicted_direction_delta": p_direction,
            "predicted_direction_task_deltas": predicted_direction_task_deltas,
            "predicted_residual_task_deltas": predicted_residual_task_deltas,
            "predicted_feasible_o1_o2_task_deltas": predicted_feasible_task_deltas,
            "predicted_direction_not_dehaze_only": predicted_direction_not_dehaze_only,
            "predicted_denoise_direction_guard": predicted_denoise_guard,
            "predicted_residual_not_dehaze_only": predicted_residual_not_dehaze_only,
            "predicted_residual_denoise_guard": predicted_residual_denoise_guard,
            "predicted_feasible_not_dehaze_only": predicted_feasible_not_dehaze_only,
            "predicted_feasible_denoise_guard": predicted_feasible_denoise_guard,
            "predicted_vs_controls_delta": p_controls,
            "predicted_vs_residual_delta": p_residual,
            "predicted_vs_deterministic_o1_o2_delta": p_feasible,
            "predicted_capture_ratio": capture_ratio,
            "predicted_paired_locked": predicted_paired,
            "predicted_paired_ci_go": predicted_paired_ci_go,
            "predicted_primary_go": primary_predicted_go,
            "predicted_controls": predicted_controls,
            "predicted_sign_abs_control_delta": predicted_sign_abs_delta,
            "predicted_direction_control_delta": predicted_direction_control_delta,
            "predicted_random_noise_control_delta": predicted_random_noise_delta,
            "predicted_control_paired_locked": predicted_control_paired,
            "predicted_control_paired_ci_go": predicted_control_paired_ci_go,
            "predicted_pca_ablation": predicted_pca,
            "predicted_fixed_random_vs_pca_delta": predicted_fixed_random_vs_pca_delta,
            "predicted_repeat_metrics": repeat_metrics,
            "predicted_three_seed_direction_consistent": repeat_direction_consistent,
            "predicted_go": predicted_go,
            "predicted_srsc": "GO" if predicted_go else "NO_GO",
            "scientific_go": "GO" if predicted_go else "NO_GO",
            "publication_go": "INCOMPLETE" if predicted_go else "NO_GO",
            "residual_code_control": residual_code_outcome(p_residual),
            "selected_model": select_predicted_model(predicted, predicted_go),
            "per_task_deltas": {
                "predicted_direction": predicted_direction_task_deltas,
                "predicted_vs_residual": predicted_residual_task_deltas,
                "predicted_vs_deterministic_o1_o2": predicted_feasible_task_deltas,
            },
            "blocking_issues": (
                ["Stage-C matched controls and joint fine-tuning have not completed"]
                if predicted_go else
                ["formal predicted feedback failed the preregistered learnability gate"]
            ),
            "next_command": (
                "Stage-C matched P7/O0 and preregistered controls"
                if predicted_go else "stop SRSC and prefer residual/control"
            ),
        })
        persist_decision(decision, decision_path)
        note(
            f"PREDICTED FORMAL {'GO' if predicted_go else 'NO-GO'}: "
            f"direction={p_direction:.4f}, controls={p_controls:.4f}, "
            f"residual={p_residual:.4f}, capture={capture_ratio}"
        )
        if not predicted_go:
            refresh_metrics_long(args.protocol, "predicted_no_go")
            return 3

        # The prompt requires a second ownership allocation to rule out a
        # result caused only by the deliberately coarse 6-block D1.  This is
        # conditional on the complete main Stage-B passing and therefore does
        # not consume compute for a rejected representation.
        if args.protocol == "aio3":
            run_aio3_capacity_robustness(
                decision, decision_path, runtime_bundle
            )

        if args.stop_after_stage_b:
            decision.update({
                "stage": "STAGE_B_COMPLETE",
                "scientific_go": "GO",
                "publication_go": "INCOMPLETE",
                "blocking_issues": [
                    "Stage-C matched controls and joint fine-tuning have not completed"
                ],
                "next_command": (
                    "freeze the second protocol Stage-B contract before either "
                    "protocol reads official test, then resume Stage-C"
                ),
            })
            persist_decision(decision, decision_path)
            note(
                f"STAGE-B COMPLETE protocol={args.protocol}; stopped before Stage-C "
                "and official test by preregistered sequencing flag"
            )
            refresh_metrics_long(args.protocol, "stage_b_complete")
            return 0

        # Only after the information and learnability gates pass, train the
        # expensive clean single-stage controls independently from scratch.
        # Stage-A's deliberately shallow D1 is not a valid Restormer baseline.
        baseline_models = {}
        baseline_pretrain_config = (
            ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"
            if args.protocol == "aio3" else config
        )
        baseline_pretrain_cfg = yaml.safe_load(baseline_pretrain_config.read_text())
        baseline_specs = (
            ("baseline", "baseline", "baseline_ft"),
            ("baseline_matched", "baseline_matched", "baseline_matched_ft"),
        )
        if args.protocol == "aio3":
            # These arms reproduce the observed 64->120 Stage-A update budget.
            # Each invocation owns all four GPUs, so clean and matched must run
            # sequentially.  Their DDP checkpoints remain uncompressed because
            # last/top3 carry the rank-integrity and 240-epoch update evidence.
            hybrid_ddp.reference.load_and_validate_config(
                baseline_pretrain_config
            )
            workers_per_rank = hybrid_ddp_workers_per_rank()
            hybrid_names = []
            for kind, pretrain_stage, _ in baseline_specs:
                pretrain_name = baseline_pretrain_run_name(
                    args.protocol, kind, int(cfg["seed"])
                )
                if not hybrid_ddp_complete(
                    pretrain_name,
                    config=baseline_pretrain_config,
                    stage=pretrain_stage,
                    workers_per_rank=workers_per_rank,
                ):
                    run(
                        hybrid_ddp_train_command(
                            baseline_pretrain_config,
                            pretrain_stage,
                            pretrain_name,
                            workers_per_rank=workers_per_rank,
                        ),
                        f"{pretrain_name}.log",
                    )
                if not hybrid_ddp_complete(
                    pretrain_name,
                    config=baseline_pretrain_config,
                    stage=pretrain_stage,
                    workers_per_rank=workers_per_rank,
                ):
                    raise RuntimeError(
                        f"hybrid DDP baseline transaction incomplete: {pretrain_name}"
                    )
                hybrid_names.append(pretrain_name)
            assert_matching_hybrid_ddp_update_digests(hybrid_names)
        else:
            # AIO-5 was registered as a constant global-batch-120 trajectory;
            # retain the generic train.py execution and compaction unchanged.
            if baseline_pretrain_cfg["effective_batch"] != 120:
                raise RuntimeError(
                    "AIO-5 baseline pretrain must use global batch 120"
                )
            baseline_pretrain_jobs = []
            for kind, pretrain_stage, _ in baseline_specs:
                pretrain_name = baseline_pretrain_run_name(
                    args.protocol, kind, int(cfg["seed"])
                )
                if not formal_complete(
                    pretrain_name, baseline_pretrain_cfg["epochs"],
                    config=baseline_pretrain_config, stage=pretrain_stage,
                    feedback="O0", init=None,
                ):
                    baseline_pretrain_jobs.append(
                        (
                            train_command(
                                baseline_pretrain_config, pretrain_stage,
                                pretrain_name, "O0", None
                            ),
                            f"{pretrain_name}.log",
                        )
                    )
            run_independent_arms(baseline_pretrain_jobs)
            for kind, pretrain_stage, _ in baseline_specs:
                pretrain_name = baseline_pretrain_run_name(
                    args.protocol, kind, int(cfg["seed"])
                )
                compact_completed_formal(
                    pretrain_name, baseline_pretrain_cfg["epochs"],
                    config=baseline_pretrain_config, stage=pretrain_stage,
                    feedback="O0", init=None,
                )
            assert_matching_replay_digests([
                baseline_pretrain_run_name(
                    args.protocol, kind, int(cfg["seed"])
                )
                for kind, _, _ in baseline_specs
            ])

        baseline_finetune_jobs = []
        for kind, _, finetune_stage in baseline_specs:
            pretrain_name = baseline_pretrain_run_name(
                args.protocol, kind, int(cfg["seed"])
            )
            finetune_name = f"{args.protocol}_{kind}_finetune_s{cfg['seed']}"
            pretrain_source = best_checkpoint(pretrain_name)
            if not formal_complete(
                finetune_name, stage_c_cfg["epochs"], config=stage_c_config,
                stage=finetune_stage, feedback="O0", init=pretrain_source,
            ):
                baseline_finetune_jobs.append(
                    (
                        train_command(
                            stage_c_config, finetune_stage, finetune_name, "O0",
                            pretrain_source,
                        ),
                        f"{finetune_name}.log",
                    )
                )
        run_independent_arms(baseline_finetune_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_{kind}_finetune_s{cfg['seed']}"
            for kind, _, _ in baseline_specs
        ])
        for kind, _, finetune_stage in baseline_specs:
            pretrain_name = baseline_pretrain_run_name(
                args.protocol, kind, int(cfg["seed"])
            )
            finetune_name = f"{args.protocol}_{kind}_finetune_s{cfg['seed']}"
            compact_completed_formal(
                finetune_name, stage_c_cfg["epochs"], config=stage_c_config,
                stage=finetune_stage, feedback="O0",
                init=best_checkpoint(pretrain_name),
            )
            baseline_models[kind] = {
                "best_metric": best_metric(finetune_name),
                "best_checkpoint": str(best_checkpoint(finetune_name)),
            }

        # Joint fine-tuning is authorized only after both formal gates.  Train
        # the matched no-state causal control under the identical Stage-C
        # schedule before any official-test invocation.
        joint_runs = {}
        joint_feedbacks = (
            "O0", "O1", "O2", "O6", "O7", "O8", "O9", "O10", "O11", "O12"
        )
        joint_jobs = []
        for feedback in joint_feedbacks:
            source_family = "oracle" if feedback in {"O1", "O2"} else "predicted"
            source_name = (
                f"{args.protocol}_{source_family}_{feedback.lower()}_formal_s{cfg['seed']}"
            )
            source = best_checkpoint(source_name)
            name = f"{args.protocol}_stage_c_{feedback.lower()}_s{cfg['seed']}"
            # A completed arm is compacted to its locked-validation-selected
            # model.  Checking the formal marker (rather than requiring a
            # retained ``last.pt``) makes the operation idempotent across an
            # orchestrator restart and prevents the eight Stage-C controls
            # from retaining three redundant Adam/RNG snapshots apiece.
            if not formal_complete(
                name, stage_c_cfg["epochs"], config=stage_c_config,
                stage="c", feedback=feedback, init=source,
            ):
                joint_jobs.append(
                    (
                        train_command(stage_c_config, "c", name, feedback, source),
                        f"{name}.log",
                    )
                )
        run_independent_arms(joint_jobs)
        assert_matching_replay_digests([
            f"{args.protocol}_stage_c_{feedback.lower()}_s{cfg['seed']}"
            for feedback in joint_feedbacks
        ])
        for feedback in joint_feedbacks:
            source_family = "oracle" if feedback in {"O1", "O2"} else "predicted"
            source_name = (
                f"{args.protocol}_{source_family}_{feedback.lower()}_formal_s{cfg['seed']}"
            )
            source = best_checkpoint(source_name)
            name = f"{args.protocol}_stage_c_{feedback.lower()}_s{cfg['seed']}"
            compact_completed_formal(
                name, stage_c_cfg["epochs"], config=stage_c_config,
                stage="c", feedback=feedback, init=source,
            )
            joint_runs[feedback] = {
                "best_metric": best_metric(name),
                "best_checkpoint": str(best_checkpoint(name)),
            }
        joint_score = {
            feedback: payload["best_metric"]["macro_psnr"]
            for feedback, payload in joint_runs.items()
        }
        joint_direction_delta = joint_score["O7"] - joint_score["O6"]
        joint_sign_control_delta = joint_score["O7"] - joint_score["O8"]
        joint_direction_control_delta = joint_score["O7"] - max(
            joint_score["O9"], joint_score["O10"]
        )
        joint_random_noise_delta = joint_score["O7"] - joint_score["O11"]
        joint_residual_delta = joint_score["O7"] - joint_score["O12"]
        joint_feasible_delta = joint_score["O7"] - max(
            joint_score["O1"], joint_score["O2"]
        )
        joint_paired = {
            feedback: locked_paired_comparison(
                f"{args.protocol}_stage_c_o7_vs_{feedback.lower()}",
                joint_runs[feedback]["best_metric"],
                joint_runs["O7"]["best_metric"],
            )
            for feedback in ("O0", "O1", "O2", "O6", "O8", "O9", "O10", "O11", "O12")
        }
        joint_paired_ci_go = all(
            payload["five_setting_bootstrap_95ci"][0] > 0.0
            for payload in joint_paired.values()
        )
        joint_task_keys = metric_task_keys(
            args.protocol, joint_runs["O7"]["best_metric"]
        )
        joint_direction_task_deltas = {
            key: (
                joint_runs["O7"]["best_metric"][key]
                - joint_runs["O6"]["best_metric"][key]
            )
            for key in joint_task_keys
        }
        joint_residual_task_deltas = {
            key: (
                joint_runs["O7"]["best_metric"][key]
                - joint_runs["O12"]["best_metric"][key]
            )
            for key in joint_task_keys
        }
        joint_non_dehaze_direction = [
            delta for key, delta in joint_direction_task_deltas.items() if key != "dehaze"
        ]
        joint_direction_not_dehaze_only = bool(joint_non_dehaze_direction) and (
            statistics.median(joint_non_dehaze_direction) >= 0.0
            and max(joint_non_dehaze_direction) >= 0.02
        )
        joint_denoise_direction = [
            delta for key, delta in joint_direction_task_deltas.items()
            if key.startswith("denoise")
        ]
        joint_denoise_guard = (
            len(joint_denoise_direction) == (3 if args.protocol == "aio3" else 1)
            and all(delta >= -0.01 for delta in joint_denoise_direction)
        )
        joint_residual_non_dehaze = [
            delta for key, delta in joint_residual_task_deltas.items()
            if key != "dehaze"
        ]
        joint_residual_not_dehaze_only = bool(joint_residual_non_dehaze) and (
            statistics.median(joint_residual_non_dehaze) >= 0.0
            and max(joint_residual_non_dehaze) >= 0.02
        )
        joint_residual_denoise = [
            delta for key, delta in joint_residual_task_deltas.items()
            if key.startswith("denoise")
        ]
        joint_residual_denoise_guard = (
            len(joint_residual_denoise) == (3 if args.protocol == "aio3" else 1)
            and all(delta >= -0.01 for delta in joint_residual_denoise)
        )
        joint_mechanism_go = (
            joint_direction_delta > 0.0
            and joint_sign_control_delta >= 0.02
            and joint_direction_control_delta >= 0.02
            and joint_random_noise_delta >= 0.02
            and joint_residual_delta >= 0.02
            and joint_feasible_delta >= 0.02
            and joint_direction_not_dehaze_only
            and joint_denoise_guard
            and joint_residual_not_dehaze_only
            and joint_residual_denoise_guard
            and joint_paired_ci_go
        )
        decision.update({
            "stage": "STAGE_C_COMPLETE",
            "stage_c": joint_runs,
            "joint_direction_delta": joint_direction_delta,
            "joint_sign_control_delta": joint_sign_control_delta,
            "joint_direction_control_delta": joint_direction_control_delta,
            "joint_random_noise_control_delta": joint_random_noise_delta,
            "joint_residual_delta": joint_residual_delta,
            "joint_feasible_o1_o2_delta": joint_feasible_delta,
            "joint_direction_task_deltas": joint_direction_task_deltas,
            "joint_residual_task_deltas": joint_residual_task_deltas,
            "joint_direction_not_dehaze_only": joint_direction_not_dehaze_only,
            "joint_denoise_direction_guard": joint_denoise_guard,
            "joint_residual_not_dehaze_only": joint_residual_not_dehaze_only,
            "joint_residual_denoise_guard": joint_residual_denoise_guard,
            "joint_paired_locked": joint_paired,
            "joint_paired_ci_go": joint_paired_ci_go,
            "joint_mechanism_go": joint_mechanism_go,
            "residual_code_control": residual_code_outcome(joint_residual_delta),
            "per_task_deltas": {
                "joint_direction": joint_direction_task_deltas,
                "joint_vs_residual": joint_residual_task_deltas,
            },
            "selected_model": "SRSC_LITE" if joint_mechanism_go else "NO_GO",
            "publication_go": "INCOMPLETE_OFFICIAL_TEST_FROZEN",
            "blocking_issues": (
                ["official test has not yet been executed"]
                if joint_mechanism_go else
                ["Stage-C failed to reproduce the preregistered mechanism ordering"]
            ),
            "next_command": "run each frozen O0/P7 checkpoint once on official test",
        })
        persist_decision(decision, decision_path)
        if not joint_mechanism_go:
            decision.update({
                # Stage-B has already established representation information
                # value and learnability.  A failed joint restoration gate
                # rejects publication strength, not that completed scientific
                # result; keep the two preregistered statuses distinct.
                "scientific_go": "GO",
                "scientific_scope": (
                    "Stage-B information-value and learnability gates passed; "
                    "Stage-C restoration mechanism ordering did not persist"
                ),
                "publication_go": "NO_GO",
                "selected_model": "NO_GO",
                "blocking_issues": [
                    "Stage-C failed to reproduce the preregistered mechanism ordering"
                ],
                "next_command": (
                    "stop before official test; report scientific-only support "
                    "and inspect Stage-C mechanism failure"
                ),
            })
            persist_decision(decision, decision_path)
            note("STAGE-C MECHANISM NO-GO: official test remains sealed")
            refresh_metrics_long(args.protocol, "stage_c_no_go")
            return 2
        review_contract(
            f"{args.protocol}_before_official_test",
            extra_contracts=runtime_bundle.contract_paths,
        )
        official_manifest_candidates = []
        for feedback in ("O0", "O1", "O2", "O7"):
            official_manifest_candidates.append({
                "candidate_id": f"{args.protocol}-stage-c-{feedback.lower()}",
                "model": "srsc",
                "checkpoint_path": joint_runs[feedback]["best_checkpoint"],
                "output_path": str(
                    ROOT / "artifacts/metrics"
                    / f"{args.protocol}_stage_c_{feedback.lower()}_official.csv"
                ),
            })
        for kind in ("baseline", "baseline_matched"):
            official_manifest_candidates.append({
                "candidate_id": f"{args.protocol}-{kind.replace('_', '-')}",
                "model": kind,
                "checkpoint_path": baseline_models[kind]["best_checkpoint"],
                "output_path": str(
                    ROOT / "artifacts/metrics" / f"{args.protocol}_{kind}_official.csv"
                ),
            })
        official_manifest = freeze_official_candidate_manifest(
            args.protocol, stage_c_config, official_manifest_candidates
        )
        decision.update({
            "official_candidate_manifest": str(official_manifest.resolve()),
            "official_candidate_manifest_sha256": _sha256_file(official_manifest),
            "blocking_issues": [
                "frozen official candidates have not all completed their one-shot evaluations"
            ],
        })
        persist_decision(decision, decision_path)
        official_outputs = {}
        for feedback in ("O0", "O1", "O2", "O7"):
            output = ROOT / "artifacts/metrics" / f"{args.protocol}_stage_c_{feedback.lower()}_official.csv"
            checkpoint = Path(joint_runs[feedback]["best_checkpoint"])
            if not official_artifacts_complete(
                args.protocol, "srsc", checkpoint, output,
                official_manifest=official_manifest,
            ):
                run([
                    sys.executable, "scripts/eval_locked.py",
                    "--config", str(stage_c_config),
                    "--checkpoint", str(checkpoint),
                    "--model", "srsc",
                    "--split", "official_test",
                    "--unlock-official-test",
                    "--official-manifest", str(official_manifest),
                    "--output", str(output),
                ], f"{args.protocol}_stage_c_{feedback.lower()}_official.log")
            if not official_artifacts_complete(
                args.protocol, "srsc", checkpoint, output,
                official_manifest=official_manifest,
            ):
                raise RuntimeError(f"official output transaction is incomplete: {output}")
            official_outputs[feedback] = str(output)
        for kind in ("baseline", "baseline_matched"):
            output = ROOT / "artifacts/metrics" / f"{args.protocol}_{kind}_official.csv"
            checkpoint = Path(baseline_models[kind]["best_checkpoint"])
            if not official_artifacts_complete(
                args.protocol, kind, checkpoint, output,
                official_manifest=official_manifest,
            ):
                run([
                    sys.executable, "scripts/eval_locked.py",
                    "--config", str(stage_c_config),
                    "--checkpoint", str(checkpoint),
                    "--model", kind,
                    "--split", "official_test",
                    "--unlock-official-test",
                    "--official-manifest", str(official_manifest),
                    "--output", str(output),
                ], f"{args.protocol}_{kind}_official.log")
            if not official_artifacts_complete(
                args.protocol, kind, checkpoint, output,
                official_manifest=official_manifest,
            ):
                raise RuntimeError(f"official output transaction is incomplete: {output}")
            official_outputs[kind] = str(output)
        # Additional OOD/local-composite evidence is strictly report-only and
        # never mixed into the standard AIO averages.  Generation was frozen
        # before any final checkpoint existed.
        local_composite_outputs = {}
        local_checkpoint_specs = {
            "O0": (joint_runs["O0"]["best_checkpoint"], "srsc"),
            "O1": (joint_runs["O1"]["best_checkpoint"], "srsc"),
            "O2": (joint_runs["O2"]["best_checkpoint"], "srsc"),
            "O7": (joint_runs["O7"]["best_checkpoint"], "srsc"),
            "baseline": (baseline_models["baseline"]["best_checkpoint"], "baseline"),
            "baseline_matched": (
                baseline_models["baseline_matched"]["best_checkpoint"],
                "baseline_matched",
            ),
        }
        for key, (checkpoint, model_kind) in local_checkpoint_specs.items():
            output = ROOT / "artifacts/metrics" / f"{args.protocol}_{key.lower()}_local_composite.csv"
            checkpoint = Path(checkpoint)
            if not local_composite_artifacts_complete(
                args.protocol, checkpoint, model_kind, output
            ):
                run([
                    sys.executable,
                    "scripts/eval_local_composite.py",
                    "--config", str(stage_c_config),
                    "--checkpoint", str(checkpoint),
                    "--model", model_kind,
                    "--output", str(output),
                ], f"{args.protocol}_{key.lower()}_local_composite.log")
            if not local_composite_artifacts_complete(
                args.protocol, checkpoint, model_kind, output
            ):
                raise RuntimeError(
                    f"local-composite output transaction is incomplete: {output}"
                )
            local_composite_outputs[key] = str(output)
        local_composite_comparisons = {}
        for baseline_key in ("O0", "O1", "O2", "baseline", "baseline_matched"):
            paired = ROOT / "artifacts/metrics" / (
                f"{args.protocol}_stage_c_o7_vs_{baseline_key.lower()}_local_composite_paired.json"
            )
            run([
                sys.executable,
                "scripts/compare_paired.py",
                "--baseline", local_composite_outputs[baseline_key],
                "--method", local_composite_outputs["O7"],
                "--output", str(paired),
            ], f"{args.protocol}_stage_c_o7_vs_{baseline_key.lower()}_local_composite.log")
            local_composite_comparisons[baseline_key] = json.loads(paired.read_text())
        local_composite_go = all(
            local_composite_comparisons[key]["macro_task_psnr_delta"] >= 0.10
            and local_composite_comparisons[key]["macro_task_ssim_delta"] >= -0.0001
            and local_composite_comparisons[key]["all_images_psnr_delta"]["bootstrap_95ci"][0] > 0.0
            for key in ("O0", "baseline_matched")
        )
        internal_comparisons = {}
        for baseline_key in ("O0", "baseline", "baseline_matched"):
            paired = ROOT / "artifacts/metrics" / f"{args.protocol}_stage_c_o7_vs_{baseline_key.lower()}_paired.json"
            run([
                sys.executable, "scripts/compare_paired.py",
                "--baseline", official_outputs[baseline_key],
                "--method", official_outputs["O7"],
                "--output", str(paired),
            ], f"{args.protocol}_stage_c_o7_vs_{baseline_key.lower()}_paired.log")
            internal_comparisons[baseline_key] = json.loads(paired.read_text())
        paired_result = internal_comparisons["O0"]
        internal_publication_go = all(
            comparison["publication_go_internal"] for comparison in internal_comparisons.values()
        ) and decision.get("capacity_robustness_go", True) and decision["joint_mechanism_go"] \
            and local_composite_go
        r2r_comparison = ROOT / "artifacts/metrics" / f"{args.protocol}_stage_c_o7_vs_r2r.json"
        run([
            sys.executable, "scripts/compare_r2r.py",
            "--protocol", args.protocol,
            "--method-summary", str(Path(official_outputs["O7"]).with_suffix(".json")),
            "--reference", str(ROOT / "artifacts/reference/r2r_cvpr2026_tables.json"),
            "--output", str(r2r_comparison),
        ], f"{args.protocol}_stage_c_o7_vs_r2r.log")
        r2r_result = json.loads(r2r_comparison.read_text())
        decision.update({
            "stage": "OFFICIAL_TEST_COMPLETE",
            "official_outputs": official_outputs,
            "paired_official_comparison": paired_result,
            "all_internal_official_comparisons": internal_comparisons,
            "local_composite_outputs": local_composite_outputs,
            "local_composite_comparisons": local_composite_comparisons,
            "local_composite_go": local_composite_go,
            "local_composite_rule": (
                "P7 vs O0 and matched baseline: PSNR>=+0.10dB, SSIM>=-0.0001, "
                "paired-bootstrap PSNR 95% CI lower bound >0; excluded from standard average"
            ),
            "r2r_table_comparison": r2r_result,
            "publication_go": "GO" if internal_publication_go else "PROMISING_NOT_PUBLICATION_GO",
            "publication_scope": "internal matched O0 gate plus separately reported external R2R table comparison",
            "user_r2r_target_met": r2r_result["user_target_met"],
            "selected_model": "SRSC_LITE",
            "residual_code_control": residual_code_outcome(joint_residual_delta),
            "per_task_deltas": paired_result["tasks"],
            "blocking_issues": (
                [] if internal_publication_go else
                ["one or more preregistered publication-strength guardrails failed"]
            ),
            "next_command": "final report and AutoSOTA eligibility audit",
        })
        persist_decision(decision, decision_path)
        note(
            f"OFFICIAL TEST COMPLETE protocol={args.protocol} internal_publication_go="
            f"{internal_publication_go} macro_delta={paired_result['macro_task_psnr_delta']:.4f}"
        )
        refresh_metrics_long(args.protocol, "official_complete")
        return 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        fcntl.flock(gpu_lock_fd, fcntl.LOCK_UN)
        os.close(gpu_lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
