#!/usr/bin/env python3
"""Full-RGB PSNR/SSIM evaluation with an explicit official-test lock."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_TASKS = {
    "aio3": ("dehaze", "derain", "denoise15", "denoise25", "denoise50"),
    "aio5": ("dehaze", "derain", "denoise25", "deblur", "lowlight"),
}

OFFICIAL_MANIFEST_SCHEMA_VERSION = 2
OFFICIAL_DATA_MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_LEDGER_SCHEMA_VERSION = 1
VALID_MODEL_KINDS = {"baseline", "baseline_matched", "srsc"}
OFFICIAL_EVAL_CODE_FILES = (
    "scripts/eval_locked.py",
    "src/data/aio_dataset.py",
    "src/net/srsc_lite.py",
    "src/net/clean_restormer_aio.py",
    "src/net/restormer_blocks.py",
    "src/net/feedback_controls.py",
)

from src.data import EXPECTED_OFFICIAL_COUNTS, build_locked_val, build_test_sets
from src.net import (
    DETERMINISTIC_FEEDBACK_MODES,
    PREDICTED_FEEDBACK_MODES,
    CleanRestormerAiO,
    SRSCLite,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["baseline", "baseline_matched", "srsc"], required=True)
    p.add_argument("--split", choices=["locked_val", "official_test"], required=True)
    p.add_argument("--unlock-official-test", action="store_true")
    p.add_argument(
        "--official-manifest",
        help=(
            "Pre-frozen candidate manifest. Required for official_test and ignored "
            "for locked_val."
        ),
    )
    p.add_argument(
        "--tile", type=int, default=0,
        help="0 is the paper-comparable full-image protocol; nonzero is an explicitly labeled fallback",
    )
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--output", required=True)
    return p.parse_args()


def build_model(cfg, kind):
    m = cfg["model"]
    common = dict(dim=m["dim"], encoder_blocks=tuple(m["encoder_blocks"]), heads=tuple(m["heads"]), expansion=m["expansion"])
    if kind in {"baseline", "baseline_matched"}:
        if kind == "baseline_matched":
            common["dim"] = m["matched_dim"]
        return CleanRestormerAiO(**common)
    return SRSCLite(
        **common,
        d1_blocks=tuple(m["d1_blocks"]),
        d2_blocks=tuple(m["d2_blocks"]),
        d2_refinement=m["d2_refinement"],
    )


def configure_srsc_inference(model, payload: dict, cfg: dict) -> None:
    """Restore the feedback intervention encoded by a predicted/joint run."""
    if not isinstance(model, SRSCLite):
        return
    checkpoint_args = payload.get("args", {})
    stage = checkpoint_args.get("stage")
    feedback = checkpoint_args.get("feedback")
    if stage not in {"b_predicted", "c"}:
        raise ValueError(
            f"deployable SRSC evaluation requires b_predicted/c checkpoint, got {stage!r}"
        )
    if stage == "c" and feedback in DETERMINISTIC_FEEDBACK_MODES:
        stats_path = Path(cfg["coordinate_stats"])
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        model.configure_deterministic_feedback(
            feedback, json.loads(stats_path.read_text())
        )
    elif feedback in PREDICTED_FEEDBACK_MODES:
        model.predicted_feedback_mode = feedback
        model.force_zero_state = feedback == "O0"
    else:
        raise ValueError(f"checkpoint has non-deployable feedback mode: {feedback!r}")


def tiled(model, x, tile, overlap):
    b, c, h, w = x.shape
    if tile <= 0 or max(h, w) <= tile:
        return model(x)
    stride = tile - overlap
    hs = list(range(0, max(h - tile, 0), stride)) + [max(h - tile, 0)]
    ws = list(range(0, max(w - tile, 0), stride)) + [max(w - tile, 0)]
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for top in sorted(set(hs)):
        for left in sorted(set(ws)):
            patch = x[..., top : top + tile, left : left + tile]
            restored = model(patch)
            out[..., top : top + restored.shape[-2], left : left + restored.shape[-1]] += restored
            weight[..., top : top + restored.shape[-2], left : left + restored.shape[-1]] += 1
    return out / weight.clamp_min(1)


def metrics(prediction, target):
    prediction = prediction.clamp(0, 1)
    mse = (prediction - target).square().mean().item()
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    p = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ssim = structural_similarity(t, p, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def official_evaluation_code_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in OFFICIAL_EVAL_CODE_FILES}


def _official_data_payload(cfg: dict) -> dict:
    """Hash the exact official identities without decoding pixels or using CUDA."""
    protocol = cfg["protocol"]
    sets = build_test_sets(cfg["data_root"], protocol)
    digest_cache: dict[Path, str] = {}

    def file_record(path: Path) -> dict:
        resolved = path.resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = sha256_file(resolved)
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": digest_cache[resolved],
        }

    records = []
    counts = {}
    for task in EXPECTED_TASKS[protocol]:
        dataset = sets[task]
        counts[task] = len(dataset)
        if hasattr(dataset, "pairs"):
            for degraded, clean in dataset.pairs:
                records.append({
                    "task": task,
                    "name": degraded.stem,
                    "sigma": 0,
                    "degraded": file_record(degraded),
                    "clean": file_record(clean),
                })
        elif hasattr(dataset, "clean_paths") and hasattr(dataset, "sigma"):
            for clean in dataset.clean_paths:
                records.append({
                    "task": task,
                    "name": clean.stem,
                    "sigma": int(dataset.sigma),
                    "degraded": None,
                    "clean": file_record(clean),
                })
        else:
            raise TypeError(f"unsupported official dataset type for {task}")
    expected_counts = EXPECTED_OFFICIAL_COUNTS[protocol]
    if counts != expected_counts:
        raise RuntimeError(
            f"official dataset identity count mismatch: {counts} != {expected_counts}"
        )
    keys = [(row["task"], row["name"]) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("official dataset identity has duplicate task/name keys")
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": OFFICIAL_DATA_MANIFEST_SCHEMA_VERSION,
        "protocol": protocol,
        "data_root": _resolved(cfg["data_root"]),
        "counts": counts,
        "record_count": len(records),
        "aggregate_sha256": aggregate,
        "records": records,
    }


def freeze_official_data_manifest(cfg: dict) -> Path:
    """Freeze official path/pair/content identity after Stage-C configuration lock."""
    payload = _official_data_payload(cfg)
    path = ROOT / "artifacts/manifests" / f"official_dataset_{cfg['protocol']}.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"corrupt official data manifest: {path}") from error
        if existing != payload:
            raise RuntimeError("official dataset drift after manifest freeze")
    else:
        atomic_write_json(path, payload)
    return path


def validate_official_data_manifest(
    path: Path, cfg: dict, expected_sha256: str,
) -> dict:
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise PermissionError("official data manifest path/SHA256 drift")
    try:
        frozen = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid official data manifest: {path}") from error
    current = _official_data_payload(cfg)
    if frozen != current:
        raise PermissionError("official pair keys, counts or image bytes drifted")
    return frozen


def validate_checkpoint_contract(payload: dict, candidate: dict) -> None:
    args = payload.get("args") or {}
    actual = {
        "stage": args.get("stage"),
        "feedback": args.get("feedback"),
        "run_contract_sha256": args.get("run_contract_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
    }
    if actual != candidate["checkpoint_contract"]:
        raise PermissionError(
            "checkpoint-internal training/run-contract binding drift"
        )


def official_control_paths(protocol: str) -> tuple[Path, Path]:
    """Return process-global controls shared by every candidate of a protocol."""
    if protocol not in EXPECTED_TASKS:
        raise ValueError(f"unsupported protocol: {protocol}")
    directory = ROOT / "artifacts/manifests"
    return (
        directory / f"official_test_{protocol}.flock",
        directory / f"official_test_{protocol}_consumption.json",
    )


@contextmanager
def protocol_file_lock(lock_path: Path):
    """Hold a non-blocking exclusive lock for the entire official evaluation."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another official-test process already holds the protocol lock: {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_frozen_official_manifest(
    manifest_path: Path,
    *,
    protocol: str,
    config_path: Path,
    config_sha256: str,
    model_kind: str,
    checkpoint_path: Path,
    output_path: Path,
) -> tuple[dict, str, dict]:
    """Validate and select one candidate without opening the checkpoint.

    The manifest deliberately binds both paths and content hashes.  Path checks
    happen before checkpoint hashing, so an unlisted checkpoint or output is
    rejected without reading checkpoint bytes or touching CUDA.
    """
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid official candidate manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("official candidate manifest must be a JSON object")
    if manifest.get("schema_version") != OFFICIAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported official candidate manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "FROZEN":
        raise PermissionError("official candidate manifest must have status='FROZEN'")
    if manifest.get("protocol") != protocol:
        raise PermissionError(
            f"official manifest protocol mismatch: {manifest.get('protocol')!r} != {protocol!r}"
        )
    if manifest.get("config_path") != _resolved(config_path):
        raise PermissionError("official manifest does not bind the requested config path")
    if manifest.get("config_sha256") != config_sha256:
        raise PermissionError("official manifest config SHA256 does not match the requested config")
    if not _is_sha256(manifest.get("config_sha256")):
        raise ValueError("official manifest config_sha256 must be a lowercase SHA256 digest")
    try:
        cfg = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid frozen official config: {config_path}") from error
    if not isinstance(cfg, dict) or cfg.get("protocol") != protocol:
        raise PermissionError("official manifest config protocol mismatch")

    stats_path = Path(manifest.get("coordinate_stats_path", ""))
    stats_sha = manifest.get("coordinate_stats_sha256")
    configured_stats = Path(cfg.get("coordinate_stats", "")).resolve()
    if (
        not stats_path.is_absolute()
        or stats_path != configured_stats
        or not stats_path.is_file()
        or not _is_sha256(stats_sha)
        or sha256_file(stats_path) != stats_sha
    ):
        raise PermissionError("official coordinate-statistics path/SHA256 drift")

    code_contract = manifest.get("evaluation_code_sha256")
    if code_contract != official_evaluation_code_hashes():
        raise PermissionError("official evaluation code closure drift")

    data_manifest_path = Path(manifest.get("official_data_manifest_path", ""))
    data_manifest_sha = manifest.get("official_data_manifest_sha256")
    if not data_manifest_path.is_absolute() or not _is_sha256(data_manifest_sha):
        raise ValueError("official data manifest path/SHA256 is invalid")
    validate_official_data_manifest(data_manifest_path, cfg, data_manifest_sha)

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("official candidate manifest requires a non-empty candidates list")
    requested_checkpoint = _resolved(checkpoint_path)
    requested_output = _resolved(output_path)
    candidate_ids: set[str] = set()
    candidate_pairs: set[tuple[str, str]] = set()
    all_output_artifacts: set[str] = set()
    matches: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"manifest candidate {index} must be an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"manifest candidate {index} requires a non-empty candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate official candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        candidate_model = candidate.get("model")
        if candidate_model not in VALID_MODEL_KINDS:
            raise ValueError(f"invalid model kind for candidate {candidate_id}: {candidate_model!r}")
        candidate_checkpoint = candidate.get("checkpoint_path")
        if not isinstance(candidate_checkpoint, str) or candidate_checkpoint != _resolved(candidate_checkpoint):
            raise ValueError(f"candidate {candidate_id} checkpoint_path must be absolute and canonical")
        pair = (candidate_model, candidate_checkpoint)
        if pair in candidate_pairs:
            raise ValueError(f"duplicate model/checkpoint candidate tuple: {pair}")
        candidate_pairs.add(pair)
        if not _is_sha256(candidate.get("checkpoint_sha256")):
            raise ValueError(f"candidate {candidate_id} has an invalid checkpoint_sha256")
        run_contract_path = candidate.get("run_contract_path")
        run_contract_sha = candidate.get("run_contract_sha256")
        if (
            not isinstance(run_contract_path, str)
            or run_contract_path != _resolved(run_contract_path)
            or not Path(run_contract_path).is_file()
            or not _is_sha256(run_contract_sha)
            or sha256_file(Path(run_contract_path)) != run_contract_sha
        ):
            raise PermissionError(
                f"candidate {candidate_id} run-contract path/SHA256 drift"
            )
        checkpoint_contract = candidate.get("checkpoint_contract")
        if not isinstance(checkpoint_contract, dict) or set(checkpoint_contract) != {
            "stage", "feedback", "run_contract_sha256", "config_sha256",
            "split_manifest_sha256",
        }:
            raise ValueError(
                f"candidate {candidate_id} has an invalid checkpoint contract"
            )
        if checkpoint_contract["run_contract_sha256"] != run_contract_sha:
            raise ValueError(
                f"candidate {candidate_id} run-contract digest disagreement"
            )
        if checkpoint_contract["config_sha256"] != config_sha256:
            raise PermissionError(
                f"candidate {candidate_id} checkpoint/config binding drift"
            )
        if not _is_sha256(checkpoint_contract["split_manifest_sha256"]):
            raise ValueError(
                f"candidate {candidate_id} split manifest digest is invalid"
            )
        output_paths = candidate.get("output_paths")
        if not isinstance(output_paths, list) or not output_paths:
            raise ValueError(f"candidate {candidate_id} requires non-empty output_paths")
        normalized_outputs: list[str] = []
        for raw_output in output_paths:
            if not isinstance(raw_output, str) or raw_output != _resolved(raw_output):
                raise ValueError(
                    f"candidate {candidate_id} output_paths must be absolute and canonical"
                )
            if Path(raw_output).suffix.lower() != ".csv":
                raise ValueError(
                    f"candidate {candidate_id} official output must use a .csv path"
                )
            if raw_output in normalized_outputs:
                raise ValueError(f"candidate {candidate_id} has duplicate output_paths")
            claimed = {raw_output, str(Path(raw_output).with_suffix(".json"))}
            overlap = claimed & all_output_artifacts
            if overlap:
                raise ValueError(
                    "official output/summary path belongs to multiple candidates: "
                    f"{sorted(overlap)}"
                )
            normalized_outputs.append(raw_output)
            all_output_artifacts.update(claimed)
        if (
            candidate_model == model_kind
            and candidate_checkpoint == requested_checkpoint
            and requested_output in normalized_outputs
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise PermissionError(
            "requested model/checkpoint/output tuple is not an allowed frozen official candidate"
        )
    return manifest, manifest_sha256, matches[0]


def _load_consumption_ledger(
    ledger_path: Path,
    *,
    protocol: str,
    manifest_path: Path,
    manifest_sha256: str,
    config_sha256: str,
) -> dict:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return {
            "schema_version": OFFICIAL_LEDGER_SCHEMA_VERSION,
            "protocol": protocol,
            "manifest_path": _resolved(manifest_path),
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "consumptions": [],
        }
    try:
        ledger = json.loads(ledger_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid official-test consumption ledger: {ledger_path}") from error
    expected = {
        "schema_version": OFFICIAL_LEDGER_SCHEMA_VERSION,
        "protocol": protocol,
        "manifest_path": _resolved(manifest_path),
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise PermissionError(
                f"official-test ledger rejects frozen manifest/config drift at {key}"
            )
    if not isinstance(ledger.get("consumptions"), list):
        raise RuntimeError("official-test consumption ledger has invalid consumptions")
    return ledger


def assert_official_candidate_available(
    ledger_path: Path,
    *,
    protocol: str,
    manifest_path: Path,
    manifest_sha256: str,
    config_sha256: str,
    candidate: dict,
    output_path: Path,
) -> None:
    """Check one-shot eligibility before reading any checkpoint bytes."""
    ledger = _load_consumption_ledger(
        ledger_path,
        protocol=protocol,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    candidate_id = candidate["candidate_id"]
    if any(item.get("candidate_id") == candidate_id for item in ledger["consumptions"]):
        raise FileExistsError(
            f"frozen official candidate has already been consumed: {candidate_id}"
        )
    output_path = Path(output_path)
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(
            f"official candidate output already exists and will not be overwritten: {output_path}"
        )


def reserve_official_candidate(
    ledger_path: Path,
    *,
    protocol: str,
    manifest_path: Path,
    manifest_sha256: str,
    config_sha256: str,
    candidate: dict,
    output_path: Path,
) -> None:
    """Atomically consume a candidate before torch.load/CUDA allocation."""
    assert_official_candidate_available(
        ledger_path,
        protocol=protocol,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        candidate=candidate,
        output_path=output_path,
    )
    ledger = _load_consumption_ledger(
        ledger_path,
        protocol=protocol,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    ledger["consumptions"].append({
        "candidate_id": candidate["candidate_id"],
        "model": candidate["model"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "output": _resolved(output_path),
        "status": "STARTED",
        "started_at": _utc_now(),
        "pid": os.getpid(),
    })
    atomic_write_json(Path(ledger_path), ledger)


def finalize_official_candidate(
    ledger_path: Path,
    *,
    candidate_id: str,
    status: str,
    details: dict | None = None,
) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(f"invalid official candidate terminal status: {status}")
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text())
    matches = [
        item for item in ledger.get("consumptions", [])
        if item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "STARTED":
        raise RuntimeError(f"cannot finalize official candidate {candidate_id!r}")
    matches[0]["status"] = status
    matches[0]["finished_at"] = _utc_now()
    if details:
        matches[0].update(details)
    atomic_write_json(ledger_path, ledger)


def summarize_rows(
    rows: list[dict], protocol: str, *, expected_counts: dict[str, int] | None = None,
) -> dict:
    if protocol not in EXPECTED_TASKS:
        raise ValueError(f"unsupported protocol: {protocol}")
    if not rows:
        raise RuntimeError("evaluation produced no rows")
    keys = [(str(row["task"]), str(row["name"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("evaluation produced duplicate (task, name) rows")
    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in ("psnr", "ssim")):
            raise RuntimeError(f"non-finite metric row: {row}")
    actual = {str(row["task"]) for row in rows}
    expected = set(EXPECTED_TASKS[protocol])
    if actual != expected:
        raise RuntimeError(
            f"evaluation task set mismatch: actual={sorted(actual)} expected={sorted(expected)}"
        )
    counts = {
        task: sum(str(row["task"]) == task for row in rows)
        for task in EXPECTED_TASKS[protocol]
    }
    if expected_counts is not None and counts != expected_counts:
        raise RuntimeError(
            f"official evaluation count mismatch: actual={counts} "
            f"expected={expected_counts}"
        )

    summary = {}
    for task in EXPECTED_TASKS[protocol]:
        subset = [row for row in rows if row["task"] == task]
        summary[task] = {
            "psnr": float(np.mean([row["psnr"] for row in subset])),
            "ssim": float(np.mean([row["ssim"] for row in subset])),
            "n": len(subset),
        }
    setting_psnr = float(np.mean([summary[key]["psnr"] for key in EXPECTED_TASKS[protocol]]))
    setting_ssim = float(np.mean([summary[key]["ssim"] for key in EXPECTED_TASKS[protocol]]))
    denoise_keys = [key for key in EXPECTED_TASKS[protocol] if key.startswith("denoise")]
    denoise_psnr = float(np.mean([summary[key]["psnr"] for key in denoise_keys]))
    denoise_ssim = float(np.mean([summary[key]["ssim"] for key in denoise_keys]))
    if protocol == "aio3":
        task_psnr = float(np.mean([summary["dehaze"]["psnr"], summary["derain"]["psnr"], denoise_psnr]))
        task_ssim = float(np.mean([summary["dehaze"]["ssim"], summary["derain"]["ssim"], denoise_ssim]))
    else:
        task_psnr, task_ssim = setting_psnr, setting_ssim
    # Keep ``macro`` as a backward-compatible alias for the R2R five-setting
    # average, but make its semantics explicit in the same artifact.
    summary["macro"] = {"psnr": setting_psnr, "ssim": setting_ssim}
    summary["aggregates"] = {
        "five_setting_mean": {"psnr": setting_psnr, "ssim": setting_ssim},
        "task_macro": {"psnr": task_psnr, "ssim": task_ssim},
        "denoise_task_mean": {"psnr": denoise_psnr, "ssim": denoise_ssim},
        "legacy_macro_semantics": "alias_of_five_setting_mean",
    }
    return summary


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def official_record_path(protocol: str, model_kind: str, checkpoint_sha256: str) -> Path:
    return (
        ROOT / "artifacts/manifests"
        / f"official_test_{protocol}_{model_kind}_{checkpoint_sha256[:16]}.json"
    )


def official_artifacts_complete(
    protocol: str,
    model_kind: str,
    checkpoint: Path,
    output: Path,
    *,
    official_manifest: Path,
) -> bool:
    """Verify a manifest-gated one-shot transaction before reusing it.

    Legacy completion records intentionally do not qualify.  Reuse is allowed
    only when the exact frozen manifest candidate, summary/CSV, completion
    record, and protocol-wide consumption ledger form one hash-bound COMPLETE
    transaction.
    """
    checkpoint = Path(checkpoint)
    output = Path(output)
    official_manifest = Path(official_manifest)
    if (
        not checkpoint.is_file()
        or not output.is_file()
        or not official_manifest.is_file()
    ):
        return False
    summary_path = output.with_suffix(".json")
    if not summary_path.is_file():
        return False
    try:
        manifest_preview = json.loads(official_manifest.read_text())
        config_path = Path(manifest_preview["config_path"])
        config_sha256 = sha256_file(config_path)
        _, manifest_sha256, candidate = validate_frozen_official_manifest(
            official_manifest,
            protocol=protocol,
            config_path=config_path,
            config_sha256=config_sha256,
            model_kind=model_kind,
            checkpoint_path=checkpoint,
            output_path=output,
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        if candidate.get("checkpoint_sha256") != checkpoint_sha256:
            return False
        record_path = official_record_path(protocol, model_kind, checkpoint_sha256)
        if not record_path.is_file():
            return False
        record = json.loads(record_path.read_text())
        summary = json.loads(summary_path.read_text())
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidate_id = candidate["candidate_id"]
        ledger_path = official_control_paths(protocol)[1]
        if record.get("official_ledger") != _resolved(ledger_path):
            return False
        ledger = json.loads(ledger_path.read_text())
        entries = [
            item for item in ledger.get("consumptions", [])
            if item.get("candidate_id") == candidate_id
        ]
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        PermissionError,
        json.JSONDecodeError,
        csv.Error,
    ):
        return False
    meta = summary.get("_meta", {})
    ledger_complete = len(entries) == 1 and all((
        ledger.get("schema_version") == OFFICIAL_LEDGER_SCHEMA_VERSION,
        ledger.get("protocol") == protocol,
        ledger.get("manifest_path") == _resolved(official_manifest),
        ledger.get("manifest_sha256") == manifest_sha256,
        ledger.get("config_sha256") == config_sha256,
        entries[0].get("status") == "COMPLETE",
        entries[0].get("model") == model_kind,
        entries[0].get("checkpoint_path") == str(checkpoint.resolve()),
        entries[0].get("checkpoint_sha256") == checkpoint_sha256,
        entries[0].get("output") == str(output.resolve()),
        entries[0].get("record") == str(record_path.resolve()),
        entries[0].get("record_sha256") == sha256_file(record_path),
        entries[0].get("csv_sha256") == sha256_file(output),
        entries[0].get("summary_sha256") == sha256_file(summary_path),
    ))
    return all((
        record.get("status") == "COMPLETE",
        record.get("protocol") == protocol,
        record.get("model") == model_kind,
        record.get("candidate_id") == candidate_id,
        record.get("official_manifest") == _resolved(official_manifest),
        record.get("official_manifest_sha256") == manifest_sha256,
        record.get("official_ledger") == _resolved(ledger_path),
        record.get("checkpoint_sha256") == checkpoint_sha256,
        record.get("paper_comparable_full_image") is True,
        record.get("rows") == len(rows) and len(rows) > 0,
        record.get("csv") == str(output.resolve()),
        record.get("summary") == str(summary_path.resolve()),
        record.get("csv_sha256") == sha256_file(output),
        record.get("summary_sha256") == sha256_file(summary_path),
        meta.get("split") == "official_test",
        meta.get("protocol") == protocol,
        meta.get("model") == model_kind,
        meta.get("candidate_id") == candidate_id,
        meta.get("official_manifest") == _resolved(official_manifest),
        meta.get("official_manifest_sha256") == manifest_sha256,
        meta.get("official_ledger") == _resolved(ledger_path),
        meta.get("checkpoint_sha256") == checkpoint_sha256,
        meta.get("paper_comparable_full_image") is True,
        ledger_complete,
    ))


def evaluate_rows(args, cfg: dict, payload: dict) -> list[dict]:
    """Build the model and evaluate after all official preflight checks pass."""
    model = build_model(cfg, args.model).cuda().eval()
    if args.model == "srsc":
        configure_srsc_inference(model, payload, cfg)
    model.load_state_dict(payload.get("model", payload), strict=True)
    if args.split == "locked_val":
        locked = build_locked_val(
            cfg["data_root"], cfg["list_root"], cfg["protocol"], cfg["split_manifest"]
        )
        sets = {"locked_val": locked}
    else:
        sets = build_test_sets(cfg["data_root"], cfg["protocol"])
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for task, dataset in sets.items():
            for item in dataset:
                x = item["degraded"].unsqueeze(0).cuda()
                gt = item["clean"].unsqueeze(0).cuda()
                pred = tiled(model, x, args.tile, args.overlap).float()
                psnr, ssim = metrics(pred, gt)
                rows.append({"task": item.get("task", task), "name": item["name"], "psnr": psnr, "ssim": ssim})
    return rows


def publish_results(
    args,
    cfg: dict,
    checkpoint_sha256: str,
    rows: list[dict],
    *,
    official_metadata: dict | None = None,
) -> tuple[dict, Path, Path]:
    output = Path(args.output)
    summary = summarize_rows(
        rows,
        cfg["protocol"],
        expected_counts=(
            EXPECTED_OFFICIAL_COUNTS[cfg["protocol"]]
            if args.split == "official_test" else None
        ),
    )
    summary["_meta"] = {
        "split": args.split,
        "protocol": cfg["protocol"],
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "tile": args.tile,
        "overlap": args.overlap,
        "paper_comparable_full_image": args.tile == 0,
    }
    if official_metadata:
        summary["_meta"].update(official_metadata)
    json_output = output.with_suffix(".json")
    atomic_write_csv(output, rows)
    atomic_write_json(json_output, summary)
    return summary, output, json_output


def _run_locked_val(args, cfg: dict) -> dict:
    checkpoint = Path(args.checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = evaluate_rows(args, cfg, payload)
    summary, _, _ = publish_results(args, cfg, checkpoint_sha256, rows)
    return summary


def _run_official_test(
    args,
    cfg: dict,
    *,
    config_path: Path,
    config_sha256: str,
) -> dict:
    protocol = cfg["protocol"]
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    manifest_path = Path(args.official_manifest)
    lock_path, ledger_path = official_control_paths(protocol)
    with protocol_file_lock(lock_path):
        _manifest, manifest_sha256, candidate = validate_frozen_official_manifest(
            manifest_path,
            protocol=protocol,
            config_path=config_path,
            config_sha256=config_sha256,
            model_kind=args.model,
            checkpoint_path=checkpoint,
            output_path=output,
        )
        # Eligibility, output collision, and ledger drift checks intentionally
        # precede even checkpoint hashing.  An unlisted or consumed candidate
        # therefore cannot read checkpoint bytes or initialize CUDA.
        assert_official_candidate_available(
            ledger_path,
            protocol=protocol,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
            candidate=candidate,
            output_path=output,
        )
        declared_record = official_record_path(
            protocol, args.model, candidate["checkpoint_sha256"]
        )
        if declared_record.exists():
            raise FileExistsError(
                "this frozen checkpoint already has an official-test completion record: "
                f"{declared_record}"
            )

        # Hash and deserialize through the same open file descriptor to avoid
        # a path-replacement race between digest verification and torch.load.
        with checkpoint.open("rb") as checkpoint_handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
            if checkpoint_sha256 != candidate["checkpoint_sha256"]:
                raise PermissionError(
                    "checkpoint SHA256 does not match the pre-frozen official candidate"
                )
            reserve_official_candidate(
                ledger_path,
                protocol=protocol,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                config_sha256=config_sha256,
                candidate=candidate,
                output_path=output,
            )
            try:
                checkpoint_handle.seek(0)
                payload = torch.load(
                    checkpoint_handle, map_location="cpu", weights_only=False
                )
                validate_checkpoint_contract(payload, candidate)
                rows = evaluate_rows(args, cfg, payload)
                official_metadata = {
                    "candidate_id": candidate["candidate_id"],
                    "official_manifest": _resolved(manifest_path),
                    "official_manifest_sha256": manifest_sha256,
                    "official_ledger": _resolved(ledger_path),
                }
                summary, csv_output, json_output = publish_results(
                    args,
                    cfg,
                    checkpoint_sha256,
                    rows,
                    official_metadata=official_metadata,
                )
                record = dict(summary["_meta"])
                record.update({
                    "status": "COMPLETE",
                    "rows": len(rows),
                    "csv": str(csv_output.resolve()),
                    "csv_sha256": sha256_file(csv_output),
                    "summary": str(json_output.resolve()),
                    "summary_sha256": sha256_file(json_output),
                })
                atomic_write_json(declared_record, record)
                finalize_official_candidate(
                    ledger_path,
                    candidate_id=candidate["candidate_id"],
                    status="COMPLETE",
                    details={
                        "record": _resolved(declared_record),
                        "record_sha256": sha256_file(declared_record),
                        "csv_sha256": record["csv_sha256"],
                        "summary_sha256": record["summary_sha256"],
                    },
                )
            except BaseException as error:
                # STARTED and FAILED are both terminal consumption states.  A
                # power loss may leave STARTED, which is deliberately also not
                # retryable: official-test exposure is a one-shot operation.
                try:
                    finalize_official_candidate(
                        ledger_path,
                        candidate_id=candidate["candidate_id"],
                        status="FAILED",
                        details={"error_type": type(error).__name__},
                    )
                except BaseException:
                    pass
                raise
    return summary


def main():
    args = parse_args()
    if args.split == "official_test" and not args.unlock_official_test:
        raise PermissionError(
            "official test is locked; pass --unlock-official-test only after configuration freeze"
        )
    if args.split == "official_test" and not args.official_manifest:
        raise PermissionError(
            "official test requires --official-manifest with a pre-frozen candidate set"
        )
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    cfg = yaml.safe_load(config_bytes)
    if not isinstance(cfg, dict) or cfg.get("protocol") not in EXPECTED_TASKS:
        raise ValueError("evaluation config must define protocol aio3 or aio5")
    if args.split == "locked_val":
        summary = _run_locked_val(args, cfg)
    else:
        summary = _run_official_test(
            args,
            cfg,
            config_path=config_path,
            config_sha256=config_sha256,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
