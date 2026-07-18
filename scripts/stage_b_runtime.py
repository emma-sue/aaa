#!/usr/bin/env python3
"""Freeze and validate the one permissible Stage-B runtime selection.

This module deliberately separates a hardware-only memory preflight from every
scientific trainer run.  The worker may report only fit/peak-memory evidence;
this parent module selects the first preregistered candidate for which every
required probe passed, writes immutable effective YAML files, and binds them to
one manifest.  Once any Stage-B artifact exists, a missing or invalid manifest
is fatal rather than an invitation to choose a different micro-batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

import yaml


MANIFEST_SCHEMA = "srsc.stage_b_runtime_bundle.v1"
WORKER_RESULT_SCHEMA = "srsc.stage_b_memory_preflight.v1"
REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# The order is scientific state: the selector must take the first candidate
# whose complete probe set passes.  Adding/reordering candidates creates a new
# protocol family and must not be done after any Stage-B artifact exists.
STAGE_B_RUNTIME_CANDIDATES: dict[str, tuple[tuple[int, int], ...]] = {
    "aio3": ((16, 4), (8, 8), (4, 16), (2, 32), (1, 64)),
    "aio5": (
        (15, 8), (12, 10), (10, 12), (8, 15), (6, 20),
        (5, 24), (4, 30), (3, 40), (2, 60), (1, 120),
    ),
}
EXPECTED_EFFECTIVE_BATCH = {"aio3": 64, "aio5": 120}
RUNTIME_ROLES = {"aio3": ("main", "capacity_10_10"), "aio5": ("main",)}


def assert_stage_b_cublas_environment(
    runtime_identity: Mapping[str, object],
) -> None:
    """Bind every frozen Stage-B arm to one deterministic CUDA workspace."""
    if (
        runtime_identity
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "frozen Stage-B runtime requires "
            "CUBLAS_WORKSPACE_CONFIG="
            f"{REQUIRED_STAGE_B_CUBLAS_WORKSPACE_CONFIG!r}"
        )
PREFLIGHT_CODE_RELATIVE_PATHS = (
    "scripts/preflight_stage_b_runtime.py",
    "scripts/train.py",
    "scripts/stage_b_runtime.py",
    "src/net/feedback_controls.py",
    "src/net/srsc_lite.py",
    "src/net/srsc_coordinates.py",
    "src/data/aio_dataset.py",
    "src/losses/objectives.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_key(micro_batch: int, accumulation: int) -> str:
    return f"m{int(micro_batch)}_a{int(accumulation)}"


def _canonical_candidates(protocol: str) -> tuple[tuple[int, int], ...]:
    try:
        return STAGE_B_RUNTIME_CANDIDATES[protocol]
    except KeyError as error:
        raise ValueError(f"unsupported Stage-B runtime protocol: {protocol!r}") from error


def first_all_pass(
    protocol: str,
    results: Mapping[tuple[int, int] | str, bool],
) -> tuple[int, int]:
    """Return the first preregistered candidate marked fully passing.

    Missing entries are failures.  The function intentionally accepts no
    score, peak-memory sorting, or fallback heuristic, which makes it
    impossible to turn scientific metrics into a batch-size search signal.
    """
    for micro_batch, accumulation in _canonical_candidates(protocol):
        passed = results.get((micro_batch, accumulation))
        if passed is None:
            passed = results.get(_candidate_key(micro_batch, accumulation), False)
        if passed is True:
            return micro_batch, accumulation
    raise RuntimeError(f"no preregistered Stage-B runtime candidate passed for {protocol}")


def required_probe_ids(protocol: str) -> tuple[str, ...]:
    probes: list[str] = []
    for role in RUNTIME_ROLES.get(protocol, ()):
        for stage in ("b_oracle", "b_predicted"):
            for feedback in ("O7", "O12"):
                for scope in ("train_step", "native_val"):
                    probes.append(f"{role}:{stage}:{feedback}:{scope}")
    if not probes:
        raise ValueError(f"unsupported Stage-B runtime protocol: {protocol!r}")
    return tuple(probes)


def stage_b_artifact_evidence(root: str | Path, protocol: str) -> tuple[str, ...]:
    """Return artifacts proving that the runtime-selection embargo has begun."""
    root = Path(root).resolve()
    prefixes = (
        f"{protocol}_oracle_",
        f"{protocol}_predicted_",
        f"{protocol}_capacity_10_10_",
    )
    evidence: set[str] = set()

    metrics = root / "artifacts/metrics"
    if metrics.is_dir():
        for path in metrics.rglob("*_locked_val.jsonl"):
            if path.name.startswith(prefixes):
                evidence.add(str(path.resolve()))

    checkpoints = root / "artifacts/checkpoints"
    if checkpoints.is_dir():
        for run_dir in checkpoints.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(prefixes):
                continue
            scientific_files = (
                list(run_dir.glob("*.pt"))
                + list(run_dir.glob("run_contract.json"))
                + list(run_dir.glob("pilot_complete.json"))
                + list(run_dir.glob("formal_complete.json"))
            )
            evidence.update(str(path.resolve()) for path in scientific_files)

    replay_root = root / "artifacts/manifests/replay_digests"
    if replay_root.is_dir():
        for run_dir in replay_root.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith(prefixes):
                evidence.update(str(path.resolve()) for path in run_dir.glob("*.json"))
    return tuple(sorted(evidence))


def assert_stage_b_artifact_embargo_clear(root: str | Path, protocol: str) -> None:
    evidence = stage_b_artifact_evidence(root, protocol)
    if evidence:
        preview = "\n".join(f"- {path}" for path in evidence[:12])
        raise RuntimeError(
            "Stage-B runtime selection is permanently embargoed after the first "
            f"trainer artifact for {protocol}; found:\n{preview}"
        )


@dataclass(frozen=True)
class StageBRuntimeBundle:
    protocol: str
    manifest_path: Path
    manifest_sha256: str
    main_config: Path
    main_config_sha256: str
    capacity_config: Path | None
    capacity_config_sha256: str | None
    micro_batch: int
    accumulation: int
    effective_batch: int
    workers: int
    worker_result: Path
    worker_result_sha256: str

    @property
    def contract_paths(self) -> tuple[Path, ...]:
        paths = [self.manifest_path, self.main_config, self.worker_result]
        if self.capacity_config is not None:
            paths.append(self.capacity_config)
        return tuple(paths)


def _bundle_paths(root: Path, protocol: str) -> dict[str, Path]:
    directory = root / "artifacts/manifests/stage_b_runtime"
    configs = directory / "frozen_configs"
    result = {
        "manifest": directory / f"stage_b_runtime_{protocol}.json",
        "main": configs / f"stage_b_{protocol}.runtime.yaml",
        # Keep the complete no-metric evidence inside the manifest backup
        # surface.  A restored clone must be able to validate the bundle
        # without an untracked artifacts/preflight directory.
        "worker_result": directory / f"stage_b_memory_preflight_{protocol}.json",
    }
    if protocol == "aio3":
        result["capacity_10_10"] = configs / "stage_b_aio3_10_10.runtime.yaml"
    return result


def _template_paths(root: Path, protocol: str) -> dict[str, Path]:
    paths = {"main": root / "configs" / f"stage_b_{protocol}.yaml"}
    if protocol == "aio3":
        paths["capacity_10_10"] = root / "configs/stage_b_aio3_10_10.yaml"
    return paths


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"runtime YAML is not a mapping: {path}")
    return payload


def preflight_code_hashes(root: str | Path) -> dict[str, str]:
    """Hash every implementation file executed by the memory probe."""
    root = Path(root).resolve()
    records: dict[str, str] = {}
    for relative in PREFLIGHT_CODE_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative] = sha256_file(path)
    return records


def preflight_input_bindings(
    root: str | Path,
    protocol: str,
    *,
    stage_a_checkpoint: str | Path,
    coordinate_stats: str | Path,
) -> dict[str, object]:
    """Build the independently reproducible input identity for every probe.

    The AIO-3 10/10 role intentionally has no scientific Stage-A checkpoint or
    dedicated coordinate statistics at memory-preflight time.  It uses random
    coarse weights and the main AIO-3 statistics strictly as a shape surrogate.
    Its future statistics path is therefore bound through the capacity template
    bytes, but the future file is allowed to appear after this bundle freezes.
    """
    root = Path(root).resolve()
    templates = _template_paths(root, protocol)
    stage_a_checkpoint = Path(stage_a_checkpoint).resolve()
    coordinate_stats = Path(coordinate_stats).resolve()
    for path in (stage_a_checkpoint, coordinate_stats, *templates.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    main_cfg = _load_yaml(templates["main"])
    if main_cfg.get("protocol") != protocol:
        raise RuntimeError("Stage-B main template protocol mismatch")
    template_stats = Path(str(main_cfg.get("coordinate_stats", ""))).resolve()
    if template_stats != coordinate_stats:
        raise RuntimeError(
            "Stage-B coordinate-statistics argument does not match the main "
            "template path"
        )
    split = Path(str(main_cfg.get("split_manifest", ""))).resolve()
    if not split.is_file():
        raise FileNotFoundError(split)

    roles: dict[str, dict[str, object]] = {
        "main": {
            "template_path": str(templates["main"].resolve()),
            "template_sha256": sha256_file(templates["main"]),
            "stage_a_checkpoint": str(stage_a_checkpoint),
            "stage_a_checkpoint_sha256": sha256_file(stage_a_checkpoint),
            "init_policy": "COARSE_ONLY_FROM_SELECTED_STAGE_A",
            "coordinate_stats_path": str(coordinate_stats),
            "coordinate_stats_sha256": sha256_file(coordinate_stats),
            "coordinate_stats_origin": "TEMPLATE_BOUND",
        }
    }
    if protocol == "aio3":
        capacity_path = templates["capacity_10_10"]
        capacity_cfg = _load_yaml(capacity_path)
        if capacity_cfg.get("protocol") != protocol:
            raise RuntimeError("Stage-B capacity template protocol mismatch")
        allowed_differences = {"model", "coordinate_stats"}
        unexpected_differences = sorted(
            key
            for key in set(main_cfg) | set(capacity_cfg)
            if key not in allowed_differences
            and main_cfg.get(key) != capacity_cfg.get(key)
        )
        if unexpected_differences:
            raise RuntimeError(
                "AIO3 10/10 Stage-B template differs outside model/statistics: "
                f"{unexpected_differences}"
            )
        declared_stats = Path(
            str(capacity_cfg.get("coordinate_stats", ""))
        ).resolve()
        roles["capacity_10_10"] = {
            "template_path": str(capacity_path.resolve()),
            "template_sha256": sha256_file(capacity_path),
            "stage_a_checkpoint": None,
            "stage_a_checkpoint_sha256": None,
            "init_policy": "MEMORY_ONLY_RANDOM_COARSE",
            "coordinate_stats_path": str(coordinate_stats),
            "coordinate_stats_sha256": sha256_file(coordinate_stats),
            "coordinate_stats_origin": "MAIN_PROTOCOL_VALUES_MEMORY_SHAPE_ONLY",
            "declared_future_coordinate_stats_path": str(declared_stats),
        }

    return {
        "protocol": protocol,
        "stage_a_checkpoint": {
            "path": str(stage_a_checkpoint),
            "sha256": sha256_file(stage_a_checkpoint),
        },
        "coordinate_stats": {
            "path": str(coordinate_stats),
            "sha256": sha256_file(coordinate_stats),
        },
        "split_manifest": {
            "path": str(split),
            "sha256": sha256_file(split),
        },
        "roles": roles,
        "code_sha256": preflight_code_hashes(root),
    }


def _generated_config(
    template: Mapping[str, object],
    *,
    protocol: str,
    role: str,
    manifest_path: Path,
    micro_batch: int,
    accumulation: int,
    workers: int,
) -> dict:
    config = dict(template)
    config.update({
        "micro_batch": int(micro_batch),
        "accumulation": int(accumulation),
        "effective_batch": EXPECTED_EFFECTIVE_BATCH[protocol],
        "workers": int(workers),
        "stage_b_runtime_manifest": str(manifest_path.resolve()),
        "stage_b_runtime_family": protocol,
        "stage_b_runtime_role": role,
    })
    return config


def _serialize_yaml(payload: Mapping[str, object]) -> str:
    return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)


def _exclusive_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _validate_worker_result(
    payload: Mapping[str, object],
    protocol: str,
    *,
    expected_bindings: Mapping[str, object] | None = None,
) -> tuple[tuple[int, int], Sequence[dict]]:
    candidates = [list(pair) for pair in _canonical_candidates(protocol)]
    if payload.get("schema") != WORKER_RESULT_SCHEMA:
        raise RuntimeError("Stage-B memory preflight schema mismatch")
    if payload.get("protocol") != protocol:
        raise RuntimeError("Stage-B memory preflight protocol mismatch")
    if (
        payload.get("scope") != "MEMORY_ONLY"
        or payload.get("scientific_authority") != "NONE"
        or payload.get("official_test_accessed") is not False
        or payload.get("quality_metrics_computed") is not False
        or payload.get("fatal_error") is not None
    ):
        raise RuntimeError(
            "Stage-B memory preflight exceeded its no-metric/no-test authority"
        )
    if payload.get("candidate_order") != candidates:
        raise RuntimeError("Stage-B memory preflight candidate order drift")
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Stage-B memory preflight lacks immutable input bindings")
    if payload.get("inputs_unchanged_through_preflight") is not True:
        raise RuntimeError("Stage-B memory preflight inputs drifted during execution")
    if expected_bindings is not None and bindings != dict(expected_bindings):
        raise RuntimeError("Stage-B memory preflight input-binding mismatch")
    attempts = payload.get("attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or len(attempts) > len(candidates)
    ):
        raise RuntimeError("Stage-B memory preflight attempt sequence is invalid")
    expected_probes = set(required_probe_ids(protocol))
    pass_map: dict[tuple[int, int], bool] = {}
    for expected, attempt in zip(candidates[: len(attempts)], attempts):
        if not isinstance(attempt, dict):
            raise RuntimeError("Stage-B memory preflight attempt is not a mapping")
        pair = [attempt.get("micro_batch"), attempt.get("accumulation")]
        if pair != expected:
            raise RuntimeError("Stage-B memory preflight attempt order drift")
        micro_batch, accumulation = (int(value) for value in expected)
        if int(attempt.get("effective_batch", -1)) != micro_batch * accumulation:
            raise RuntimeError("Stage-B memory preflight effective-batch mismatch")
        probes = attempt.get("probes")
        if not isinstance(probes, list):
            raise RuntimeError("Stage-B memory preflight attempt lacks probes")
        by_id = {
            probe.get("probe_id"): probe
            for probe in probes if isinstance(probe, dict)
        }
        if len(probes) != len(expected_probes) or set(by_id) != expected_probes:
            raise RuntimeError(
                "Stage-B memory preflight probe coverage mismatch: "
                f"missing={sorted(expected_probes - set(by_id))} "
                f"extra={sorted(set(by_id) - expected_probes)}"
            )
        computed = all(probe.get("passed") is True for probe in by_id.values())
        if attempt.get("all_pass") is not computed:
            raise RuntimeError("Stage-B memory preflight all_pass is not probe-derived")
        pass_map[(micro_batch, accumulation)] = computed
    selected = first_all_pass(protocol, pass_map)
    selected_index = candidates.index(list(selected))
    if selected_index != len(attempts) - 1:
        raise RuntimeError(
            "Stage-B memory preflight must stop immediately at first all-pass"
        )
    expected_selected = {
        "micro_batch": selected[0],
        "accumulation": selected[1],
        "effective_batch": selected[0] * selected[1],
    }
    if payload.get("selected") != expected_selected:
        raise RuntimeError("worker selected runtime disagrees with first-all-pass policy")
    if payload.get("selected_candidate") != list(selected):
        raise RuntimeError("worker-selected candidate disagrees with first-all-pass policy")
    if payload.get("all_pass") is not True:
        raise RuntimeError("Stage-B memory preflight did not publish an all-pass result")
    return selected, attempts


def _validate_generated_role(
    *,
    role: str,
    record: Mapping[str, object],
    config_path: Path,
    template_path: Path,
    manifest_path: Path,
    protocol: str,
    micro_batch: int,
    accumulation: int,
    workers: int,
) -> None:
    if str(config_path.resolve()) != record.get("path"):
        raise RuntimeError(f"Stage-B runtime {role} config path mismatch")
    if not config_path.is_file() or sha256_file(config_path) != record.get("sha256"):
        raise RuntimeError(f"Stage-B runtime {role} config SHA256 mismatch")
    if not template_path.is_file() or sha256_file(template_path) != record.get(
        "template_sha256"
    ):
        raise RuntimeError(f"Stage-B runtime {role} template drift")
    template = _load_yaml(template_path)
    expected = _generated_config(
        template,
        protocol=protocol,
        role=role,
        manifest_path=manifest_path,
        micro_batch=micro_batch,
        accumulation=accumulation,
        workers=workers,
    )
    if _load_yaml(config_path) != expected:
        raise RuntimeError(f"Stage-B runtime {role} effective config drift")


def validate_frozen_stage_b_runtime(
    root: str | Path,
    protocol: str,
    *,
    stage_a_checkpoint: str | Path,
    coordinate_stats: str | Path,
) -> StageBRuntimeBundle:
    root = Path(root).resolve()
    paths = _bundle_paths(root, protocol)
    templates = _template_paths(root, protocol)
    manifest_path = paths["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("status") != "FROZEN":
        raise RuntimeError("Stage-B runtime manifest is not a frozen v1 bundle")
    if payload.get("protocol") != protocol:
        raise RuntimeError("Stage-B runtime manifest protocol mismatch")
    expected_candidates = [list(pair) for pair in _canonical_candidates(protocol)]
    if payload.get("candidate_order") != expected_candidates:
        raise RuntimeError("Stage-B runtime manifest candidate order drift")
    if payload.get("required_probe_ids") != list(required_probe_ids(protocol)):
        raise RuntimeError("Stage-B runtime manifest probe contract drift")
    if protocol == "aio3" and payload.get("capacity_preflight_scope") != {
        "role": "capacity_10_10",
        "authority": "MEMORY_ONLY_NOT_SCIENTIFIC_INITIALIZATION",
        "coarse_weights": "RANDOM_STRUCTURE_ONLY",
        "coordinate_statistics": "MAIN_AIO3_VALUES_FOR_MEMORY_SHAPE_ONLY",
        "future_binding": (
            "FORMAL_10_10_RUN_CONTRACT_MUST_BIND_ITS_OWN_STAGE_A_AND_STATS"
        ),
        "batch_reselection": "FORBIDDEN",
    }:
        raise RuntimeError("AIO3 10/10 memory-only preflight disclosure drift")

    stage_a_checkpoint = Path(stage_a_checkpoint).resolve()
    coordinate_stats = Path(coordinate_stats).resolve()
    expected_preflight_bindings = preflight_input_bindings(
        root,
        protocol,
        stage_a_checkpoint=stage_a_checkpoint,
        coordinate_stats=coordinate_stats,
    )
    if payload.get("preflight_input_bindings") != expected_preflight_bindings:
        raise RuntimeError("Stage-B runtime preflight input bindings drift")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Stage-B runtime manifest lacks bindings")
    expected_bindings = {
        "stage_a_checkpoint": (stage_a_checkpoint, sha256_file(stage_a_checkpoint)),
        "coordinate_stats": (coordinate_stats, sha256_file(coordinate_stats)),
        "split_manifest": (
            Path(_load_yaml(templates["main"])["split_manifest"]).resolve(),
            sha256_file(_load_yaml(templates["main"])["split_manifest"]),
        ),
    }
    for key, (path, digest) in expected_bindings.items():
        record = bindings.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"Stage-B runtime manifest lacks {key} binding")
        if record.get("path") != str(path) or record.get("sha256") != digest:
            raise RuntimeError(f"Stage-B runtime manifest {key} binding drift")

    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("Stage-B runtime manifest lacks selected runtime")
    micro_batch = int(selected.get("micro_batch", -1))
    accumulation = int(selected.get("accumulation", -1))
    effective_batch = int(selected.get("effective_batch", -1))
    workers = int(selected.get("workers", -1))
    if (micro_batch, accumulation) not in _canonical_candidates(protocol):
        raise RuntimeError("Stage-B runtime selected pair was not preregistered")
    if micro_batch * accumulation != EXPECTED_EFFECTIVE_BATCH[protocol]:
        raise RuntimeError("Stage-B runtime selected effective batch is invalid")
    if effective_batch != EXPECTED_EFFECTIVE_BATCH[protocol] or workers <= 0:
        raise RuntimeError("Stage-B runtime selected fields are invalid")

    worker_record = payload.get("worker_result")
    if not isinstance(worker_record, dict):
        raise RuntimeError("Stage-B runtime manifest lacks worker result")
    worker_path = Path(str(worker_record.get("path", ""))).resolve()
    if worker_path != paths["worker_result"].resolve() or not worker_path.is_file():
        raise RuntimeError("Stage-B runtime worker-result path drift")
    if sha256_file(worker_path) != worker_record.get("sha256"):
        raise RuntimeError("Stage-B runtime worker-result SHA256 drift")
    worker_payload = json.loads(worker_path.read_text())
    worker_selected, _ = _validate_worker_result(
        worker_payload,
        protocol,
        expected_bindings=expected_preflight_bindings,
    )
    if worker_selected != (micro_batch, accumulation):
        raise RuntimeError("Stage-B runtime selection differs from worker first-all-pass")
    worker_code = Path(str(worker_record.get("worker_code_path", ""))).resolve()
    if (
        not worker_code.is_file()
        or sha256_file(worker_code) != worker_record.get("worker_code_sha256")
    ):
        raise RuntimeError("Stage-B memory worker code drift")

    config_records = payload.get("configs")
    if not isinstance(config_records, dict) or set(config_records) != set(
        RUNTIME_ROLES[protocol]
    ):
        raise RuntimeError("Stage-B runtime generated-config role mismatch")
    for role in RUNTIME_ROLES[protocol]:
        _validate_generated_role(
            role=role,
            record=config_records[role],
            config_path=paths[role],
            template_path=templates[role],
            manifest_path=manifest_path,
            protocol=protocol,
            micro_batch=micro_batch,
            accumulation=accumulation,
            workers=workers,
        )

    return StageBRuntimeBundle(
        protocol=protocol,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=sha256_file(manifest_path),
        main_config=paths["main"].resolve(),
        main_config_sha256=str(config_records["main"]["sha256"]),
        capacity_config=(
            paths["capacity_10_10"].resolve() if protocol == "aio3" else None
        ),
        capacity_config_sha256=(
            str(config_records["capacity_10_10"]["sha256"])
            if protocol == "aio3" else None
        ),
        micro_batch=micro_batch,
        accumulation=accumulation,
        effective_batch=effective_batch,
        workers=workers,
        worker_result=worker_path,
        worker_result_sha256=str(worker_record["sha256"]),
    )


def _preflight_command(
    *,
    root: Path,
    protocol: str,
    stage_a_checkpoint: Path,
    templates: Mapping[str, Path],
    output: Path,
) -> list[str]:
    worker = root / "scripts/preflight_stage_b_runtime.py"
    if not worker.is_file():
        raise FileNotFoundError(
            f"Stage-B memory preflight worker is missing: {worker}"
        )
    command = [
        sys.executable,
        str(worker),
        "--protocol", protocol,
        "--root", str(root),
        "--stage-a-checkpoint", str(stage_a_checkpoint),
        "--main-template", str(templates["main"].resolve()),
        "--candidates-json", json.dumps(
            [list(pair) for pair in _canonical_candidates(protocol)],
            separators=(",", ":"),
        ),
        "--output", str(output.resolve()),
    ]
    if "capacity_10_10" in templates:
        command.extend([
            "--capacity-template", str(templates["capacity_10_10"].resolve())
        ])
    return command


def ensure_stage_b_runtime_bundle(
    root: str | Path,
    protocol: str,
    *,
    stage_a_checkpoint: str | Path,
    coordinate_stats: str | Path,
    runner: Callable[[list[str], str], None],
) -> StageBRuntimeBundle:
    """Create once before Stage-B, otherwise strictly validate and reuse."""
    root = Path(root).resolve()
    stage_a_checkpoint = Path(stage_a_checkpoint).resolve()
    coordinate_stats = Path(coordinate_stats).resolve()
    paths = _bundle_paths(root, protocol)
    templates = _template_paths(root, protocol)
    manifest_path = paths["manifest"]
    if manifest_path.is_file():
        return validate_frozen_stage_b_runtime(
            root,
            protocol,
            stage_a_checkpoint=stage_a_checkpoint,
            coordinate_stats=coordinate_stats,
        )

    assert_stage_b_artifact_embargo_clear(root, protocol)
    if any(paths[role].exists() for role in RUNTIME_ROLES[protocol]):
        raise RuntimeError(
            "orphaned Stage-B runtime config exists without its frozen manifest; "
            "refusing an implicit reselection"
        )
    for path in (stage_a_checkpoint, coordinate_stats, *templates.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    bindings_before = preflight_input_bindings(
        root,
        protocol,
        stage_a_checkpoint=stage_a_checkpoint,
        coordinate_stats=coordinate_stats,
    )

    worker_output = paths["worker_result"]
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    if worker_output.exists():
        raise RuntimeError(
            "orphaned Stage-B memory-preflight output exists without a frozen "
            "manifest; archive it before a new preregistered family"
        )
    command = _preflight_command(
        root=root,
        protocol=protocol,
        stage_a_checkpoint=stage_a_checkpoint,
        templates=templates,
        output=worker_output,
    )
    runner(command, f"{protocol}_stage_b_runtime_preflight.log")
    if not worker_output.is_file():
        raise RuntimeError("Stage-B memory preflight did not atomically publish output")
    bindings_after = preflight_input_bindings(
        root,
        protocol,
        stage_a_checkpoint=stage_a_checkpoint,
        coordinate_stats=coordinate_stats,
    )
    if bindings_after != bindings_before:
        raise RuntimeError(
            "Stage-B memory-preflight inputs drifted while the probe was running"
        )
    worker_payload = json.loads(worker_output.read_text())
    (micro_batch, accumulation), attempts = _validate_worker_result(
        worker_payload,
        protocol,
        expected_bindings=bindings_after,
    )
    workers_by_role = {
        int(_load_yaml(path).get("workers", -1)) for path in templates.values()
    }
    if len(workers_by_role) != 1 or next(iter(workers_by_role)) <= 0:
        raise RuntimeError("Stage-B templates do not share one positive worker count")
    workers = next(iter(workers_by_role))

    created: list[Path] = []
    try:
        config_records: dict[str, dict] = {}
        for role in RUNTIME_ROLES[protocol]:
            generated = _generated_config(
                _load_yaml(templates[role]),
                protocol=protocol,
                role=role,
                manifest_path=manifest_path,
                micro_batch=micro_batch,
                accumulation=accumulation,
                workers=workers,
            )
            _exclusive_write(paths[role], _serialize_yaml(generated))
            created.append(paths[role])
            config_records[role] = {
                "path": str(paths[role].resolve()),
                "sha256": sha256_file(paths[role]),
                "template_path": str(templates[role].resolve()),
                "template_sha256": sha256_file(templates[role]),
            }

        worker = root / "scripts/preflight_stage_b_runtime.py"
        main_template = _load_yaml(templates["main"])
        split = Path(str(main_template["split_manifest"])).resolve()
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "FROZEN",
            "protocol": protocol,
            "created_utc": utc_now(),
            "selection_policy": "FIRST_PREREGISTERED_CANDIDATE_WITH_ALL_PROBES_PASSING",
            "candidate_order": [list(pair) for pair in _canonical_candidates(protocol)],
            "required_probe_ids": list(required_probe_ids(protocol)),
            "selected": {
                "micro_batch": micro_batch,
                "accumulation": accumulation,
                "effective_batch": EXPECTED_EFFECTIVE_BATCH[protocol],
                "workers": workers,
                "precision": main_template.get("precision"),
                "crop_size": main_template.get("crop_size"),
            },
            "bindings": {
                "stage_a_checkpoint": {
                    "path": str(stage_a_checkpoint),
                    "sha256": sha256_file(stage_a_checkpoint),
                },
                "coordinate_stats": {
                    "path": str(coordinate_stats),
                    "sha256": sha256_file(coordinate_stats),
                },
                "split_manifest": {
                    "path": str(split),
                    "sha256": sha256_file(split),
                },
            },
            "worker_result": {
                "path": str(worker_output.resolve()),
                "sha256": sha256_file(worker_output),
                "worker_code_path": str(worker.resolve()),
                "worker_code_sha256": sha256_file(worker),
            },
            "preflight_input_bindings": bindings_after,
            "configs": config_records,
            "attempts": attempts,
            "capacity_preflight_scope": (
                {
                    "role": "capacity_10_10",
                    "authority": "MEMORY_ONLY_NOT_SCIENTIFIC_INITIALIZATION",
                    "coarse_weights": "RANDOM_STRUCTURE_ONLY",
                    "coordinate_statistics": (
                        "MAIN_AIO3_VALUES_FOR_MEMORY_SHAPE_ONLY"
                    ),
                    "future_binding": (
                        "FORMAL_10_10_RUN_CONTRACT_MUST_BIND_ITS_OWN_STAGE_A_AND_STATS"
                    ),
                    "batch_reselection": "FORBIDDEN",
                }
                if protocol == "aio3" else None
            ),
            "metric_embargo_check": {
                "checked_utc": utc_now(),
                "evidence": [],
                "rule": "NO_STAGE_B_METRIC_RUN_CONTRACT_CHECKPOINT_OR_REPLAY_DIGEST",
            },
        }
        _exclusive_write(manifest_path, json.dumps(
            manifest, indent=2, sort_keys=True
        ) + "\n")
        created.append(manifest_path)
    except BaseException:
        # This is a single-process transaction under the orchestrator GPU lock.
        # Remove only files created by this invocation; never touch preexisting
        # scientific artifacts or the worker's durable diagnostic result.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise

    return validate_frozen_stage_b_runtime(
        root,
        protocol,
        stage_a_checkpoint=stage_a_checkpoint,
        coordinate_stats=coordinate_stats,
    )


def runtime_identity_for_config(
    config_path: str | Path,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return immutable fields that Stage-B run contracts must carry."""
    config_path = Path(config_path).resolve()
    cfg = dict(config) if config is not None else _load_yaml(config_path)
    manifest_value = cfg.get("stage_b_runtime_manifest")
    if manifest_value is None:
        return {}
    manifest_path = Path(str(manifest_value)).resolve()
    if not manifest_path.is_file():
        raise RuntimeError("Stage-B runtime manifest referenced by config is missing")
    manifest = json.loads(manifest_path.read_text())
    family = cfg.get("stage_b_runtime_family")
    role = cfg.get("stage_b_runtime_role")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "FROZEN"
        or manifest.get("protocol") != family
        or role not in RUNTIME_ROLES.get(str(family), ())
    ):
        raise RuntimeError("Stage-B config/runtime manifest identity mismatch")
    record = manifest.get("configs", {}).get(role)
    if not isinstance(record, dict):
        raise RuntimeError("Stage-B runtime manifest lacks config role")
    if record.get("path") != str(config_path) or record.get("sha256") != sha256_file(
        config_path
    ):
        raise RuntimeError("Stage-B runtime manifest does not bind this config")
    template_path = Path(str(record.get("template_path", ""))).resolve()
    if (
        not template_path.is_file()
        or sha256_file(template_path) != record.get("template_sha256")
    ):
        raise RuntimeError("Stage-B runtime template binding drift")
    worker_record = manifest.get("worker_result")
    if not isinstance(worker_record, dict):
        raise RuntimeError("Stage-B runtime manifest lacks worker evidence")
    worker_result = Path(str(worker_record.get("path", ""))).resolve()
    worker_code = Path(str(worker_record.get("worker_code_path", ""))).resolve()
    if (
        not worker_result.is_file()
        or sha256_file(worker_result) != worker_record.get("sha256")
        or not worker_code.is_file()
        or sha256_file(worker_code) != worker_record.get("worker_code_sha256")
    ):
        raise RuntimeError("Stage-B runtime worker evidence drift")
    selected = manifest.get("selected", {})
    for key in ("micro_batch", "accumulation", "effective_batch", "workers"):
        if cfg.get(key) != selected.get(key):
            raise RuntimeError(f"Stage-B runtime config field drift: {key}")
    worker_record = manifest.get("worker_result")
    if not isinstance(worker_record, dict):
        raise RuntimeError("Stage-B runtime manifest lacks worker-result binding")
    worker_path = Path(str(worker_record.get("path", ""))).resolve()
    if (
        not worker_path.is_file()
        or sha256_file(worker_path) != worker_record.get("sha256")
    ):
        raise RuntimeError("Stage-B runtime worker-result binding drift")
    worker_code = Path(str(worker_record.get("worker_code_path", ""))).resolve()
    if (
        not worker_code.is_file()
        or sha256_file(worker_code) != worker_record.get("worker_code_sha256")
    ):
        raise RuntimeError("Stage-B runtime worker-code binding drift")
    worker_selected, _attempts = _validate_worker_result(
        json.loads(worker_path.read_text()), str(family)
    )
    if worker_selected != (
        int(selected.get("micro_batch", -1)),
        int(selected.get("accumulation", -1)),
    ):
        raise RuntimeError("Stage-B runtime worker/manifest selection drift")
    template_path = Path(str(record.get("template_path", ""))).resolve()
    _validate_generated_role(
        role=str(role),
        record=record,
        config_path=config_path,
        template_path=template_path,
        manifest_path=manifest_path,
        protocol=str(family),
        micro_batch=int(selected.get("micro_batch", -1)),
        accumulation=int(selected.get("accumulation", -1)),
        workers=int(selected.get("workers", -1)),
    )
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Stage-B runtime manifest lacks prerequisite bindings")
    for name in ("stage_a_checkpoint", "coordinate_stats", "split_manifest"):
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise RuntimeError(f"Stage-B runtime manifest lacks {name} binding")
        bound_path = Path(str(binding.get("path", ""))).resolve()
        if (
            not bound_path.is_file()
            or sha256_file(bound_path) != binding.get("sha256")
        ):
            raise RuntimeError(f"Stage-B runtime prerequisite drift: {name}")
    return {
        "stage_b_runtime_family": family,
        "stage_b_runtime_role": role,
        "stage_b_runtime_manifest_path": str(manifest_path),
        "stage_b_runtime_manifest_sha256": sha256_file(manifest_path),
    }


def assert_no_runtime_worker_override(
    config: Mapping[str, object], environment: Mapping[str, str] | None = None,
) -> None:
    """Forbid ambient worker changes for a frozen Stage-B config."""
    if "stage_b_runtime_manifest" not in config:
        return
    environment = os.environ if environment is None else environment
    if str(environment.get("SRSC_TRAIN_WORKERS", "")).strip():
        raise RuntimeError(
            "SRSC_TRAIN_WORKERS is forbidden for a frozen Stage-B runtime; "
            "the generated YAML owns the worker count"
        )
