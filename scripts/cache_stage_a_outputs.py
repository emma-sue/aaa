#!/usr/bin/env python3
"""Build an audited, locked-validation-only Stage-A ``y1`` cache.

The cache is deliberately narrower than a generic prediction cache:

* only the preregistered ``locked_val`` split is supported;
* tensors are stored losslessly as contiguous FP32 ``.npy`` shards;
* every input, target, output, shard and source contract is SHA256-bound;
* a new cache is committed only after a second full online replay agrees
  bit-for-bit with every staged shard;
* an existing cache is fully checked on CPU and returned idempotently;
* no official-test dataset builder is imported or reachable.

Full dynamic train caching and official-test caching are intentionally not
implemented.  Their formal replacement is documented in
``reports/CACHE_CONTRACT_REVISION_V1.md``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import only the held-out training split constructor.  Importing or calling
# build_test_sets here would create an unaudited path around the official-test
# one-shot ledger in scripts/eval_locked.py.
from src.data.aio_dataset import build_locked_val  # noqa: E402
from src.net.restormer_blocks import crop_to_shape, pad_to_multiple  # noqa: E402


CACHE_SCHEMA_VERSION = 1
CACHE_SCOPE = "locked_val"
CACHE_STATUS = "COMPLETE_TWO_PASS_VERIFIED"

MODEL_CODE_PATHS = (
    ROOT / "src/net/srsc_lite.py",
    ROOT / "src/net/restormer_blocks.py",
    ROOT / "src/net/feedback_controls.py",
    ROOT / "src/data/aio_dataset.py",
    ROOT / "scripts/train.py",
    Path(__file__),
)


class LockedDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


DatasetFactory = Callable[[], LockedDataset]
Predictor = Callable[[torch.Tensor], torch.Tensor]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def validate_cache_scope(scope: str) -> str:
    if scope != CACHE_SCOPE:
        raise PermissionError(
            "Stage-A output caching is locked_val-only; official_test is never "
            "reachable from this tool"
        )
    return scope


def validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lowered = tuple(part.lower() for part in resolved.parts)
    if any(
        "official_test" in part or "official-test" in part for part in lowered
    ):
        raise PermissionError("cache output path must not target official_test")
    return resolved


def tensor_to_float32_array(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    array = tensor.detach().float().cpu().contiguous().numpy()
    # Normalize byte order as part of the portable on-disk contract.
    return np.ascontiguousarray(array.astype("<f4", copy=False))


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def write_npy_atomic(path: Path, array: np.ndarray) -> str:
    if array.dtype != np.dtype("<f4") or not array.flags.c_contiguous:
        raise ValueError("cache shards must be contiguous little-endian FP32 arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def load_npy_strict(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
        if handle.read(1):
            raise RuntimeError(f"trailing bytes in cache shard: {path}")
    if array.dtype != np.dtype("<f4"):
        raise RuntimeError(f"cache shard is not little-endian FP32: {path}")
    if not array.flags.c_contiguous:
        raise RuntimeError(f"cache shard is not contiguous: {path}")
    return array


def stable_item_key(item: Mapping[str, object]) -> tuple[str, str, str]:
    task = str(item.get("task", ""))
    name = str(item.get("name", ""))
    if not task or not name:
        raise ValueError("locked_val cache items require non-empty task and name")
    key = f"{task}\0{name}"
    return task, name, sha256_bytes(key.encode("utf-8"))


def validate_item_tensors(item: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    degraded = item.get("degraded")
    clean = item.get("clean")
    if not isinstance(degraded, torch.Tensor) or not isinstance(clean, torch.Tensor):
        raise TypeError("locked_val item must contain degraded/clean tensors")
    if degraded.ndim != 3 or clean.ndim != 3 or degraded.shape != clean.shape:
        raise ValueError(
            f"invalid locked pair shapes: {tuple(degraded.shape)} vs {tuple(clean.shape)}"
        )
    if degraded.shape[0] != 3:
        raise ValueError(f"locked cache expects RGB CHW tensors: {tuple(degraded.shape)}")
    degraded_array = tensor_to_float32_array(degraded)
    clean_array = tensor_to_float32_array(clean)
    if not np.isfinite(degraded_array).all() or not np.isfinite(clean_array).all():
        raise FloatingPointError("locked cache input/GT contains NaN or Inf")
    return degraded_array, clean_array


def cache_aggregate_sha256(bindings: Mapping[str, object], items: list[dict]) -> str:
    material = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scope": CACHE_SCOPE,
        "bindings": bindings,
        "items": items,
    }
    return sha256_bytes(canonical_json_bytes(material))


def _manifest_without_time(bindings: Mapping[str, object], items: list[dict]) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scope": CACHE_SCOPE,
        "status": CACHE_STATUS,
        "official_test_forbidden": True,
        "storage": {
            "format": "npy",
            "dtype": "float32-little-endian",
            "clamped": False,
            "lossless": True,
            "sharding": "one variable-shape tensor per locked observation",
        },
        "verification": {
            "creation_passes": 2,
            "second_pass": "full online model replay; exact array equality",
            "existing_cache": "full CPU contract/input/GT/shard verification",
        },
        "bindings": dict(bindings),
        "item_count": len(items),
        "items": items,
        "aggregate_sha256": cache_aggregate_sha256(bindings, items),
    }


def _write_staged_manifest(staging: Path, manifest: dict) -> None:
    committed = dict(manifest)
    committed["created_at_utc"] = utc_now()
    manifest_path = staging / "manifest.json"
    atomic_write_json(manifest_path, committed)
    atomic_write_bytes(
        staging / "manifest.sha256",
        (sha256_file(manifest_path) + "  manifest.json\n").encode("ascii"),
    )


def _read_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "manifest.json"
    digest_path = cache_dir / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise RuntimeError(f"partial cache transaction: {cache_dir}")
    fields = digest_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != "manifest.json":
        raise RuntimeError(f"invalid cache manifest digest sidecar: {digest_path}")
    if fields[0] != sha256_file(manifest_path):
        raise RuntimeError(f"cache manifest SHA256 mismatch: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid cache manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("cache manifest must be a JSON object")
    return manifest


def verify_cache_offline(
    cache_dir: Path,
    bindings: Mapping[str, object],
    dataset: LockedDataset,
) -> dict:
    """Verify contract, source tensors and every shard without a GPU/model."""
    cache_dir = validate_output_path(cache_dir)
    manifest = _read_manifest(cache_dir)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("unsupported Stage-A output cache schema")
    if manifest.get("scope") != CACHE_SCOPE or manifest.get("status") != CACHE_STATUS:
        raise RuntimeError("cache is not a complete locked_val two-pass transaction")
    if manifest.get("official_test_forbidden") is not True:
        raise RuntimeError("cache lost the official-test prohibition")
    if manifest.get("storage") != _manifest_without_time(bindings, [])["storage"]:
        raise RuntimeError("cache storage contract drift")
    if manifest.get("verification") != _manifest_without_time(bindings, [])["verification"]:
        raise RuntimeError("cache verification contract drift")
    if manifest.get("bindings") != dict(bindings):
        raise RuntimeError("Stage-A output cache binding drift")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != len(dataset):
        raise RuntimeError("cache item count does not match locked_val")
    if manifest.get("item_count") != len(items):
        raise RuntimeError("cache manifest item_count mismatch")
    if manifest.get("aggregate_sha256") != cache_aggregate_sha256(bindings, items):
        raise RuntimeError("cache aggregate SHA256 mismatch")

    expected_files = set()
    seen_keys = set()
    for index, record in enumerate(items):
        if not isinstance(record, dict) or int(record.get("index", -1)) != index:
            raise RuntimeError(f"invalid cache record at index {index}")
        item = dataset[index]
        task, name, key_sha = stable_item_key(item)
        degraded, clean = validate_item_tensors(item)
        expected_identity = {
            "task": task,
            "name": name,
            "key_sha256": key_sha,
            "input_sha256": array_sha256(degraded),
            "gt_sha256": array_sha256(clean),
        }
        for key, value in expected_identity.items():
            if record.get(key) != value:
                raise RuntimeError(f"cache {key} drift at index {index}")
        if key_sha in seen_keys:
            raise RuntimeError(f"duplicate cache item identity: {task}/{name}")
        seen_keys.add(key_sha)

        relative = Path(str(record.get("shard", "")))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("shards",):
            raise RuntimeError(f"unsafe cache shard path: {relative}")
        shard = cache_dir / relative
        expected_files.add(relative.as_posix())
        if not shard.is_file() or sha256_file(shard) != record.get("shard_sha256"):
            raise RuntimeError(f"cache shard SHA256 mismatch: {shard}")
        y1 = load_npy_strict(shard)
        if record.get("dtype") != "float32":
            raise RuntimeError(f"cache record dtype mismatch: {shard}")
        if list(y1.shape) != record.get("shape"):
            raise RuntimeError(f"cache shard shape mismatch: {shard}")
        if array_sha256(y1) != record.get("y1_sha256"):
            raise RuntimeError(f"cache y1 tensor SHA256 mismatch: {shard}")
        if y1.shape != degraded.shape:
            raise RuntimeError(f"cache y1/input shape mismatch at index {index}")
        if not np.isfinite(y1).all():
            raise FloatingPointError(f"cache y1 contains NaN or Inf: {shard}")

    actual_files = {
        path.relative_to(cache_dir).as_posix()
        for path in (cache_dir / "shards").glob("**/*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(
            "cache shard set mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )
    allowed_root_files = {"manifest.json", "manifest.sha256"}
    unexpected_root = {
        path.name for path in cache_dir.iterdir()
        if path.is_file() and path.name not in allowed_root_files
    }
    if unexpected_root:
        raise RuntimeError(f"unexpected files in cache root: {sorted(unexpected_root)}")
    unexpected_directories = {
        path.name for path in cache_dir.iterdir()
        if path.is_dir() and path.name != "shards"
    }
    if unexpected_directories:
        raise RuntimeError(
            f"unexpected directories in cache root: {sorted(unexpected_directories)}"
        )
    return manifest


def verify_cache_online(
    cache_dir: Path,
    manifest: Mapping[str, object],
    dataset: LockedDataset,
    predict_y1: Predictor,
) -> None:
    """Replay the selected Stage-A operator over every held-out observation."""
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != len(dataset):
        raise RuntimeError("online replay item count mismatch")
    for index, record in enumerate(items):
        item = dataset[index]
        degraded, clean = validate_item_tensors(item)
        if array_sha256(degraded) != record.get("input_sha256"):
            raise RuntimeError(f"online replay input drift at index {index}")
        if array_sha256(clean) != record.get("gt_sha256"):
            raise RuntimeError(f"online replay GT drift at index {index}")
        prediction = tensor_to_float32_array(
            predict_y1(torch.from_numpy(degraded.copy()))
        )
        shard = cache_dir / str(record["shard"])
        cached = load_npy_strict(shard)
        if prediction.shape != cached.shape or not np.array_equal(prediction, cached):
            maximum = (
                float(np.max(np.abs(prediction - cached)))
                if prediction.shape == cached.shape and prediction.size else float("inf")
            )
            raise RuntimeError(
                f"second-pass Stage-A replay mismatch at index {index}: max_abs={maximum}"
            )
        if array_sha256(prediction) != record.get("y1_sha256"):
            raise RuntimeError(f"online replay y1 SHA256 drift at index {index}")


@contextmanager
def cache_creation_lock(output_dir: Path):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Stage-A cache transaction is already active: {output_dir}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def ensure_locked_val_cache(
    output_dir: str | Path,
    bindings: Mapping[str, object],
    dataset_factory: DatasetFactory,
    predict_y1: Predictor,
    *,
    online_reverify_existing: bool = False,
) -> tuple[dict, bool]:
    """Create or validate a cache; return ``(manifest, created)``.

    Creation is a two-pass transaction.  Existing caches take the CPU-only
    path unless explicit online reverification is requested.
    """
    validate_cache_scope(CACHE_SCOPE)
    output_dir = validate_output_path(Path(output_dir))
    with cache_creation_lock(output_dir):
        if output_dir.exists():
            if not output_dir.is_dir():
                raise RuntimeError(f"cache output exists but is not a directory: {output_dir}")
            manifest = verify_cache_offline(output_dir, bindings, dataset_factory())
            if online_reverify_existing:
                verify_cache_online(
                    output_dir, manifest, dataset_factory(), predict_y1
                )
            return manifest, False

        staging = output_dir.parent / (
            f".{output_dir.name}.staging.{os.getpid()}.{time.time_ns()}"
        )
        if staging.exists():
            raise RuntimeError(f"refusing pre-existing cache staging directory: {staging}")
        staging.mkdir(parents=False)
        (staging / "shards").mkdir()
        try:
            dataset = dataset_factory()
            items: list[dict] = []
            seen_keys = set()
            for index in range(len(dataset)):
                item = dataset[index]
                task, name, key_sha = stable_item_key(item)
                if key_sha in seen_keys:
                    raise RuntimeError(f"duplicate locked cache identity: {task}/{name}")
                seen_keys.add(key_sha)
                degraded, clean = validate_item_tensors(item)
                y1 = tensor_to_float32_array(
                    predict_y1(torch.from_numpy(degraded.copy()))
                )
                if y1.shape != degraded.shape:
                    raise RuntimeError(
                        f"Stage-A y1 shape mismatch at index {index}: "
                        f"{tuple(y1.shape)} vs {tuple(degraded.shape)}"
                    )
                relative = Path("shards") / f"{index:06d}_{key_sha[:16]}.npy"
                shard = staging / relative
                shard_sha = write_npy_atomic(shard, y1)
                items.append({
                    "index": index,
                    "task": task,
                    "name": name,
                    "key_sha256": key_sha,
                    "input_sha256": array_sha256(degraded),
                    "gt_sha256": array_sha256(clean),
                    "shape": list(y1.shape),
                    "dtype": "float32",
                    "y1_sha256": array_sha256(y1),
                    "shard": relative.as_posix(),
                    "shard_sha256": shard_sha,
                })

            manifest = _manifest_without_time(bindings, items)
            # A second independently constructed dataset traversal and a full
            # model replay are mandatory before any complete marker exists.
            verify_cache_online(staging, manifest, dataset_factory(), predict_y1)
            _write_staged_manifest(staging, manifest)
            verify_cache_offline(staging, bindings, dataset_factory())
            fsync_directory(staging / "shards")
            fsync_directory(staging)
            os.replace(staging, output_dir)
            fsync_directory(output_dir.parent)
            committed = verify_cache_offline(output_dir, bindings, dataset_factory())
            return committed, True
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _verified_list_bindings(cfg: Mapping[str, object], split_payload: dict) -> dict:
    declared = split_payload.get("list_sha256")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError("locked split manifest has no list_sha256 contract")
    list_root = Path(str(cfg["list_root"])).expanduser().resolve()
    result = {}
    for relative, expected in sorted(declared.items()):
        path = (list_root / relative).resolve()
        if list_root not in path.parents:
            raise RuntimeError(f"unsafe list path in split manifest: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"dataset list SHA256 drift: {path}")
        result[relative] = {"path": str(path), "sha256": actual}
    return result


def _selected_checkpoint_binding(checkpoint: Path, payload: dict, cfg: dict) -> dict:
    if payload.get("validation_pending") is not None:
        raise RuntimeError("selected Stage-A checkpoint has pending validation")
    checkpoint_args = payload.get("args", {})
    if checkpoint_args.get("stage") != "a":
        raise RuntimeError("Stage-A output cache requires a Stage-A checkpoint")
    if payload.get("config") != cfg:
        raise RuntimeError("Stage-A checkpoint effective config drift")
    run_dir = checkpoint.parent
    evidence: dict[str, object] = {}
    if payload.get("checkpoint_kind") == "formal_locked_val_best_model_only":
        record = payload.get("selected_top3_record")
        if not isinstance(record, dict):
            raise RuntimeError("formal Stage-A checkpoint lost selected_top3_record")
        marker = run_dir / "formal_complete.json"
        if not marker.is_file():
            raise RuntimeError("formal Stage-A checkpoint has no completion marker")
        marker_payload = json.loads(marker.read_text())
        if int(marker_payload.get("completed_epochs", -1)) != int(cfg["epochs"]):
            raise RuntimeError("formal Stage-A completion epoch drift")
        if marker_payload.get("selected_top3_record") != record:
            raise RuntimeError("formal Stage-A selection marker drift")
        evidence = {
            "selection_kind": "formal_locked_val_best_model_only",
            "selected_top3_record": record,
            "formal_complete_path": str(marker.resolve()),
            "formal_complete_sha256": sha256_file(marker),
        }
    else:
        top3 = run_dir / "top3.json"
        if not top3.is_file():
            raise RuntimeError("selected Stage-A checkpoint has no top3 index")
        records = json.loads(top3.read_text())
        if not isinstance(records, list) or not records:
            raise RuntimeError("selected Stage-A top3 index is empty")
        record = records[0]
        if record.get("checkpoint") != checkpoint.name:
            raise RuntimeError("checkpoint is not the current locked-val best Stage-A model")
        if int(record.get("epoch", -1)) != int(payload.get("epoch", -2)):
            raise RuntimeError("selected Stage-A epoch disagrees with top3")
        if int(record.get("step", -1)) != int(payload.get("step", -2)):
            raise RuntimeError("selected Stage-A step disagrees with top3")
        run_name = checkpoint_args.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            raise RuntimeError("selected Stage-A checkpoint lost its run name")
        metrics = ROOT / "artifacts/metrics" / f"{run_name}_locked_val.jsonl"
        if not metrics.is_file():
            raise RuntimeError("selected Stage-A checkpoint has no locked-val ledger")
        metric_rows = [
            json.loads(line) for line in metrics.read_text().splitlines()
            if line.strip()
        ]
        if not metric_rows or not any(
            int(row.get("epoch", -1)) == int(cfg["epochs"])
            for row in metric_rows
        ):
            raise RuntimeError("Stage-A has not committed its final locked validation")
        selected_metric = max(
            metric_rows, key=lambda row: float(row["macro_psnr"])
        )
        if (
            int(selected_metric.get("epoch", -1)) != int(record["epoch"])
            or int(selected_metric.get("step", -1)) != int(record["step"])
            or float(selected_metric.get("macro_psnr")) != float(record["score"])
        ):
            raise RuntimeError("top3 rank zero is not the locked-val-best metric row")
        evidence = {
            "selection_kind": "top3_locked_val_rank_0",
            "selected_top3_record": record,
            "top3_path": str(top3.resolve()),
            "top3_sha256": sha256_file(top3),
            "locked_val_ledger_path": str(metrics.resolve()),
            "locked_val_ledger_sha256": sha256_file(metrics),
            "selected_locked_val": selected_metric,
        }
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "step": int(payload.get("step", -1)),
        **evidence,
    }


def build_runtime_bindings(
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[dict, dict, dict]:
    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    if not isinstance(cfg, dict) or cfg.get("protocol") not in {"aio3", "aio5"}:
        raise ValueError("cache config must define protocol aio3 or aio5")
    if cfg.get("official_test_locked") is not True:
        raise PermissionError("cache construction requires official_test_locked=true")
    split_path = Path(str(cfg["split_manifest"])).expanduser().resolve()
    split_bytes = split_path.read_bytes()
    split_payload = json.loads(split_bytes)
    if split_payload.get("protocol") != cfg["protocol"]:
        raise RuntimeError("locked split protocol mismatch")

    checkpoint_sha = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("formal Stage-A checkpoint must contain a model state")
    config_sha = sha256_bytes(config_bytes)
    split_sha = sha256_bytes(split_bytes)
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("Stage-A checkpoint config SHA256 drift")
    if payload.get("split_manifest_sha256") != split_sha:
        raise RuntimeError("Stage-A checkpoint split SHA256 drift")
    selected = _selected_checkpoint_binding(checkpoint_path, payload, cfg)
    if selected["sha256"] != checkpoint_sha:
        raise RuntimeError("checkpoint changed during cache preflight")

    bindings = {
        "protocol": cfg["protocol"],
        "scope": CACHE_SCOPE,
        "config": {"path": str(config_path), "sha256": config_sha},
        "split_manifest": {"path": str(split_path), "sha256": split_sha},
        "dataset_lists": _verified_list_bindings(cfg, split_payload),
        "stage_a_checkpoint": selected,
        "model_code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in MODEL_CODE_PATHS
        },
        "forward_contract": {
            "operator": "selected frozen encoder + D1",
            "padding_multiple": 8,
            "autocast": "cuda_bfloat16",
            "stored_dtype": "float32",
            "clamp": False,
            "target_used_by_model": False,
        },
    }
    return cfg, payload, bindings


class LazyCudaStageAPredictor:
    """Load the selected model only if creation/online replay actually needs it."""

    def __init__(self, cfg: dict, payload: dict):
        self.cfg = cfg
        self.payload = payload
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to reproduce the registered BF16 forward")
        # Imported lazily so CPU-only validation of an existing cache never
        # builds the full model or initializes CUDA.
        from scripts.train import build_model

        model = build_model(self.cfg, "a").cuda().eval()
        model.load_state_dict(self.payload["model"], strict=True)
        self.model = model
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def __call__(self, degraded: torch.Tensor) -> torch.Tensor:
        self._load()
        assert self.model is not None
        if degraded.ndim != 3:
            raise ValueError("Stage-A predictor expects one CHW image")
        x = degraded.unsqueeze(0).cuda(non_blocking=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            padded, original = pad_to_multiple(x, 8)
            features = self.model.encoder(padded)
            delta0, _ = self.model.d1(features)
            y1 = crop_to_shape(padded + delta0, original)
        return y1.squeeze(0).float().cpu()


def default_output_dir(protocol: str, checkpoint_sha256: str) -> Path:
    return (
        ROOT / "artifacts/cache/stage_a_y1" / protocol
        / checkpoint_sha256[:16] / CACHE_SCOPE
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-a-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="content-bound cache directory; defaults below artifacts/cache/stage_a_y1",
    )
    parser.add_argument(
        "--online-reverify-existing",
        action="store_true",
        help="also repeat the full CUDA Stage-A replay for an already valid cache",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_cache_scope(CACHE_SCOPE)
    cfg, payload, bindings = build_runtime_bindings(
        args.config, args.stage_a_checkpoint
    )
    checkpoint_sha = bindings["stage_a_checkpoint"]["sha256"]
    output_dir = validate_output_path(
        args.output_dir
        if args.output_dir is not None
        else default_output_dir(cfg["protocol"], checkpoint_sha)
    )
    def dataset_factory():
        return build_locked_val(
            cfg["data_root"],
            cfg["list_root"],
            cfg["protocol"],
            cfg["split_manifest"],
        )
    predictor = LazyCudaStageAPredictor(cfg, payload)
    manifest, created = ensure_locked_val_cache(
        output_dir,
        bindings,
        dataset_factory,
        predictor,
        online_reverify_existing=args.online_reverify_existing,
    )
    print(json.dumps({
        "status": "CREATED" if created else "VERIFIED_IDEMPOTENT",
        "scope": CACHE_SCOPE,
        "output_dir": str(output_dir),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "item_count": manifest["item_count"],
        "online_reverified_existing": bool(
            not created and args.online_reverify_existing
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
