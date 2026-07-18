#!/usr/bin/env python3
"""Exact four-GPU AIO-3 Stage-A trainer for the 10/10 capacity control.

This is an intentionally isolated adapter around the already audited hybrid
DDP engine.  It reuses that engine's canonical per-update raw-index schedule,
safe four-rank micro-batches, transaction-safe validation, resume cursors and
cross-rank integrity checks.  It does *not* widen the clean/matched baseline
trainer's public stage set or alter either baseline run contract.

Only Encoder+D1 participate in the loss.  The DDP adapter nevertheless owns
the complete SRSCLite module so DDP broadcasts every frozen parameter/buffer
before the engine hashes the full rank-zero state.  Checkpoints are written
from the unwrapped SRSCLite object and therefore retain the exact, prefix-free
state_dict expected by Stage-B ``load_formal_init``.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml
from torch import nn

from scripts import train_baseline_hybrid as reference
from scripts import train_baseline_hybrid_ddp as engine
from scripts.train import build_model as build_srsc_model


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_STAGE = "a"
ALLOWED_STAGES = (CAPACITY_STAGE,)
CAPACITY_TRAINING_ORIGIN = (
    "fresh_aio3_stage_a_10_10_hybrid_ddp_exact_update_safe_micro_v1"
)
CAPACITY_PURPOSE = "aio3_10_10_capacity_stage_a_exact_hybrid_ddp"
BASELINE_HYBRID_CONFIG = ROOT / "configs/protocol_aio3_baseline_hybrid.yaml"
CAPACITY_DEFINITION_CONFIG = ROOT / "configs/protocol_aio3_10_10.yaml"

_BASE_ALLOWED_STAGES = reference.ALLOWED_STAGES
_BASE_LOAD_CONFIG = reference.load_and_validate_config
_BASE_RUN_CONTRACT = engine.run_contract_payload
_BASE_DDP = engine.DDP


class CapacityStageACoarseForward(nn.Module):
    """Expose the coarse Stage-A graph while registering the complete model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # Registering the full object makes DDP's initial state broadcast cover
        # frozen assessor/D2 parameters too.  The checkpoint still serializes
        # ``model`` directly, never this adapter's ``model.``-prefixed view.
        self.model = model

    def forward(self, image):
        _, coarse = self.model._encode_coarse(image)
        return coarse


def _training_protocol(payload: dict) -> dict:
    # ``coordinate_stats`` is a downstream Stage-B binding.  It is present so
    # this one capacity config can own Stage-A, cache and statistics, but it is
    # not consumed by the Stage-A optimizer/data path.
    return {
        key: value for key, value in payload.items()
        if key not in {"model", "coordinate_stats"}
    }


def per_scale_total_blocks(payload: dict) -> tuple[int, int, int]:
    """Return level3/level2/level1 totals after assigning blocks to D1/D2."""
    d1 = tuple(map(int, payload["d1_blocks"]))
    d2 = tuple(map(int, payload["d2_blocks"]))
    return (
        d1[0] + d2[0],
        d1[1] + d2[1],
        d1[2] + d2[2] + int(payload["d2_refinement"]),
    )


def load_and_validate_config(path: str | Path) -> dict:
    """Require the main hybrid protocol with only 10/10 ownership changed."""

    cfg = _BASE_LOAD_CONFIG(path)
    baseline = _BASE_LOAD_CONFIG(BASELINE_HYBRID_CONFIG)
    if _training_protocol(cfg) != _training_protocol(baseline):
        raise ValueError(
            "10/10 Stage-A hybrid protocol differs from the primary AIO-3 "
            "hybrid protocol outside the model ownership definition"
        )

    definition = yaml.safe_load(CAPACITY_DEFINITION_CONFIG.read_text())
    if cfg.get("model") != definition.get("model"):
        raise ValueError(
            "10/10 Stage-A hybrid model differs from the preregistered "
            "capacity definition"
        )
    if cfg.get("coordinate_stats") != definition.get("coordinate_stats"):
        raise ValueError("capacity coordinate-statistics binding has drifted")
    model = cfg["model"]
    baseline_model = baseline["model"]
    ownership_fields = {"d1_blocks", "d2_blocks", "d2_refinement"}
    if {
        key: value for key, value in model.items() if key not in ownership_fields
    } != {
        key: value
        for key, value in baseline_model.items()
        if key not in ownership_fields
    }:
        raise ValueError(
            "capacity model differs from the primary model outside block ownership"
        )

    if per_scale_total_blocks(model) != per_scale_total_blocks(baseline_model):
        raise ValueError(
            "capacity model changes the per-scale total Transformer-block budget"
        )
    if sum(map(int, model["d1_blocks"])) != 10:
        raise ValueError("capacity D1 must own exactly ten Transformer blocks")
    if sum(map(int, model["d2_blocks"])) + int(model["d2_refinement"]) != 10:
        raise ValueError("capacity D2 must own exactly ten Transformer blocks")
    return cfg


def stable_engine_args(
    config: str | Path, run_name: str, workers_per_rank: int
) -> argparse.Namespace:
    """Return the exact namespace serialized by the shared DDP engine."""

    return argparse.Namespace(
        config=str(Path(config).resolve()),
        stage=CAPACITY_STAGE,
        run_name=run_name,
        workers_per_rank=int(workers_per_rank),
    )


def run_contract_payload(
    cfg: dict, args: argparse.Namespace, workers_per_rank: int
) -> dict:
    if args.stage != CAPACITY_STAGE:
        raise ValueError("capacity Stage-A contract only accepts stage='a'")
    contract = _BASE_RUN_CONTRACT(cfg, args, workers_per_rank)
    contract.update({
        "schema": 3,
        "purpose": CAPACITY_PURPOSE,
        "training_origin": CAPACITY_TRAINING_ORIGIN,
        "capacity_definition": {
            "d1_blocks": list(cfg["model"]["d1_blocks"]),
            "d2_blocks": list(cfg["model"]["d2_blocks"]),
            "d2_refinement": int(cfg["model"]["d2_refinement"]),
            "d1_total": 10,
            "d2_total": 10,
            "definition_config": str(CAPACITY_DEFINITION_CONFIG.resolve()),
            "definition_config_sha256": engine.sha256_file(
                CAPACITY_DEFINITION_CONFIG
            ),
            "primary_hybrid_config": str(BASELINE_HYBRID_CONFIG.resolve()),
            "primary_hybrid_config_sha256": engine.sha256_file(
                BASELINE_HYBRID_CONFIG
            ),
            "only_registered_change": "D1/D2 block ownership 6/14 -> 10/10",
        },
        "checkpoint_model_state": (
            "complete prefix-free SRSCLite.state_dict; Stage-B imports exactly "
            "encoder.* and d1.*"
        ),
        "equivalence_claim": (
            "relative to primary AIO-3 Stage-A, every optimizer update has the "
            "same canonical raw-index set; total 330500 updates, 33034920 raw "
            "samples, continuous Adam, clipping, objective, epoch LR, split and "
            "48 locked-validation boundaries are matched; only D1/D2 block "
            "ownership changes"
        ),
    })
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/net/feedback_controls.py",
    ):
        contract["code_sha256"][str(path.relative_to(ROOT))] = (
            engine.sha256_file(path)
        )
    return contract


def ensure_run_contract_rank0(
    run_dir: Path,
    cfg: dict,
    args: argparse.Namespace,
    workers_per_rank: int,
) -> str:
    contract = run_contract_payload(cfg, args, workers_per_rank)
    path = run_dir / "run_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("immutable 10/10 Stage-A hybrid run contract mismatch")
    else:
        if any(run_dir.glob("*.pt")):
            raise RuntimeError(
                "10/10 Stage-A checkpoint exists without its hybrid run contract"
            )
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    return engine.sha256_file(path)


def build_model(cfg: dict, stage: str = CAPACITY_STAGE):
    if stage != CAPACITY_STAGE:
        raise ValueError("capacity Stage-A model only accepts stage='a'")
    return build_srsc_model(cfg, "a")


@contextmanager
def _capacity_origin() -> Iterator[None]:
    prior = engine.TRAINING_ORIGIN
    engine.TRAINING_ORIGIN = CAPACITY_TRAINING_ORIGIN
    try:
        yield
    finally:
        engine.TRAINING_ORIGIN = prior


def validate_resume_payload(*args, **kwargs) -> None:
    """Apply the shared transaction validator under the capacity origin."""

    with _capacity_origin():
        engine.validate_resume_payload(*args, **kwargs)


def _capacity_ddp(raw_model, *args, **kwargs):
    return _BASE_DDP(CapacityStageACoarseForward(raw_model), *args, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--fresh", action="store_true")
    start.add_argument("--resume")
    parser.add_argument("--workers-per-rank", type=int, default=8)
    return parser.parse_args(argv)


def _engine_argv(args: argparse.Namespace) -> list[str]:
    values = [
        "--config", str(Path(args.config).resolve()),
        "--stage", CAPACITY_STAGE,
        "--run-name", args.run_name,
        "--workers-per-rank", str(int(args.workers_per_rank)),
    ]
    if args.fresh:
        values.append("--fresh")
    else:
        values.extend(["--resume", str(Path(args.resume).resolve())])
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Fail on protocol/model drift before installing process-local hooks or
    # reaching any CUDA initialization in the shared engine.
    load_and_validate_config(args.config)
    prior_allowed = reference.ALLOWED_STAGES
    prior_loader = reference.load_and_validate_config
    prior_contract = engine.run_contract_payload
    prior_ddp = engine.DDP
    prior_origin = engine.TRAINING_ORIGIN
    try:
        reference.ALLOWED_STAGES = tuple(dict.fromkeys((*prior_allowed, CAPACITY_STAGE)))
        reference.load_and_validate_config = load_and_validate_config
        engine.run_contract_payload = run_contract_payload
        engine.DDP = _capacity_ddp
        engine.TRAINING_ORIGIN = CAPACITY_TRAINING_ORIGIN
        return engine.main(_engine_argv(args))
    finally:
        reference.ALLOWED_STAGES = prior_allowed
        reference.load_and_validate_config = prior_loader
        engine.run_contract_payload = prior_contract
        engine.DDP = prior_ddp
        engine.TRAINING_ORIGIN = prior_origin


if __name__ == "__main__":
    raise SystemExit(main())
