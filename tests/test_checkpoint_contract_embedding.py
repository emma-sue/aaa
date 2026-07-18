from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import train as trainer


def _args(config, run_name: str):
    return SimpleNamespace(
        config=str(config),
        stage="a",
        feedback="O7",
        run_name=run_name,
        resume=None,
        init=None,
        max_steps=0,
        workers_override=None,
        seed_override=None,
        allow_incomplete_data=False,
        source_init_path=None,
        source_init_sha256=None,
    )


def test_checkpoint_embeds_run_code_data_and_runtime_contracts(
    tmp_path, monkeypatch,
):
    split = tmp_path / "locked_split.json"
    split.write_text('{"protocol":"aio3"}\n')
    config = tmp_path / "protocol.yaml"
    cfg = {
        "protocol": "aio3",
        "split_manifest": str(split),
        "micro_batch": 1,
        "accumulation": 1,
        "effective_batch": 1,
    }
    config.write_text(yaml.safe_dump(cfg, sort_keys=True))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _args(config, "contract_embedding")
    contract = trainer.ensure_run_contract(run_dir, cfg, args)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    monkeypatch.setattr(trainer, "runtime_snapshot", lambda: {"gpu_hours": 0.0})
    checkpoint = run_dir / "last.pt"
    trainer.save_checkpoint(
        checkpoint, model, optimizer, scheduler,
        epoch=1, batch_in_epoch=2, step=3, cfg=cfg, args=args,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_contract_sha = hashlib.sha256(
        (run_dir / "run_contract.json").read_bytes()
    ).hexdigest()
    assert payload["code_contract"] == contract["code_sha256"]
    assert payload["data_contract"]["config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert payload["data_contract"]["split_manifest_sha256"] == hashlib.sha256(
        split.read_bytes()
    ).hexdigest()
    assert payload["runtime_contract"]["run_contract_sha256"] == run_contract_sha
    assert payload["args"]["run_contract_sha256"] == run_contract_sha


def test_checkpoint_rejects_run_contract_drift(tmp_path, monkeypatch):
    split = tmp_path / "locked_split.json"
    split.write_text("{}\n")
    config = tmp_path / "protocol.yaml"
    cfg = {
        "protocol": "aio3",
        "split_manifest": str(split),
        "micro_batch": 1,
        "accumulation": 1,
        "effective_batch": 1,
    }
    config.write_text(yaml.safe_dump(cfg, sort_keys=True))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _args(config, "contract_drift")
    trainer.ensure_run_contract(run_dir, cfg, args)
    contract_path = run_dir / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["feedback"] = "O6"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    monkeypatch.setattr(trainer, "runtime_snapshot", lambda: {})
    with pytest.raises(RuntimeError, match="SHA256 carrier mismatch"):
        trainer.save_checkpoint(
            run_dir / "last.pt", model, optimizer, scheduler,
            epoch=1, batch_in_epoch=0, step=1, cfg=cfg, args=args,
        )
