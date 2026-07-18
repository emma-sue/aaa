import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"transaction_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pilot_cache_fixture(tmp_path, monkeypatch):
    orchestrator = load_script("orchestrate.py")
    trainer = load_script("train.py")
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "note", lambda _message: None)
    monkeypatch.delenv("SRSC_TRAIN_WORKERS", raising=False)

    name = "aio3_oracle_o7_pilot_n1000_s1"
    run_dir = tmp_path / "artifacts/checkpoints" / name
    metric_dir = tmp_path / "artifacts/metrics"
    run_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    split = tmp_path / "split.json"
    stats = tmp_path / "stats.json"
    source = tmp_path / "stage_a.pt"
    split.write_text('{"split": 1}\n')
    stats.write_text('{"stats": 1}\n')
    source.write_bytes(b"stage-a-source")
    cfg = {
        "protocol": "aio3",
        "seed": 1,
        "epochs": 240,
        "workers": 4,
        "split_manifest": str(split),
        "coordinate_stats": str(stats),
    }
    config = tmp_path / "protocol_aio3.yaml"
    config.write_text(json.dumps(cfg, sort_keys=True) + "\n")
    args = SimpleNamespace(
        config=str(config.resolve()),
        stage="b_oracle",
        feedback="O7",
        run_name=name,
        resume=None,
        init=str(source.resolve()),
        max_steps=1000,
        seed_override=None,
        workers_override=None,
        allow_incomplete_data=False,
        source_init_path=str(source.resolve()),
        source_init_sha256=trainer.sha256_file(source),
    )
    trainer.ensure_run_contract(run_dir, cfg, args)
    checkpoint = {
        "model": {"weight": torch.ones(1)},
        "epoch": 0,
        "batch_in_epoch": 1000,
        "step": 1000,
        "validation_pending": None,
        "config": cfg,
        "config_sha256": trainer.sha256_file(config),
        "split_manifest_sha256": trainer.sha256_file(split),
        "args": vars(args).copy(),
    }
    metric = metric_dir / f"{name}_locked_val.jsonl"
    metric.write_text(
        json.dumps({"epoch": 0, "step": 1000, "macro_psnr": 30.0}) + "\n"
    )
    torch.save(checkpoint, run_dir / "last.pt")
    kwargs = {
        "config": config,
        "stage": "b_oracle",
        "feedback": "O7",
        "init": source,
    }
    return orchestrator, name, run_dir, checkpoint, config, split, stats, source, kwargs


def test_checkpoint_complete_rejects_pending_final_validation(tmp_path, monkeypatch):
    module = load_script("orchestrate.py")
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {"epoch": 30, "batch_in_epoch": 0, "validation_pending": "epoch"},
        checkpoint,
    )
    assert not module.checkpoint_complete(checkpoint, 30)
    payload = torch.load(checkpoint, weights_only=False)
    payload["validation_pending"] = None
    torch.save(payload, checkpoint)
    assert module.checkpoint_complete(checkpoint, 30)


def test_epoch_replay_digest_is_ordered_atomic_and_immutable(tmp_path, monkeypatch):
    trainer = load_script("train.py")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    split = tmp_path / "split.json"
    split.write_text('{"locked": true}\n')
    cfg = {
        "protocol": "aio3",
        "seed": 7,
        "split_manifest": str(split),
    }
    batch = {
        "sample_index": torch.tensor([3, 1]),
        "sample_key": ["a" * 64, "b" * 64],
    }
    digest = hashlib.sha256()
    assert trainer.update_replay_digest(digest, batch) == 2
    path = trainer.commit_epoch_replay_digest(
        "run", 1, 10, 2, digest.hexdigest(), cfg
    )
    payload = json.loads(path.read_text())
    assert payload["sample_count"] == 2
    assert payload["ordered_sample_identity_sha256"] == digest.hexdigest()
    assert trainer.commit_epoch_replay_digest(
        "run", 1, 10, 2, digest.hexdigest(), cfg
    ) == path
    with pytest.raises(RuntimeError, match="replay digest drift"):
        trainer.commit_epoch_replay_digest("run", 1, 10, 2, "0" * 64, cfg)

    reversed_digest = hashlib.sha256()
    trainer.update_replay_digest(
        reversed_digest,
        {"sample_index": torch.tensor([1, 3]), "sample_key": ["b" * 64, "a" * 64]},
    )
    assert reversed_digest.hexdigest() != digest.hexdigest()


def test_orchestrator_rejects_paired_arm_replay_drift(tmp_path, monkeypatch):
    orchestrator = load_script("orchestrate.py")
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "note", lambda _message: None)
    root = tmp_path / "artifacts/manifests/replay_digests"
    common = {
        "epoch": 1,
        "optimizer_step_end": 10,
        "sample_count": 64,
        "ordered_sample_identity_sha256": "a" * 64,
        "protocol": "aio3",
        "seed": 7,
        "split_manifest_sha256": "b" * 64,
    }
    for name in ("arm_a", "arm_b"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "epoch001.json").write_text(
            json.dumps({"run_name": name, **common}) + "\n"
        )
    orchestrator.assert_matching_replay_digests(["arm_a", "arm_b"])
    drifted = {**common, "ordered_sample_identity_sha256": "c" * 64}
    (root / "arm_b/epoch001.json").write_text(json.dumps(drifted) + "\n")
    with pytest.raises(RuntimeError, match="paired-arm replay drift"):
        orchestrator.assert_matching_replay_digests(["arm_a", "arm_b"])


def test_pilot_complete_requires_exact_budget_and_nonpending_state(tmp_path, monkeypatch):
    module, name, run_dir, payload, config, split, stats, source, kwargs = (
        _pilot_cache_fixture(tmp_path, monkeypatch)
    )
    payload["validation_pending"] = "max_steps"
    torch.save(payload, run_dir / "last.pt")
    assert not module.pilot_complete(name, 1000, **kwargs)
    payload["validation_pending"] = None
    torch.save(payload, run_dir / "last.pt")
    assert module.pilot_complete(name, 1000, **kwargs)
    assert not module.pilot_complete(name, 999, **kwargs)

    module.compact_completed_pilot(name, 1000, **kwargs)
    assert module.pilot_complete(name, 1000, **kwargs)
    assert not (run_dir / "last.pt").exists()
    marker = json.loads((run_dir / "pilot_complete.json").read_text())
    marker["selected_locked_val"]["macro_psnr"] = -1.0
    (run_dir / "pilot_complete.json").write_text(json.dumps(marker) + "\n")
    assert not module.pilot_complete(name, 1000, **kwargs)


def test_run_cache_contract_rejects_every_registered_drift(tmp_path, monkeypatch):
    module, name, run_dir, payload, config, split, stats, source, kwargs = (
        _pilot_cache_fixture(tmp_path, monkeypatch)
    )
    saved_args = payload["args"]

    def matches(**overrides):
        values = dict(
            run_name=name,
            config=config,
            stage="b_oracle",
            feedback="O7",
            init=source,
            max_steps=1000,
            seed_override=None,
            epochs=None,
            checkpoint_args=saved_args,
        )
        values.update(overrides)
        return module.run_contract_matches_current(run_dir, **values)

    assert matches()
    assert not matches(run_name=name + "_different")
    assert not matches(stage="b_predicted")
    assert not matches(feedback="O6")
    assert not matches(max_steps=999)
    assert not matches(seed_override=2)

    same_bytes_config = tmp_path / "same_bytes_different_path.yaml"
    same_bytes_config.write_bytes(config.read_bytes())
    assert not matches(config=same_bytes_config)

    alternative_source = tmp_path / "same_bytes_different_source.pt"
    alternative_source.write_bytes(source.read_bytes())
    assert not matches(init=alternative_source)

    monkeypatch.setenv("SRSC_TRAIN_WORKERS", "8")
    assert not matches()
    monkeypatch.delenv("SRSC_TRAIN_WORKERS")

    original = split.read_bytes()
    split.write_bytes(original + b"drift")
    assert not matches()
    split.write_bytes(original)

    original = stats.read_bytes()
    stats.write_bytes(original + b"drift")
    assert not matches()
    stats.write_bytes(original)

    original = source.read_bytes()
    source.write_bytes(original + b"drift")
    assert not matches()
    source.write_bytes(original)

    original = config.read_bytes()
    config.write_bytes(original + b"\n")
    assert not matches()
    config.write_bytes(original)

    real_hashes = module.current_train_code_hashes
    monkeypatch.setattr(module, "current_train_code_hashes", lambda: {"drift": "1"})
    assert not matches()
    monkeypatch.setattr(module, "current_train_code_hashes", real_hashes)


def test_validation_upsert_and_top3_are_idempotent(tmp_path, monkeypatch):
    module = load_script("train.py")
    metric = tmp_path / "metric.jsonl"
    first = {"epoch": 2, "step": 20, "macro_psnr": 30.0}
    second = {"epoch": 2, "step": 20, "macro_psnr": 31.0}
    module.upsert_validation_record(metric, first)
    module.upsert_validation_record(metric, second)
    rows = [json.loads(line) for line in metric.read_text().splitlines()]
    assert rows == [second]

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_save(path, *_args, **_kwargs):
        Path(path).write_text("checkpoint")

    monkeypatch.setattr(module, "save_checkpoint", fake_save)
    placeholder = object()
    module.update_top3(
        run_dir, 30.0, 2, 20, placeholder, placeholder, placeholder, {}, placeholder
    )
    module.update_top3(
        run_dir, 31.0, 2, 20, placeholder, placeholder, placeholder, {}, placeholder
    )
    records = json.loads((run_dir / "top3.json").read_text())
    assert len(records) == 1
    assert records[0]["score"] == 31.0


def test_commit_pending_validation_replays_without_duplicates(tmp_path, monkeypatch):
    module = load_script("train.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    events = []

    monkeypatch.setattr(
        module,
        "validate_locked",
        lambda *_args, **_kwargs: (
            {
                "macro_psnr": 32.0,
                "denoise15": 32.0,
                "setting_ssim": {"denoise15": 0.9},
                "five_setting_mean_ssim": 0.9,
            },
            [
                {
                    "task": "denoise15", "name": "sample",
                    "psnr": 32.0, "ssim": 0.9,
                }
            ],
        ),
    )
    monkeypatch.setattr(
        module,
        "update_top3",
        lambda *_args, **_kwargs: events.append("top3"),
    )

    def fake_save(_path, *_args, **kwargs):
        events.append(("save", kwargs.get("validation_pending")))

    monkeypatch.setattr(module, "save_checkpoint", fake_save)
    kwargs = dict(
        model=object(), locked_val=object(), stage="b_oracle", builder=object(),
        feedback="O7", feedback_stats={}, run_dir=tmp_path / "run",
        optimizer=object(), scheduler=object(), cfg={"protocol": "aio3"},
        args=SimpleNamespace(run_name="transaction"), epoch=30,
        batch_in_epoch=0, step=300, kind="epoch",
    )
    (tmp_path / "run").mkdir()
    module.commit_pending_validation(**kwargs)
    module.commit_pending_validation(**kwargs)
    rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts/metrics/transaction_locked_val.jsonl")
        .read_text().splitlines()
    ]
    assert len(rows) == 1
    assert events.count("top3") == 2
    assert events[-1] == ("save", None)


def test_locked_summary_keeps_psnr_keys_and_adds_full_rgb_ssim(tmp_path):
    module = load_script("train.py")
    target = torch.zeros(1, 3, 8, 8)
    identical = module.full_rgb_ssim(target, target)
    changed = target.clone()
    changed[..., ::2, ::2] = 1.0
    assert identical == pytest.approx(1.0)
    assert module.full_rgb_ssim(changed, target) < identical

    tasks = ("dehaze", "derain", "denoise15", "denoise25", "denoise50")
    per_task_psnr = {task: [30.0, 32.0] for task in tasks}
    per_task_ssim = {task: [0.90, 0.92] for task in tasks}
    summary = module.summarize_locked_metrics(
        per_task_psnr, per_task_ssim, tasks
    )
    for task in tasks:
        assert summary[task] == pytest.approx(31.0)
        assert summary["setting_ssim"][task] == pytest.approx(0.91)
    assert summary["macro_psnr"] == pytest.approx(31.0)
    assert summary["five_setting_mean_ssim"] == pytest.approx(0.91)

    rows = [
        {"task": "dehaze", "name": "one", "psnr": 31.0, "ssim": 0.91}
    ]
    output = tmp_path / "locked_rows.csv"
    digest = module.atomic_write_locked_rows(output, rows)
    with output.open(newline="") as handle:
        restored = list(csv.DictReader(handle))
    assert list(restored[0]) == ["task", "name", "psnr", "ssim"]
    assert restored[0]["ssim"] == "0.91"
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()

    invalid = [dict(rows[0], ssim=float("nan"))]
    with pytest.raises(RuntimeError, match="non-finite"):
        module.atomic_write_locked_rows(tmp_path / "invalid.csv", invalid)


def test_resume_csv_discards_uncheckpointed_and_duplicate_rows(tmp_path):
    module = load_script("train.py")
    path = tmp_path / "train.csv"
    fields = ["time", "epoch", "step", "loss"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"time": 1, "epoch": 0, "step": 50, "loss": 1},
            {"time": 2, "epoch": 0, "step": 100, "loss": 2},
            {"time": 3, "epoch": 0, "step": 100, "loss": 3},
            {"time": 4, "epoch": 0, "step": 150, "loss": 4},
        ])
    module.reconcile_training_csv(path, 100)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [50, 100]
    assert rows[-1]["loss"] == "3"


def test_ddp_runtime_contract_and_fresh_guard(tmp_path):
    module = load_script("train_stage_a_ddp.py")
    args = SimpleNamespace(
        per_gpu_batch=30, accumulation=1, workers_per_rank=6
    )
    runtime = module.expected_distributed_runtime(args, world_size=4)
    assert runtime == {
        "world_size": 4,
        "per_gpu_batch": 30,
        "accumulation": 1,
        "global_effective_batch": 120,
        "workers_per_rank": 6,
    }
    payload = {"batch_in_epoch": 0, "distributed_runtime": dict(runtime)}
    module.validate_distributed_resume_runtime(payload, args, world_size=4)
    payload["distributed_runtime"]["per_gpu_batch"] = 29
    with pytest.raises(RuntimeError, match="identical distributed runtime"):
        module.validate_distributed_resume_runtime(payload, args, world_size=4)

    # A legacy epoch-boundary AIO-3 checkpoint is the only no-runtime case
    # accepted by the migration path.
    module.validate_distributed_resume_runtime(
        {"batch_in_epoch": 0}, args, world_size=4
    )
    with pytest.raises(RuntimeError, match="mid-epoch"):
        module.validate_distributed_resume_runtime(
            {"batch_in_epoch": 1}, args, world_size=4
        )

    run_dir = tmp_path / "run"
    log = tmp_path / "run.csv"
    metric = tmp_path / "metric.jsonl"
    module.assert_fresh_run_is_empty(run_dir, log, metric)
    run_dir.mkdir()
    (run_dir / "partial.pt").write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="refuses to overwrite"):
        module.assert_fresh_run_is_empty(run_dir, log, metric)
    module.enforce_training_origin("fresh", "fresh")
    module.enforce_training_origin("legacy", None)
    with pytest.raises(RuntimeError, match="training-origin mismatch"):
        module.enforce_training_origin("legacy", "fresh")
    assert module.legacy_final_checkpoint_needs_validation_replay(
        {"epoch": 240, "batch_in_epoch": 0}, epochs=240
    )
    assert not module.legacy_final_checkpoint_needs_validation_replay(
        {"epoch": 240, "batch_in_epoch": 0,
         "validation_transaction_schema": 1},
        epochs=240,
    )
    assert not module.legacy_final_checkpoint_needs_validation_replay(
        {"epoch": 239, "batch_in_epoch": 0}, epochs=240
    )


def test_ddp_top3_is_idempotent_and_atomic(tmp_path, monkeypatch):
    module = load_script("train_stage_a_ddp.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_checkpoint(path, *_args, **_kwargs):
        Path(path).write_text("checkpoint")

    monkeypatch.setattr(module, "atomic_ddp_checkpoint", fake_checkpoint)
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    placeholder = object()
    common = dict(
        run_dir=run_dir,
        epoch=5,
        step=50,
        raw_model=placeholder,
        optimizer=placeholder,
        scheduler=placeholder,
        cfg={},
        args=placeholder,
        rank=0,
        world_size=4,
    )
    module.update_top3_ddp(score=30.0, **common)
    module.update_top3_ddp(score=31.0, **common)
    records = json.loads((run_dir / "top3.json").read_text())
    assert len(records) == 1
    assert records[0]["score"] == 31.0
    assert not list(run_dir.glob("top3.json.tmp.*"))


def test_generic_stage_a_validation_artifact_integrity(tmp_path, monkeypatch):
    module = load_script("verify_stage_a_checkpoint.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run_name = "aio5_stage_a_coarse_seed1415926"
    run_dir = tmp_path / "artifacts/checkpoints" / run_name
    metric_dir = tmp_path / "artifacts/metrics"
    run_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    checkpoint = run_dir / "val_epoch240_step0427440.pt"
    torch.save({
        "epoch": 240,
        "step": 427440,
        "batch_in_epoch": 0,
        "validation_pending": None,
        "args": {"run_name": run_name, "stage": "a"},
    }, checkpoint)
    (run_dir / "top3.json").write_text(json.dumps([
        {"score": 31.0, "epoch": 240, "step": 427440,
         "checkpoint": checkpoint.name}
    ]) + "\n")
    (metric_dir / f"{run_name}_locked_val.jsonl").write_text(
        json.dumps({"macro_psnr": 31.0, "epoch": 240, "step": 427440})
        + "\n"
    )
    checks, details = module.validation_artifact_integrity(
        run_name, expected_epoch=240, expected_step=427440
    )
    assert all(checks.values()), (checks, details)
    checkpoint.unlink()
    checks, _ = module.validation_artifact_integrity(
        run_name, expected_epoch=240, expected_step=427440
    )
    assert not checks["top3_checkpoints_exist"]


def test_stage_a_validation_rejects_wrong_top3_selection(tmp_path, monkeypatch):
    module = load_script("verify_stage_a_checkpoint.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run_name = "aio5_stage_a_coarse_seed1415926"
    run_dir = tmp_path / "artifacts/checkpoints" / run_name
    metric_dir = tmp_path / "artifacts/metrics"
    run_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    rows = [
        {"macro_psnr": 35.0, "epoch": 235, "step": 420_000},
        {"macro_psnr": 30.0, "epoch": 240, "step": 427_440},
    ]
    for row in rows:
        checkpoint = run_dir / (
            f"val_epoch{row['epoch']:03d}_step{row['step']:07d}.pt"
        )
        torch.save({
            "epoch": row["epoch"],
            "step": row["step"],
            "batch_in_epoch": 0,
            "validation_pending": None,
            "args": {"run_name": run_name, "stage": "a"},
        }, checkpoint)
        row["checkpoint"] = checkpoint.name
    (metric_dir / f"{run_name}_locked_val.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in row.items() if k != "checkpoint"}) + "\n"
                for row in rows)
    )
    # Deliberately put the worse checkpoint first while keeping every file and
    # payload internally self-consistent.
    (run_dir / "top3.json").write_text(json.dumps([
        {"score": rows[1]["macro_psnr"], "epoch": rows[1]["epoch"],
         "step": rows[1]["step"], "checkpoint": rows[1]["checkpoint"]},
        {"score": rows[0]["macro_psnr"], "epoch": rows[0]["epoch"],
         "step": rows[0]["step"], "checkpoint": rows[0]["checkpoint"]},
    ]) + "\n")
    checks, details = module.validation_artifact_integrity(
        run_name, expected_epoch=240, expected_step=427_440
    )
    assert checks["top3_checkpoints_exist"]
    assert checks["top3_payload_provenance_valid"]
    assert not checks["top3_scores_valid"]
    assert not checks["top3_selection_valid"]
    assert details["top3_score_errors"] == []


def test_run_contract_is_immutable_and_resume_feedback_mismatch_fails(tmp_path):
    module = load_script("train.py")
    split = tmp_path / "split.json"
    config = tmp_path / "config.yaml"
    split.write_text("{}\n")
    config.write_text("protocol: aio3\n")
    cfg = {"protocol": "aio3", "split_manifest": str(split), "seed": 1}
    args = SimpleNamespace(
        config=str(config), run_name="contract", stage="b_oracle", feedback="O7",
        max_steps=0, seed_override=None, workers_override=8,
        allow_incomplete_data=False, source_init_path=None,
        source_init_sha256=None, init=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    module.ensure_run_contract(run_dir, cfg, args)
    assert (run_dir / "run_contract.json").is_file()
    args.feedback = "O6"
    with pytest.raises(RuntimeError, match="immutable run contract mismatch"):
        module.ensure_run_contract(run_dir, cfg, args)

    saved_args = {
        "stage": "b_oracle", "feedback": "O7", "run_name": "contract",
        "max_steps": 0, "seed_override": None, "workers_override": 8,
        "allow_incomplete_data": False, "init": None,
    }
    payload = {
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "config": cfg,
        "args": saved_args,
    }
    with pytest.raises(RuntimeError, match="feedback"):
        module.validate_resume_contract(payload, cfg, args)


def test_parallel_gpu_ids_are_canonical_and_single_job_is_bound(tmp_path, monkeypatch):
    module = load_script("orchestrate.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("SRSC_PARALLEL_GPUS", "0,00")
    with pytest.raises(ValueError, match="duplicate"):
        module.configured_parallel_gpus()

    monkeypatch.setenv("SRSC_PARALLEL_GPUS", "3")
    module.run_independent_arms([
        ([sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"], "one.log")
    ])
    assert (tmp_path / "artifacts/logs/one.log").read_text().strip() == "3"


def test_parallel_child_failure_terminates_sibling_group(tmp_path, monkeypatch):
    module = load_script("orchestrate.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("SRSC_PARALLEL_GPUS", "0,1")
    sentinel = tmp_path / "orphan_wrote.txt"
    jobs = [
        ([sys.executable, "-c", "import time,sys; time.sleep(.2); sys.exit(7)"], "fail.log"),
        ([sys.executable, "-c", f"import time,pathlib; time.sleep(8); pathlib.Path({str(sentinel)!r}).write_text('bad')"], "sibling.log"),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        module.run_independent_arms(jobs)
    assert not sentinel.exists()
