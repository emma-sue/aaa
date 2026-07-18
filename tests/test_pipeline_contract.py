import json
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_orchestrator_module():
    path = ROOT / "scripts/orchestrate.py"
    spec = importlib.util.spec_from_file_location("srsc_orchestrate_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_train_module():
    path = ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("srsc_train_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_orchestrator_lock_survives_stale_pid_file(tmp_path):
    module = _load_orchestrator_module()
    lock = tmp_path / ".orchestrate_aio3.lock"
    lock.write_text("99999999")

    first_fd = module.acquire_process_lock(lock)
    assert lock.read_text() == str(module.os.getpid())
    with pytest.raises(RuntimeError, match="lock is held"):
        module.acquire_process_lock(lock)

    module.fcntl.flock(first_fd, module.fcntl.LOCK_UN)
    module.os.close(first_fd)
    second_fd = module.acquire_process_lock(lock)
    module.fcntl.flock(second_fd, module.fcntl.LOCK_UN)
    module.os.close(second_fd)


def test_formal_stage_b_configs_are_locked_and_capacity_identical():
    for protocol in ("aio3", "aio5"):
        pretrain = yaml.safe_load((ROOT / f"configs/protocol_{protocol}.yaml").read_text())
        formal = yaml.safe_load((ROOT / f"configs/stage_b_{protocol}.yaml").read_text())
        assert formal["protocol"] == protocol
        assert formal["epochs"] == 30
        assert formal["validate_every_epochs"] == 2
        assert formal["crop_size"] == 128
        assert formal["micro_batch"] * formal["accumulation"] == formal["effective_batch"]
        assert formal["model"] == pretrain["model"]
        assert formal["split_manifest"] == pretrain["split_manifest"]
        assert formal["coordinate_stats"] == pretrain["coordinate_stats"]


def test_aio3_capacity_10_10_is_preregistered_and_budget_matched():
    main = yaml.safe_load((ROOT / "configs/protocol_aio3.yaml").read_text())
    alternate = yaml.safe_load((ROOT / "configs/protocol_aio3_10_10.yaml").read_text())
    alternate_b = yaml.safe_load((ROOT / "configs/stage_b_aio3_10_10.yaml").read_text())

    assert alternate_b["model"] == alternate["model"]
    assert alternate_b["split_manifest"] == alternate["split_manifest"] == main["split_manifest"]
    assert alternate_b["coordinate_stats"] == alternate["coordinate_stats"]
    assert alternate["coordinate_stats"] != main["coordinate_stats"]
    assert alternate["model"]["d1_blocks"] == [3, 3, 4]
    assert alternate["model"]["d2_blocks"] == [3, 3, 3]
    assert alternate["model"]["d2_refinement"] == 1
    assert sum(main["model"]["d1_blocks"]) == 6
    assert sum(main["model"]["d2_blocks"]) + main["model"]["d2_refinement"] == 14
    assert sum(alternate["model"]["d1_blocks"]) == 10
    assert sum(alternate["model"]["d2_blocks"]) + alternate["model"]["d2_refinement"] == 10
    # Per-scale totals are preserved: deep, middle, and shallow/refinement.
    assert main["model"]["d1_blocks"][0] + main["model"]["d2_blocks"][0] == 6
    assert alternate["model"]["d1_blocks"][0] + alternate["model"]["d2_blocks"][0] == 6
    assert main["model"]["d1_blocks"][1] + main["model"]["d2_blocks"][1] == 6
    assert alternate["model"]["d1_blocks"][1] + alternate["model"]["d2_blocks"][1] == 6
    assert (
        main["model"]["d1_blocks"][2] + main["model"]["d2_blocks"][2]
        + main["model"]["d2_refinement"]
    ) == 8
    assert (
        alternate["model"]["d1_blocks"][2] + alternate["model"]["d2_blocks"][2]
        + alternate["model"]["d2_refinement"]
    ) == 8


def test_orchestrator_has_non_authoritative_pilot_and_formal_gates():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    assert "PIPELINE_ONLY_NO_SCIENTIFIC_GATE" in source
    assert '"ORACLE_FORMAL_TIER1"' in source
    assert '"ORACLE_FORMAL_COMPLETE"' in source
    assert '"PREDICTED_FORMAL"' in source
    assert '"STAGE_C_COMPLETE"' in source
    assert '"OFFICIAL_TEST_COMPLETE"' in source
    assert source.index("ORACLE_PILOT_COMPLETE") < source.index("ORACLE_FORMAL_TIER1")
    assert source.index("PREDICTED_FORMAL") < source.index("STAGE_C_COMPLETE")
    assert source.index("STAGE_C_COMPLETE") < source.index("OFFICIAL_TEST_COMPLETE")
    assert "oracle_sign_abs_control_delta" in source
    assert "oracle_random_noise_control_delta" in source
    assert "predicted_direction_control_delta" in source
    assert 'for feedback in ("O8", "O9", "O10", "O11")' in source
    assert '"pca_ablation_authority": "REPORT_ONLY_NOT_A_GO_CRITERION"' in source
    assert '"O15"' in source
    trainer = (ROOT / "scripts/train.py").read_text()
    assert "predicted_supervision_mode(args.feedback)" in trainer
    assert '{"O7", "O8", "O15"}' in trainer
    controls = (ROOT / "src/net/feedback_controls.py").read_text()
    assert 'return "O7" if interface_mode in {"O9", "O10", "O11"}' in controls
    assert "apply_predicted_feedback_interface" in controls
    assert "direction_weights_from_coordinates(coords, target_mode)" in trainer
    assert 'for feedback in ("O6", "O7", "O12")' in source
    assert '"capacity_robustness_rule": "P7>P6 and P7>P12; no hyperparameter search"' in source
    assert source.index("if not predicted_go:") < source.rindex("run_aio3_capacity_robustness(decision")
    assert '("O13", "full_e_projected_diagnostic")' in source
    assert '("O14", "direct_gt_correction_ceiling")' in source
    assert "DIAGNOSTIC_OR_CEILING_ONLY_NOT_DEPLOYABLE_NOT_A_GO_CRITERION" in source
    assert "oracle_direction_not_dehaze_only" in source
    assert "predicted_direction_not_dehaze_only" in source
    assert '"O0", "O1", "O2", "O6", "O7", "O8", "O9", "O10", "O11", "O12"' in source
    assert "for feedback in joint_feedbacks" in source
    assert '"joint_mechanism_go": joint_mechanism_go' in source
    assert 'and decision["joint_mechanism_go"]' in source
    stage_c_failure = source[source.index("if not joint_mechanism_go:"):]
    stage_c_failure = stage_c_failure[:stage_c_failure.index("review_contract(")]
    assert '"scientific_go": "GO"' in stage_c_failure
    assert '"publication_go": "NO_GO"' in stage_c_failure
    assert '"scientific_go": "NO_GO"' not in stage_c_failure
    assert "scientific-only support" in stage_c_failure
    assert '"local_composite_go": local_composite_go' in source
    assert 'for key in ("O0", "baseline_matched")' in source
    assert "excluded from standard average" in source


def test_task_metric_whitelist_ignores_transaction_provenance():
    module = _load_orchestrator_module()
    metric = {
        "dehaze": 30.0,
        "derain": 31.0,
        "denoise15": 32.0,
        "denoise25": 33.0,
        "denoise50": 34.0,
        "macro_psnr": 32.0,
        "epoch": 2,
        "step": 10,
        "paired_rows_path": "/tmp/rows.csv",
        "paired_rows_sha256": "a" * 64,
    }
    assert module.metric_task_keys("aio3", metric) == [
        "dehaze", "derain", "denoise15", "denoise25", "denoise50"
    ]


def test_task_metric_whitelist_rejects_missing_or_nonfinite_task():
    module = _load_orchestrator_module()
    metric = {key: 1.0 for key in module.EXPECTED_TASKS["aio5"]}
    metric.pop("lowlight")
    with pytest.raises(KeyError, match="lowlight"):
        module.metric_task_keys("aio5", metric)
    metric["lowlight"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.metric_task_keys("aio5", metric)


def test_oracle_feasible_guard_uses_each_paired_task_median():
    module = _load_orchestrator_module()
    comparison = {
        "tasks": {
            task: {"median": 0.01}
            for task in module.EXPECTED_TASKS["aio3"]
        }
    }
    assert module.paired_task_median_guard("aio3", comparison)
    comparison["tasks"]["denoise50"]["median"] = -1e-4
    assert not module.paired_task_median_guard("aio3", comparison)
    comparison["tasks"].pop("derain")
    with pytest.raises(KeyError, match="task mismatch"):
        module.paired_task_median_guard("aio3", comparison)


def test_oracle_tier1_does_not_make_o12_an_unregistered_kill_gate():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    start = source.index("provisional_oracle_go = (")
    end = source.index("oracle_sign_tier1_go", start)
    gate = source[start:end]
    assert "oracle_residual_delta" not in gate
    assert "O12" not in gate
    assert "feasible_median_guard" in gate
    assert "edit_delta > 0" in gate
    assert "magnitude_delta > 0" in gate
    assert "REQUIRED_FAIR_DIAGNOSTIC_NOT_AN_ORACLE_TIER1_GO_CRITERION" in source


def test_stage_a_selection_uses_locked_validation_checkpoint():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    assert 'stage_a = best_checkpoint(stage_a_name)' in source
    assert "for statistics and Stage-B" in source


def test_aio5_stage_a_ddp_preserves_registered_budget_and_is_wired_before_orchestrator():
    cfg = yaml.safe_load((ROOT / "configs/protocol_aio5.yaml").read_text())
    assert cfg["effective_batch"] == 120
    assert cfg["micro_batch"] * cfg["accumulation"] == 120
    # Full manifest has 213,779 samples. Both 15x8 single-GPU and 30x4x1 DDP
    # consume 213,720 samples and make exactly 1,781 optimizer steps/epoch.
    train_samples = 213779
    single_steps = (train_samples // cfg["micro_batch"]) // cfg["accumulation"]
    per_rank = train_samples // 4
    ddp_steps = per_rank // 30
    assert single_steps == ddp_steps == 1781
    assert ddp_steps * cfg["epochs"] == 427440

    launcher = (ROOT / "scripts/launch_aio5_stage_a_4x4090.sh").read_text()
    assert "PER_GPU_BATCH=30" in launcher
    assert "ACCUMULATION=1" in launcher
    assert "WORKERS_PER_RANK=6" in launcher
    assert "--enforce-config-effective-batch" in launcher
    assert "--require-training-origin fresh" in launcher
    assert "--expected-training-origin fresh" in launcher
    assert "FINAL_STEP=427440" in launcher

    chain = (ROOT / "scripts/launch_when_data_ready.sh").read_text()
    aio3 = chain.index("orchestrate.py --protocol aio3")
    aio5_stage_a = chain.index("launch_aio5_stage_a_4x4090.sh")
    aio5 = chain.index("orchestrate.py --protocol aio5")
    assert aio3 < aio5_stage_a < aio5


def test_aio3_handoff_requires_exact_four_gpu_final_transaction():
    chain = (ROOT / "scripts/launch_when_data_ready.sh").read_text()
    for token in (
        "--minimum-step 330500",
        "--expected-epoch 240",
        "--expected-step 330500",
        "--expected-world-size 4",
        "--expected-global-effective-batch 120",
        "--expected-per-gpu-batch 30",
        "--expected-accumulation 1",
        "--expected-workers-per-rank 8",
        "--expected-backend nccl",
        "--require-validation-complete",
    ):
        assert token in chain

    source = (ROOT / "scripts/orchestrate.py").read_text()
    main_start = source.index("def main():")
    start = source.index("stage_a_last =", main_start)
    end = source.index("stage_a = best_checkpoint", start)
    handoff = source[start:end]
    assert "ensure_protocol_stage_a_handoff(" in handoff
    assert "checkpoint_complete(" not in handoff
    assert "train_command(" not in handoff
    assert '"expected_step": 330_500' in source
    assert '"expected_world_size": 4' in source
    assert '"expected_global_effective_batch": 120' in source


def test_both_stage_b_contracts_freeze_before_first_full_orchestration():
    chain = (ROOT / "scripts/launch_when_data_ready.sh").read_text()
    aio3_gate = chain.index("orchestrate.py --protocol aio3 --pilot-steps 1000")
    aio5_stage_a = chain.index("launch_aio5_stage_a_4x4090.sh", aio3_gate)
    assert "--stop-after-stage-b" in chain[aio3_gate:aio5_stage_a]

    aio5_gate = chain.index(
        "orchestrate.py --protocol aio5 --pilot-steps 1000", aio5_stage_a
    )
    aio3_full = chain.index(
        "orchestrate.py --protocol aio3 --pilot-steps 1000", aio3_gate + 1
    )
    assert "--stop-after-stage-b" in chain[aio5_gate:aio3_full]
    assert aio3_gate < aio5_stage_a < aio5_gate < aio3_full

    source = (ROOT / "scripts/orchestrate.py").read_text()
    assert '"--stop-after-stage-b"' in source
    assert '"stage": "STAGE_B_COMPLETE"' in source
    capacity_call = source.index(
        "run_aio3_capacity_robustness(decision, decision_path)"
    )
    stop_gate = source.index("if args.stop_after_stage_b:", capacity_call)
    baseline_start = source.index("baseline_models = {}", stop_gate)
    assert capacity_call < stop_gate < baseline_start


def test_aio3_baseline_pretrain_uses_sequential_four_gpu_hybrid_only():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    start = source.index("baseline_models = {}")
    end = source.index("baseline_finetune_jobs = []", start)
    pretrain = source[start:end]
    aio3_start = pretrain.index('if args.protocol == "aio3":')
    aio5_start = pretrain.index("else:", aio3_start)
    aio3 = pretrain[aio3_start:aio5_start]
    aio5 = pretrain[aio5_start:]

    assert "protocol_aio3_baseline_hybrid.yaml" in pretrain
    assert "hybrid_ddp_train_command(" in aio3
    assert aio3.count("hybrid_ddp_complete(") == 2
    assert "assert_matching_hybrid_ddp_update_digests(hybrid_names)" in aio3
    assert "hybrid_names.append(pretrain_name)" in aio3
    assert "run_independent_arms(" not in aio3
    assert "formal_complete(" not in aio3
    assert "compact_completed_formal(" not in aio3

    # AIO-5 retains the established generic train.py path and compaction.
    assert "train_command(" in aio5
    assert "formal_complete(" in aio5
    assert "run_independent_arms(baseline_pretrain_jobs)" in aio5
    assert "compact_completed_formal(" in aio5


def test_hybrid_ddp_command_is_four_rank_and_resume_or_fresh(tmp_path, monkeypatch):
    module = _load_orchestrator_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    config = tmp_path / "hybrid.yaml"
    config.write_text("protocol: aio3\n")
    run_name = "aio3_baseline_hybrid_ddp_pretrain_s1415926"

    fresh = module.hybrid_ddp_train_command(
        config, "baseline", run_name, workers_per_rank=8
    )
    assert fresh[:5] == [
        module.sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
    ]
    assert "scripts/train_baseline_hybrid_ddp.py" in fresh
    assert fresh[-1] == "--fresh"
    assert "--resume" not in fresh
    assert fresh[fresh.index("--workers-per-rank") + 1] == "8"

    last = tmp_path / "artifacts/checkpoints" / run_name / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"resume-boundary")
    resume = module.hybrid_ddp_train_command(
        config, "baseline", run_name, workers_per_rank=8
    )
    assert "--fresh" not in resume
    assert resume[-2:] == ["--resume", str(last)]


def test_stage_c_uses_idempotent_formal_completion_and_compaction():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    loop = source.index("for feedback in joint_feedbacks:")
    end = source.index("joint_score =", loop)
    stage_c = source[loop:end]
    assert "if not formal_complete(" in stage_c
    assert 'name, stage_c_cfg["epochs"], config=stage_c_config' in stage_c
    assert 'stage="c", feedback=feedback, init=source' in stage_c
    assert "compact_completed_formal(" in stage_c
    assert 'checkpoint_complete(last, stage_c_cfg["epochs"])' not in stage_c
    assert stage_c.index("compact_completed_formal") < stage_c.index('"best_checkpoint"')


def test_formal_compaction_preserves_locked_val_best_model_and_provenance(tmp_path, monkeypatch):
    module = _load_orchestrator_module()
    trainer = _load_train_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    monkeypatch.delenv("SRSC_TRAIN_WORKERS", raising=False)

    run_name = "aio3_predicted_o7_formal_s1415926"
    run_dir = tmp_path / "artifacts" / "checkpoints" / run_name
    metric_dir = tmp_path / "artifacts" / "metrics"
    run_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    split = tmp_path / "split.json"
    stats = tmp_path / "stats.json"
    source = tmp_path / "stage_a.pt"
    split.write_text('{"split": 1}\n')
    stats.write_text('{"stats": 1}\n')
    source.write_bytes(b"frozen-stage-a")
    cfg = {
        "protocol": "aio3",
        "seed": 1415926,
        "epochs": 30,
        "workers": 4,
        "split_manifest": str(split),
        "coordinate_stats": str(stats),
    }
    config = tmp_path / "stage_b_aio3.yaml"
    config.write_text(json.dumps(cfg, sort_keys=True) + "\n")
    args = SimpleNamespace(
        config=str(config.resolve()),
        stage="b_predicted",
        feedback="O7",
        run_name=run_name,
        resume=None,
        init=str(source.resolve()),
        max_steps=0,
        seed_override=None,
        workers_override=None,
        allow_incomplete_data=False,
        source_init_path=str(source.resolve()),
        source_init_sha256=trainer.sha256_file(source),
    )
    trainer.ensure_run_contract(run_dir, cfg, args)

    low = torch.nn.Linear(2, 2, bias=False)
    high = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        low.weight.fill_(1.0)
        high.weight.fill_(7.0)
    common = {
        "batch_in_epoch": 0,
        "validation_pending": None,
        "config": cfg,
        "config_sha256": trainer.sha256_file(config),
        "split_manifest_sha256": trainer.sha256_file(split),
        "args": vars(args).copy(),
    }
    torch.save(
        dict(common, epoch=28, step=280, model=low.state_dict()),
        run_dir / "val_epoch028_step0000280.pt",
    )
    torch.save(
        dict(common, epoch=30, step=300, model=high.state_dict()),
        run_dir / "val_epoch030_step0000300.pt",
    )
    # Completion authority comes from the resumable last checkpoint, whereas
    # selection authority comes only from locked validation/top3.
    torch.save(
        dict(common, epoch=30, step=300, model=low.state_dict()), run_dir / "last.pt"
    )
    (run_dir / "top3.json").write_text(json.dumps([
        {"score": 31.0, "epoch": 30, "step": 300, "checkpoint": "val_epoch030_step0000300.pt"},
        {"score": 29.0, "epoch": 28, "step": 280, "checkpoint": "val_epoch028_step0000280.pt"},
    ]))
    (metric_dir / f"{run_name}_locked_val.jsonl").write_text(
        json.dumps({"macro_psnr": 29.0, "epoch": 28, "step": 280}) + "\n"
        + json.dumps({"macro_psnr": 31.0, "epoch": 30, "step": 300}) + "\n"
    )

    cache_kwargs = {
        "config": config,
        "stage": "b_predicted",
        "feedback": "O7",
        "init": source,
    }
    assert module.formal_complete(run_name, 30, **cache_kwargs)
    module.compact_completed_formal(run_name, epochs=30, **cache_kwargs)
    compact_path = run_dir / "formal_best_model.pt"
    marker_path = run_dir / "formal_complete.json"
    assert compact_path.is_file() and marker_path.is_file()
    assert module.formal_complete(run_name, 30, **cache_kwargs)
    assert module.best_checkpoint(run_name) == compact_path
    assert sorted(path.name for path in run_dir.glob("*.pt")) == ["formal_best_model.pt"]

    payload = torch.load(compact_path, map_location="cpu", weights_only=False)
    restored = torch.nn.Linear(2, 2, bias=False)
    restored.load_state_dict(payload["model"], strict=True)
    assert torch.equal(restored.weight, high.weight)
    assert payload["selected_locked_val"]["macro_psnr"] == 31.0
    assert payload["config"] == cfg
    assert payload["config_sha256"] == trainer.sha256_file(config)
    assert payload["split_manifest_sha256"] == trainer.sha256_file(split)
    assert payload["args"] == vars(args)
    assert payload["selected_top3_record"]["checkpoint"] == "val_epoch030_step0000300.pt"
    assert payload["source_checkpoint"] == "val_epoch030_step0000300.pt"
    assert len(payload["source_checkpoint_sha256"]) == 64
    assert payload["checkpoint_kind"] == "formal_locked_val_best_model_only"

    # A restart must recognize the compact marker and must not rewrite the
    # selected model or require a deleted last.pt.
    original_bytes = compact_path.read_bytes()
    module.compact_completed_formal(run_name, epochs=30, **cache_kwargs)
    assert compact_path.read_bytes() == original_bytes

    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    marker["selected_top3_record"]["score"] = -1.0
    marker_path.write_text(json.dumps(marker) + "\n")
    assert not module.formal_complete(run_name, 30, **cache_kwargs)
    marker_path.write_bytes(marker_bytes)

    payload = torch.load(compact_path, map_location="cpu", weights_only=False)
    payload["args"]["feedback"] = "O6"
    torch.save(payload, compact_path)
    marker = json.loads(marker_path.read_text())
    marker["model_sha256"] = hashlib.sha256(compact_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker) + "\n")
    assert not module.formal_complete(run_name, 30, **cache_kwargs)
    compact_path.write_bytes(original_bytes)
    marker_path.write_bytes(marker_bytes)

    top3_path = run_dir / "top3.json"
    top3_bytes = top3_path.read_bytes()
    top3 = json.loads(top3_bytes)
    top3[0]["score"] = -1.0
    top3_path.write_text(json.dumps(top3) + "\n")
    assert not module.formal_complete(run_name, 30, **cache_kwargs)
    top3_path.write_bytes(top3_bytes)


def test_contract_review_hashes_scientific_definition_and_parity_reports():
    module = _load_orchestrator_module()
    contracted = {path.resolve() for path in module.CONTRACTS}
    for relative in (
        "reports/AUDIT.md", "reports/ARCHITECTURE.md", "reports/BASELINE_PARITY.md",
        "reports/AUTOSOTA_STRATEGY_LIBRARY.md",
        "reports/PRE_STAGE_B_RELOAD_REQUIRED.md",
        "reports/PROTOCOL_CORRECTION_CENTER_CROP.md",
        "reports/CACHE_CONTRACT_REVISION_V1.md",
    ):
        path = (ROOT / relative).resolve()
        assert path in contracted
        assert path.is_file() and path.stat().st_size > 0
    for relative in (
        "src/net/clean_restormer_aio.py", "src/net/restormer_blocks.py",
        "src/net/srsc_lite.py",
        "src/net/srsc_coordinates.py", "src/data/aio_dataset.py",
        "scripts/compute_coordinate_stats.py", "scripts/verify_promptir_baseline.py",
        "scripts/launch_when_data_ready.sh", "scripts/reload_pipeline_at_checkpoint.sh",
        "scripts/train_stage_a_ddp.py", "scripts/launch_aio3_stage_a_4x4090.sh",
        "scripts/launch_aio5_stage_a_4x4090.sh",
        "scripts/watchdog.sh", "configs/protocol_aio3_10_10.yaml",
        "configs/stage_b_aio3_10_10.yaml",
        "scripts/eval_local_composite.py",
        "scripts/cache_stage_a_outputs.py", "scripts/train_baseline_hybrid.py",
        "scripts/train_baseline_hybrid_ddp.py",
        "scripts/export_metrics_long.py",
        "configs/protocol_aio3_baseline_hybrid.yaml",
        "artifacts/manifests/aio3.json",
        "artifacts/manifests/locked_split_aio3.json",
    ):
        path = (ROOT / relative).resolve()
        assert path in contracted
        assert path.is_file() and path.stat().st_size > 0


def test_stage_a_cache_and_statistics_precede_every_stage_b_contract_review():
    source = (ROOT / "scripts/orchestrate.py").read_text()
    main_select = source.index(
        'note(f"SELECT Stage-A locked-val checkpoint `{stage_a}` for statistics and Stage-B")'
    )
    main_cache = source.index("ensure_stage_a_locked_val_cache(", main_select)
    main_stats = source.index("ensure_coordinate_stats(", main_cache)
    main_review = source.index(
        'review_contract(f"{args.protocol}_before_stage_b")', main_stats
    )
    assert main_select < main_cache < main_stats < main_review

    capacity_select = source.index(
        'note(f"SELECT 10/10 Stage-A locked-val checkpoint `{stage_a}`")'
    )
    capacity_cache = source.index("ensure_stage_a_locked_val_cache(", capacity_select)
    capacity_stats = source.index("ensure_coordinate_stats(", capacity_cache)
    capacity_review = source.index(
        'review_contract("aio3_before_capacity_10_10_stage_b")', capacity_stats
    )
    assert capacity_select < capacity_cache < capacity_stats < capacity_review


def test_coordinate_stats_completion_is_checkpoint_bound_and_fail_closed(tmp_path):
    module = _load_orchestrator_module()
    checkpoint = tmp_path / "stage_a.pt"
    checkpoint.write_bytes(b"selected-stage-a")
    split = tmp_path / "split.json"
    split.write_text('{"protocol":"aio3"}\n')
    stats = tmp_path / "stats.json"
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump({
        "protocol": "aio3",
        "seed": 1415926,
        "split_manifest": str(split),
        "coordinate_stats": str(stats),
    }))
    modes = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O12", "O13", "O15")
    payload = {
        "protocol": "aio3",
        "seed": 1415926,
        "stage_a_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "tau_v": 0.01,
        "tau_e": 0.02,
        "pca_direction_matrix": [[0.0] * 81 for _ in range(6)],
        "pca_direction_mean": [0.0] * 81,
        "normalization": {
            mode: {"center": [0.0] * 8, "scale": [1.0] * 8}
            for mode in modes
        },
    }
    stats.write_text(json.dumps(payload) + "\n")
    assert module.coordinate_stats_complete(config, checkpoint)
    payload["stage_a_checkpoint_sha256"] = "0" * 64
    stats.write_text(json.dumps(payload) + "\n")
    assert not module.coordinate_stats_complete(config, checkpoint)


def test_stage_a_cache_gate_records_bound_manifest_and_rejects_drift(
    tmp_path, monkeypatch,
):
    module = _load_orchestrator_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "note", lambda _message: None)
    checkpoint = tmp_path / "stage_a.pt"
    checkpoint.write_bytes(b"selected")
    config = tmp_path / "protocol_aio3.yaml"
    config.write_text(yaml.safe_dump({"protocol": "aio3"}))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cache_dir = (
        tmp_path / "artifacts/cache/stage_a_y1/aio3"
        / checkpoint_sha[:16] / "locked_val"
    )

    def fake_run(_command, _log):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "COMPLETE_TWO_PASS_VERIFIED",
            "scope": "locked_val",
            "official_test_forbidden": True,
            "item_count": 2,
            "aggregate_sha256": "a" * 64,
            "bindings": {"stage_a_checkpoint": {"sha256": checkpoint_sha}},
        }
        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        (cache_dir / "manifest.sha256").write_text(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            + "  manifest.json\n"
        )

    monkeypatch.setattr(module, "run", fake_run)
    manifest = module.ensure_stage_a_locked_val_cache(config, checkpoint, "cache.log")
    assert manifest == cache_dir / "manifest.json"
    evidence = tmp_path / "artifacts/manifests/aio3_stage_a_locked_val_cache.json"
    assert json.loads(evidence.read_text())["stage_a_checkpoint_sha256"] == checkpoint_sha

    def corrupt_run(_command, _log):
        fake_run(_command, _log)
        (cache_dir / "manifest.sha256").write_text(
            "0" * 64 + "  manifest.json\n"
        )

    monkeypatch.setattr(module, "run", corrupt_run)
    with pytest.raises(RuntimeError, match="failed closed"):
        module.ensure_stage_a_locked_val_cache(config, checkpoint, "cache.log")


def test_r2r_reference_has_exact_protocol_tasks():
    payload = json.loads((ROOT / "artifacts/reference/r2r_cvpr2026_tables.json").read_text())
    assert set(payload["table1_aio3"]) == {
        "dehaze", "derain", "denoise15", "denoise25", "denoise50", "reported_average"
    }
    assert set(payload["table2_aio5"]) == {
        "dehaze", "derain", "denoise25", "deblur", "lowlight", "reported_average"
    }
    assert payload["table1_aio3"]["reported_average"]["psnr"] == 32.53
    assert payload["table2_aio5"]["reported_average"]["psnr"] == 30.48
