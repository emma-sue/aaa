import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from scripts.train import (
    build_model,
    direction_valid_masks,
    direction_weights_from_coordinates,
    feedback_from_coordinates,
    load_formal_init,
    optimizer_groups,
    r2r_pretrain_epoch_ratio,
    restoration_l1,
    validate_coordinate_statistics,
)
from scripts.compute_coordinate_stats import fit_pca_projection
from src.net import SRSCCoordinateBuilder
from src.losses import state_loss
from src.net import SRSCLite


def test_pretrain_lr_curve_exactly_matches_public_r2r_closed_form():
    scheduler_path = Path("/root/R2R/utils/schedulers.py")
    spec = importlib.util.spec_from_file_location("public_r2r_schedulers", scheduler_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parameter = torch.nn.Parameter(torch.ones(()))
    lr, warmup, maximum, warmup_start = 2e-4, 15, 270, 1e-7
    optimizer = torch.optim.Adam([parameter], lr=lr)
    scheduler = module.LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup,
        max_epochs=maximum,
        warmup_start_lr=warmup_start,
        eta_min=0.0,
    )
    for epoch in range(240):
        scheduler.step(epoch)
        expected = scheduler.get_last_lr()[0]
        actual = lr * r2r_pretrain_epoch_ratio(
            epoch, lr, warmup, maximum, warmup_start, 0.0
        )
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_restoration_loss_is_exact_promptir_l1():
    prediction = torch.tensor([0.0, 0.25, 1.0])
    target = torch.tensor([1.0, 0.0, 0.5])
    assert torch.equal(restoration_l1(prediction, target), (prediction - target).abs().mean())


def test_aio3_matched_baseline_uses_same_batch_and_step_budget_as_ddp_stage_a():
    config = yaml.safe_load(
        Path("configs/protocol_aio3_baseline_b120.yaml").read_text()
    )
    assert config["micro_batch"] * config["accumulation"] == 120
    samples = 137669
    micro_batches = samples // config["micro_batch"]
    usable_micro_batches = (
        micro_batches // config["accumulation"] * config["accumulation"]
    )
    assert usable_micro_batches // config["accumulation"] == 1147
    assert usable_micro_batches * config["micro_batch"] == 137640


def test_stage_c_encoder_and_d1_have_tenth_learning_rate():
    model = SRSCLite(dim=8, encoder_blocks=(1, 1, 1, 1), d1_blocks=(1, 1, 1), d2_blocks=(1, 1, 1), d2_refinement=1)
    groups = optimizer_groups(model, "c", 2e-4)
    assert [group["lr"] for group in groups] == [2e-5, 2e-4]
    slow_ids = {id(p) for p in groups[0]["params"]}
    expected = {id(p) for module in (model.encoder, model.d1) for p in module.parameters()}
    assert slow_ids == expected
    assert not slow_ids.intersection(id(p) for p in groups[1]["params"])


def test_stage_b_loads_only_trained_coarse_path_and_preserves_seeded_d2():
    kwargs = dict(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
    )
    torch.manual_seed(11)
    stage_a = SRSCLite(**kwargs)
    torch.manual_seed(29)
    stage_b = SRSCLite(**kwargs)
    fresh_d2 = {key: value.clone() for key, value in stage_b.d2.state_dict().items()}

    scope = load_formal_init(stage_b, stage_a.state_dict(), "b_predicted")
    assert scope == "coarse_only_fresh_seeded_feedback_path"
    for key, value in stage_a.encoder.state_dict().items():
        assert torch.equal(stage_b.encoder.state_dict()[key], value)
    for key, value in stage_a.d1.state_dict().items():
        assert torch.equal(stage_b.d1.state_dict()[key], value)
    for key, value in fresh_d2.items():
        assert torch.equal(stage_b.d2.state_dict()[key], value)
    assert any(
        not torch.equal(stage_b.d2.state_dict()[key], stage_a.d2.state_dict()[key])
        for key in fresh_d2
    )

    torch.manual_seed(37)
    joint = SRSCLite(**kwargs)
    assert load_formal_init(joint, stage_a.state_dict(), "c") == "full"
    for key, value in stage_a.state_dict().items():
        assert torch.equal(joint.state_dict()[key], value)


def test_coordinate_statistics_are_bound_to_split_and_stage_a_checkpoint(tmp_path):
    split = tmp_path / "split.json"
    split.write_text("locked split")
    checkpoint = tmp_path / "stage_a.pt"
    checkpoint.write_bytes(b"selected locked-val checkpoint")
    cfg = {"protocol": "aio3", "split_manifest": str(split)}
    payload = {
        "protocol": "aio3",
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "stage_a_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    validate_coordinate_statistics(cfg, payload, checkpoint)

    wrong_split = dict(payload, split_manifest_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="locked-split mismatch"):
        validate_coordinate_statistics(cfg, wrong_split, checkpoint)
    wrong_checkpoint = dict(payload, stage_a_checkpoint_sha256="f" * 64)
    with pytest.raises(RuntimeError, match="Stage-A checkpoint mismatch"):
        validate_coordinate_statistics(cfg, wrong_checkpoint, checkpoint)


def test_non_srsc_code_has_no_direction_cosine_penalty():
    prediction = [torch.randn(2, 8, 4, 4)]
    target = [torch.randn(2, 8, 4, 4)]
    loss, parts = state_loss(prediction, target, direction_cosine_weight=0.0)
    assert torch.allclose(loss, parts["state_base"])


def test_direction_cosine_uses_raw_norm_mask_and_zero_then_full_map_mean():
    # Two pixels: the first has opposite predicted/target directions (cos= -1,
    # loss=2); the second is invalid.  Zeroing the invalid position and then
    # averaging the full map must give 1, not the old conditional mean of 2.
    prediction = torch.zeros(1, 8, 1, 2)
    target = torch.zeros_like(prediction)
    prediction[:, 2, 0, 0] = -1.0
    target[:, 2, 0, 0] = 1.0
    raw = torch.zeros_like(target)
    raw[:, 2, 0, 0] = 5e-4  # valid under the registered ||d_tilde||>=1e-6 rule
    weights = [torch.ones(1, 1, 1, 2)]
    masks = direction_valid_masks([raw], weights)
    assert masks[0][0, 0, 0, 0]
    assert not masks[0][0, 0, 0, 1]
    _, parts = state_loss(
        [prediction], [target], direction_cosine_weight=1.0,
        direction_weights=weights, direction_valid_masks=masks,
    )
    assert parts["state_cos"] == pytest.approx(torch.tensor(1.0))


def test_direction_raw_norm_below_one_e_minus_six_is_invalid():
    raw = torch.zeros(1, 8, 1, 2)
    raw[:, 2, 0, 0] = 0.5e-6
    raw[:, 2, 0, 1] = 1.5e-6
    masks = direction_valid_masks([raw], [torch.ones(1, 1, 1, 2)])[0]
    assert not masks[0, 0, 0, 0]
    assert masks[0, 0, 0, 1]


def test_all_registered_feedback_codes_have_eight_channels():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    y1 = (x + gt) / 2
    builder_with_pca = SRSCCoordinateBuilder(
        tau_v=1e-4, tau_e=1e-4, pca_projection=builder.P.clone()
    )
    outputs = builder_with_pca(x, y1, gt, [(8, 8)])
    model = SRSCLite(dim=8, encoder_blocks=(1, 1, 1, 1), d1_blocks=(1, 1, 1), d2_blocks=(1, 1, 1), d2_refinement=1)
    for mode in ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13", "O14", "O15"):
        state = feedback_from_coordinates(outputs, mode, model.oracle_ceiling_adapter)[0]
        assert state.shape == (1, 8, 8, 8)


def test_o14_ceiling_adapter_is_trainable_from_detached_oracle_coordinates():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    outputs = builder(x, (x + gt) / 2, gt, [(8, 8)], requested={"O14"})
    model = SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
    )
    state = feedback_from_coordinates(outputs, "O14", model.oracle_ceiling_adapter)[0]
    state.square().mean().backward()
    gradient = model.oracle_ceiling_adapter.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_pca_feedback_is_frozen_row_orthogonal_and_matches_equal_basis():
    random_builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    pca_builder = SRSCCoordinateBuilder(
        tau_v=1e-4, tau_e=1e-4, pca_projection=random_builder.P.clone()
    )
    assert torch.allclose(
        pca_builder.P_pca @ pca_builder.P_pca.T,
        torch.eye(6),
        atol=1e-5,
    )
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    output = pca_builder(x, (x + gt) / 2, gt, [(8, 8)])[0]
    assert torch.allclose(feedback_from_coordinates([output], "O15")[0], output.state)


def test_train_only_pca_fit_is_centered_orthogonal_and_deterministic():
    generator = torch.Generator().manual_seed(20260719)
    samples = torch.randn(128, 81, generator=generator)
    samples = samples / samples.norm(dim=1, keepdim=True)
    vnorm = torch.ones(128)
    mraw = torch.ones(128)
    first_p, first_mean, first_n = fit_pca_projection(samples, vnorm, mraw, 1e-4)
    second_p, second_mean, second_n = fit_pca_projection(samples, vnorm, mraw, 1e-4)
    assert first_p.shape == (6, 81) and first_mean.shape == (81,)
    assert torch.allclose(first_p @ first_p.T, torch.eye(6), atol=1e-5)
    assert torch.equal(first_p, second_p)
    assert torch.equal(first_mean, second_mean)
    assert first_n == second_n == 128


def test_o3_o4_controls_use_registered_raw_definitions_without_srsc_gate():
    builder = SRSCCoordinateBuilder(tau_v=10.0, tau_e=10.0)
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    y1 = (x + gt) / 2
    output = builder(x, y1, gt, [(8, 8)])[0]
    o3 = feedback_from_coordinates([output], "O3")[0]
    o4 = feedback_from_coordinates([output], "O4")[0]
    assert torch.allclose(o3[:, :1], output.error_magnitude)
    assert torch.allclose(o4[:, :1], torch.relu(output.p_raw))
    assert torch.allclose(o4[:, 1:2], output.m_raw + torch.relu(-output.p_raw))


def test_o9_is_corrupted_for_batch_one_and_cross_sample_for_batches():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    output = builder(x, (x + gt) / 2, gt, [(8, 8)])[0]
    o9 = feedback_from_coordinates([output], "O9")[0]
    assert torch.equal(o9[:, :2], output.state[:, :2])
    assert not torch.equal(o9[:, 2:], output.state[:, 2:])

    doubled = type(output)(**{
        field: (torch.cat((value, value + 1), 0) if isinstance(value, torch.Tensor) else value)
        for field, value in vars(output).items()
    })
    shuffled = feedback_from_coordinates([doubled], "O9")[0]
    assert torch.equal(shuffled[:, 2:], torch.roll(doubled.state[:, 2:], 1, 0))
    weights = direction_weights_from_coordinates([doubled], "O9")[0]
    assert torch.equal(weights, torch.roll(doubled.w_dir, 1, 0))


def test_o11_is_repeatable_without_consuming_global_rng():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 8, 8)
    gt = torch.rand_like(x)
    output = builder(x, (x + gt) / 2, gt, [(8, 8)])[0]
    before = torch.get_rng_state().clone()
    first = feedback_from_coordinates([output], "O11")[0]
    after = torch.get_rng_state().clone()
    second = feedback_from_coordinates([output], "O11")[0]
    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_matched_plain_baseline_is_within_half_percent_parameters():
    cfg = {
        "model": {
            "dim": 48, "matched_dim": 52, "encoder_blocks": [4, 6, 6, 8],
            "d1_blocks": [2, 2, 2], "d2_blocks": [4, 4, 4], "d2_refinement": 2,
            "heads": [1, 2, 4, 8], "expansion": 2.66,
        }
    }
    matched = sum(p.numel() for p in build_model(cfg, "baseline_matched").parameters())
    srsc = sum(p.numel() for p in build_model(cfg, "c").parameters())
    assert abs(matched - srsc) / srsc < 0.005
