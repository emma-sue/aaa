import torch

from scripts.eval_local_composite import elliptical_mask, make_local_composite


def test_local_composite_mask_is_feathered_and_spatially_local():
    mask = elliptical_mask(65, 81)
    assert mask.shape == (1, 65, 81)
    assert float(mask.min()) == 0.0
    assert float(mask.max()) == 1.0
    assert ((mask > 0.0) & (mask < 1.0)).any()
    assert mask[0, 32, 28] == 1.0
    assert mask[0, 0, -1] == 0.0


def test_local_composite_is_deterministic_and_does_not_consume_global_rng():
    degraded = torch.full((3, 32, 40), 0.5)
    before = torch.get_rng_state().clone()
    first = make_local_composite(degraded, "rain-001.png")
    after = torch.get_rng_state().clone()
    second = make_local_composite(degraded, "rain-001.png")
    different = make_local_composite(degraded, "rain-002.png")
    assert torch.equal(before, after)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert first.shape == degraded.shape
    assert torch.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
