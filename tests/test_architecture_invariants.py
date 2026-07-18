from pathlib import Path

import torch
import yaml

from src.net import SRSCCoordinateBuilder, SRSCLite


ROOT = Path(__file__).resolve().parents[1]


def tiny_model() -> SRSCLite:
    return SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
    )


def test_full_f_g_s_pyramids_and_encoder_executes_once():
    model = tiny_model().eval()
    encoder_calls = []
    y1_pyramids = []
    handles = [
        model.encoder.register_forward_hook(
            lambda _module, _inputs, _output: encoder_calls.append(True)
        ),
        model.y1_pyramid.register_forward_hook(
            lambda _module, _inputs, output: y1_pyramids.append(output)
        ),
    ]
    try:
        with torch.no_grad():
            details = model.forward_details(torch.rand(1, 3, 64, 72))
    finally:
        for handle in handles:
            handle.remove()

    assert len(encoder_calls) == 1
    assert len(y1_pyramids) == 1
    assert [tuple(feature.shape) for feature in details.features] == [
        (1, 8, 64, 72),
        (1, 16, 32, 36),
        (1, 32, 16, 18),
        (1, 64, 8, 9),
    ]
    assert [tuple(feature.shape) for feature in y1_pyramids[0]] == [
        (1, 24, 64, 72),
        (1, 48, 32, 36),
        (1, 96, 16, 18),
        (1, 192, 8, 9),
    ]
    assert [tuple(state.shape) for state in details.states] == [
        (1, 8, 64, 72),
        (1, 8, 32, 36),
        (1, 8, 16, 18),
        (1, 8, 8, 9),
    ]


def test_d1_and_d2_are_distinct_decoders_without_weight_sharing():
    model = tiny_model()
    d1_parameters = {id(parameter) for parameter in model.d1.parameters()}
    d2_parameters = {id(parameter) for parameter in model.d2.parameters()}
    assert d1_parameters
    assert d2_parameters
    assert d1_parameters.isdisjoint(d2_parameters)


def test_registered_main_decoder_budget_is_exactly_six_plus_fourteen():
    config = yaml.safe_load((ROOT / "configs/protocol_aio3.yaml").read_text())
    model = config["model"]
    d1_blocks = sum(model["d1_blocks"])
    d2_blocks = sum(model["d2_blocks"]) + model["d2_refinement"]
    assert d1_blocks == 6
    assert d2_blocks == 14
    assert d1_blocks + d2_blocks == 20


def test_orthogonal_chromatic_perturbation_increases_transverse_magnitude():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.full((1, 3, 24, 24), 0.2)
    gt = torch.full_like(x, 0.5)
    on_target_line = x + 0.5 * (gt - x)
    with_transverse = on_target_line.clone()
    # The target edit is equal in RGB.  Adding +delta to red and -delta to
    # green has zero dot product with that local RGB edit, while introducing a
    # genuine transverse component in the same 81-D descriptor space.
    with_transverse[:, 0] += 0.03
    with_transverse[:, 1] -= 0.03

    base = builder(x, on_target_line, gt, [(24, 24)])[0]
    perturbed = builder(x, with_transverse, gt, [(24, 24)])[0]
    assert perturbed.m_raw.mean() > base.m_raw.mean() + 0.05
    assert perturbed.orthogonal_relative_error.median() < 1e-4
