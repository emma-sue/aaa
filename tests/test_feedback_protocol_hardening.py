import pytest
import torch

from src.net.feedback_controls import (
    apply_predicted_feedback_interface,
    isotropic_direction_normalization,
)
from src.net.srsc_lite import SRSCLite
from scripts.compute_coordinate_stats import build_normalization_statistics
from scripts.train import normalize_feedback


def tiny_model(mode=None):
    return SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
        predicted_feedback_mode=mode,
    )


def native_feedback(batch=1, height=32, width=40):
    return [
        torch.zeros(batch, 8, height // 2**index, width // 2**index)
        for index in range(4)
    ]


def test_forward_with_feedback_rejects_non_native_scale_without_running_d2():
    model = tiny_model().eval()
    feedback = native_feedback()
    feedback[2] = torch.zeros(1, 8, 9, 10)  # native S3 is 8x10
    d2_calls = []
    handle = model.d2.register_forward_hook(lambda *_: d2_calls.append(True))
    try:
        with pytest.raises(ValueError, match=r"S3.*expected native size.*interpolation is forbidden"):
            model.forward_with_feedback(torch.rand(1, 3, 32, 40), feedback)
    finally:
        handle.remove()
    assert d2_calls == []


def test_forward_with_feedback_accepts_exact_native_four_scale_interface():
    model = tiny_model().eval()
    with torch.no_grad():
        output = model.forward_with_feedback(
            torch.rand(1, 3, 32, 40), native_feedback()
        )
    assert output.y2.shape == (1, 3, 32, 40)
    assert [tuple(state.shape) for state in output.states] == [
        (1, 8, 32, 40),
        (1, 8, 16, 20),
        (1, 8, 8, 10),
        (1, 8, 4, 5),
    ]


@pytest.mark.parametrize("mode", ["O7", "O15"])
def test_direction_normalization_uses_one_isotropic_scale(mode):
    center, scale = isotropic_direction_normalization(
        mode,
        [1.0, 2.0, 9.0, -3.0, 2.0, 7.0, 4.0, 5.0],
        [2.0, 3.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    )
    assert center[:2] == [1.0, 2.0]
    assert center[2:] == [0.0] * 6
    # Sorted direction scales are [1,2,4,8,16,32], median=(4+8)/2=6.
    assert scale[:2] == [2.0, 3.0]
    assert scale[2:] == [6.0] * 6

    direction_a = torch.tensor([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])
    direction_b = torch.tensor([-2.0, 1.0, 4.0, -3.0, 2.0, 5.0])
    cosine_before = torch.nn.functional.cosine_similarity(
        direction_a.unsqueeze(0), direction_b.unsqueeze(0)
    )
    isotropic = torch.tensor(scale[2:])
    cosine_after = torch.nn.functional.cosine_similarity(
        (direction_a / isotropic).unsqueeze(0),
        (direction_b / isotropic).unsqueeze(0),
    )
    assert torch.allclose(cosine_before, cosine_after, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "mode,statistics_key",
    [("O7", "O7"), ("O8", "O7"), ("O15", "O15")],
)
def test_isotropic_statistics_survive_the_actual_training_consumer(
    mode, statistics_key
):
    raw_scales = [2.0, 3.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    collected = {}
    for producer_mode in ("O7", "O15"):
        collected[producer_mode] = [
            [torch.full((32,), channel_scale)]
            for channel_scale in raw_scales
        ]
    statistics = {
        "normalization": build_normalization_statistics(collected)
    }
    assert len(set(statistics["normalization"][statistics_key]["scale"][2:])) == 1
    raw = torch.zeros(1, 8, 1, 2)
    raw[:, 2:, 0, 0] = torch.tensor([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])
    raw[:, 2:, 0, 1] = torch.tensor([-2.0, 1.0, 4.0, -3.0, 2.0, 5.0])
    consumed = normalize_feedback([raw], mode, statistics)[0]
    cosine_before = torch.nn.functional.cosine_similarity(
        raw[:, 2:, 0, 0], raw[:, 2:, 0, 1]
    )
    cosine_after = torch.nn.functional.cosine_similarity(
        consumed[:, 2:, 0, 0], consumed[:, 2:, 0, 1]
    )
    assert torch.allclose(cosine_before, cosine_after, atol=1e-7, rtol=1e-7)


def test_zero_feedback_keeps_full_four_scale_eight_channel_interface_and_graph():
    model = tiny_model("O0").train()
    calls = {"assessor": 0, "pyramid": 0, "mods": 0}
    mod_states = []

    def count(name):
        def hook(_module, _inputs, _output):
            calls[name] += 1
        return hook

    def capture_mod(_module, inputs):
        calls["mods"] += 1
        mod_states.append(inputs[1])

    handles = [
        model.assessor.register_forward_hook(count("assessor")),
        model.y1_pyramid.register_forward_hook(count("pyramid")),
    ]
    handles.extend(
        module.register_forward_pre_hook(capture_mod)
        for module in (model.d2.mod4, model.d2.mod3, model.d2.mod2, model.d2.mod1)
    )
    try:
        details = model.forward_details(torch.rand(1, 3, 32, 40))
        details.y2.mean().backward()
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"assessor": 1, "pyramid": 1, "mods": 4}
    assert len(details.states) == len(mod_states) == 4
    assert all(state.shape[1] == 8 for state in mod_states)
    assert all(torch.count_nonzero(state) == 0 for state in mod_states)
    # The zero intervention happens after the assessor. Its parameters and raw
    # eight-channel outputs remain instantiated/executed for capacity matching.
    assert any(torch.count_nonzero(state.detach()) > 0 for state in details.states)
    assert any(parameter.numel() for parameter in model.assessor.parameters())


def test_interface_rejects_non_eight_channel_state_before_zeroing():
    with pytest.raises(ValueError, match="Bx8xHxW"):
        apply_predicted_feedback_interface([torch.zeros(1, 7, 4, 4)], "O0")
