import pytest
import torch

from src.net import (
    DeterministicFeedbackEncoder,
    SRSCCoordinateBuilder,
    apply_predicted_feedback_interface,
    predicted_supervision_mode,
)
from src.net.srsc_lite import SRSCLite


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


def test_predicted_channel_masks_are_hard_at_common_interface():
    state = torch.arange(2 * 8 * 4 * 6, dtype=torch.float32).reshape(2, 8, 4, 6)
    state.requires_grad_()
    o6 = apply_predicted_feedback_interface([state], "O6")[0]
    assert torch.equal(o6[:, :2], state[:, :2])
    assert torch.count_nonzero(o6[:, 2:]) == 0
    o6.sum().backward()
    assert torch.count_nonzero(state.grad[:, :2]) > 0
    assert torch.count_nonzero(state.grad[:, 2:]) == 0

    o10 = apply_predicted_feedback_interface([state.detach()], "O10")[0]
    assert torch.equal(o10[:, :2], state.detach()[:, :2])
    assert torch.count_nonzero(o10[:, 2:]) == 0


def test_sign_shuffle_and_noise_controls_cannot_bypass_interface():
    state = torch.randn(2, 8, 5, 7)
    o8 = apply_predicted_feedback_interface([state], "O8")[0]
    assert torch.all(o8[:, :1] >= 0)
    assert torch.equal(o8[:, 1:], state[:, 1:])

    o9 = apply_predicted_feedback_interface([state], "O9")[0]
    assert torch.equal(o9[:, :2], state[:, :2])
    assert torch.equal(o9[:, 2:], torch.roll(state[:, 2:], 1, 0))

    one = state[:1]
    batch_one_o9 = apply_predicted_feedback_interface([one], "O9")[0]
    assert not torch.equal(batch_one_o9[:, 2:], one[:, 2:])

    before = torch.get_rng_state().clone()
    first = apply_predicted_feedback_interface([state], "O11")[0]
    after = torch.get_rng_state().clone()
    second = apply_predicted_feedback_interface([state + 123.0], "O11")[0]
    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_negative_controls_supervise_full_state_but_intervene_before_d2():
    assert predicted_supervision_mode("O8") == "O8"
    assert predicted_supervision_mode("O9") == "O7"
    assert predicted_supervision_mode("O10") == "O7"
    assert predicted_supervision_mode("O11") == "O7"
    with pytest.raises(ValueError, match="unsupported predicted"):
        predicted_supervision_mode("O14")


def test_model_preserves_raw_assessor_outputs_and_masks_only_d2_inputs():
    model = tiny_model("O6").eval()
    state_inputs = []

    def capture(_module, inputs):
        state_inputs.append(inputs[1].detach().clone())

    handles = [
        module.register_forward_pre_hook(capture)
        for module in (model.d2.mod4, model.d2.mod3, model.d2.mod2, model.d2.mod1)
    ]
    try:
        with torch.no_grad():
            details = model.forward_details(torch.rand(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert len(details.states) == len(state_inputs) == 4
    assert any(torch.count_nonzero(state[:, 2:]) > 0 for state in details.states)
    assert all(torch.count_nonzero(state[:, 2:]) == 0 for state in state_inputs)


@pytest.mark.parametrize("mode,field", [("O1", "y1_code"), ("O2", "edit_code")])
def test_deployable_o1_o2_exactly_match_training_coordinate_codes(mode, field):
    x = torch.rand(2, 3, 16, 16)
    y1 = torch.rand_like(x)
    gt = torch.rand_like(x)
    sizes = [(16, 16), (8, 8), (4, 4)]
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    oracle = builder(x, y1, gt, sizes, requested={mode})
    stats = {
        "normalization": {
            mode: {"center": [0.0] * 8, "scale": [1.0] * 8}
        }
    }
    encoder = DeterministicFeedbackEncoder()
    encoder.configure(mode, stats)
    deployed = encoder(x, y1, sizes, mode)
    expected = [getattr(output, field) for output in oracle]
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(deployed, expected))


def test_deterministic_feedback_buffers_do_not_break_legacy_checkpoint_schema():
    model = tiny_model()
    assert not any(key.startswith("deterministic_feedback.") for key in model.state_dict())
