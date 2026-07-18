import torch

from src.net.srsc_lite import SRSCMod, SRSCLite


def tiny_model(force_zero_state=False):
    return SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
        force_zero_state=force_zero_state,
    )


def test_zero_init_is_identity():
    mod = SRSCMod(16)
    f = torch.randn(2, 16, 11, 13)
    s = torch.randn(2, 8, 11, 13)
    assert torch.allclose(mod(f, s), f, atol=1e-7)


def test_oracle_feedback_requires_eight_channels():
    model = tiny_model()
    x = torch.rand(1, 3, 32, 32)
    bad = [torch.zeros(1, 7, 32 // 2**i, 32 // 2**i) for i in range(4)]
    try:
        model.forward_with_feedback(x, bad)
    except ValueError as exc:
        assert "8 channels" in str(exc)
    else:
        raise AssertionError("invalid feedback width was accepted")


def test_state_loss_does_not_reach_encoder_d1():
    model = tiny_model()
    x = torch.rand(1, 3, 32, 32)
    xpad = x
    features, y1 = model._encode_coarse(xpad)
    states = model.assessor(xpad, y1, features)
    sum(s.square().mean() for s in states).backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.d1.parameters())
    assert any(p.grad is not None for p in model.assessor.parameters())


def test_final_loss_reaches_all_restoration_stages():
    model = tiny_model()
    x = torch.rand(1, 3, 32, 32)
    model(x).mean().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.d1.parameters())
    assert any(p.grad is not None for p in model.d2.parameters())


def test_dummy_zero_state_keeps_assessor_parameters():
    active = tiny_model(False)
    dummy = tiny_model(True)
    assert sum(p.numel() for p in active.parameters()) == sum(p.numel() for p in dummy.parameters())


def test_dummy_zero_state_executes_assessor_pyramid_d2_and_all_modulators():
    model = tiny_model(True).eval()
    calls = {"assessor": 0, "pyramid": 0, "mods": 0}
    state_inputs = []

    def count_assessor(_module, _inputs, _output):
        calls["assessor"] += 1

    def count_pyramid(_module, _inputs, _output):
        calls["pyramid"] += 1

    def capture_mod(_module, inputs):
        calls["mods"] += 1
        state_inputs.append(inputs[1].detach().clone())

    handles = [
        model.assessor.register_forward_hook(count_assessor),
        model.y1_pyramid.register_forward_hook(count_pyramid),
    ]
    handles.extend(
        module.register_forward_pre_hook(capture_mod)
        for module in (model.d2.mod1, model.d2.mod2, model.d2.mod3, model.d2.mod4)
    )
    try:
        with torch.no_grad():
            details = model.forward_details(torch.rand(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"assessor": 1, "pyramid": 1, "mods": 4}
    assert len(details.states) == len(state_inputs) == 4
    assert all(state.shape[1] == 8 for state in state_inputs)
    assert all(torch.count_nonzero(state) == 0 for state in state_inputs)
    # The assessor output is retained for state supervision/reporting even
    # though the common D2 interface is zeroed for the matched O0 control.
    assert any(torch.count_nonzero(state) > 0 for state in details.states)
