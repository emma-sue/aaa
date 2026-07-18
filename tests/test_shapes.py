import torch

from src.net import CleanRestormerAiO, SRSCLite


def tiny_srsc(**kwargs):
    return SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
        **kwargs,
    )


def test_srsc_even_and_odd_shapes():
    model = tiny_srsc().eval()
    with torch.no_grad():
        for shape in ((1, 3, 128, 128), (2, 3, 127, 131)):
            x = torch.rand(shape)
            y = model(x)
            assert y.shape == x.shape


def test_clean_baseline_shape():
    model = CleanRestormerAiO(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
    ).eval()
    x = torch.rand(1, 3, 65, 67)
    with torch.no_grad():
        assert model(x).shape == x.shape


def test_four_state_scales_and_width():
    model = tiny_srsc().eval()
    x = torch.rand(1, 3, 64, 72)
    with torch.no_grad():
        out = model.forward_details(x)
    assert [s.shape for s in out.states] == [
        torch.Size((1, 8, 64, 72)),
        torch.Size((1, 8, 32, 36)),
        torch.Size((1, 8, 16, 18)),
        torch.Size((1, 8, 8, 9)),
    ]


def test_inference_accepts_only_x():
    model = tiny_srsc().eval()
    assert model.forward.__code__.co_argcount == 2
    with torch.no_grad():
        model(torch.rand(1, 3, 32, 32))
