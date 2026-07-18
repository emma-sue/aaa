import io

import torch

from src.net import SRSCLite


def make_model():
    return SRSCLite(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        d1_blocks=(1, 1, 1),
        d2_blocks=(1, 1, 1),
        d2_refinement=1,
        heads=(1, 1, 2, 4),
        expansion=2.0,
    ).eval()


def test_checkpoint_roundtrip():
    torch.manual_seed(3)
    model = make_model()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(x)
    assert torch.allclose(actual, expected, atol=1e-7)
