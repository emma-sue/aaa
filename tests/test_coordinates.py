import torch
from torch.nn import functional as F

from src.net import SRSCCoordinateBuilder


def test_projection_shapes_and_orthogonality():
    builder = SRSCCoordinateBuilder()
    assert builder.P.shape == (6, 81)
    assert builder.Pr.shape == (8, 81)
    assert torch.allclose(builder.P @ builder.P.T, torch.eye(6), atol=1e-5)
    assert torch.allclose(builder.Pr @ builder.Pr.T, torch.eye(8), atol=1e-5)
    for projection in (builder.Py1, builder.Pedit, builder.Pe):
        assert projection.shape == (8, 81)
        assert torch.allclose(projection @ projection.T, torch.eye(8), atol=1e-5)


def test_descriptor_unfold_shape():
    builder = SRSCCoordinateBuilder()
    x = torch.rand(2, 3, 17, 19)
    assert builder.unfold(builder.descriptor(x)).shape == (2, 81, 17, 19)


def test_completed_state_is_zero():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 24, 24)
    gt = torch.rand_like(x)
    out = builder(x, gt, gt, [(24, 24)])[0]
    assert out.p.abs().mean() < 1e-4
    assert out.m_raw.mean() < 5e-3


def test_under_and_over_correction_sign():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 24, 24) * 0.3
    gt = x + 0.3
    under = builder(x, x + 0.5 * (gt - x), gt, [(24, 24)])[0]
    over = builder(x, x + 1.5 * (gt - x), gt, [(24, 24)])[0]
    assert under.p_raw.mean() > 0.45
    assert over.p_raw.mean() < -0.45


def test_transverse_is_orthogonal():
    builder = SRSCCoordinateBuilder(tau_v=1e-4, tau_e=1e-4)
    x = torch.rand(1, 3, 24, 24)
    gt = torch.rand_like(x)
    y1 = 0.7 * gt + 0.3 * x + 0.03 * torch.randn_like(x)
    out = builder(x, y1, gt, [(24, 24)])[0]
    valid = out.q > 0.5
    assert out.orthogonal_relative_error[valid].median() < 1e-4


def test_state_is_gated_in_clean_region():
    builder = SRSCCoordinateBuilder(tau_v=0.05, tau_e=0.05)
    x = torch.full((1, 3, 16, 16), 0.5)
    out = builder(x, x, x, [(16, 16)])[0]
    assert out.state.abs().max() < 1e-6
    assert torch.isfinite(out.state).all()
def test_descriptor_uses_half_raw_sobel_without_hidden_divisor():
    builder = SRSCCoordinateBuilder()
    image = torch.arange(25.0).reshape(1, 1, 5, 5)
    rgb = image.repeat(1, 3, 1, 1)
    descriptor = builder.descriptor(rgb)
    kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).reshape(1, 1, 3, 3)
    expected = 0.5 * F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel)
    assert torch.allclose(descriptor[:, 3:4], expected)
