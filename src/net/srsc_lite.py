"""End-to-end SRSC-Lite v1.4 model.

Inference graph: x -> shared encoder -> D1 -> assessor -> D2 -> y2.
No GT, task labels, prompts, experts, recurrence, or dynamic stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .feedback_controls import (
    DETERMINISTIC_FEEDBACK_MODES,
    DeterministicFeedbackEncoder,
    apply_predicted_feedback_interface,
    validate_native_feedback_pyramid,
)
from .restormer_blocks import (
    Downsample,
    OverlapPatchEmbed,
    Upsample,
    crop_to_shape,
    make_blocks,
    pad_to_multiple,
)


class SharedEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        self.dim = dim
        widths = [dim * 2**i for i in range(4)]
        self.widths = widths
        self.patch = OverlapPatchEmbed(3, dim, bias)
        self.level1 = make_blocks(widths[0], blocks[0], heads[0], expansion, bias, norm_type)
        self.down12 = Downsample(widths[0])
        self.level2 = make_blocks(widths[1], blocks[1], heads[1], expansion, bias, norm_type)
        self.down23 = Downsample(widths[1])
        self.level3 = make_blocks(widths[2], blocks[2], heads[2], expansion, bias, norm_type)
        self.down34 = Downsample(widths[2])
        self.level4 = make_blocks(widths[3], blocks[3], heads[3], expansion, bias, norm_type)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f1 = self.level1(self.patch(x))
        f2 = self.level2(self.down12(f1))
        f3 = self.level3(self.down23(f2))
        f4 = self.level4(self.down34(f3))
        return f1, f2, f3, f4


class RestorationDecoder(nn.Module):
    """Ordinary decoder used for D1 and the clean single-stage baseline."""

    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (2, 2, 2),
        refinement: int = 0,
        heads: Sequence[int] = (1, 2, 4),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        c1, c2, c3, c4 = [dim * 2**i for i in range(4)]
        self.up43 = Upsample(c4)
        self.fuse3 = nn.Conv2d(c3 * 2, c3, 1, bias=bias)
        self.dec3 = make_blocks(c3, blocks[0], heads[2], expansion, bias, norm_type)
        self.up32 = Upsample(c3)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, 1, bias=bias)
        self.dec2 = make_blocks(c2, blocks[1], heads[1], expansion, bias, norm_type)
        self.up21 = Upsample(c2)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, 1, bias=bias)
        self.dec1 = make_blocks(c1, blocks[2], heads[0], expansion, bias, norm_type)
        self.refine = make_blocks(c1, refinement, heads[0], expansion, bias, norm_type)
        self.head = nn.Conv2d(c1, 3, 3, padding=1, bias=bias)

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        f1, f2, f3, f4 = features
        z3 = self.dec3(self.fuse3(torch.cat((self.up43(f4), f3), dim=1)))
        z2 = self.dec2(self.fuse2(torch.cat((self.up32(z3), f2), dim=1)))
        z1 = self.dec1(self.fuse1(torch.cat((self.up21(z2), f1), dim=1)))
        z1 = self.refine(z1)
        return self.head(z1), z1


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.act(self.dw(x)))


class StateAssessor(nn.Module):
    def __init__(self, dim: int = 48, evidence_widths: Sequence[int] = (32, 48, 64, 96)):
        super().__init__()
        e1, e2, e3, e4 = evidence_widths
        self.stem = nn.Sequential(nn.Conv2d(9, e1, 3, padding=1), nn.GELU(), nn.Conv2d(e1, e1, 3, padding=1))
        self.down12 = nn.Sequential(nn.Conv2d(e1, e2, 3, 2, 1), nn.GELU())
        self.down23 = nn.Sequential(nn.Conv2d(e2, e3, 3, 2, 1), nn.GELU())
        self.down34 = nn.Sequential(nn.Conv2d(e3, e4, 3, 2, 1), nn.GELU())
        enc_widths = [dim * 2**i for i in range(4)]
        self.enc_proj = nn.ModuleList([nn.Conv2d(c, e, 1) for c, e in zip(enc_widths, evidence_widths)])
        self.fuse = nn.ModuleList([nn.Conv2d(e * 2, e, 1) for e in evidence_widths])
        self.body = nn.ModuleList(
            [nn.Sequential(DepthwiseResidual(e), DepthwiseResidual(e)) for e in evidence_widths]
        )
        self.head = nn.ModuleList([nn.Conv2d(e, 8, 3, padding=1) for e in evidence_widths])

    def forward(
        self, x: torch.Tensor, y1: torch.Tensor, features: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        evidence = torch.cat((x.detach(), y1.detach(), (y1 - x).detach()), dim=1)
        e1 = self.stem(evidence)
        e2 = self.down12(e1)
        e3 = self.down23(e2)
        e4 = self.down34(e3)
        pyramid = [e1, e2, e3, e4]
        states = []
        for e, f, proj, fuse, body, head in zip(
            pyramid, features, self.enc_proj, self.fuse, self.body, self.head
        ):
            z = fuse(torch.cat((e, proj(f.detach())), dim=1))
            states.append(head(body(z)))
        return states


class Y1Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.GELU())
        self.g2 = nn.Sequential(nn.Conv2d(24, 48, 3, 2, 1), nn.GELU())
        self.g3 = nn.Sequential(nn.Conv2d(48, 96, 3, 2, 1), nn.GELU())
        self.g4 = nn.Sequential(nn.Conv2d(96, 192, 3, 2, 1), nn.GELU())

    def forward(self, y1: torch.Tensor) -> list[torch.Tensor]:
        g1 = self.g1(y1)
        g2 = self.g2(g1)
        g3 = self.g3(g2)
        g4 = self.g4(g3)
        return [g1, g2, g3, g4]


class SRSCMod(nn.Module):
    def __init__(self, channels: int, state_channels: int = 8, gamma_scale: float = 0.1):
        super().__init__()
        hidden = min(64, max(16, channels // 2))
        self.adapter = nn.Sequential(
            nn.Conv2d(state_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.gamma = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Conv2d(hidden, channels, 1)
        self.gamma_scale = gamma_scale
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feature: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        s = self.adapter(state)
        gamma = self.gamma_scale * torch.tanh(self.gamma(s))
        return (1.0 + gamma) * feature + self.beta(s)


class CorrectionDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (4, 4, 4),
        refinement: int = 2,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        c1, c2, c3, c4 = [dim * 2**i for i in range(4)]
        self.fuse4 = nn.Conv2d(c4 + 192, c4, 1, bias=bias)
        self.mod4 = SRSCMod(c4)
        self.up43 = Upsample(c4)
        self.fuse3 = nn.Conv2d(c3 + c3 + 96, c3, 1, bias=bias)
        self.mod3 = SRSCMod(c3)
        self.dec3 = make_blocks(c3, blocks[0], heads[2], expansion, bias, norm_type)
        self.up32 = Upsample(c3)
        self.fuse2 = nn.Conv2d(c2 + c2 + 48, c2, 1, bias=bias)
        self.mod2 = SRSCMod(c2)
        self.dec2 = make_blocks(c2, blocks[1], heads[1], expansion, bias, norm_type)
        self.up21 = Upsample(c2)
        self.fuse1 = nn.Conv2d(c1 + c1 + 24, c1, 1, bias=bias)
        self.mod1 = SRSCMod(c1)
        self.dec1 = make_blocks(c1, blocks[2], heads[0], expansion, bias, norm_type)
        self.refine = make_blocks(c1, refinement, heads[0], expansion, bias, norm_type)
        self.head = nn.Conv2d(c1, 3, 3, padding=1, bias=bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        y1_features: Sequence[torch.Tensor],
        states: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        f1, f2, f3, f4 = features
        g1, g2, g3, g4 = y1_features
        s1, s2, s3, s4 = states
        z4 = self.mod4(self.fuse4(torch.cat((f4, g4), dim=1)), s4)
        z3 = self.fuse3(torch.cat((self.up43(z4), f3, g3), dim=1))
        z3 = self.dec3(self.mod3(z3, s3))
        z2 = self.fuse2(torch.cat((self.up32(z3), f2, g2), dim=1))
        z2 = self.dec2(self.mod2(z2, s2))
        z1 = self.fuse1(torch.cat((self.up21(z2), f1, g1), dim=1))
        z1 = self.refine(self.dec1(self.mod1(z1, s1)))
        return self.head(z1)


@dataclass
class SRSCOutput:
    y1: torch.Tensor
    y2: torch.Tensor
    states: list[torch.Tensor]
    features: tuple[torch.Tensor, ...]


class SRSCLite(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        d1_blocks: Sequence[int] = (2, 2, 2),
        d2_blocks: Sequence[int] = (4, 4, 4),
        d2_refinement: int = 2,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        force_zero_state: bool = False,
        predicted_feedback_mode: str | None = None,
    ):
        super().__init__()
        self.encoder = SharedEncoder(dim, encoder_blocks, heads, expansion, bias, norm_type)
        self.d1 = RestorationDecoder(dim, d1_blocks, 0, heads[:3], expansion, bias, norm_type)
        self.assessor = StateAssessor(dim)
        self.y1_pyramid = Y1Pyramid()
        self.d2 = CorrectionDecoder(
            dim, d2_blocks, d2_refinement, heads, expansion, bias, norm_type
        )
        # Instantiated for every variant so the non-capacity-matched O14
        # ceiling cannot silently change the model definition at Tier-2.
        # It is never called by the deployable inference forward.
        self.oracle_ceiling_adapter = nn.Conv2d(81, 8, 1, bias=False)
        self.deterministic_feedback = DeterministicFeedbackEncoder()
        self.force_zero_state = force_zero_state
        self.predicted_feedback_mode = predicted_feedback_mode
        self.deterministic_feedback_mode: str | None = None

    def configure_deterministic_feedback(self, mode: str, statistics: dict) -> None:
        if mode not in DETERMINISTIC_FEEDBACK_MODES:
            raise ValueError(mode)
        self.predicted_feedback_mode = None
        self.deterministic_feedback_mode = mode
        self.deterministic_feedback.configure(mode, statistics)

    def _encode_coarse(self, x: torch.Tensor):
        features = self.encoder(x)
        delta0, _ = self.d1(features)
        return features, x + delta0

    def _run_d2(
        self,
        x: torch.Tensor,
        y1: torch.Tensor,
        features: Sequence[torch.Tensor],
        states: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if self.predicted_feedback_mode is not None:
            states = apply_predicted_feedback_interface(
                states, self.predicted_feedback_mode
            )
        if self.force_zero_state:
            states = [torch.zeros_like(s) for s in states]
        delta1 = self.d2(features, self.y1_pyramid(y1), states)
        return y1 + delta1

    def forward_details(self, x: torch.Tensor) -> SRSCOutput:
        x_pad, original = pad_to_multiple(x, 8)
        features, y1 = self._encode_coarse(x_pad)
        states = self.assessor(x_pad, y1, features)
        if self.deterministic_feedback_mode is not None:
            d2_states = self.deterministic_feedback(
                x_pad,
                y1,
                [feature.shape[-2:] for feature in features],
                self.deterministic_feedback_mode,
            )
        else:
            d2_states = states
        y2 = self._run_d2(x_pad, y1, features, d2_states)
        return SRSCOutput(
            y1=crop_to_shape(y1, original),
            y2=crop_to_shape(y2, original),
            states=states,
            features=features,
        )

    def forward_with_feedback(
        self, x: torch.Tensor, feedback: Sequence[torch.Tensor]
    ) -> SRSCOutput:
        """Training-only oracle/control path; feedback order is S1..S4."""
        x_pad, original = pad_to_multiple(x, 8)
        features, y1 = self._encode_coarse(x_pad)
        expected = [f.shape[-2:] for f in features]
        states = validate_native_feedback_pyramid(
            feedback, expected, batch_size=x_pad.shape[0]
        )
        y2 = self._run_d2(x_pad, y1, features, states)
        return SRSCOutput(crop_to_shape(y1, original), crop_to_shape(y2, original), states, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x).y2
