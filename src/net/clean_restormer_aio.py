"""Clean prompt-free Restormer-AiO baseline."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .restormer_blocks import crop_to_shape, pad_to_multiple
from .srsc_lite import RestorationDecoder, SharedEncoder


class CleanRestormerAiO(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        self.encoder = SharedEncoder(dim, encoder_blocks, heads, expansion, bias, norm_type)
        self.decoder = RestorationDecoder(
            dim, decoder_blocks, refinement, heads[:3], expansion, bias, norm_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, original = pad_to_multiple(x, 8)
        delta, _ = self.decoder(self.encoder(x_pad))
        return crop_to_shape(x_pad + delta, original)
