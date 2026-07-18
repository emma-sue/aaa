"""Generic Restormer blocks adapted from PromptIR commit 106159a.

This file intentionally contains no PromptIR prompt generation or conditioning.
"""

from __future__ import annotations

import numbers
from typing import Iterable

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, norm_type: str):
        super().__init__()
        body = BiasFreeLayerNorm if norm_type == "BiasFree" else WithBiasLayerNorm
        self.body = body(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float, bias: bool):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, bias: bool):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        pattern = "b (head c) h w -> b head c (h w)"
        q = rearrange(q, pattern, head=self.heads)
        k = rearrange(k, pattern, head=self.heads)
        v = rearrange(v, pattern, head=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = rearrange(
            attn @ v,
            "b head c (h w) -> b (head c) h w",
            head=self.heads,
            h=h,
            w=w,
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attn = Attention(dim, heads, bias)
        self.norm2 = LayerNorm(dim, norm_type)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def make_blocks(
    dim: int,
    count: int,
    heads: int,
    expansion: float = 2.66,
    bias: bool = False,
    norm_type: str = "WithBias",
) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(dim, heads, expansion, bias, norm_type)
            for _ in range(count)
        ]
    )


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        mode = "reflect" if h > ph and w > pw else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (h, w)


def crop_to_shape(x: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    h, w = shape
    return x[..., :h, :w]
