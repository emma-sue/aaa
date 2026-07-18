"""Training-only SRSC target construction in a strict 81-D local space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CoordinateOutput:
    state: torch.Tensor
    p_raw: torch.Tensor
    p: torch.Tensor
    m_raw: torch.Tensor
    m: torch.Tensor
    d: torch.Tensor
    q: torch.Tensor
    w_dir: torch.Tensor
    orthogonal_relative_error: torch.Tensor
    residual_code: torch.Tensor | None
    error_magnitude: torch.Tensor | None
    vstar_norm: torch.Tensor
    y1_code: torch.Tensor | None
    edit_code: torch.Tensor | None
    transverse_code: torch.Tensor | None
    correction: torch.Tensor | None
    state_pca: torch.Tensor | None
    unit_deviation: torch.Tensor | None


def _orthogonal_rows(rows: int, cols: int, seed: int) -> torch.Tensor:
    if rows > cols:
        raise ValueError("rows must not exceed cols")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=generator), mode="reduced")
    return q.transpose(0, 1).contiguous()


class SRSCCoordinateBuilder(nn.Module):
    """Build multi-scale GT-derived targets; never called by inference forward."""

    def __init__(
        self,
        patch_size: int = 3,
        grad_weight: float = 0.5,
        direction_dim: int = 6,
        tau_v: float = 0.05,
        tau_e: float = 0.05,
        eps: float = 1e-6,
        pca_projection: torch.Tensor | None = None,
        pca_mean: torch.Tensor | None = None,
    ):
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        self.patch_size = patch_size
        self.grad_weight = grad_weight
        self.direction_dim = direction_dim
        self.tau_v = tau_v
        self.tau_e = tau_e
        self.eps = eps
        descriptor_dim = 9 * patch_size * patch_size
        self.descriptor_dim = descriptor_dim
        self.register_buffer("P", _orthogonal_rows(direction_dim, descriptor_dim, 20260713))
        self.register_buffer("Pr", _orthogonal_rows(8, descriptor_dim, 20260714))
        self.register_buffer("Py1", _orthogonal_rows(8, descriptor_dim, 20260715))
        self.register_buffer("Pedit", _orthogonal_rows(8, descriptor_dim, 20260716))
        self.register_buffer("Pe", _orthogonal_rows(8, descriptor_dim, 20260717))
        if pca_projection is None:
            pca_projection = torch.empty(0, descriptor_dim)
        pca_projection = torch.as_tensor(pca_projection, dtype=torch.float32)
        if pca_projection.shape not in {(0, descriptor_dim), (direction_dim, descriptor_dim)}:
            raise ValueError(
                f"pca_projection must be empty or {(direction_dim, descriptor_dim)}, "
                f"got {tuple(pca_projection.shape)}"
            )
        if pca_projection.numel():
            identity = torch.eye(direction_dim, dtype=pca_projection.dtype)
            if not torch.allclose(pca_projection @ pca_projection.T, identity, atol=1e-4, rtol=1e-4):
                raise ValueError("pca_projection rows must be orthonormal")
        self.register_buffer("P_pca", pca_projection.contiguous())
        if pca_mean is None:
            pca_mean = torch.zeros(descriptor_dim)
        pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32).reshape(-1)
        if pca_mean.shape != (descriptor_dim,):
            raise ValueError(
                f"pca_mean must have shape {(descriptor_dim,)}, got {tuple(pca_mean.shape)}"
            )
        if not pca_projection.numel() and torch.count_nonzero(pca_mean):
            raise ValueError("pca_mean requires pca_projection")
        self.register_buffer("pca_mean", pca_mean.contiguous())
        # The preregistered descriptor is [RGB, 0.5*Sobel_x,
        # 0.5*Sobel_y].  Do not silently normalize the Sobel kernel again.
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        kernels = torch.stack((sobel_x, sobel_y)).unsqueeze(1)
        self.register_buffer("sobel", kernels)

    def descriptor(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        flat = image.reshape(b * c, 1, h, w)
        flat = F.pad(flat, (1, 1, 1, 1), mode="reflect")
        grad = F.conv2d(flat, self.sobel).reshape(b, c, 2, h, w)
        gx, gy = grad[:, :, 0], grad[:, :, 1]
        return torch.cat((image, self.grad_weight * gx, self.grad_weight * gy), dim=1)

    def unfold(self, descriptor: torch.Tensor) -> torch.Tensor:
        p = self.patch_size // 2
        descriptor = F.pad(descriptor, (p, p, p, p), mode="reflect")
        patches = F.unfold(descriptor, kernel_size=self.patch_size)
        b, q, hw = patches.shape
        h = descriptor.shape[-2] - 2 * p
        w = descriptor.shape[-1] - 2 * p
        if hw != h * w or q != self.descriptor_dim:
            raise RuntimeError("unexpected unfolded descriptor shape")
        return patches.reshape(b, q, h, w)

    def _single(
        self,
        x: torch.Tensor,
        y1: torch.Tensor,
        gt: torch.Tensor,
        requested: set[str] | None = None,
    ) -> CoordinateOutput:
        with torch.no_grad():
            build_all = requested is None
            dx = self.descriptor(x)
            vstar = self.unfold(self.descriptor(gt) - dx)
            v1 = self.unfold(self.descriptor(y1.detach()) - dx)
            nstar = vstar.square().sum(1, keepdim=True)
            alpha = (v1 * vstar).sum(1, keepdim=True) / (nstar + self.eps)
            p_raw = 1.0 - alpha
            p = 2.0 * torch.tanh(p_raw / 2.0)
            e = v1 - alpha * vstar
            enorm2 = e.square().sum(1, keepdim=True)
            m_raw = torch.sqrt(enorm2 + self.eps) / (
                torch.sqrt(nstar + self.eps) + self.eps
            )
            m = 2.0 * torch.tanh(m_raw / 2.0)
            q = nstar / (nstar + self.tau_v**2)
            w_dir = q * m_raw / (m_raw + self.tau_e)
            u = e / (torch.sqrt(enorm2) + self.eps)
            d = torch.einsum("dq,bqhw->bdhw", self.P.to(u), u)
            p_eff = q * p
            m_eff = q * m
            d_eff = w_dir * d
            state = torch.cat((p_eff, m_eff, d_eff), dim=1)
            state_pca = None
            if self.P_pca.numel():
                centered_u = u - self.pca_mean.to(u).view(1, -1, 1, 1)
                d_pca = torch.einsum("dq,bqhw->bdhw", self.P_pca.to(u), centered_u)
                state_pca = torch.cat((p_eff, m_eff, w_dir * d_pca), dim=1)
            unit_deviation = (
                u if build_all or (requested is not None and "PCA_STATS" in requested) else None
            )
            y1_code = edit_code = transverse_code = correction = None
            residual_code = error_magnitude = None
            if build_all or "O1" in requested:
                y1_local = self.unfold(self.descriptor(y1.detach()))
                y1_code = torch.einsum("dq,bqhw->bdhw", self.Py1.to(y1_local), y1_local)
            if build_all or "O2" in requested:
                edit_local = self.unfold(self.descriptor(y1.detach()) - dx)
                edit_code = torch.einsum("dq,bqhw->bdhw", self.Pedit.to(edit_local), edit_local)
            if build_all or "O13" in requested:
                transverse_code = torch.einsum("dq,bqhw->bdhw", self.Pe.to(e), e)
            if build_all or requested.intersection({"O3", "O12", "O14"}):
                correction = self.unfold(self.descriptor(gt) - self.descriptor(y1.detach()))
                if build_all or "O12" in requested:
                    residual_code = torch.einsum("dq,bqhw->bdhw", self.Pr.to(correction), correction)
                if build_all or "O3" in requested:
                    error_magnitude = torch.sqrt(
                        correction.square().sum(1, keepdim=True) + self.eps
                    )
            dot_e = (e * vstar).sum(1, keepdim=True).abs()
            denom = torch.sqrt(enorm2 * nstar + self.eps)
            orth_error = dot_e / denom
        return CoordinateOutput(
            state=state,
            p_raw=p_raw,
            p=p,
            m_raw=m_raw,
            m=m,
            d=d,
            q=q,
            w_dir=w_dir,
            orthogonal_relative_error=orth_error,
            residual_code=residual_code,
            error_magnitude=error_magnitude,
            vstar_norm=torch.sqrt(nstar),
            y1_code=y1_code,
            edit_code=edit_code,
            transverse_code=transverse_code,
            correction=correction,
            state_pca=state_pca,
            unit_deviation=unit_deviation,
        )

    def forward(
        self,
        x: torch.Tensor,
        y1: torch.Tensor,
        gt: torch.Tensor,
        sizes: Sequence[tuple[int, int]],
        requested: set[str] | None = None,
    ) -> list[CoordinateOutput]:
        outputs = []
        for size in sizes:
            args = [F.interpolate(t, size=size, mode="area") for t in (x, y1, gt)]
            outputs.append(self._single(*args, requested=requested))
        return outputs
