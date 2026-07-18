"""Hard, representation-specific controls at the predicted D2 interface.

The assessor always executes and keeps an eight-channel output for capacity
matching and state supervision.  A control arm must nevertheless prevent the
restoration loss from hiding forbidden information in nominally unused
channels.  These transforms are therefore applied *after* the assessor and
immediately before D2, during training, validation, and deployment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


PREDICTED_FEEDBACK_MODES = frozenset(
    {"O0", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O15"}
)
DETERMINISTIC_FEEDBACK_MODES = frozenset({"O1", "O2"})
DIRECTIONAL_FEEDBACK_MODES = frozenset({"O7", "O8", "O9", "O10", "O15"})


def validate_native_feedback_pyramid(
    states: Sequence[torch.Tensor],
    expected_sizes: Sequence[tuple[int, int]],
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    """Validate the native S1..S4 interface without resizing feedback.

    A spatial resize is not an innocuous convenience here: each state target is
    constructed independently after downsampling ``x/y1/gt`` to its native
    encoder scale.  Interpolating a state from another scale therefore changes
    the registered information set.  Keep this check at the common interface
    and fail closed before D2 executes.
    """

    if len(states) != 4 or len(expected_sizes) != 4:
        raise ValueError("feedback must contain exactly native S1..S4")
    validated: list[torch.Tensor] = []
    for scale_index, (state, expected_size) in enumerate(
        zip(states, expected_sizes), start=1
    ):
        if not isinstance(state, torch.Tensor) or state.ndim != 4:
            raise ValueError(f"S{scale_index} feedback must be a Bx8xHxW tensor")
        if state.shape[0] != batch_size:
            raise ValueError(
                f"S{scale_index} feedback batch {state.shape[0]} does not match "
                f"input batch {batch_size}"
            )
        if state.shape[1] != 8:
            raise ValueError(f"S{scale_index} feedback must have exactly 8 channels")
        actual_size = tuple(state.shape[-2:])
        native_size = tuple(expected_size)
        if actual_size != native_size:
            raise ValueError(
                f"S{scale_index} feedback has spatial size {actual_size}; "
                f"expected native size {native_size}; interpolation is forbidden"
            )
        validated.append(state)
    return validated


def isotropic_direction_normalization(
    mode: str,
    center: Sequence[float],
    scale: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Canonicalize an 8-D feedback normalization without warping direction.

    Channels 2..7 form one six-dimensional direction vector for SRSC and PCA
    controls.  Independent channel scales apply a diagonal linear transform and
    change angles/cosines.  A shared scalar preserves their geometry.  Scalar
    progress/magnitude channels remain independently normalized.

    The shared value is the ordinary median of the six positive finite scales,
    matching the robust-statistics policy while remaining dependency-free.
    """

    normalized_center = [float(value) for value in center]
    normalized_scale = [float(value) for value in scale]
    if len(normalized_center) != 8 or len(normalized_scale) != 8:
        raise ValueError("feedback normalization must contain exactly 8 channels")
    scale_tensor = torch.tensor(normalized_scale, dtype=torch.float64)
    if not torch.isfinite(scale_tensor).all() or torch.any(scale_tensor < 1e-4):
        raise ValueError("feedback normalization scales must be finite and >= 1e-4")
    center_tensor = torch.tensor(normalized_center, dtype=torch.float64)
    if not torch.isfinite(center_tensor).all():
        raise ValueError("feedback normalization centers must be finite")
    if mode in DIRECTIONAL_FEEDBACK_MODES:
        ordered = sorted(normalized_scale[2:8])
        shared_scale = 0.5 * (ordered[2] + ordered[3])
        # Gated direction coordinates use exact zero as their neutral value.
        # A non-zero direction center would move that neutral point and is not
        # permitted by the scale-only preregistered normalization.
        normalized_center[2:8] = [0.0] * 6
        normalized_scale[2:8] = [shared_scale] * 6
    return normalized_center, normalized_scale


def _orthogonal_rows(rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(cols, rows, generator=generator), mode="reduced"
    )
    return q.transpose(0, 1).contiguous()


class DeterministicFeedbackEncoder(nn.Module):
    """Deployable O1/O2 codes computed only from x and the coarse output y1."""

    def __init__(self, patch_size: int = 3, grad_weight: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        self.grad_weight = grad_weight
        descriptor_dim = 9 * patch_size * patch_size
        self.descriptor_dim = descriptor_dim
        # Non-persistent fixed buffers preserve strict compatibility with the
        # Stage-A checkpoints created before this deployable control was added.
        self.register_buffer(
            "Py1", _orthogonal_rows(8, descriptor_dim, 20260715), persistent=False
        )
        self.register_buffer(
            "Pedit", _orthogonal_rows(8, descriptor_dim, 20260716), persistent=False
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        self.register_buffer(
            "sobel",
            torch.stack((sobel_x, sobel_x.transpose(0, 1))).unsqueeze(1),
            persistent=False,
        )
        self.register_buffer("center", torch.zeros(8), persistent=False)
        self.register_buffer("scale", torch.ones(8), persistent=False)
        self.configured_mode: str | None = None

    def configure(self, mode: str, statistics: dict) -> None:
        if mode not in DETERMINISTIC_FEEDBACK_MODES:
            raise ValueError(f"deterministic feedback only supports O1/O2, got {mode}")
        spec = statistics["normalization"][mode]
        center = torch.as_tensor(spec["center"], dtype=self.center.dtype, device=self.center.device)
        scale = torch.as_tensor(spec["scale"], dtype=self.scale.dtype, device=self.scale.device)
        if center.shape != (8,) or scale.shape != (8,) or torch.any(scale < 1e-4):
            raise ValueError("invalid deterministic feedback normalization statistics")
        self.center.copy_(center)
        self.scale.copy_(scale)
        self.configured_mode = mode

    def descriptor(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        flat = image.reshape(batch * channels, 1, height, width)
        flat = F.pad(flat, (1, 1, 1, 1), mode="reflect")
        gradient = F.conv2d(flat, self.sobel).reshape(
            batch, channels, 2, height, width
        )
        return torch.cat(
            (image, self.grad_weight * gradient[:, :, 0], self.grad_weight * gradient[:, :, 1]),
            dim=1,
        )

    def unfold(self, descriptor: torch.Tensor) -> torch.Tensor:
        padding = self.patch_size // 2
        padded = F.pad(
            descriptor, (padding, padding, padding, padding), mode="reflect"
        )
        patches = F.unfold(padded, kernel_size=self.patch_size)
        batch, channels, _ = patches.shape
        height, width = descriptor.shape[-2:]
        return patches.reshape(batch, channels, height, width)

    def forward(
        self,
        x: torch.Tensor,
        y1: torch.Tensor,
        sizes: Sequence[tuple[int, int]],
        mode: str,
    ) -> list[torch.Tensor]:
        if self.configured_mode != mode:
            raise RuntimeError(
                f"deterministic feedback normalization is not configured for {mode}"
            )
        projection = self.Py1 if mode == "O1" else self.Pedit
        center = self.center.to(x).view(1, 8, 1, 1)
        scale = self.scale.to(x).view(1, 8, 1, 1)
        outputs = []
        for size in sizes:
            xd = F.interpolate(x.detach(), size=size, mode="area")
            yd = F.interpolate(y1.detach(), size=size, mode="area")
            if mode == "O1":
                local = self.unfold(self.descriptor(yd))
            else:
                local = self.unfold(self.descriptor(yd) - self.descriptor(xd))
            code = torch.einsum("dq,bqhw->bdhw", projection.to(local), local)
            outputs.append((code - center) / scale)
        return outputs


def corrupt_direction_control(tensor: torch.Tensor) -> torch.Tensor:
    """Misalign direction without ever becoming identity at batch size one."""
    if tensor.shape[0] > 1:
        return torch.roll(tensor, shifts=1, dims=0)
    height, width = tensor.shape[-2:]
    return torch.roll(
        tensor,
        shifts=(max(1, height // 2), max(1, width // 2)),
        dims=(-2, -1),
    )


def fixed_random_state_like(tensor: torch.Tensor, scale_index: int) -> torch.Tensor:
    """Return repeatable unit noise without consuming the global RNG stream."""
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(20260718 + scale_index)
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )


def predicted_supervision_mode(interface_mode: str) -> str:
    """Target used by the assessor for a predicted feedback/control arm.

    O9/O10/O11 isolate what D2 is allowed to consume.  Their assessor is
    therefore supervised with the same full SRSC target as O7 while the
    corruption/zero/noise intervention is active at the D2 interface from the
    first training step.  This avoids conflating a negative control with an
    impossible or lower-bandwidth assessor task.
    """
    if interface_mode not in PREDICTED_FEEDBACK_MODES:
        raise ValueError(f"unsupported predicted feedback mode: {interface_mode}")
    return "O7" if interface_mode in {"O9", "O10", "O11"} else interface_mode


def apply_predicted_feedback_interface(
    states: Sequence[torch.Tensor], mode: str
) -> list[torch.Tensor]:
    """Enforce the registered information set immediately before D2."""
    if mode not in PREDICTED_FEEDBACK_MODES:
        raise ValueError(f"unsupported predicted feedback mode: {mode}")
    transformed: list[torch.Tensor] = []
    for scale_index, state in enumerate(states):
        if state.ndim != 4 or state.shape[1] != 8:
            raise ValueError("predicted feedback must be Bx8xHxW at every scale")
        if mode == "O0":
            controlled = torch.zeros_like(state)
        elif mode in {"O3", "O5"}:
            controlled = torch.zeros_like(state)
            controlled[:, :1] = state[:, :1]
        elif mode in {"O4", "O6"}:
            controlled = torch.zeros_like(state)
            controlled[:, :2] = state[:, :2]
        elif mode in {"O7", "O12", "O15"}:
            controlled = state
        elif mode == "O8":
            controlled = state.clone()
            controlled[:, :1] = state[:, :1].abs()
        elif mode == "O9":
            controlled = state.clone()
            controlled[:, 2:] = corrupt_direction_control(state[:, 2:])
        elif mode == "O10":
            controlled = state.clone()
            controlled[:, 2:] = 0
        elif mode == "O11":
            controlled = fixed_random_state_like(state, scale_index)
        else:  # guarded above; keeps future additions fail-closed
            raise AssertionError(mode)
        transformed.append(controlled)
    return transformed
