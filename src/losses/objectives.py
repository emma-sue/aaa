import torch
from torch.nn import functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + eps**2).mean()


def state_loss(
    predictions,
    targets,
    direction_cosine_weight: float = 0.1,
    direction_weights=None,
    direction_valid_masks=None,
):
    base = sum(F.smooth_l1_loss(p, t) for p, t in zip(predictions, targets)) / len(targets)
    cosine_terms = []
    for scale_index, (prediction, target) in enumerate(zip(predictions, targets)):
        pd, td = prediction[:, 2:8], target[:, 2:8]
        cosine = 1.0 - F.cosine_similarity(pd, td, dim=1, eps=1e-6).unsqueeze(1)
        if direction_valid_masks is not None:
            valid = direction_valid_masks[scale_index].detach().bool()
        else:
            # Fallback for callers without raw-coordinate masks.  Formal SRSC
            # training passes an explicit mask built before normalization.
            valid = td.float().norm(dim=1, keepdim=True) >= 1e-6
        if direction_weights is not None:
            weight = direction_weights[scale_index].detach().float()
            valid = valid & (weight >= 1e-3)
            cosine = cosine * weight
        # The specification sets invalid positions to zero and then aggregates
        # the loss map.  Conditional averaging over only valid pixels would
        # silently amplify sparse direction regions.
        cosine_terms.append(torch.where(valid, cosine, torch.zeros_like(cosine)).mean())
    direction = torch.stack(cosine_terms).mean() if cosine_terms else base.new_zeros(())
    return base + direction_cosine_weight * direction, {"state_base": base.detach(), "state_cos": direction.detach()}
