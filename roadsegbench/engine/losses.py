"""Manuscript-specified compound segmentation loss for RoadSegBench:
L = 1.0 * CrossEntropy (BCEWithLogits) + 0.5 * SoftDice + 0.75 * LovaszHinge
Reference: Berman et al., 'The Lovasz-Softmax loss: A tractable surrogate for the
optimization of the intersection-over-union measure in neural networks', CVPR 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    targets_f = targets.float()
    dims = (1, 2, 3)
    inter = (probs * targets_f).sum(dims)
    denom = probs.sum(dims) + targets_f.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    if logits_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = F.relu(1.0 - logits_flat.float() * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(errors_sorted, grad)


def lovasz_hinge_loss(logits: torch.Tensor, targets: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    if per_image and logits.ndim >= 2:
        b = logits.shape[0]
        if b == 0:
            return logits.sum() * 0.0
        log_2d = logits.float().reshape(b, -1)
        tgt_2d = targets.float().reshape(b, -1)
        signs = 2.0 * tgt_2d - 1.0
        errors = F.relu(1.0 - log_2d * signs)
        errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
        gt_sorted = torch.gather(tgt_2d, 1, perm)
        gts = gt_sorted.sum(dim=1, keepdim=True)
        intersection = gts - gt_sorted.cumsum(dim=1)
        union = gts + (1.0 - gt_sorted).cumsum(dim=1)
        jaccard = 1.0 - intersection / union.clamp_min(1e-8)
        if jaccard.shape[1] > 1:
            jaccard[:, 1:] = jaccard[:, 1:] - jaccard[:, :-1]
        return (errors_sorted * jaccard).sum(dim=1).mean()
    return lovasz_hinge_flat(logits.reshape(-1), targets.reshape(-1))


class RoadSegBenchCompoundLoss(nn.Module):
    """1.0 * BCEWithLogits + 0.5 * SoftDice + 0.75 * LovaszHinge."""
    def __init__(self, w_ce: float = 1.0, w_dice: float = 0.5, w_lovasz: float = 0.75):
        super().__init__()
        self.w_ce = float(w_ce)
        self.w_dice = float(w_dice)
        self.w_lovasz = float(w_lovasz)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        targets_f = targets.float()
        l_ce = self.bce(logits_f, targets_f)
        l_dice = soft_dice_loss(logits_f, targets_f)
        l_lov = lovasz_hinge_loss(logits_f, targets_f, per_image=True)
        return self.w_ce * l_ce + self.w_dice * l_dice + self.w_lovasz * l_lov
