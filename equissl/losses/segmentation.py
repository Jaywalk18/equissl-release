"""
Segmentation loss functions for EquiSSL finetuning.
- FocalLoss: down-weights easy examples, helps rare classes
- CEDiceLoss: CE + Dice hybrid for better boundary segmentation
- build_class_weights: inverse-frequency weighting from training set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss: -alpha * (1-p)^gamma * log(p)"""

    def __init__(self, gamma=2.0, weight=None, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             ignore_index=self.ignore_index, reduction="none")
        p = torch.exp(-ce)
        focal = ((1 - p) ** self.gamma) * ce
        return focal.mean()


class CEDiceLoss(nn.Module):
    """CE + Dice hybrid loss."""

    def __init__(self, ce_weight=1.0, dice_weight=0.5, ignore_index=0, num_classes=14):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # Dice loss (per-class, then average)
        probs = F.softmax(logits, dim=1) if logits.dim() > 2 else F.softmax(logits, dim=-1)
        if logits.dim() == 2:
            # (N, C) -> one-hot targets (N, C)
            valid = targets != self.ignore_index
            probs = probs[valid]
            t = targets[valid]
            one_hot = F.one_hot(t, self.num_classes).float()
        else:
            valid = targets != self.ignore_index
            probs = probs[valid]
            t = targets[valid]
            one_hot = F.one_hot(t, self.num_classes).float()

        dice_loss = 0.0
        count = 0
        for c in range(1, self.num_classes):  # skip class 0
            p_c = probs[:, c] if probs.dim() > 1 else probs
            g_c = one_hot[:, c]
            intersection = (p_c * g_c).sum()
            union = p_c.sum() + g_c.sum()
            if union > 0:
                dice_loss += 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
                count += 1
        if count > 0:
            dice_loss = dice_loss / count

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def build_class_weights(dataset, num_classes=14, ignore_index=0):
    """Compute inverse-frequency class weights from training set."""
    print("Computing class weights from training set...")
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample["sphere_gt_sem"].numpy()
        for c in range(num_classes):
            counts[c] += (labels == c).sum()

    # Inverse frequency, normalized
    valid_counts = counts.copy()
    valid_counts[ignore_index] = 0
    valid_mask = valid_counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if valid_mask.any():
        freq = valid_counts[valid_mask]
        inv_freq = 1.0 / freq
        inv_freq = inv_freq / inv_freq.sum() * valid_mask.sum()
        weights[valid_mask] = inv_freq

    weights[ignore_index] = 0.0
    print(f"Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)
