"""Dice + Focal Loss Kombination für unbalancierte Muskel-Segmentierung."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muscle_seg.labels import NUM_CLASSES


class DiceFocalLoss(nn.Module):
    """Dice (BG ausgeschlossen) + gewichteter Focal Loss.

    Focal Loss bestraft falsch klassifizierte Voxel stärker (gamma > 0) und
    gewichtet seltene Klassen hoch (alpha). Kombiniert mit Dice für stabile
    Segmentierung kleiner Muskeln.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        smooth: float = 1e-5,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.gamma = gamma
        self.focal_weight = focal_weight

        if alpha is None:
            alpha = [1.0] * num_classes
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # --- Dice (BG ausgeschlossen) ---
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_oh).sum(dims)
        cardinality = probs.sum(dims) + targets_oh.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice[1:].mean()

        # --- Focal Loss ---
        log_probs = F.log_softmax(logits, dim=1)
        # Wahrscheinlichkeit der korrekten Klasse pro Voxel
        p_t = (probs * targets_oh).sum(dim=1)          # (B, H, W, D)
        log_p_t = (log_probs * targets_oh).sum(dim=1)  # (B, H, W, D)
        # Klassengewicht pro Voxel
        alpha_t = (self.alpha.to(logits.device).view(1, -1, 1, 1, 1) * targets_oh).sum(dim=1)
        focal = -alpha_t * (1.0 - p_t) ** self.gamma * log_p_t
        focal_loss = focal.mean()

        return dice_loss + self.focal_weight * focal_loss
