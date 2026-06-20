"""Multi-Klassen Dice Loss (Hintergrund ausgeschlossen)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muscle_seg.labels import NUM_CLASSES


class MultiClassDiceLoss(nn.Module):
    """Softmax-Dice über Vordergrund-Klassen (BG ausgeschlossen).

    Hintergrund (Klasse 0) hat Dice ~0.99 und würde den Loss-Gradienten
    für die kleinen Muskelklassen verwässern.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_oh).sum(dims)
        cardinality = probs.sum(dims) + targets_oh.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Klasse 0 (BG) ausschliessen — nur Klassen 1..N mitteln
        return 1.0 - dice[1:].mean()
