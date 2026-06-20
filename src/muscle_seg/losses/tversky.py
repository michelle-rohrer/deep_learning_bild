"""Focal-Tversky Loss (BG ausgeschlossen, β > α → FN stärker gewichten)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muscle_seg.labels import NUM_CLASSES


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky Loss über Vordergrund-Klassen 1..N.

    Tversky-Index pro Klasse:
        TI_c = (TP + ε) / (TP + α·FP + β·FN + ε)

    Focal-Exponent γ < 1 verstärkt schwierige (niedrige TI) Klassen.
    Standard-Werte: α=0.3, β=0.7, γ=0.75 (Abraham & Khan 2019).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(
            targets.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        tp = (probs * targets_oh).sum(dims)
        fp = (probs * (1.0 - targets_oh)).sum(dims)
        fn = ((1.0 - probs) * targets_oh).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = (1.0 - tversky) ** self.gamma

        # Klasse 0 (BG) ausschliessen
        return focal_tversky[1:].mean()
