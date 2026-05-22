"""Evaluation: Macro-Dice über linke Muskelklassen 1–8 (BG ausgeschlossen)."""

from __future__ import annotations

import torch

from muscle_seg.labels import LEFT_MUSCLE_LABELS


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> float:
    p = (pred == class_id).float()
    t = (target == class_id).float()
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return float("nan")
    return float((2 * inter + smooth) / (denom + smooth))


def macro_dice_left_muscles(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, dict[int, float]]:
    """
    Mittlerer Dice über Klassen 1–8.
    Klassen ohne Voxel in der Ground Truth werden übersprungen.
    """
    per_class: dict[int, float] = {}
    scores: list[float] = []
    for c in LEFT_MUSCLE_LABELS:
        if (target == c).sum() == 0:
            continue
        d = dice_per_class(pred, target, c)
        per_class[c] = d
        if d == d:  # not nan
            scores.append(d)
    macro = sum(scores) / len(scores) if scores else 0.0
    return macro, per_class


def aggregate_macro_dice(values: list[float]) -> tuple[float, float]:
    """Mittelwert und Standardabweichung über Folds / Runs."""
    if not values:
        return 0.0, 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))
