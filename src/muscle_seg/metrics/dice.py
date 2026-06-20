"""Evaluation: Macro-Dice über linke Muskelklassen 1–8 (BG ausgeschlossen)."""

from __future__ import annotations

import numpy as np
import torch

from muscle_seg.labels import LABEL_NAMES, LEFT_MUSCLE_LABELS


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


def compute_subject_metrics_left(
    subject: str,
    gt: np.ndarray,
    pred: np.ndarray,
    present_classes: list[int] | None = None,
) -> dict:
    """
    Pro-Klasse-Metriken auf dem ganzen Volume (Sabina-Stil, nur links 1–8).

    Klassen ohne GT und ohne Pred werden übersprungen.
    Nur-GT oder nur-Pred zählen (Dice kann 0 sein).
    """
    classes = list(present_classes) if present_classes is not None else list(LEFT_MUSCLE_LABELS)
    per_class: dict[str, dict[str, float]] = {}
    dice_vals: list[float] = []

    for c in classes:
        if c not in LEFT_MUSCLE_LABELS:
            continue
        p = pred == c
        g = gt == c
        p_sum = float(p.sum())
        g_sum = float(g.sum())
        if p_sum == 0 and g_sum == 0:
            continue
        tp = float((p & g).sum())
        denom = p_sum + g_sum
        dice = (2.0 * tp) / denom if denom > 0 else float("nan")
        prec = (tp / p_sum) if p_sum > 0 else float("nan")
        rec = (tp / g_sum) if g_sum > 0 else float("nan")
        name = LABEL_NAMES.get(c, str(c))
        per_class[name] = {
            "dice": dice,
            "precision": prec,
            "recall": rec,
            "gt_voxels": g_sum,
            "pred_voxels": p_sum,
        }
        if dice == dice:
            dice_vals.append(dice)

    macro = float(np.mean(dice_vals)) if dice_vals else float("nan")
    return {
        "subject": subject,
        "macro_dice_left": macro,
        "per_class": per_class,
        "n_classes_scored": len(dice_vals),
    }
