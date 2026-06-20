"""Expected Calibration Error (ECE) für MC-Dropout Segmentierung."""

from __future__ import annotations

import numpy as np


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ignore_bg: bool = True,
) -> float:
    """ECE über ein ganzes 3D-Volumen.

    Args:
        probs:     (C, H, W, D) Softmax-Wahrscheinlichkeiten (MC-Mittel).
        labels:    (H, W, D)    Ground-Truth-Labels (int).
        n_bins:    Anzahl Konfidenz-Bins (gleichmässig 0..1).
        ignore_bg: Hintergrund-Voxel (Label 0) von der Berechnung ausschliessen.

    Returns:
        ECE ∈ [0, 1] — 0 = perfekt kalibriert.
    """
    confidence = probs.max(axis=0).ravel()   # höchste Klassen-Prob pro Voxel
    predicted = probs.argmax(axis=0).ravel() # vorhergesagte Klasse
    gt = labels.ravel()

    if ignore_bg:
        mask = gt > 0
        confidence = confidence[mask]
        predicted = predicted[mask]
        gt = gt[mask]

    if len(confidence) == 0:
        return float("nan")

    correct = (predicted == gt).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(confidence)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence >= lo) & (confidence < hi)
        if in_bin.sum() == 0:
            continue
        avg_conf = float(confidence[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        ece += in_bin.sum() / n_total * abs(avg_conf - avg_acc)

    return float(ece)
