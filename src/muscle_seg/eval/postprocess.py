"""Connected-Component Postprocessing: per-class Rauschentfernung."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from muscle_seg.labels import NUM_CLASSES


def cc_cleanup(
    pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    mode: str = "largest",
    min_voxels: int = 50,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Per-class Connected-Component Cleanup auf Argmax-Volume.

    mode='largest'  : pro Klasse nur die grösste Komponente behalten.
    mode='threshold': alle Komponenten >= min_voxels behalten (sicherer für
                      kleine/fragmentierte Muskeln wie BfSH, SM).

    Returns (cleaned_pred, stats) mit stats[class_id] = {
        'total_pred': Voxel vor Cleanup,
        'removed':    entfernte Voxel (fast nur FP),
        'kept_components': Anzahl behaltener Komponenten,
    }
    """
    out = np.zeros_like(pred)
    stats: dict[int, dict] = {}

    for c in range(1, num_classes):
        mask_c = pred == c
        total = int(mask_c.sum())

        if total == 0:
            stats[c] = {"total_pred": 0, "removed": 0, "kept_components": 0}
            continue

        labeled, n_comp = ndimage.label(mask_c)
        comp_sizes = np.array(
            ndimage.sum(mask_c, labeled, range(1, n_comp + 1)), dtype=np.int64
        )

        if mode == "largest":
            keep_ids = {int(np.argmax(comp_sizes)) + 1}
        else:
            keep_ids = {i + 1 for i, sz in enumerate(comp_sizes) if sz >= min_voxels}

        if keep_ids:
            kept_mask = np.isin(labeled, sorted(keep_ids))
            out[kept_mask] = c
            kept = int(kept_mask.sum())
        else:
            kept = 0

        stats[c] = {
            "total_pred": total,
            "removed": total - kept,
            "kept_components": len(keep_ids),
        }

    return out, stats
