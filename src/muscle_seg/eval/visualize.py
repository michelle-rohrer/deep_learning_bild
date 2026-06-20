"""Rückwärtskompatibel: Overlay-Helfer + TensorBoard-Logging (TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muscle_seg.eval.tensorboard_viz import (
    LABEL_COLORS,
    TfEvalVisualizer,
    annotated_z_indices,
    build_patch_val_mosaic,
    build_subject_mosaic,
    log_subjects_to_tensorboard,
    overlay_axial_slice,
    water_contrast,
)

__all__ = [
    "LABEL_COLORS",
    "TfEvalVisualizer",
    "annotated_z_indices",
    "build_patch_val_mosaic",
    "build_subject_mosaic",
    "log_subjects_to_tensorboard",
    "log_subject_to_tensorboard",
    "overlay_axial_slice",
    "water_contrast",
]


def log_subject_to_tensorboard(
    log_dir: Path,
    subject: str,
    water: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: dict,
    *,
    n_slices: int = 6,
    step: int = 0,
) -> Path:
    """Einzelnes Subject nach TensorBoard loggen."""
    with TfEvalVisualizer(log_dir) as viz:
        viz.log_subject(subject, water, gt, pred, metrics, step=step, n_slices=n_slices)
    return log_dir
