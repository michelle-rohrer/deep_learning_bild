"""Visualisierungshelfer für TensorBoard: Mosaike, Overlays, Uncertainty-Maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from muscle_seg.labels import LABEL_NAMES, LEFT_MUSCLE_LABELS

# ---------------------------------------------------------------------------
# Farbpalette (0=BG schwarz, 1–8 je distinkt)
# ---------------------------------------------------------------------------

LABEL_COLORS: np.ndarray = np.array(
    [
        [0.00, 0.00, 0.00],  # 0: BG
        [0.90, 0.15, 0.15],  # 1: VL_R  rot
        [0.15, 0.70, 0.15],  # 2: RF_R  grün
        [0.15, 0.15, 0.90],  # 3: VM_R  blau
        [0.95, 0.60, 0.10],  # 4: VI_R  orange
        [0.70, 0.10, 0.90],  # 5: BfSH_R lila
        [0.10, 0.85, 0.85],  # 6: BfLH_R cyan
        [0.90, 0.90, 0.10],  # 7: ST_R  gelb
        [0.95, 0.40, 0.75],  # 8: SM_R  pink
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Bildhelfer
# ---------------------------------------------------------------------------

def water_contrast(img_2d: np.ndarray) -> np.ndarray:
    """Percentile-Clip + Normalisierung auf [0, 1] float32."""
    lo, hi = np.percentile(img_2d, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(img_2d, dtype=np.float32)
    return np.clip((img_2d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def overlay_axial_slice(
    water_2d: np.ndarray,
    label_2d: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Farbiges Label-Overlay auf Graustufen-Wasserbild.

    Returns:
        (H, W, 3) float32 [0, 1]
    """
    gray = water_contrast(water_2d)
    rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)

    fg = label_2d > 0
    if fg.any():
        colors = LABEL_COLORS[label_2d.clip(0, len(LABEL_COLORS) - 1)]
        rgb[fg] = (1 - alpha) * rgb[fg] + alpha * colors[fg]

    return rgb.astype(np.float32)


def annotated_z_indices(mask_3d: np.ndarray, n: int = 6) -> list[int]:
    """N gleichmässig verteilte Z-Indizes mit Annotation; fallback: mittlere Slices."""
    annotated = np.where(np.any(mask_3d > 0, axis=(0, 1)))[0]
    if len(annotated) == 0:
        mid = mask_3d.shape[2] // 2
        start = max(0, mid - n // 2)
        return list(range(start, min(mask_3d.shape[2], start + n)))
    step = max(1, len(annotated) // n)
    indices = annotated[::step][:n].tolist()
    return indices


# ---------------------------------------------------------------------------
# Mosaik-Builder
# ---------------------------------------------------------------------------

def build_patch_val_mosaic(
    water_3d: np.ndarray,
    gt_3d: np.ndarray,
    pred_3d: np.ndarray,
    uncertainty_3d: np.ndarray | None = None,
) -> np.ndarray:
    """Mittlerer Z-Slice: [Water | GT | Pred | Uncertainty(optional)].

    Args:
        water_3d:       (H, W, D) float
        gt_3d:          (H, W, D) int
        pred_3d:        (H, W, D) int
        uncertainty_3d: (H, W, D) float (MC-Varianz), optional

    Returns:
        (H, W*n_cols, 3) float32 [0, 1]
    """
    z = water_3d.shape[2] // 2
    w2d = water_contrast(water_3d[:, :, z])
    gray3 = np.stack([w2d, w2d, w2d], axis=-1)

    gt_ov = overlay_axial_slice(water_3d[:, :, z], gt_3d[:, :, z])
    pr_ov = overlay_axial_slice(water_3d[:, :, z], pred_3d[:, :, z])

    panels = [gray3, gt_ov, pr_ov]

    if uncertainty_3d is not None:
        u2d = uncertainty_3d[:, :, z].astype(np.float32)
        u_max = u2d.max()
        if u_max > 0:
            u2d = u2d / u_max
        # Heatmap: blau (niedrig) → rot (hoch) via einfaches RGB-Mapping
        r = u2d
        g = np.zeros_like(u2d)
        b = 1.0 - u2d
        panels.append(np.stack([r, g, b], axis=-1))

    return np.concatenate(panels, axis=1).astype(np.float32)


def build_subject_mosaic(
    water_3d: np.ndarray,
    gt_3d: np.ndarray,
    pred_3d: np.ndarray,
    n_slices: int = 6,
    uncertainty_3d: np.ndarray | None = None,
) -> np.ndarray:
    """Grid mit N annotierten Slices: jede Zeile = [Water | GT | Pred | Unc?].

    Returns:
        (H*n_slices, W*n_cols, 3) float32 [0, 1]
    """
    z_indices = annotated_z_indices(gt_3d, n=n_slices)
    rows = []
    for z in z_indices:
        w2d = water_contrast(water_3d[:, :, z])
        gray3 = np.stack([w2d, w2d, w2d], axis=-1)
        gt_ov = overlay_axial_slice(water_3d[:, :, z], gt_3d[:, :, z])
        pr_ov = overlay_axial_slice(water_3d[:, :, z], pred_3d[:, :, z])
        panels = [gray3, gt_ov, pr_ov]

        if uncertainty_3d is not None:
            u2d = uncertainty_3d[:, :, z].astype(np.float32)
            u_max = uncertainty_3d.max()
            u2d = u2d / u_max if u_max > 0 else u2d
            r = u2d
            g = np.zeros_like(u2d)
            b = 1.0 - u2d
            panels.append(np.stack([r, g, b], axis=-1))

        rows.append(np.concatenate(panels, axis=1))

    return np.concatenate(rows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# TfEvalVisualizer – Context Manager für TF-basiertes Image-Logging
# ---------------------------------------------------------------------------

class TfEvalVisualizer:
    """TensorFlow-Summary-Writer für Eval-Bilder (als Context Manager)."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer = None

    def __enter__(self) -> "TfEvalVisualizer":
        import tensorflow as tf

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.create_file_writer(str(self._log_dir))
        return self

    def __exit__(self, *_) -> None:
        if self._writer is not None:
            self._writer.flush()

    def log_subject(
        self,
        subject: str,
        water: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        metrics: dict,
        *,
        step: int = 0,
        n_slices: int = 6,
        uncertainty: np.ndarray | None = None,
    ) -> None:
        import tensorflow as tf

        mosaic = build_subject_mosaic(water, gt, pred, n_slices=n_slices, uncertainty_3d=uncertainty)
        img = mosaic[np.newaxis].astype(np.float32)  # (1, H, W, 3)

        dice_str = f"{metrics.get('macro_dice_left', float('nan')):.3f}"
        tag = f"eval/{subject}_dice{dice_str}"

        with self._writer.as_default():
            tf.summary.image(tag, img, step=step, max_outputs=1)

            # Per-Klasse Dice als Text-Tabelle
            pc = metrics.get("per_class", {})
            if pc:
                rows = [f"| Muskel | Dice | Prec | Recall |", "|---|---|---|---|"]
                for name, vals in sorted(pc.items()):
                    rows.append(
                        f"| {name} | {vals.get('dice', float('nan')):.3f} "
                        f"| {vals.get('precision', float('nan')):.3f} "
                        f"| {vals.get('recall', float('nan')):.3f} |"
                    )
                tf.summary.text(f"eval/{subject}_per_class", "\n".join(rows), step=step)


# ---------------------------------------------------------------------------
# log_subjects_to_tensorboard – Haupt-Einstiegspunkt für Eval
# ---------------------------------------------------------------------------

def log_subjects_to_tensorboard(
    tensorboard_log_dir: Path,
    experiment_name: str,
    fold_idx: int,
    subjects_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]],
    *,
    fold_mean_dice: float,
    n_slices: int = 6,
) -> Path:
    """Für jeden Subject: Mosaik-Bild + per-Klasse Metriken nach TensorBoard.

    Args:
        subjects_data: Liste von (subject_id, water, gt, pred, metrics_dict)

    Returns:
        Pfad zum TensorBoard-Log-Verzeichnis
    """
    log_dir = (
        Path(tensorboard_log_dir)
        / experiment_name
        / f"fold{fold_idx}_eval"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    with TfEvalVisualizer(log_dir) as viz:
        for sid, water, gt, pred, metrics in subjects_data:
            viz.log_subject(sid, water, gt, pred, metrics, step=fold_idx, n_slices=n_slices)

    return log_dir
