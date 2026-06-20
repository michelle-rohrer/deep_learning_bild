#!/usr/bin/env python3
"""Confusion-Matrix auf Foreground-Voxeln (BG ausgeschlossen).

Zeigt wo das Modell Muskeln verwechselt vs. in Background kollabiert.
Verwendung:
    python scripts/eval_confusion.py --config configs/hpt_focal.yaml
    python scripts/eval_confusion.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from muscle_seg.config import TrainConfig
from muscle_seg.data.splits import load_folds
from muscle_seg.data.volume_io import load_subject_volume
from muscle_seg.eval.volume_predict import predict_volume_sliding_window
from muscle_seg.labels import LABEL_NAMES, NUM_CLASSES
from muscle_seg.models.bayesian_unet import BayesianUNet3D
from muscle_seg.train.trainer import _device


MUSCLE_NAMES = [LABEL_NAMES[c] for c in range(1, NUM_CLASSES)]  # 8 Namen


def _load_model(cfg: TrainConfig, ckpt: Path) -> BayesianUNet3D:
    device = _device(cfg.device)
    model = BayesianUNet3D(
        in_channels=1, num_classes=NUM_CLASSES,
        base_channels=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def confusion_foreground(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """8×8 Matrix: Zeile = GT-Klasse 1-8, Spalte = Pred-Klasse 0-8.
    Spalte 0 = BG (FN-Kollaps), Spalten 1-8 = Muskel-zu-Muskel."""
    # gt foreground mask
    fg = gt > 0
    gt_fg = gt[fg].astype(np.int64) - 1      # 0..7
    pred_fg = pred[fg].astype(np.int64)       # 0..8 (0=BG)

    mat = np.zeros((8, NUM_CLASSES), dtype=np.int64)  # 8 rows × 9 cols
    for g, p in zip(gt_fg, pred_fg):
        mat[g, p] += 1
    return mat


def print_confusion(mat: np.ndarray, title: str = "") -> None:
    """Druckt normalisierte Confusion-Matrix (Zeilen = Recall-Verteilung)."""
    row_sums = mat.sum(axis=1, keepdims=True).clip(1)
    normed = (mat / row_sums * 100).astype(int)

    col_headers = ["BG"] + MUSCLE_NAMES
    col_w = 7

    if title:
        print(f"\n{title}")
    header = f"{'GT\\Pred':<10}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(MUSCLE_NAMES):
        row = f"{name:<10}" + "".join(f"{normed[i, j]:>{col_w}}" for j in range(NUM_CLASSES))
        # highlight diagonal and BG column
        print(row)
    print()
    # Summary: %BG-collapse vs %correct vs %confused
    bg_col = normed[:, 0]
    diag = np.array([normed[i, i+1] for i in range(8)])
    confused = 100 - bg_col - diag
    print(f"{'Klasse':<10} {'→BG%':>6} {'Correct%':>9} {'Confused%':>10}")
    print("-" * 38)
    for i, name in enumerate(MUSCLE_NAMES):
        print(f"{name:<10} {bg_col[i]:>6} {diag[i]:>9} {confused[i]:>10}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config).resolve_paths()
    device = _device(cfg.device)
    stride = cfg.eval_stride or tuple(s // 2 for s in cfg.patch_size)
    folds_data = load_folds(cfg.splits_path)

    print(f"\n{'='*60}")
    print(f"Confusion Matrix: {cfg.experiment_name}")
    print(f"{'='*60}")

    mat_total = np.zeros((8, NUM_CLASSES), dtype=np.int64)
    fold_results = []

    for fold_idx in args.folds:
        ckpt = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{fold_idx}" / "best.pt"
        if not ckpt.exists():
            print(f"[fold {fold_idx}] Checkpoint fehlt, übersprungen.")
            continue

        val_subjects = list(folds_data[fold_idx]["val"])
        model = _load_model(cfg, ckpt)
        mat_fold = np.zeros((8, NUM_CLASSES), dtype=np.int64)

        for sid in val_subjects:
            image, gt = load_subject_volume(
                cfg.data_dir, sid,
                percentiles=cfg.intensity_clip_percentile,
                z_crop=cfg.eval_z_crop, z_margin=cfg.eval_z_margin,
                bbox_crop=cfg.eval_bbox_crop, bbox_margin=cfg.eval_bbox_margin,
            )
            pred = predict_volume_sliding_window(
                model, image, cfg.patch_size, device, stride=stride,
                use_mc=False, mc_samples=1,
            )
            mat_sid = confusion_foreground(gt, pred)
            mat_fold += mat_sid

            # Per-subject BG-collapse rate for Hamstrings
            row_sums = mat_sid.sum(axis=1).clip(1)
            bg_rates = mat_sid[:, 0] / row_sums * 100
            print(f"  [{fold_idx}] {sid}: BG-Kollaps Hamstrings — "
                  f"BfSH={bg_rates[4]:.0f}% BfLH={bg_rates[5]:.0f}% "
                  f"ST={bg_rates[6]:.0f}% SM={bg_rates[7]:.0f}%")

        print_confusion(mat_fold, title=f"Fold {fold_idx} (val: {val_subjects})")
        mat_total += mat_fold
        fold_results.append({"fold": fold_idx, "matrix": mat_fold.tolist()})

    print_confusion(mat_total, title="GESAMT (alle Folds)")

    # Save
    out = cfg.checkpoint_dir / cfg.experiment_name / "confusion_matrix.json"
    out.write_text(json.dumps({
        "experiment": cfg.experiment_name,
        "class_names": MUSCLE_NAMES,
        "col_headers": ["BG"] + MUSCLE_NAMES,
        "note": "rows=GT_class(1-8), cols=pred_class(0=BG,1-8=muscle)",
        "per_fold": fold_results,
        "total": mat_total.tolist(),
    }, indent=2))
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
