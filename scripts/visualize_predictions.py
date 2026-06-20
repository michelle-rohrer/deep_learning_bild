#!/usr/bin/env python3
"""
Predictions in TensorBoard loggen (TensorFlow tf.summary.image).

Anzeigen:
  python scripts/launch_tensorboard.py --logdir runs/baseline
  → Tab „Images“: eval/<subject>/mosaic_water_gt_pred
"""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.config import TrainConfig
from muscle_seg.data.splits import load_folds
from muscle_seg.data.volume_io import load_subject_volume, present_left_classes
from muscle_seg.eval.run_eval import _load_model
from muscle_seg.eval.tensorboard_viz import log_subjects_to_tensorboard
from muscle_seg.eval.volume_predict import predict_volume_sliding_window
from muscle_seg.metrics.dice import compute_subject_metrics_left
from muscle_seg.train.trainer import _device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLBS Predictions → TensorBoard (TensorFlow)",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Komma-getrennt, leer = alle Val-Subjects des Folds",
    )
    parser.add_argument("--mc", action="store_true", help="MC-Dropout-Inferenz (langsamer)")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config).resolve_paths(Path.cwd())
    folds = load_folds(cfg.splits_path)
    val_subjects = list(folds[args.fold]["val"])
    if args.subjects.strip():
        val_subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    ckpt = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{args.fold}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint fehlt: {ckpt}")

    model = _load_model(cfg, ckpt)
    device = _device(cfg.device)
    stride = cfg.eval_stride or tuple(s // 2 for s in cfg.patch_size)
    use_mc = args.mc or cfg.eval_use_mc

    tb_data: list[tuple] = []
    for sid in val_subjects:
        print(f"[tf-viz] Inferenz {sid} …")
        image, gt = load_subject_volume(
            cfg.data_dir,
            sid,
            percentiles=cfg.intensity_clip_percentile,
            z_crop=cfg.eval_z_crop,
            z_margin=cfg.eval_z_margin,
        )
        pred = predict_volume_sliding_window(
            model,
            image,
            cfg.patch_size,
            device,
            stride=stride,
            use_mc=use_mc,
            mc_samples=cfg.mc_samples,
            show_progress=True,
        )
        metrics = compute_subject_metrics_left(
            sid, gt, pred, present_classes=present_left_classes(gt),
        )
        tb_data.append((sid, image, gt, pred, metrics))
        print(f"  macro_dice_left={metrics['macro_dice_left']:.4f}")

    log_dir = log_subjects_to_tensorboard(
        cfg.tensorboard_log_dir,
        cfg.experiment_name,
        args.fold,
        tb_data,
        n_slices=cfg.eval_n_slices,
    )
    print(f"\nTensorBoard-Logs: {log_dir.resolve()}")
    print("Starten:")
    print(f"  python scripts/launch_tensorboard.py --logdir {cfg.tensorboard_log_dir / cfg.experiment_name}")


if __name__ == "__main__":
    main()
