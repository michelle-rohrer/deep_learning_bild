"""Evaluation: Patch (schnell) oder Volume + Sliding Window (Sabina-Stil)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from muscle_seg.config import TrainConfig
from muscle_seg.data.dataset import build_datasets
from muscle_seg.data.splits import load_folds
from muscle_seg.data.volume_io import load_subject_volume, present_left_classes
from muscle_seg.eval.volume_predict import predict_volume_sliding_window
from muscle_seg.metrics.calibration import compute_ece
from muscle_seg.eval.tensorboard_viz import log_subjects_to_tensorboard
from muscle_seg.labels import NUM_CLASSES
from muscle_seg.metrics.dice import aggregate_macro_dice, compute_subject_metrics_left
from muscle_seg.models.bayesian_unet import BayesianUNet3D
from muscle_seg.train.trainer import Trainer, _collate, _device


def _load_model(cfg: TrainConfig, checkpoint_path: Path) -> BayesianUNet3D:
    device = _device(cfg.device)
    model = BayesianUNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def _eval_patch(
    cfg: TrainConfig,
    model: BayesianUNet3D,
    train_subjects: list[str],
    val_subjects: list[str],
) -> dict:
    """Legacy: Dice auf Val-Patches (zentriert auf GT-Muskeln)."""
    _, val_ds = build_datasets(
        cfg.data_dir,
        train_subjects,
        val_subjects,
        patch_size=cfg.patch_size,
        patches_per_volume=cfg.patches_per_volume,
        percentiles=cfg.intensity_clip_percentile,
        seed=cfg.seed,
    )
    loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
    )
    trainer = Trainer(cfg)
    return trainer.evaluate(model, loader, use_mc=cfg.eval_use_mc)


def _eval_volume(
    cfg: TrainConfig,
    model: BayesianUNet3D,
    val_subjects: list[str],
    *,
    log_tensorboard: bool = False,
    fold_idx: int = 0,
) -> dict:
    """Vollvolumen + Sliding Window; pro Subject Sabina-ähnliche Metriken."""
    device = _device(cfg.device)
    stride = cfg.eval_stride or tuple(s // 2 for s in cfg.patch_size)
    per_subject: list[dict] = []
    tb_subjects: list[tuple[str, object, object, object, dict]] = []

    for sid in val_subjects:
        image, gt = load_subject_volume(
            cfg.data_dir,
            sid,
            percentiles=cfg.intensity_clip_percentile,
            z_crop=cfg.eval_z_crop,
            z_margin=cfg.eval_z_margin,
            bbox_crop=cfg.eval_bbox_crop,
            bbox_margin=cfg.eval_bbox_margin,
        )
        result = predict_volume_sliding_window(
            model,
            image,
            cfg.patch_size,
            device,
            stride=stride,
            use_mc=cfg.eval_use_mc,
            mc_samples=cfg.mc_samples,
            show_progress=len(val_subjects) == 1,
            return_probs=cfg.eval_use_mc,  # Probs nur bei MC-Dropout für ECE
        )
        if cfg.eval_use_mc:
            pred, probs = result
            ece = compute_ece(probs, gt)
        else:
            pred = result
            ece = float("nan")

        present = present_left_classes(gt)
        m = compute_subject_metrics_left(sid, gt, pred, present_classes=present)
        m["mode"] = "volume"
        m["ece"] = ece
        per_subject.append(m)
        tb_subjects.append((sid, image, gt, pred, m))

    dices = [s["macro_dice_left"] for s in per_subject if s["macro_dice_left"] == s["macro_dice_left"]]
    mean_d, std_d = aggregate_macro_dice(dices)
    ece_vals = [s["ece"] for s in per_subject if s["ece"] == s["ece"]]
    mean_ece = float(np.mean(ece_vals)) if ece_vals else float("nan")

    tensorboard_log_dir: str | None = None
    if log_tensorboard and tb_subjects:
        tb_path = log_subjects_to_tensorboard(
            cfg.tensorboard_log_dir,
            cfg.experiment_name,
            fold_idx,
            tb_subjects,  # type: ignore[arg-type]
            fold_mean_dice=mean_d,
            n_slices=cfg.eval_n_slices,
        )
        tensorboard_log_dir = str(tb_path)

    return {
        "mode": "volume",
        "macro_dice_left": mean_d,
        "macro_dice_left_std": std_d,
        "mean_ece": mean_ece,
        "per_subject": per_subject,
        "tensorboard_log_dir": tensorboard_log_dir,
        "n_subjects": len(per_subject),
    }


def evaluate_checkpoint(
    cfg: TrainConfig,
    checkpoint_path: Path,
    fold_idx: int,
    *,
    log_tensorboard: bool = False,
) -> dict:
    cfg = cfg.resolve_paths()
    folds = load_folds(cfg.splits_path)
    fold = folds[fold_idx]
    train_subjects = list(fold["train"])
    val_subjects = list(fold["val"])
    model = _load_model(cfg, checkpoint_path)

    if cfg.eval_mode == "patch":
        metrics = _eval_patch(cfg, model, train_subjects, val_subjects)
        result = {
            "checkpoint": str(checkpoint_path),
            "fold": fold_idx,
            "val_subjects": val_subjects,
            **metrics,
        }
    else:
        result = {
            "checkpoint": str(checkpoint_path),
            "fold": fold_idx,
            "val_subjects": val_subjects,
            **_eval_volume(
                cfg,
                model,
                val_subjects,
                log_tensorboard=log_tensorboard,
                fold_idx=fold_idx,
            ),
        }
    return result


def evaluate_all_folds(
    cfg: TrainConfig,
    *,
    log_tensorboard: bool | None = None,
) -> dict:
    cfg = cfg.resolve_paths()
    if log_tensorboard is None:
        log_tensorboard = cfg.eval_log_tensorboard
    results = []
    for fold_idx in range(3):
        ckpt = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{fold_idx}" / "best.pt"
        if not ckpt.exists():
            continue
        results.append(
            evaluate_checkpoint(
                cfg,
                ckpt,
                fold_idx,
                log_tensorboard=log_tensorboard,
            )
        )

    dices = [r["macro_dice_left"] for r in results]
    mean_d, std_d = aggregate_macro_dice(dices)
    report = {
        "experiment": cfg.experiment_name,
        "eval_mode": cfg.eval_mode,
        "per_fold": results,
        "mean_macro_dice_left": mean_d,
        "std_macro_dice_left": std_d,
    }
    out = cfg.checkpoint_dir / cfg.experiment_name / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    return report
