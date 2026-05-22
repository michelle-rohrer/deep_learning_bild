"""Standalone-Evaluation geladener Checkpoints (nach Training / vor Tuning)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from muscle_seg.config import TrainConfig
from muscle_seg.data.dataset import build_datasets
from muscle_seg.data.splits import load_folds
from muscle_seg.labels import NUM_CLASSES
from muscle_seg.metrics.dice import aggregate_macro_dice
from muscle_seg.models.bayesian_unet import BayesianUNet3D
from muscle_seg.train.trainer import Trainer, _collate, _device


def evaluate_checkpoint(
    cfg: TrainConfig,
    checkpoint_path: Path,
    fold_idx: int,
) -> dict:
    cfg = cfg.resolve_paths()
    folds = load_folds(cfg.splits_path)
    fold = folds[fold_idx]
    _, val_ds = build_datasets(
        cfg.data_dir,
        fold["train"],
        fold["val"],
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

    model = BayesianUNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(_device(cfg.device))

    state = torch.load(checkpoint_path, map_location=_device(cfg.device), weights_only=False)
    model.load_state_dict(state["model"])

    trainer = Trainer(cfg)
    metrics = trainer.evaluate(model, loader)
    return {
        "checkpoint": str(checkpoint_path),
        "fold": fold_idx,
        **metrics,
    }


def evaluate_all_folds(cfg: TrainConfig) -> dict:
    cfg = cfg.resolve_paths()
    results = []
    for fold_idx in range(3):
        ckpt = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{fold_idx}" / "best.pt"
        if not ckpt.exists():
            continue
        results.append(evaluate_checkpoint(cfg, ckpt, fold_idx))

    dices = [r["macro_dice_left"] for r in results]
    mean_d, std_d = aggregate_macro_dice(dices)
    report = {
        "experiment": cfg.experiment_name,
        "per_fold": results,
        "mean_macro_dice_left": mean_d,
        "std_macro_dice_left": std_d,
    }
    out = cfg.checkpoint_dir / cfg.experiment_name / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    return report
