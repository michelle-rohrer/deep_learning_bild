#!/usr/bin/env python3
"""CC-Postprocessing auf bestehenden Checkpoints – kein Retraining.

Lädt Prediction via Sliding Window, wendet CC-Cleanup an,
berechnet Dice vorher/nachher und zeigt entferntes Volumen pro Klasse.

Verwendung:
    python scripts/eval_cc.py --config configs/baseline.yaml --mode largest
    python scripts/eval_cc.py --config configs/hpt_focal.yaml --mode threshold --min-voxels 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from muscle_seg.config import TrainConfig
from muscle_seg.data.splits import load_folds
from muscle_seg.data.volume_io import load_subject_volume, present_left_classes
from muscle_seg.eval.postprocess import cc_cleanup
from muscle_seg.eval.volume_predict import predict_volume_sliding_window
from muscle_seg.labels import LABEL_NAMES, NUM_CLASSES
from muscle_seg.metrics.dice import compute_subject_metrics_left
from muscle_seg.models.bayesian_unet import BayesianUNet3D
from muscle_seg.train.trainer import _device


def _load_model(cfg: TrainConfig, ckpt: Path) -> BayesianUNet3D:
    device = _device(cfg.device)
    model = BayesianUNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def eval_fold_cc(
    cfg: TrainConfig,
    fold_idx: int,
    mode: str,
    min_voxels: int,
) -> dict:
    device = _device(cfg.device)
    stride = cfg.eval_stride or tuple(s // 2 for s in cfg.patch_size)
    ckpt = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{fold_idx}" / "best.pt"
    if not ckpt.exists():
        print(f"  [fold {fold_idx}] Checkpoint fehlt: {ckpt}")
        return {}

    folds = load_folds(cfg.splits_path)
    val_subjects = list(folds[fold_idx]["val"])
    model = _load_model(cfg, ckpt)

    results_before, results_after = [], []
    cc_summary: list[dict] = []

    for sid in val_subjects:
        image, gt = load_subject_volume(
            cfg.data_dir, sid,
            percentiles=cfg.intensity_clip_percentile,
            z_crop=cfg.eval_z_crop,
            z_margin=cfg.eval_z_margin,
            bbox_crop=cfg.eval_bbox_crop,
            bbox_margin=cfg.eval_bbox_margin,
        )
        pred_raw = predict_volume_sliding_window(
            model, image, cfg.patch_size, device, stride=stride,
            use_mc=False, mc_samples=1,
        )
        pred_cc, stats = cc_cleanup(pred_raw, mode=mode, min_voxels=min_voxels)

        present = present_left_classes(gt)
        m_before = compute_subject_metrics_left(sid, gt, pred_raw, present_classes=present)
        m_after  = compute_subject_metrics_left(sid, gt, pred_cc,  present_classes=present)

        results_before.append(m_before["macro_dice_left"])
        results_after.append(m_after["macro_dice_left"])

        # Per-class Dice-Delta + entferntes Volumen
        per_class_info = {}
        for c in range(1, NUM_CLASSES):
            name = LABEL_NAMES.get(c, str(c))
            st = stats.get(c, {})
            d_before = m_before["per_class"].get(name, {}).get("dice", float("nan"))
            d_after  = m_after["per_class"].get(name, {}).get("dice", float("nan"))
            per_class_info[name] = {
                "dice_before": round(d_before, 4) if d_before == d_before else None,
                "dice_after":  round(d_after, 4)  if d_after  == d_after  else None,
                "removed_voxels": st.get("removed", 0),
                "total_pred": st.get("total_pred", 0),
            }
        cc_summary.append({"subject": sid, "per_class": per_class_info})

    mean_before = float(np.mean(results_before)) if results_before else float("nan")
    mean_after  = float(np.mean(results_after))  if results_after  else float("nan")

    print(f"  Fold {fold_idx}: Dice before={mean_before:.4f}  after={mean_after:.4f}  "
          f"delta={mean_after - mean_before:+.4f}")
    for subj_data in cc_summary:
        print(f"    {subj_data['subject']}:")
        for cls_name, info in subj_data["per_class"].items():
            if info["total_pred"] == 0 and info["dice_before"] is None:
                continue
            print(
                f"      {cls_name:8s}  "
                f"dice {info['dice_before']} -> {info['dice_after']}  "
                f"removed={info['removed_voxels']}/{info['total_pred']} voxels"
            )
    return {
        "fold": fold_idx,
        "mean_dice_before": mean_before,
        "mean_dice_after": mean_after,
        "delta": mean_after - mean_before,
        "per_subject": cc_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["largest", "threshold"], default="largest")
    parser.add_argument("--min-voxels", type=int, default=100,
                        help="Nur bei --mode threshold: Mindestgrösse einer Komponente")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config).resolve_paths()
    print(f"\nCC-Postprocessing: {cfg.experiment_name}  mode={args.mode}")

    all_results = []
    for fold_idx in args.folds:
        r = eval_fold_cc(cfg, fold_idx, args.mode, args.min_voxels)
        if r:
            all_results.append(r)

    if all_results:
        deltas = [r["delta"] for r in all_results]
        befores = [r["mean_dice_before"] for r in all_results]
        afters  = [r["mean_dice_after"]  for r in all_results]
        print(f"\nGesamt: Dice before={np.mean(befores):.4f}  "
              f"after={np.mean(afters):.4f}  delta={np.mean(deltas):+.4f}")

        out = cfg.checkpoint_dir / cfg.experiment_name / f"cc_report_{args.mode}.json"
        out.write_text(json.dumps({"mode": args.mode, "folds": all_results}, indent=2))
        print(f"Report: {out}")


if __name__ == "__main__":
    main()
