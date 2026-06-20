#!/usr/bin/env python3
"""Evaluation aller Fold-Checkpoints (+ optional TensorBoard-Bilder)."""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.config import TrainConfig
from muscle_seg.eval.run_eval import evaluate_all_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="DLBS Evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Predictions als tf.summary.image nach runs/<experiment>/eval_fold_* loggen",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Keine Bilder loggen (nur eval_report.json)",
    )
    parser.add_argument(
        "--patch-eval",
        action="store_true",
        help="Schnelle Patch-Eval statt Vollvolumen (legacy)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config).resolve_paths(Path.cwd())
    if args.patch_eval:
        cfg.eval_mode = "patch"

    log_tb: bool | None = None
    if args.tensorboard:
        log_tb = True
    if args.no_tensorboard:
        log_tb = False

    report = evaluate_all_folds(cfg, log_tensorboard=log_tb)
    print(report)

    tb_dirs = [
        r.get("tensorboard_log_dir")
        for r in report.get("per_fold", [])
        if r.get("tensorboard_log_dir")
    ]
    if tb_dirs:
        print("\nTensorBoard (Tab „Images“):")
        print(f"  python scripts/launch_tensorboard.py --logdir {cfg.tensorboard_log_dir / cfg.experiment_name}")
        for d in tb_dirs:
            print(f"    → {d}")


if __name__ == "__main__":
    main()
