#!/usr/bin/env python3
"""Overfitting-Test auf einem Fall (Dice >= 0.90)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from muscle_seg.config import TrainConfig
from muscle_seg.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfitting-Validierung")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/overfit_single.yaml"),
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    overrides: dict = {}
    if args.subject:
        overrides["subjects"] = [args.subject]
        overrides["overfit_subject"] = args.subject
    if args.no_tensorboard:
        overrides["use_tensorboard"] = False

    cfg = TrainConfig.from_yaml(args.config, overrides=overrides).resolve_paths(Path.cwd())
    report = Trainer(cfg).run()
    dice = report.get("final_macro_dice_left", 0.0)
    target = cfg.overfit_target_dice
    print(f"Final Dice (links): {dice:.4f}  (Ziel: {target:.2f})")
    if dice < target:
        sys.exit(1)


if __name__ == "__main__":
    main()
