#!/usr/bin/env python3
"""Evaluation aller Fold-Checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.config import TrainConfig
from muscle_seg.eval.run_eval import evaluate_all_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="DLBS Evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config).resolve_paths(Path.cwd())
    report = evaluate_all_folds(cfg)
    print(report)


if __name__ == "__main__":
    main()
