#!/usr/bin/env python3
"""Training starten (Baseline oder beliebige YAML-Config)."""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.config import TrainConfig
from muscle_seg.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="DLBS Training")
    parser.add_argument("--config", type=Path, required=True, help="YAML-Konfiguration")
    parser.add_argument("--fold", type=int, default=None, help="Nur einen Fold trainieren (0–2)")
    parser.add_argument("--no-tensorboard", action="store_true", help="TensorBoard deaktivieren")
    args = parser.parse_args()

    overrides: dict = {}
    if args.fold is not None:
        overrides["fold"] = args.fold
    if args.no_tensorboard:
        overrides["use_tensorboard"] = False

    cfg = TrainConfig.from_yaml(args.config, overrides=overrides).resolve_paths(Path.cwd())
    report = Trainer(cfg).run()
    print(report)


if __name__ == "__main__":
    main()
