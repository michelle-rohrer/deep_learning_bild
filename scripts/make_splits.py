#!/usr/bin/env python3
"""Erzeugt reproduzierbare 3-Fold-Splits (Fall-Ebene)."""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.data.splits import generate_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Fold CV Splits erzeugen")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("splits/folds.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    root = Path.cwd()
    folds = generate_and_save(
        (root / args.data_dir).resolve(),
        (root / args.out).resolve(),
        n_splits=args.n_splits,
        seed=args.seed,
    )
    for f in folds:
        print(f"Fold {f['fold']}: train={len(f['train'])}, val={len(f['val'])}")


if __name__ == "__main__":
    main()
