#!/usr/bin/env python3
"""Prüft, ob die Umgebung für GPU-Training und TensorBoard-Visualisierung bereit ist."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path.cwd()
    ok = True

    print("=== DLBS Umgebungs-Check ===\n")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA verfügbar: {cuda}")
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Keine GPU – Training nur mit configs/*_cpu.yaml sinnvoll (sehr langsam).")
            ok = False
    except ImportError:
        print("❌ PyTorch nicht installiert")
        ok = False

    try:
        from torch.utils.tensorboard.writer import SummaryWriter  # noqa: F401

        print("TensorBoard (SummaryWriter): ok")
    except ImportError:
        print("⚠️  tensorboard fehlt (pip install tensorboard)")
        ok = False

    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__} (tf.summary für Eval-Bilder)")
    except ImportError:
        print("⚠️  tensorflow fehlt (pip install -e .)")
        ok = False

    for data_name in ("data", "../cleaned_data_2"):
        data = (root / data_name).resolve()
        if not data.is_dir():
            continue
        for sub in ("water", "masks"):
            p = data / sub
            n = len([x for x in p.iterdir() if x.is_dir()]) if p.is_dir() else 0
            status = "ok" if n > 0 else "FEHLT"
            print(f"{data_name}/{sub}: {status} ({n} Subjects)")
            if n == 0:
                ok = False
        break
    else:
        print("Daten: FEHLT (data/ oder ../cleaned_data_2 mit water/, masks/)")
        ok = False

    splits = root / "splits" / "folds.json"
    print(f"splits/folds.json: {'ok' if splits.exists() else 'fehlt – python scripts/make_splits.py'}")

    print("\nPipeline:")
    print("  python scripts/train.py --config configs/baseline.yaml")
    print("  python scripts/evaluate.py --config configs/baseline.yaml")
    print("  python scripts/visualize_predictions.py --fold 0")
    print("  python scripts/launch_tensorboard.py --logdir runs/baseline")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
