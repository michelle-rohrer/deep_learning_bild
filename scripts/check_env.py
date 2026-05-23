#!/usr/bin/env python3
"""Prüft, ob die Umgebung für GPU-Training bereit ist."""

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
        import tensorboard  # noqa: F401

        print("TensorBoard: ok")
    except ImportError:
        print("⚠️  tensorboard fehlt (pip install tensorboard)")

    data = root / "data"
    for sub in ("water", "masks"):
        p = data / sub
        n = len(list(p.iterdir())) if p.is_dir() else 0
        status = "ok" if n > 0 else "FEHLT"
        print(f"data/{sub}: {status} ({n} Einträge)")
        if n == 0:
            ok = False

    splits = root / "splits" / "folds.json"
    print(f"splits/folds.json: {'ok' if splits.exists() else 'fehlt – python scripts/make_splits.py'}")

    print("\nEmpfohlene Configs auf GPU/Cluster:")
    print("  configs/overfit_single.yaml")
    print("  configs/baseline.yaml")
    print("\nNur Laptop/CPU:")
    print("  configs/overfit_single_cpu.yaml")
    print("  configs/baseline_cpu.yaml")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
