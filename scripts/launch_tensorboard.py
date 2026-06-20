#!/usr/bin/env python3
"""TensorBoard lokal starten (Trainings in Browser ansehen)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorBoard starten")
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs"),
        help="Log-Verzeichnis (Standard: runs/)",
    )
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    logdir = args.logdir.resolve()
    if not logdir.exists():
        print(f"Logdir existiert noch nicht: {logdir}", file=sys.stderr)
        print("Zuerst Training starten, dann TensorBoard erneut aufrufen.", file=sys.stderr)

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--port",
        str(args.port),
        "--bind_all",
    ]
    print(f"TensorBoard: http://localhost:{args.port}")
    print(f"Logdir: {logdir}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
