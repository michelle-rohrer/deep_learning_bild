#!/usr/bin/env python3
"""Erzeugt reproduzierbare 3-Fold-Splits (Fall-Ebene)."""

from __future__ import annotations

import argparse
from pathlib import Path

from muscle_seg.data.annotation_audit import audit_cohort
from muscle_seg.data.splits import generate_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Fold CV Splits erzeugen")
    parser.add_argument("--data-dir", type=Path, default=Path("../cleaned_data_2"))
    parser.add_argument("--out", type=Path, default=Path("splits/folds.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    audit = audit_cohort(data_dir)
    if audit["excluded_no_right"]:
        print("Ausgeschlossen (keine rechten Labels in Rechts-Maske):")
        for sid in audit["excluded_no_right"]:
            print(f"  {sid}")
    if audit["partial_right"]:
        print("Partielle rechte Annotation (nur Teilmenge 1–8):")
        for r in audit["per_subject"]:
            if r["status"] == "partial":
                print(f"  {r['subject']}: {', '.join(r['present_names'])}")

    folds = generate_and_save(
        data_dir,
        (root / args.out).resolve(),
        n_splits=args.n_splits,
        seed=args.seed,
    )
    for f in folds:
        print(f"Fold {f['fold']}: train={len(f['train'])}, val={len(f['val'])}")
    print(f"CV-Subjects gesamt: {sum(len(f['train']) + len(f['val']) for f in folds) // 2}")


if __name__ == "__main__":
    main()
