#!/usr/bin/env python3
"""Erstellt Abb. A4, A5, A6 für den Bericht.

Speichert nach docs/figures/
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
OUT  = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MUSCLE_NAMES = ["VL_R", "RF_R", "VM_R", "VI_R", "BfSH_R", "BfLH_R", "ST_R", "SM_R"]
MUSCLE_LABELS = ["VL", "RF", "VM", "VI", "BfSH", "BfLH", "ST", "SM"]
COL_HEADERS = ["BG"] + MUSCLE_LABELS  # 9 pred columns

EXPERIMENTS = [
    ("baseline_right", "Baseline"),
    ("hpt_focal",      "hpt_focal"),
    ("hpt_lr_scheduler","hpt_lr"),
    ("hpt_dropout",    "hpt_dropout"),
    ("hpt_augmentation","hpt_aug"),
    ("hpt_combined",   "hpt_combined"),
]

# ── helper ────────────────────────────────────────────────────────────────────

def fold_mean_dice(exp_key: str) -> tuple[float, float]:
    """Returns (mean, std) over folds from eval_report.json."""
    path = CKPT / exp_key / "eval_report.json"
    d = json.loads(path.read_text())
    vals = [f["macro_dice_left"] for f in d["per_fold"]]
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return m, s


def cc_mean(exp_key: str) -> float:
    """Returns mean dice-after-CC averaged over folds."""
    path = CKPT / exp_key / "cc_report_largest.json"
    d = json.loads(path.read_text())
    vals = [f["mean_dice_after"] for f in d["folds"]]
    return sum(vals) / len(vals)


def per_class_dice(exp_key: str) -> dict[str, float]:
    """Returns mean dice per class over all val subjects."""
    path = CKPT / exp_key / "eval_report.json"
    d = json.loads(path.read_text())
    sums: dict[str, list[float]] = {m: [] for m in MUSCLE_NAMES}
    for fold in d["per_fold"]:
        for subj in fold["per_subject"]:
            for m in MUSCLE_NAMES:
                v = subj["per_class"][m]["dice"]
                if isinstance(v, float) and not math.isnan(v):
                    sums[m].append(v)
    return {m: (sum(v) / len(v) if v else 0.0) for m, v in sums.items()}


def confusion_normed(exp_key: str) -> np.ndarray:
    """Returns 8×9 row-normalised confusion matrix (%)."""
    path = CKPT / exp_key / "confusion_matrix.json"
    d = json.loads(path.read_text())
    mat = np.array(d["total"], dtype=float)  # 8×9
    row_sums = mat.sum(axis=1, keepdims=True).clip(1)
    return mat / row_sums * 100


# ── Abb. A4: Mean Dice ± Std vor / nach CC ───────────────────────────────────

def plot_a4():
    labels = [label for _, label in EXPERIMENTS]
    befores, stds, afters = [], [], []
    for key, _ in EXPERIMENTS:
        m, s = fold_mean_dice(key)
        befores.append(m)
        stds.append(s)
        afters.append(cc_mean(key))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - w/2, befores, w, yerr=stds, capsize=4,
                   color="#4C72B0", label="Ohne CC", error_kw={"elinewidth": 1.2})
    bars2 = ax.bar(x + w/2, afters, w,
                   color="#DD8452", label="Mit CC (largest)")

    ax.set_ylabel("Mean Dice (BG ausgeschlossen)", fontsize=11)
    ax.set_title("Abb. A4 – Mean Dice ± Std aller Experimente, vor und nach CC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.55)
    ax.axhline(0.60, color="red", linestyle="--", linewidth=1, label="Ziel RQ1 (0.60)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = OUT / "abb_a4_dice_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Abb. A5: Per-Klassen-Dice: Baseline vs. hpt_focal vs. hpt_combined ───────

def plot_a5():
    experiments = [
        ("baseline_right", "Baseline",     "#4C72B0"),
        ("hpt_focal",      "hpt_focal",    "#DD8452"),
        ("hpt_combined",   "hpt_combined", "#55A868"),
    ]
    all_dices = {key: per_class_dice(key) for key, _, _ in experiments}

    x = np.arange(len(MUSCLE_NAMES))
    n = len(experiments)
    w = 0.25
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    fig, ax = plt.subplots(figsize=(12, 5))

    for (key, label, color), offset in zip(experiments, offsets):
        vals = [all_dices[key][m] for m in MUSCLE_NAMES]
        bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("Mean Dice", fontsize=11)
    ax.set_title("Abb. A5 – Per-Klassen-Dice: Baseline vs. hpt_focal vs. hpt_combined", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MUSCLE_LABELS, fontsize=11)
    ax.set_ylim(0, 0.90)
    ax.legend(fontsize=10)
    ax.axvline(3.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.5, 0.85, "Quadrizeps", ha="center", fontsize=9, color="gray")
    ax.text(5.5, 0.85, "Hamstrings", ha="center", fontsize=9, color="gray")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT / "abb_a5_per_class_dice.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Abb. A6: Drei Confusion-Matrix-Heatmaps nebeneinander ────────────────────

def plot_a6():
    configs = [
        ("baseline_right", "Baseline"),
        ("hpt_focal",      "hpt_focal"),
        ("hpt_combined",   "hpt_combined"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Abb. A6 – Confusion-Matrix (Foreground-Voxel, Zeilen-normiert in %)",
                 fontsize=13, y=1.01)

    for ax, (key, title) in zip(axes, configs):
        mat = confusion_normed(key)  # 8×9
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(9))
        ax.set_xticklabels(COL_HEADERS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(8))
        ax.set_yticklabels(MUSCLE_LABELS, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("GT", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(8):
            for j in range(9):
                val = int(mat[i, j])
                if val > 0:
                    color = "white" if mat[i, j] > 55 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=8, color=color)

        # Highlight diagonal (correct predictions, col offset +1 because col 0 = BG)
        for i in range(8):
            ax.add_patch(plt.Rectangle(
                (i + 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.5
            ))

        plt.colorbar(im, ax=ax, shrink=0.8, label="%")

    fig.tight_layout()
    path = OUT / "abb_a6_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Erstelle Abbildungen...")
    plot_a4()
    plot_a5()
    plot_a6()
    print("Fertig. Alle Abbildungen in:", OUT)
