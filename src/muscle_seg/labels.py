"""Label-Definitionen (links, Evaluation nur Klassen 1–8)."""

from __future__ import annotations

# Original-Masken-IDs (links)
LEFT_MUSCLE_LABELS: tuple[int, ...] = tuple(range(1, 9))
NUM_CLASSES: int = 9  # 0 = Hintergrund, 1–8 = linke Muskeln

LABEL_NAMES: dict[int, str] = {
    0: "BG",
    1: "VL_L",
    2: "RF_L",
    3: "VM_L",
    4: "VI_L",
    5: "BfSH_L",
    6: "BfLH_L",
    7: "ST_L",
    8: "SM_L",
}

# Subjects ohne vollständige linke Annotation (EDA)
EXCLUDED_SUBJECTS: frozenset[str] = frozenset({"543"})


def remap_mask_to_classes(mask) -> "object":
    """
    Mappt Roh-Maske auf Klassen 0–8.
    Rechte Labels (>=9) und unbekannte IDs werden zu 0 (Hintergrund).
    """
    import numpy as np

    out = np.zeros(mask.shape, dtype=np.int64)
    for lbl in LEFT_MUSCLE_LABELS:
        out[mask == lbl] = lbl
    return out
