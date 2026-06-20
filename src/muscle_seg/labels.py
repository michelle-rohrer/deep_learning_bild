"""Label-Definitionen (rechts, Evaluation Klassen 1–8 = remapped rechte Muskeln)."""

from __future__ import annotations

# Original-Masken-IDs rechts (9–16), werden zu Klassen 1–8 umgemappt
LEFT_MUSCLE_LABELS: tuple[int, ...] = tuple(range(1, 9))   # Klassen im Modell-Output
RIGHT_MUSCLE_LABELS: tuple[int, ...] = tuple(range(9, 17)) # Original-IDs in Masken
NUM_CLASSES: int = 9  # 0 = Hintergrund, 1–8 = rechte Muskeln (remapped)

LABEL_NAMES: dict[int, str] = {
    0: "BG",
    1: "VL_R",
    2: "RF_R",
    3: "VM_R",
    4: "VI_R",
    5: "BfSH_R",
    6: "BfLH_R",
    7: "ST_R",
    8: "SM_R",
}

# Kein manueller Ausschluss mehr (cleaned_data_2 hat rechte Labels für alle)
EXCLUDED_SUBJECTS: frozenset[str] = frozenset()


def remap_mask_to_classes(mask) -> "object":
    """
    Mappt rechte Masken-IDs (9–16) auf Klassen 1–8.
    Linke Labels (1–8) und unbekannte IDs werden zu 0 (Hintergrund).
    """
    import numpy as np

    out = np.zeros(mask.shape, dtype=np.int64)
    for cls, lbl in enumerate(RIGHT_MUSCLE_LABELS, start=1):  # 9→1, 10→2, …, 16→8
        out[mask == lbl] = cls
    return out
