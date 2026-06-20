"""Sliding-Window-Inferenz auf ganzen 3D-Volumina (Bayesian 3D U-Net).

Zwei Modi:
  - uniform  : jeder Patch-Pixel zählt gleich (schnell, Nahtartefakte möglich)
  - shrink   : Overlap-Tile-Strategie (Ronneberger et al. 2015) — das Modell
               sieht den ganzen Patch als Kontext, aber nur der zentrale Bereich
               ohne Zero-Padding-Rand fliesst in die Prediction ein.
               → keine Nahtartefakte, mehr Patches nötig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from tqdm import tqdm

from muscle_seg.data.patching import extract_patch
from muscle_seg.labels import NUM_CLASSES

if TYPE_CHECKING:
    from muscle_seg.models.bayesian_unet import BayesianUNet3D


# ---------------------------------------------------------------------------
# Interne Helfer
# ---------------------------------------------------------------------------

def _default_stride(patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(1, s // 2) for s in patch_size)


def _shrink_stride(
    patch_size: tuple[int, int, int],
    shrink_margin: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Stride = Patch-Mitte ohne Rand → jeder Voxel genau einmal abgedeckt."""
    return tuple(max(1, s - 2 * m) for s, m in zip(patch_size, shrink_margin))


def _patch_centers(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> list[np.ndarray]:
    centers: list[np.ndarray] = []
    for d in range(0, shape[2], stride[2]):
        for h in range(0, shape[0], stride[0]):
            for w in range(0, shape[1], stride[1]):
                cz = min(d + patch_size[2] // 2, shape[2] - 1)
                ch = min(h + patch_size[0] // 2, shape[0] - 1)
                cw = min(w + patch_size[1] // 2, shape[1] - 1)
                centers.append(np.array([ch, cw, cz], dtype=int))
    return centers


def _run_patch(
    model: "BayesianUNet3D",
    patch_img: np.ndarray,
    device: torch.device,
    use_mc: bool,
    mc_samples: int,
) -> np.ndarray:
    """Einzelnen Patch durch das Modell — gibt (C, H, W, D) Softmax-Probs zurück."""
    x = torch.from_numpy(patch_img[None, None]).to(device)
    if use_mc:
        model.train()
        probs = torch.stack(
            [torch.softmax(model(x), dim=1) for _ in range(mc_samples)], dim=0
        ).mean(dim=0)
    else:
        model.eval()
        probs = torch.softmax(model(x), dim=1)
    return probs.squeeze(0).cpu().numpy()  # (C, H, W, D)


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_volume_sliding_window(
    model: "BayesianUNet3D",
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    device: torch.device,
    *,
    stride: tuple[int, int, int] | None = None,
    use_mc: bool = False,
    mc_samples: int = 5,
    show_progress: bool = False,
    mode: Literal["uniform", "shrink"] = "shrink",
    shrink_margin: tuple[int, int, int] | None = None,
    return_probs: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Sliding-Window-Inferenz über das gesamte 3D-Volumen.

    Args:
        mode:
            ``"uniform"``  — gleichmässige Gewichtung (alt, schnell).
            ``"shrink"``   — Overlap-Tile: Modell sieht ganzen Patch als Kontext,
                             nur der zentrale Bereich ohne Zero-Padding-Rand wird
                             genutzt. Keine Nahtartefakte.
        shrink_margin:
            Randbreite in Voxeln die verworfen wird (pro Seite, pro Achse).
            Default: Viertel der Patch-Grösse je Achse, z.B. (32, 32, 8) für
            Patch (128, 128, 32).
    """
    shape = image.shape

    if mode == "shrink":
        if shrink_margin is None:
            shrink_margin = tuple(max(1, s // 4) for s in patch_size)
        effective_stride = _shrink_stride(patch_size, shrink_margin)
    else:
        shrink_margin = (0, 0, 0)
        effective_stride = stride or _default_stride(patch_size)

    centers = _patch_centers(shape, patch_size, effective_stride)

    prob_sum = np.zeros((NUM_CLASSES, *shape), dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)

    iterator = tqdm(centers, desc="patches", leave=False) if show_progress else centers

    margin = np.array(shrink_margin, dtype=int)
    patch_arr = np.array(patch_size, dtype=int)
    vol_shape = np.array(shape, dtype=int)

    for center in iterator:
        patch_img, _ = extract_patch(image, None, center, patch_size)
        probs_np = _run_patch(model, patch_img, device, use_mc, mc_samples)

        half = patch_arr // 2
        vol_starts = center.astype(int) - half
        vol_ends = vol_starts + patch_arr

        if mode == "shrink":
            # Nur den zentralen Bereich (ohne Rand) ins Volume schreiben
            inner_vol_start = vol_starts + margin
            inner_vol_end = vol_ends - margin
            inner_patch_start = margin

            src_s = np.maximum(inner_vol_start, 0)
            src_e = np.minimum(inner_vol_end, vol_shape)
            dst_s = inner_patch_start + (src_s - inner_vol_start)
            dst_e = dst_s + (src_e - src_s)
        else:
            src_s = np.maximum(vol_starts, 0)
            src_e = np.minimum(vol_ends, vol_shape)
            dst_s = src_s - vol_starts
            dst_e = dst_s + (src_e - src_s)

        sl_vol = tuple(slice(int(a), int(b)) for a, b in zip(src_s, src_e))
        sl_patch = tuple(slice(int(a), int(b)) for a, b in zip(dst_s, dst_e))

        if all(sl.start < sl.stop for sl in sl_vol):
            prob_sum[(slice(None), *sl_vol)] += probs_np[:, *sl_patch]
            weight[sl_vol] += 1.0

    weight = np.maximum(weight, 1e-8)
    mean_probs = (prob_sum / weight[None, ...]).astype(np.float32)
    pred = mean_probs.argmax(axis=0).astype(np.int64)
    if return_probs:
        return pred, mean_probs
    return pred
