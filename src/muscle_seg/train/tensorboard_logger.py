"""TensorBoard-Logging (TensorFlow-Ökosystem, ersetzt WandB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from muscle_seg.config import TrainConfig

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


def _summary_writer():
    """Direkt aus writer importieren (robuster als torch.utils.tensorboard auf manchen Cluster-Knoten)."""
    from torch.utils.tensorboard.writer import SummaryWriter

    return SummaryWriter


class TBLogger:
    """Schreibt Skalare, Bilder und Histogramme nach runs/<experiment>/."""

    def __init__(self, cfg: TrainConfig, run_name: str, hparams: dict[str, Any] | None = None):
        self.enabled = cfg.use_tensorboard
        self.writer: SummaryWriter | None = None
        if not self.enabled:
            return

        log_root = (cfg.project_root / cfg.tensorboard_log_dir / cfg.experiment_name).resolve()
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = _summary_writer()(log_dir=str(log_dir))

        if hparams:
            for key, val in hparams.items():
                self.writer.add_text(f"config/{key}", str(val))

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(key, val, step)

    def log_image(self, tag: str, image_hwc: np.ndarray, step: int) -> None:
        """RGB-Mosaik (H, W, 3) float32 [0,1] direkt via PyTorch SummaryWriter."""
        if self.writer is None:
            return
        img = np.asarray(image_hwc, dtype=np.float32).clip(0.0, 1.0)
        # SummaryWriter erwartet (C, H, W)
        self.writer.add_image(tag, img.transpose(2, 0, 1), global_step=step, dataformats="CHW")

    def log_image_tf(
        self,
        tag: str,
        image_hwc: "object",
        step: int,
    ) -> None:
        """RGB-Mosaik (H,W,3) float32 [0,1] via TensorFlow tf.summary.image."""
        if self.writer is None:
            return
        import tensorflow as tf

        img = np.asarray(image_hwc, dtype=np.float32)
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        log_dir = Path(self.writer.log_dir)
        tf_writer = tf.summary.create_file_writer(str(log_dir / "train_images"))
        with tf_writer.as_default():
            tf.summary.image(tag, img, step=step, max_outputs=1)
        tf_writer.flush()

    def log_histogram(self, tag: str, values: "object", step: int) -> None:
        """Gewichts- oder Gradienten-Histogramm."""
        if self.writer is None:
            return
        import torch

        v = values
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32).ravel()
        if v.size == 0:
            return
        self.writer.add_histogram(tag, v, global_step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
