"""TensorBoard-Logging (TensorFlow-Ökosystem, ersetzt WandB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from muscle_seg.config import TrainConfig

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Schreibt Skalare und Hyperparameter nach runs/<experiment>/."""

    def __init__(self, cfg: TrainConfig, run_name: str, hparams: dict[str, Any] | None = None):
        self.enabled = cfg.use_tensorboard
        self.writer: SummaryWriter | None = None
        if not self.enabled:
            return

        from torch.utils.tensorboard import SummaryWriter

        log_root = (cfg.project_root / cfg.tensorboard_log_dir / cfg.experiment_name).resolve()
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        if hparams:
            for key, val in hparams.items():
                self.writer.add_text(f"config/{key}", str(val))

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(key, val, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
