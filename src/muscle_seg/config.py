"""Konfiguration laden (YAML) – Karpathy: Hyperparameter aus dem Code, nicht verstreut."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Alle Trainings-Parameter an einem Ort (später: Tuning via TensorBoard HParams)."""

    # Pfade
    project_root: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    splits_path: Path = field(default_factory=lambda: Path("splits/folds.json"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Experiment
    experiment_name: str = "baseline"
    seed: int = 42
    device: str = "cuda"
    fold: int | None = None  # None = alle Folds

    # Daten
    patch_size: tuple[int, int, int] = (128, 128, 32)
    patches_per_volume: int = 4
    intensity_clip_percentile: tuple[float, float] = (0.5, 99.5)
    num_workers: int = 0

    # Modell (Bayesian 3D U-Net)
    base_channels: int = 16
    depth: int = 4
    dropout: float = 0.3

    # Training
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_train_steps_per_epoch: int | None = None  # Debug / Overfit-Test

    # Loss / Optim
    loss_name: str = "dice"  # baseline: ungewichteter Dice

    # Evaluation / MC Dropout
    mc_samples: int = 20
    eval_every_epochs: int = 1

    # Overfitting-Validierung (ein Fall)
    overfit_subject: str = "512"
    overfit_target_dice: float = 0.90

    # Augmentation (Baseline: aus)
    augmentation: bool = False

    # TensorBoard (TensorFlow-Ökosystem)
    use_tensorboard: bool = True
    tensorboard_log_dir: Path = field(default_factory=lambda: Path("runs"))

    # Subset
    subjects: list[str] | None = None  # z.B. nur ein Fall für Overfit

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            raw = {**raw, **overrides}
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        raw = {k: v for k, v in raw.items() if k in valid}
        return cls(**_coerce_fields(raw))

    def resolve_paths(self, root: Path | None = None) -> TrainConfig:
        root = (root or Path.cwd()).resolve()
        self.project_root = root
        self.data_dir = (root / self.data_dir).resolve()
        self.splits_path = (root / self.splits_path).resolve()
        self.checkpoint_dir = (root / self.checkpoint_dir).resolve()
        self.tensorboard_log_dir = (root / self.tensorboard_log_dir).resolve()
        return self


def _coerce_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """YAML-Typen in Python-Typen (Tupel, Pfade)."""
    out = dict(raw)
    for key in ("patch_size", "intensity_clip_percentile"):
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    for key in ("project_root", "data_dir", "splits_path", "checkpoint_dir", "tensorboard_log_dir"):
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    return out
