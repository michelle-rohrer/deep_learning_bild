"""Trainings-Loop – lesbar, ein Einstiegspunkt (Karpathy-Stil)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from muscle_seg.config import TrainConfig
from muscle_seg.data.dataset import build_datasets
from muscle_seg.data.splits import load_folds
from muscle_seg.labels import NUM_CLASSES
from muscle_seg.losses.dice import MultiClassDiceLoss
from muscle_seg.metrics.dice import aggregate_macro_dice, macro_dice_left_muscles
from muscle_seg.models.bayesian_unet import BayesianUNet3D, mc_predict
from muscle_seg.train.tensorboard_logger import TBLogger


def _config_dict(cfg: TrainConfig, **extra) -> dict:
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
    d.update(extra)
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "subject": [b["subject"] for b in batch],
    }


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg.resolve_paths()
        set_seed(cfg.seed)
        self.device = _device(cfg.device)
        self.criterion = MultiClassDiceLoss(num_classes=NUM_CLASSES)
        self.history: list[dict] = []

    def _build_model(self) -> BayesianUNet3D:
        return BayesianUNet3D(
            in_channels=1,
            num_classes=NUM_CLASSES,
            base_channels=self.cfg.base_channels,
            depth=self.cfg.depth,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def train_one_fold(self, fold_idx: int, train_subj: list[str], val_subj: list[str]) -> dict:
        cfg = self.cfg
        run_name = f"fold{fold_idx}"
        ckpt_dir = cfg.checkpoint_dir / cfg.experiment_name / f"fold_{fold_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_ds, val_ds = build_datasets(
            cfg.data_dir,
            train_subj,
            val_subj,
            patch_size=cfg.patch_size,
            patches_per_volume=cfg.patches_per_volume,
            percentiles=cfg.intensity_clip_percentile,
            seed=cfg.seed + fold_idx,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
        )

        model = self._build_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        tb = TBLogger(
            cfg,
            run_name,
            hparams=_config_dict(cfg, fold=fold_idx, train_subjects=train_subj, val_subjects=val_subj),
        )

        best_val_dice = -1.0
        best_path = ckpt_dir / "best.pt"

        for epoch in range(1, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            model.train()
            train_losses: list[float] = []
            steps = 0
            pbar = tqdm(train_loader, desc=f"{cfg.experiment_name}_{run_name} ep{epoch}", leave=False)
            for batch in pbar:
                x = batch["image"].to(self.device)
                y = batch["mask"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
                pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
                steps += 1
                if cfg.max_train_steps_per_epoch and steps >= cfg.max_train_steps_per_epoch:
                    break

            if epoch % cfg.eval_every_epochs == 0:
                val_metrics = self.evaluate(model, val_loader)
                val_dice = val_metrics["macro_dice_left"]
                log = {
                    "train/loss": float(np.mean(train_losses)),
                    "val/macro_dice_left": val_dice,
                }
                self.history.append({"fold": fold_idx, "epoch": epoch, **log})
                tb.log_scalars(log, step=epoch)
                print(
                    f"[{cfg.experiment_name}/{run_name}] epoch {epoch}: "
                    f"loss={log['train/loss']:.4f} val_dice={val_dice:.4f}",
                    flush=True,
                )

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg.__dict__,
                            "fold": fold_idx,
                            "epoch": epoch,
                            "val_macro_dice_left": val_dice,
                        },
                        best_path,
                    )

        tb.close()

        summary = {
            "fold": fold_idx,
            "best_val_macro_dice_left": best_val_dice,
            "checkpoint": str(best_path),
            "tensorboard_run": str(cfg.tensorboard_log_dir / cfg.experiment_name / run_name),
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    @torch.no_grad()
    def evaluate(self, model: BayesianUNet3D, loader: DataLoader) -> dict:
        """Validierung mit MC Dropout (N Samples, Mittel-Vorhersage)."""
        cfg = self.cfg
        dice_scores: list[float] = []

        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["mask"].to(self.device)
            pred, _variance = mc_predict(model, x, n_samples=cfg.mc_samples)
            for i in range(x.shape[0]):
                d, _ = macro_dice_left_muscles(pred[i].cpu(), y[i].cpu())
                dice_scores.append(d)

        mean_d, std_d = aggregate_macro_dice(dice_scores)
        return {
            "macro_dice_left": mean_d,
            "macro_dice_left_std": std_d,
            "n_batches": len(loader),
        }

    def run(self) -> dict:
        folds = load_folds(self.cfg.splits_path)
        if self.cfg.subjects:
            run_name = "overfit"
            tb = TBLogger(self.cfg, run_name, hparams=_config_dict(self.cfg))
            train_ds, _ = build_datasets(
                self.cfg.data_dir,
                self.cfg.subjects,
                self.cfg.subjects,
                patch_size=self.cfg.patch_size,
                patches_per_volume=self.cfg.patches_per_volume,
                percentiles=self.cfg.intensity_clip_percentile,
                seed=self.cfg.seed,
            )
            loader = DataLoader(
                train_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=_collate,
                drop_last=True,
            )
            model = self._build_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
            target = self.cfg.overfit_target_dice
            dice = 0.0

            for epoch in range(1, self.cfg.epochs + 1):
                train_ds.set_epoch(epoch)
                model.train()
                losses = []
                steps = 0
                for batch in loader:
                    x = batch["image"].to(self.device)
                    y = batch["mask"].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    steps += 1
                    if self.cfg.max_train_steps_per_epoch and steps >= self.cfg.max_train_steps_per_epoch:
                        break

                metrics = self.evaluate(model, loader)
                dice = metrics["macro_dice_left"]
                tb.log_scalars(
                    {"train/loss": float(np.mean(losses)), "val/macro_dice_left": dice},
                    step=epoch,
                )
                print(
                    f"[overfit] epoch {epoch}: loss={float(np.mean(losses)):.4f} dice={dice:.4f}",
                    flush=True,
                )
                if dice >= target:
                    break

            tb.close()
            return {
                "mode": "overfit",
                "final_macro_dice_left": dice,
                "target": target,
                "tensorboard_run": str(
                    self.cfg.tensorboard_log_dir / self.cfg.experiment_name / run_name
                ),
            }

        fold_indices = (
            [self.cfg.fold] if self.cfg.fold is not None else list(range(len(folds)))
        )
        all_summaries = []
        for fi in fold_indices:
            fold = folds[fi]
            summary = self.train_one_fold(fi, fold["train"], fold["val"])
            all_summaries.append(summary)

        mean_best, std_best = aggregate_macro_dice(
            [s["best_val_macro_dice_left"] for s in all_summaries]
        )
        report = {
            "experiment": self.cfg.experiment_name,
            "folds": all_summaries,
            "mean_best_val_macro_dice_left": mean_best,
            "std_best_val_macro_dice_left": std_best,
            "tensorboard_logdir": str(
                self.cfg.project_root / self.cfg.tensorboard_log_dir / self.cfg.experiment_name
            ),
        }
        out = self.cfg.checkpoint_dir / self.cfg.experiment_name / "cv_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        return report
