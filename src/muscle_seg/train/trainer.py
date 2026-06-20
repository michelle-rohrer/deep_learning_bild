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
from muscle_seg.losses.focal import DiceFocalLoss
from muscle_seg.losses.tversky import FocalTverskyLoss
from muscle_seg.labels import LABEL_NAMES, LEFT_MUSCLE_LABELS
from muscle_seg.metrics.dice import aggregate_macro_dice, macro_dice_left_muscles
from muscle_seg.models.bayesian_unet import BayesianUNet3D, mc_predict
from muscle_seg.eval.tensorboard_viz import build_patch_val_mosaic
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
        if cfg.loss_name == "dice_focal":
            self.criterion = DiceFocalLoss(
                num_classes=NUM_CLASSES,
                gamma=cfg.focal_gamma,
                alpha=cfg.focal_alpha,
            )
        elif cfg.loss_name == "focal_tversky":
            self.criterion = FocalTverskyLoss(
                num_classes=NUM_CLASSES,
                alpha=cfg.tversky_alpha,
                beta=cfg.tversky_beta,
                gamma=cfg.tversky_gamma,
            )
        else:
            self.criterion = MultiClassDiceLoss(num_classes=NUM_CLASSES)
        self.history: list[dict] = []
        self._amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp)

    @torch.no_grad()
    def _log_val_patch_mosaic(
        self,
        tb: TBLogger,
        model: BayesianUNet3D,
        val_loader: DataLoader,
        epoch: int,
        *,
        use_mc: bool,
    ) -> None:
        """Erstes Val-Batch → TensorBoard Images (Water | GT | Pred | Uncertainty)."""
        if not self.cfg.train_log_val_images:
            return
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return
        x = batch["image"].to(self.device)
        y = batch["mask"].to(self.device)
        if use_mc:
            pred, variance = mc_predict(model, x, n_samples=min(3, self.cfg.mc_samples))
            unc = variance[0].detach().cpu().numpy()
        else:
            model.eval()
            pred = model(x).argmax(dim=1)
            unc = None
        water = x[0, 0].detach().cpu().numpy()
        gt = y[0].detach().cpu().numpy()
        pr = pred[0].detach().cpu().numpy()

        # Patch hat (H,W,D) als letzte Dim — sicherstellen
        if water.ndim == 2:
            water = water[:, :, np.newaxis]
            gt = gt[:, :, np.newaxis]
            pr = pr[:, :, np.newaxis]
            if unc is not None:
                unc = unc[:, :, np.newaxis]

        mosaic = build_patch_val_mosaic(water, gt, pr, uncertainty_3d=unc)
        tb.log_image("val/patch_water_gt_pred_unc", mosaic, epoch)

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
            augmentation=cfg.augmentation,
            class_balanced_sampling=cfg.class_balanced_sampling,
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
        if cfg.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 1e-2
            )
        elif cfg.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10,
                min_lr=cfg.learning_rate * 1e-2,
            )
        else:
            scheduler = None

        tb = TBLogger(
            cfg,
            run_name,
            hparams=_config_dict(cfg, fold=fold_idx, train_subjects=train_subj, val_subjects=val_subj),
        )

        best_val_dice = -1.0
        best_path = ckpt_dir / "best.pt"
        global_step = 0
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            model.train()
            train_losses: list[float] = []
            grad_norms: list[float] = []
            steps = 0
            pbar = tqdm(train_loader, desc=f"{cfg.experiment_name}_{run_name} ep{epoch}", leave=False)
            for batch in pbar:
                x = batch["image"].to(self.device)
                y = batch["mask"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp):
                    logits = model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()

                # Gradient-Norm (nach unscale, vor step)
                self.scaler.unscale_(optimizer)
                total_norm = float(
                    torch.norm(
                        torch.stack([
                            p.grad.detach().norm()
                            for p in model.parameters()
                            if p.grad is not None
                        ])
                    ).item()
                )
                grad_norms.append(total_norm)

                self.scaler.step(optimizer)
                self.scaler.update()
                loss_val = float(loss.item())
                train_losses.append(loss_val)
                tb.log_scalars({"train/loss_batch": loss_val}, step=global_step)
                global_step += 1
                pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")
                steps += 1
                if cfg.max_train_steps_per_epoch and steps >= cfg.max_train_steps_per_epoch:
                    break

            # Cosine Scheduler: step nach jeder Epoche
            if scheduler is not None and cfg.lr_scheduler == "cosine":
                scheduler.step()

            # Epoch-Skalare
            epoch_log: dict[str, float] = {
                "train/loss": float(np.mean(train_losses)),
                "train/grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            tb.log_scalars(epoch_log, step=epoch)

            # Weight-Histogramme alle 10 Epochen
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    tb.log_histogram(f"weights/{name}", param, epoch)
                    if param.grad is not None:
                        tb.log_histogram(f"grads/{name}", param.grad, epoch)

            if epoch % cfg.eval_every_epochs == 0:
                val_metrics = self.evaluate(model, val_loader, use_mc=cfg.eval_use_mc)
                val_dice = val_metrics["macro_dice_left"]

                val_log: dict[str, float] = {
                    "val/macro_dice": val_dice,
                    "val/macro_dice_std": val_metrics.get("macro_dice_left_std", 0.0),
                }
                # Per-Klasse Dice
                for c, name in LABEL_NAMES.items():
                    if c == 0:
                        continue
                    key = f"val/dice_{name}"
                    val_log[key] = val_metrics.get("per_class_dice", {}).get(c, float("nan"))

                self.history.append({"fold": fold_idx, "epoch": epoch, **epoch_log, **val_log})
                tb.log_scalars(val_log, step=epoch)
                self._log_val_patch_mosaic(
                    tb, model, val_loader, epoch, use_mc=cfg.eval_use_mc
                )
                print(
                    f"[{cfg.experiment_name}/{run_name}] epoch {epoch}: "
                    f"loss={epoch_log['train/loss']:.4f} val_dice={val_dice:.4f} "
                    f"grad_norm={epoch_log['train/grad_norm']:.3f}",
                    flush=True,
                )

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    patience_counter = 0
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
                else:
                    patience_counter += cfg.eval_every_epochs

                # Plateau Scheduler: step nach Val
                if scheduler is not None and cfg.lr_scheduler == "plateau":
                    scheduler.step(val_dice)

                # Early Stopping
                if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
                    print(
                        f"[{cfg.experiment_name}/{run_name}] Early stopping ep {epoch} "
                        f"(keine Verbesserung seit {patience_counter} Epochen)",
                        flush=True,
                    )
                    break

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
    def evaluate(
        self,
        model: BayesianUNet3D,
        loader: DataLoader,
        *,
        use_mc: bool | None = None,
    ) -> dict:
        """Validierung: MC-Dropout (Baseline) oder deterministisch (Overfit-Sanity-Check)."""
        cfg = self.cfg
        if use_mc is None:
            use_mc = True
        dice_scores: list[float] = []
        per_class_accum: dict[int, list[float]] = {c: [] for c in LEFT_MUSCLE_LABELS}

        if use_mc:
            for batch in loader:
                x = batch["image"].to(self.device)
                y = batch["mask"].to(self.device)
                pred, _variance = mc_predict(model, x, n_samples=cfg.mc_samples)
                for i in range(x.shape[0]):
                    d, pc = macro_dice_left_muscles(pred[i].cpu(), y[i].cpu())
                    dice_scores.append(d)
                    for c, v in pc.items():
                        if v == v:  # not nan
                            per_class_accum[c].append(v)
        else:
            model.eval()
            for batch in loader:
                x = batch["image"].to(self.device)
                y = batch["mask"].to(self.device)
                pred = model(x).argmax(dim=1)
                for i in range(x.shape[0]):
                    d, pc = macro_dice_left_muscles(pred[i].cpu(), y[i].cpu())
                    dice_scores.append(d)
                    for c, v in pc.items():
                        if v == v:
                            per_class_accum[c].append(v)

        mean_d, std_d = aggregate_macro_dice(dice_scores)
        per_class_mean = {
            c: float(np.mean(vals)) if vals else float("nan")
            for c, vals in per_class_accum.items()
        }
        return {
            "macro_dice_left": mean_d,
            "macro_dice_left_std": std_d,
            "per_class_dice": per_class_mean,
            "n_batches": len(loader),
        }

    def run(self) -> dict:
        folds = load_folds(self.cfg.splits_path)
        if self.cfg.subjects:
            cfg = self.cfg
            run_name = "overfit"
            ckpt_dir = cfg.checkpoint_dir / cfg.experiment_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_path = ckpt_dir / "best.pt"

            tb = TBLogger(cfg, run_name, hparams=_config_dict(cfg))
            train_ds, eval_ds = build_datasets(
                cfg.data_dir,
                cfg.subjects,
                cfg.subjects,
                patch_size=cfg.patch_size,
                patches_per_volume=cfg.patches_per_volume,
                percentiles=cfg.intensity_clip_percentile,
                seed=cfg.seed,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                collate_fn=_collate,
                drop_last=True,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=_collate,
            )
            use_mc_eval = cfg.overfit_eval_mc

            model = self._build_model()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            target = cfg.overfit_target_dice
            dice = 0.0
            best_dice = -1.0
            stop_epoch = cfg.epochs

            print(
                f"[overfit] {len(train_loader)} train batches/epoch, "
                f"{len(eval_loader)} eval batches (mc_eval={use_mc_eval})",
                flush=True,
            )

            for epoch in range(1, cfg.epochs + 1):
                train_ds.set_epoch(epoch)
                eval_ds.set_epoch(epoch)
                model.train()
                losses = []
                steps = 0
                for batch in train_loader:
                    x = batch["image"].to(self.device)
                    y = batch["mask"].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp):
                        loss = self.criterion(model(x), y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    losses.append(float(loss.item()))
                    steps += 1
                    if cfg.max_train_steps_per_epoch and steps >= cfg.max_train_steps_per_epoch:
                        break

                metrics = self.evaluate(model, eval_loader, use_mc=use_mc_eval)
                dice = metrics["macro_dice_left"]
                tb.log_scalars(
                    {"train/loss": float(np.mean(losses)), "val/macro_dice_left": dice},
                    step=epoch,
                )
                self._log_val_patch_mosaic(
                    tb, model, eval_loader, epoch, use_mc=use_mc_eval
                )
                print(
                    f"[overfit] epoch {epoch}: loss={float(np.mean(losses)):.4f} dice={dice:.4f}",
                    flush=True,
                )

                if dice > best_dice:
                    best_dice = dice
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg.__dict__,
                            "epoch": epoch,
                            "val_macro_dice_left": dice,
                        },
                        best_path,
                    )

                if dice >= target:
                    stop_epoch = epoch
                    break

            tb.close()
            return {
                "mode": "overfit",
                "final_macro_dice_left": dice,
                "best_macro_dice_left": best_dice,
                "target": target,
                "epochs_run": stop_epoch,
                "checkpoint": str(best_path),
                "tensorboard_run": str(cfg.tensorboard_log_dir / cfg.experiment_name / run_name),
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
