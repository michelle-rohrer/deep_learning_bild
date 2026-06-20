# dlbs Mini-Challenge – 3D-Segmentierung der Oberschenkelmuskulatur

Sabina Grüner & Michelle Rohrer · FHNW · FS 2026

Automatische voxelweise Segmentierung von **8 rechten Oberschenkelmuskeln** (Quadrizeps + Hamstrings) aus wassergewichteten 3D-MRI-Aufnahmen. Methode: Bayesian 3D U-Net mit Monte Carlo Dropout, 3-Fold Cross-Validation, systematische Hyperparameter-Experimente nach Karpathy-Methode.

---

## Ergebnisse (Mean Dice, BG ausgeschlossen, 3-Fold CV)

| Experiment | ohne CC | mit CC |
|---|---|---|
| Baseline | 0.234 ± 0.026 | 0.266 |
| hpt_focal | 0.266 ± 0.031 | 0.294 |
| hpt_lr_scheduler | 0.213 ± 0.010 | 0.254 |
| hpt_dropout p=0.5 | 0.130 ± 0.009 | 0.154 |
| hpt_augmentation | 0.126 ± 0.022 | 0.156 |
| **hpt_combined** | **0.308 ± 0.061** | **0.355** |

CC = Connected-Component-Postprocessing (largest component per class).  
Beste Einzelklassen mit hpt_combined: VL 0.743, VI 0.716, VM 0.415.

---

## Setup

```bash
# Cluster (empfohlen)
cd /cluster/group/vised/muscles-seg/code/deep_learning_bild
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Daten verlinken (einmalig)
bash scripts/link_data_to_cleaned.sh
```

Voraussetzungen prüfen:

```bash
python scripts/check_env.py
```

Daten liegen unter `data/water/` und `data/masks/` — nicht im Git.

---

## Pipeline

### 1. Fold-Splits erzeugen

```bash
python scripts/make_splits.py
# → splits/folds.json  (bereits versioniert)
```

### 2. Overfit-Test (Sanity Check, Ziel: Dice ≥ 0.90)

```bash
sbatch cluster/slurm_overfit.sh
# oder lokal:
python scripts/train.py --config configs/overfit_single.yaml
```

### 3. Training

```bash
# Baseline
sbatch cluster/slurm_baseline.sh

# HPT-Experimente (je ein Parameter vs. Baseline)
sbatch cluster/slurm_hpt_focal.sh
sbatch cluster/slurm_hpt_augmentation.sh
sbatch cluster/slurm_hpt_lr_scheduler.sh
sbatch cluster/slurm_hpt_dropout.sh
sbatch cluster/slurm_hpt_combined.sh    # Focal-Tversky + klass.-bal. Sampling, 200 Ep.
```

### 4. Evaluation

```bash
python scripts/evaluate.py --config configs/<experiment>.yaml
# → checkpoints/<experiment>/eval_report.json
```

### 5. Post-Processing & Analyse

```bash
# Connected-Component-Cleanup (largest mode)
python scripts/eval_cc.py --config configs/<experiment>.yaml --mode largest
# → checkpoints/<experiment>/cc_report_largest.json

# Confusion-Matrix auf Foreground-Voxeln
python scripts/eval_confusion.py --config configs/<experiment>.yaml
# → checkpoints/<experiment>/confusion_matrix.json

# Abbildungen für den Bericht (Abb. A4, A5, A6)
python scripts/plot_figures.py
# → docs/figures/abb_a4_dice_overview.png
# → docs/figures/abb_a5_per_class_dice.png
# → docs/figures/abb_a6_confusion_matrices.png
```

### 6. TensorBoard

```bash
python scripts/launch_tensorboard.py --logdir runs/<experiment>
# → http://localhost:6006
```

---

## Projektstruktur

```
configs/                    # Ein YAML pro Experiment
  baseline.yaml
  overfit_single.yaml
  hpt_focal.yaml
  hpt_augmentation.yaml
  hpt_lr_scheduler.yaml
  hpt_dropout.yaml
  hpt_combined.yaml

scripts/
  train.py                  # Trainings-Entrypoint (--config, --fold)
  evaluate.py               # Sliding-Window Evaluation
  eval_cc.py                # CC-Postprocessing Analyse
  eval_confusion.py         # Confusion-Matrix auf Foreground-Voxeln
  plot_figures.py           # Berichts-Abbildungen (A4, A5, A6)
  make_splits.py            # 3-Fold CV Splits erzeugen
  check_env.py              # GPU / Package Check
  visualize_predictions.py  # Slice-Visualisierung (TensorBoard)
  launch_tensorboard.py

cluster/
  slurm_*.sh                # SLURM-Job-Scripts (performance partition, 23h)

src/muscle_seg/
  config.py                 # TrainConfig (Dataclass, from_yaml)
  labels.py                 # Muskel-Namen, Label-Mapping
  data/
    dataset.py              # MusclePatchDataset, class-balanced Sampling
    volume_io.py            # NIfTI laden, Crop, Normalisierung
    splits.py               # Fold-Zuteilung laden
  models/
    bayesian_unet.py        # 3D U-Net mit MC-Dropout
  losses/
    dice.py                 # MultiClassDiceLoss
    focal.py                # DiceFocalLoss
    tversky.py              # FocalTverskyLoss
  eval/
    volume_predict.py       # Sliding-Window Inferenz
    postprocess.py          # CC-Postprocessing
    run_eval.py             # Evaluation Loop
  train/
    trainer.py              # Trainer (AMP, Checkpoints, TensorBoard)
    tensorboard_logger.py

notebooks/
  EDA.ipynb                 # Exploratory Data Analysis (Notebook-Abgabe)

docs/
  bericht.md                # Schriftlicher Bericht (M4)
  figures/                  # Abbildungen A4–A6 (aus plot_figures.py)
  Aufgabenstellung.pdf
  bewertungsraster.xlsx
  Sabina_Michelle_dlbs.docx

splits/
  folds.json                # Versioniert: reproduzierbare Fold-Zuteilung
```

---

## Konfiguration

Alle Hyperparameter stehen in `configs/*.yaml`. Wichtigste Felder:

```yaml
experiment_name: hpt_combined
loss_name: focal_tversky       # dice | dice_focal | focal_tversky
learning_rate: 0.0001
dropout: 0.3
epochs: 200
class_balanced_sampling: true  # uniform über Klassen 1–8
augmentation: false
eval_bbox_crop: true
```

---

## Daten

- 10 Probanden, rechtes Bein (Labels 9–16 → Modell-Klassen 1–8)
- 8 Zielklassen: VL, RF, VM, VI (Quadrizeps) + BfSH, BfLH, ST, SM (Hamstrings)
- 3-Fold CV auf Probandenebene (kein Slice-Leakage)
- Daten nicht im Repository (`data/` ist gitignored)
