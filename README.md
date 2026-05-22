# Deep Learning Bild & Signal – Muskel-Segmentierung

Reproduzierbare 3D-Pipeline für die Mini-Challenge (Bayesian U-Net, linke Muskeln, 3-Fold CV).  
Aufbau gemäss Projektdokument bis **Baseline** inkl. Overfitting-Test und WandB-Monitoring.

## Prinzipien (Karpathy)

1. **Ein Config-File** pro Experiment (`configs/*.yaml`) – später Hyperparameter-Sweeps über WandB.
2. **Kleine, lesbare Module** – Daten, Modell, Loss, Train-Loop getrennt.
3. **Reproduzierbarkeit** – fester Seed, versionierte `splits/folds.json`, Checkpoints + JSON-Reports.
4. **Erst Overfit, dann Generalisierung** – technischer Sanity-Check vor 3-Fold-Baseline.

## Setup

```bash
cd /Users/michellerohrer/Code/deep_learning_bild
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# WandB (Trainings in der Cloud verfolgen)
wandb login
```

Daten unter `data/water/` und `data/masks/` (nicht im Git).

## Pipeline (Reihenfolge)

### 1. Fold-Splits erzeugen (Fall-Ebene, kein Slice-Leakage)

```bash
python scripts/make_splits.py
# → splits/folds.json
```

### 2. Overfitting-Test (ein Fall, Ziel Dice ≥ 0.90)

```bash
python scripts/run_overfit.py --config configs/overfit_single.yaml
```

Dashboard: [wandb.ai](https://wandb.ai) → Projekt `dlbs-muscle-seg`, Tag `overfit-test`.

### 3. Baseline-Training (3-Fold CV)

```bash
# alle Folds
python scripts/train.py --config configs/baseline.yaml

# ein Fold (Debug)
python scripts/train.py --config configs/baseline.yaml --fold 0
```

Baseline laut Dokument: Dropout 0.3, Dice Loss, Adam 1e-4, MC Samples 20, **keine** Augmentation.

### 4. Evaluation

```bash
python scripts/evaluate.py --config configs/baseline.yaml
# → checkpoints/baseline/eval_report.json
```

## Projektstruktur

```
configs/           # Experiment-Configs (Baseline, Overfit, später Tuning)
scripts/           # CLI-Einstiegspunkte
src/muscle_seg/
  data/            # Inventar, Splits, Patch-Dataset
  models/          # Bayesian 3D U-Net + MC-Inferenz
  losses/          # Dice Loss
  metrics/         # Macro-Dice links (Klassen 1–8)
  train/           # Trainer + WandB-Logging
  eval/            # Checkpoint-Evaluation
splits/            # folds.json
checkpoints/       # Modelle & Reports
notebooks/         # EDA
```

## WandB

- Projekt: `dlbs-muscle-seg`
- Geloggte Metriken: `train_loss`, `val_macro_dice_left`
- Ohne Login: `--no-wandb`

Für Hyperparameter-Tuning später z. B.:

```bash
wandb sweep configs/sweep_baseline.yaml  # (optional, noch nicht angelegt)
```

## Hinweise

- **Nur linke Muskeln** (Labels 1–8); Subject `543` ausgeschlossen (unvollständige linke Quadrizeps-Labels).
- **Patch-basiertes 3D** wegen Volumengrösse (~704×508×640); kompakter 3D-Ansatz wie im Dokument.
- Rechte Labels werden in der Maske auf Hintergrund gemappt und nicht in den Dice einbezogen.

## Referenz

FHNW Mini-Challenge – Dokument `doc/Sabina_Michelle_dlbs.docx` (Kap. 3.3 Baseline, 3.5 Evaluation).
