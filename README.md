# Deep Learning Bild & Signal – Muskel-Segmentierung

Reproduzierbare 3D-Pipeline für die Mini-Challenge (Bayesian U-Net, linke Muskeln, 3-Fold CV).  
Aufbau gemäss Projektdokument bis **Baseline** inkl. Overfitting-Test und **TensorBoard**-Monitoring.

## Prinzipien (Karpathy)

1. **Ein Config-File** pro Experiment (`configs/*.yaml`) – Hyperparameter zentral, nicht im Code verstreut.
2. **Kleine, lesbare Module** – Daten, Modell, Loss, Train-Loop getrennt.
3. **Reproduzierbarkeit** – fester Seed, versionierte `splits/folds.json`, Checkpoints + JSON-Reports.
4. **Erst Overfit, dann Generalisierung** – technischer Sanity-Check vor 3-Fold-Baseline.

## Setup

```bash
cd /Users/michellerohrer/Code/deep_learning_bild
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Daten unter `data/water/` und `data/masks/` (nicht im Git).

### Wo trainieren?

| Rechner | Empfehlung |
|---------|------------|
| **Laptop (nur CPU)** | Zu langsam für echte 3D-Baseline → nur Pipeline testen (`configs/*_cpu.yaml`) |
| **Starker PC mit NVIDIA-GPU** | `configs/baseline.yaml` + `bash scripts/run_pipeline.sh all` |
| **Cluster (SLURM)** | `sbatch cluster/slurm_overfit.sh` dann `slurm_baseline.sh` |

Ausführlich: **[docs/RUN_GPU_CLUSTER.md](docs/RUN_GPU_CLUSTER.md)** · Check: `python scripts/check_env.py`

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

### 3. Baseline-Training (3-Fold CV)

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml --fold 0   # nur ein Fold
```

### 4. Evaluation

```bash
python scripts/evaluate.py --config configs/baseline.yaml
# → checkpoints/baseline/eval_report.json
```

## TensorBoard (Trainings im Browser)

Metriken werden nach `runs/<experiment>/` geschrieben (`train/loss`, `val/macro_dice_left`).

```bash
# in separatem Terminal, während oder nach dem Training:
python scripts/tensorboard.py
# → http://localhost:6006
```

Nur ein Experiment anzeigen:

```bash
python scripts/tensorboard.py --logdir runs/baseline
```

Ohne Logging: `--no-tensorboard` bei `train.py` / `run_overfit.py`.

## Projektstruktur

```
configs/           # Experiment-Configs (Baseline, Overfit)
scripts/           # CLI + tensorboard.py
src/muscle_seg/
  data/            # Inventar, Splits, Patch-Dataset
  models/          # Bayesian 3D U-Net + MC-Inferenz
  losses/          # Dice Loss
  metrics/         # Macro-Dice links (Klassen 1–8)
  train/           # Trainer + TensorBoard-Logging
  eval/            # Checkpoint-Evaluation
runs/              # TensorBoard-Logs
checkpoints/       # Modelle & Reports
```

## Hinweise

- **Modell:** PyTorch (3D U-Net). **Monitoring:** TensorBoard (TensorFlow-Ökosystem, kein WandB).
- **Nur linke Muskeln** (Labels 1–8); Subject `543` ausgeschlossen.
- **Patch-basiertes 3D** wegen Volumengrösse (~704×508×640).

## Referenz

FHNW Mini-Challenge – `doc/Sabina_Michelle_dlbs.docx` (Kap. 3.3 Baseline, 3.5 Evaluation).
