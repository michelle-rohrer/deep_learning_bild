# Training auf GPU-PC oder Cluster

Auf dem **Laptop (nur CPU)** dauert 3D-Training praktisch zu lange.  
Die **offiziellen Configs** (`configs/overfit_single.yaml`, `configs/baseline.yaml`) sind für **CUDA-GPU** gedacht.

| Umgebung | Configs | Erwartung |
|----------|---------|-----------|
| GPU-PC / Cluster | `overfit_single.yaml`, `baseline.yaml` | Stunden (Baseline 3-Fold) |
| Laptop nur CPU | `*_cpu.yaml` | Nur zum Testen der Pipeline |

---

## A) Starker PC (Windows / Linux) mit NVIDIA-GPU

### 1. Repo + Daten

```bash
git clone <dein-repo> deep_learning_bild
cd deep_learning_bild

# Daten vom Laptop/Cluster kopieren (ca. 2.6 GB Water + Masken):
# rsync -av laptop:~/Code/deep_learning_bild/data/ ./data/
```

### 2. PyTorch **mit CUDA** installieren

Wichtig: nicht nur `pip install torch` vom CPU-Wheel – CUDA-Version zum Treiber passend wählen:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Beispiel CUDA 12.1 – siehe https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 3. Check

```bash
python scripts/check_env.py
# CUDA verfügbar: True
# GPU: <Name>
```

### 4. Pipeline

```bash
bash scripts/run_pipeline.sh all
# oder Schrittweise:
bash scripts/run_pipeline.sh overfit
bash scripts/run_pipeline.sh baseline
```

### 5. TensorBoard

```bash
python scripts/tensorboard.py --logdir runs
# Browser: http://localhost:6006
```

---

## B) Cluster (SLURM, z. B. Vised / FHNW)

### 1. Projekt auf den Cluster bringen

```bash
# vom Laptop:
rsync -av --exclude .venv --exclude data \
  ~/Code/deep_learning_bild/ user@cluster:~/deep_learning_bild/

rsync -av ~/Code/deep_learning_bild/data/ user@cluster:~/deep_learning_bild/data/
```

Oder: Git push + auf dem Cluster `git clone` + Daten separat nach `data/`.

### 2. Umgebung auf dem Cluster

```bash
cd ~/deep_learning_bild
module load python/3.11    # je nach Site
module load cuda/12.1      # je nach Site

python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python scripts/check_env.py
```

### 3. Jobs einreichen

```bash
mkdir -p logs
sbatch cluster/slurm_overfit.sh    # zuerst (~ wenige Stunden)
sbatch cluster/slurm_baseline.sh   # danach (~ bis 24 h, 3 Folds)
squeue -u $USER
```

**Fold parallel** (wenn 3 GPUs / 3 Jobs erlaubt):

```bash
for f in 0 1 2; do
  sbatch --job-name=dlbs-f$f --wrap \
    "cd $PWD && source .venv/bin/activate && python scripts/train.py --config configs/baseline.yaml --fold $f"
done
python scripts/evaluate.py --config configs/baseline.yaml
```

### 4. TensorBoard vom Cluster auf den Laptop

```bash
# auf dem Laptop:
ssh -L 6006:localhost:6006 user@cluster
# auf dem Cluster (Login-Node oder Port-Forward nach Compute-Node):
cd ~/deep_learning_bild && source .venv/bin/activate
python scripts/tensorboard.py --logdir runs --port 6006
```

Dann im Laptop-Browser: http://localhost:6006

### 5. Ergebnisse zurückholen

```bash
rsync -av user@cluster:~/deep_learning_bild/checkpoints/ ./checkpoints/
rsync -av user@cluster:~/deep_learning_bild/runs/ ./runs/
```

Trage die Zahlen in `docs/EXPERIMENT_RESULTS.md` ein (Vorlage unten).

---

## C) Ergebnisse dokumentieren

Nach dem Lauf auf GPU/Cluster:

| Datei | Inhalt |
|-------|--------|
| `checkpoints/baseline/cv_report.json` | Mean Dice ± Std über 3 Folds |
| `checkpoints/baseline/eval_report.json` | Evaluation aller Checkpoints |
| `checkpoints/overfit_single/` | Overfit-Run (falls gespeichert) |
| `runs/baseline/fold*/` | TensorBoard-Kurven |

**Interpretation (Kurz):**

- **Overfit Dice ≥ 0.90** → Pipeline + Modell können lernen (technisch ok).
- **Baseline Mean Dice** → Ziel laut Dokument ≥ 0.60 (linke Muskeln); Std über Folds zeigt Stabilität bei kleinem n.
- **Train-Loss sinkt, Val-Dice flach** → Overfitting / zu wenig Daten → später Augmentation & Regularisierung (Kap. 3.4).

---

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `CUDA verfügbar: False` | PyTorch mit CUDA-Wheel neu installieren; Treiber `nvidia-smi` prüfen |
| CUDA OOM | `batch_size: 1` oder `patch_size: [96, 96, 24]` in YAML |
| Job bricht ab | `logs/slurm-*.err` lesen; `--mem` erhöhen |
| Langsam auf CPU | Auf GPU wechseln, nicht `baseline_cpu.yaml` für finale Ergebnisse |
