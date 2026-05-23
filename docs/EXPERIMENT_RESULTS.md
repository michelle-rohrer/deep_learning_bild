# Experiment-Ergebnisse & Interpretation

**Projekt:** 3D Muskel-Segmentierung (linke Oberschenkelmuskulatur)  
**Status:** Ergebnisse auf **GPU-PC oder Cluster** ausstehend (Laptop-CPU zu langsam für 3D-Baseline).

Nach dem Lauf auf GPU: Zahlen aus `checkpoints/*/cv_report.json` hier eintragen.

---

## Setup (geplant)

| | Laptop | GPU-PC / Cluster |
|---|--------|------------------|
| Config Overfit | `overfit_single_cpu.yaml` (Test) | `overfit_single.yaml` |
| Config Baseline | `baseline_cpu.yaml` (Test) | `baseline.yaml` |
| Patch-Grösse | 64×64×16 | 128×128×32 |
| MC Samples (Eval) | 3–5 | 20 |

**Splits:** `splits/folds.json` — 38 Subjects, 3 Folds (Seed 42), Subject `543` ausgeschlossen.

| Fold | Train | Val |
|------|-------|-----|
| 0 | 25 | 13 |
| 1 | 25 | 13 |
| 2 | 26 | 12 |

---

## 1 · Overfitting-Test

**Ziel (Dokument):** Ein Fall, Macro-Dice links ≥ **0.90** → Datenpipeline + Loss + Modell funktionieren.

| Metrik | Ergebnis | Ziel | ✓/✗ |
|--------|----------|------|-----|
| Subject | 512 | — | |
| Final Macro-Dice (links) | *eintragen* | ≥ 0.90 | |
| Epochen bis Ziel | *eintragen* | — | |

**Interpretation (nach Ausfüllen):**

- **≥ 0.90:** Technische Validierung bestanden; sinnvoll mit 3-Fold-Baseline weiter.
- **< 0.90:** Patch-Grösse, Lernrate, Epochen oder Subject prüfen; TensorBoard `runs/overfit_single/` ansehen.

---

## 2 · Baseline (3-Fold CV)

**Ziel (Dokument):** Mean Macro-Dice links ≥ **0.60** über Validierungs-Folds.

| Fold | Best Val Dice (links) | Checkpoint |
|------|----------------------|------------|
| 0 | *eintragen* | `checkpoints/baseline/fold_0/best.pt` |
| 1 | *eintragen* | `checkpoints/baseline/fold_1/best.pt` |
| 2 | *eintragen* | `checkpoints/baseline/fold_2/best.pt` |
| **Mean ± Std** | *aus cv_report.json* | |

**Interpretation:**

- **Mean ≥ 0.60:** Forschungsfrage 1 (kompakter 3D-Ansatz) erfüllt unter diesen Bedingungen.
- **Hohe Std zwischen Folds:** erwartbar bei n≈12–13 Val-Fällen; bilateral/unvollständige Fälle sind in Splits verteilt.
- **Train-Loss ↓, Val-Dice stagniert:** typisch bei kleinem Datensatz → Kap. 3.4 (Augmentation, Focal Loss, LR-Schedule).

---

## 3 · Evaluation (aggregiert)

Aus `checkpoints/baseline/eval_report.json`:

| Metrik | Wert |
|--------|------|
| Mean Macro-Dice (links) | *eintragen* |
| Std über Folds | *eintragen* |

---

## Gesamt-Fazit (nach GPU-Lauf)

*Kurz zusammenfassen: technische Validierung, Baseline vs. Ziel 0.60, Stabilität über Folds, nächste Schritte (Tuning Kap. 3.4).*

---

## Reproduktion

```bash
# GPU-PC / Cluster
python scripts/check_env.py
bash scripts/run_pipeline.sh all
# oder: sbatch cluster/slurm_overfit.sh && sbatch cluster/slurm_baseline.sh
```

Logs: `runs/`, Reports: `checkpoints/`, Anleitung: `docs/RUN_GPU_CLUSTER.md`
