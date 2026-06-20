#!/bin/bash
#SBATCH --job-name=dlbs-pipeline
#SBATCH -p performance
#SBATCH --output=logs/slurm-pipeline-%j.out
#SBATCH --error=logs/slurm-pipeline-%j.err
#SBATCH --time=14:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Nach DAW/cleaned_data-Update: Overfit → Baseline-Train → Eval (ein Job)

set -euo pipefail
PROJECT="/cluster/group/vised/muscles-seg/code/deep_learning_bild"
cd "$PROJECT"
mkdir -p logs
PY="${PROJECT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

echo "=== Umgebung ==="
"${PY}" scripts/check_env.py

echo "=== Splits (Annotation-Audit) ==="
"${PY}" scripts/make_splits.py

echo "=== 1/3 Overfit (Subject 512) ==="
"${PY}" scripts/run_overfit.py --config configs/overfit_single.yaml

echo "=== 2/3 Baseline 3-Fold CV ==="
"${PY}" scripts/train.py --config configs/baseline.yaml

echo "=== 3/3 Evaluation + TensorBoard-Bilder ==="
"${PY}" scripts/evaluate.py --config configs/baseline.yaml

echo "Pipeline fertig."
echo "  Overfit:  checkpoints/overfit_single/best.pt"
echo "  Baseline: checkpoints/baseline/eval_report.json"
echo "  TB:       python scripts/launch_tensorboard.py --logdir runs/baseline"
