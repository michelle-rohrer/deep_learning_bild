#!/bin/bash
#SBATCH --job-name=dlbs-baseline-right
#SBATCH -p performance
#SBATCH --output=logs/slurm-baseline-right-%j.out
#SBATCH --error=logs/slurm-baseline-right-%j.err
#SBATCH --time=23:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
PROJECT="/cluster/group/vised/muscles-seg/code/deep_learning_bild"
cd "$PROJECT"
mkdir -p logs
PY="${PROJECT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

"${PY}" scripts/check_env.py
# Splits sind fest (10 Subjects, rechtes Bein) — kein make_splits nötig
"${PY}" scripts/train.py --config configs/baseline.yaml
"${PY}" scripts/evaluate.py --config configs/baseline.yaml

echo "Baseline-Right fertig."
echo "  Metriken: checkpoints/baseline_right/eval_report.json"
echo "  TensorBoard: python scripts/launch_tensorboard.py --logdir runs/baseline_right"
