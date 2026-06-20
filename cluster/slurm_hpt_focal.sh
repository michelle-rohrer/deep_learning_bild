#!/bin/bash
#SBATCH --job-name=dlbs-hpt-focal
#SBATCH -p performance
#SBATCH --output=logs/slurm-hpt-focal-%j.out
#SBATCH --error=logs/slurm-hpt-focal-%j.err
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
"${PY}" scripts/train.py --config configs/hpt_focal.yaml
"${PY}" scripts/evaluate.py --config configs/hpt_focal.yaml

echo "HPT Focal fertig."
echo "  Metriken: checkpoints/hpt_focal/eval_report.json"
echo "  TensorBoard: python scripts/launch_tensorboard.py --logdir runs/hpt_focal"
