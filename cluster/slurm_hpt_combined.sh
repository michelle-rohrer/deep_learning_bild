#!/bin/bash
#SBATCH --job-name=dlbs-hpt-combined
#SBATCH -p performance
#SBATCH --output=logs/slurm-hpt-combined-%j.out
#SBATCH --error=logs/slurm-hpt-combined-%j.err
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
"${PY}" scripts/train.py --config configs/hpt_combined.yaml
"${PY}" scripts/evaluate.py --config configs/hpt_combined.yaml

echo "HPT-Combined fertig."
echo "  Metriken: checkpoints/hpt_combined/eval_report.json"
