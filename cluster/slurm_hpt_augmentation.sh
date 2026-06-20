#!/bin/bash
#SBATCH --job-name=dlbs-hpt-aug
#SBATCH -p performance
#SBATCH --output=logs/slurm-hpt-aug-%j.out
#SBATCH --error=logs/slurm-hpt-aug-%j.err
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
"${PY}" scripts/train.py --config configs/hpt_augmentation.yaml
"${PY}" scripts/evaluate.py --config configs/hpt_augmentation.yaml

echo "HPT Augmentation fertig."
