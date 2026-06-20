#!/bin/bash
#SBATCH --job-name=dlbs-overfit
#SBATCH -p performance
#SBATCH --output=logs/slurm-overfit-%j.out
#SBATCH --error=logs/slurm-overfit-%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
PROJECT="/cluster/group/vised/muscles-seg/code/deep_learning_bild"
cd "$PROJECT"
mkdir -p logs
PY="${PROJECT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

"${PY}" scripts/check_env.py
"${PY}" scripts/run_overfit.py --config configs/overfit_single.yaml

echo "Overfit fertig. Logs: runs/overfit_single/"
