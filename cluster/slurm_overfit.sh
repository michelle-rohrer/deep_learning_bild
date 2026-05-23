#!/bin/bash
#SBATCH --job-name=dlbs-overfit
#SBATCH --output=logs/slurm-overfit-%j.out
#SBATCH --error=logs/slurm-overfit-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Anpassen für euren Cluster (Module, Partition):
# #SBATCH --partition=gpu
# module load python/3.11 cuda/12.1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/check_env.py
python scripts/run_overfit.py --config configs/overfit_single.yaml

echo "Overfit fertig. Logs: runs/overfit_single/"
