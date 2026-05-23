#!/bin/bash
#SBATCH --job-name=dlbs-baseline
#SBATCH --output=logs/slurm-baseline-%j.out
#SBATCH --error=logs/slurm-baseline-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Optional: 3 separate Jobs (ein Fold pro Job), schneller bei Job-Limits:
#   python scripts/train.py --config configs/baseline.yaml --fold 0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/check_env.py
python scripts/make_splits.py
python scripts/train.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml

echo "Baseline fertig. Report: checkpoints/baseline/cv_report.json"
