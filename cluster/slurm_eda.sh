#!/bin/bash
#SBATCH --job-name=dlbs-eda
#SBATCH -p performance
#SBATCH --output=logs/slurm-eda-%j.out
#SBATCH --error=logs/slurm-eda-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
PROJECT="/cluster/group/vised/muscles-seg/code/deep_learning_bild"
cd "$PROJECT"
mkdir -p logs results
PY="${PROJECT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "=== EDA (nach DAW) ==="
"${PY}" scripts/run_eda.py

echo "EDA fertig."
echo "  Notebook: notebooks/EDA_executed.ipynb"
echo "  Plots:    results/eda_*.png"
