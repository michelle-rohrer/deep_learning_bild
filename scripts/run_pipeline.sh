#!/usr/bin/env bash
# Volle Pipeline auf GPU-PC oder Cluster-Login-Node (interaktiv).
# Nutzung: bash scripts/run_pipeline.sh [overfit|baseline|all]

set -euo pipefail
cd "$(dirname "$0")/.."

STEP="${1:-all}"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== Umgebung ==="
python scripts/check_env.py || true
echo

run_splits() {
  echo "=== 1/4 Splits ==="
  python scripts/make_splits.py
}

run_overfit() {
  echo "=== 2/4 Overfitting-Test ==="
  python scripts/run_overfit.py --config configs/overfit_single.yaml
}

run_baseline() {
  echo "=== 3/4 Baseline (3-Fold) ==="
  python scripts/train.py --config configs/baseline.yaml
}

run_eval() {
  echo "=== 4/4 Evaluation ==="
  python scripts/evaluate.py --config configs/baseline.yaml
}

case "$STEP" in
  splits)   run_splits ;;
  overfit)  run_splits; run_overfit ;;
  baseline) run_splits; run_baseline ;;
  eval)     run_eval ;;
  all)
    run_splits
    run_overfit
    run_baseline
    run_eval
    ;;
  *)
    echo "Unbekannter Schritt: $STEP"
    echo "Nutze: splits | overfit | baseline | eval | all"
    exit 1
    ;;
esac

echo
echo "Fertig. TensorBoard: python scripts/tensorboard.py --logdir runs"
echo "Reports: checkpoints/*/cv_report.json, docs/EXPERIMENT_RESULTS.md"
