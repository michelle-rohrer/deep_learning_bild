#!/usr/bin/env bash
# Verknüpft <repo>/data → ../cleaned_data (gemeinsame DAW-Daten, kein Duplikat).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLEANED="$(cd "${REPO}/../cleaned_data" && pwd)"
cd "$REPO"
if [[ -L data ]]; then
  echo "data ist bereits ein Symlink: $(readlink -f data)"
  exit 0
fi
if [[ -d data ]] || [[ -e data ]]; then
  echo "Entferne vorhandenes data/ (ersetze durch Symlink nach ${CLEANED})"
  rm -rf data
fi
ln -s ../cleaned_data data
echo "OK: ${REPO}/data -> $(readlink -f data)"
