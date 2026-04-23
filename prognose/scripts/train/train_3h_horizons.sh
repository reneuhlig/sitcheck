#!/usr/bin/env bash
# Train LGBM quantile models for all 12 fifteen-minute horizons (h15..h180).
# Runs locally — no Docker required. Activate your virtualenv first.
#
# Usage:
#   ./scripts/train/train_3h_horizons.sh
#   ./scripts/train/train_3h_horizons.sh --skip-existing
#   ./scripts/train/train_3h_horizons.sh --dry-run
#   ./scripts/train/train_3h_horizons.sh --horizons 30,45,75
#   ZONE_ID=my-zone ./scripts/train/train_3h_horizons.sh
#
# Extra arguments are forwarded verbatim to train_multihorizon.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORECAST_DIR="${PROGNOSE_DIR}/services/forecast"

ZONE_ID="${ZONE_ID:-default-zone}"
TRAINING_DATA="${PROGNOSE_DIR}/training_data.parquet"
MODEL_DIR="${FORECAST_DIR}/models"
TRAIN_SCRIPT="${FORECAST_DIR}/train_multihorizon.py"

# ── Pick Python interpreter ───────────────────────────────────────────────────

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "[ERROR] No Python interpreter found. Activate your virtualenv." >&2
    exit 1
fi

# ── Preflight checks ──────────────────────────────────────────────────────────

if [ ! -f "${TRAINING_DATA}" ]; then
    echo "[ERROR] training_data.parquet not found at: ${TRAINING_DATA}" >&2
    echo "        Generate it first:" >&2
    echo "          cd ${PROGNOSE_DIR}" >&2
    echo "          python scripts/data/prepare_training_data.py" >&2
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] train_multihorizon.py not found at: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"

# ── Run training ──────────────────────────────────────────────────────────────

echo "[INFO] Python:        ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "[INFO] Training data: ${TRAINING_DATA}"
echo "[INFO] Model dir:     ${MODEL_DIR}"
echo "[INFO] Zone:          ${ZONE_ID}"
[ $# -gt 0 ] && echo "[INFO] Extra args:    $*"
echo ""

"${PYTHON}" "${TRAIN_SCRIPT}" \
    --data "${TRAINING_DATA}" \
    --model-dir "${MODEL_DIR}" \
    --zone-id "${ZONE_ID}" \
    "$@"

echo ""
echo "[INFO] Training complete. Models saved to: ${MODEL_DIR}"
