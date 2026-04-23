#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORECAST_DIR="${PROGNOSE_DIR}/services/forecast"

ZONE_ID="${ZONE_ID:-default-zone}"
TRAINING_DATA="${PROGNOSE_DIR}/training_data.parquet"
MODEL_DIR="${FORECAST_DIR}/models"
TRAIN_SCRIPT="${FORECAST_DIR}/train_multihorizon.py"

if command -v python3 &>/dev/null; then PYTHON=python3; else PYTHON=python; fi

if [ ! -f "${TRAINING_DATA}" ]; then
    echo "[ERROR] training_data.parquet not found at: ${TRAINING_DATA}" >&2; exit 1
fi

mkdir -p "${MODEL_DIR}"

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
