#!/usr/bin/env bash
# Train LGBM quantile models for all 12 fifteen-minute horizons (h15..h180).
# Run from anywhere — the script locates itself and the prognose project root.
#
# Usage:
#   ./scripts/train/train_3h_horizons.sh
#   ./scripts/train/train_3h_horizons.sh --skip-existing
#   ./scripts/train/train_3h_horizons.sh --dry-run
#   ./scripts/train/train_3h_horizons.sh --horizons 30,45,75
#   ZONE_ID=my-zone ./scripts/train/train_3h_horizons.sh
#
# All extra arguments are forwarded verbatim to train_multihorizon.py.
# Models are written to the forecast_models Docker volume at /models/<zone>/<horizon>/.
#
# After training, restart the forecast container to pick up the new models:
#   docker compose restart forecast

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ZONE_ID="${ZONE_ID:-default-zone}"
TRAINING_DATA="${PROGNOSE_DIR}/training_data.parquet"
CONTAINER_DATA_PATH="/app/training_data.parquet"
TRAIN_SCRIPT="/app/train_multihorizon.py"

# ── Preflight checks ──────────────────────────────────────────────────────────

if [ ! -f "${TRAINING_DATA}" ]; then
    echo "[ERROR] training_data.parquet not found at: ${TRAINING_DATA}" >&2
    echo "        Generate it first:" >&2
    echo "          cd ${PROGNOSE_DIR}" >&2
    echo "          python scripts/data/prepare_training_data.py" >&2
    exit 1
fi

cd "${PROGNOSE_DIR}"

echo "[INFO] Resolving forecast container..."
CONTAINER_ID="$(docker compose ps -q forecast 2>/dev/null || true)"
if [ -z "${CONTAINER_ID}" ]; then
    echo "[ERROR] The 'forecast' service is not running." >&2
    echo "        Start it with: docker compose up -d forecast" >&2
    exit 1
fi
echo "[INFO] Container ID: ${CONTAINER_ID}"

# ── Copy training data into container ─────────────────────────────────────────

echo "[INFO] Copying training_data.parquet into container..."
docker cp "${TRAINING_DATA}" "${CONTAINER_ID}:${CONTAINER_DATA_PATH}"
echo "[INFO] Copy complete."

# ── Run training ──────────────────────────────────────────────────────────────

echo ""
echo "[INFO] Starting multi-horizon training for zone '${ZONE_ID}'..."
[ $# -gt 0 ] && echo "[INFO] Extra args: $*"
echo ""

docker compose exec -T forecast python "${TRAIN_SCRIPT}" \
    --data "${CONTAINER_DATA_PATH}" \
    --model-dir /models \
    --zone-id "${ZONE_ID}" \
    "$@"

echo ""
echo "[INFO] Training complete."
echo "[INFO] To activate new models: docker compose restart forecast"
