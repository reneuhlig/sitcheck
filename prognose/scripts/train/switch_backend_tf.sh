#!/usr/bin/env bash
set -euo pipefail

TARGET_BACKEND="${1:-tf_mlp}"
if [[ "${TARGET_BACKEND}" != "tf_mlp" && "${TARGET_BACKEND}" != "baseline" ]]; then
  echo "Usage: $0 [tf_mlp|baseline]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Restart Sitcheck with FORECAST_MODEL_BACKEND=${TARGET_BACKEND}"
SITCHECK_FORECAST_MODEL_BACKEND="${TARGET_BACKEND}" "${ROOT_DIR}/sitcheckctl.sh" restart
