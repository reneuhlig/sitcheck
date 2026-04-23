#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="$(cd "${PROGNOSE_DIR}/.." && pwd)"
SITCHECKCTL="${ROOT_DIR}/sitcheckctl.sh"

FORECAST_URL="${FORECAST_URL:-http://127.0.0.1:8001}"
ZONE_ID="${ZONE_ID:-default-zone}"
HISTORY_HOURS="${HISTORY_HOURS:-8760}"
RETRAIN_FULL="${RETRAIN_FULL:-true}"

if [ ! -x "${SITCHECKCTL}" ]; then
    echo "missing orchestrator: ${SITCHECKCTL}" >&2
    exit 1
fi

MAINTENANCE_ACTIVE=0
RELOCK_DONE=0

relock_on_exit() {
    if [ "${MAINTENANCE_ACTIVE}" = "1" ] && [ "${RELOCK_DONE}" = "0" ]; then
        echo "[WARN] Restoring locked training mode..."
        SITCHECK_FORECAST_MODEL_BACKEND=tf_mlp \
        SITCHECK_FORECAST_TRAINING_MODE=locked \
        SITCHECK_FORECAST_TRAINER_ENABLED=0 \
        "${SITCHECKCTL}" restart >/dev/null 2>&1 || true
    fi
}
trap relock_on_exit EXIT

request_json() {
    local method="$1"
    local url="$2"
    local payload="${3:-}"
    local tmp_body
    tmp_body="$(mktemp)"
    local http_code

    if [ -n "${payload}" ]; then
        http_code="$(curl -sS -o "${tmp_body}" -w '%{http_code}' -X "${method}" "${url}" -H 'content-type: application/json' -d "${payload}")"
    else
        http_code="$(curl -sS -o "${tmp_body}" -w '%{http_code}' -X "${method}" "${url}")"
    fi

    if [ "${http_code}" -ge 400 ]; then
        echo "${http_code}" >&2
        cat "${tmp_body}" >&2
        rm -f "${tmp_body}"
        return 1
    fi

    cat "${tmp_body}"
    rm -f "${tmp_body}"
    return 0
}

echo "[INFO] Verify forecast health and backend..."
health_json="$(request_json GET "${FORECAST_URL}/health")"
python3 - "$health_json" <<'PY'
from __future__ import annotations

import json
import sys

payload = json.loads(sys.argv[1])
backend = str(payload.get("backend", ""))
if backend != "tf_mlp":
    raise SystemExit(f"forecast backend must be tf_mlp, got {backend!r}")
print(f"[OK] Forecast backend={backend}, training_mode={payload.get('training_mode')}")
PY

echo "[INFO] Restart sitcheck in maintenance mode (training enabled for controlled run)..."
SITCHECK_FORECAST_MODEL_BACKEND=tf_mlp \
SITCHECK_FORECAST_TRAINING_MODE=maintenance \
SITCHECK_FORECAST_TRAINER_ENABLED=0 \
"${SITCHECKCTL}" restart
MAINTENANCE_ACTIVE=1

maint_health_json="$(request_json GET "${FORECAST_URL}/health")"
python3 - "$maint_health_json" <<'PY'
from __future__ import annotations

import json
import sys

payload = json.loads(sys.argv[1])
backend = str(payload.get("backend", ""))
mode = str(payload.get("training_mode", ""))
if backend != "tf_mlp":
    raise SystemExit(f"backend changed unexpectedly: {backend!r}")
if mode != "maintenance":
    raise SystemExit(f"training mode is not maintenance: {mode!r}")
print("[OK] Maintenance mode active.")
PY

EVALUATE_PAYLOAD="$(python3 - <<PY
import json
print(json.dumps({
    "zone_id": "${ZONE_ID}",
    "horizons": [60],
    "history_hours": int(${HISTORY_HOURS}),
    "include_lecture_impact": True,
    "save_report": True,
}))
PY
)"

FALLBACK_PAYLOAD="$(python3 - <<PY
import json
print(json.dumps({
    "zone_id": "${ZONE_ID}",
    "horizons": [60],
    "history_hours": int(${HISTORY_HOURS}),
    "folds": 3,
    "train_days": 14,
    "val_days": 3,
    "test_days": 3,
    "include_lecture_impact": True,
    "save_report": True,
}))
PY
)"

echo "[INFO] Run scientific evaluate for H60..."
eval_file="$(mktemp)"
eval_code="$(curl -sS -o "${eval_file}" -w '%{http_code}' -X POST "${FORECAST_URL}/v1/train/evaluate" \
    -H 'content-type: application/json' -d "${EVALUATE_PAYLOAD}")"
if [ "${eval_code}" -ge 400 ]; then
    echo "[WARN] primary evaluate failed (HTTP ${eval_code}), checking for insufficient history..."
    if ! python3 - "${eval_file}" <<'PY'
from __future__ import annotations

import json
import sys

try:
    payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)
detail = str(payload.get("detail", "")).lower()
ok = "insufficient_history" in detail or "insufficient history" in detail
raise SystemExit(0 if ok else 1)
PY
    then
        cat "${eval_file}" >&2
        rm -f "${eval_file}"
        exit 1
    fi
    echo "[INFO] Re-run evaluate with reduced split configuration..."
    fallback_code="$(curl -sS -o "${eval_file}" -w '%{http_code}' -X POST "${FORECAST_URL}/v1/train/evaluate" \
        -H 'content-type: application/json' -d "${FALLBACK_PAYLOAD}")"
    if [ "${fallback_code}" -ge 400 ]; then
        cat "${eval_file}" >&2
        rm -f "${eval_file}"
        exit 1
    fi
fi

echo "[INFO] Evaluate result:"
python3 -m json.tool < "${eval_file}"

run_id="$(python3 - "${eval_file}" <<'PY'
from __future__ import annotations

import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
decision = payload.get("decision", {}) if isinstance(payload, dict) else {}
scientific_pass = bool(decision.get("scientific_pass", False))
champion = str(decision.get("champion_model", ""))
run_id = str(payload.get("run_id", "")).strip()
if not run_id:
    raise SystemExit("missing run_id in evaluate response")
if not scientific_pass:
    raise SystemExit(f"scientific gate failed for run {run_id}")
if champion != "tf_mlp":
    raise SystemExit(f"champion must be tf_mlp, got {champion!r}")
print(run_id)
PY
)"

PROMOTE_PAYLOAD="$(python3 - <<PY
import json
print(json.dumps({
    "zone_id": "${ZONE_ID}",
    "run_id": "${run_id}",
    "horizons": [60],
    "full_retrain": ${RETRAIN_FULL},
}))
PY
)"

echo "[INFO] Promote evaluated run ${run_id} (H60 only)..."
promote_json="$(request_json POST "${FORECAST_URL}/v1/train/promote" "${PROMOTE_PAYLOAD}")"
python3 -m json.tool <<<"${promote_json}"

echo "[INFO] Lock training mode again..."
SITCHECK_FORECAST_MODEL_BACKEND=tf_mlp \
SITCHECK_FORECAST_TRAINING_MODE=locked \
SITCHECK_FORECAST_TRAINER_ENABLED=0 \
"${SITCHECKCTL}" restart
RELOCK_DONE=1
MAINTENANCE_ACTIVE=0

final_health_json="$(request_json GET "${FORECAST_URL}/health")"
python3 - "$final_health_json" "${run_id}" <<'PY'
from __future__ import annotations

import json
import sys

health = json.loads(sys.argv[1])
run_id = sys.argv[2]
print(f"[OK] H60 retrain flow completed. run_id={run_id}")
print(f"[OK] forecast backend={health.get('backend')} training_mode={health.get('training_mode')}")
PY

rm -f "${eval_file}"
