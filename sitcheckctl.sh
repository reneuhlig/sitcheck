#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="${SCRIPT_DIR}/prognose"
BILDAUSWERTUNG_CTL="${SCRIPT_DIR}/bildauswertung/start_system.sh"
DASHBOARD_CTL="${SCRIPT_DIR}/bildauswertung/realtime/start_dashboard.sh"
PORTAL_CTL="${SCRIPT_DIR}/website-dashboard/portal/start_portal.sh"
RUNTIME_ROOT="${SCRIPT_DIR}/website-dashboard/runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
SITCHECKCTL_LOG="${LOG_DIR}/sitcheckctl.log"
PROGNOSE_LOCAL_DB_PATH="${SITCHECK_PROGNOSE_DB_PATH:-${PROGNOSE_DIR}/runtime_local.db}"
PROGNOSE_DATABASE_URL="${SITCHECK_PROGNOSE_DATABASE_URL:-sqlite:///${PROGNOSE_LOCAL_DB_PATH}}"
PROGNOSE_VENV="${SITCHECK_PROGNOSE_VENV:-${PROGNOSE_DIR}/.venv}"
PROGNOSE_PY="${PROGNOSE_VENV}/bin/python"
FORECAST_MODEL_BACKEND="${SITCHECK_FORECAST_MODEL_BACKEND:-tf_mlp}"
TF_MIN_TRAIN_POINTS="${SITCHECK_TF_MIN_TRAIN_POINTS:-1000}"
FORECAST_TRAINING_MODE="${SITCHECK_FORECAST_TRAINING_MODE:-locked}"
FORECAST_SNAPSHOT_HORIZONS="${SITCHECK_FORECAST_SNAPSHOT_HORIZONS:-60}"
FORECAST_TRAINER_ENABLED_RAW="${SITCHECK_FORECAST_TRAINER_ENABLED:-0}"
if [[ "${FORECAST_TRAINER_ENABLED_RAW,,}" =~ ^(1|true|yes|on)$ ]]; then
    FORECAST_TRAINER_ENABLED=1
else
    FORECAST_TRAINER_ENABLED=0
fi

API_HEALTH_URL="${SITCHECK_API_HEALTH_URL:-http://127.0.0.1:8000/health}"
API_LEGACY_HEALTH_URL="${SITCHECK_API_LEGACY_HEALTH_URL:-http://127.0.0.1:5000/health}"
CALENDAR_INGEST_HEALTH_URL="${SITCHECK_CALENDAR_INGEST_HEALTH_URL:-http://127.0.0.1:8010/health}"
MCP_HEALTH_URL="${SITCHECK_MCP_HEALTH_URL:-http://127.0.0.1:8081/health}"
LECTURE_INGEST_HEALTH_URL="${SITCHECK_LECTURE_INGEST_HEALTH_URL:-http://127.0.0.1:8012/health}"
FORECAST_TRAINER_HEALTH_URL="${SITCHECK_FORECAST_TRAINER_HEALTH_URL:-http://127.0.0.1:8013/health}"
FORECAST_TRAINER_STATUS_URL="${SITCHECK_FORECAST_TRAINER_STATUS_URL:-http://127.0.0.1:8013/status}"
REALTIME_HEALTH_URL="${SITCHECK_REALTIME_HEALTH_URL:-http://127.0.0.1:8080/health}"
PORTAL_HEALTH_URL="${SITCHECK_PORTAL_HEALTH_URL:-http://127.0.0.1:8090/health}"
STREAMLIT_HEALTH_URL="${SITCHECK_STREAMLIT_HEALTH_URL:-http://127.0.0.1:8501/_stcore/health}"
REALTIME_STATE_URL="${SITCHECK_REALTIME_STATE_URL:-http://127.0.0.1:8080/api/state}"

ORCH_MODE="${SITCHECK_ORCH_MODE:-local}" # local | docker
START_STANDALONE_TRACKING="${SITCHECK_START_STANDALONE_TRACKING:-0}"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SITCHECKCTL_LOG}"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log "[ERROR] Benötigter Befehl fehlt: ${cmd}"
        exit 1
    fi
}

http_ok() {
    local url="$1"
    curl -fsS --max-time 3 "$url" >/dev/null 2>&1
}

wait_for_http() {
    local url="$1"
    local label="$2"
    local timeout_seconds="$3"
    local optional="$4"
    local deadline=$((SECONDS + timeout_seconds))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if http_ok "$url"; then
            log "[OK] ${label} erreichbar: ${url}"
            return 0
        fi
        sleep 1
    done

    if [ "$optional" = "1" ]; then
        log "[WARN] ${label} nicht erreichbar (optional): ${url}"
        return 0
    fi
    log "[ERROR] ${label} Health fehlgeschlagen: ${url}"
    return 1
}

port_in_use() {
    local port="$1"
    ss -ltn "( sport = :${port} )" | grep -q ":${port}"
}

pid_alive() {
    local pid="$1"
    [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1
}

service_pid_file() {
    local name="$1"
    echo "${LOG_DIR}/prognose-${name}.pid"
}

service_log_file() {
    local name="$1"
    echo "${LOG_DIR}/prognose-${name}.log"
}

start_local_service() {
    local name="$1"
    local port="$2"
    local workdir="$3"
    local command="$4"
    local pid_file
    local log_file
    pid_file="$(service_pid_file "$name")"
    log_file="$(service_log_file "$name")"

    if [ -f "${pid_file}" ]; then
        local pid
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if pid_alive "${pid}"; then
            log "[INFO] ${name} läuft bereits (PID: ${pid}, Port: ${port})"
            return 0
        fi
        rm -f "${pid_file}"
    fi

    if port_in_use "${port}"; then
        log "[WARN] Port ${port} bereits belegt; ${name} wird nicht neu gestartet (externer Prozess)."
        return 0
    fi

    log "[INFO] Starte ${name} auf Port ${port} (local/no-docker)."
    rm -f "${pid_file}"
    setsid -f bash -lc "cd '${workdir}' && echo \$\$ > '${pid_file}' && exec ${command}" >>"${log_file}" 2>&1 < /dev/null

    local pid=""
    for _ in $(seq 1 20); do
        if [ -f "${pid_file}" ]; then
            pid="$(cat "${pid_file}" 2>/dev/null || true)"
            break
        fi
        sleep 0.1
    done

    if [ -n "${pid}" ] && pid_alive "${pid}"; then
        log "[OK] ${name} gestartet (PID: ${pid})"
        return 0
    fi

    if [ -z "${pid}" ] && port_in_use "${port}"; then
        log "[OK] ${name} läuft (Port ${port} aktiv), aber kein PID-File geschrieben."
        return 0
    fi

    log "[ERROR] ${name} konnte nicht gestartet werden. Siehe Log: ${log_file}"
    return 1
}

stop_local_service() {
    local name="$1"
    local pid_file
    pid_file="$(service_pid_file "$name")"
    if [ ! -f "${pid_file}" ]; then
        log "[INFO] ${name}: kein PID-File vorhanden."
        return 0
    fi

    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if pid_alive "${pid}"; then
        kill "${pid}" >/dev/null 2>&1 || true
        sleep 0.5
        if pid_alive "${pid}"; then
            kill -9 "${pid}" >/dev/null 2>&1 || true
        fi
        log "[OK] ${name} gestoppt (PID: ${pid})"
    else
        log "[INFO] ${name}: PID ${pid} war nicht aktiv."
    fi
    rm -f "${pid_file}"
}

start_prognose_local_stack() {
    require_cmd curl
    require_cmd ss
    if [ ! -x "${PROGNOSE_PY}" ]; then
        log "[ERROR] Prognose-VENV unvollständig: ${PROGNOSE_VENV}"
        log "[ERROR] Erwartet Python: ${PROGNOSE_PY}"
        exit 1
    fi

    mkdir -p "$(dirname "${PROGNOSE_LOCAL_DB_PATH}")"
    log "[INFO] Prognose local DB URL: ${PROGNOSE_DATABASE_URL}"
    log "[INFO] Forecast backend: ${FORECAST_MODEL_BACKEND}"
    log "[INFO] Forecast training mode: ${FORECAST_TRAINING_MODE}"

    start_local_service "api-gateway" 8000 "${PROGNOSE_DIR}/apps/api-gateway" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' DEFAULT_ZONE_ID='default-zone' DEFAULT_ZONE_CAPACITY='100' FORECAST_SERVICE_URL='http://127.0.0.1:8001' XAI_SERVICE_URL='http://127.0.0.1:8002' RECOMMENDATIONS_SERVICE_URL='http://127.0.0.1:8003' SCHEDULER_SERVICE_URL='http://127.0.0.1:8011' INTERNAL_API_TOKEN='change_me_internal' OLLAMA_ENABLED='true' OLLAMA_BASE_URL='http://127.0.0.1:11434' OLLAMA_MODEL='gemma3:4b' EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS='180' EXPLAINABILITY_FALLBACK_ENABLED='false' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8000"
    wait_for_http "${API_HEALTH_URL}" "api-gateway" 120 0

    # Legacy API alias for Cloudflare ingress compatibility.
    start_local_service "api-gateway-legacy" 5000 "${PROGNOSE_DIR}/apps/api-gateway" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' DEFAULT_ZONE_ID='default-zone' DEFAULT_ZONE_CAPACITY='100' FORECAST_SERVICE_URL='http://127.0.0.1:8001' XAI_SERVICE_URL='http://127.0.0.1:8002' RECOMMENDATIONS_SERVICE_URL='http://127.0.0.1:8003' SCHEDULER_SERVICE_URL='http://127.0.0.1:8011' INTERNAL_API_TOKEN='change_me_internal' OLLAMA_ENABLED='true' OLLAMA_BASE_URL='http://127.0.0.1:11434' OLLAMA_MODEL='gemma3:4b' EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS='180' EXPLAINABILITY_FALLBACK_ENABLED='false' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 5000"
    wait_for_http "${API_LEGACY_HEALTH_URL}" "api-gateway legacy (:5000)" 120 1

    start_local_service "forecast" 8001 "${PROGNOSE_DIR}/services/forecast" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' FORECAST_MODEL_BACKEND='${FORECAST_MODEL_BACKEND}' FORECAST_TRAINING_MODE='${FORECAST_TRAINING_MODE}' TF_MIN_TRAIN_POINTS='${TF_MIN_TRAIN_POINTS}' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8001"
    wait_for_http "http://127.0.0.1:8001/health" "forecast" 120 0

    start_local_service "xai" 8002 "${PROGNOSE_DIR}/services/xai" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' FORECAST_SERVICE_URL='http://127.0.0.1:8001' XAI_SHAP_ENABLED='false' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8002"
    wait_for_http "http://127.0.0.1:8002/health" "xai" 120 0

    start_local_service "recommendations" 8003 "${PROGNOSE_DIR}/services/recommendations" \
      "env '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8003"
    wait_for_http "http://127.0.0.1:8003/health" "recommendations" 120 0

    start_local_service "lecture-ingest" 8012 "${PROGNOSE_DIR}/services/lecture-ingest" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' LECTURE_ZONE_ID='default-zone' LECTURE_SITE_CODE='MA' LECTURE_API_BASE_URL='https://api.dhbw.app' LECTURE_BACKFILL_ENABLED='false' LECTURE_REFRESH_HISTORY_DAYS='7' LECTURE_REFRESH_FUTURE_DAYS='14' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8012"
    wait_for_http "${LECTURE_INGEST_HEALTH_URL}" "lecture-ingest" 120 0

    start_local_service "calendar-ingest" 8010 "${PROGNOSE_DIR}/services/calendar-ingest" \
      "env DATABASE_URL='${PROGNOSE_DATABASE_URL}' DEFAULT_ZONE_ID='default-zone' DEFAULT_ZONE_CAPACITY='100' CALENDAR_INGEST_PORT='8010' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8010"
    wait_for_http "${CALENDAR_INGEST_HEALTH_URL}" "calendar-ingest" 120 1

    start_local_service "mcp-sitcheck" 8081 "${PROGNOSE_DIR}/apps/mcp-sitcheck" \
      "env API_BASE_URL='http://127.0.0.1:8000' OLLAMA_ENABLED='false' OLLAMA_BASE_URL='http://127.0.0.1:11434' OLLAMA_MODEL='gemma3:4b' MCP_HEALTH_PORT='8081' sh -c \"tail -f /dev/null | node server.mjs\""
    wait_for_http "${MCP_HEALTH_URL}" "mcp-sitcheck" 120 1

    start_local_service "forecast-scheduler" 8011 "${PROGNOSE_DIR}/services/forecast-scheduler" \
      "env API_BASE_URL='http://127.0.0.1:8000' INTERNAL_API_TOKEN='change_me_internal' FORECAST_SNAPSHOT_INTERVAL_SECONDS='300' FORECAST_SNAPSHOT_HORIZONS='${FORECAST_SNAPSHOT_HORIZONS}' SCHEDULER_PORT='8011' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8011"
    wait_for_http "http://127.0.0.1:8011/health" "forecast-scheduler" 120 1

    start_local_service "dashboard" 8501 "${PROGNOSE_DIR}/apps/dashboard" \
      "env API_BASE_URL='http://127.0.0.1:8000' DEFAULT_ZONE_ID='default-zone' OLLAMA_ENABLED='true' OLLAMA_BASE_URL='http://127.0.0.1:11434' OLLAMA_MODEL='gemma3:4b' ASSISTANT_LOCAL_TEMPLATE_FALLBACK='false' DASHBOARD_API_TIMEOUT_SECONDS='240' '${PROGNOSE_PY}' -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"
    wait_for_http "${STREAMLIT_HEALTH_URL}" "streamlit dashboard" 180 0

    if [ "${FORECAST_TRAINER_ENABLED}" = "1" ]; then
        start_local_service "forecast-trainer" 8013 "${PROGNOSE_DIR}/services/forecast-trainer" \
          "env FORECAST_TRAINER_API_BASE_URL='http://127.0.0.1:8001' FORECAST_TRAINER_ZONE_ID='default-zone' FORECAST_TRAINER_ENABLED='true' '${PROGNOSE_PY}' -m uvicorn main:app --host 0.0.0.0 --port 8013"
        wait_for_http "${FORECAST_TRAINER_HEALTH_URL}" "forecast-trainer" 120 0
    else
        log "[INFO] forecast-trainer deaktiviert (SITCHECK_FORECAST_TRAINER_ENABLED=${FORECAST_TRAINER_ENABLED_RAW})."
    fi
}

stop_prognose_local_stack() {
    stop_local_service "forecast-trainer"
    stop_local_service "dashboard"
    stop_local_service "forecast-scheduler"
    stop_local_service "mcp-sitcheck"
    stop_local_service "calendar-ingest"
    stop_local_service "lecture-ingest"
    stop_local_service "api-gateway-legacy"
    stop_local_service "api-gateway"
    stop_local_service "recommendations"
    stop_local_service "xai"
    stop_local_service "forecast"
}

start_prognose_stack() {
    if [ "${ORCH_MODE}" = "docker" ]; then
        require_cmd docker
        log "[INFO] Starte Prognose per Docker Compose."
        (
            cd "${PROGNOSE_DIR}" || exit 1
            docker compose up -d
        ) >>"${SITCHECKCTL_LOG}" 2>&1
        wait_for_http "${API_HEALTH_URL}" "api-gateway" 240 0
        wait_for_http "${STREAMLIT_HEALTH_URL}" "streamlit dashboard" 240 1
        return 0
    fi

    log "[INFO] Starte Prognose lokal ohne Docker."
    start_prognose_local_stack
}

stop_prognose_stack() {
    if [ "${ORCH_MODE}" = "docker" ]; then
        log "[INFO] Stoppe Prognose Docker Compose."
        (
            cd "${PROGNOSE_DIR}" || exit 1
            docker compose stop
        ) >>"${SITCHECKCTL_LOG}" 2>&1 || true
        return 0
    fi

    log "[INFO] Stoppe Prognose local/no-docker Stack."
    stop_prognose_local_stack
}

start_realtime_stack() {
    if [ "${START_STANDALONE_TRACKING}" = "1" ]; then
        log "[INFO] Starte zusätzlich standalone Tracking."
        "${BILDAUSWERTUNG_CTL}" start >>"${SITCHECKCTL_LOG}" 2>&1 || true
    else
        log "[INFO] Standalone Tracking deaktiviert (SITCHECK_START_STANDALONE_TRACKING=0)."
    fi

    "${DASHBOARD_CTL}" start >>"${SITCHECKCTL_LOG}" 2>&1
    wait_for_http "${REALTIME_HEALTH_URL}" "realtime dashboard" 120 0
}

stop_realtime_stack() {
    "${DASHBOARD_CTL}" stop >>"${SITCHECKCTL_LOG}" 2>&1 || true
    if [ "${START_STANDALONE_TRACKING}" = "1" ]; then
        "${BILDAUSWERTUNG_CTL}" stop >>"${SITCHECKCTL_LOG}" 2>&1 || true
    fi
}

start_portal_stack() {
    "${PORTAL_CTL}" start >>"${SITCHECKCTL_LOG}" 2>&1
    wait_for_http "${PORTAL_HEALTH_URL}" "portal hauptseite" 120 0
}

stop_portal_stack() {
    "${PORTAL_CTL}" stop >>"${SITCHECKCTL_LOG}" 2>&1 || true
}

start_all() {
    require_cmd curl
    require_cmd python3
    mkdir -p "${RUNTIME_ROOT}/dash" "${RUNTIME_ROOT}/hls" "${RUNTIME_ROOT}/original-site-out" "${LOG_DIR}"

    log "[INFO] Start Sitcheck (Mode=${ORCH_MODE}, kein Port-80-Binding)"
    start_prognose_stack
    start_realtime_stack
    start_portal_stack

    log "[OK] Sitcheck gestartet. Main=:8090 | Realtime=:8080 | Analytics=:8501 | API=:8000"
    log "[INFO] Direkte Ports (:8080/:8501/:8000) sind Advanced/Debug. Standardzugang bleibt :8090."
}

stop_all() {
    log "[INFO] Stop Sitcheck (portal -> realtime -> prognose)"
    stop_portal_stack
    stop_realtime_stack
    stop_prognose_stack
    log "[OK] Stop ausgeführt"
}

status_health_line() {
    local label="$1"
    local url="$2"
    if http_ok "$url"; then
        echo "[OK] ${label}: ${url}"
    else
        echo "[WARN] ${label}: nicht erreichbar (${url})"
    fi
}

print_local_service_status() {
    local name="$1"
    local port="$2"
    local pid_file
    pid_file="$(service_pid_file "${name}")"
    if [ -f "${pid_file}" ]; then
        local pid
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if pid_alive "${pid}"; then
            echo "[OK] prognose/${name}: PID=${pid}, Port=${port}"
            return 0
        fi
        echo "[WARN] prognose/${name}: PID-File vorhanden, Prozess tot (PID=${pid})"
        return 0
    fi

    if port_in_use "${port}"; then
        echo "[INFO] prognose/${name}: Port ${port} aktiv (externer Prozess, kein PID-File)."
    else
        echo "[WARN] prognose/${name}: nicht aktiv."
    fi
}

print_integration_metrics() {
    if ! http_ok "${REALTIME_STATE_URL}"; then
        echo "integration_writer: state endpoint nicht erreichbar (${REALTIME_STATE_URL})"
        return 0
    fi
    curl -fsS --max-time 4 "${REALTIME_STATE_URL}" | python3 -c '
import json
import sys
try:
    payload = json.load(sys.stdin)
except Exception:
    print("integration_writer: state payload ungültig")
    raise SystemExit(0)
writer = payload.get("integration_writer", {}) or {}
if not writer:
    print("integration_writer: keine Daten")
    raise SystemExit(0)
print("integration_writer.enabled=%s" % writer.get("enabled", False))
print("integration_writer.last_successful_write_at=%s" % writer.get("last_successful_write_at"))
print("integration_writer.current_write_rate=%s" % writer.get("current_write_rate"))
print("integration_writer.spool_size=%s" % writer.get("spool_size"))
print("integration_writer.spool_flush_status=%s" % writer.get("spool_flush_status"))
print("integration_writer.last_error_class=%s" % writer.get("last_error_class"))
print("integration_writer.last_error_message=%s" % writer.get("last_error_message"))
'
}

print_forecast_trainer_metrics() {
    if ! http_ok "${FORECAST_TRAINER_STATUS_URL}"; then
        echo "forecast_trainer: status endpoint nicht erreichbar (${FORECAST_TRAINER_STATUS_URL})"
        return 0
    fi
    curl -fsS --max-time 4 "${FORECAST_TRAINER_STATUS_URL}" | python3 -c '
import json
import sys
try:
    payload = json.load(sys.stdin)
except Exception:
    print("forecast_trainer: status payload ungültig")
    raise SystemExit(0)
state = payload.get("state", {}) or {}
last = state.get("last_result") if isinstance(state.get("last_result"), dict) else {}
print("forecast_trainer.status=%s" % state.get("status"))
print("forecast_trainer.scheduler_status=%s" % state.get("scheduler_status"))
print("forecast_trainer.last_run_at=%s" % state.get("last_run_at"))
print("forecast_trainer.last_success_at=%s" % state.get("last_success_at"))
print("forecast_trainer.last_error_class=%s" % state.get("last_error_class"))
print("forecast_trainer.last_error_message=%s" % state.get("last_error_message"))
if last:
    print("forecast_trainer.with_lecture_run_id=%s" % last.get("with_lecture_run_id"))
    print("forecast_trainer.without_lecture_run_id=%s" % last.get("without_lecture_run_id"))
    ablation = last.get("ablation", {}) if isinstance(last.get("ablation"), dict) else {}
    print("forecast_trainer.mae_gain_primary_horizon=%s" % ablation.get("mae_gain_primary_horizon"))
    print("forecast_trainer.pinball_gain_primary_horizon=%s" % ablation.get("pinball_gain_primary_horizon"))
    print("forecast_trainer.coverage_delta_primary_horizon=%s" % ablation.get("coverage_delta_primary_horizon"))
'
}

show_status() {
    echo "sitcheckctl status ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "Mode: ${ORCH_MODE} (no-docker gewünscht)"
    echo "Portpolitik: kein Port 80 Binding durch Sitcheck (osTicket/Apache bleibt separat)."
    echo "Forecast backend: ${FORECAST_MODEL_BACKEND}"
    echo "Forecast training_mode: ${FORECAST_TRAINING_MODE}"
    echo "Forecast trainer enabled: ${FORECAST_TRAINER_ENABLED}"
    status_health_line "api-gateway" "${API_HEALTH_URL}"
    status_health_line "api-gateway legacy" "${API_LEGACY_HEALTH_URL}"
    status_health_line "calendar-ingest" "${CALENDAR_INGEST_HEALTH_URL}"
    status_health_line "mcp-sitcheck" "${MCP_HEALTH_URL}"
    status_health_line "lecture-ingest" "${LECTURE_INGEST_HEALTH_URL}"
    if [ "${FORECAST_TRAINER_ENABLED}" = "1" ]; then
        status_health_line "forecast-trainer" "${FORECAST_TRAINER_HEALTH_URL}"
    else
        echo "[INFO] forecast-trainer: deaktiviert (SITCHECK_FORECAST_TRAINER_ENABLED=${FORECAST_TRAINER_ENABLED_RAW})"
    fi
    status_health_line "portal hauptseite" "${PORTAL_HEALTH_URL}"
    status_health_line "realtime dashboard" "${REALTIME_HEALTH_URL}"
    status_health_line "streamlit analytics" "${STREAMLIT_HEALTH_URL}"
    echo

    echo "[bildauswertung]"
    "${BILDAUSWERTUNG_CTL}" status || true
    echo "[website-dashboard]"
    "${PORTAL_CTL}" status || true
    "${DASHBOARD_CTL}" status || true
    echo

    if [ "${ORCH_MODE}" = "local" ]; then
        echo "[prognose local services]"
        print_local_service_status "forecast" 8001
        print_local_service_status "xai" 8002
        print_local_service_status "recommendations" 8003
        print_local_service_status "api-gateway" 8000
        print_local_service_status "api-gateway-legacy" 5000
        print_local_service_status "lecture-ingest" 8012
        print_local_service_status "calendar-ingest" 8010
        print_local_service_status "mcp-sitcheck" 8081
        print_local_service_status "forecast-scheduler" 8011
        print_local_service_status "dashboard" 8501
        if [ "${FORECAST_TRAINER_ENABLED}" = "1" ]; then
            print_local_service_status "forecast-trainer" 8013
        else
            echo "[INFO] prognose/forecast-trainer: deaktiviert."
        fi
    else
        echo "[prognose docker compose]"
        if ! bash -lc "cd '${PROGNOSE_DIR}' && docker compose ps" 2>/dev/null; then
            echo "[WARN] docker compose ps nicht verfügbar"
        fi
    fi
    echo
    echo "[integration metrics]"
    print_integration_metrics
    echo
    echo "[forecast trainer]"
    if [ "${FORECAST_TRAINER_ENABLED}" = "1" ]; then
        print_forecast_trainer_metrics
    else
        echo "forecast_trainer: deaktiviert"
    fi
}

show_logs() {
    touch "${SITCHECKCTL_LOG}" "${LOG_DIR}/tracking.log" "${LOG_DIR}/dashboard.log" "${LOG_DIR}/portal.log" "${LOG_DIR}/integration_writer.log"
    touch "$(service_log_file forecast)" "$(service_log_file xai)" "$(service_log_file recommendations)" "$(service_log_file api-gateway)" "$(service_log_file api-gateway-legacy)" "$(service_log_file lecture-ingest)" "$(service_log_file calendar-ingest)" "$(service_log_file mcp-sitcheck)" "$(service_log_file forecast-scheduler)" "$(service_log_file dashboard)" "$(service_log_file forecast-trainer)"
    echo "Tailing logs aus ${LOG_DIR}"
    tail -F \
      "${SITCHECKCTL_LOG}" \
      "${LOG_DIR}/tracking.log" \
      "${LOG_DIR}/dashboard.log" \
      "${LOG_DIR}/portal.log" \
      "${LOG_DIR}/integration_writer.log" \
      "$(service_log_file forecast)" \
      "$(service_log_file xai)" \
      "$(service_log_file recommendations)" \
      "$(service_log_file api-gateway)" \
      "$(service_log_file api-gateway-legacy)" \
      "$(service_log_file lecture-ingest)" \
      "$(service_log_file calendar-ingest)" \
      "$(service_log_file mcp-sitcheck)" \
      "$(service_log_file forecast-scheduler)" \
      "$(service_log_file dashboard)" \
      "$(service_log_file forecast-trainer)"
}

case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 1
        start_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo "Main entry: http://<host>:8090"
        echo "Optional env: SITCHECK_ORCH_MODE=local|docker (default: local)"
        echo "Optional env: SITCHECK_FORECAST_MODEL_BACKEND=baseline|tf_mlp (default: tf_mlp)"
        echo "Optional env: SITCHECK_FORECAST_TRAINING_MODE=locked|maintenance (default: locked)"
        echo "Optional env: SITCHECK_FORECAST_SNAPSHOT_HORIZONS=60[,1440,...] (default: 60)"
        echo "Optional env: SITCHECK_FORECAST_TRAINER_ENABLED=0|1 (default: 0)"
        echo "Optional env: SITCHECK_TF_MIN_TRAIN_POINTS=<int> (default: 1000)"
        exit 1
        ;;
esac
