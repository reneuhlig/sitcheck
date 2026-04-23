#!/bin/bash

# Unified start/stop wrapper for tracking, dashboard and simulation-remote.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BILDAUSWERTUNG_ROOT="$SCRIPT_DIR"
WORKSPACE_ROOT="$(cd "${BILDAUSWERTUNG_ROOT}/.." && pwd)"
RUNTIME_ROOT="${BILDAUSWERTUNG_ROOT}/runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"

CONFIG_PATH="${BILDAUSWERTUNG_ROOT}/config.yaml"

TRACKING_LOG="${LOG_DIR}/tracking.log"
TRACKING_PID_FILE="${LOG_DIR}/tracking.pid"

DASHBOARD_LOG="${LOG_DIR}/dashboard.log"
DASHBOARD_PID_FILE="${LOG_DIR}/dashboard.pid"

SIM_REMOTE_LOG="${LOG_DIR}/simulation_remote.log"
SIM_REMOTE_PID_FILE="${LOG_DIR}/simulation_remote.pid"

DASHBOARD_APP_PATH="${BILDAUSWERTUNG_ROOT}/realtime/dashboard_app.py"
SIM_REMOTE_APP_PATH="${BILDAUSWERTUNG_ROOT}/realtime/simulation_remote_app.py"

HEADLESS_MODE="${SITCHECK_HEADLESS:-1}"
RUNTIME_VENV="${SITCHECK_RUNTIME_VENV:-/tmp/sitcheck_runtime_venv}"
PYTHON_BIN=""

DASHBOARD_HOST="${SITCHECK_DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${SITCHECK_DASHBOARD_PORT:-8080}"
SIM_REMOTE_HOST="${SITCHECK_SIM_REMOTE_HOST:-0.0.0.0}"
SIM_REMOTE_PORT="${SITCHECK_SIM_REMOTE_PORT:-8091}"
SIM_REMOTE_WITH_SYSTEM="${SITCHECK_SIM_REMOTE_WITH_SYSTEM:-${SITCHECK_SIM_REMOTE_WITH_DASHBOARD:-0}}"
TRACKING_WITH_SYSTEM="${SITCHECK_TRACKING_WITH_SYSTEM:-0}"
DASHBOARD_API_BASE="${SITCHECK_DASHBOARD_API_BASE:-http://127.0.0.1:${DASHBOARD_PORT}}"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

service_access_host() {
    local host="$1"
    if [ "$host" = "0.0.0.0" ]; then
        echo "127.0.0.1"
    else
        echo "$host"
    fi
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local retries="${3:-60}"
    local sleep_seconds="${4:-0.2}"

    for _ in $(seq 1 "$retries"); do
        if "$PYTHON_BIN" - <<PYCODE > /dev/null 2>&1
import urllib.request
urllib.request.urlopen("$url", timeout=1.5).read(1)
PYCODE
        then
            log_message "[OK] ${name} API bereit (${url})"
            return 0
        fi
        sleep "$sleep_seconds"
    done

    log_message "[ERROR] ${name} API nicht bereit: ${url}"
    return 1
}

check_python() {
    if ! command -v python3 > /dev/null 2>&1; then
        log_message "[ERROR] python3 nicht gefunden"
        exit 1
    fi
}

resolve_python_runtime() {
    if [ -n "${SITCHECK_PYTHON:-}" ]; then
        if [ ! -x "$SITCHECK_PYTHON" ]; then
            log_message "[ERROR] SITCHECK_PYTHON ist nicht ausfuehrbar: $SITCHECK_PYTHON"
            exit 1
        fi
        PYTHON_BIN="$SITCHECK_PYTHON"
        return
    fi

    if [ ! -x "$RUNTIME_VENV/bin/python" ]; then
        log_message "[INFO] Erzeuge Runtime-VENV: $RUNTIME_VENV"
        python3 -m venv --system-site-packages "$RUNTIME_VENV"
    fi
    PYTHON_BIN="$RUNTIME_VENV/bin/python"
}

check_dependencies() {
    MISSING=$($PYTHON_BIN - <<PYCODE
missing = []
for mod, pkg in [
    ("cv2", "opencv-python"),
    ("ultralytics", "ultralytics"),
    ("lap", "lapx"),
    ("pg8000", "pg8000"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
    ("yaml", "pyyaml"),
    ("yt_dlp", "yt-dlp"),
    ("flask", "flask"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
]:
    try:
        __import__(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print(" ".join(missing))
PYCODE
)

    if [ -n "$MISSING" ]; then
        log_message "[INFO] Installiere fehlende Pakete in Runtime-VENV: $MISSING"
        if ! $PYTHON_BIN -m pip install $MISSING; then
            log_message "[ERROR] Paketinstallation fehlgeschlagen: $MISSING"
            exit 1
        fi
    fi

    log_message "[OK] Abhaengigkeiten vorhanden"
}

create_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "${RUNTIME_ROOT}/dash" "${RUNTIME_ROOT}/hls"
}

is_running() {
    local pid_file="$1"
    if [ ! -f "$pid_file" ]; then
        return 1
    fi
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        return 1
    fi
    ps -p "$pid" > /dev/null 2>&1
}

start_component() {
    local name="$1"
    local pid_file="$2"
    local log_file="$3"
    local command="$4"

    if is_running "$pid_file"; then
        local existing_pid
        existing_pid="$(cat "$pid_file")"
        log_message "[INFO] ${name} laeuft bereits (PID: ${existing_pid})"
        return
    fi

    rm -f "$pid_file"
    setsid -f bash -lc "echo \$\$ > '$pid_file' && ${command}" >> "$log_file" 2>&1 < /dev/null

    for _ in $(seq 1 25); do
        if is_running "$pid_file"; then
            local started_pid
            started_pid="$(cat "$pid_file")"
            log_message "[OK] ${name} gestartet (PID: ${started_pid})"
            return
        fi
        sleep 0.1
    done

    log_message "[ERROR] ${name} konnte nicht gestartet werden. Siehe ${log_file}"
    exit 1
}

stop_component() {
    local name="$1"
    local pid_file="$2"

    if [ ! -f "$pid_file" ]; then
        log_message "[INFO] Kein PID fuer ${name} gefunden"
        return
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        rm -f "$pid_file"
        log_message "[INFO] Leere PID-Datei fuer ${name} entfernt"
        return
    fi

    if ps -p "$pid" > /dev/null 2>&1; then
        local pgid
        pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" || true
        if [ -n "$pgid" ] && [ "$pgid" != "0" ] && [ "$pgid" != "1" ]; then
            kill -- "-$pgid" > /dev/null 2>&1 || true
        else
            kill "$pid" > /dev/null 2>&1 || true
        fi
        sleep 0.5
        if ps -p "$pid" > /dev/null 2>&1; then
            if [ -n "$pgid" ] && [ "$pgid" != "0" ] && [ "$pgid" != "1" ]; then
                kill -9 -- "-$pgid" > /dev/null 2>&1 || true
            else
                kill -9 "$pid" > /dev/null 2>&1 || true
            fi
        fi
        log_message "[OK] ${name} gestoppt (PID: ${pid}, PGID: ${pgid:-?})"
    else
        log_message "[INFO] ${name}-PID laeuft nicht mehr (PID: ${pid})"
    fi
    rm -f "$pid_file"
}

start_tracking() {
    local headless_flag=""
    if [ "$HEADLESS_MODE" != "0" ]; then
        headless_flag="--headless"
    fi

    start_component \
        "Tracking" \
        "$TRACKING_PID_FILE" \
        "$TRACKING_LOG" \
        "cd '$BILDAUSWERTUNG_ROOT' && exec env PYTHONPATH='$BILDAUSWERTUNG_ROOT' '$PYTHON_BIN' '$BILDAUSWERTUNG_ROOT/run_live_detection.py' --config '$CONFIG_PATH' ${headless_flag}"
}

start_dashboard() {
    if [ -z "${SITCHECK_PROGNOSE_API_TIMEOUT_SECONDS:-}" ]; then
        export SITCHECK_PROGNOSE_API_TIMEOUT_SECONDS="3"
    fi
    if [ -z "${SITCHECK_PROGNOSE_DB_MAX_WPS:-}" ]; then
        export SITCHECK_PROGNOSE_DB_MAX_WPS="2"
    fi

    start_component \
        "Dashboard" \
        "$DASHBOARD_PID_FILE" \
        "$DASHBOARD_LOG" \
        "cd '$WORKSPACE_ROOT' && exec env PYTHONPATH='$BILDAUSWERTUNG_ROOT' '$PYTHON_BIN' '$DASHBOARD_APP_PATH' --config '$CONFIG_PATH' --host '$DASHBOARD_HOST' --port '$DASHBOARD_PORT'"

    local dashboard_host
    dashboard_host="$(service_access_host "$DASHBOARD_HOST")"
    wait_for_http "Dashboard" "http://${dashboard_host}:${DASHBOARD_PORT}/health" 100 0.2 || exit 1
}

start_sim_remote() {
    if [ "$SIM_REMOTE_WITH_SYSTEM" != "1" ]; then
        log_message "[INFO] Simulation-Remote Autostart deaktiviert (SITCHECK_SIM_REMOTE_WITH_SYSTEM=$SIM_REMOTE_WITH_SYSTEM)"
        return
    fi

    start_component \
        "Simulation-Remote" \
        "$SIM_REMOTE_PID_FILE" \
        "$SIM_REMOTE_LOG" \
        "cd '$WORKSPACE_ROOT' && exec env PYTHONPATH='$BILDAUSWERTUNG_ROOT' '$PYTHON_BIN' '$SIM_REMOTE_APP_PATH' --host '$SIM_REMOTE_HOST' --port '$SIM_REMOTE_PORT' --dashboard-base-url '$DASHBOARD_API_BASE'"

    local remote_host
    remote_host="$(service_access_host "$SIM_REMOTE_HOST")"
    wait_for_http "Simulation-Remote" "http://${remote_host}:${SIM_REMOTE_PORT}/health" 80 0.2 || exit 1
}

reclaim_port() {
    local port="$1"
    local label="$2"
    local pids
    pids="$(ss -lptn "( sport = :${port} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)" || true
    if [ -z "${pids}" ]; then
        return 0
    fi
    while read -r pid; do
        [ -z "${pid}" ] && continue
        if ! ps -p "${pid}" > /dev/null 2>&1; then
            continue
        fi
        log_message "[WARN] Port ${port} (${label}) belegt durch Altprozess (PID=${pid}); beende."
        kill "${pid}" > /dev/null 2>&1 || true
    done <<< "${pids}"
    sleep 0.5
    pids="$(ss -lptn "( sport = :${port} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)" || true
    while read -r pid; do
        [ -z "${pid}" ] && continue
        if ! ps -p "${pid}" > /dev/null 2>&1; then
            continue
        fi
        log_message "[WARN] Port ${port} (${label}) SIGKILL (PID=${pid})."
        kill -9 "${pid}" > /dev/null 2>&1 || true
    done <<< "${pids}"
}

reclaim_managed_ports() {
    log_message "[INFO] Bereinige verwaltete Ports vor Neustart."
    reclaim_port "${DASHBOARD_PORT}" "dashboard"
    reclaim_port "${SIM_REMOTE_PORT}" "simulation-remote"
}

stop_tracking() {
    stop_component "Tracking" "$TRACKING_PID_FILE"
}

stop_dashboard() {
    stop_component "Dashboard" "$DASHBOARD_PID_FILE"
}

stop_sim_remote() {
    stop_component "Simulation-Remote" "$SIM_REMOTE_PID_FILE"
}

show_component_status() {
    local name="$1"
    local pid_file="$2"
    local url="$3"

    if is_running "$pid_file"; then
        local pid
        pid="$(cat "$pid_file")"
        if [ -n "$url" ]; then
            log_message "[OK] ${name} aktiv (PID: ${pid}, URL: ${url})"
        else
            log_message "[OK] ${name} aktiv (PID: ${pid})"
        fi
    else
        log_message "[INFO] ${name} nicht aktiv"
    fi
}

show_status_all() {
    local dashboard_host
    local remote_host
    dashboard_host="$(service_access_host "$DASHBOARD_HOST")"
    remote_host="$(service_access_host "$SIM_REMOTE_HOST")"

    show_component_status "Tracking" "$TRACKING_PID_FILE" ""
    show_component_status "Dashboard" "$DASHBOARD_PID_FILE" "http://${dashboard_host}:${DASHBOARD_PORT}"
    show_component_status "Simulation-Remote" "$SIM_REMOTE_PID_FILE" "http://${remote_host}:${SIM_REMOTE_PORT}"
}

show_room_state() {
    PYTHONPATH="$BILDAUSWERTUNG_ROOT" "$PYTHON_BIN" - <<PYCODE
from ConfigManager import ConfigManager
from DatabaseHandler import DatabaseHandler

cfg = ConfigManager("$CONFIG_PATH").load()
db_cfg = cfg.get("database", {})
if not db_cfg.get("enabled", False):
    print("[INFO] Datenbank in config deaktiviert")
    raise SystemExit(0)

try:
    db = DatabaseHandler(
        host=db_cfg["host"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        database=db_cfg["database"],
        port=int(db_cfg["port"]),
    )
    if not db.connect():
        print("[ERROR] Keine DB-Verbindung")
        raise SystemExit(1)
    snapshot = db.get_occupancy_snapshot()
    print(f"Occupancy: {snapshot['current_occupancy']}")
    print(f"Updated: {snapshot['updated_at']}")
    print(f"Reason: {snapshot['change_reason']}")
    db.close()
except Exception as exc:
    print("[ERROR]", exc)
    raise SystemExit(1)
PYCODE
}

show_logs() {
    local target="${2:-all}"
    touch "$TRACKING_LOG" "$DASHBOARD_LOG" "$SIM_REMOTE_LOG"

    case "$target" in
        tracking)
            tail -f "$TRACKING_LOG"
            ;;
        dashboard)
            tail -f "$DASHBOARD_LOG"
            ;;
        remote|simulation_remote|simremote)
            tail -f "$SIM_REMOTE_LOG"
            ;;
        all)
            tail -f "$TRACKING_LOG" "$DASHBOARD_LOG" "$SIM_REMOTE_LOG"
            ;;
        *)
            echo "Usage: $0 logs [all|tracking|dashboard|remote]"
            exit 1
            ;;
    esac
}

start_all() {
    if [ "$TRACKING_WITH_SYSTEM" = "1" ]; then
        start_tracking
    else
        log_message "[INFO] Standalone-Tracking Autostart aus (SITCHECK_TRACKING_WITH_SYSTEM=$TRACKING_WITH_SYSTEM)"
    fi
    start_dashboard
    start_sim_remote
}

stop_all() {
    stop_sim_remote
    stop_dashboard
    stop_tracking
}

case "$1" in
    start)
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        reclaim_managed_ports
        sleep 1
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_all
        ;;
    status)
        show_status_all
        ;;
    room)
        check_python
        resolve_python_runtime
        show_room_state
        ;;
    logs)
        show_logs "$@"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|room|logs [all|tracking|dashboard|remote]}"
        exit 1
        ;;
esac
