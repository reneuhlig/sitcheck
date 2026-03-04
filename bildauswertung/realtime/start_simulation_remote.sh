#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BILDAUSWERTUNG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${BILDAUSWERTUNG_ROOT}/runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
APP_LOG="${LOG_DIR}/simulation_remote.log"
APP_PID_FILE="${LOG_DIR}/simulation_remote.pid"
APP_PATH="${SCRIPT_DIR}/simulation_remote_app.py"
HOST="${SITCHECK_SIM_REMOTE_HOST:-0.0.0.0}"
PORT="${SITCHECK_SIM_REMOTE_PORT:-8091}"
DASHBOARD_API_BASE="${SITCHECK_DASHBOARD_API_BASE:-http://127.0.0.1:8080}"
RUNTIME_VENV="${SITCHECK_RUNTIME_VENV:-/tmp/sitcheck_runtime_venv}"
PYTHON_BIN=""

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_message "[ERROR] python3 nicht gefunden"
        exit 1
    fi
}

resolve_python_runtime() {
    if [ -n "${SITCHECK_PYTHON:-}" ]; then
        if [ ! -x "$SITCHECK_PYTHON" ]; then
            log_message "[ERROR] SITCHECK_PYTHON ist nicht ausführbar: $SITCHECK_PYTHON"
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
import sys
missing = []
for mod, pkg in [
    ("flask", "flask"),
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
        log_message "[INFO] Installiere fehlende Pakete: $MISSING"
        if ! $PYTHON_BIN -m pip install $MISSING; then
            log_message "[ERROR] Paketinstallation fehlgeschlagen: $MISSING"
            exit 1
        fi
    fi

    echo "[OK] Abhängigkeiten vorhanden"
}

create_directories() {
    mkdir -p "$LOG_DIR"
}

start_app() {
    if [ -f "$APP_PID_FILE" ]; then
        PID=$(cat "$APP_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[INFO] Simulation-Remote läuft bereits (PID: $PID)"
            exit 0
        fi
    fi

    log_message "[INFO] Starte Simulation-Remote auf ${HOST}:${PORT} (Dashboard: ${DASHBOARD_API_BASE})..."
    rm -f "$APP_PID_FILE"
    setsid -f bash -lc "cd '$WORKSPACE_ROOT' && echo \$\$ > '$APP_PID_FILE' && exec env PYTHONPATH='$WORKSPACE_ROOT/bildauswertung${PYTHONPATH:+:$PYTHONPATH}' '$PYTHON_BIN' '$APP_PATH' --host '$HOST' --port '$PORT' --dashboard-base-url '$DASHBOARD_API_BASE'" >> "$APP_LOG" 2>&1 < /dev/null

    for _ in $(seq 1 20); do
        if [ -f "$APP_PID_FILE" ]; then
            PID=$(cat "$APP_PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                log_message "[OK] Simulation-Remote gestartet (PID: $PID)"
                return
            fi
        fi
        sleep 0.1
    done

    log_message "[ERROR] Simulation-Remote konnte nicht gestartet werden. Siehe $APP_LOG"
    exit 1
}

stop_app() {
    if [ -f "$APP_PID_FILE" ]; then
        PID=$(cat "$APP_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID" >/dev/null 2>&1 || true
            sleep 0.5
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" >/dev/null 2>&1 || true
            fi
            log_message "[OK] Simulation-Remote gestoppt (PID: $PID)"
        else
            log_message "[INFO] PID $PID läuft nicht mehr"
        fi
        rm -f "$APP_PID_FILE"
    else
        log_message "[INFO] Kein Simulation-Remote-PID gefunden"
    fi
}

show_status() {
    if [ -f "$APP_PID_FILE" ]; then
        PID=$(cat "$APP_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[OK] Simulation-Remote aktiv (PID: $PID, URL: http://${HOST}:${PORT})"
            return
        fi
    fi
    log_message "[INFO] Simulation-Remote nicht aktiv"
}

show_logs() {
    touch "$APP_LOG"
    tail -f "$APP_LOG"
}

case "$1" in
    start)
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 1
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_app
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
