#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${WORKSPACE_ROOT}/website-dashboard/runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
DASH_LOG="${LOG_DIR}/dashboard.log"
DASH_PID_FILE="${LOG_DIR}/dashboard.pid"
CONFIG_PATH="${WORKSPACE_ROOT}/bildauswertung/config.yaml"
APP_PATH="${SCRIPT_DIR}/dashboard_app.py"
HOST="${SITCHECK_DASHBOARD_HOST:-0.0.0.0}"
PORT="${SITCHECK_DASHBOARD_PORT:-8080}"
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
    ("cv2", "opencv-python"),
    ("ultralytics", "ultralytics"),
    ("pg8000", "pg8000"),
    ("numpy", "numpy"),
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
    mkdir -p "${RUNTIME_ROOT}/dash" "${RUNTIME_ROOT}/hls"
}

start_dashboard() {
    if [ -f "$DASH_PID_FILE" ]; then
        PID=$(cat "$DASH_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[INFO] Dashboard läuft bereits (PID: $PID)"
            exit 0
        fi
    fi

    if [ -z "${SITCHECK_VIDEO_SOURCE:-}" ] && [ -e "/dev/video0" ]; then
        export SITCHECK_VIDEO_SOURCE="0"
    fi
    if [ -z "${SITCHECK_VIDEO_FALLBACK_SOURCE:-}" ] && [ -e "/dev/video0" ]; then
        export SITCHECK_VIDEO_FALLBACK_SOURCE="0"
    fi
    if [ -z "${SITCHECK_PROGNOSE_API_TIMEOUT_SECONDS:-}" ]; then
        export SITCHECK_PROGNOSE_API_TIMEOUT_SECONDS="3"
    fi
    if [ -z "${SITCHECK_PROGNOSE_DB_MAX_WPS:-}" ]; then
        export SITCHECK_PROGNOSE_DB_MAX_WPS="2"
    fi

    log_message "[INFO] Starte Dashboard auf ${HOST}:${PORT}..."
    rm -f "$DASH_PID_FILE"
    setsid -f bash -lc "cd '$WORKSPACE_ROOT' && echo \$\$ > '$DASH_PID_FILE' && exec env PYTHONPATH='$WORKSPACE_ROOT/bildauswertung${PYTHONPATH:+:$PYTHONPATH}' '$PYTHON_BIN' '$APP_PATH' --config '$CONFIG_PATH' --host '$HOST' --port '$PORT'" >> "$DASH_LOG" 2>&1 < /dev/null

    for _ in $(seq 1 20); do
        if [ -f "$DASH_PID_FILE" ]; then
            PID=$(cat "$DASH_PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                log_message "[OK] Dashboard gestartet (PID: $PID)"
                return
            fi
        fi
        sleep 0.1
    done

    log_message "[ERROR] Dashboard konnte nicht gestartet werden. Siehe $DASH_LOG"
    exit 1
}

stop_dashboard() {
    if [ -f "$DASH_PID_FILE" ]; then
        PID=$(cat "$DASH_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID" >/dev/null 2>&1 || true
            sleep 0.5
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" >/dev/null 2>&1 || true
            fi
            log_message "[OK] Dashboard gestoppt (PID: $PID)"
        else
            log_message "[INFO] PID $PID läuft nicht mehr"
        fi
        rm -f "$DASH_PID_FILE"
    else
        log_message "[INFO] Kein Dashboard-PID gefunden"
    fi
}

show_status() {
    if [ -f "$DASH_PID_FILE" ]; then
        PID=$(cat "$DASH_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[OK] Dashboard aktiv (PID: $PID, URL: http://${HOST}:${PORT})"
            return
        fi
    fi
    log_message "[INFO] Dashboard nicht aktiv"
}

show_logs() {
    touch "$DASH_LOG"
    tail -f "$DASH_LOG"
}

case "$1" in
    start)
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_dashboard
        ;;
    stop)
        stop_dashboard
        ;;
    restart)
        stop_dashboard
        sleep 1
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_dashboard
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
