#!/bin/bash

LOG_DIR="logs"
DASH_LOG="${LOG_DIR}/dashboard.log"
DASH_PID_FILE="${LOG_DIR}/dashboard.pid"
CONFIG_PATH="config.yaml"
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
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("yt_dlp", "yt-dlp"),
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

start_dashboard() {
    if [ -f "$DASH_PID_FILE" ]; then
        PID=$(cat "$DASH_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[INFO] Dashboard läuft bereits (PID: $PID)"
            exit 0
        fi
    fi

    log_message "[INFO] Starte Dashboard auf ${HOST}:${PORT}..."
    "$PYTHON_BIN" dashboard_app.py --config "$CONFIG_PATH" --host "$HOST" --port "$PORT" > "$DASH_LOG" 2>&1 &
    PID=$!
    echo "$PID" > "$DASH_PID_FILE"
    log_message "[OK] Dashboard gestartet (PID: $PID)"
}

stop_dashboard() {
    if [ -f "$DASH_PID_FILE" ]; then
        PID=$(cat "$DASH_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
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
