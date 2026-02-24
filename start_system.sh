#!/bin/bash
# start_system.sh - Start/stop wrapper for config-driven YOLO tracking runtime

LOG_DIR="logs"
TRACKING_LOG="${LOG_DIR}/tracking.log"
TRACKING_PID_FILE="${LOG_DIR}/tracking.pid"
CONFIG_PATH="config.yaml"
RUNTIME_VENV="${SITCHECK_RUNTIME_VENV:-/tmp/sitcheck_runtime_venv}"
PYTHON_BIN=""
HEADLESS_MODE="${SITCHECK_HEADLESS:-1}"

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

    echo "[OK] Abhängigkeiten vorhanden"
}

create_directories() {
    mkdir -p "$LOG_DIR"
}

start_tracking() {
    if [ -f "$TRACKING_PID_FILE" ]; then
        PID=$(cat "$TRACKING_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[INFO] Tracking läuft bereits (PID: $PID)"
            exit 0
        fi
    fi

    log_message "[INFO] Starte YOLO Live-Tracking..."
    HEADLESS_ARGS=()
    if [ "$HEADLESS_MODE" != "0" ]; then
        HEADLESS_ARGS+=("--headless")
    fi

    "$PYTHON_BIN" run_live_detection.py --config "$CONFIG_PATH" "${HEADLESS_ARGS[@]}" > "$TRACKING_LOG" 2>&1 &

    PID=$!
    echo "$PID" > "$TRACKING_PID_FILE"
    log_message "[OK] Tracking gestartet (PID: $PID)"
}

stop_tracking() {
    if [ -f "$TRACKING_PID_FILE" ]; then
        PID=$(cat "$TRACKING_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
            log_message "[OK] Tracking gestoppt (PID: $PID)"
        else
            log_message "[INFO] PID $PID läuft nicht mehr"
        fi
        rm -f "$TRACKING_PID_FILE"
    else
        log_message "[INFO] Kein Tracking-PID gefunden"
    fi
}

show_status() {
    if [ -f "$TRACKING_PID_FILE" ]; then
        PID=$(cat "$TRACKING_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_message "[OK] Tracking aktiv (PID: $PID)"
            return
        fi
    fi
    log_message "[INFO] Tracking nicht aktiv"
}

show_room_state() {
    "$PYTHON_BIN" - <<PYCODE
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
    touch "$TRACKING_LOG"
    tail -f "$TRACKING_LOG"
}

case "$1" in
    start)
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_tracking
        ;;
    stop)
        stop_tracking
        ;;
    restart)
        stop_tracking
        sleep 1
        check_python
        resolve_python_runtime
        check_dependencies
        create_directories
        start_tracking
        ;;
    status)
        show_status
        ;;
    room)
        show_room_state
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|room|logs}"
        exit 1
        ;;
esac
