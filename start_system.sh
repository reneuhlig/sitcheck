#!/bin/bash
# start_system.sh - Startet das komplette Live-System
# Startet sowohl die Live-Detection als auch die Zeitreihenanalyse

# =============================================================================
# KONFIGURATION
# =============================================================================

DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="DHBW1234!?"
DB_NAME="ai_detection"
DB_PORT=5432

INPUT_X="input_x"
INPUT_Y="input_y"

YOLO_MODEL="yolov8n.pt"
CONFIDENCE_THRESHOLD=0.5
POLL_INTERVAL=0.5

ANALYSIS_INTERVAL=10  # Sekunden zwischen Zeitreihenanalysen

# Log-Dateien
LOG_DIR="logs"
DETECTION_LOG="${LOG_DIR}/detection.log"
ANALYSIS_LOG="${LOG_DIR}/analysis.log"

# =============================================================================
# FUNKTIONEN
# =============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_message "✗ Python3 nicht gefunden"
        exit 1
    fi
    log_message "✓ Python3 gefunden: $(python3 --version)"
}

check_dependencies() {
    log_message "Prüfe Python-Abhängigkeiten..."
    
    python3 - <<PYCODE
import sys
missing = []

try:
    import cv2
except ImportError:
    missing.append('opencv-python')

try:
    from ultralytics import YOLO
except ImportError:
    missing.append('ultralytics')

try:
    import pg8000
except ImportError:
    missing.append('pg8000')

try:
    import numpy
except ImportError:
    missing.append('numpy')

if missing:
    print(f"✗ Fehlende Pakete: {', '.join(missing)}")
    print(f"  Installiere mit: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("✓ Alle Abhängigkeiten installiert")
    sys.exit(0)
PYCODE

    if [ $? -ne 0 ]; then
        exit 1
    fi
}

create_directories() {
    log_message "Erstelle Verzeichnisse..."
    mkdir -p "$INPUT_X" "$INPUT_Y" "$LOG_DIR"
    log_message "✓ Verzeichnisse erstellt"
}

test_database() {
    log_message "Teste Datenbankverbindung..."
    
    python3 - <<PYCODE
import pg8000
import sys

try:
    conn = pg8000.connect(
        host='$DB_HOST',
        port=$DB_PORT,
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME',
        timeout=5
    )
    print('✓ Datenbankverbindung erfolgreich')
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'✗ Datenbankverbindung fehlgeschlagen: {e}')
    sys.exit(1)
PYCODE

    return $?
}

start_detection() {
    log_message "Starte Live-Detection..."
    
    python3 run_live_detection.py \
        --db-host "$DB_HOST" \
        --db-user "$DB_USER" \
        --db-password "$DB_PASSWORD" \
        --db-name "$DB_NAME" \
        --db-port "$DB_PORT" \
        --input-x "$INPUT_X" \
        --input-y "$INPUT_Y" \
        --yolo-model "$YOLO_MODEL" \
        --confidence-threshold "$CONFIDENCE_THRESHOLD" \
        --poll-interval "$POLL_INTERVAL" \
        --verbose \
        > "$DETECTION_LOG" 2>&1 &
    
    DETECTION_PID=$!
    echo $DETECTION_PID > "${LOG_DIR}/detection.pid"
    log_message "✓ Detection gestartet (PID: $DETECTION_PID)"
}

start_analysis() {
    log_message "Starte Zeitreihenanalyse..."
    
    # Warte kurz, damit erste Detections vorhanden sind
    sleep 5
    
    python3 TimeSeriesAnalyzer.py \
        --db-host "$DB_HOST" \
        --db-user "$DB_USER" \
        --db-password "$DB_PASSWORD" \
        --db-name "$DB_NAME" \
        --db-port "$DB_PORT" \
        --interval "$ANALYSIS_INTERVAL" \
        > "$ANALYSIS_LOG" 2>&1 &
    
    ANALYSIS_PID=$!
    echo $ANALYSIS_PID > "${LOG_DIR}/analysis.pid"
    log_message "✓ Analyse gestartet (PID: $ANALYSIS_PID)"
}

stop_system() {
    log_message "Stoppe System..."
    
    if [ -f "${LOG_DIR}/detection.pid" ]; then
        DETECTION_PID=$(cat "${LOG_DIR}/detection.pid")
        if ps -p $DETECTION_PID > /dev/null 2>&1; then
            kill $DETECTION_PID
            log_message "✓ Detection gestoppt (PID: $DETECTION_PID)"
        fi
        rm "${LOG_DIR}/detection.pid"
    fi
    
    if [ -f "${LOG_DIR}/analysis.pid" ]; then
        ANALYSIS_PID=$(cat "${LOG_DIR}/analysis.pid")
        if ps -p $ANALYSIS_PID > /dev/null 2>&1; then
            kill $ANALYSIS_PID
            log_message "✓ Analyse gestoppt (PID: $ANALYSIS_PID)"
        fi
        rm "${LOG_DIR}/analysis.pid"
    fi
}

show_status() {
    log_message "System-Status:"
    
    if [ -f "${LOG_DIR}/detection.pid" ]; then
        DETECTION_PID=$(cat "${LOG_DIR}/detection.pid")
        if ps -p $DETECTION_PID > /dev/null 2>&1; then
            echo "  ✓ Detection läuft (PID: $DETECTION_PID)"
        else
            echo "  ✗ Detection nicht aktiv"
        fi
    else
        echo "  ✗ Detection nicht gestartet"
    fi
    
    if [ -f "${LOG_DIR}/analysis.pid" ]; then
        ANALYSIS_PID=$(cat "${LOG_DIR}/analysis.pid")
        if ps -p $ANALYSIS_PID > /dev/null 2>&1; then
            echo "  ✓ Analyse läuft (PID: $ANALYSIS_PID)"
        else
            echo "  ✗ Analyse nicht aktiv"
        fi
    else
        echo "  ✗ Analyse nicht gestartet"
    fi
    
    echo ""
    echo "Log-Dateien:"
    echo "  Detection: $DETECTION_LOG"
    echo "  Analyse: $ANALYSIS_LOG"
}

tail_logs() {
    log_message "Zeige Logs (Ctrl+C zum Beenden)..."
    tail -f "$DETECTION_LOG" "$ANALYSIS_LOG"
}

# =============================================================================
# HAUPTLOGIK
# =============================================================================

case "${1:-start}" in
    "start")
        log_message "=== STARTE LIVE-PERSONENERKENNUNG-SYSTEM ==="
        check_python
        check_dependencies
        create_directories
        
        if ! test_database; then
            log_message "✗ Datenbanktest fehlgeschlagen - breche ab"
            exit 1
        fi
        
        start_detection
        start_analysis
        
        echo ""
        log_message "✓ System erfolgreich gestartet"
        echo ""
        show_status
        echo ""
        log_message "Verwende './start_system.sh logs' um Logs zu verfolgen"
        log_message "Verwende './start_system.sh stop' um das System zu stoppen"
        ;;
        
    "stop")
        stop_system
        ;;
        
    "restart")
        stop_system
        sleep 2
        "$0" start
        ;;
        
    "status")
        show_status
        ;;
        
    "logs")
        tail_logs
        ;;
        
    "test-db")
        test_database
        ;;
        
    "help"|*)
        echo "Usage: $0 {start|stop|restart|status|logs|test-db}"
        echo ""
        echo "Befehle:"
        echo "  start     - Startet das komplette System"
        echo "  stop      - Stoppt das System"
        echo "  restart   - Neustart des Systems"
        echo "  status    - Zeigt den aktuellen Status"
        echo "  logs      - Verfolgt die Log-Dateien"
        echo "  test-db   - Testet die Datenbankverbindung"
        ;;
esac