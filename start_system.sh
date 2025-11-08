#!/bin/bash
# start_system.sh - Startet das komplette Live-System mit Bewegungserkennung
# Startet sowohl die Live-Detection als auch die Bewegungsanalyse

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

ANALYSIS_INTERVAL=30  # Sekunden zwischen Bewegungsanalysen (erhöht auf 30s)

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
        log_message "ERROR: Python3 nicht gefunden"
        exit 1
    fi
    log_message "OK: Python3 gefunden: $(python3 --version)"
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
    print(f"ERROR: Fehlende Pakete: {', '.join(missing)}")
    print(f"  Installiere mit: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("OK: Alle Abhängigkeiten installiert")
    sys.exit(0)
PYCODE

    if [ $? -ne 0 ]; then
        exit 1
    fi
}

check_files() {
    log_message "Prüfe benötigte Dateien..."
    
    REQUIRED_FILES=(
        "run_live_detection.py"
        "run_time_series_analysis.py"
        "LiveProcessor.py"
        "TimeSeriesAnalyzer.py"
        "MovementDetector.py"
        "RoomOccupancyManager.py"
        "DatabaseHandler.py"
        "UltralyticsPersonDetector.py"
        "BaseDetector.py"
        "DataLoader.py"
    )
    
    MISSING=()
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            MISSING+=("$file")
        fi
    done
    
    if [ ${#MISSING[@]} -gt 0 ]; then
        log_message "ERROR: Fehlende Dateien:"
        for file in "${MISSING[@]}"; do
            echo "    - $file"
        done
        exit 1
    fi
    
    log_message "OK: Alle benötigten Dateien vorhanden"
}

create_directories() {
    log_message "Erstelle Verzeichnisse..."
    mkdir -p "$INPUT_X" "$INPUT_Y" "$LOG_DIR"
    log_message "OK: Verzeichnisse erstellt"
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
    print('OK: Datenbankverbindung erfolgreich')
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'ERROR: Datenbankverbindung fehlgeschlagen: {e}')
    sys.exit(1)
PYCODE

    return $?
}

initialize_database() {
    log_message "Initialisiere Datenbank-Tabellen..."
    
    python3 - <<PYCODE
import sys
from DatabaseHandler import DatabaseHandler

try:
    db = DatabaseHandler(
        host='$DB_HOST',
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME',
        port=$DB_PORT
    )
    
    if not db.connect():
        print('ERROR: Verbindung fehlgeschlagen')
        sys.exit(1)
    
    if not db.create_tables():
        print('ERROR: Tabellenerstellung fehlgeschlagen')
        sys.exit(1)
    
    db.close()
    print('OK: Datenbank initialisiert')
    sys.exit(0)
    
except Exception as e:
    print(f'ERROR: Fehler: {e}')
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
    log_message "OK: Live-Detection gestartet (PID: $DETECTION_PID)"
}

start_analysis() {
    log_message "Starte Bewegungsanalyse..."
    
    # Warte kurz, damit erste Detections vorhanden sind
    sleep 3
    
    # Hinweis: run_time_series_analysis.py hat DB-Config bereits integriert
    # Falls du diese konfigurierbar machen willst, musst du die Datei anpassen
    python3 run_time_series_analysis.py \
        > "$ANALYSIS_LOG" 2>&1 &
    
    ANALYSIS_PID=$!
    echo $ANALYSIS_PID > "${LOG_DIR}/analysis.pid"
    log_message "OK: Bewegungsanalyse gestartet (PID: $ANALYSIS_PID)"
    log_message "  (Analysiert alle ${ANALYSIS_INTERVAL}s)"
}

stop_system() {
    log_message "Stoppe System..."
    
    if [ -f "${LOG_DIR}/detection.pid" ]; then
        DETECTION_PID=$(cat "${LOG_DIR}/detection.pid")
        if ps -p $DETECTION_PID > /dev/null 2>&1; then
            kill $DETECTION_PID
            log_message "OK: Detection gestoppt (PID: $DETECTION_PID)"
        fi
        rm "${LOG_DIR}/detection.pid"
    fi
    
    if [ -f "${LOG_DIR}/analysis.pid" ]; then
        ANALYSIS_PID=$(cat "${LOG_DIR}/analysis.pid")
        if ps -p $ANALYSIS_PID > /dev/null 2>&1; then
            kill $ANALYSIS_PID
            log_message "OK: Bewegungsanalyse gestoppt (PID: $ANALYSIS_PID)"
        fi
        rm "${LOG_DIR}/analysis.pid"
    fi
}

show_status() {
    log_message "System-Status:"
    
    if [ -f "${LOG_DIR}/detection.pid" ]; then
        DETECTION_PID=$(cat "${LOG_DIR}/detection.pid")
        if ps -p $DETECTION_PID > /dev/null 2>&1; then
            echo "  OK: Live-Detection läuft (PID: $DETECTION_PID)"
        else
            echo "  ERROR: Live-Detection nicht aktiv"
        fi
    else
        echo "  ERROR: Live-Detection nicht gestartet"
    fi
    
    if [ -f "${LOG_DIR}/analysis.pid" ]; then
        ANALYSIS_PID=$(cat "${LOG_DIR}/analysis.pid")
        if ps -p $ANALYSIS_PID > /dev/null 2>&1; then
            echo "  OK: Bewegungsanalyse läuft (PID: $ANALYSIS_PID)"
        else
            echo "  ERROR: Bewegungsanalyse nicht aktiv"
        fi
    else
        echo "  ERROR: Bewegungsanalyse nicht gestartet"
    fi
    
    echo ""
    echo "Raumzustand abfragen:"
    echo "  psql -U $DB_USER -d $DB_NAME -c 'SELECT * FROM room_state ORDER BY timestamp DESC LIMIT 1;'"
    
    echo ""
    echo "Log-Dateien:"
    echo "  Detection: $DETECTION_LOG"
    echo "  Analyse:   $ANALYSIS_LOG"
}

tail_logs() {
    log_message "Zeige Logs (Ctrl+C zum Beenden)..."
    tail -f "$DETECTION_LOG" "$ANALYSIS_LOG"
}

show_room_state() {
    log_message "Aktueller Raumzustand:"
    
    python3 - <<PYCODE
import sys
from DatabaseHandler import DatabaseHandler

try:
    db = DatabaseHandler(
        host='$DB_HOST',
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME',
        port=$DB_PORT
    )
    
    if not db.connect():
        print('ERROR: Verbindung fehlgeschlagen')
        sys.exit(1)
    
    state = db.get_latest_room_state()
    
    if state:
        print(f"  Personen im Raum: {state['total_persons']}")
        print(f"  Zeitpunkt: {state['timestamp']}")
        print(f"  Grund: {state['change_reason']}")
        if state['confidence']:
            print(f"  Konfidenz: {state['confidence']:.2f}")
    else:
        print('  Noch keine Daten vorhanden')
    
    db.close()
    sys.exit(0)
    
except Exception as e:
    print(f'ERROR: Fehler: {e}')
    sys.exit(1)
PYCODE
}

show_statistics() {
    log_message "Statistiken (letzte 5 Minuten):"
    
    python3 - <<PYCODE
import sys
from DatabaseHandler import DatabaseHandler

try:
    db = DatabaseHandler(
        host='$DB_HOST',
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME',
        port=$DB_PORT
    )
    
    if not db.connect():
        print('ERROR: Verbindung fehlgeschlagen')
        sys.exit(1)
    
    cursor = db.connection.cursor()
    
    # Detections
    cursor.execute("""
        SELECT source, COUNT(*) 
        FROM live_detections 
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        GROUP BY source
    """)
    
    print("\n  Detections:")
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]}")
    
    # Bewegungen
    cursor.execute("""
        SELECT movement_type, COUNT(*), SUM(person_count)
        FROM movement_tracking 
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        GROUP BY movement_type
    """)
    
    print("\n  Bewegungen:")
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]} Ereignisse, {row[2]} Personen")
    
    # Unverarbeitet
    cursor.execute("""
        SELECT COUNT(*) 
        FROM live_detections 
        WHERE processed = FALSE
    """)
    
    unprocessed = cursor.fetchone()[0]
    print(f"\n  Unverarbeitete Detections: {unprocessed}")
    
    cursor.close()
    db.close()
    sys.exit(0)
    
except Exception as e:
    print(f'ERROR: Fehler: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYCODE
}

# =============================================================================
# HAUPTLOGIK
# =============================================================================

case "${1:-start}" in
    "start")
        log_message "=== STARTE LIVE-BEWEGUNGSERKENNUNGS-SYSTEM ==="
        check_python
        check_dependencies
        check_files
        create_directories
        
        if ! test_database; then
            log_message "ERROR: Datenbanktest fehlgeschlagen - breche ab"
            exit 1
        fi
        
        if ! initialize_database; then
            log_message "ERROR: Datenbank-Initialisierung fehlgeschlagen - breche ab"
            exit 1
        fi
        
        start_detection
        start_analysis
        
        echo ""
        log_message "OK: System erfolgreich gestartet"
        echo ""
        show_status
        echo ""
        log_message "Nützliche Befehle:"
        echo "  ./start_system.sh logs       - Logs verfolgen"
        echo "  ./start_system.sh status     - System-Status anzeigen"
        echo "  ./start_system.sh room       - Raumzustand anzeigen"
        echo "  ./start_system.sh stats      - Statistiken anzeigen"
        echo "  ./start_system.sh stop       - System stoppen"
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
        
    "room")
        show_room_state
        ;;
        
    "stats")
        show_statistics
        ;;
        
    "test-db")
        test_database
        ;;
        
    "init-db")
        initialize_database
        ;;
        
    "help"|*)
        echo "Usage: $0 {start|stop|restart|status|logs|room|stats|test-db|init-db}"
        echo ""
        echo "Befehle:"
        echo "  start     - Startet das komplette System (Detection + Bewegungsanalyse)"
        echo "  stop      - Stoppt das System"
        echo "  restart   - Neustart des Systems"
        echo "  status    - Zeigt den aktuellen Status"
        echo "  logs      - Verfolgt die Log-Dateien (Ctrl+C zum Beenden)"
        echo "  room      - Zeigt aktuellen Raumzustand (Personenanzahl)"
        echo "  stats     - Zeigt Statistiken der letzten 5 Minuten"
        echo "  test-db   - Testet die Datenbankverbindung"
        echo "  init-db   - Initialisiert die Datenbank-Tabellen"
        echo ""
        echo "Konfiguration:"
        echo "  Bearbeite die Variablen am Anfang dieser Datei:"
        echo "    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT"
        echo "    INPUT_X, INPUT_Y"
        echo "    YOLO_MODEL, CONFIDENCE_THRESHOLD"
        echo ""
        echo "Hinweis:"
        echo "  Die DB-Konfiguration in run_time_series_analysis.py muss separat"
        echo "  angepasst werden (Zeile 11-17)!"
        ;;
esac