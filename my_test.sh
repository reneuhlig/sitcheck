#!/bin/bash
# test_models.sh - Separate Testläufe für alle KI-Modelle inkl. Ollama
# Angepasst: Thread-Limits, faulthandler, core-dumps, CPU-safe option

# =============================================================================
# KONFIGURATION - BITTE ANPASSEN
# =============================================================================

# Datenbankverbindung
DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="DHBW1234!?"
DB_NAME="ai_detection"

# Datenpfad zu klassifizierten Bildern
DATA_DIR="/blob"

# Test-Parameter (sicherer Default für Tests)
TEST_MAX_IMAGES=100    # moderater Default (statt 500000) — für schnelle, sichere Tests
CONFIDENCE_THRESHOLD=0.5

# Optional: Standard CPU-Safe-Mode (leave empty to allow GPU)
# Wenn gesetzt auf "" werden GPUs ausgeblendet -> CPU-only (vermeidet viele CUDA-Segfaults)
DEFAULT_CUDA_VISIBLE_DEVICES=""

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Setze sichere Umgebungsvariablen vor jedem Run
env_preamble() {
    # Thread-Limits für native Bibliotheken
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    # Ultralytics safe mode & Python faulthandler
    export ULTRALYTICS_SAFE_MODE=1
    export PYTHONFAULTHANDLER=1

    # Erlaube Core-Dumps für Post-Mortem-Analyse (kann große Dateien erzeugen)
    ulimit -c unlimited
}

# Testet ein einzelnes Modell; 2. Param = zusätzliche CLI-Arg-Strings, 3. Param = zusätzliche ENV-Variablen
test_model() {
    local model=$1
    local extra_args="$2"
    local env_args="$3"   # e.g. "CUDA_VISIBLE_DEVICES=\"\""

    log_message "=== Teste $model Modell ==="

    # sichere envs setzen
    env_preamble

    # falls env_args übergeben -> eval um Environment vor Python zu setzen
    if [ -n "$env_args" ]; then
        eval $env_args python3 run_person_detection.py \
            --model "$model" \
            --db-host "$DB_HOST" \
            --db-user "$DB_USER" \
            --db-password "$DB_PASSWORD" \
            --db-name "$DB_NAME" \
            --data-dir "$DATA_DIR" \
            --max-images "$TEST_MAX_IMAGES" \
            --confidence-threshold "$CONFIDENCE_THRESHOLD" \
            --run-name "test_${model}_$(date +%Y%m%d_%H%M%S)" \
            $extra_args
    else
        python3 run_person_detection.py \
            --model "$model" \
            --db-host "$DB_HOST" \
            --db-user "$DB_USER" \
            --db-password "$DB_PASSWORD" \
            --db-name "$DB_NAME" \
            --data-dir "$DATA_DIR" \
            --max-images "$TEST_MAX_IMAGES" \
            --confidence-threshold "$CONFIDENCE_THRESHOLD" \
            --run-name "test_${model}_$(date +%Y%m%d_%H%M%S)" \
            $extra_args
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_message "✓ $model Test erfolgreich"
    else
        log_message "✗ $model Test fehlgeschlagen (Exit Code: $exit_code)"
    fi

    echo
    return $exit_code
}

# =============================================================================
# EINZELNE TESTFUNKTIONEN
# =============================================================================

test_ultralytics() {
    log_message "Starte Ultralytics YOLO Test..."

    # CPU-safe run (vermeidet CUDA/Treiber-Probleme)
    if [ -n "$DEFAULT_CUDA_VISIBLE_DEVICES" ]; then
        # explizit GPU erlauben - Leerstring bedeutet CPU-only
        CUDA_ARG="CUDA_VISIBLE_DEVICES=\"$DEFAULT_CUDA_VISIBLE_DEVICES\""
    else
        CUDA_ARG="CUDA_VISIBLE_DEVICES=\"\""
    fi

    # Nutze das kleinste Modell für Tests
    test_model "ultralytics" "--yolo-model-path yolov8n.pt" "$CUDA_ARG"
}

test_deepface() {
    log_message "Starte DeepFace Test..."

    echo "1. Test mit OpenCV Backend (CPU-safe):"
    # DeepFace nutzt häufig GPU falls verfügbar; hier CPU erzwingen per CUDA env
    if [ -n "$DEFAULT_CUDA_VISIBLE_DEVICES" ]; then
        CUDA_ARG="CUDA_VISIBLE_DEVICES=\"$DEFAULT_CUDA_VISIBLE_DEVICES\""
    else
        CUDA_ARG="CUDA_VISIBLE_DEVICES=\"\""
    fi

    test_model "deepface" "--deepface-backend opencv" "$CUDA_ARG"
}

test_ollama() {
    log_message "Starte Ollama Test (Gemma 3 Modelle)..."

    # Ollama läuft üblicherweise extern; kein CUDA-Override nötig, aber wir verwenden safe envs
    test_model "ollama-gemma3" "--ollama-model gemma3:4b"
}

test_database_connection() {
    log_message "Teste Datenbankverbindung..."

    python3 - <<PYCODE
import mysql.connector,sys
try:
    conn = mysql.connector.connect(
        host='$DB_HOST',
        user='$DB_USER',
        password='$DB_PASSWORD',
        database='$DB_NAME'
    )
    if conn.is_connected():
        print('✓ Datenbankverbindung erfolgreich')
        conn.close()
        sys.exit(0)
    else:
        print('✗ Datenbankverbindung fehlgeschlagen')
        sys.exit(1)
except Exception as e:
    print(f'✗ Datenbankverbindung fehlgeschlagen: {e}')
    sys.exit(1)
PYCODE
}

check_prerequisites() {
    log_message "Prüfe Voraussetzungen..."

    python3 - <<PYCODE
ok = True
try:
    import cv2, numpy, PIL, psutil, mysql.connector
    print('✓ Core-Bibliotheken verfügbar')
except ImportError as e:
    print(f'✗ Import-Fehler: {e}')
    ok = False

try:
    from ultralytics import YOLO
    print('✓ Ultralytics verfügbar')
except ImportError:
    print('✗ Ultralytics nicht verfügbar')

try:
    import deepface
    print('✓ DeepFace verfügbar')
except ImportError:
    print('✗ DeepFace nicht verfügbar')

try:
    import ollama
    print('✓ Ollama Python-Bibliothek verfügbar')
except ImportError:
    print('✗ Ollama Python-Bibliothek nicht verfügbar')

if not ok:
    raise SystemExit(1)
PYCODE

    # Datenverzeichnis prüfen
    if [ ! -d "$DATA_DIR" ]; then
        log_message "✗ Datenverzeichnis nicht gefunden: $DATA_DIR"
        return 1
    else
        log_message "✓ Datenverzeichnis gefunden: $DATA_DIR"
        image_count=$(find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        log_message "  Gefunden: $image_count Bilder"
        if [ $image_count -eq 0 ]; then
            log_message "⚠ Keine Bilder gefunden - erstelle Testbilder?"
        fi
    fi
}

# =============================================================================
# HAUPTLOGIK
# =============================================================================

case "${1:-all}" in
    "prerequisites"|"prereq")
        check_prerequisites
        ;;
    "database"|"db")
        test_database_connection
        ;;
    "ultralytics"|"yolo")
        test_ultralytics
        ;;
    "deepface"|"df")
        test_deepface
        ;;
    "ollama")
        test_ollama
        ;;
    "quick")
        log_message "=== SCHNELLTEST ALLER MODELLE ==="
        TEST_MAX_IMAGES=3
        check_prerequisites
        test_database_connection
        test_ultralytics
        test_deepface
        #test_ollama
        ;;
    "all"|"")
        log_message "=== VOLLSTÄNDIGER TEST ALLER MODELLE ==="
        check_prerequisites
        test_database_connection
        if [ $? -eq 0 ]; then
            test_ultralytics
            sleep 5
            test_deepface
            sleep 5
            test_ollama
        else
            log_message "✗ Datenbank-Test fehlgeschlagen - breche ab"
            exit 1
        fi
        ;;
    "help"|*)
        echo "Usage: $0 {prerequisites|database|ultralytics|deepface|ollama|quick|all}"
        ;;
esac
