#!/bin/bash
# =============================================================================
# Sitcheck – Installations-Script
# Installiert und konfiguriert alle Komponenten und Abhängigkeiten.
#
# Verwendung:
#   ./install.sh                      # Interaktiver Modus
#   ./install.sh --mode local         # Lokaler Betrieb (SQLite, kein Docker)
#   ./install.sh --mode docker        # Docker-Betrieb (TimescaleDB)
#   ./install.sh --skip-ml            # TensorFlow/SHAP/LightGBM überspringen
#   ./install.sh --skip-vision        # OpenCV/YOLO überspringen
#   ./install.sh --skip-website       # Next.js-Website nicht bauen
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGNOSE_DIR="${SCRIPT_DIR}/prognose"
BILDAUSWERTUNG_DIR="${SCRIPT_DIR}/bildauswertung"
WEBSITE_DIR="${SCRIPT_DIR}/website-dashboard"
NEXTAPP_DIR="${WEBSITE_DIR}/original-site/nextapp"

# Venvs — persistent, nicht in /tmp
PROGNOSE_VENV="${PROGNOSE_DIR}/.venv"
RUNTIME_VENV="${WEBSITE_DIR}/runtime/venv"   # bildauswertung + portal

# --- Farben ------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()     { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

# --- Argumente ---------------------------------------------------------------
INSTALL_MODE=""
SKIP_ML=0; SKIP_VISION=0; SKIP_WEBSITE=0

for arg in "$@"; do
    case "$arg" in
        --mode=*)      INSTALL_MODE="${arg#--mode=}" ;;
        --skip-ml)     SKIP_ML=1 ;;
        --skip-vision) SKIP_VISION=1 ;;
        --skip-website)SKIP_WEBSITE=1 ;;
        --help|-h)
            sed -n '/^# Verwendung/,/^# ===/p' "$0" | grep -E '^\s+\./install' | sed 's/#//'
            exit 0 ;;
    esac
done

# =============================================================================
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║        Sitcheck – Installations-Script           ║"
echo "  ║    Occupancy Intelligence · DHBW Mannheim        ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

# =============================================================================
section "1 · Betriebsmodus wählen"

if [[ -z "${INSTALL_MODE}" ]]; then
    echo "  Welchen Betriebsmodus möchtest du installieren?"
    echo "  [1] local  – Direkt auf dem Host, SQLite-Datenbank  (Standard)"
    echo "  [2] docker – Docker Compose, TimescaleDB"
    read -r -p "  Auswahl [1]: " mode_choice
    case "${mode_choice:-1}" in
        2|docker) INSTALL_MODE="docker" ;;
        *)        INSTALL_MODE="local"  ;;
    esac
fi
[[ "${INSTALL_MODE}" != "local" && "${INSTALL_MODE}" != "docker" ]] \
    && die "Unbekannter Modus: '${INSTALL_MODE}'. Erlaubt: local, docker"

ok "Modus: ${INSTALL_MODE}"
[[ "${SKIP_ML}"      == "1" ]] && warn "  --skip-ml:      TensorFlow / LightGBM / SHAP werden übersprungen"
[[ "${SKIP_VISION}"  == "1" ]] && warn "  --skip-vision:  OpenCV / YOLO werden übersprungen"
[[ "${SKIP_WEBSITE}" == "1" ]] && warn "  --skip-website: Next.js-Build wird übersprungen"

# =============================================================================
section "2 · Systemvoraussetzungen prüfen"

check_cmd() {
    local cmd="$1"; local label="${2:-$1}"; local required="${3:-1}"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "${label}: $(command -v "$cmd")"; return 0
    fi
    [[ "$required" == "1" ]] && die "${label} nicht gefunden. Bitte installieren."
    warn "${label} nicht gefunden (optional)."; return 1
}

# Python 3.11+
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c "import sys; print('%d%d' % sys.version_info[:2])" 2>/dev/null) || continue
        [[ "$ver" -ge 311 ]] && { PYTHON_BIN="$(command -v "$candidate")"; break; }
    fi
done
[[ -z "${PYTHON_BIN}" ]] && die "Python 3.11+ nicht gefunden. Bitte installieren."
ok "Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

# Node.js 18+
check_cmd node "Node.js" 1
NODE_MAJOR=$(node -e "process.stdout.write(String(process.versions.node.split('.')[0]))")
[[ "${NODE_MAJOR}" -lt 18 ]] && die "Node.js 18+ benötigt (gefunden: $(node --version))."
ok "Node.js: $(node --version)"
check_cmd npm "npm" 1
check_cmd curl "curl" 1
check_cmd ss "ss (iproute2)" 1

if [[ "${INSTALL_MODE}" == "docker" ]]; then
    check_cmd docker "Docker" 1
    docker compose version >/dev/null 2>&1 || die "Docker Compose v2 fehlt ('docker compose' nicht verfügbar)."
    ok "Docker Compose: $(docker compose version --short 2>/dev/null || echo 'ok')"
fi

# =============================================================================
section "3 · Python-Venv: Prognose-Stack  (${PROGNOSE_VENV})"
# Dieses Venv enthält alle Python-Dienste des Prognose-Stacks:
# api-gateway, forecast, xai, recommendations, scheduler, trainer,
# lecture-ingest, calendar-ingest, streamlit-dashboard

if [[ -x "${PROGNOSE_VENV}/bin/python" ]]; then
    info "Venv bereits vorhanden – überspringe Erstellung."
else
    info "Erstelle Virtualenv ..."
    "${PYTHON_BIN}" -m venv "${PROGNOSE_VENV}"
    ok "Virtualenv erstellt."
fi

PY="${PROGNOSE_VENV}/bin/python"
PIP="${PROGNOSE_VENV}/bin/pip"

info "Aktualisiere pip / setuptools ..."
"${PY}" -m pip install --upgrade pip setuptools wheel --quiet
ok "pip aktuell."

# Alle requirements.txt aus dem prognose/-Verzeichnis zusammenführen
TMP_REQ=$(mktemp /tmp/sitcheck-req-XXXXXX.txt)
trap 'rm -f "${TMP_REQ}"' EXIT

info "Sammle Pakete aus allen requirements.txt ..."
find "${PROGNOSE_DIR}" \
    \( -path "${PROGNOSE_DIR}/.venv" -o -path "${PROGNOSE_DIR}/node_modules" \) -prune \
    -o -name "requirements.txt" -print \
    | sort \
    | while read -r f; do
        info "  → ${f#"${SCRIPT_DIR}/"}"; cat "$f"; echo
    done > "${TMP_REQ}"

if [[ "${SKIP_ML}" == "1" ]]; then
    grep -v -iE '^(tensorflow|shap|lightgbm)' "${TMP_REQ}" > "${TMP_REQ}.f" && mv "${TMP_REQ}.f" "${TMP_REQ}"
    warn "ML-Pakete herausgefiltert (tensorflow, shap, lightgbm)."
fi

PKG_COUNT=$(grep -cE '^[a-zA-Z]' "${TMP_REQ}" 2>/dev/null || echo "?")
info "Installiere ~${PKG_COUNT} Pakete (pip dedupliziert automatisch) ..."
[[ "${SKIP_ML}" == "0" ]] && warn "TensorFlow ~600 MB – das kann einige Minuten dauern."

"${PIP}" install --quiet -r "${TMP_REQ}" \
    || die "pip install (Prognose) fehlgeschlagen."

ok "Alle Prognose-Pakete installiert."

# Schnell-Prüfung
"${PY}" -c "import fastapi, uvicorn, sqlalchemy, pandas, numpy" \
    && ok "Kern-Pakete (FastAPI, SQLAlchemy, pandas, numpy) verfügbar."
if [[ "${SKIP_ML}" == "0" ]]; then
    "${PY}" -c "import lightgbm" 2>/dev/null && ok "LightGBM verfügbar." \
        || warn "LightGBM nicht importierbar – Fallback-Modus aktiv."
    "${PY}" -c "import tensorflow" 2>/dev/null && ok "TensorFlow verfügbar." \
        || warn "TensorFlow nicht importierbar – nur LGBM-Backend nutzbar."
fi

# =============================================================================
section "4 · Python-Venv: Runtime-Stack  (${RUNTIME_VENV})"
# Gemeinsames Venv für bildauswertung (OpenCV, YOLO) und das Web-Portal (Flask).
# Persistenter Pfad statt /tmp — überlebt Neustarts.

mkdir -p "$(dirname "${RUNTIME_VENV}")"

if [[ -x "${RUNTIME_VENV}/bin/python" ]]; then
    info "Runtime-Venv bereits vorhanden – überspringe Erstellung."
else
    info "Erstelle Runtime-Virtualenv ..."
    "${PYTHON_BIN}" -m venv --system-site-packages "${RUNTIME_VENV}"
    ok "Runtime-Venv erstellt."
fi

RPY="${RUNTIME_VENV}/bin/python"
RPIP="${RUNTIME_VENV}/bin/pip"

"${RPY}" -m pip install --upgrade pip --quiet

# Portal: Flask
info "Installiere Portal-Abhängigkeiten (Flask) ..."
"${RPIP}" install --quiet flask
ok "Flask installiert."

# Bildauswertung: Vision-Pipeline
if [[ "${SKIP_VISION}" == "1" ]]; then
    warn "Vision-Pakete übersprungen (--skip-vision)."
else
    info "Installiere Vision-Abhängigkeiten (OpenCV, YOLO, Flask, etc.) ..."
    warn "ultralytics/YOLO ist ein großes Paket – kann einige Minuten dauern."
    "${RPIP}" install --quiet \
        "opencv-python" \
        "ultralytics" \
        "lapx" \
        "pg8000" \
        "numpy" \
        "openpyxl" \
        "pyyaml" \
        "yt-dlp" \
        "flask" \
        "imageio-ffmpeg" \
        || die "pip install (Vision) fehlgeschlagen."
    ok "Vision-Pakete installiert."

    # YOLO-Modell prüfen
    YOLO_MODEL="${BILDAUSWERTUNG_DIR}/models/yolo26n.pt"
    if [[ -f "${YOLO_MODEL}" ]]; then
        ok "YOLO-Modell vorhanden: ${YOLO_MODEL}"
    else
        warn "YOLO-Modell fehlt: ${YOLO_MODEL}"
        warn "Das Modell wird beim ersten Start der Bildauswertung automatisch heruntergeladen."
        warn "Alternativ: Datei manuell nach ${BILDAUSWERTUNG_DIR}/models/ kopieren."
    fi
fi

# =============================================================================
section "5 · Node.js-Abhängigkeiten installieren"

info "prognose/ (Monorepo-Root – ajv, schema-Tests) ..."
(cd "${PROGNOSE_DIR}" && npm install --silent)
ok "prognose/ Root-Module installiert."

info "apps/mcp-sitcheck ..."
(cd "${PROGNOSE_DIR}/apps/mcp-sitcheck" && npm install --silent)
ok "mcp-sitcheck installiert."

info "apps/command-center-web ..."
(cd "${PROGNOSE_DIR}/apps/command-center-web" && npm install --silent)
ok "command-center-web installiert."

if [[ -f "${NEXTAPP_DIR}/package.json" ]]; then
    info "website-dashboard/original-site/nextapp ..."
    (cd "${NEXTAPP_DIR}" && npm install --silent)
    ok "Next.js-App Module installiert."
fi

# =============================================================================
section "6 · Frontend-Apps bauen"

# Command Center React-App
info "Baue Command Center Web (React/Vite) ..."
(
    cd "${PROGNOSE_DIR}/apps/command-center-web"
    VITE_DEFAULT_ZONE_ID=default-zone \
    VITE_DEFAULT_HORIZON=210 \
    npm run build --silent
)
ok "Command Center gebaut (dist/ erstellt)."

# Next.js statische Website
if [[ "${SKIP_WEBSITE}" == "1" ]]; then
    warn "Next.js-Website übersprungen (--skip-website)."
elif [[ ! -f "${NEXTAPP_DIR}/package.json" ]]; then
    warn "Next.js-Quellen nicht gefunden – Website-Build übersprungen."
else
    info "Baue statische Website (Next.js) ..."
    BUILD_SCRIPT="${WEBSITE_DIR}/original-site/scripts/build_static_site.sh"
    if [[ -x "${BUILD_SCRIPT}" ]]; then
        bash "${BUILD_SCRIPT}"
        ok "Statische Website gebaut."
    else
        warn "Build-Script nicht gefunden: ${BUILD_SCRIPT}"
    fi
fi

# =============================================================================
section "7 · Umgebungsvariablen konfigurieren (.env)"

ENV_FILE="${PROGNOSE_DIR}/.env"
ENV_EXAMPLE="${PROGNOSE_DIR}/.env.example"

if [[ -f "${ENV_FILE}" ]]; then
    warn ".env bereits vorhanden – wird nicht überschrieben."
else
    info "Erstelle ${ENV_FILE} aus .env.example ..."
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"

    if [[ "${INSTALL_MODE}" == "docker" ]]; then
        echo
        warn "Wichtig: Passe das Datenbankpasswort in ${ENV_FILE} an."
        read -r -p "  DB-Passwort für 'sitcheck' [change_me]: " db_pass
        db_pass="${db_pass:-change_me}"
        if [[ "${db_pass}" != "change_me" ]]; then
            sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${db_pass}/" "${ENV_FILE}"
            sed -i "s|change_me@timescaledb|${db_pass}@timescaledb|g" "${ENV_FILE}"
            ok "DB-Passwort gesetzt."
        else
            warn "Standard-Passwort 'change_me' beibehalten – vor Produktionseinsatz ändern!"
        fi
    else
        # Lokaler Modus: SQLite-URL eintragen + persistentes Runtime-VENV
        {
            echo ""
            echo "# Lokaler Betrieb: SQLite (automatisch angelegt beim ersten Start)"
            echo "DATABASE_URL=sqlite:///${PROGNOSE_DIR}/runtime_local.db"
            echo ""
            echo "# Persistentes Runtime-VENV (bildauswertung + portal)"
            echo "SITCHECK_RUNTIME_VENV=${RUNTIME_VENV}"
        } >> "${ENV_FILE}"
        ok "SQLite-URL und Runtime-VENV in .env eingetragen."
    fi

    ok ".env erstellt."
fi

# INTERNAL_API_TOKEN-Hinweis
if grep -q "INTERNAL_API_TOKEN=change_me_internal" "${ENV_FILE}" 2>/dev/null; then
    warn "INTERNAL_API_TOKEN ist noch 'change_me_internal' – für Produktion bitte ändern."
fi

# =============================================================================
section "8 · Datenbank"

if [[ "${INSTALL_MODE}" == "local" ]]; then
    info "Lokaler Modus: SQLite-Datenbank."
    info "Die Datenbank wird beim ersten Start der api-gateway automatisch angelegt:"
    info "  → Base.metadata.create_all() + Schema-Erweiterungen"
    info "  → Default-Zone 'default-zone' und Seed-Kalender werden eingerichtet"
    DB_PATH="${PROGNOSE_DIR}/runtime_local.db"
    if [[ -f "${DB_PATH}" ]]; then
        ok "Datenbank bereits vorhanden: ${DB_PATH} ($(du -sh "${DB_PATH}" | cut -f1))"
    else
        info "Datenbank wird angelegt unter: ${DB_PATH}"
        # Frühzeitige Initialisierung über api-gateway startup (einmalig, dann sofort stoppen)
        info "Initialisiere SQLite-Schema jetzt (api-gateway einmalig starten) ..."
        INIT_LOG=$(mktemp /tmp/sitcheck-db-init-XXXXXX.log)
        (
            cd "${PROGNOSE_DIR}/apps/api-gateway"
            env DATABASE_URL="sqlite:///${DB_PATH}" \
                DEFAULT_ZONE_ID="default-zone" \
                DEFAULT_ZONE_CAPACITY="100" \
                FORECAST_SERVICE_URL="http://127.0.0.1:8001" \
                XAI_SERVICE_URL="http://127.0.0.1:8002" \
                RECOMMENDATIONS_SERVICE_URL="http://127.0.0.1:8003" \
                SCHEDULER_SERVICE_URL="http://127.0.0.1:8011" \
                INTERNAL_API_TOKEN="change_me_internal" \
                OLLAMA_ENABLED="false" \
            timeout 15 "${PY}" -c "
import sys, os
sys.path.insert(0, '.')
# Importiere nur das Datenbankschema, starte keinen Server
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
# Lade DB-Init-Logik aus main
import importlib.util, types
# Minimaler Import: nur ORM-Modelle + create_all
os.environ.setdefault('DATABASE_URL', 'sqlite:///${DB_PATH}')
from main import Base, engine, _ensure_default_zone, _seed_mock_calendar, _ensure_sqlite_indexes, _ensure_sqlite_schema_extensions
Base.metadata.create_all(bind=engine)
with Session(engine) as db:
    zone = _ensure_default_zone(db)
    _seed_mock_calendar(db, zone.zone_id)
    _ensure_sqlite_indexes(db)
    _ensure_sqlite_schema_extensions(db)
    db.commit()
print('DB-Init OK')
" 2>&1 || true
        ) > "${INIT_LOG}" 2>&1

        if grep -q "DB-Init OK" "${INIT_LOG}" 2>/dev/null; then
            ok "SQLite-Datenbank initialisiert: ${DB_PATH}"
        else
            warn "Automatische DB-Initialisierung nicht möglich (Dienste fehlen noch)."
            info "Die Datenbank wird beim ersten 'sitcheckctl.sh start' automatisch angelegt."
        fi
        rm -f "${INIT_LOG}"
    fi
else
    # Docker: TimescaleDB + Migrations über docker-entrypoint-initdb.d
    info "Docker-Modus: TimescaleDB."
    info "Starte TimescaleDB und wende Migrations an ..."
    (
        cd "${PROGNOSE_DIR}"
        docker compose up -d timescaledb
    )
    info "Warte auf TimescaleDB (max. 60 Sek.) ..."
    for i in $(seq 1 60); do
        if docker compose -f "${PROGNOSE_DIR}/docker-compose.yml" exec -T timescaledb \
            pg_isready -U sitcheck -d sitcheck >/dev/null 2>&1; then
            ok "TimescaleDB bereit."
            break
        fi
        [[ "$i" == "60" ]] && die "TimescaleDB nicht bereit nach 60 Sekunden."
        sleep 1
    done
    info "Migrations wurden automatisch über docker-entrypoint-initdb.d angewendet."
    info "  001_init.sql – Kern-Schema (zones, counts, forecasts, ...)"
    info "  002 – Indizes | 003 – Vorlesungen | 004 – Lineage | 005 – Auth"
    ok "Datenbank bereit."
fi

# =============================================================================
section "9 · Scripts ausführbar machen"

for sh_file in \
    "${SCRIPT_DIR}/sitcheckctl.sh" \
    "${BILDAUSWERTUNG_DIR}/start_system.sh" \
    "${WEBSITE_DIR}/portal/start_portal.sh" \
    "${WEBSITE_DIR}/original-site/scripts/build_static_site.sh"
do
    [[ -f "$sh_file" ]] && chmod +x "$sh_file" && ok "  +x $(basename "$sh_file")"
done

# =============================================================================
section "10 · Ollama (optional – für LLM-Narrative)"

OLLAMA_BIN="$(command -v ollama 2>/dev/null || true)"
if [[ -n "${OLLAMA_BIN}" ]]; then
    ok "Ollama bereits installiert: ${OLLAMA_BIN}"
    info "Prüfe ob Modell 'qwen2.5:0.5b' vorhanden ..."
    if ollama list 2>/dev/null | grep -q "qwen2.5:0.5b"; then
        ok "Modell qwen2.5:0.5b vorhanden."
    else
        info "Lade Modell qwen2.5:0.5b herunter (~400 MB) ..."
        if ollama pull qwen2.5:0.5b; then
            ok "Modell geladen."
        else
            warn "Modell-Download fehlgeschlagen. Manuell: ollama pull qwen2.5:0.5b"
        fi
    fi
else
    warn "Ollama nicht gefunden."
    echo ""
    echo "  Ollama wird für LLM-basierte Narrativ-Texte benötigt."
    echo "  Ohne Ollama läuft das System mit Template-Fallback (eingeschränkt)."
    echo ""
    echo "  Installation:"
    echo "    Linux/macOS:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "    Windows:      https://ollama.com/download"
    echo ""
    echo "  Nach der Installation:"
    echo "    ollama pull qwen2.5:0.5b"
    echo ""
    echo "  Dann in prognose/.env setzen:"
    echo "    OLLAMA_ENABLED=true"
fi

# =============================================================================
section "Zusammenfassung"
echo ""
echo -e "  ${GREEN}${BOLD}Installation abgeschlossen.${RESET}"
echo ""
echo -e "  ${BOLD}Modus:${RESET}           ${INSTALL_MODE}"
echo -e "  ${BOLD}Prognose-Venv:${RESET}   ${PROGNOSE_VENV}"
echo -e "  ${BOLD}Runtime-Venv:${RESET}    ${RUNTIME_VENV}"
echo -e "  ${BOLD}Config:${RESET}          ${PROGNOSE_DIR}/.env"

echo ""
echo -e "  ${BOLD}Nächste Schritte:${RESET}"
echo ""

if [[ "${INSTALL_MODE}" == "local" ]]; then
    echo "  System starten:"
    echo "    ./sitcheckctl.sh start"
    echo ""
    echo "  Status prüfen:"
    echo "    ./sitcheckctl.sh status"
    echo ""
    echo "  Logs:"
    echo "    ./sitcheckctl.sh logs"
    echo ""
    echo "  Erreichbar nach dem Start:"
    echo "    Portal      →  http://localhost:8090"
    echo "    Realtime    →  http://localhost:8080"
    echo "    Analytics   →  http://localhost:8501"
    echo "    API         →  http://localhost:8000/docs"
else
    echo "  Alle Dienste starten:"
    echo "    cd prognose && docker compose up -d"
    echo ""
    echo "  Mit optionalen Diensten:"
    echo "    docker compose --profile dev up -d       # Demo-Datengenerator"
    echo "    docker compose --profile ollama up -d    # Lokales LLM (Ollama)"
    echo "    docker compose --profile calendar up -d  # ICS-Kalender-Import"
    echo ""
    echo "  Oder alles über das Orchestrations-Script:"
    echo "    SITCHECK_ORCH_MODE=docker ./sitcheckctl.sh start"
    echo ""
    echo "  Status:"
    echo "    docker compose ps"
    echo "    docker compose logs -f"
fi

echo ""
[[ "${SKIP_ML}" == "1" ]] \
    && warn "ML-Pakete fehlen → Prognosemodell läuft im Baseline-Modus. Nachinstallieren: ./install.sh --mode ${INSTALL_MODE}"
[[ "${SKIP_VISION}" == "1" ]] \
    && warn "Vision-Pakete fehlen → Bildauswertung (YOLO) nicht verfügbar. Nachinstallieren: ./install.sh --mode ${INSTALL_MODE}"
echo ""
