#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${WORKSPACE_ROOT}/website-dashboard/runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
PORTAL_LOG="${LOG_DIR}/portal.log"
PORTAL_PID_FILE="${LOG_DIR}/portal.pid"
APP_PATH="${SCRIPT_DIR}/portal_app.py"
BUILD_SCRIPT="${WORKSPACE_ROOT}/website-dashboard/original-site/scripts/build_static_site.sh"
STATIC_OUT_DIR="${RUNTIME_ROOT}/original-site-out"
HOST="${SITCHECK_PORTAL_HOST:-0.0.0.0}"
PORT="${SITCHECK_PORTAL_PORT:-8090}"
RUNTIME_VENV="${SITCHECK_RUNTIME_VENV:-/tmp/sitcheck_runtime_venv}"
PYTHON_BIN=""

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_message "[ERROR] python3 nicht gefunden"
        exit 1
    fi
}

resolve_python_runtime() {
    if [ -n "${SITCHECK_PYTHON:-}" ]; then
        if [ ! -x "${SITCHECK_PYTHON}" ]; then
            log_message "[ERROR] SITCHECK_PYTHON ist nicht ausführbar: ${SITCHECK_PYTHON}"
            exit 1
        fi
        PYTHON_BIN="${SITCHECK_PYTHON}"
        return
    fi

    if [ ! -x "${RUNTIME_VENV}/bin/python" ]; then
        log_message "[INFO] Erzeuge Runtime-VENV: ${RUNTIME_VENV}"
        python3 -m venv --system-site-packages "${RUNTIME_VENV}"
    fi
    PYTHON_BIN="${RUNTIME_VENV}/bin/python"
}

check_dependencies() {
    MISSING="$(${PYTHON_BIN} - <<PYCODE
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
)"

    if [ -n "${MISSING}" ]; then
        log_message "[INFO] Installiere fehlende Pakete: ${MISSING}"
        if ! "${PYTHON_BIN}" -m pip install ${MISSING}; then
            log_message "[ERROR] Paketinstallation fehlgeschlagen: ${MISSING}"
            exit 1
        fi
    fi
}

ensure_runtime_dirs() {
    mkdir -p "${LOG_DIR}" "${STATIC_OUT_DIR}"
}

ensure_static_site() {
    if [ -f "${STATIC_OUT_DIR}/index.html" ]; then
        return
    fi
    log_message "[INFO] Statische Original-Website fehlt, starte Build."
    if [ ! -x "${BUILD_SCRIPT}" ]; then
        log_message "[ERROR] Build-Skript fehlt oder ist nicht ausführbar: ${BUILD_SCRIPT}"
        exit 1
    fi
    "${BUILD_SCRIPT}" >>"${PORTAL_LOG}" 2>&1
}

start_portal() {
    if [ -f "${PORTAL_PID_FILE}" ]; then
        PID="$(cat "${PORTAL_PID_FILE}")"
        if ps -p "${PID}" >/dev/null 2>&1; then
            log_message "[INFO] Portal läuft bereits (PID: ${PID})"
            exit 0
        fi
    fi

    rm -f "${PORTAL_PID_FILE}"
    log_message "[INFO] Starte Portal auf ${HOST}:${PORT}..."
    setsid -f bash -lc "cd '${WORKSPACE_ROOT}' && echo \$\$ > '${PORTAL_PID_FILE}' && exec '${PYTHON_BIN}' '${APP_PATH}' --host '${HOST}' --port '${PORT}'" >>"${PORTAL_LOG}" 2>&1 < /dev/null

    for _ in $(seq 1 20); do
        if [ -f "${PORTAL_PID_FILE}" ]; then
            PID="$(cat "${PORTAL_PID_FILE}")"
            if ps -p "${PID}" >/dev/null 2>&1; then
                log_message "[OK] Portal gestartet (PID: ${PID})"
                return
            fi
        fi
        sleep 0.1
    done

    log_message "[ERROR] Portal konnte nicht gestartet werden. Siehe ${PORTAL_LOG}"
    exit 1
}

stop_portal() {
    if [ -f "${PORTAL_PID_FILE}" ]; then
        PID="$(cat "${PORTAL_PID_FILE}")"
        if ps -p "${PID}" >/dev/null 2>&1; then
            local pgid
            pgid="$(ps -o pgid= -p "${PID}" 2>/dev/null | tr -d ' ')" || true
            if [ -n "${pgid}" ] && [ "${pgid}" != "0" ] && [ "${pgid}" != "1" ]; then
                kill -- "-${pgid}" >/dev/null 2>&1 || true
            else
                kill "${PID}" >/dev/null 2>&1 || true
            fi
            sleep 0.5
            if ps -p "${PID}" >/dev/null 2>&1; then
                if [ -n "${pgid}" ] && [ "${pgid}" != "0" ] && [ "${pgid}" != "1" ]; then
                    kill -9 -- "-${pgid}" >/dev/null 2>&1 || true
                else
                    kill -9 "${PID}" >/dev/null 2>&1 || true
                fi
            fi
            log_message "[OK] Portal gestoppt (PID: ${PID}, PGID: ${pgid:-?})"
        else
            log_message "[INFO] PID ${PID} läuft nicht mehr"
        fi
        rm -f "${PORTAL_PID_FILE}"
    else
        log_message "[INFO] Kein Portal-PID gefunden"
    fi
}

reclaim_portal_port() {
    local pids
    pids="$(ss -lptn "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)" || true
    if [ -z "${pids}" ]; then
        return 0
    fi
    while read -r pid; do
        [ -z "${pid}" ] && continue
        if ! ps -p "${pid}" >/dev/null 2>&1; then
            continue
        fi
        log_message "[WARN] Portal-Port ${PORT} belegt durch Altprozess (PID=${pid}); beende."
        kill "${pid}" >/dev/null 2>&1 || true
    done <<< "${pids}"
    sleep 0.5
    pids="$(ss -lptn "( sport = :${PORT} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)" || true
    while read -r pid; do
        [ -z "${pid}" ] && continue
        if ! ps -p "${pid}" >/dev/null 2>&1; then
            continue
        fi
        log_message "[WARN] Portal-Port ${PORT} SIGKILL (PID=${pid})."
        kill -9 "${pid}" >/dev/null 2>&1 || true
    done <<< "${pids}"
}

show_status() {
    if [ -f "${PORTAL_PID_FILE}" ]; then
        PID="$(cat "${PORTAL_PID_FILE}")"
        if ps -p "${PID}" >/dev/null 2>&1; then
            log_message "[OK] Portal aktiv (PID: ${PID}, URL: http://${HOST}:${PORT})"
            return
        fi
    fi
    log_message "[INFO] Portal nicht aktiv"
}

show_logs() {
    touch "${PORTAL_LOG}"
    tail -f "${PORTAL_LOG}"
}

case "${1:-}" in
    start)
        check_python
        resolve_python_runtime
        check_dependencies
        ensure_runtime_dirs
        ensure_static_site
        start_portal
        ;;
    stop)
        stop_portal
        ;;
    restart)
        stop_portal
        reclaim_portal_port
        sleep 1
        check_python
        resolve_python_runtime
        check_dependencies
        ensure_runtime_dirs
        ensure_static_site
        start_portal
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
