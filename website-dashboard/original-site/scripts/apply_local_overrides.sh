#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SITE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NEXTAPP_DIR="${ORIGINAL_SITE_DIR}/nextapp"
OVERRIDES_DIR="${ORIGINAL_SITE_DIR}/local-overrides"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log "[ERROR] Benötigter Befehl fehlt: ${cmd}"
        exit 1
    fi
}

if [ ! -d "${NEXTAPP_DIR}" ]; then
    log "[ERROR] Next.js Quelle fehlt: ${NEXTAPP_DIR}"
    exit 1
fi

if [ ! -d "${OVERRIDES_DIR}" ]; then
    log "[INFO] Keine lokalen Overrides vorhanden (${OVERRIDES_DIR})."
    exit 0
fi

require_cmd rsync

log "[INFO] Wende lokale Overrides an (${OVERRIDES_DIR} -> ${NEXTAPP_DIR})"
rsync -a "${OVERRIDES_DIR}/" "${NEXTAPP_DIR}/"
log "[OK] Lokale Overrides angewendet."
