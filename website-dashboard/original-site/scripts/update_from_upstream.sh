#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SITE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UPSTREAM_URL="${SITCHECK_UPSTREAM_URL:-https://github.com/PSSCoding/SitCheck}"
UPSTREAM_REF="${SITCHECK_UPSTREAM_REF:-}"
TMP_DIR="$(mktemp -d /tmp/sitcheck_upstream_sync_XXXXXX)"
UPSTREAM_DIR="${TMP_DIR}/repo"
APPLY_OVERRIDES_SCRIPT="${SCRIPT_DIR}/apply_local_overrides.sh"
BUILD_SCRIPT="${SCRIPT_DIR}/build_static_site.sh"

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

require_cmd git
require_cmd rsync

log "[INFO] Klone Upstream: ${UPSTREAM_URL}"
git clone --depth 1 "${UPSTREAM_URL}" "${UPSTREAM_DIR}"

if [ -n "${UPSTREAM_REF}" ]; then
    log "[INFO] Checke expliziten Upstream-Ref aus: ${UPSTREAM_REF}"
    (
        cd "${UPSTREAM_DIR}" || exit 1
        git fetch --depth 1 origin "${UPSTREAM_REF}"
        git checkout FETCH_HEAD
    )
fi

if [ ! -d "${UPSTREAM_DIR}/nextapp" ]; then
    log "[ERROR] Upstream enthält kein nextapp/-Verzeichnis"
    exit 1
fi

log "[INFO] Synchronisiere nextapp/ nach website-dashboard/original-site/nextapp"
rsync -a --delete \
    --exclude ".git" \
    --exclude ".next-root-backup" \
    --exclude "out-root-backup" \
    --exclude "out" \
    "${UPSTREAM_DIR}/nextapp/" "${ORIGINAL_SITE_DIR}/nextapp/"

if [ -x "${APPLY_OVERRIDES_SCRIPT}" ]; then
    "${APPLY_OVERRIDES_SCRIPT}"
else
    log "[WARN] Override-Skript fehlt oder nicht ausführbar: ${APPLY_OVERRIDES_SCRIPT}"
fi

if [ -x "${BUILD_SCRIPT}" ]; then
    "${BUILD_SCRIPT}"
else
    log "[ERROR] Build-Skript fehlt oder ist nicht ausführbar: ${BUILD_SCRIPT}"
    exit 1
fi

PINNED_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)"
echo "${PINNED_COMMIT}" > "${ORIGINAL_SITE_DIR}/UPSTREAM_PINNED_COMMIT"

log "[OK] Snapshot aktualisiert, Overrides angewendet und Build abgeschlossen (Commit: ${PINNED_COMMIT})"
log "[INFO] Temporäres Verzeichnis (optional löschen): ${TMP_DIR}"
