#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SITE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ORIGINAL_SITE_DIR}/../.." && pwd)"
NEXTAPP_DIR="${ORIGINAL_SITE_DIR}/nextapp"
RUNTIME_OUT_DIR="${WORKSPACE_ROOT}/website-dashboard/runtime/original-site-out"

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

require_cmd npm
require_cmd python3

log "[INFO] Installiere Node-Abhängigkeiten (npm ci)"
(
    cd "${NEXTAPP_DIR}" || exit 1
    npm ci
)

log "[INFO] Baue statische Website (npm run build)"
(
    cd "${NEXTAPP_DIR}" || exit 1
    npm run build
)

EXPORT_DIR="${NEXTAPP_DIR}/out"
if [ ! -d "${EXPORT_DIR}" ]; then
    # Bei distDir='.next-local' liegt der Export ebenfalls in diesem Ordner.
    if [ -f "${NEXTAPP_DIR}/.next-local/index.html" ]; then
        EXPORT_DIR="${NEXTAPP_DIR}/.next-local"
    else
        log "[ERROR] Build-Ausgabe fehlt: ${NEXTAPP_DIR}/out (und kein .next-local Export gefunden)"
        exit 1
    fi
fi

mkdir -p "${RUNTIME_OUT_DIR}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${EXPORT_DIR}/" "${RUNTIME_OUT_DIR}/"
else
    python3 - <<PY
import os
import shutil

src = "${EXPORT_DIR}"
dst = "${RUNTIME_OUT_DIR}"

for name in os.listdir(dst):
    path = os.path.join(dst, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

for name in os.listdir(src):
    s = os.path.join(src, name)
    d = os.path.join(dst, name)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
PY
fi

touch "${RUNTIME_OUT_DIR}/.gitkeep"
log "[OK] Statische Website bereit unter: ${RUNTIME_OUT_DIR}"
