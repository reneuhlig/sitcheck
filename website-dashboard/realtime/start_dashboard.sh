#!/bin/bash
# Compatibility wrapper: realtime dashboard moved to bildauswertung/realtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec "${WORKSPACE_ROOT}/bildauswertung/realtime/start_dashboard.sh" "$@"
