#!/bin/bash
# Compatibility wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/bildauswertung/start_system.sh" "$@"
