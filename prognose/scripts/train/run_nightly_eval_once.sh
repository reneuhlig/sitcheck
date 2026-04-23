#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${FORECAST_TRAINER_URL:-http://127.0.0.1:8013}"

echo "Trigger manual nightly-evaluate run via ${TRAINER_URL}/run-once"
tmp_file="$(mktemp)"
http_code="$(curl -sS -o "${tmp_file}" -w '%{http_code}' -X POST "${TRAINER_URL}/run-once")"

if python3 -m json.tool < "${tmp_file}"; then
  :
else
  cat "${tmp_file}"
fi

rm -f "${tmp_file}"
if [ "${http_code}" -ge 400 ]; then
  echo "run-once failed with HTTP ${http_code}" >&2
  exit 1
fi
