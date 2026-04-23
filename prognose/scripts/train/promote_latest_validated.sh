#!/usr/bin/env bash
set -euo pipefail

FORECAST_URL="${FORECAST_URL:-http://127.0.0.1:8001}"
ZONE_ID="${ZONE_ID:-default-zone}"
PRIMARY_HORIZON="${PRIMARY_HORIZON:-60}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.08}"

python3 - "$FORECAST_URL" "$ZONE_ID" "$PRIMARY_HORIZON" "$MIN_IMPROVEMENT" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def _request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected payload type from {url}: {type(parsed)}")
    return parsed


forecast_url = sys.argv[1].rstrip("/")
zone_id = sys.argv[2]
primary_horizon = int(sys.argv[3])
min_improvement = float(sys.argv[4])

report_url = (
    f"{forecast_url}/v1/model/report/latest?"
    + urllib.parse.urlencode({"zone_id": zone_id, "horizon": primary_horizon})
)
try:
    report = _request_json(report_url)
except urllib.error.HTTPError as exc:
    raise SystemExit(f"failed to read latest report: HTTP {exc.code}") from exc

run_id = str(report.get("run_id") or "")
decision = report.get("decision", {}) if isinstance(report.get("decision"), dict) else {}
comparison = report.get("comparison", {}) if isinstance(report.get("comparison"), dict) else {}
champion = str(decision.get("champion_model") or "")
scientific_pass = bool(decision.get("scientific_pass", False))
improvement_map = comparison.get("improvement_vs_baseline_mae", {})
improvement = None
if isinstance(improvement_map, dict):
    try:
        improvement = float(improvement_map.get(champion))
    except Exception:
        improvement = None

if not run_id:
    raise SystemExit("latest report has no run_id")
if not scientific_pass:
    raise SystemExit(f"run {run_id} is not scientifically approved")
if champion != "tf_mlp":
    raise SystemExit(f"run {run_id} champion is '{champion}', expected 'tf_mlp'")
if improvement is None or improvement < min_improvement:
    raise SystemExit(
        f"run {run_id} improvement_vs_baseline_mae={improvement} below required {min_improvement}"
    )

promote_url = f"{forecast_url}/v1/train/promote"
payload = {"zone_id": zone_id, "run_id": run_id}
try:
    result = _request_json(promote_url, method="POST", payload=payload)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="ignore")
    raise SystemExit(f"promotion failed: HTTP {exc.code} {body}") from exc

print(json.dumps(result, indent=2))
PY
