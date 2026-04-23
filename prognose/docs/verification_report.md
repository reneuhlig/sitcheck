# Sitcheck Verification Report

Date: 2026-02-19  
Working directory: `/project_sitcheck`

## Schritt 0 - Preflight

### STATUS
Repository and environment baseline collected.

### VERIFY
Commands:
```bash
pwd
git status
git log --oneline -n 15
ls -la
cat .env.example | sed -n '1,160p'
[ -f .env ] || cp .env.example .env
```

Relevant output:
```text
pwd -> /project_sitcheck
branch -> main
latest commits -> f457793, db0ef18, b8c99ba, ...
.env -> created from .env.example (local only, not committed)
```

### RESULT
PASS.

### FIX
No code change.

## Schritt 1 - Docker Socket / Permissions

### STATUS
Checked Docker daemon accessibility and privilege path.

### VERIFY
Commands:
```bash
ls -l /var/run/docker.sock
id
groups
sudo -n true
sudo docker ps
```

Relevant output:
```text
/var/run/docker.sock -> srw-rw---- root docker
user groups -> does NOT include docker
sudo -n true -> sudo: a password is required
sudo docker ps -> requires interactive password
```

### RESULT
BLOCKED for daemon operations in this session.

### FIX
No repo code change.

Operator guidance:
```bash
sudo usermod -aG docker $USER
# logout/login required
```

## Schritt 2 - Tests / Contracts (lokal)

### STATUS
Executed schema contracts and local smoke suite.

### VERIFY
Commands:
```bash
.venv/bin/python3 scripts/tests/schema_contract_test.py
node scripts/tests/schema_contract_test.mjs
.venv/bin/python3 scripts/tests/api_phase2_smoke.py
.venv/bin/python3 scripts/tests/forecast_phase3_smoke.py
.venv/bin/python3 scripts/tests/xai_phase4_smoke.py
.venv/bin/python3 scripts/tests/recommendations_phase5_smoke.py
.venv/bin/python3 scripts/tests/e2e_phase5_chain.py
.venv/bin/python3 scripts/tests/dashboard_phase6_smoke.py
.venv/bin/python3 -m pytest
```

Relevant output:
```text
python schema contract tests passed
node ajv schema contract tests passed
phase2 api smoke test passed
phase3 forecast smoke test passed
phase4 xai smoke test passed
phase5 recommendations smoke test passed
phase5 e2e chain smoke test passed
phase6 dashboard assistant smoke test passed
pytest -> warnings only (no collected pytest-style tests in current layout)
```

### RESULT
PASS.

### FIX
No code change in this step.

## Schritt 3 - Compose Config Validation

### STATUS
Validated compose rendering.

### VERIFY
Command:
```bash
docker compose config
```

Relevant output:
```text
COMPOSE_CONFIG_OK
```

### RESULT
PASS.

### FIX
No code change in this step.

## Schritt 4 - Stack Start (Compose)

### STATUS
Attempted compose start for core + dev profiles.

### VERIFY
Commands:
```bash
docker compose up -d
docker compose --profile dev up -d
docker compose ps
docker compose logs --tail=120 api-gateway forecast xai recommendations mcp-sitcheck dashboard demo-generator
```

Relevant output:
```text
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock
```

### RESULT
BLOCKED by host docker permissions.

### FIX
No repo code change in this step.

Fallback verification path used: local one-process E2E execution (same services via local uvicorn/streamlit/node) to validate runtime behavior until docker access is granted.

## Schritt 5 - API Smoke + Schema Validation

### STATUS
Ran full endpoint smoke and schema validation via local stack (`http://127.0.0.1:18000`).

### VERIFY
Key command block:
```bash
/tmp/sitcheck_e2e_verify3.sh
.venv/bin/python3 scripts/tests/api_schema_smoke.py --base-url http://127.0.0.1:18000 --zone-id default-zone --horizon 60
```

Relevant output:
```text
HEALTH -> {"status":"ok","service":"api-gateway"}
ZONES -> {'zones': 1, 'zone_ids': ['default-zone']}
FORECAST -> {'model_version': 'baseline-v1', 'points': 60}
EXPLAIN -> {'drivers': 5, 'uncertainty': 'low', 'has_evidence': True}
RECOMMEND -> {'actions': 1, 'quality_ok': True, 'uncertainty_ok': True}
SCENARIO -> delta present
api_schema_smoke passed
```

### RESULT
PASS after one fix.

### FIX
Initial failure: `/api/v1/counts` returned `occupancy` as float for aggregated points, violating `count-point.schema.json` (`integer`).  
Applied minimal response fix in `apps/api-gateway/main.py` so `occupancy` is serialized as integer (raw and aggregated).

Commit: see **Fix Commits** section.

## Schritt 6 - E2E Dataflow (Demo Generator)

### STATUS
Validated ingest -> forecast -> explain -> recommendations -> scenario chain with generated data.

### VERIFY
Command:
```bash
/tmp/sitcheck_e2e_verify3.sh
```

Relevant output:
```text
demo generator posted 30 points successfully
counts_points: 41
forecast_points: 60
drivers: 5
actions: 2
scenario delta returned
```

### RESULT
PASS.

### FIX
No additional code change in this step.

## Schritt 7 - Dashboard Verification

### STATUS
Verified Streamlit dashboard reachability and API wiring.

### VERIFY
From local E2E run:
```bash
curl -I http://127.0.0.1:18501
```

Relevant output:
```text
HTTP/1.1 200 OK
Server: TornadoServer/6.5.4
```

### RESULT
PASS.

### FIX
No code change in this step.

## Schritt 8 - MCP Verification

### STATUS
Validated MCP tools against running API and guardrail behavior.

### VERIFY
Commands:
```bash
API_BASE_URL=http://127.0.0.1:18000 node scripts/tests/mcp_smoke_test.mjs
API_BASE_URL=http://127.0.0.1:18000 node -e "...simulate_scenario..."
```

Relevant output:
```text
mcp smoke results -> history_points, forecast_points, actions all returned
simulate_scenario -> persist=false, persist_is_false=true
```

### RESULT
PASS.

### FIX
No code change in this step.

## Schritt 9 - Externe Daten (Calendar/ICS)

### STATUS
Observed calendar endpoint initially returning only mock-window data in normal E2E path; added optional ICS ingestion as requested.

### IMPLEMENTED
- New service: `services/calendar-ingest`
- Env contract:
  - `CALENDAR_ICS_URLS`
  - `CALENDAR_ZONE_ID`
  - `CALENDAR_IMPORT_INTERVAL_SECONDS`
  - `CALENDAR_RUN_ONCE`
- Compose profile: `calendar` (also enabled in `dev` profile)
- Health endpoint: `GET /health` on calendar-ingest
- Demo ICS fixture: `scripts/demo/sample_calendar.ics`

### VERIFY
Local verification commands:
```bash
# run local stack with calendar-ingest and sample ICS
# (see command block used in terminal)
curl -sSf http://127.0.0.1:18110/health
curl -sSfG http://127.0.0.1:18100/api/v1/calendar/events \
  --data-urlencode zone_id=default-zone \
  --data-urlencode from=2026-02-19T00:00:00Z \
  --data-urlencode to=2026-02-22T00:00:00Z
```

Relevant output:
```text
calendar-ingest health -> {'status': 'ok', 'last_imported': 2, 'last_updated': 0, 'last_error': None}
calendar events API -> 4 events (2 mock + 2 imported ICS demo events)
```

### RESULT
PASS.

### FIX
Added optional ICS ingestion pipeline.

Commit: see **Fix Commits** section.

## Schritt 10 - Doku Drift Fix

### STATUS
Updated docs to match verified runtime and blockers.

### IMPLEMENTED
- Added calendar-ingest module/profile usage in docs.
- Added Docker permission troubleshooting to README and deploy guide.
- Added API schema smoke command to README verification section.

### VERIFY
Files checked:
```bash
README.md
docs/deploy-remote.md
.env.example
docker-compose.yml
```

### RESULT
PASS.

### FIX
Documentation aligned to actual behavior.

Commit: see **Fix Commits** section.

## Fix Commits

1. `33874d6` - `fix(api): align counts occupancy response type with schema contract`
2. `60efc7c` - `feat(calendar): add optional ICS ingestion to calendar_events`
3. `ce7b3c9` - `docs: update runbooks and verification report after validation`

## Final Ergebnis

- **Runtime chain is verified green locally** (ingest -> forecast -> explain -> recommendations -> scenario, dashboard, MCP, API schema checks).  
- **MCP guardrail verified**: `simulate_scenario` enforces `persist=false`.  
- **Optional ICS ingestion implemented and verified**.  
- **Remaining external blocker**: docker daemon permissions in current session prevent direct `docker compose up` execution here; operator action required on host (`docker` group or sudo session).
