# Remote Deployment Guide (Ubuntu)

This guide deploys Sitcheck on a remote Ubuntu host using Docker Compose.

## 1. Prerequisites

- Ubuntu 22.04+ server
- SSH access with sudo
- Open ports (at least):
  - `8000` API
  - `8501` Dashboard
  - `8081` MCP health (optional external)
  - `8011` Forecast scheduler health (optional external)
  - `8012` Lecture ingest health (optional external)
  - `5432` Postgres (optional external; usually keep internal)
  - `11434` Ollama (optional external)

## 2. Install Docker + Compose

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Re-login after group change.

## 3. Deploy

```bash
git clone <your-repo-url> /opt/sitcheck
cd /opt/sitcheck
cp .env.example .env
```

Edit `.env` with production values:

- `POSTGRES_PASSWORD`
- service URLs/ports
- zone defaults
- optional `CALENDAR_ICS_URLS` (comma-separated ICS URLs)
- optional `OLLAMA_ENABLED`
- `INTERNAL_API_TOKEN` (required for internal forecast snapshot writes)
- optional snapshot tuning: `FORECAST_SNAPSHOT_INTERVAL_SECONDS`, `FORECAST_SNAPSHOT_HORIZONS`, `FORECAST_SNAPSHOT_ZONES`, `MAX_FORECAST_HORIZON_MINUTES`, `LONG_HORIZON_STEP_MINUTES`
- optional `FORECAST_MODEL_BACKEND=tf_mlp` + `TF_*` vars for TensorFlow forecasting
- lecture enrichment defaults:
  - `LECTURE_SITE_CODE=MA`
  - `LECTURE_API_BASE_URL=https://api.dhbw.app`
  - `LECTURE_BACKFILL_ENABLED=true`
  - `LECTURE_REFRESH_INTERVAL_SECONDS=1800`

## 4. Start Stack

Core stack:

```bash
docker compose up -d
```

Core stack includes `forecast-scheduler` (constant snapshot generation).

Include demo generator:

```bash
docker compose --profile dev up -d
```

Include optional ICS calendar importer:

```bash
docker compose --profile calendar up -d
```

Run dev + calendar together:

```bash
docker compose --profile dev --profile calendar up -d
```

Include Ollama:

```bash
docker compose --profile ollama up -d
```

Include optional TF periodic trainer:

```bash
docker compose --profile ml-train up -d
```

Run one-shot multi-horizon training:

```bash
curl -X POST "http://localhost:8001/v1/train/batch" \
  -H "content-type: application/json" \
  -d '{"zone_id":"default-zone","horizons":[60,1440,10080,20160],"history_hours":720,"full_retrain":false}'
```

Backfill historical Excel counts:

```bash
.venv/bin/python scripts/data/import_excel_counts.py \
  --file /project_sitcheck/KI_Projekt_Daten_einJahr.xlsx \
  --api-base-url http://localhost:8000 \
  --zone-id default-zone
```

## 5. Verify

```bash
docker compose ps
curl http://localhost:8000/health
curl "http://localhost:8000/api/v1/zones"
curl "http://localhost:8000/api/v1/lectures/activity?zone_id=default-zone&from=2026-02-24T00:00:00Z&to=2026-02-24T23:59:59Z&granularity=15m"
curl "http://localhost:8000/api/v1/forecast?zone_id=default-zone&horizon=60"
curl "http://localhost:8000/api/v1/forecast/latest?zone_id=default-zone&horizon=60"
curl "http://localhost:8000/api/v1/dashboard/command-center?zone_id=default-zone&horizon=60&history_minutes=180&stale_seconds=900&long_term_days=14"
curl "http://localhost:8011/health"
curl "http://localhost:8012/health"
curl -I "http://localhost:8501"
```

MCP smoke (when API is up):

```bash
API_BASE_URL=http://localhost:8000 node scripts/tests/mcp_smoke_test.mjs
```

## 6. Ollama Notes

Check models on host:

```bash
ollama list
```

If needed, pull a model:

```bash
ollama pull gemma3:4b
```

Set:

- `OLLAMA_ENABLED=true`
- `OLLAMA_MODEL=gemma3:4b`
- `OLLAMA_BASE_URL=http://ollama:11434` (inside compose network)

## 7. Upgrade / Rollback

Upgrade:

```bash
git pull
docker compose build
docker compose up -d
```

Rollback:

```bash
git checkout <previous_commit_or_tag>
docker compose build
docker compose up -d
```

## 8. Operational Guardrails

- Keep MCP read-only in production.
- Do not expose raw DB publicly.
- Rotate secrets in `.env`.
- Keep API behind reverse proxy/TLS if internet exposed.

## 9. Troubleshooting Docker Permissions

If Docker commands fail with `/var/run/docker.sock: permission denied`:

```bash
sudo usermod -aG docker $USER
# logout/login required
groups
docker ps
```

If group update is not possible in your current session, run deployment commands with sudo.

## 10. Codex Execution Runbook

For iterative delivery on remote:

1. Implement one phase.
2. Run phase-local verification commands.
3. Commit with scoped message.
4. Deploy and run smoke checks.
5. Continue to next phase.
