# Portal Contract

## Scope
Der Portal-Service liefert die originale SitCheck-Webseite als Hauptzugang auf Port `8090`
und verbindet Realtime- und Prognose-Stack ueber einen einheitlichen Einstieg.

## Runtime
- Startskript: `website-dashboard/portal/start_portal.sh`
- App: `website-dashboard/portal/portal_app.py`
- Host/Port Defaults: `0.0.0.0:8090`
- Log: `website-dashboard/runtime/logs/portal.log`
- PID: `website-dashboard/runtime/logs/portal.pid`

## Environment Variablen
- `SITCHECK_PORTAL_HOST` (Default `0.0.0.0`)
- `SITCHECK_PORTAL_PORT` (Default `8090`)
- `SITCHECK_ORIGINAL_SITE_OUT` (Default `website-dashboard/runtime/original-site-out`)
- `SITCHECK_REALTIME_BASE_URL` (Default `http://127.0.0.1:8080`)
- `SITCHECK_PROGNOSE_API_BASE_URL` (Default `http://127.0.0.1:8000`)
- `SITCHECK_DEFAULT_ZONE_ID` (Default `default-zone`)
- `SITCHECK_CC_HORIZON` (Default `60`)
- `SITCHECK_CC_HISTORY_MINUTES` (Default `180`)
- `SITCHECK_HUB_HISTORY_MINUTES` (Default `180`, History-Fenster fuer `/api/hub/overview`)
- `SITCHECK_PROXY_TIMEOUT_SECONDS` (Default `15`)
- `SITCHECK_COMMAND_CENTER_TIMEOUT_SECONDS` (Default wie `SITCHECK_PROXY_TIMEOUT_SECONDS`)
- `SITCHECK_HUB_COMMAND_CENTER_TIMEOUT_SECONDS` (Default `2.5`, Timeout nur fuer Command-Center im Hub)
- `SITCHECK_ANALYTICS_REDIRECT_URL` (optional, ueberschreibt dynamischen Redirect)

## Public Routes
- `GET /`
  - Statische Original-Website (`index.html`) aus `runtime/original-site-out`.
- `GET /health`
  - Portal-Health mit statischem Root-Pfad.
- `GET /api/hub/overview`
  - Aggregierte Hub-Ansicht (Service-Health, Live-Occupancy, Realtime-KPIs, Degraded-Status).
- `GET /api/test`
  - Legacy-Testantwort mit UTC-Zeit.
- `GET /api/occupancy`
  - Legacy-kompatibles Occupancy-Format:
    - `averagePersons`
    - `currentPersons`
    - `lastUpdated`
    - `history: [{persons, timestamp}]`
- `GET /analytics`
  - Redirect auf `http://<host>:8501` (oder `SITCHECK_ANALYTICS_REDIRECT_URL`).
- `/realtime`, `/realtime/` und `/realtime/*`
  - Transparenter Proxy auf Realtime (`:8080`).
- `/api/v1` und `/api/v1/*`
  - Transparenter Proxy auf Prognose API-Gateway (`:8000`).

## Static Routing Rules
Fuer Pfade ohne Dateiendung:
1. `<path>.html`
2. `<path>/index.html`
3. Kein stiller `index.html`-Fallback fuer unbekannte Pfade.

Fuer Pfade mit Dateiendung:
1. Direkter Dateipfad

Sonderregel:
- Unbekannte Systempfade (`/api/*`, `/realtime/*`, `/dash/*`) liefern 4xx (maschinenlesbar) statt statischem Fallback.
- Unbekannte UI-Pfade liefern bevorzugt `404.html` (falls vorhanden).

## Compatibility Mapping `/api/occupancy`
Quelle:
- `GET /api/v1/dashboard/command-center?zone_id=default-zone&horizon=60&history_minutes=180`

Mapping:
- `currentPersons` = `live.occupancy` (int, fallback `0`)
- `averagePersons` = Mittelwert der letzten `history.points[*].occupancy` (2 Nachkommastellen), fallback `currentPersons`
- `lastUpdated` = `live.timestamp`, fallback `meta.generated_at`
- `history` = letzte 20 Punkte, aufsteigend, als `{persons, timestamp}`

Fehlerfall:
- HTTP `503`
- Body: `{ "error": "Noch keine Auslastungsdaten verfügbar." }`

## Hub-Overview Contract (`/api/hub/overview`)
Responsefelder:
- `status`: `ok` oder `degraded`
- `generated_at`: ISO-UTC
- `zone_id`: aktive Zone (Default `default-zone`)
- `services`: Objekt mit Status je Dienst (`portal`, `realtime`, `realtime_state`, `api_gateway`, `analytics`, `command_center`)
- `occupancy`: Legacy-kompatible Sicht (`averagePersons`, `currentPersons`, `lastUpdated`, `history`)
- `forecast`: kompakte Forecast-Sicht aus `forecast_latest`/`forecast_long_term`:
  - `horizon`, `generated_at`, `summary`, `model_version`, `source`, `age_seconds`, `stale`
  - `current_yhat`, `peak_yhat`
  - `points: [{timestamp, yhat, pi_low, pi_high}]`
  - `long_term: [{horizon, model_version, generated_at, first_point}]`
- `realtime_state`: kompakte Realtime-Metriken (`occupancy`, `fps`, `inference_fps`, `tracks`, `entries_total`, `exits_total`, ...)
- `errors`: Liste degradierter/down Services inkl. Fehlermeldung
