# Website/Dashboard Map

## 1. Portal Hauptseite (Flask Gateway)
Pfad:
- `website-dashboard/portal/portal_app.py`

Aufgabe:
- Hauptzugang auf Port `8090`
- Liefert statische Original-Website aus `website-dashboard/runtime/original-site-out`
- Proxied Realtime unter `/realtime/*`
- Redirect auf Analytics unter `/analytics`
- Legacy-Kompatibilitaet (`/api/occupancy`, `/api/test`)
- Hub-Aggregation (`/api/hub/overview`) fuer die Hauptseite

Start:
- `./website-dashboard/portal/start_portal.sh start`
- oder zentral: `./sitcheckctl.sh start`

## 2. Realtime Dashboard (Flask)
Pfad:
- `bildauswertung/realtime/dashboard_app.py`

Aufgabe:
- Live-Trackingfeed (DASH)
- Bedienung für Zone/ROI
- Realtime Endpunkte (`/api/state`, `/api/zone`, `/api/tracking-roi`, `/health`)

Runtime-Artefakte:
- `website-dashboard/runtime/dash`
- `website-dashboard/runtime/hls`
- `website-dashboard/runtime/logs`

Start:
- `./sitcheckctl.sh start`

## 3. Analytics/Executive Dashboard (Streamlit)
Pfad:
- `prognose/apps/dashboard/app.py`

Aufgabe:
- Forecast-/XAI-/Recommendation-Visualisierung
- Command Center + Assistant-Flows

Start (im Prognose-Kontext, no-docker via Root-Orchestrator):
- `./sitcheckctl.sh start`

## 4. Zusammenspiel
- Realtime Flask nutzt Trackingmodule aus `bildauswertung`.
- Realtime schreibt optional via Direct DB Write nach `prognose.counts` (`integration.prognose_db.*` in `bildauswertung/config.yaml`).
- Streamlit-Dashboard bleibt in `prognose` und konsumiert Prognose-APIs.
- Portal verbindet Original-Website, Realtime und Prognose-APIs unter einem Einstieg.
- Hauptzugang fuer Nutzer: `http://<host>:8090` (One Flow).

## 5. Feste UI-Verknüpfung
- Original-Website enthält Header/Footer Links auf `/realtime`, `/analytics` und `/api/v1/dashboard/command-center?...`
- Realtime UI enthält Link zur Hauptseite `http://<host>:8090`
- Realtime UI enthält Link: `Analytics öffnen` -> `http://<host>:8501`
- Realtime UI enthält Link: `API/Command Center` -> `http://<host>:8000/api/v1/dashboard/command-center?...`
- Direktzugriffe `:8080`, `:8501`, `:8000` gelten als Advanced/Debug-Pfade.
