# Sitcheck Workspace

Dieses Workspace bleibt in drei Hauptbereiche gegliedert:

## 1) Prognose
Pfad: `./prognose`

Inhalt:
- Forecasting, XAI, Recommendations
- API-Gateway auf `:8000`
- Lecture-Ingest auf `:8012`
- Forecast-Trainer (nightly scientific evaluate + ablation) auf `:8013`
- Streamlit-Analytics-Dashboard auf `:8501`

## 2) Bildauswertung
Pfad: `./bildauswertung`

Inhalt:
- YOLO-Tracking
- Trajektorien- und Event-Analyse (Entry/Exit)
- Occupancy-State + optionale Direct-DB-Integration in `prognose.counts`

## 3) Website/Dashboard
Pfad: `./website-dashboard`

Inhalt:
- Portal/Gateway (`website-dashboard/portal`) auf `:8090`
- Realtime Flask Dashboard (`website-dashboard/realtime`) auf `:8080`
- Original-Website Snapshot (`website-dashboard/original-site/nextapp`)
- Runtime-Ausgaben unter `website-dashboard/runtime`

## One Flow (empfohlen, no-docker)
Zentraler Start/Stop über Root-Orchestrator:

```bash
./sitcheckctl.sh start
./sitcheckctl.sh status
./sitcheckctl.sh logs
./sitcheckctl.sh stop
```

Scientific-Training Utilities:

```bash
./prognose/scripts/train/run_nightly_eval_once.sh
./prognose/scripts/train/promote_latest_validated.sh
./prognose/scripts/train/switch_backend_tf.sh tf_mlp
```

Hauptzugänge:
- Hauptseite (Portal + Original-Website): `http://<host>:8090`
- Realtime direkt: `http://<host>:8080`
- Analytics (Streamlit): `http://<host>:8501`
- API/Command Center: `http://<host>:8000/api/v1/dashboard/command-center?zone_id=default-zone&horizon=60&history_minutes=180`
- Lecture-Impact (latest): `http://<host>:8000/api/v1/lectures/impact/latest?zone_id=default-zone`
- Hub-API der Hauptseite: `http://<host>:8090/api/hub/overview`
- Forecast-Trainer Health: `http://<host>:8013/health`

Hinweis:
- Sitcheck bindet keinen Port `80` (osTicket/Apache bleibt unangetastet).
- Direkte Ports `8080/8501/8000` sind als Advanced-/Debug-Pfade gedacht; Standardzugang ist `:8090`.

## Original-Website Build
Statische Ausgabe fuer das Portal erzeugen:

```bash
./website-dashboard/original-site/scripts/build_static_site.sh
```

Upstream-Snapshot aktualisieren:

```bash
./website-dashboard/original-site/scripts/update_from_upstream.sh
```

Lokale Website-Overrides (update-sicher):
- Quelle: `website-dashboard/original-site/local-overrides/`
- Anwenden: `./website-dashboard/original-site/scripts/apply_local_overrides.sh`
- `update_from_upstream.sh` führt Overrides + Build automatisch aus.

Fixierter Upstream-Stand:
- `website-dashboard/original-site/UPSTREAM_PINNED_COMMIT`

## Kompatibilitäts-Wrapper (Root)
Die alten Root-Kommandos bleiben verfügbar und delegieren intern:
- `./start_system.sh ...` -> `./bildauswertung/start_system.sh ...`
- `./start_dashboard.sh ...` -> `./website-dashboard/realtime/start_dashboard.sh ...`

## Architektur- und Analyse-Dokumente
- Bereichsarchitektur Bildauswertung: `bildauswertung/docs/architecture.md`
- Dashboard-Mapping: `website-dashboard/docs/dashboard-map.md`
- Portal-Contract: `website-dashboard/portal/docs/portal-contract.md`
- Integrationsvertrag Vision -> Prognose: `docs/integration/vision-to-prognose-contract.md`
- Lecture-Impact Vertrag: `prognose/docs/lecture-impact-contract.md`
- Vollanalyse-Artefakte:
  - `docs/overview/01_inventory.md`
  - `docs/overview/02_dependency-map.md`
  - `docs/overview/03_area-mapping.csv`
