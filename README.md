# Sitcheck

End-to-end System fuer Live-Bildauswertung, Belegungszaehlung, Prognose und Web-Ausspielung.

## Zielbild

Sitcheck beantwortet in Echtzeit:

- Wie viele Personen sind aktuell im Bereich?
- Wie entwickelt sich die Auslastung kurzfristig (Forecast)?
- Welche Handlungsempfehlungen lassen sich daraus ableiten?

Das System kombiniert dazu drei technische Domainen:

- Bildauswertung (YOLO + Eventlogik)
- Prognose-Services (Forecast, XAI, Recommendations)
- Web/Portal-Schicht (Hauptseite + Realtime + Analytics)

## Bereinigte Repo-Struktur

Die Struktur ist auf einen konsistenten Hauptfluss reduziert.

```text
sitcheck/
  sitcheckctl.sh                     # zentraler Orchestrator (einziger Root-Startpunkt)
  bildauswertung/                    # Vision-Pipeline + Realtime-App
    start_system.sh                  # Subsystem-Controller fuer Vision-Teil
    run_live_detection.py            # Standalone-CLI (Debug/Entwicklung)
    realtime/
      dashboard_app.py               # Realtime Flask + DASH + API
    integration/
      prognose_db_writer.py          # Direct-DB-Write in prognose.counts
    config.yaml                      # Zonen, Tracking, Integrations- und Runtime-Config
    models/                          # YOLO Gewichte
  website-dashboard/
    portal/                          # Hauptportal/Gateway auf :8090
      start_portal.sh
      portal_app.py
    original-site/                   # statische Originalseite + Build-Skripte
    runtime/                         # zentrale Runtime-Artefakte fuer Web/Portal
  prognose/                          # separates Prognose-Subprojekt
    apps/
    services/
    scripts/
  docs/                              # Integrations- und Strukturdokumente
```

Wichtige Bereinigungen:

- Root-Wrapper wurden entfernt, es bleibt nur noch `sitcheckctl.sh` als Root-Einstieg.
- Alte doppelte Realtime-Komponente unter `website-dashboard/realtime` wurde entfernt.
- Leerer Root-Ordner `models` wurde entfernt.

## Komponenten und Funktionsweise

## 1) Zentraler Orchestrator: sitcheckctl.sh

Datei: `sitcheckctl.sh`

Aufgabe:

- Startet und stoppt den gesamten Stack in definierter Reihenfolge.
- Fuehrt Health-Checks aus.
- Stellt aggregierten Status und zentrale Logs bereit.

Startreihenfolge:

1. Prognose-Stack (lokal oder Docker, standardmaessig lokal)
2. Realtime/Vision-Stack
3. Portal/Hauptseite

Stop-Reihenfolge:

1. Portal
2. Realtime/Vision
3. Prognose

Warum das wichtig ist:

- Vermeidet Race Conditions beim Boot.
- Verhindert teilweise gestartete UI bei nicht erreichbarer API.

## 2) Bildauswertung (Vision Core)

Ordner: `bildauswertung/`

Kernmodule:

- `VideoInputModule.py`: Quelle oeffnen/lesen (Datei, Stream, etc.).
- `UltralyticsPersonDetector.py`: YOLO-Inferenzadapter.
- `YOLOTrackingModule.py`: Tracking-Layer inkl. Stabilisierung.
- `TrajectoryEntryAnalysisModule.py`: Richtung/Zonen-Crossing erkennen.
- `OccupancyStateModule.py`: Zustandsautomat fuer Belegung.
- `VisualizationOutputModule.py`: Overlay-Rendering + Interaktion.
- `LiveProcessor.py`: orchestriert den Pipeline-Lauf.

Funktionskette pro Frame:

1. Frame wird gelesen.
2. YOLO detektiert Personen + IDs.
3. Trajektorien werden ueber mehrere Frames aufgebaut.
4. Zonenlogik entscheidet Entry/Exit-Events.
5. Occupancy-State wird eventbasiert aktualisiert.
6. Ergebnis wird visualisiert und optional persistiert.

Warum eventbasiert:

- Nicht die Anzahl der Boxen im Frame zaehlt, sondern verifizierte Uebertritte.
- Das reduziert Fehlzaehlungen durch Passanten, die nur vorbeilaufen.

## 3) Realtime Dashboard (Web + API + Stream)

Datei: `bildauswertung/realtime/dashboard_app.py`

Aufgabe:

- Bietet Live-UI zur Visualisierung und Kalibrierung.
- Stellt APIs fuer Zustand und Zone bereit.
- Erzeugt DASH-Ausgabe fuer den Live-Stream.

Wesentliche Endpunkte:

- `GET /health`
- `GET /api/state`
- `GET/POST /api/zone`
- `GET/POST /api/tracking-roi`
- `GET/POST /api/video-source`
- `GET /dash/*`

Besonderheit:

- Dashboard und Analyse laufen in einem gemeinsamen Runtime-Kontext,
  dadurch bleiben Bedienung und Ergebniszustand synchron.

## 4) Integration in Prognose-Datenbank

Datei: `bildauswertung/integration/prognose_db_writer.py`

Aufgabe:

- Schreibt Occupancy/Events optional direkt in `prognose.counts`.
- Nutzt Spool/Retry-Mechanik fuer robustes Schreiben bei temporaeren Ausfaellen.

Konfiguration:

- In `bildauswertung/config.yaml` unter `integration.prognose_db`.
- Log/Spool liegen unter `website-dashboard/runtime/logs`.

## 5) Website-Dashboard / Portal

Ordner: `website-dashboard/portal`

Aufgabe:

- Hauptzugang auf Port `8090`.
- Aggregiert/veraehnigt Einstiegspunkte:
  - Hauptseite
  - Realtime
  - Analytics
  - Hub-API

Dateien:

- `website-dashboard/portal/portal_app.py`
- `website-dashboard/portal/start_portal.sh`

## 6) Prognose-Subsystem

Ordner: `prognose/`

Aufgabe:

- Prognosemodelle und Services fuer Vorhersage + Erklaerbarkeit + Empfehlungen.

Typische Services (lokal):

- API Gateway `:8000` (+ legacy alias `:5000`)
- Forecast `:8001`
- XAI `:8002`
- Recommendations `:8003`
- Lecture ingest `:8012`
- Calendar ingest `:8010`
- Scheduler `:8011`
- Streamlit Dashboard `:8501`
- Optional Forecast Trainer `:8013`

## Betriebsmodus (One Flow)

Zentraler Betrieb erfolgt ausschliesslich ueber:

```bash
./sitcheckctl.sh start
./sitcheckctl.sh status
./sitcheckctl.sh logs
./sitcheckctl.sh stop
```

Zugaenge:

- Hauptseite: `http://<host>:8090`
- Realtime: `http://<host>:8080`
- Analytics: `http://<host>:8501`
- API: `http://<host>:8000`

Hinweis:

- Port 80 wird von Sitcheck nicht gebunden.

## Standalone-Modi (nur fuer Debug)

Vision-Subsystem direkt:

```bash
./bildauswertung/start_system.sh start
./bildauswertung/start_system.sh status
./bildauswertung/start_system.sh logs
./bildauswertung/start_system.sh stop
```

Wichtig:

- Nicht parallel `sitcheckctl.sh start` und `bildauswertung/start_system.sh start` gegen dieselbe Quelle laufen lassen,
  sonst droht doppelte Inferenzlast.

## Konfigurationslogik

Primare Vision-Konfiguration:

- `bildauswertung/config.yaml`

ENV-Overrides (Auszug):

- `SITCHECK_ORCH_MODE=local|docker`
- `SITCHECK_START_STANDALONE_TRACKING=0|1`
- `SITCHECK_TRACKING_WITH_SYSTEM=0|1`
- `SITCHECK_DASHBOARD_HOST`, `SITCHECK_DASHBOARD_PORT`
- `SITCHECK_PORTAL_HOST`, `SITCHECK_PORTAL_PORT`

## Troubleshooting

### Realtime ist nicht erreichbar

1. `./sitcheckctl.sh status`
2. Health fuer `:8080` pruefen
3. Logs ansehen mit `./sitcheckctl.sh logs`

### Doppelte Zaehllast / hohe GPU-Last

Pruefen, ob gleichzeitig laufen:

- `./sitcheckctl.sh start`
- `./bildauswertung/start_system.sh start`

Nur einen Startweg aktiv halten.

### API laeuft, Portal aber nicht

- Portal-Status pruefen in `sitcheckctl status`.
- Portal-Log unter `website-dashboard/runtime/logs/portal.log` ansehen.

## Weiterfuehrende Dokumente

- `Agent.md`
- `bildauswertung/docs/architecture.md`
- `website-dashboard/docs/dashboard-map.md`
- `website-dashboard/portal/docs/portal-contract.md`
- `docs/integration/vision-to-prognose-contract.md`
- `docs/overview/01_inventory.md`
- `docs/overview/02_dependency-map.md`
- `docs/overview/03_area-mapping.csv`
