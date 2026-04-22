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

### Tracking- und Eventlogik (praesentationsgeeignet, Schritt fuer Schritt)

Die Realtime-Engine in `bildauswertung/realtime/dashboard_app.py` arbeitet in drei entkoppelten Loops:

1. Capture-Loop
- Liest kontinuierlich Frames aus der Quelle (YouTube oder Livefeed Simulation).
- Puffert Frames in einer Analyse-Queue.
- Bei Backpressure werden alte Frames verworfen (Queue-Schutz), damit Latenz stabil bleibt.

2. Inference-Loop
- Zieht Frames aus der Queue und fuehrt Tracking nur auf jedem n-ten Frame aus (`process_every_n_frames`).
- Unter Last wird `n` dynamisch erhoeht (dynamic skip), um Echtzeitfaehigkeit zu halten.
- Optional wird vor der Inferenz ein ROI-Crop angewendet, danach werden Track-Koordinaten auf das Originalbild zurueckgemappt.

3. Packetizer/Stream-Loop
- Nimmt den letzten Visual- oder Fallback-Raw-Frame und erzeugt DASH-Segmente.
- Dadurch bleibt der Stream fluessig, auch wenn Inferenz kurzzeitig aussetzt.

Evententscheidung (Entry/Exit):

1. Tracker liefert pro Person stabile Track-ID und Trajektorie.
2. `TrajectoryEntryAnalysisModule` prueft Crossing gegen die konfigurierte Zone:
- `line`: gerichtetes Ueberschreiten einer Linie.
- `polygon`: Eintritt/Austritt ueber Polygonregeln.
- `dual_polygon`: getrennte Entry- und Exit-Polygone (robusteste Option fuer Tuerszenarien).
3. Event wird erst akzeptiert, wenn zeitliche und geometrische Mindestkriterien erfuellt sind.
4. Akzeptierte Events aktualisieren Zaehler (`entries_total`, `exits_total`) und Belegungszustand.

Occupancy-Quellen:

- `occupancy_state`: strikt eventbasierter Zustand.
- `frame_near`: naehert ueber aktive Tracks im Frame an.
- `profile_simulation` aktiv: live Occupancy folgt Simulationsprofil (siehe unten), Detection-Events wirken als temporale Korrektur.

### Technisches Detail: Frame-, Direction-, Trajectory- und ReID-Logik

Dieser Abschnitt beschreibt die Entscheidungslogik auf Implementierungsebene.

1. Frame-Sampling und Laststeuerung
- Capture liefert bis `capture_max_fps` Frames in die Analyse-Queue.
- Inferenz verarbeitet nur jeden n-ten Frame (`process_every_n_frames`).
- Unter Queue-Druck wird `n` dynamisch bis `dynamic_skip_max_n` angehoben.
- Effekt: stabile Latenz statt wachsender Verzugszeit.

2. Track-Stabilisierung in `YOLOTrackingModule`
- Jede sichtbare ID wird mit EMA geglaettet:
  - `bbox_t = alpha * bbox_current + (1-alpha) * bbox_prev`
  - `center_t = alpha * center_current + (1-alpha) * center_prev`
- Kurzzeitverluste werden ueber `track_hold_frames` gepuffert.
- Die Confidence faellt in Hold-Frames exponentiell (`hold_confidence_decay`).
- `trail_length` speichert die letzten Zentren als Trajektorie.

3. Bewegungsrichtung (Direction)
- Aus den geglaetteten Zentren wird pro Frame ein Bewegungsvektor berechnet.
- Unterhalb `motion_min_pixels` wird ein Track als "still" markiert.
- Oberhalb der Schwelle wird in Hauptachsen/Diagonalen klassifiziert.
- Diese Richtung ist kein Event an sich, aber wichtig fuer Plausibilisierung.

4. Trajectory-basierte Event-Entscheidung
- `TrajectoryEntryAnalysisModule` verarbeitet nur nicht-stale Tracks.
- Pro Track wird eine Historie (`max_history`) als Punktfolge aufgebaut.
- Erst bei ausreichender Evidenz wird ein Event erzeugt:
  - Mindestpunkte (`min_track_points`),
  - Mindestverschiebung (`min_crossing_displacement_px`),
  - Cooldown (`min_event_cooldown_frames`).

5. Line-Mode
- Signierte Distanz zur Linie wird fuer aufeinanderfolgende Punkte berechnet.
- Vorzeichenwechsel entspricht Linienkreuzung.
- Entry/Exit folgt `line_entry_direction`.
- Ohne ausreichende Normalverschiebung kein Event (Jitter-Schutz).

6. Dual-Polygon-Mode
- Punkt wird gegen Entry- und Exit-Polygon klassifiziert.
- Zone-Wechsel wird ueber kurze Historie stabilisiert (Mehrheitsentscheidung).
- Event entsteht nur bei klarer Transition zwischen Entry<->Exit.
- Zusaetzlich prueft die Logik eine Divider-Linienkreuzung als starke Evidenz.

7. Fast-Transition fuer schnelle Personen
- Fuer kurze Historien (2-3 Punkte) wird ein Fast-Path aktiv.
- Direkter oder quasi-direkter Wechsel Exit->Entry bzw. Entry->Exit wird akzeptiert,
  wenn Schrittdistanz und Divider-Crossing plausibel sind.
- Dadurch werden schnelle Durchgaenge erkannt, die sonst wegen kurzer Sichtbarkeit verloren gingen.

8. ReID-Stitching bei ID-Wechseln
- Verschwindende Tracks landen in einer Lost-Map mit Zeitstempel und letzter Geschwindigkeit.
- Neue IDs werden gegen diese Lost-Kandidaten gematcht (Zeitluecke + Distanz + Positionsprognose).
- Score basiert auf Distanz zur vorhergesagten Position und zur letzten Position.
- Ambiguitaetsfilter: wenn der zweitbeste Kandidat zu nah am besten liegt,
  wird bewusst nicht gestitcht, um False-Merges in dichten Gruppen zu vermeiden.

9. Warum diese Kombination robust ist
- Kurzfristige Detektor-Aussetzer werden durch Track-Hold/EMA abgefedert.
- Schnelle Personen profitieren vom Fast-Transition-Pfad.
- Dichte Szenen profitieren vom konservativen ReID-Ambiguitaetsfilter.
- Event-Cooldowns verhindern Mehrfachzaehlung bei Oszillation an der Grenzlinie.

### Verhalten bei pausierter Auswertung (wichtig fuer Demo)

Wenn `analysis_enabled=false` gesetzt wird:

- Es werden keine Inferenz-API-Aufrufe mehr ausgefuehrt.
- Die Profilsimulation laeuft weiter und erzeugt Tick-basierte Entry/Exit-Deltas.
- Diese Deltas werden weiterhin an den Integration-Writer uebergeben, inklusive Heartbeat-Schreibungen.
- Damit bleiben `prognose.counts` und nachgelagerte Forecast-Services mit realistisch wirkendem Signal versorgt, auch im "Pause"-Modus.

Das ermoeglicht Live-Demos mit stabiler Prognose, ohne GPU/Inference-Druck.

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
- `SITCHECK_RECLAIM_PORTS=0|1` (Default: `1`, raeumt belegte Sitcheck-Ports vor Start auf)

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

### Scheduler ist degraded / Snapshot schlaegt fehl

Symptom:

- `sitcheckctl status` zeigt Warnung fuer `forecast-scheduler`.

Haeufige Ursache:

- Extern gestarteter API-Gateway-Prozess ohne `INTERNAL_API_TOKEN` belegt Port `8000`.

Loesung:

1. Mit Default-Start (`SITCHECK_RECLAIM_PORTS=1`) uebernimmt `sitcheckctl.sh` die Sitcheck-Ports deterministisch.
2. Danach `./sitcheckctl.sh restart` ausfuehren.
3. Erneut Status pruefen; Scheduler sollte Snapshot-Aufrufe wieder erfolgreich fahren.

## Weiterfuehrende Dokumente

- `Agent.md`
- `bildauswertung/docs/architecture.md`
- `website-dashboard/docs/dashboard-map.md`
- `website-dashboard/portal/docs/portal-contract.md`
- `docs/integration/vision-to-prognose-contract.md`
- `docs/overview/01_inventory.md`
- `docs/overview/02_dependency-map.md`
- `docs/overview/03_area-mapping.csv`
