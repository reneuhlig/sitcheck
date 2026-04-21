# Bildauswertung Architektur

## Zweck
`bildauswertung` kapselt die komplette Video-/YOLO-basierte Erkennung, Tracking- und Ereignislogik.

Seit der letzten Integration liegt auch das Realtime-Dashboard direkt in diesem Modul unter `bildauswertung/realtime`.

## Pipeline
1. Video Ingest: `VideoInputModule.py`
2. Detektion/Tracking: `UltralyticsPersonDetector.py`, `YOLOTrackingModule.py`
3. Ereignislogik: `TrajectoryEntryAnalysisModule.py`
4. Zustand: `OccupancyStateModule.py`
5. Visualisierung/Kalibrierung: `VisualizationOutputModule.py`
6. Orchestrierung: `LiveProcessor.py` bzw. `run_live_detection.py`

### Realtime-Stack
- Dashboard/API/Overlay: `realtime/dashboard_app.py`
- Separate Remote-Webapp (Clip-Steuerung): `realtime/simulation_remote_app.py`
- Gemeinsamer Start/Stop Wrapper: `start_system.sh`
- DASH-Ausgabe: `runtime/dash/`
- Logs/PID: `runtime/logs/`

## Konfiguration
- Primär über `config.yaml`
- Laden/Speichern über `ConfigManager.py`
- Modell ist fest auf `yolo26n.pt` ausgelegt (abweichende Modellnamen werden abgewiesen)

### Videoquellen-Modi
- `youtube`: Stream über YouTube-URL (Auflösung via `yt-dlp`, optional Cookie-Datei)
- `livefeed_simulation`: lokaler Simulations-Feed über Video/Clips
- Simulation-Control (`video.simulation.control_mode`):
	- `remote_control` (jetzt): externer Controller/Webapp kann aktiv Clip wählen
	- `auto_rules` (später): regelbasierte automatische Clip-Auswahl
- Kein Kamera-Fallback: Bei Ausfall der primären Quelle wird nicht auf `/dev/video*` gewechselt

### LiveFeed Simulation Struktur (Stand 04.03.2026)
- Verzeichnis: `LiveFeed Simulation/`
- Enthält redundante Ordner-Sichten (`1 Person`, `Rein gehen`, `T ...` usw.)
- Bestand: **59 Video-Dateien**, davon **23 eindeutige Inhalte** (hash-basiert), **36 Duplikate**
- Neutraler Basis-Feed: `LiveFeed Simulation/Leerlauf.mov`

### Zugriffskonzept für Simulation (erweiterbar)
- Zentrale Registry: `simulation_video_registry.py`
- Verhalten:
	1. Scan des Simulations-Ordners nach Video-Dateien
	2. Deduplizierung per Dateihash (SHA-1)
	3. Auswahl eines kanonischen Pfads pro Inhalt (Alias-Pfade bleiben referenzierbar)
	4. Vergabe stabiler `clip_id` für API/Webapp-Steuerung
- Fallback-Reihenfolge für `livefeed_simulation`:
	1. expliziter `video.source`
	2. `video.simulation.default_clip_id`
	3. `Leerlauf.mov` (bzw. `Leerlauf.mp4`)
	4. erster verfügbarer Clip

### Dashboard-/API-Steuerung Simulation
- Video-Source API erweitert um Simulation-Control:
	- `GET/POST /api/video-source`
- Neuer Simulations-Katalog:
	- `GET /api/simulation/catalog` (kanonische Clips + Duplikatinfo)
- Neuer Remote-Playback-Endpunkt (für separate Webapp/Fernsteuerung):
	- `POST /api/simulation/select` mit `clip_id` und optional `control_mode`
	- Verhalten: ausgewählter Clip wird **einmal** abgespielt (`scheduled`), danach automatische Rückkehr zu `Leerlauf`

## Startpfade
- Primär (Tracking + Dashboard + Simulation-Remote): `./bildauswertung/start_system.sh start`
- Direkt: `python3 bildauswertung/run_live_detection.py --config bildauswertung/config.yaml`
- Status prüfen: `./bildauswertung/start_system.sh status`
- Logs ansehen: `./bildauswertung/start_system.sh logs [all|tracking|dashboard|remote]`

### Separate Remote-Webapp (MVP)
- Zweck: Fernsteuerung der Simulationsclips ohne Zugriff auf das volle Dashboard
- Standard-URL: `http://<host>:8091`
- Nutzt Dashboard als Backend (Default `SITCHECK_DASHBOARD_API_BASE=http://127.0.0.1:8080`)
- Wird standardmäßig mit `start_system.sh` mitgestartet (`SITCHECK_SIM_REMOTE_WITH_SYSTEM=1`, kompatibel zu `SITCHECK_SIM_REMOTE_WITH_DASHBOARD=1`)
- Funktionen:
	- Katalog laden (`/api/simulation/catalog` via Proxy)
	- Clip auswählen (`/api/simulation/select` via Proxy)
	- `control_mode` setzen (`remote_control`/`auto_rules`)

### Aktivierungslogik Simulations-Feed
- Der Feed wird **nur** als Simulations-Feed erzeugt, wenn im Tracking-Dashboard `video_mode=livefeed_simulation` gesetzt ist.
- Läuft das Dashboard in `video_mode=youtube`, bleibt die Simulation inaktiv (auch wenn die Remote-Webapp mitläuft).
- Wenn keine Fernbedienungs-Auswahl aktiv ist, läuft automatisch `Leerlauf`.
- Fernbedienungs-Auswahl überschreibt den laufenden Clip einmalig und fällt nach Clip-Ende auf `Leerlauf` zurück.

## Persistenz
- Optionales PostgreSQL-Logging über `DatabaseHandler.py`

## Laufzeitabhängigkeiten (relevant)
- `ultralytics`, `opencv-python`, `flask`, `yt-dlp`
- Für ByteTrack in dieser Umgebung zusätzlich: Paket `lapx` (stellt Modul `lap` bereit)

## Troubleshooting (kurz)
- **Keine Boxen / keine Bewegungsrichtung:**
	- `/api/state` prüfen: `track_ok` und `track_error`
	- Bei `No module named 'lap'`: `lapx` im Runtime-Venv installieren und Dienst neu starten
- **Schwarzes Bild bei YouTube:**
	- Cookie-Datei und URL in `config.yaml` prüfen
	- Dashboard-Log unter `runtime/logs/dashboard.log` auf `yt-dlp`-Fehler prüfen
- **Polygon-Overlay versetzt:**
	- Dashboard nutzt `object-fit: contain`-Mapping; bei Browser-Zoom/Resize Seite neu laden

## Nicht-Ziele
- Keine Forecasting-/XAI-/Recommendation-Logik in diesem Bereich
- Keine Streamlit-Analytics-UI in diesem Bereich
