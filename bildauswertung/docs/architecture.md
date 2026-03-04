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
- Start/Stop Wrapper: `realtime/start_dashboard.sh`
- DASH-Ausgabe: `runtime/dash/`
- Logs/PID: `runtime/logs/`, `runtime/pids/`

## Konfiguration
- Primär über `config.yaml`
- Laden/Speichern über `ConfigManager.py`
- Modell ist fest auf `yolo26n.pt` ausgelegt (abweichende Modellnamen werden abgewiesen)

### Videoquellen-Modi
- `youtube`: Stream über YouTube-URL (Auflösung via `yt-dlp`, optional Cookie-Datei)
- `livefeed_simulation`: lokaler Simulations-Feed über Video/Clips
- Kein Kamera-Fallback: Bei Ausfall der primären Quelle wird nicht auf `/dev/video*` gewechselt

## Startpfade
- Primär: `./bildauswertung/start_system.sh start`
- Direkt: `python3 bildauswertung/run_live_detection.py --config bildauswertung/config.yaml`
- Dashboard: `./bildauswertung/realtime/start_dashboard.sh start`

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
