# Bildauswertung Architektur

## Zweck
`bildauswertung` kapselt die komplette Video-/YOLO-basierte Erkennung, Tracking- und Ereignislogik.

## Pipeline
1. Video Ingest: `VideoInputModule.py`
2. Detektion/Tracking: `UltralyticsPersonDetector.py`, `YOLOTrackingModule.py`
3. Ereignislogik: `TrajectoryEntryAnalysisModule.py`
4. Zustand: `OccupancyStateModule.py`
5. Visualisierung/Kalibrierung: `VisualizationOutputModule.py`
6. Orchestrierung: `LiveProcessor.py` bzw. `run_live_detection.py`

## Konfiguration
- Primär über `config.yaml`
- Laden/Speichern über `ConfigManager.py`
- Modellpfad standardmäßig: `models/yolo26n.pt`

## Startpfade
- Primär: `./bildauswertung/start_system.sh start`
- Direkt: `python3 bildauswertung/run_live_detection.py --config bildauswertung/config.yaml`

## Persistenz
- Optionales PostgreSQL-Logging über `DatabaseHandler.py`

## Nicht-Ziele
- Keine Forecasting-/XAI-/Recommendation-Logik in diesem Bereich
- Keine Streamlit-Analytics-UI in diesem Bereich
