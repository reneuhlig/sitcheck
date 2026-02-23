# Library Occupancy Tracking (YOLO Track)

Config-driven real-time people tracking and occupancy counting for a library entrance (entry + exit capable).

## Core flow
- Video source abstraction (`VideoInputModule.py`) with YouTube/RTSP/webcam/file support
- Ultralytics tracking via `model.track(...)` (person class only)
- Trajectory-based transition detection (`line`, `polygon`, or `dual_polygon` with entry+exit areas)
- Event-based occupancy management
- Optional PostgreSQL persistence for webpage + analysis

## Run
```bash
python3 run_live_detection.py --config config.yaml
```

Default tracking runtime is configured for CPU with YOLO26:
- `tracking.model_path: yolo26n.pt`
- `tracking.device: cpu`

Für bessere Erkennung von weiter entfernten Personen und weniger Box-Flimmern:
- `tracking.imgsz` erhöhen (z. B. `640`, `768`, `960`, `1280`; meist Vielfache von 32 sinnvoll)
- `preprocess.enabled: true` mit leichtem `preprocess.upscale` (z. B. `1.2-1.4`)
- `tracking.stabilization_enabled: true` für temporale Glättung + kurze Hold-Phase
- `tracking.track_hold_frames`, `tracking.box_ema_alpha`, `tracking.hold_confidence_decay` feinjustieren
- `tracking.trail_length` und `tracking.motion_min_pixels` für Bewegungsrichtung-Overlay abstimmen
- optional `tracking.tta_enabled: true` nur bei genügend CPU/GPU-Reserve

Dashboard ROI-Cropping (spart Rechenzeit und reduziert Störbereiche):
- Im Dashboard unter **Analysis ROI** den Analysebereich (`x_min`, `y_min`, `x_max`, `y_max`) setzen
- Mit **Save ROI** speichern; Werte werden in `config.yaml` unter `tracking.analysis_roi` persistiert
- Nur dieser Bildausschnitt wird von Ultralytics analysiert, Overlays/Zonen bleiben im Vollbild sichtbar

Tracker-Auswahl nach Ultralytics `track`-Doku:
- `tracking.tracker: bytetrack_entrance.yaml` (hier Standard) ist robust bei niedrigen Confidence-Werten und kurzen Aussetzern
- `tracking.tracker: botsort.yaml` kann bei stärkeren Okklusionen helfen (optional mit ReID im Tracker-YAML)
- Für eigene Profile: YAML aus Ultralytics-Trackern kopieren und nur Parameter (nicht `tracker_type`) anpassen

Production start wrapper uses its own runtime venv automatically:
```bash
./start_system.sh start
```

Optional overrides:
- `SITCHECK_RUNTIME_VENV=/path/to/venv` to change managed runtime venv location
- `SITCHECK_PYTHON=/path/to/python` to use a fixed interpreter directly

## Live calibration tool
With window enabled (`ui.show_window: true`), you can calibrate zone live:
- `L` line mode
- `P` polygon mode
- `D` toggle line direction
- left click set/add points
- right click remove polygon point
- `S` save
- `Q`/`ESC` quit

Zone changes are persisted immediately into `config.yaml`.

## Web dashboard (headless server)
Use browser dashboard for live stream + zone editing on servers without GUI:

```bash
./start_dashboard.sh start
./start_dashboard.sh status
```

Default URL: `http://<server-ip>:8080`

Das Dashboard nutzt jetzt primär HLS (HTML5-Video + `hls.js`) mit Server-seitigen Segmenten.
Voraussetzung: `ffmpeg` (System) oder das Python-Paket `imageio-ffmpeg` (wird über `start_dashboard.sh` automatisch installiert).

Für flüssiges HLS-Web-Streaming kannst du in `config.yaml` unter `dashboard` anpassen:
- `stream_fps` (Ziel-Ausgabe-FPS im Browser)
- `jpeg_quality` (40-95, niedriger = schneller/kleinere Frames)
- `stream_max_width` (maximale Stream-Breite in Pixeln; kleiner = deutlich weniger Bandbreite)
- `analysis_queue_frames` (Frame-Puffer für Ultralytics-Analyse)
- `analysis_skip_threshold_frames` (0 = keine Analyse-Skips, >0 = bei Überlast ältere Analyse-Frames verwerfen)
- `hls.enabled` (HLS-Generierung an/aus)
- `hls.output_dir` (Segment-/Playlist-Ordner)
- `hls.segment_time` (Segmentlänge in Sekunden)
- `hls.list_size` (Anzahl Segmente in Live-Playlist)

Hinweis: Das Dashboard verwendet jetzt ausschließlich den HLS-Pfad (kein MJPEG-/Packet-Fallback mehr).

Effizienter Internet-Preset (ähnlich „adaptive light“):
- `stream_fps: 20`
- `jpeg_quality: 65`
- `stream_max_width: 960`

- `dual_polygon` mode supports separate `entry_polygon` and `exit_polygon`
- Occupancy increases on outside→inside crossing of `entry_polygon`
- Occupancy decreases on outside→inside crossing of `exit_polygon`
- Counting includes cooldown/displacement filters to reduce false toggles

## Webpage data functions
`DatabaseHandler.py` provides:
- `get_occupancy_snapshot()`
- `get_occupancy_timeseries(minutes)`
- `get_entry_summary(minutes)`
- `get_recent_entry_events(limit)`

For full architecture, assumptions, and business rules, see `Agent.md`.
