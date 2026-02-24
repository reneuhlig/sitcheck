# Agent.md

## 1) Project Overview

### Purpose
This project performs **real-time people tracking and occupancy counting** for a library entrance using **Ultralytics YOLO tracking mode** (`model.track(...)`).

### Business Goal
Answer one operational KPI reliably:
- **How many people are currently inside the library?**

Counting is event-based and must avoid false positives from pedestrians passing by the entrance.

---

## 2) Camera Context & Assumptions

- Camera is outside, facing the entrance area.
- Scene contains both entering people and pass-by traffic.
- Not every person in frame is an entry.
- A person counts as entry only when trajectory confirms entry-direction crossing of configured zone.

---

## 3) Runtime Architecture

### Modules
1. **Video source handling** → `VideoInputModule.py`
2. **YOLO tracking** → `UltralyticsPersonDetector.py`, `YOLOTrackingModule.py`
3. **Trajectory & entry analysis** → `TrajectoryEntryAnalysisModule.py`
4. **Occupancy state** → `OccupancyStateModule.py`
5. **Visualization + interactive editor** → `VisualizationOutputModule.py`
6. **Orchestration** → `LiveProcessor.py`
7. **Configuration layer** → `ConfigManager.py`, `config.yaml`
8. **Persistence/analytics** → `DatabaseHandler.py` (optional)

### Current execution model (important)
- `dashboard_app.py` contains the full live pipeline for web usage:
  video ingest + YOLO tracking + trajectory analysis + overlay rendering + DASH output.
- `run_live_detection.py` / `start_system.sh` is a standalone local/CLI runtime.
- To avoid duplicate inference load and lag, do **not** run `start_dashboard.sh` and `start_system.sh` at the same time on the same source.

### Separation Principles
- Video source is abstracted and replaceable via configuration.
- Tracking and trajectory logic are isolated from UI.
- UI edits zone definition and persists configuration, but does not contain counting logic.
- Occupancy is event-based and independent from number of active tracks in current frame.

---

## 4) Video Source Abstraction

### Initial development source
Default source in `config.yaml`:
- `https://www.youtube.com/watch?v=8JCk5M_xrBs`

### Supported source types
- YouTube URL (resolved via `yt-dlp` to stream URL)
- RTSP stream
- webcam/USB index (`0`, `1`, ...)
- local video file path

### Configuration inputs
- YAML (`config.yaml`) is primary.
- ENV overrides supported (e.g. `SITCHECK_VIDEO_SOURCE`, `SITCHECK_TRACKER`, `SITCHECK_DB_ENABLED`, ...).
- Optional CLI overrides in `run_live_detection.py`.

---

## 5) YOLO Tracking Requirements (Implemented)

- Uses official Ultralytics API: `model.track(...)`
- `persist=True` for stable track IDs
- Tracks person class only (`classes=[0]`)
- Uses supported tracker config (`botsort.yaml` by default, configurable)
- No external tracking framework used

---

## 6) Entry Zone Logic

## Zone Modes
- **Line mode**
  - Two normalized points (`line.p1`, `line.p2`)
  - Direction filter (`negative_to_positive` or `positive_to_negative`)
- **Polygon mode**
  - Normalized polygon points
  - Entry defined as **outside → inside** transition

## Trajectory-based Entry Decision
Per `track_id`:
1. Build center-point trajectory history over multiple frames.
2. Require minimum history length (`min_track_points`).
3. Require zone crossing behavior:
   - line: side-change across line
   - polygon: outside→inside
4. Require minimum displacement (`min_crossing_displacement_px`) to reduce jitter-triggered false counts.
5. Count each `track_id` once per entry event.

## Pass-by Suppression
- People who never cross zone are ignored.
- Wrong crossing direction in line mode is ignored.
- Near-border jitter without enough displacement is ignored.

---

## 7) Interactive Live Visualization & Calibration Tool

Provided by `VisualizationOutputModule.py`:
- Live feed rendering (minimal latency via OpenCV loop)
- Bounding boxes + persistent track IDs
- Occupancy and entry counters
- Live editable zone geometry via mouse and keyboard

### Controls
- `L`: switch to line mode
- `P`: switch to polygon mode
- `D`: toggle line entry direction
- Left click:
  - line mode: set p1 then p2
  - polygon mode: append point
- Right click (polygon mode): remove last point
- `S`: save current zone (also auto-saved on changes)
- `Q` or `ESC`: quit

### Persistence behavior
Zone edits are immediately written to `config.yaml` through `ConfigManager.update_zone(...)`, so calibration persists across restarts.

---

## 8) Occupancy Logic

- Maintains current occupancy as numeric state (`OccupancyStateModule.current_occupancy`).
- Increments only on confirmed entry events.
- Uses track-id deduplication (`tracking_events` unique constraint and in-memory set).
- Does not infer occupancy from active detections in frame.
- Exit decrement is not enabled in current baseline.

---

## 9) Database Data for Webpage & Analysis

Optional DB integration (`database.enabled: true`) writes:
- `tracking_events` (entry events per track)
- `movement_tracking` (event log)
- `room_state` (occupancy timeline)

Minimal query surface for webpage/business analytics in `DatabaseHandler.py`:
- `get_occupancy_snapshot()`
- `get_occupancy_timeseries(minutes)`
- `get_entry_summary(minutes)`
- `get_recent_entry_events(limit)`

---

## 10) Runtime Usage

## Standalone tracker (without web dashboard)
```bash
python3 run_live_detection.py --config config.yaml
```

## Override source quickly
```bash
python3 run_live_detection.py --config config.yaml --video-source 0
python3 run_live_detection.py --config config.yaml --video-source "rtsp://..."
python3 run_live_detection.py --config config.yaml --video-source "video.mp4"
```

## Service helper (standalone mode)
```bash
./start_system.sh start
./start_system.sh status
./start_system.sh room
./start_system.sh logs
./start_system.sh stop
```

## Recommended production pipeline (web + tracking + DASH in one process)
```bash
./start_dashboard.sh start
./start_dashboard.sh status
./start_dashboard.sh logs
./start_dashboard.sh stop
```

Default web URL:
- `http://<server-ip>:8080`

---

## 11) Known Limitations

- Very dense crowding may still cause track ID switches.
- YouTube stream stability depends on network and source availability.
- Exit logic is not part of current baseline; occupancy only increases on entries.
- Accurate operation requires camera-specific zone calibration.

---

## 12) Non-Trivial Logic Notes

- Line crossing uses signed distance/side change + displacement gating.
- Polygon mode uses point-in-polygon transition (outside→inside).
- Zone editor writes updates immediately to config to keep runtime and persisted state consistent.
- Source abstraction resolves YouTube URLs with `yt-dlp` but remains replaceable via config/env.
