# Sitcheck Dependency Map (Phase 1)

## 1. Bildauswertung Internal Graph
Core module dependencies:
- `run_live_detection.py` -> `ConfigManager`, `LiveProcessor`, `EntranceZoneConfig`, `UltralyticsPersonDetector`
- `LiveProcessor.py` -> `VideoInputModule`, `YOLOTrackingModule`, `TrajectoryEntryAnalysisModule`, `OccupancyStateModule`, `VisualizationOutputModule`, optional `DatabaseHandler`
- `YOLOTrackingModule.py` -> detector abstraction + OpenCV preprocessing/stabilization
- `UltralyticsPersonDetector.py` -> `BaseDetector` + Ultralytics `YOLO`
- `VisualizationOutputModule.py` -> `EntranceZoneConfig` for overlay + zone editing

## 2. Website/Realtime Graph
- `website-dashboard/realtime/dashboard_app.py` imports tracking modules from `bildauswertung`.
- Config source for realtime dashboard: `bildauswertung/config.yaml`.
- DASH output target: `website-dashboard/runtime/dash`.

## 3. Prognose Boundary
`/sitcheck/prognose` is a separate project/repo with its own:
- app layer (`apps/api-gateway`, `apps/dashboard`, `apps/mcp-sitcheck`)
- services (`forecast`, `xai`, `recommendations`, ingest services)
- schemas (`packages/shared/schemas`)
- infra (`docker-compose.yml`, DB migrations)

## 4. Cross-Area Coupling Rules
Current intentional couplings:
- `website-dashboard/realtime` -> `bildauswertung` (shared runtime detection stack)
- `bildauswertung` and `prognose` are logically complementary but code-decoupled.
- Root wrappers (`start_system.sh`, `start_dashboard.sh`) delegate to new area paths.

## 5. Public Runtime Interfaces
Unchanged HTTP routes (realtime dashboard):
- `/`
- `/api/state`
- `/api/zone`
- `/api/tracking-roi`
- `/health`
- `/dash/*`

Prognose APIs remain in `/sitcheck/prognose` (e.g. `/api/v1/forecast`, `/api/v1/dashboard/command-center`).
