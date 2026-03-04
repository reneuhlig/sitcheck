# Sitcheck Inventory (Phase 1)

## Scope
Inventory of `/sitcheck` with artifact excludes for analysis clarity.

Excluded for this overview:
- virtual environments: `.venv`, `prognose/.venv`, `prognose/excel_forecasting/.venv`
- package caches: `node_modules`
- bytecode: `__pycache__`
- runtime stream/output: `website-dashboard/runtime`
- VCS internals: `.git`
- generated DB files and model binaries are not used for complexity scoring

## Top-Level Structure
- `/sitcheck/bildauswertung`
- `/sitcheck/website-dashboard`
- `/sitcheck/prognose` (own Git repo)
- `/sitcheck/docs`

## File Count (analysis scope)
- Files in scope: `154`

## Size/Complexity Snapshot (LOC)
Largest code/config files in scope:
- `1423` lines: `bildauswertung/realtime/dashboard_app.py`
- `1198` lines: `prognose/apps/api-gateway/main.py`
- `749` lines: `prognose/services/forecast/main.py`
- `462` lines: `prognose/services/lecture-ingest/main.py`
- `460` lines: `prognose/apps/dashboard/explainability.py`
- `432` lines: `bildauswertung/TrajectoryEntryAnalysisModule.py`
- `395` lines: `prognose/apps/mcp-sitcheck/tools.mjs`
- `337` lines: `bildauswertung/DatabaseHandler.py`
- `331` lines: `bildauswertung/VisualizationOutputModule.py`
- `224` lines: `bildauswertung/YOLOTrackingModule.py`

## Entrypoints (`if __name__ == "__main__"`)
Primary runtime entrypoints:
- `bildauswertung/run_live_detection.py`
- `bildauswertung/realtime/dashboard_app.py`

Additional service/test entrypoints live under:
- `prognose/services/*`
- `prognose/scripts/*`
- `prognose/excel_forecasting/scripts/*`

## Observations
- Realtime pipeline and tracking logic are now physically separated from dashboard runtime orchestration.
- `prognose` remains structurally independent and service-oriented.
- Heavy runtime artifacts are isolated under `website-dashboard/runtime`.
