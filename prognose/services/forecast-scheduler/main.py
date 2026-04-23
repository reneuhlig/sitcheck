from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse


API_BASE_URL = os.getenv("API_BASE_URL", "http://api-gateway:8000").rstrip("/")
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
FORECAST_SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("FORECAST_SNAPSHOT_INTERVAL_SECONDS", "300"))
FORECAST_SNAPSHOT_HORIZONS = os.getenv("FORECAST_SNAPSHOT_HORIZONS", "210")
FORECAST_SNAPSHOT_ZONES = os.getenv("FORECAST_SNAPSHOT_ZONES", "auto")
FORECAST_SNAPSHOT_MAX_HORIZON_MINUTES = int(os.getenv("FORECAST_SNAPSHOT_MAX_HORIZON_MINUTES", "43200"))
SCHEDULER_PORT = int(os.getenv("SCHEDULER_PORT", "8011"))


@dataclass
class SchedulerState:
    status: str = "starting"
    last_run_at: str | None = None
    last_success_at: str | None = None
    consecutive_failures: int = 0
    total_snapshots: int = 0
    last_error: str | None = None


app = FastAPI(title="sitcheck-forecast-scheduler", version="0.1.0")
_state = SchedulerState()
_lock = threading.Lock()
_stop_event = threading.Event()
_worker: threading.Thread | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_horizons(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 1 or value > FORECAST_SNAPSHOT_MAX_HORIZON_MINUTES:
            raise ValueError(
                f"invalid horizon {value}; expected 1..{FORECAST_SNAPSHOT_MAX_HORIZON_MINUTES}"
            )
        values.append(value)
    if not values:
        raise ValueError("FORECAST_SNAPSHOT_HORIZONS must define at least one horizon")
    return values


def _resolve_zones() -> list[str]:
    if FORECAST_SNAPSHOT_ZONES.strip().lower() != "auto":
        zones = [z.strip() for z in FORECAST_SNAPSHOT_ZONES.split(",") if z.strip()]
        if not zones:
            raise ValueError("FORECAST_SNAPSHOT_ZONES is empty")
        return zones

    response = requests.get(f"{API_BASE_URL}/api/v1/zones", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("/api/v1/zones did not return a list")

    zones = [str(item.get("zone_id", "")).strip() for item in payload]
    zones = [zone for zone in zones if zone]
    if not zones:
        raise ValueError("auto zone resolution returned no zone_ids")
    return zones


def _snapshot(zone_id: str, horizon: int) -> None:
    headers = {"X-Internal-Token": INTERNAL_API_TOKEN}
    payload = {"zone_id": zone_id, "horizon": horizon}
    response = requests.post(
        f"{API_BASE_URL}/api/v1/internal/forecast/snapshot",
        json=payload,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()


def _snapshot_with_retry(zone_id: str, horizon: int) -> None:
    attempts = 3
    backoff = 1.0
    for idx in range(attempts):
        try:
            _snapshot(zone_id=zone_id, horizon=horizon)
            return
        except Exception:
            if idx == attempts - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


def _run_once(horizons: list[int]) -> int:
    zones = _resolve_zones()
    processed = 0
    for zone_id in zones:
        for horizon in horizons:
            _snapshot_with_retry(zone_id=zone_id, horizon=horizon)
            processed += 1
    return processed


def _worker_loop() -> None:
    if not INTERNAL_API_TOKEN:
        with _lock:
            _state.status = "error"
            _state.last_error = "INTERNAL_API_TOKEN not configured"
            _state.consecutive_failures += 1
        return

    try:
        horizons = _parse_horizons(FORECAST_SNAPSHOT_HORIZONS)
    except Exception as exc:
        with _lock:
            _state.status = "error"
            _state.last_error = f"invalid horizon config: {exc}"
            _state.consecutive_failures += 1
        return

    with _lock:
        _state.status = "running"

    while not _stop_event.is_set():
        run_ts = _utc_now_iso()
        try:
            processed = _run_once(horizons=horizons)
            with _lock:
                _state.last_run_at = run_ts
                _state.last_success_at = _utc_now_iso()
                _state.consecutive_failures = 0
                _state.total_snapshots += processed
                _state.last_error = None
                _state.status = "running"
        except Exception as exc:
            with _lock:
                _state.last_run_at = run_ts
                _state.consecutive_failures += 1
                _state.last_error = str(exc)
                _state.status = "degraded"

        _stop_event.wait(FORECAST_SNAPSHOT_INTERVAL_SECONDS)


@app.on_event("startup")
def on_startup() -> None:
    global _worker
    _stop_event.clear()
    _worker = threading.Thread(target=_worker_loop, daemon=True, name="forecast-scheduler")
    _worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    if _worker is not None:
        _worker.join(timeout=5)


@app.get("/health")
def health() -> Any:
    with _lock:
        payload = {
            "status": "ok" if _state.status in {"running", "starting"} else "error",
            "service": "forecast-scheduler",
            "scheduler_status": _state.status,
            "last_run_at": _state.last_run_at,
            "last_success_at": _state.last_success_at,
            "consecutive_failures": _state.consecutive_failures,
            "total_snapshots": _state.total_snapshots,
            "interval_seconds": FORECAST_SNAPSHOT_INTERVAL_SECONDS,
            "zones_mode": FORECAST_SNAPSHOT_ZONES,
            "horizons": FORECAST_SNAPSHOT_HORIZONS,
            "last_error": _state.last_error,
        }
    status_code = 200 if payload["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SCHEDULER_PORT)
