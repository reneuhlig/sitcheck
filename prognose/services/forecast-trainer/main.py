from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_horizons(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError(f"invalid horizon '{value}', expected >= 1")
        values.append(value)
    if not values:
        raise ValueError("no horizons configured")
    return sorted(set(values))


def _parse_hhmm(raw: str) -> tuple[int, int]:
    token = (raw or "").strip()
    if ":" not in token:
        raise ValueError("nightly time must match HH:MM")
    hour_text, minute_text = token.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23:
        raise ValueError("nightly hour must be 0..23")
    if minute < 0 or minute > 59:
        raise ValueError("nightly minute must be 0..59")
    return hour, minute


FORECAST_API_BASE_URL = os.getenv("FORECAST_TRAINER_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
FORECAST_TRAINER_PORT = int(os.getenv("FORECAST_TRAINER_PORT", "8013"))
FORECAST_TRAINER_ENABLED = _env_bool("FORECAST_TRAINER_ENABLED", True)
FORECAST_TRAINER_ZONE_ID = os.getenv("FORECAST_TRAINER_ZONE_ID", "default-zone").strip() or "default-zone"
FORECAST_TRAINER_NIGHTLY_UTC_RAW = os.getenv("FORECAST_TRAINER_NIGHTLY_UTC", "02:15")
FORECAST_TRAINER_HORIZONS_RAW = os.getenv("FORECAST_TRAINER_HORIZONS", "210,1440")
FORECAST_TRAINER_INCLUDE_LECTURE_IMPACT = _env_bool("FORECAST_TRAINER_INCLUDE_LECTURE_IMPACT", True)
FORECAST_TRAINER_RUN_ABLATION = _env_bool("FORECAST_TRAINER_RUN_ABLATION", True)
FORECAST_TRAINER_HISTORY_HOURS = int(os.getenv("FORECAST_TRAINER_HISTORY_HOURS", str(24 * 120)))
FORECAST_TRAINER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("FORECAST_TRAINER_REQUEST_TIMEOUT_SECONDS", "1800"))
FORECAST_TRAINER_PRIMARY_HORIZON = int(os.getenv("FORECAST_TRAINER_PRIMARY_HORIZON", "210"))
FORECAST_TRAINER_IMPROVEMENT_THRESHOLD = float(os.getenv("FORECAST_TRAINER_IMPROVEMENT_THRESHOLD", "0.08"))
FORECAST_TRAINER_STATE_FILE = Path(
    os.getenv("FORECAST_TRAINER_STATE_FILE", "./runtime/forecast_trainer_state.json")
).resolve()

try:
    FORECAST_TRAINER_HORIZONS = _parse_horizons(FORECAST_TRAINER_HORIZONS_RAW)
    NIGHTLY_HOUR_UTC, NIGHTLY_MINUTE_UTC = _parse_hhmm(FORECAST_TRAINER_NIGHTLY_UTC_RAW)
    if FORECAST_TRAINER_PRIMARY_HORIZON not in FORECAST_TRAINER_HORIZONS:
        FORECAST_TRAINER_PRIMARY_HORIZON = FORECAST_TRAINER_HORIZONS[0]
    CONFIG_ERROR: str | None = None
except Exception as exc:
    FORECAST_TRAINER_HORIZONS = [210, 1440]
    NIGHTLY_HOUR_UTC, NIGHTLY_MINUTE_UTC = 2, 15
    if FORECAST_TRAINER_PRIMARY_HORIZON not in FORECAST_TRAINER_HORIZONS:
        FORECAST_TRAINER_PRIMARY_HORIZON = 210
    CONFIG_ERROR = f"invalid trainer config: {exc}"


@dataclass
class TrainerState:
    status: str = "starting"
    scheduler_status: str = "starting"
    last_run_at: str | None = None
    last_success_at: str | None = None
    next_run_at: str | None = None
    total_runs: int = 0
    consecutive_failures: int = 0
    last_error_class: str | None = None
    last_error_message: str | None = None
    last_result: dict[str, Any] | None = None


class TrainerRunError(RuntimeError):
    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


app = FastAPI(title="sitcheck-forecast-trainer", version="0.1.0")
_state = TrainerState()
_state_lock = threading.Lock()
_run_lock = threading.Lock()
_stop_event = threading.Event()
_worker: threading.Thread | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _extract_error_detail(response: requests.Response) -> str:
    text = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail is not None:
                text = str(detail)
            else:
                text = json.dumps(payload)
        else:
            text = str(payload)
    except Exception:
        text = response.text.strip()
    return text[:500] if text else f"http {response.status_code}"


def _classify_http_error(status_code: int, detail: str, stage: str) -> str:
    lowered = detail.lower()
    if status_code in {502, 503, 504}:
        return "api_unreachable"
    if status_code == 400 and any(
        token in lowered for token in ("no history", "insufficient history", "no history left", "insufficient")
    ):
        return "insufficient_history"
    if stage == "ablation":
        return "ablation_failed"
    return "evaluation_failed"


def _next_nightly_run(now: datetime) -> datetime:
    candidate = now.replace(hour=NIGHTLY_HOUR_UTC, minute=NIGHTLY_MINUTE_UTC, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


def _read_report_metric(report: dict[str, Any]) -> dict[str, Any]:
    comparison = report.get("comparison", {}) if isinstance(report, dict) else {}
    decision = report.get("decision", {}) if isinstance(report, dict) else {}
    models = report.get("models", {}) if isinstance(report, dict) else {}

    primary_horizon = int(comparison.get("primary_horizon") or FORECAST_TRAINER_PRIMARY_HORIZON)
    champion_model = str(decision.get("champion_model") or "")
    horizon_payload = (
        models.get(champion_model, {})
        .get("horizons", {})
        .get(str(primary_horizon), {})
    )
    metrics = horizon_payload.get("test_metrics", {}) if isinstance(horizon_payload, dict) else {}
    improvement_map = comparison.get("improvement_vs_baseline_mae", {})
    improvement = None
    if isinstance(improvement_map, dict):
        improvement = _safe_float(improvement_map.get(champion_model))

    return {
        "run_id": str(report.get("run_id") or ""),
        "primary_horizon": primary_horizon,
        "scientific_pass": bool(decision.get("scientific_pass", False)),
        "champion_model": champion_model,
        "mae": _safe_float(metrics.get("mae")),
        "pinball": _safe_float(metrics.get("pinball")),
        "coverage90": _safe_float(metrics.get("coverage90")),
        "improvement_vs_baseline_mae": improvement,
    }


def _build_ablation_summary(
    with_report: dict[str, Any],
    without_report: dict[str, Any],
) -> dict[str, Any]:
    with_metrics = _read_report_metric(with_report)
    without_metrics = _read_report_metric(without_report)

    mae_gain = None
    if with_metrics["mae"] is not None and without_metrics["mae"] is not None:
        mae_gain = float(without_metrics["mae"] - with_metrics["mae"])

    pinball_gain = None
    if with_metrics["pinball"] is not None and without_metrics["pinball"] is not None:
        pinball_gain = float(without_metrics["pinball"] - with_metrics["pinball"])

    coverage_delta = None
    if with_metrics["coverage90"] is not None and without_metrics["coverage90"] is not None:
        coverage_delta = float(with_metrics["coverage90"] - without_metrics["coverage90"])

    return {
        "with_lecture": with_metrics,
        "without_lecture": without_metrics,
        "mae_gain_primary_horizon": mae_gain,
        "pinball_gain_primary_horizon": pinball_gain,
        "coverage_delta_primary_horizon": coverage_delta,
    }


def _evaluate_once(include_lecture_impact: bool, stage: str) -> dict[str, Any]:
    payload = {
        "zone_id": FORECAST_TRAINER_ZONE_ID,
        "horizons": FORECAST_TRAINER_HORIZONS,
        "history_hours": FORECAST_TRAINER_HISTORY_HOURS,
        "primary_horizon": FORECAST_TRAINER_PRIMARY_HORIZON,
        "save_report": True,
        "include_lecture_impact": include_lecture_impact,
    }
    url = f"{FORECAST_API_BASE_URL}/v1/train/evaluate"
    try:
        response = requests.post(url, json=payload, timeout=FORECAST_TRAINER_REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise TrainerRunError("api_unreachable", f"{stage} request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        error_class = _classify_http_error(response.status_code, detail, stage=stage)
        raise TrainerRunError(
            error_class,
            f"{stage} returned HTTP {response.status_code}: {detail}",
        )

    try:
        report = response.json()
    except Exception as exc:
        raise TrainerRunError("evaluation_failed", f"{stage} returned invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise TrainerRunError("evaluation_failed", f"{stage} returned invalid payload type")
    return report


def _persist_state() -> None:
    FORECAST_TRAINER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _utc_now_iso(),
        "state": asdict(_state),
        "config": {
            "enabled": FORECAST_TRAINER_ENABLED,
            "zone_id": FORECAST_TRAINER_ZONE_ID,
            "nightly_utc": FORECAST_TRAINER_NIGHTLY_UTC_RAW,
            "horizons": FORECAST_TRAINER_HORIZONS,
            "primary_horizon": FORECAST_TRAINER_PRIMARY_HORIZON,
            "include_lecture_impact": FORECAST_TRAINER_INCLUDE_LECTURE_IMPACT,
            "run_ablation": FORECAST_TRAINER_RUN_ABLATION,
            "api_base_url": FORECAST_API_BASE_URL,
            "history_hours": FORECAST_TRAINER_HISTORY_HOURS,
            "improvement_threshold": FORECAST_TRAINER_IMPROVEMENT_THRESHOLD,
        },
    }
    tmp_path = FORECAST_TRAINER_STATE_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(FORECAST_TRAINER_STATE_FILE)


def _run_evaluation_pair(trigger: str) -> dict[str, Any]:
    started_at = _utc_now_iso()

    primary_include = bool(FORECAST_TRAINER_INCLUDE_LECTURE_IMPACT)
    primary_report = _evaluate_once(include_lecture_impact=primary_include, stage="evaluation")
    result: dict[str, Any] = {
        "trigger": trigger,
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "zone_id": FORECAST_TRAINER_ZONE_ID,
        "horizons": FORECAST_TRAINER_HORIZONS,
        "primary_horizon": FORECAST_TRAINER_PRIMARY_HORIZON,
        "run_ablation": FORECAST_TRAINER_RUN_ABLATION,
        "include_lecture_impact": primary_include,
        "with_lecture_run_id": str(primary_report.get("run_id") or "") if primary_include else None,
        "without_lecture_run_id": None,
        "ablation": None,
    }
    if not primary_include:
        result["without_lecture_run_id"] = str(primary_report.get("run_id") or "")

    if FORECAST_TRAINER_RUN_ABLATION:
        secondary_report = _evaluate_once(include_lecture_impact=not primary_include, stage="ablation")
        if primary_include:
            with_report = primary_report
            without_report = secondary_report
        else:
            with_report = secondary_report
            without_report = primary_report

        result["with_lecture_run_id"] = str(with_report.get("run_id") or "")
        result["without_lecture_run_id"] = str(without_report.get("run_id") or "")
        result["ablation"] = _build_ablation_summary(with_report=with_report, without_report=without_report)

    result["finished_at"] = _utc_now_iso()
    return result


def _update_next_run(now: datetime) -> None:
    with _state_lock:
        _state.next_run_at = _next_nightly_run(now).isoformat()


def _execute_run(trigger: str) -> dict[str, Any]:
    if not _run_lock.acquire(blocking=False):
        raise TrainerRunError("evaluation_failed", "trainer run already in progress")

    try:
        with _state_lock:
            _state.scheduler_status = "running"
            _state.last_run_at = _utc_now_iso()
            _state.status = "running"

        result = _run_evaluation_pair(trigger=trigger)
        with _state_lock:
            _state.total_runs += 1
            _state.last_success_at = _utc_now_iso()
            _state.consecutive_failures = 0
            _state.last_error_class = None
            _state.last_error_message = None
            _state.last_result = result
            _state.status = "running"
            _state.scheduler_status = "idle"
        _persist_state()
        return result
    except TrainerRunError as exc:
        with _state_lock:
            _state.total_runs += 1
            _state.consecutive_failures += 1
            _state.last_error_class = exc.error_class
            _state.last_error_message = str(exc)
            _state.status = "degraded"
            _state.scheduler_status = "idle"
        _persist_state()
        raise
    except Exception as exc:  # pragma: no cover
        with _state_lock:
            _state.total_runs += 1
            _state.consecutive_failures += 1
            _state.last_error_class = "evaluation_failed"
            _state.last_error_message = str(exc)
            _state.status = "degraded"
            _state.scheduler_status = "idle"
        _persist_state()
        raise TrainerRunError("evaluation_failed", str(exc)) from exc
    finally:
        _run_lock.release()


def _worker_loop() -> None:
    if not FORECAST_TRAINER_ENABLED:
        with _state_lock:
            _state.status = "disabled"
            _state.scheduler_status = "disabled"
        _persist_state()
        return

    if CONFIG_ERROR:
        with _state_lock:
            _state.status = "error"
            _state.scheduler_status = "error"
            _state.last_error_class = "config_error"
            _state.last_error_message = CONFIG_ERROR
        _persist_state()
        return

    with _state_lock:
        _state.status = "running"
        _state.scheduler_status = "idle"
    _persist_state()

    while not _stop_event.is_set():
        now = datetime.now(UTC)
        next_run = _next_nightly_run(now)
        with _state_lock:
            _state.next_run_at = next_run.isoformat()
        _persist_state()

        while not _stop_event.is_set():
            now = datetime.now(UTC)
            wait_seconds = (next_run - now).total_seconds()
            if wait_seconds <= 0:
                break
            _stop_event.wait(min(60.0, max(1.0, wait_seconds)))

        if _stop_event.is_set():
            break

        try:
            _execute_run(trigger="nightly")
        except TrainerRunError:
            continue

    with _state_lock:
        if _state.status not in {"error", "disabled"}:
            _state.scheduler_status = "stopped"
    _persist_state()


@app.on_event("startup")
def on_startup() -> None:
    global _worker
    _stop_event.clear()
    _update_next_run(datetime.now(UTC))
    _worker = threading.Thread(target=_worker_loop, daemon=True, name="forecast-trainer")
    _worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    if _worker is not None:
        _worker.join(timeout=5)


@app.get("/health")
def health() -> Any:
    with _state_lock:
        payload = {
            "status": "ok" if _state.status in {"starting", "running", "degraded", "disabled"} else "error",
            "service": "forecast-trainer",
            "trainer_status": _state.status,
            "scheduler_status": _state.scheduler_status,
            "zone_id": FORECAST_TRAINER_ZONE_ID,
            "horizons": FORECAST_TRAINER_HORIZONS,
            "primary_horizon": FORECAST_TRAINER_PRIMARY_HORIZON,
            "nightly_utc": FORECAST_TRAINER_NIGHTLY_UTC_RAW,
            "last_run_at": _state.last_run_at,
            "last_success_at": _state.last_success_at,
            "next_run_at": _state.next_run_at,
            "total_runs": _state.total_runs,
            "consecutive_failures": _state.consecutive_failures,
            "last_error_class": _state.last_error_class,
            "last_error_message": _state.last_error_message,
            "last_result": _state.last_result,
        }
    status_code = 200 if payload["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/status")
def status() -> Any:
    with _state_lock:
        payload = {
            "service": "forecast-trainer",
            "state": asdict(_state),
            "config": {
                "api_base_url": FORECAST_API_BASE_URL,
                "enabled": FORECAST_TRAINER_ENABLED,
                "zone_id": FORECAST_TRAINER_ZONE_ID,
                "nightly_utc": FORECAST_TRAINER_NIGHTLY_UTC_RAW,
                "horizons": FORECAST_TRAINER_HORIZONS,
                "primary_horizon": FORECAST_TRAINER_PRIMARY_HORIZON,
                "include_lecture_impact": FORECAST_TRAINER_INCLUDE_LECTURE_IMPACT,
                "run_ablation": FORECAST_TRAINER_RUN_ABLATION,
                "history_hours": FORECAST_TRAINER_HISTORY_HOURS,
                "request_timeout_seconds": FORECAST_TRAINER_REQUEST_TIMEOUT_SECONDS,
                "state_file": str(FORECAST_TRAINER_STATE_FILE),
                "config_error": CONFIG_ERROR,
                "improvement_threshold": FORECAST_TRAINER_IMPROVEMENT_THRESHOLD,
            },
        }
    return payload


@app.post("/run-once")
def run_once() -> Any:
    if not FORECAST_TRAINER_ENABLED:
        raise HTTPException(status_code=409, detail="forecast trainer is disabled")
    if CONFIG_ERROR:
        raise HTTPException(status_code=500, detail=CONFIG_ERROR)
    try:
        result = _execute_run(trigger="manual")
        return {
            "status": "ok",
            "message": "manual evaluation run completed",
            "result": result,
        }
    except TrainerRunError as exc:
        raise HTTPException(status_code=503, detail={"error_class": exc.error_class, "message": str(exc)}) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=FORECAST_TRAINER_PORT)
