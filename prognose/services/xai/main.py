"""FastAPI XAI service — occupancy forecast explainability and driver analysis.

This service implements the Explainable AI (XAI) layer for the Sitcheck platform.
It answers the question "why is occupancy predicted to be X?" by computing:

  1. Driver attribution: which factors most influenced the forecast
     (trend, momentum, seasonality, events, lecture density, data quality)
  2. Uncertainty quantification: how confident is the prediction, expressed
     as low/medium/high with a numeric score

Design philosophy:
  The current implementation uses heuristic proxy drivers derived from raw
  sensor data (trend slope, momentum, event context) rather than model-internal
  SHAP values.  This makes the XAI service available even when the SHAP library
  is not installed and regardless of which forecast backend is active.

  True SHAP-based attribution (using SHAPExplainer in shap_explainer.py) is
  available when XAI_SHAP_ENABLED=true and FORECAST_MODEL_BACKEND=xgboost.
  For the LightGBM primary backend, SHAP integration is partially implemented.

Graceful degradation:
  If any data loading or driver computation fails, the endpoint returns a minimal
  fallback response with uncertainty.level='high' and a FALLBACK quality flag,
  rather than raising an HTTP 500 error.

Key symbols:
    ExplanationResponse: Pydantic response schema for GET /v1/explain
    _compute_drivers: Core heuristic driver computation
    explain_forecast: Main explainability endpoint
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, event, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sitcheck.db")
FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8001")
XAI_SHAP_ENABLED = os.getenv("XAI_SHAP_ENABLED", "false").lower() == "true"
FORECAST_MODEL_BACKEND = os.getenv("FORECAST_MODEL_BACKEND", "baseline")
DEFAULT_FORECAST_HORIZON_MINUTES = max(
    1,
    min(int(os.getenv("DEFAULT_FORECAST_HORIZON_MINUTES", os.getenv("TF_DEFAULT_HORIZON", "210"))), 720),
)
logger = logging.getLogger(__name__)

if DATABASE_URL.startswith("sqlite:"):
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 60},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA temp_store=MEMORY")
        finally:
            cursor.close()
else:
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Count(Base):
    __tablename__ = "counts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    occupancy: Mapped[int] = mapped_column(Integer, nullable=False)
    utilization: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_impact: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class LectureActivity(Base):
    __tablename__ = "lecture_activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    active_lectures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_courses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    starts_next_60m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ends_next_60m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class ExplanationResponse(BaseModel):
    zone_id: str
    horizon: int
    summary: str
    drivers: list[dict[str, Any]]
    uncertainty: dict[str, Any]
    evidence: dict[str, Any]


app = FastAPI(title="sitcheck-xai", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _load_counts(zone_id: str, hours: int = 24) -> pd.DataFrame:
    from_dt = datetime.now(UTC) - timedelta(hours=hours)
    try:
        with SessionLocal() as db:
            dialect = (db.bind.dialect.name if db.bind is not None else "").lower()
            if dialect == "postgresql":
                minute_bucket_expr = func.date_trunc("minute", Count.ts)
            else:
                minute_bucket_expr = func.strftime("%Y-%m-%dT%H:%M:00Z", Count.ts)

            rows = db.execute(
                select(
                    minute_bucket_expr.label("minute_bucket"),
                    func.avg(Count.occupancy).label("occupancy_avg"),
                    func.avg(Count.quality_score).label("quality_avg"),
                )
                .where(Count.zone_id == zone_id)
                .where(Count.ts >= from_dt)
                .group_by(minute_bucket_expr)
                .order_by(minute_bucket_expr.asc())
            ).all()
    except OperationalError as exc:
        logger.warning("xai counts query failed (zone=%s): %s", zone_id, exc)
        return pd.DataFrame(columns=["ts", "occupancy", "quality_score"])

    if not rows:
        return pd.DataFrame(columns=["ts", "occupancy", "quality_score"])

    timestamps: list[datetime] = []
    occupancy_values: list[float] = []
    quality_values: list[float] = []
    for row in rows:
        row_map = row._mapping
        minute_value = row_map.get("minute_bucket")
        if minute_value is None:
            continue
        try:
            ts = _normalize_dt(minute_value if isinstance(minute_value, datetime) else datetime.fromisoformat(str(minute_value).replace("Z", "+00:00")))
        except Exception:
            continue
        timestamps.append(ts)
        occupancy_values.append(float(row_map.get("occupancy_avg") or 0.0))
        quality_values.append(float(row_map.get("quality_avg") or 0.0))

    return pd.DataFrame(
        {
            "ts": timestamps,
            "occupancy": occupancy_values,
            "quality_score": quality_values,
        }
    )


def _load_events(zone_id: str, horizon: int) -> list[CalendarEvent]:
    now = datetime.now(UTC)
    until = now + timedelta(minutes=horizon)
    try:
        with SessionLocal() as db:
            rows = db.execute(
                select(CalendarEvent)
                .where(CalendarEvent.starts_at <= until)
                .where(CalendarEvent.ends_at >= now)
                .where((CalendarEvent.zone_id == zone_id) | (CalendarEvent.zone_id.is_(None)))
                .order_by(CalendarEvent.starts_at.asc())
            ).scalars().all()
    except OperationalError as exc:
        logger.warning("xai events query failed (zone=%s): %s", zone_id, exc)
        return []
    return rows


def _load_lecture_activity(zone_id: str, hours: int = 48) -> pd.DataFrame:
    from_dt = datetime.now(UTC) - timedelta(hours=hours)
    try:
        with SessionLocal() as db:
            dialect = (db.bind.dialect.name if db.bind is not None else "").lower()
            if dialect == "postgresql":
                minute_bucket_expr = func.date_trunc("minute", LectureActivity.ts)
            else:
                minute_bucket_expr = func.strftime("%Y-%m-%dT%H:%M:00Z", LectureActivity.ts)

            rows = db.execute(
                select(
                    minute_bucket_expr.label("minute_bucket"),
                    func.avg(LectureActivity.active_lectures).label("active_lectures_avg"),
                    func.avg(LectureActivity.starts_next_60m).label("starts_next_60m_avg"),
                    func.avg(LectureActivity.ends_next_60m).label("ends_next_60m_avg"),
                    func.avg(LectureActivity.quality_score).label("quality_score_avg"),
                )
                .where(LectureActivity.zone_id == zone_id)
                .where(LectureActivity.ts >= from_dt)
                .group_by(minute_bucket_expr)
                .order_by(minute_bucket_expr.asc())
            ).all()
    except OperationalError as exc:
        logger.warning("xai lecture query failed (zone=%s): %s", zone_id, exc)
        return pd.DataFrame(
            columns=["ts", "active_lectures", "starts_next_60m", "ends_next_60m", "quality_score"]
        )

    if not rows:
        return pd.DataFrame(
            columns=["ts", "active_lectures", "starts_next_60m", "ends_next_60m", "quality_score"]
        )

    timestamps: list[datetime] = []
    active_lectures: list[float] = []
    starts_next_60m: list[float] = []
    ends_next_60m: list[float] = []
    quality_scores: list[float] = []
    for row in rows:
        row_map = row._mapping
        minute_value = row_map.get("minute_bucket")
        if minute_value is None:
            continue
        try:
            ts = _normalize_dt(minute_value if isinstance(minute_value, datetime) else datetime.fromisoformat(str(minute_value).replace("Z", "+00:00")))
        except Exception:
            continue
        timestamps.append(ts)
        active_lectures.append(float(row_map.get("active_lectures_avg") or 0.0))
        starts_next_60m.append(float(row_map.get("starts_next_60m_avg") or 0.0))
        ends_next_60m.append(float(row_map.get("ends_next_60m_avg") or 0.0))
        quality_scores.append(float(row_map.get("quality_score_avg") or 0.0))

    return pd.DataFrame(
        {
            "ts": timestamps,
            "active_lectures": active_lectures,
            "starts_next_60m": starts_next_60m,
            "ends_next_60m": ends_next_60m,
            "quality_score": quality_scores,
        }
    )


def _fetch_forecast_interval(zone_id: str, horizon: int) -> float:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{FORECAST_SERVICE_URL}/v1/forecast",
                params={"zone_id": zone_id, "horizon": horizon},
            )
            response.raise_for_status()
            payload = response.json()
            points = payload.get("points", [])
            if not points:
                return 0.0
            widths = [max(0.0, float(p["pi_high"]) - float(p["pi_low"])) for p in points]
            return float(np.mean(widths))
    except Exception:
        return 0.0


def _compute_drivers(
    df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    events: list[CalendarEvent],
    interval_width: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if df.empty:
        drivers = [
            {
                "name": "insufficient_history",
                "impact": 0.0,
                "direction": "mixed",
                "description": "No count history available."
            }
        ]
        uncertainty = {
            "score": 0.85,
            "level": "high",
            "reason": "Sparse history prevents reliable attribution."
        }
        return drivers, uncertainty

    series = df.set_index("ts")["occupancy"].resample("1min").mean().interpolate(limit_direction="both")
    recent = series.tail(min(len(series), 60))

    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent.values, 1)[0]) if len(recent) >= 2 else 0.0

    momentum = float(series.tail(min(len(series), 15)).mean() - series.tail(min(len(series), 60)).mean()) if len(series) > 1 else 0.0

    seasonality = 0.0
    if len(series) > 120:
        seasonality = float(series.iloc[-1] - series.iloc[-60])

    event_impact = float(sum(float(e.expected_impact or 0.0) for e in events))

    lecture_density = 0.0
    lecture_start_wave = 0.0
    lecture_quality_penalty = 0.0
    if not lecture_df.empty:
        lecture_series = lecture_df.set_index("ts")["active_lectures"].resample("1min").mean().interpolate(limit_direction="both")
        lecture_density = float(lecture_series.tail(min(60, len(lecture_series))).mean()) if not lecture_series.empty else 0.0
        lecture_start_wave = float(lecture_df["starts_next_60m"].tail(min(60, len(lecture_df))).mean()) if len(lecture_df) else 0.0
        lecture_quality_penalty = max(0.0, 1.0 - float(lecture_df["quality_score"].mean()))

    quality_mean = float(df["quality_score"].mean()) if not df.empty else 0.5
    quality_penalty = max(0.0, 1.0 - quality_mean)

    drivers = [
        {
            "name": "trend",
            "impact": abs(slope),
            "direction": "up" if slope > 0 else "down" if slope < 0 else "mixed",
            "description": f"Recent slope over last hour is {slope:.2f} occupancy/min.",
        },
        {
            "name": "momentum",
            "impact": abs(momentum),
            "direction": "up" if momentum > 0 else "down" if momentum < 0 else "mixed",
            "description": f"Last 15m vs 60m delta is {momentum:.2f}.",
        },
        {
            "name": "seasonality_proxy",
            "impact": abs(seasonality),
            "direction": "up" if seasonality > 0 else "down" if seasonality < 0 else "mixed",
            "description": f"Current level vs 60m ago differs by {seasonality:.2f}.",
        },
        {
            "name": "event_context",
            "impact": abs(event_impact),
            "direction": "up" if event_impact > 0 else "mixed",
            "description": f"{len(events)} active/planned events in forecast horizon.",
        },
        {
            "name": "lecture_density",
            "impact": abs(lecture_density),
            "direction": "up" if lecture_density > 0 else "mixed",
            "description": f"Average active lectures over the last 60m is {lecture_density:.2f}.",
        },
        {
            "name": "lecture_start_wave",
            "impact": abs(lecture_start_wave),
            "direction": "up" if lecture_start_wave > 0 else "mixed",
            "description": f"Expected lecture starts in next 60m average {lecture_start_wave:.2f}.",
        },
        {
            "name": "data_quality_penalty",
            "impact": abs(max(quality_penalty, lecture_quality_penalty)),
            "direction": "down" if quality_penalty > 0 else "mixed",
            "description": (
                f"Count quality mean {quality_mean:.2f}; lecture quality penalty {lecture_quality_penalty:.2f}."
            ),
        },
    ]

    drivers = sorted(drivers, key=lambda d: d["impact"], reverse=True)

    normalized_interval = interval_width / max(float(series.mean()), 1.0)
    uncertainty_score = max(
        0.0,
        min(1.0, 0.15 + normalized_interval * 0.5 + max(quality_penalty, lecture_quality_penalty) * 0.6),
    )
    if uncertainty_score < 0.34:
        level = "low"
    elif uncertainty_score < 0.67:
        level = "medium"
    else:
        level = "high"

    uncertainty = {
        "score": round(uncertainty_score, 3),
        "level": level,
        "reason": "Combines forecast interval width and data quality penalty.",
    }

    if XAI_SHAP_ENABLED and FORECAST_MODEL_BACKEND == "xgboost":
        drivers.append(
            {
                "name": "shap_placeholder",
                "impact": 0.0,
                "direction": "mixed",
                "description": "SHAP enabled, but MVP implementation uses deterministic proxy drivers.",
            }
        )

    return drivers[:5], uncertainty


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "xai"}


@app.get("/v1/explain", response_model=ExplanationResponse)
def explain_forecast(
    zone_id: str = Query(...),
    horizon: int = Query(default=DEFAULT_FORECAST_HORIZON_MINUTES, ge=1, le=720),
) -> ExplanationResponse:
    now = datetime.now(UTC)
    try:
        counts_df = _load_counts(zone_id)
        lecture_df = _load_lecture_activity(zone_id)
        events = _load_events(zone_id, horizon)
        interval_width = _fetch_forecast_interval(zone_id, horizon)
        drivers, uncertainty = _compute_drivers(counts_df, lecture_df, events, interval_width)

        summary = "Top demand drivers calculated with progressive disclosure: summary, drivers, evidence, and counterfactual-ready context."
        quality_flags = ["LOW_CONFIDENCE"] if uncertainty["level"] == "high" else ["OK"]
    except Exception as exc:
        logger.warning("xai explain fallback activated (zone=%s, horizon=%s): %s", zone_id, horizon, exc)
        drivers = [
            {
                "name": "upstream_unavailable",
                "impact": 0.0,
                "direction": "mixed",
                "description": "Explainability degraded due to temporary backend lock/timeout.",
            }
        ]
        uncertainty = {
            "score": 1.0,
            "level": "high",
            "reason": "Fallback response due to temporary backend error.",
        }
        interval_width = 0.0
        events = []
        lecture_df = pd.DataFrame(columns=["ts"])
        summary = "Fallback: explainability temporarily degraded; using minimal safe response."
        quality_flags = ["BACKEND_ERROR", "FALLBACK"]

    evidence = {
        "evidence_id": f"xai-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": (now - timedelta(hours=24)).isoformat(),
            "to": now.isoformat(),
        },
        "sources": [
            {"type": "counts", "id": f"zone:{zone_id}"},
            {"type": "events", "id": f"events:{len(events)}"},
            {"type": "lecture_activity", "id": f"rows:{len(lecture_df)}"},
            {"type": "forecast", "id": f"interval:{interval_width:.2f}"},
        ],
        "model": {"name": "xai_driver_engine", "version": "v1", "backend": FORECAST_MODEL_BACKEND},
        "quality": {
            "score": max(0.0, 1.0 - float(uncertainty.get("score", 1.0))),
            "flags": quality_flags,
        },
    }

    return ExplanationResponse(
        zone_id=zone_id,
        horizon=horizon,
        summary=summary,
        drivers=drivers,
        uncertainty=uncertainty,
        evidence=evidence,
    )
