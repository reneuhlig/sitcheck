from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine, event, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

from explainability import (
    LLMNarrativeUnavailableError,
    LLMQualityGateError,
    NarrativeService,
    PromptRegistry,
    build_explainability_context_v2,
)
from weekly_explain import build_weekly_explainability


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sitcheck.db")
DEFAULT_ZONE_ID = os.getenv("DEFAULT_ZONE_ID", "default-zone")
DEFAULT_ZONE_CAPACITY = int(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))
FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8001")
XAI_SERVICE_URL = os.getenv("XAI_SERVICE_URL", "http://xai:8002")
RECOMMENDATIONS_SERVICE_URL = os.getenv("RECOMMENDATIONS_SERVICE_URL", "http://recommendations:8003")
SCHEDULER_SERVICE_URL = os.getenv("SCHEDULER_SERVICE_URL", "http://forecast-scheduler:8011")
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
FORECAST_SERVICE_TIMEOUT_SECONDS = float(os.getenv("FORECAST_SERVICE_TIMEOUT_SECONDS", "75"))
FORECAST_SNAPSHOT_RETENTION_DAYS = int(os.getenv("FORECAST_SNAPSHOT_RETENTION_DAYS", "14"))
FORECAST_STALE_THRESHOLD_SECONDS = int(os.getenv("FORECAST_STALE_THRESHOLD_SECONDS", "900"))
MAX_FORECAST_HORIZON_MINUTES = int(os.getenv("MAX_FORECAST_HORIZON_MINUTES", "43200"))
DEFAULT_FORECAST_HORIZON_MINUTES = max(
    1,
    min(
        int(os.getenv("DEFAULT_FORECAST_HORIZON_MINUTES", os.getenv("TF_DEFAULT_HORIZON", "210"))),
        MAX_FORECAST_HORIZON_MINUTES,
    ),
)
DEFAULT_SHORT_FORECAST_HORIZON_MINUTES = max(1, min(DEFAULT_FORECAST_HORIZON_MINUTES, 720))
WEEKLY_FORECAST_DAYS_DEFAULT = int(os.getenv("WEEKLY_FORECAST_DAYS_DEFAULT", "7"))
WEEKLY_FORECAST_SLOT_MINUTES = int(os.getenv("WEEKLY_FORECAST_SLOT_MINUTES", "60"))
EXPLAINABILITY_TEMPLATE_SET = os.getenv("EXPLAINABILITY_TEMPLATE_SET", "explainability-de-v2")
EXPLAINABILITY_RESPONSE_MODE = os.getenv("EXPLAINABILITY_RESPONSE_MODE", "free")
EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS", "90"))
EXPLAINABILITY_FALLBACK_ENABLED = os.getenv("EXPLAINABILITY_FALLBACK_ENABLED", "true").lower() == "true"
EXPLAINABILITY_DEBUG_PROMPT_PREVIEW = (
    os.getenv("EXPLAINABILITY_DEBUG_PROMPT_PREVIEW", "false").lower() == "true"
)
EXPLAINABILITY_PROFESSOR_MODE_ENABLED = (
    os.getenv("EXPLAINABILITY_PROFESSOR_MODE_ENABLED", "true").lower() == "true"
)
EXPLAINABILITY_INTENT_ROUTER_ENABLED = (
    os.getenv("EXPLAINABILITY_INTENT_ROUTER_ENABLED", "true").lower() == "true"
)
EXPLAINABILITY_DYNAMIC_RENDER_ENABLED = (
    os.getenv("EXPLAINABILITY_DYNAMIC_RENDER_ENABLED", "true").lower() == "true"
)
EXPLAINABILITY_LLM_MIN_FIELD_COVERAGE = int(os.getenv("EXPLAINABILITY_LLM_MIN_FIELD_COVERAGE", "3"))
EXPLAINABILITY_LLM_RETRY_ON_LOW_COVERAGE = int(os.getenv("EXPLAINABILITY_LLM_RETRY_ON_LOW_COVERAGE", "1"))
EXPLAINABILITY_STRICT_LLM_GATE = (
    os.getenv("EXPLAINABILITY_STRICT_LLM_GATE", "true").lower() == "true"
)
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
PRODUCT_SHORT_TERM = "short_term"
PRODUCT_WEEKLY = "weekly_slot"
SESSION_COOKIE_NAME = "sitcheck_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
UserRole = Literal["user", "admin"]

if DATABASE_URL.startswith("sqlite:"):
    # For local no-docker runtime, avoid QueuePool starvation under bursty dashboard polling.
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


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


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


class ScenarioRun(Base):
    __tablename__ = "scenario_runs"

    scenario_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    persist: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class ForecastSnapshot(Base):
    __tablename__ = "forecasts"

    forecast_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    product: Mapped[str] = mapped_column(String, nullable=False, default=PRODUCT_SHORT_TERM)
    slot_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class ReferenceObject(Base):
    __tablename__ = "reference_objects"

    reference_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reference_type: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    uri_or_path: Mapped[str | None] = mapped_column(String, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class ModelRun(Base):
    __tablename__ = "model_runs"

    model_run_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    product: Mapped[str] = mapped_column(String, nullable=False, default=PRODUCT_SHORT_TERM)
    horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_backend: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="trained")
    scientific_status: Mapped[str] = mapped_column(String, nullable=False, default="training_only")
    include_lecture_impact: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    feature_set_version: Mapped[str] = mapped_column(String, nullable=False)
    history_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    history_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evaluation_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promotion_source: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    lineage_json: Mapped[dict[str, Any]] = mapped_column("lineage", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    username_normalized: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False, default="user")
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    password_salt: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class Booking(Base):
    __tablename__ = "bookings"

    booking_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    party_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String, nullable=False, default="confirmed", index=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class EvidenceTimeWindow(BaseModel):
    from_: datetime = Field(alias="from")
    to: datetime


class EvidenceSource(BaseModel):
    type: str
    id: str
    uri: str | None = None
    note: str | None = None
    metadata: dict[str, Any] | None = None


class EvidenceModelSpec(BaseModel):
    name: str
    version: str
    backend: str | None = None


class EvidenceQuality(BaseModel):
    score: float
    flags: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    generated_at: datetime
    time_window: EvidenceTimeWindow
    sources: list[EvidenceSource]
    model: EvidenceModelSpec
    quality: EvidenceQuality


class ZoneResponse(BaseModel):
    zone_id: str
    name: str
    capacity: int
    is_active: bool
    metadata: dict[str, Any]


class CountIngestPoint(BaseModel):
    timestamp: datetime | None = None
    zone_id: str
    occupancy: int = Field(ge=0)
    utilization: float | None = Field(default=None, ge=0)
    source: str = "vision-counter"
    quality_score: float = Field(default=1.0, ge=0, le=1)
    quality_flags: list[str] = Field(default_factory=list)
    evidence: Evidence


class CountIngestRequest(BaseModel):
    points: list[CountIngestPoint]


class CountResponsePoint(BaseModel):
    timestamp: datetime
    zone_id: str
    occupancy: int
    utilization: float
    source: str
    quality_score: float
    quality_flags: list[str]
    evidence: dict[str, Any]


class CountsResponse(BaseModel):
    zone_id: str
    from_: datetime = Field(alias="from")
    to: datetime
    granularity: str
    points: list[CountResponsePoint]


class CalendarEventResponse(BaseModel):
    event_id: str
    zone_id: str | None = None
    title: str
    category: str | None = None
    starts_at: datetime
    ends_at: datetime
    expected_impact: float | None = None
    source: str
    metadata: dict[str, Any]


class LectureActivityPoint(BaseModel):
    timestamp: datetime
    zone_id: str
    active_lectures: int
    active_courses: int
    starts_next_60m: int
    ends_next_60m: int
    source: str
    quality_score: float
    quality_flags: list[str]
    metadata: dict[str, Any]


class LectureActivityResponse(BaseModel):
    zone_id: str
    from_: datetime = Field(alias="from")
    to: datetime
    granularity: str
    points: list[LectureActivityPoint]


class ScenarioInputRequest(BaseModel):
    zone_id: str
    horizon: int = Field(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720)
    persist: bool = False
    changes: dict[str, Any]


class ForecastSnapshotRequest(BaseModel):
    zone_id: str
    horizon: int = Field(default=DEFAULT_FORECAST_HORIZON_MINUTES, ge=1, le=MAX_FORECAST_HORIZON_MINUTES)


class ForecastLatestResponse(BaseModel):
    zone_id: str
    horizon: int
    generated_at: datetime
    age_seconds: int
    stale: bool
    summary: str
    model_version: str
    points: list[dict[str, Any]]
    evidence: dict[str, Any]
    lineage: dict[str, Any] | None = None
    source: Literal["snapshot", "on_demand_fallback"]


class ForecastHistoryResponse(BaseModel):
    zone_id: str
    horizon: int
    from_: datetime = Field(alias="from")
    to: datetime
    items: list[ForecastLatestResponse]


class WeeklyForecastPoint(BaseModel):
    timestamp: datetime
    yhat: float
    pi_low: float
    pi_high: float
    day_of_week: int
    slot_of_day: int
    event_active: float | None = None
    event_impact_sum: float | None = None
    lecture_net_pull: float | None = None
    quality_score: float | None = None


class WeeklyForecastDaySummary(BaseModel):
    date: str
    peak_slot: str | None = None
    peak_yhat: float
    avg_yhat: float
    risk_level: str
    data_quality: str


class WeeklyForecastResponse(BaseModel):
    zone_id: str
    product: str
    days: int
    slot_minutes: int
    generated_at: datetime
    age_seconds: int
    stale: bool
    summary: str
    model_version: str
    points: list[WeeklyForecastPoint]
    daily_summaries: list[WeeklyForecastDaySummary]
    evidence: dict[str, Any]
    lineage: dict[str, Any]
    source: Literal["snapshot", "on_demand_fallback"]


class WeeklyForecastHistoryResponse(BaseModel):
    zone_id: str
    days: int
    slot_minutes: int
    from_: datetime = Field(alias="from")
    to: datetime
    items: list[WeeklyForecastResponse]


class TrainingDataReferenceRegisterRequest(BaseModel):
    zone_id: str | None = None
    source_type: str = Field(min_length=2, max_length=40)
    label: str = Field(min_length=2, max_length=240)
    uri_or_path: str | None = None
    checksum: str | None = None
    imported_at: datetime | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
    row_count: int | None = Field(default=None, ge=0)
    ingest_job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WeeklyExplainResponse(BaseModel):
    zone_id: str
    days: int
    slot_minutes: int
    generated_at: datetime
    summary: str
    week_overview: dict[str, Any]
    daily_highlights: list[dict[str, Any]]
    drivers: list[dict[str, Any]]
    uncertainty: dict[str, Any]
    quality: dict[str, Any]
    references: list[dict[str, Any]]
    lineage: dict[str, Any]
    evidence: dict[str, Any]


class ExplainContextResponse(BaseModel):
    context_version: str
    context: dict[str, Any]


class ExplainNarrativeRequest(BaseModel):
    zone_id: str
    horizon: int = Field(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720)
    audience: Literal["ops", "executive", "enduser", "professor"] = "ops"
    query: str = ""
    language: Literal["de", "en"] = "de"
    response_mode: Literal["free"] = "free"
    ollama_model: str | None = Field(default=None, min_length=1, max_length=120)
    require_ollama: bool = False


class ExplainPromptPreviewRequest(BaseModel):
    zone_id: str
    horizon: int = Field(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720)
    audience: Literal["ops", "executive", "enduser", "professor"] = "ops"
    query: str = ""
    language: Literal["de", "en"] = "de"


class AuthCredentialsRequest(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=4, max_length=240)


class AuthRegisterRequest(AuthCredentialsRequest):
    role: UserRole = "user"


class UserPublic(BaseModel):
    user_id: str
    username: str
    role: UserRole


class AuthResponse(BaseModel):
    authenticated: bool
    user: UserPublic | None = None


class BookingCreateRequest(BaseModel):
    starts_at: datetime
    ends_at: datetime


class BookingResponse(BaseModel):
    booking_id: str
    zone_id: str
    starts_at: datetime
    ends_at: datetime
    status: Literal["confirmed", "cancelled"]
    created_at: datetime
    cancelled_at: datetime | None = None


class AdminBookingResponse(BookingResponse):
    user_id: str
    username: str


app = FastAPI(title="sitcheck-api-gateway", version="0.1.0")
PROMPT_REGISTRY = PromptRegistry(template_set_id=EXPLAINABILITY_TEMPLATE_SET)
NARRATIVE_SERVICE = NarrativeService(
    prompt_registry=PROMPT_REGISTRY,
    ollama_enabled=OLLAMA_ENABLED,
    ollama_base_url=OLLAMA_BASE_URL,
    ollama_model=OLLAMA_MODEL,
    ollama_timeout_seconds=EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS,
    fallback_enabled=EXPLAINABILITY_FALLBACK_ENABLED,
    intent_router_enabled=EXPLAINABILITY_INTENT_ROUTER_ENABLED,
    dynamic_render_enabled=EXPLAINABILITY_DYNAMIC_RENDER_ENABLED,
    llm_min_field_coverage=EXPLAINABILITY_LLM_MIN_FIELD_COVERAGE,
    llm_retry_on_low_coverage=EXPLAINABILITY_LLM_RETRY_ON_LOW_COVERAGE,
    strict_llm_gate=EXPLAINABILITY_STRICT_LLM_GATE,
)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _normalize_dt(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_granularity(value: str) -> str:
    allowed = {"raw", "1m", "5m", "15m", "60m"}
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid granularity '{value}'")
    return value


def _default_evidence(now: datetime, zone_id: str) -> dict[str, Any]:
    return {
        "evidence_id": f"ev-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {"from": (now - timedelta(minutes=60)).isoformat(), "to": now.isoformat()},
        "sources": [{"type": "api", "id": f"ingest:{zone_id}"}],
        "model": {"name": "ingestion", "version": "v1"},
        "quality": {"score": 1.0, "flags": ["DEFAULT"]},
    }


def _normalize_evidence_payload(evidence: dict[str, Any] | None, now: datetime, zone_id: str) -> dict[str, Any]:
    payload = dict(evidence or _default_evidence(now, zone_id))
    time_window = payload.get("time_window")
    if isinstance(time_window, dict):
        tw = dict(time_window)
        if "from" not in tw and "from_" in tw:
            tw["from"] = tw["from_"]
        tw.pop("from_", None)
        payload["time_window"] = tw

    sources = payload.get("sources")
    if isinstance(sources, list):
        sanitized_sources = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            sanitized_source = {k: v for k, v in source.items() if v is not None}
            if "type" in sanitized_source and "id" in sanitized_source:
                sanitized_sources.append(sanitized_source)
        if sanitized_sources:
            payload["sources"] = sanitized_sources

    model = payload.get("model")
    if isinstance(model, dict):
        model_payload = {k: v for k, v in model.items() if v is not None}
        payload["model"] = model_payload

    quality = payload.get("quality")
    if isinstance(quality, dict):
        quality_payload = dict(quality)
        if "flags" in quality_payload and quality_payload["flags"] is None:
            quality_payload["flags"] = []
        payload["quality"] = quality_payload
    return payload


def _coerce_optional_dt(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return _normalize_dt(value)
    try:
        return _normalize_dt(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except Exception:
        return None


def _float_or_zero(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    return parsed if parsed == parsed else 0.0


def _reference_payload_to_public(row: ReferenceObject) -> dict[str, Any]:
    return {
        "reference_id": row.reference_id,
        "zone_id": row.zone_id,
        "reference_type": row.reference_type,
        "source_type": row.source_type,
        "label": row.label,
        "uri_or_path": row.uri_or_path,
        "checksum": row.checksum,
        "imported_at": row.imported_at.isoformat() if row.imported_at else None,
        "time_from": row.time_from.isoformat() if row.time_from else None,
        "time_to": row.time_to.isoformat() if row.time_to else None,
        "row_count": row.row_count,
        "metadata": row.metadata_json or {},
        "created_at": row.created_at.isoformat(),
    }


def _normalize_lineage_payload(value: Any, default_product: str) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("product"):
        return value
    return {"product": default_product}


def _normalize_username_input(username: str) -> tuple[str, str]:
    cleaned = username.strip()
    if len(cleaned) < 3:
        raise HTTPException(status_code=400, detail="username_too_short")
    return cleaned, cleaned.casefold()


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    ).hex()


def _create_password_credentials(password: str) -> tuple[str, str]:
    salt_hex = secrets.token_hex(16)
    return salt_hex, _hash_password(password, salt_hex)


def _verify_password(password: str, *, salt_hex: str, expected_hash: str) -> bool:
    return secrets.compare_digest(_hash_password(password, salt_hex), expected_hash)


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_user_role(role: str | None) -> UserRole:
    normalized = str(role or "user").strip().lower()
    return "admin" if normalized == "admin" else "user"


def _serialize_user(user: User | None) -> UserPublic | None:
    if user is None:
        return None
    return UserPublic(
        user_id=user.user_id,
        username=user.username,
        role=_normalize_user_role(user.role),
    )


def _serialize_booking(booking: Booking) -> BookingResponse:
    return BookingResponse(
        booking_id=booking.booking_id,
        zone_id=booking.zone_id,
        starts_at=_normalize_dt(booking.starts_at),
        ends_at=_normalize_dt(booking.ends_at),
        status=str(booking.status),
        created_at=_normalize_dt(booking.created_at),
        cancelled_at=_normalize_dt(booking.cancelled_at) if booking.cancelled_at else None,
    )


def _serialize_admin_booking(booking: Booking, user: User) -> AdminBookingResponse:
    return AdminBookingResponse(
        booking_id=booking.booking_id,
        user_id=user.user_id,
        username=user.username,
        zone_id=booking.zone_id,
        starts_at=_normalize_dt(booking.starts_at),
        ends_at=_normalize_dt(booking.ends_at),
        status=str(booking.status),
        created_at=_normalize_dt(booking.created_at),
        cancelled_at=_normalize_dt(booking.cancelled_at) if booking.cancelled_at else None,
    )


def _set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/", samesite="lax")


def _delete_expired_sessions(db: Session, *, now: datetime | None = None) -> None:
    cutoff = datetime.now(UTC) if now is None else _normalize_dt(now)
    db.query(UserSession).filter(UserSession.expires_at <= cutoff).delete(synchronize_session=False)
    db.commit()


def _create_user_session(db: Session, *, user_id: str, now: datetime | None = None) -> tuple[UserSession, str]:
    issued_at = datetime.now(UTC) if now is None else _normalize_dt(now)
    raw_token = secrets.token_urlsafe(32)
    session = UserSession(
        session_id=f"sess-{uuid.uuid4()}",
        user_id=user_id,
        token_hash=_hash_session_token(raw_token),
        created_at=issued_at,
        expires_at=issued_at + timedelta(seconds=SESSION_TTL_SECONDS),
        last_used_at=issued_at,
    )
    db.add(session)
    return session, raw_token


def _resolve_session_user(request: Request, db: Session) -> tuple[User | None, UserSession | None]:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME, "").strip()
    if not raw_token:
        return None, None

    now = datetime.now(UTC)
    token_hash = _hash_session_token(raw_token)
    session = (
        db.execute(select(UserSession).where(UserSession.token_hash == token_hash).limit(1))
        .scalars()
        .first()
    )
    if session is None:
        return None, None

    if _normalize_dt(session.expires_at) <= now:
        db.delete(session)
        db.commit()
        return None, None

    user = db.get(User, session.user_id)
    if user is None:
        db.delete(session)
        db.commit()
        return None, None
    return user, session


def _require_session_user(request: Request, db: Session) -> tuple[User, UserSession]:
    user, session = _resolve_session_user(request=request, db=db)
    if user is None or session is None:
        raise HTTPException(status_code=401, detail="authentication_required")
    return user, session


def _require_admin_user(request: Request, db: Session) -> tuple[User, UserSession]:
    user, session = _require_session_user(request=request, db=db)
    if _normalize_user_role(user.role) != "admin":
        raise HTTPException(status_code=403, detail="admin_required")
    return user, session


def _local_day_start_utc(now: datetime | None = None) -> datetime:
    current = datetime.now().astimezone() if now is None else _normalize_dt(now).astimezone()
    local_day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_day_start.astimezone(UTC)


def _load_confirmed_bookings(
    db: Session,
    *,
    zone_id: str,
    window_start: datetime,
    window_end: datetime,
) -> list[Booking]:
    if window_end <= window_start:
        return []
    return (
        db.execute(
            select(Booking)
            .where(Booking.zone_id == zone_id)
            .where(Booking.status == "confirmed")
            .where(Booking.ends_at > window_start)
            .where(Booking.starts_at < window_end)
            .order_by(Booking.starts_at.asc())
        )
        .scalars()
        .all()
    )


def _booking_overlay_sentence(booking_count: int, peak_increment: int) -> str:
    booking_label = "Buchung" if booking_count == 1 else "Buchungen"
    person_label = "Person" if peak_increment == 1 else "Personen"
    return (
        f"{booking_count} bestätigte {booking_label} überlappen den Prognosehorizont "
        f"und erhöhen die Erwartung in einzelnen Slots um bis zu {peak_increment} {person_label}."
    )


def _augment_evidence_with_booking_overlay(
    evidence: dict[str, Any] | None,
    *,
    zone_id: str,
    booking_count: int,
    peak_increment: int,
    window_from: datetime,
    window_to: datetime,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    payload = _normalize_evidence_payload(evidence, now=now, zone_id=zone_id)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    if not any(isinstance(source, dict) and str(source.get("id")) == "bookings:overlay" for source in sources):
        sources.append(
            {
                "type": "api",
                "id": "bookings:overlay",
                "note": "Runtime booking overlay for short-term forecast and explainability.",
                "metadata": {
                    "booking_count": booking_count,
                    "peak_increment": peak_increment,
                    "window_from": window_from.isoformat(),
                    "window_to": window_to.isoformat(),
                },
            }
        )
    payload["sources"] = sources

    quality = payload.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
    flags = quality.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    if "BOOKING_OVERLAY" not in flags:
        flags.append("BOOKING_OVERLAY")
    quality["flags"] = flags
    payload["quality"] = quality
    return payload


def _compute_booking_overlay(
    db: Session,
    *,
    zone_id: str,
    horizon: int,
    forecast_points: list[dict[str, Any]],
) -> dict[str, Any]:
    if horizon > DEFAULT_SHORT_FORECAST_HORIZON_MINUTES or not forecast_points:
        return {"applied": False, "counts": [0 for _ in forecast_points]}

    point_times: list[tuple[int, datetime]] = []
    for idx, point in enumerate(forecast_points):
        point_dt = _coerce_optional_dt(point.get("timestamp"))
        if point_dt is None:
            continue
        point_times.append((idx, point_dt))

    if not point_times:
        return {"applied": False, "counts": [0 for _ in forecast_points]}

    window_start = min(point_dt for _, point_dt in point_times)
    window_end = max(point_dt for _, point_dt in point_times) + timedelta(minutes=1)
    bookings = _load_confirmed_bookings(
        db,
        zone_id=zone_id,
        window_start=window_start,
        window_end=window_end,
    )
    counts = [0 for _ in forecast_points]
    for idx, point_dt in point_times:
        counts[idx] = sum(
            int(booking.party_size)
            for booking in bookings
            if _normalize_dt(booking.starts_at) <= point_dt < _normalize_dt(booking.ends_at)
        )

    peak_increment = max(counts, default=0)
    return {
        "applied": peak_increment > 0,
        "counts": counts,
        "booking_count": len(bookings),
        "peak_increment": peak_increment,
        "window_from": window_start,
        "window_to": window_end,
    }


def _apply_booking_overlay_to_forecast(
    db: Session,
    *,
    zone_id: str,
    horizon: int,
    forecast: ForecastLatestResponse,
) -> tuple[ForecastLatestResponse, dict[str, Any]]:
    payload = forecast.model_dump(mode="json")
    points_raw = payload.get("points", [])
    if not isinstance(points_raw, list):
        return forecast, {"applied": False, "counts": []}

    overlay = _compute_booking_overlay(
        db,
        zone_id=zone_id,
        horizon=horizon,
        forecast_points=[point if isinstance(point, dict) else {} for point in points_raw],
    )
    if not overlay.get("applied"):
        return forecast, overlay

    adjusted_points: list[dict[str, Any]] = []
    for idx, point in enumerate(points_raw):
        point_payload = dict(point) if isinstance(point, dict) else {}
        increment = int(overlay["counts"][idx]) if idx < len(overlay["counts"]) else 0
        for key in ("yhat", "pi_low", "pi_high"):
            try:
                point_payload[key] = round(float(point_payload.get(key, 0.0)) + increment, 4)
            except Exception:
                continue
        adjusted_points.append(point_payload)

    payload["points"] = adjusted_points
    payload["summary"] = (
        f"{str(payload.get('summary', '')).strip()} "
        f"{_booking_overlay_sentence(int(overlay['booking_count']), int(overlay['peak_increment']))}"
    ).strip()
    payload["evidence"] = _augment_evidence_with_booking_overlay(
        payload.get("evidence"),
        zone_id=zone_id,
        booking_count=int(overlay["booking_count"]),
        peak_increment=int(overlay["peak_increment"]),
        window_from=_normalize_dt(overlay["window_from"]),
        window_to=_normalize_dt(overlay["window_to"]),
    )
    return ForecastLatestResponse.model_validate(payload), overlay


def _apply_booking_overlay_to_explanation(
    explanation: dict[str, Any],
    *,
    zone_id: str,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    if not overlay.get("applied"):
        return explanation

    payload = dict(explanation)
    drivers = payload.get("drivers", [])
    if not isinstance(drivers, list):
        drivers = []
    drivers = [
        driver
        for driver in drivers
        if not (isinstance(driver, dict) and str(driver.get("name", "")).strip().casefold() == "buchungen")
    ]
    drivers.append(
        {
            "name": "Buchungen",
            "impact": float(overlay["peak_increment"]),
            "direction": "up",
            "description": (
                f"{int(overlay['booking_count'])} bestätigte Buchungen überlappen die nächsten 60 Minuten "
                f"und addieren in einzelnen Slots bis zu {int(overlay['peak_increment'])} Personen."
            ),
        }
    )
    payload["drivers"] = sorted(
        drivers,
        key=lambda item: abs(_float_or_zero(item.get("impact", 0.0))) if isinstance(item, dict) else 0.0,
        reverse=True,
    )
    payload["summary"] = (
        f"{str(payload.get('summary', '')).strip()} "
        f"{_booking_overlay_sentence(int(overlay['booking_count']), int(overlay['peak_increment']))}"
    ).strip()
    payload["evidence"] = _augment_evidence_with_booking_overlay(
        payload.get("evidence"),
        zone_id=zone_id,
        booking_count=int(overlay["booking_count"]),
        peak_increment=int(overlay["peak_increment"]),
        window_from=_normalize_dt(overlay["window_from"]),
        window_to=_normalize_dt(overlay["window_to"]),
    )
    return payload


def _verify_internal_token(x_internal_token: str | None) -> None:
    if not INTERNAL_API_TOKEN:
        raise HTTPException(status_code=503, detail="INTERNAL_API_TOKEN is not configured")
    if x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


def _ensure_default_zone(db: Session) -> Zone:
    zone = db.get(Zone, DEFAULT_ZONE_ID)
    if zone:
        return zone

    zone = Zone(
        zone_id=DEFAULT_ZONE_ID,
        name="Default Zone",
        capacity=DEFAULT_ZONE_CAPACITY,
        is_active=True,
        metadata_json={"seeded": True},
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return zone


def _seed_mock_calendar(db: Session, zone_id: str) -> None:
    existing = db.execute(select(func.count()).select_from(CalendarEvent)).scalar_one()
    if existing and existing > 0:
        return

    base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    events = [
        CalendarEvent(
            event_id=f"evt-{uuid.uuid4()}",
            zone_id=zone_id,
            title="Vorlesungswechsel",
            category="campus-flow",
            starts_at=base + timedelta(hours=1),
            ends_at=base + timedelta(hours=2),
            expected_impact=0.25,
            source="mock",
            metadata_json={"origin": "seed"},
        ),
        CalendarEvent(
            event_id=f"evt-{uuid.uuid4()}",
            zone_id=zone_id,
            title="Mensa Peak",
            category="food",
            starts_at=base + timedelta(hours=3),
            ends_at=base + timedelta(hours=4),
            expected_impact=0.35,
            source="mock",
            metadata_json={"origin": "seed"},
        ),
    ]
    db.add_all(events)
    db.commit()


def _ensure_sqlite_indexes(db: Session) -> None:
    if not DATABASE_URL.startswith("sqlite:"):
        return
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_counts_zone_ts ON counts(zone_id, ts)"))
    db.execute(
        text("CREATE INDEX IF NOT EXISTS idx_lecture_activity_zone_ts ON lecture_activity(zone_id, ts)")
    )
    db.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_zone_horizon_generated_at "
            "ON forecasts(zone_id, horizon, generated_at)"
        )
    )
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_username_normalized ON users(username_normalized)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_bookings_user_id ON bookings(user_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_bookings_zone_window ON bookings(zone_id, status, starts_at, ends_at)"))
    db.execute(text("PRAGMA optimize"))
    db.commit()


def _sqlite_has_column(db: Session, table_name: str, column_name: str) -> bool:
    rows = db.execute(text(f"PRAGMA table_info({table_name})")).all()
    return any(str(row._mapping.get("name")) == column_name for row in rows)


def _ensure_sqlite_schema_extensions(db: Session) -> None:
    if not DATABASE_URL.startswith("sqlite:"):
        return

    user_columns = {
        "role": "TEXT NOT NULL DEFAULT 'user'",
    }
    for column_name, ddl in user_columns.items():
        if not _sqlite_has_column(db, "users", column_name):
            db.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {ddl}"))

    forecast_columns = {
        "product": "TEXT NOT NULL DEFAULT 'short_term'",
        "slot_minutes": "INTEGER",
        "days": "INTEGER",
        "model_run_id": "TEXT",
    }
    for column_name, ddl in forecast_columns.items():
        if not _sqlite_has_column(db, "forecasts", column_name):
            db.execute(text(f"ALTER TABLE forecasts ADD COLUMN {column_name} {ddl}"))

    db.execute(text("CREATE INDEX IF NOT EXISTS idx_forecasts_zone_product_generated_at ON forecasts(zone_id, product, generated_at)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_model_runs_zone_product_created_at ON model_runs(zone_id, product, created_at)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_reference_objects_zone_type_created_at ON reference_objects(zone_id, reference_type, created_at)"))
    db.commit()


def _get_zone_or_404(db: Session, zone_id: str) -> Zone:
    zone = db.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"zone not found: {zone_id}")
    return zone


async def _fetch_forecast(zone_id: str, horizon: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=FORECAST_SERVICE_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(
                f"{FORECAST_SERVICE_URL}/v1/forecast",
                params={"zone_id": zone_id, "horizon": horizon},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"forecast service unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"forecast service error: {response.text}")
    return response.json()


async def _fetch_weekly_forecast(zone_id: str, days: int, slot_minutes: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=FORECAST_SERVICE_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(
                f"{FORECAST_SERVICE_URL}/v1/forecast/weekly",
                params={"zone_id": zone_id, "days": days, "slot_minutes": slot_minutes},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"forecast weekly service unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"forecast weekly service error: {response.text}")
    return response.json()


async def _fetch_explanation(zone_id: str, horizon: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{XAI_SERVICE_URL}/v1/explain",
                params={"zone_id": zone_id, "horizon": horizon},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"xai service unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"xai service error: {response.text}")
    return response.json()


async def _fetch_recommendations(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{RECOMMENDATIONS_SERVICE_URL}/v1/recommendations", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"recommendations service unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"recommendations service error: {response.text}")
    return response.json()


async def _service_health(service: str, base_url: str) -> dict[str, Any]:
    start = time.perf_counter()
    url = f"{base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(url)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            return {
                "service": service,
                "status": "degraded",
                "latency_ms": latency_ms,
                "detail": f"http_{response.status_code}",
            }
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "service": service,
            "status": "ok",
            "latency_ms": latency_ms,
            "detail": str(payload.get("status", "ok")),
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "service": service,
            "status": "down",
            "latency_ms": latency_ms,
            "detail": str(exc),
        }


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _normalize_dt(dt)


def _to_latest_response(
    payload: dict[str, Any],
    source: Literal["snapshot", "on_demand_fallback"],
    stale_seconds: int,
) -> ForecastLatestResponse:
    generated_at = _coerce_dt(payload.get("generated_at", datetime.now(UTC).isoformat()))
    age_seconds = max(0, int((datetime.now(UTC) - generated_at).total_seconds()))
    return ForecastLatestResponse(
        zone_id=str(payload.get("zone_id", "")),
        horizon=int(payload.get("horizon", 0)),
        generated_at=generated_at,
        age_seconds=age_seconds,
        stale=age_seconds > stale_seconds,
        summary=str(payload.get("summary", "")),
        model_version=str(payload.get("model_version", "")),
        points=list(payload.get("points", [])),
        evidence=dict(payload.get("evidence", {})),
        lineage=_normalize_lineage_payload(payload.get("lineage"), PRODUCT_SHORT_TERM),
        source=source,
    )


def _to_weekly_latest_response(
    payload: dict[str, Any],
    source: Literal["snapshot", "on_demand_fallback"],
    stale_seconds: int,
) -> WeeklyForecastResponse:
    generated_at = _coerce_dt(payload.get("generated_at", datetime.now(UTC).isoformat()))
    age_seconds = max(0, int((datetime.now(UTC) - generated_at).total_seconds()))
    return WeeklyForecastResponse(
        zone_id=str(payload.get("zone_id", "")),
        product=str(payload.get("product", PRODUCT_WEEKLY)),
        days=int(payload.get("days", WEEKLY_FORECAST_DAYS_DEFAULT)),
        slot_minutes=int(payload.get("slot_minutes", WEEKLY_FORECAST_SLOT_MINUTES)),
        generated_at=generated_at,
        age_seconds=age_seconds,
        stale=age_seconds > stale_seconds,
        summary=str(payload.get("summary", "")),
        model_version=str(payload.get("model_version", "")),
        points=list(payload.get("points", [])),
        daily_summaries=list(payload.get("daily_summaries", [])),
        evidence=dict(payload.get("evidence", {})),
        lineage=_normalize_lineage_payload(payload.get("lineage"), PRODUCT_WEEKLY),
        source=source,
    )


def _forecast_quality_flags(payload: dict[str, Any]) -> set[str]:
    evidence = payload.get("evidence", {}) if isinstance(payload.get("evidence"), dict) else {}
    quality = evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}
    return {str(flag).upper() for flag in quality.get("flags", []) if isinstance(flag, str)}


def _forecast_payload_has_model_fallback(payload: dict[str, Any]) -> bool:
    model_version = str(payload.get("model_version", "")).lower()
    flags = _forecast_quality_flags(payload)
    return (
        "baseline" in model_version
        or any(flag == "TF_FALLBACK" or flag.startswith("TF_FALLBACK:") for flag in flags)
        or any(flag == "LGBM_FALLBACK" or flag.startswith("LGBM_FALLBACK:") for flag in flags)
    )


def _latest_forecast_needs_refresh(forecast: ForecastLatestResponse) -> bool:
    return forecast.stale or _forecast_payload_has_model_fallback(forecast.model_dump(mode="json"))


async def _resolve_latest_forecast(
    db: Session,
    zone_id: str,
    horizon: int,
    stale_seconds: int,
) -> ForecastLatestResponse:
    snapshot = (
        db.query(ForecastSnapshot)
        .filter(
            ForecastSnapshot.zone_id == zone_id,
            ForecastSnapshot.product == PRODUCT_SHORT_TERM,
            ForecastSnapshot.horizon == horizon,
        )
        .order_by(ForecastSnapshot.generated_at.desc())
        .first()
    )
    if snapshot is not None:
        db.rollback()
        resolved = _to_latest_response(payload=snapshot.payload, source="snapshot", stale_seconds=stale_seconds)
        if _latest_forecast_needs_refresh(resolved):
            try:
                payload = await _fetch_forecast(zone_id=zone_id, horizon=horizon)
                try:
                    _persist_forecast_snapshot(db=db, payload=payload)
                except Exception:
                    db.rollback()
                resolved = _to_latest_response(payload=payload, source="on_demand_fallback", stale_seconds=stale_seconds)
            except Exception:
                db.rollback()
        adjusted, _ = _apply_booking_overlay_to_forecast(db, zone_id=zone_id, horizon=horizon, forecast=resolved)
        return adjusted

    db.rollback()
    payload = await _fetch_forecast(zone_id=zone_id, horizon=horizon)
    try:
        _persist_forecast_snapshot(db=db, payload=payload)
    except Exception:
        db.rollback()
    resolved = _to_latest_response(payload=payload, source="on_demand_fallback", stale_seconds=stale_seconds)
    adjusted, _ = _apply_booking_overlay_to_forecast(db, zone_id=zone_id, horizon=horizon, forecast=resolved)
    return adjusted


async def _resolve_latest_weekly_forecast(
    db: Session,
    zone_id: str,
    *,
    days: int,
    slot_minutes: int,
    stale_seconds: int,
) -> WeeklyForecastResponse:
    snapshot = (
        db.query(ForecastSnapshot)
        .filter(
            ForecastSnapshot.zone_id == zone_id,
            ForecastSnapshot.product == PRODUCT_WEEKLY,
            ForecastSnapshot.days == days,
            ForecastSnapshot.slot_minutes == slot_minutes,
        )
        .order_by(ForecastSnapshot.generated_at.desc())
        .first()
    )
    if snapshot is not None:
        db.rollback()
        return _to_weekly_latest_response(payload=snapshot.payload, source="snapshot", stale_seconds=stale_seconds)

    db.rollback()
    payload = await _fetch_weekly_forecast(zone_id=zone_id, days=days, slot_minutes=slot_minutes)
    try:
        _persist_forecast_snapshot(db=db, payload=payload)
    except Exception:
        db.rollback()
    return _to_weekly_latest_response(payload=payload, source="on_demand_fallback", stale_seconds=stale_seconds)


async def _resolve_adjusted_explanation(
    db: Session,
    *,
    zone_id: str,
    horizon: int,
    forecast: ForecastLatestResponse | None = None,
) -> dict[str, Any]:
    resolved_forecast = forecast
    if resolved_forecast is None:
        resolved_forecast = await _resolve_latest_forecast(
            db=db,
            zone_id=zone_id,
            horizon=horizon,
            stale_seconds=FORECAST_STALE_THRESHOLD_SECONDS,
        )
    raw_explanation = await _fetch_explanation(zone_id=zone_id, horizon=horizon)
    _, overlay = _apply_booking_overlay_to_forecast(
        db,
        zone_id=zone_id,
        horizon=horizon,
        forecast=resolved_forecast,
    )
    return _apply_booking_overlay_to_explanation(raw_explanation, zone_id=zone_id, overlay=overlay)


def _persist_forecast_snapshot(db: Session, payload: dict[str, Any]) -> ForecastSnapshot:
    generated_at = _coerce_dt(payload.get("generated_at", datetime.now(UTC).isoformat()))
    lineage = payload.get("lineage", {}) if isinstance(payload.get("lineage"), dict) else {}
    snapshot = ForecastSnapshot(
        forecast_id=f"fc-{uuid.uuid4()}",
        zone_id=str(payload.get("zone_id", "")),
        horizon=int(payload.get("horizon", 0)),
        product=str(payload.get("product", PRODUCT_SHORT_TERM)),
        slot_minutes=int(payload.get("slot_minutes")) if payload.get("slot_minutes") is not None else None,
        days=int(payload.get("days")) if payload.get("days") is not None else None,
        model_run_id=str(lineage.get("model_run_id")) if lineage.get("model_run_id") else None,
        generated_at=generated_at,
        model_version=str(payload.get("model_version", "")),
        payload=payload,
        evidence=dict(payload.get("evidence", {})),
    )
    db.add(snapshot)

    retention_cutoff = datetime.now(UTC) - timedelta(days=FORECAST_SNAPSHOT_RETENTION_DAYS)
    db.query(ForecastSnapshot).filter(ForecastSnapshot.created_at < retention_cutoff).delete(synchronize_session=False)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def _latest_live_state(db: Session, zone_id: str, history_minutes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    now = datetime.now(UTC)
    start = now - timedelta(minutes=history_minutes)
    dialect = (db.bind.dialect.name if db.bind is not None else "").lower()
    if dialect == "postgresql":
        minute_bucket_expr = func.date_trunc("minute", Count.ts)
    else:
        minute_bucket_expr = func.strftime("%Y-%m-%dT%H:%M:00Z", Count.ts)

    aggregated_rows = db.execute(
        select(
            minute_bucket_expr.label("minute_bucket"),
            func.avg(Count.occupancy).label("occupancy_avg"),
            func.avg(Count.utilization).label("utilization_avg"),
            func.avg(Count.quality_score).label("quality_avg"),
            func.count(Count.id).label("sample_count"),
        )
        .where(Count.zone_id == zone_id)
        .where(Count.ts >= start)
        .where(Count.ts <= now)
        .group_by(minute_bucket_expr)
        .order_by(minute_bucket_expr.asc())
    ).all()

    history_points: list[dict[str, Any]] = []
    raw_sample_count = 0
    for row in aggregated_rows:
        row_map = row._mapping
        minute_value = row_map.get("minute_bucket")
        minute_iso = None
        if minute_value is not None:
            try:
                minute_iso = _coerce_dt(minute_value).isoformat()
            except Exception:
                minute_iso = str(minute_value)
        if minute_iso is None:
            continue

        try:
            occupancy_value = int(round(float(row_map.get("occupancy_avg") or 0.0)))
        except Exception:
            occupancy_value = 0
        try:
            utilization_value = float(row_map.get("utilization_avg") or 0.0)
        except Exception:
            utilization_value = 0.0
        try:
            quality_value = float(row_map.get("quality_avg") or 0.0)
        except Exception:
            quality_value = 0.0
        try:
            sample_count = int(row_map.get("sample_count") or 0)
        except Exception:
            sample_count = 0
        raw_sample_count += max(sample_count, 0)

        history_points.append(
            {
                "timestamp": minute_iso,
                "zone_id": zone_id,
                "occupancy": occupancy_value,
                "utilization": utilization_value,
                "source": "aggregated-1m",
                "quality_score": quality_value,
                "quality_flags": ["AGGREGATED_1M"],
                "evidence": _normalize_evidence_payload({}, now=now, zone_id=zone_id),
            }
        )

    latest = (
        db.execute(
            select(Count)
            .where(Count.zone_id == zone_id)
            .where(Count.ts >= start)
            .where(Count.ts <= now)
            .order_by(Count.ts.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    if latest is None:
        live = {
            "timestamp": None,
            "occupancy": 0,
            "utilization": 0.0,
            "quality_score": 0.0,
            "quality_flags": ["NO_DATA"],
            "point_count": 0,
            "raw_point_count": 0,
        }
        return history_points, live

    live = {
        "timestamp": _normalize_dt(latest.ts).isoformat(),
        "occupancy": int(latest.occupancy),
        "utilization": float(latest.utilization),
        "quality_score": float(latest.quality_score),
        "quality_flags": latest.quality_flags or [],
        "point_count": len(history_points),
        "raw_point_count": raw_sample_count,
    }
    return history_points, live


def _build_alerts(
    *,
    live: dict[str, Any],
    latest_forecast: ForecastLatestResponse,
    explanation: dict[str, Any],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []

    if live.get("point_count", 0) == 0:
        alerts.append({"code": "NO_DATA", "level": "warn", "message": "Keine aktuellen Count-Daten vorhanden."})

    quality_score = float(live.get("quality_score", 0.0))
    live_flags = {str(flag).upper() for flag in live.get("quality_flags", [])}
    hard_live_flags = {"TRACK_ERROR", "SERIALIZATION_ERROR", "BACKLOG_OVERFLOW", "ZONE_MISSING"}
    if quality_score < 0.7 or bool(live_flags & hard_live_flags):
        alerts.append(
            {
                "code": "QUALITY_RISK",
                "level": "warn",
                "message": "Datenqualität ist reduziert. Empfehlungen mit Vorsicht verwenden.",
            }
        )

    if latest_forecast.stale:
        alerts.append({"code": "STALE", "level": "risk", "message": "Forecast-Snapshot ist veraltet."})

    model_version = latest_forecast.model_version.lower()
    forecast_flags = {
        str(flag).upper()
        for flag in latest_forecast.evidence.get("quality", {}).get("flags", [])
        if isinstance(flag, str)
    }
    fallback_active = (
        "baseline" in model_version
        or any(flag == "TF_FALLBACK" or flag.startswith("TF_FALLBACK:") for flag in forecast_flags)
        or any(flag == "LGBM_FALLBACK" or flag.startswith("LGBM_FALLBACK:") for flag in forecast_flags)
    )
    if fallback_active:
        alerts.append(
            {
                "code": "BASELINE_FALLBACK",
                "level": "info",
                "message": "Fallback aktiv: Snapshot stammt nicht aus primärem Modellpfad.",
            }
        )
    if {"CONTEXT_STALE", "NO_CONTEXT_DB"} & forecast_flags:
        alerts.append(
            {
                "code": "FORECAST_CONTEXT_DEGRADED",
                "level": "risk",
                "message": "Forecast-Kontext ist veraltet oder fehlt. Prognose nicht als normal belastbar behandeln.",
            }
        )
    if {"LOW_INPUT_QUALITY", "CONTEXT_LOW_QUALITY"} & forecast_flags:
        alerts.append(
            {
                "code": "FORECAST_QUALITY_DEGRADED",
                "level": "risk",
                "message": "Forecast-Datenqualität ist reduziert. Prognose nicht als normale Entscheidungsgrundlage verwenden.",
            }
        )

    uncertainty = explanation.get("uncertainty", {}) if isinstance(explanation, dict) else {}
    level = str(uncertainty.get("level", "high")).lower()
    score = float(uncertainty.get("score", 1.0) or 1.0)
    if level == "high" or score >= 0.7:
        alerts.append(
            {
                "code": "UNCERTAINTY_HIGH",
                "level": "warn",
                "message": "Unsicherheit ist hoch. Harte Maßnahmen sollten abgesichert werden.",
            }
        )

    if not alerts:
        alerts.append({"code": "ALL_CLEAR", "level": "ok", "message": "Keine kritischen Warnungen."})
    return alerts


def _normalize_explain_audience(audience: str) -> str:
    if audience == "professor" and not EXPLAINABILITY_PROFESSOR_MODE_ENABLED:
        return "executive"
    return audience


def _read_lecture_impact_latest(db: Session, zone_id: str) -> dict[str, Any] | None:
    row = (
        db.execute(
            select(LectureActivity)
            .where(LectureActivity.zone_id == zone_id)
            .order_by(LectureActivity.ts.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if row is None:
        return None

    metadata = row.metadata_json or {}

    def _meta_float(key: str, default: float = 0.0) -> float:
        try:
            value = float(metadata.get(key, default))
        except Exception:
            return default
        return value if value == value else default

    return {
        "zone_id": zone_id,
        "timestamp": _normalize_dt(row.ts).isoformat(),
        "source": row.source,
        "quality_score": float(row.quality_score),
        "quality_flags": row.quality_flags or [],
        "active_lectures": int(row.active_lectures),
        "active_courses": int(row.active_courses),
        "starts_next_60m": int(row.starts_next_60m),
        "ends_next_60m": int(row.ends_next_60m),
        "impact": {
            "heavy_active_lectures": int(round(_meta_float("heavy_active_lectures", 0.0))),
            "heavy_ended_last_60m": int(round(_meta_float("heavy_ended_last_60m", 0.0))),
            "lecture_pull_regular": _meta_float("lecture_pull_regular", 0.0),
            "heavy_bib_bonus": _meta_float("heavy_bib_bonus", 0.0),
            "lecture_net_pull": _meta_float("lecture_net_pull", 0.0),
            "impact_model_version": str(metadata.get("impact_model_version") or ""),
        },
        "metadata": metadata,
    }


def _latest_model_lineage(db: Session, zone_id: str, product: str, horizon: int | None = None) -> dict[str, Any] | None:
    query = select(ModelRun).where(ModelRun.zone_id == zone_id).where(ModelRun.product == product)
    if horizon is not None:
        query = query.where((ModelRun.horizon == horizon) | (ModelRun.horizon.is_(None)))
    row = db.execute(query.order_by(ModelRun.promoted_at.desc(), ModelRun.created_at.desc())).scalars().first()
    if row is not None:
        payload = {
            "model_run_id": row.model_run_id,
            "zone_id": row.zone_id,
            "product": row.product,
            "horizon": row.horizon,
            "model_backend": row.model_backend,
            "model_version": row.model_version,
            "status": row.status,
            "scientific_status": row.scientific_status,
            "include_lecture_impact": row.include_lecture_impact,
            "feature_set_version": row.feature_set_version,
            "history_from": row.history_from.isoformat() if row.history_from else None,
            "history_to": row.history_to.isoformat() if row.history_to else None,
            "raw_rows": row.raw_rows,
            "train_rows": row.train_rows,
            "val_rows": row.val_rows,
            "test_rows": row.test_rows,
            "evaluation_run_id": row.evaluation_run_id,
            "promoted": row.promoted,
            "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
            "promotion_source": row.promotion_source,
            "metadata": row.metadata_json or {},
            "lineage": row.lineage_json or {},
            "created_at": row.created_at.isoformat(),
        }
        references = payload["lineage"].get("reference_objects", []) if isinstance(payload["lineage"], dict) else []
        payload["reference_objects"] = references if isinstance(references, list) else []
        return payload

    snapshot_query = db.query(ForecastSnapshot).filter(ForecastSnapshot.zone_id == zone_id, ForecastSnapshot.product == product)
    if product == PRODUCT_SHORT_TERM and horizon is not None:
        snapshot_query = snapshot_query.filter(ForecastSnapshot.horizon == horizon)
    snapshot = snapshot_query.order_by(ForecastSnapshot.generated_at.desc()).first()
    if snapshot is None:
        return None
    payload = snapshot.payload if isinstance(snapshot.payload, dict) else {}
    lineage = payload.get("lineage", {}) if isinstance(payload.get("lineage"), dict) else {}
    if not lineage:
        return None
    return {
        "model_run_id": lineage.get("model_run_id"),
        "zone_id": zone_id,
        "product": product,
        "horizon": payload.get("horizon"),
        "model_backend": lineage.get("backend", ""),
        "model_version": payload.get("model_version", ""),
        "status": "snapshot_only",
        "scientific_status": lineage.get("scientific_status", ""),
        "include_lecture_impact": bool(lineage.get("include_lecture_impact", True)),
        "feature_set_version": lineage.get("feature_set_version", ""),
        "history_from": lineage.get("history_window", {}).get("from"),
        "history_to": lineage.get("history_window", {}).get("to"),
        "raw_rows": None,
        "train_rows": None,
        "val_rows": None,
        "test_rows": None,
        "evaluation_run_id": None,
        "promoted": bool(lineage.get("model_run_id")),
        "promoted_at": snapshot.generated_at.isoformat(),
        "promotion_source": "snapshot_fallback",
        "metadata": {},
        "lineage": lineage,
        "reference_objects": lineage.get("reference_objects", []) if isinstance(lineage.get("reference_objects"), list) else [],
        "created_at": snapshot.created_at.isoformat(),
    }


async def _build_explainability_context(
    *,
    db: Session,
    zone_id: str,
    horizon: int,
    audience: str,
    language: str,
    query: str,
) -> dict[str, Any]:
    zone = _get_zone_or_404(db, zone_id)
    history_minutes = max(180, horizon * 3)
    history_points, live = _latest_live_state(db=db, zone_id=zone_id, history_minutes=history_minutes)
    db.rollback()
    latest_forecast = await _resolve_latest_forecast(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        stale_seconds=FORECAST_STALE_THRESHOLD_SECONDS,
    )
    db.rollback()
    explanation = await _resolve_adjusted_explanation(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        forecast=latest_forecast,
    )
    recommendations = await _fetch_recommendations(
        {
            "zone_id": zone_id,
            "horizon": horizon,
            "capacity": zone.capacity,
            "forecast": latest_forecast.model_dump(mode="json"),
            "explanation": explanation,
        }
    )
    alerts = _build_alerts(live=live, latest_forecast=latest_forecast, explanation=explanation)
    lecture_impact = _read_lecture_impact_latest(db=db, zone_id=zone_id)
    return build_explainability_context_v2(
        zone_id=zone_id,
        zone_capacity=zone.capacity,
        horizon=horizon,
        audience=audience,
        language=language,
        query=query,
        forecast_latest=latest_forecast.model_dump(mode="json"),
        explanation=explanation,
        recommendation=recommendations,
        history_points=history_points,
        live_state=live,
        alerts=alerts,
        lecture_impact=lecture_impact,
        template_set_id=PROMPT_REGISTRY.template_set_id,
        prompt_version=PROMPT_REGISTRY.prompt_version,
    )


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        zone = _ensure_default_zone(db)
        _seed_mock_calendar(db, zone.zone_id)
        _ensure_sqlite_indexes(db)
        _ensure_sqlite_schema_extensions(db)


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "service": "api-gateway"}


@app.get("/api/v1/zones", response_model=list[ZoneResponse])
def get_zones(db: Session = Depends(get_db)) -> list[ZoneResponse]:
    zones = db.execute(select(Zone).order_by(Zone.zone_id.asc())).scalars().all()
    return [
        ZoneResponse(
            zone_id=z.zone_id,
            name=z.name,
            capacity=z.capacity,
            is_active=z.is_active,
            metadata=z.metadata_json,
        )
        for z in zones
    ]


@app.post("/api/v1/auth/register", response_model=AuthResponse, status_code=201)
def auth_register(
    payload: AuthRegisterRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    _delete_expired_sessions(db)
    username, username_normalized = _normalize_username_input(payload.username)
    existing = (
        db.execute(select(User).where(User.username_normalized == username_normalized).limit(1))
        .scalars()
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="username_exists")

    salt_hex, password_hash = _create_password_credentials(payload.password)
    user = User(
        user_id=f"user-{uuid.uuid4()}",
        username=username,
        username_normalized=username_normalized,
        role=_normalize_user_role(payload.role),
        password_hash=password_hash,
        password_salt=salt_hex,
    )
    db.add(user)
    _, raw_token = _create_user_session(db, user_id=user.user_id)
    db.commit()
    _set_session_cookie(response, request, raw_token)
    return AuthResponse(authenticated=True, user=_serialize_user(user))


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def auth_login(
    payload: AuthCredentialsRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    _delete_expired_sessions(db)
    _, username_normalized = _normalize_username_input(payload.username)
    user = (
        db.execute(select(User).where(User.username_normalized == username_normalized).limit(1))
        .scalars()
        .first()
    )
    if user is None or not _verify_password(
        payload.password,
        salt_hex=user.password_salt,
        expected_hash=user.password_hash,
    ):
        raise HTTPException(status_code=401, detail="invalid_credentials")

    _, raw_token = _create_user_session(db, user_id=user.user_id)
    db.commit()
    _set_session_cookie(response, request, raw_token)
    return AuthResponse(authenticated=True, user=_serialize_user(user))


@app.get("/api/v1/auth/me", response_model=AuthResponse)
def auth_me(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    _delete_expired_sessions(db)
    user, _ = _resolve_session_user(request=request, db=db)
    if user is None:
        _clear_session_cookie(response)
        return AuthResponse(authenticated=False, user=None)
    return AuthResponse(authenticated=True, user=_serialize_user(user))


@app.post("/api/v1/auth/logout", response_model=AuthResponse)
def auth_logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthResponse:
    _, session = _resolve_session_user(request=request, db=db)
    if session is not None:
        db.delete(session)
        db.commit()
    _clear_session_cookie(response)
    return AuthResponse(authenticated=False, user=None)


@app.get("/api/v1/bookings", response_model=list[BookingResponse])
def list_bookings(
    request: Request,
    db: Session = Depends(get_db),
) -> list[BookingResponse]:
    user, _ = _require_session_user(request=request, db=db)
    bookings = (
        db.execute(
            select(Booking)
            .where(Booking.user_id == user.user_id)
            .order_by(Booking.starts_at.asc(), Booking.created_at.asc())
        )
        .scalars()
        .all()
    )
    return [_serialize_booking(booking) for booking in bookings]


@app.get("/api/v1/admin/bookings", response_model=list[AdminBookingResponse])
def list_admin_bookings(
    request: Request,
    db: Session = Depends(get_db),
) -> list[AdminBookingResponse]:
    _require_admin_user(request=request, db=db)
    day_start = _local_day_start_utc()
    rows = (
        db.execute(
            select(Booking, User)
            .join(User, User.user_id == Booking.user_id)
            .where(Booking.status == "confirmed")
            .where(Booking.ends_at >= day_start)
            .order_by(Booking.starts_at.asc(), Booking.created_at.asc())
        )
        .all()
    )
    return [_serialize_admin_booking(booking, user) for booking, user in rows]


@app.post("/api/v1/bookings", response_model=BookingResponse, status_code=201)
def create_booking(
    payload: BookingCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> BookingResponse:
    user, _ = _require_session_user(request=request, db=db)
    zone = _get_zone_or_404(db, DEFAULT_ZONE_ID)
    starts_at = _normalize_dt(payload.starts_at)
    ends_at = _normalize_dt(payload.ends_at)
    now = datetime.now(UTC)
    if starts_at < now:
        raise HTTPException(status_code=400, detail="booking_must_start_in_future")
    if ends_at <= starts_at:
        raise HTTPException(status_code=400, detail="booking_end_before_start")

    booking = Booking(
        booking_id=f"booking-{uuid.uuid4()}",
        user_id=user.user_id,
        zone_id=zone.zone_id,
        starts_at=starts_at,
        ends_at=ends_at,
        party_size=1,
        status="confirmed",
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return _serialize_booking(booking)


@app.delete("/api/v1/bookings/{booking_id}", response_model=BookingResponse)
def cancel_booking(
    booking_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> BookingResponse:
    user, _ = _require_session_user(request=request, db=db)
    booking = (
        db.execute(
            select(Booking)
            .where(Booking.booking_id == booking_id)
            .where(Booking.user_id == user.user_id)
            .limit(1)
        )
        .scalars()
        .first()
    )
    if booking is None:
        raise HTTPException(status_code=404, detail="booking_not_found")
    if str(booking.status) == "cancelled":
        return _serialize_booking(booking)
    if _normalize_dt(booking.ends_at) <= datetime.now(UTC):
        raise HTTPException(status_code=409, detail="booking_already_completed")

    booking.status = "cancelled"
    booking.cancelled_at = datetime.now(UTC)
    db.commit()
    db.refresh(booking)
    return _serialize_booking(booking)


@app.post("/api/v1/references/training-data/register")
def register_training_data_reference(
    request: TrainingDataReferenceRegisterRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if request.zone_id:
        _ = _get_zone_or_404(db, request.zone_id)

    seed = "|".join(
        [
            str(request.zone_id or ""),
            "training_data",
            str(request.source_type or ""),
            str(request.label or ""),
            str(request.uri_or_path or ""),
            str(request.checksum or ""),
        ]
    )
    reference_id = f"ref-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]}"
    row = db.get(ReferenceObject, reference_id)
    if row is None:
        row = ReferenceObject(reference_id=reference_id)
        db.add(row)
    row.zone_id = request.zone_id
    row.reference_type = "training_data"
    row.source_type = request.source_type
    row.label = request.label
    row.uri_or_path = request.uri_or_path
    row.checksum = request.checksum
    row.imported_at = _coerce_optional_dt(request.imported_at)
    row.time_from = _coerce_optional_dt(request.time_from)
    row.time_to = _coerce_optional_dt(request.time_to)
    row.row_count = request.row_count
    row.metadata_json = {
        **(request.metadata or {}),
        "ingest_job_id": request.ingest_job_id,
    }
    db.commit()
    db.refresh(row)
    return {"status": "ok", "reference": _reference_payload_to_public(row)}


@app.post("/api/v1/ingest/counts")
def ingest_counts(payload: CountIngestRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not payload.points:
        raise HTTPException(status_code=400, detail="points must not be empty")

    inserted = 0
    now = datetime.now(UTC)

    for point in payload.points:
        zone = db.get(Zone, point.zone_id)
        if zone is None:
            raise HTTPException(status_code=404, detail=f"zone not found: {point.zone_id}")

        ts = _normalize_dt(point.timestamp)
        util = point.utilization if point.utilization is not None else point.occupancy / max(zone.capacity, 1)
        evidence = (
            _normalize_evidence_payload(point.evidence.model_dump(mode="json", by_alias=True), now=now, zone_id=point.zone_id)
            if point.evidence
            else _default_evidence(now, point.zone_id)
        )

        db.add(
            Count(
                ts=ts,
                zone_id=point.zone_id,
                occupancy=point.occupancy,
                utilization=util,
                source=point.source,
                quality_score=point.quality_score,
                quality_flags=point.quality_flags,
                evidence=evidence,
            )
        )
        inserted += 1

    db.commit()
    return {"status": "ok", "inserted": inserted}


@app.get("/api/v1/counts", response_model=CountsResponse)
def get_counts(
    zone_id: str,
    from_: datetime = Query(alias="from"),
    to: datetime = Query(),
    granularity: Literal["raw", "1m", "5m", "15m", "60m"] = Query(default="1m"),
    db: Session = Depends(get_db),
) -> CountsResponse:
    _safe_granularity(granularity)
    _ = _get_zone_or_404(db, zone_id)

    from_dt = _normalize_dt(from_)
    to_dt = _normalize_dt(to)

    if granularity == "raw":
        rows = db.execute(
            select(Count)
            .where(Count.zone_id == zone_id)
            .where(Count.ts >= from_dt)
            .where(Count.ts <= to_dt)
            .order_by(Count.ts.asc())
        ).scalars().all()

        if not rows:
            return CountsResponse.model_validate(
                {
                    "zone_id": zone_id,
                    "from": from_dt,
                    "to": to_dt,
                    "granularity": granularity,
                    "points": [],
                }
            )

        points = [
            {
                "timestamp": r.ts,
                "zone_id": r.zone_id,
                "occupancy": int(r.occupancy),
                "utilization": float(r.utilization),
                "source": r.source,
                "quality_score": float(r.quality_score),
                "quality_flags": r.quality_flags or [],
                "evidence": _normalize_evidence_payload(r.evidence, now=datetime.now(UTC), zone_id=zone_id),
            }
            for r in rows
        ]
    else:
        rows = db.execute(
            select(
                Count.ts.label("timestamp"),
                Count.occupancy.label("occupancy"),
                Count.utilization.label("utilization"),
                Count.quality_score.label("quality_score"),
            )
            .where(Count.zone_id == zone_id)
            .where(Count.ts >= from_dt)
            .where(Count.ts <= to_dt)
            .order_by(Count.ts.asc())
        ).all()

        if not rows:
            return CountsResponse.model_validate(
                {
                    "zone_id": zone_id,
                    "from": from_dt,
                    "to": to_dt,
                    "granularity": granularity,
                    "points": [],
                }
            )

        df = pd.DataFrame(
            [
                {
                    "timestamp": _normalize_dt(row._mapping["timestamp"]),
                    "occupancy": float(row._mapping["occupancy"]),
                    "utilization": float(row._mapping["utilization"]),
                    "quality_score": float(row._mapping["quality_score"]),
                }
                for row in rows
            ]
        )
        df = df.set_index("timestamp").sort_index()
        freq = {"1m": "1min", "5m": "5min", "15m": "15min", "60m": "60min"}[granularity]
        agg = df.resample(freq).mean(numeric_only=True).dropna()

        points = [
            {
                "timestamp": idx.to_pydatetime().replace(tzinfo=UTC),
                "zone_id": zone_id,
                "occupancy": int(round(float(row["occupancy"]))),
                "utilization": float(row["utilization"]),
                "source": "aggregated",
                "quality_score": float(row["quality_score"]),
                "quality_flags": ["AGGREGATED"],
                "evidence": _default_evidence(datetime.now(UTC), zone_id),
            }
            for idx, row in agg.iterrows()
        ]

    return CountsResponse.model_validate(
        {
            "zone_id": zone_id,
            "from": from_dt,
            "to": to_dt,
            "granularity": granularity,
            "points": points,
        }
    )


@app.get("/api/v1/lectures/activity", response_model=LectureActivityResponse)
def get_lecture_activity(
    zone_id: str,
    from_: datetime = Query(alias="from"),
    to: datetime = Query(),
    granularity: Literal["raw", "1m", "5m", "15m", "60m"] = Query(default="1m"),
    db: Session = Depends(get_db),
) -> LectureActivityResponse:
    _safe_granularity(granularity)
    _ = _get_zone_or_404(db, zone_id)

    from_dt = _normalize_dt(from_)
    to_dt = _normalize_dt(to)

    rows = (
        db.execute(
            select(LectureActivity)
            .where(LectureActivity.zone_id == zone_id)
            .where(LectureActivity.ts >= from_dt)
            .where(LectureActivity.ts <= to_dt)
            .order_by(LectureActivity.ts.asc())
        )
        .scalars()
        .all()
    )

    if not rows:
        return LectureActivityResponse.model_validate(
            {
                "zone_id": zone_id,
                "from": from_dt,
                "to": to_dt,
                "granularity": granularity,
                "points": [],
            }
        )

    if granularity in {"raw", "1m"}:
        points = [
            {
                "timestamp": row.ts,
                "zone_id": row.zone_id,
                "active_lectures": row.active_lectures,
                "active_courses": row.active_courses,
                "starts_next_60m": row.starts_next_60m,
                "ends_next_60m": row.ends_next_60m,
                "source": row.source,
                "quality_score": row.quality_score,
                "quality_flags": row.quality_flags or [],
                "metadata": row.metadata_json or {},
            }
            for row in rows
        ]
        return LectureActivityResponse.model_validate(
            {
                "zone_id": zone_id,
                "from": from_dt,
                "to": to_dt,
                "granularity": granularity,
                "points": points,
            }
        )

    df = pd.DataFrame(
        {
            "timestamp": [row.ts for row in rows],
            "active_lectures": [row.active_lectures for row in rows],
            "active_courses": [row.active_courses for row in rows],
            "starts_next_60m": [row.starts_next_60m for row in rows],
            "ends_next_60m": [row.ends_next_60m for row in rows],
            "quality_score": [row.quality_score for row in rows],
        }
    )
    bucket = {"5m": "5min", "15m": "15min", "60m": "60min"}[granularity]
    grouped = (
        df.set_index("timestamp")
        .resample(bucket)
        .mean(numeric_only=True)
        .dropna(how="all")
        .reset_index()
    )
    points = [
        {
            "timestamp": rec["timestamp"],
            "zone_id": zone_id,
            "active_lectures": int(round(float(rec["active_lectures"]))),
            "active_courses": int(round(float(rec["active_courses"]))),
            "starts_next_60m": int(round(float(rec["starts_next_60m"]))),
            "ends_next_60m": int(round(float(rec["ends_next_60m"]))),
            "source": "aggregated",
            "quality_score": float(rec["quality_score"]) if rec["quality_score"] == rec["quality_score"] else 0.0,
            "quality_flags": ["AGGREGATED"],
            "metadata": {},
        }
        for rec in grouped.to_dict(orient="records")
    ]
    return LectureActivityResponse.model_validate(
        {
            "zone_id": zone_id,
            "from": from_dt,
            "to": to_dt,
            "granularity": granularity,
            "points": points,
        }
    )


@app.get("/api/v1/lectures/impact/latest")
def get_lecture_impact_latest(
    zone_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ = _get_zone_or_404(db, zone_id)
    payload = _read_lecture_impact_latest(db=db, zone_id=zone_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no lecture activity found for zone={zone_id}")
    return payload


@app.get("/api/v1/calendar/events", response_model=list[CalendarEventResponse])
def get_calendar_events(
    zone_id: str | None = None,
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[CalendarEventResponse]:
    from_dt = _normalize_dt(from_) if from_ else datetime.now(UTC) - timedelta(days=1)
    to_dt = _normalize_dt(to) if to else datetime.now(UTC) + timedelta(days=7)

    query = select(CalendarEvent).where(CalendarEvent.ends_at >= from_dt).where(CalendarEvent.starts_at <= to_dt)
    if zone_id:
        query = query.where((CalendarEvent.zone_id == zone_id) | (CalendarEvent.zone_id.is_(None)))

    events = db.execute(query.order_by(CalendarEvent.starts_at.asc())).scalars().all()
    return [
        CalendarEventResponse(
            event_id=e.event_id,
            zone_id=e.zone_id,
            title=e.title,
            category=e.category,
            starts_at=e.starts_at,
            ends_at=e.ends_at,
            expected_impact=e.expected_impact,
            source=e.source,
            metadata=e.metadata_json,
        )
        for e in events
    ]


@app.get("/api/v1/forecast")
async def get_forecast(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
) -> dict[str, Any]:
    return await _fetch_forecast(zone_id=zone_id, horizon=horizon)


@app.get("/api/v1/forecast/weekly")
async def get_weekly_forecast(
    zone_id: str,
    days: int = Query(default=WEEKLY_FORECAST_DAYS_DEFAULT, ge=1, le=14),
    slot_minutes: int = Query(default=WEEKLY_FORECAST_SLOT_MINUTES, ge=15, le=240),
) -> dict[str, Any]:
    return await _fetch_weekly_forecast(zone_id=zone_id, days=days, slot_minutes=slot_minutes)


@app.get("/api/v1/forecast/latest", response_model=ForecastLatestResponse)
async def get_forecast_latest(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    db: Session = Depends(get_db),
) -> ForecastLatestResponse:
    _ = _get_zone_or_404(db, zone_id)
    return await _resolve_latest_forecast(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        stale_seconds=stale_seconds,
    )


@app.get("/api/v1/forecast/weekly/latest", response_model=WeeklyForecastResponse)
async def get_weekly_forecast_latest(
    zone_id: str,
    days: int = Query(default=WEEKLY_FORECAST_DAYS_DEFAULT, ge=1, le=14),
    slot_minutes: int = Query(default=WEEKLY_FORECAST_SLOT_MINUTES, ge=15, le=240),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    db: Session = Depends(get_db),
) -> WeeklyForecastResponse:
    _ = _get_zone_or_404(db, zone_id)
    return await _resolve_latest_weekly_forecast(
        db=db,
        zone_id=zone_id,
        days=days,
        slot_minutes=slot_minutes,
        stale_seconds=stale_seconds,
    )


@app.get("/api/v1/forecast/history", response_model=ForecastHistoryResponse)
def get_forecast_history(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
    from_: datetime = Query(alias="from"),
    to: datetime = Query(),
    limit: int = Query(default=100, ge=1, le=1000),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    db: Session = Depends(get_db),
) -> ForecastHistoryResponse:
    _ = _get_zone_or_404(db, zone_id)
    from_dt = _normalize_dt(from_)
    to_dt = _normalize_dt(to)

    rows = (
        db.query(ForecastSnapshot)
        .filter(
            ForecastSnapshot.zone_id == zone_id,
            ForecastSnapshot.product == PRODUCT_SHORT_TERM,
            ForecastSnapshot.horizon == horizon,
            ForecastSnapshot.generated_at >= from_dt,
            ForecastSnapshot.generated_at <= to_dt,
        )
        .order_by(ForecastSnapshot.generated_at.desc())
        .limit(limit)
        .all()
    )
    items = [_to_latest_response(payload=r.payload, source="snapshot", stale_seconds=stale_seconds) for r in rows]
    return ForecastHistoryResponse.model_validate(
        {
            "zone_id": zone_id,
            "horizon": horizon,
            "from": from_dt,
            "to": to_dt,
            "items": items,
        }
    )


@app.get("/api/v1/forecast/weekly/history", response_model=WeeklyForecastHistoryResponse)
def get_weekly_forecast_history(
    zone_id: str,
    days: int = Query(default=WEEKLY_FORECAST_DAYS_DEFAULT, ge=1, le=14),
    slot_minutes: int = Query(default=WEEKLY_FORECAST_SLOT_MINUTES, ge=15, le=240),
    from_: datetime = Query(alias="from"),
    to: datetime = Query(),
    limit: int = Query(default=30, ge=1, le=200),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    db: Session = Depends(get_db),
) -> WeeklyForecastHistoryResponse:
    _ = _get_zone_or_404(db, zone_id)
    from_dt = _normalize_dt(from_)
    to_dt = _normalize_dt(to)
    rows = (
        db.query(ForecastSnapshot)
        .filter(
            ForecastSnapshot.zone_id == zone_id,
            ForecastSnapshot.product == PRODUCT_WEEKLY,
            ForecastSnapshot.days == days,
            ForecastSnapshot.slot_minutes == slot_minutes,
            ForecastSnapshot.generated_at >= from_dt,
            ForecastSnapshot.generated_at <= to_dt,
        )
        .order_by(ForecastSnapshot.generated_at.desc())
        .limit(limit)
        .all()
    )
    items = [_to_weekly_latest_response(payload=row.payload, source="snapshot", stale_seconds=stale_seconds) for row in rows]
    return WeeklyForecastHistoryResponse.model_validate(
        {
            "zone_id": zone_id,
            "days": days,
            "slot_minutes": slot_minutes,
            "from": from_dt,
            "to": to_dt,
            "items": items,
        }
    )


@app.post("/api/v1/internal/forecast/snapshot", response_model=ForecastLatestResponse)
async def create_forecast_snapshot(
    request: ForecastSnapshotRequest,
    db: Session = Depends(get_db),
    x_internal_token: str | None = Header(default=None),
) -> ForecastLatestResponse:
    _verify_internal_token(x_internal_token)
    _ = _get_zone_or_404(db, request.zone_id)

    payload = await _fetch_forecast(zone_id=request.zone_id, horizon=request.horizon)
    _persist_forecast_snapshot(db=db, payload=payload)
    return _to_latest_response(payload=payload, source="snapshot", stale_seconds=FORECAST_STALE_THRESHOLD_SECONDS)


@app.get("/api/v1/dashboard/command-center")
async def get_dashboard_command_center(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720),
    history_minutes: int = Query(default=180, ge=30, le=1440),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    long_term_days: int = Query(default=14, ge=1, le=42),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    zone = _get_zone_or_404(db, zone_id)
    generated_at = datetime.now(UTC)

    history_points, live = _latest_live_state(db=db, zone_id=zone_id, history_minutes=history_minutes)
    db.rollback()
    latest_forecast = await _resolve_latest_forecast(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        stale_seconds=stale_seconds,
    )
    weekly_forecast = await _resolve_latest_weekly_forecast(
        db=db,
        zone_id=zone_id,
        days=max(7, long_term_days),
        slot_minutes=WEEKLY_FORECAST_SLOT_MINUTES,
        stale_seconds=stale_seconds,
    )
    db.rollback()
    explanation = await _resolve_adjusted_explanation(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        forecast=latest_forecast,
    )
    recommendations = await _fetch_recommendations(
        {
            "zone_id": zone_id,
            "horizon": horizon,
            "capacity": zone.capacity,
            "forecast": latest_forecast.model_dump(mode="json"),
            "explanation": explanation,
        }
    )
    weekly_lineage = _latest_model_lineage(db=db, zone_id=zone_id, product=PRODUCT_WEEKLY)
    short_term_lineage = _latest_model_lineage(db=db, zone_id=zone_id, product=PRODUCT_SHORT_TERM, horizon=horizon)
    lecture_impact = _read_lecture_impact_latest(db=db, zone_id=zone_id)
    reference_rows = (
        db.execute(
            select(ReferenceObject)
            .where((ReferenceObject.zone_id == zone_id) | (ReferenceObject.zone_id.is_(None)))
            .order_by(ReferenceObject.imported_at.desc(), ReferenceObject.created_at.desc())
            .limit(8)
        )
        .scalars()
        .all()
    )
    reference_objects = [_reference_payload_to_public(row) for row in reference_rows]
    lecture_metadata = lecture_impact.get("metadata", {}) if isinstance(lecture_impact, dict) else {}
    for idx, item in enumerate(lecture_metadata.get("external_references", []) if isinstance(lecture_metadata, dict) else [], start=1):
        if not isinstance(item, dict):
            continue
        reference_objects.append(
            {
                "reference_id": f"ext-command-center-{idx}",
                "zone_id": zone_id,
                "reference_type": str(item.get("reference_type") or "external_reference"),
                "source_type": str(item.get("source_type") or "external"),
                "label": str(item.get("label") or "External reference"),
                "uri_or_path": item.get("url"),
                "checksum": None,
                "imported_at": generated_at.isoformat(),
                "time_from": None,
                "time_to": None,
                "row_count": None,
                "metadata": item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                "created_at": generated_at.isoformat(),
            }
        )

    db.rollback()
    calendar_events = get_calendar_events(
        zone_id=zone_id,
        from_=generated_at - timedelta(minutes=history_minutes),
        to=generated_at + timedelta(days=long_term_days),
        db=db,
    )
    weekly_explanation = build_weekly_explainability(
        zone_id=zone_id,
        weekly_forecast=weekly_forecast.model_dump(mode="json"),
        lecture_impact=lecture_impact,
        lineage=weekly_lineage or weekly_forecast.lineage,
        calendar_events=[event.model_dump(mode="json") for event in calendar_events],
    )

    long_horizons = sorted(
        {
            60,
            1440,
            10080,
            min(MAX_FORECAST_HORIZON_MINUTES, max(60, long_term_days * 24 * 60)),
        }
    )

    forecast_long_term: list[dict[str, Any]] = []
    for hz in long_horizons:
        if hz == horizon:
            forecast_long_term.append(latest_forecast.model_dump(mode="json"))
            continue

        snapshot = (
            db.query(ForecastSnapshot)
            .filter(
                ForecastSnapshot.zone_id == zone_id,
                ForecastSnapshot.product == PRODUCT_SHORT_TERM,
                ForecastSnapshot.horizon == hz,
            )
            .order_by(ForecastSnapshot.generated_at.desc())
            .first()
        )
        if snapshot is None:
            continue
        item = _to_latest_response(payload=snapshot.payload, source="snapshot", stale_seconds=stale_seconds)
        forecast_long_term.append(item.model_dump(mode="json"))
    db.rollback()

    upstream_health = await asyncio.gather(
        _service_health("forecast", FORECAST_SERVICE_URL),
        _service_health("xai", XAI_SERVICE_URL),
        _service_health("recommendations", RECOMMENDATIONS_SERVICE_URL),
        _service_health("scheduler", SCHEDULER_SERVICE_URL),
    )

    service_health = [
        {"service": "api-gateway", "status": "ok", "latency_ms": 0, "detail": "local"},
        *upstream_health,
    ]
    alerts = _build_alerts(live=live, latest_forecast=latest_forecast, explanation=explanation)

    return {
        "meta": {
            "generated_at": generated_at.isoformat(),
            "zone_id": zone_id,
            "horizon": horizon,
            "history_minutes": history_minutes,
            "long_term_days": long_term_days,
            "stale_seconds": stale_seconds,
            "environment": os.getenv("APP_ENV", "dev"),
        },
        "service_health": service_health,
        "live": live,
        "history": {
            "zone_id": zone_id,
            "from": (generated_at - timedelta(minutes=history_minutes)).isoformat(),
            "to": generated_at.isoformat(),
            "granularity": "1m",
            "points": history_points,
        },
        "forecast_latest": latest_forecast.model_dump(mode="json"),
        "forecast_long_term": forecast_long_term,
        "weekly_forecast": weekly_forecast.model_dump(mode="json"),
        "weekly_explanation": weekly_explanation,
        "explanation": explanation,
        "recommendations": recommendations,
        "model_lineage": {
            "short_term": short_term_lineage or latest_forecast.lineage or {},
            "weekly_slot": weekly_lineage or weekly_forecast.lineage or {},
        },
        "reference_objects": reference_objects,
        "calendar_events": [event.model_dump(mode="json") for event in calendar_events],
        "alerts": alerts,
    }


@app.get("/api/v1/explain")
async def get_explain(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ = _get_zone_or_404(db, zone_id)
    forecast = await _resolve_latest_forecast(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        stale_seconds=FORECAST_STALE_THRESHOLD_SECONDS,
    )
    db.rollback()
    return await _resolve_adjusted_explanation(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        forecast=forecast,
    )


@app.get("/api/v1/explain/weekly", response_model=WeeklyExplainResponse)
async def get_weekly_explain(
    zone_id: str,
    days: int = Query(default=WEEKLY_FORECAST_DAYS_DEFAULT, ge=1, le=14),
    slot_minutes: int = Query(default=WEEKLY_FORECAST_SLOT_MINUTES, ge=15, le=240),
    stale_seconds: int = Query(default=FORECAST_STALE_THRESHOLD_SECONDS, ge=30, le=86400),
    db: Session = Depends(get_db),
) -> WeeklyExplainResponse:
    _ = _get_zone_or_404(db, zone_id)
    weekly = await _resolve_latest_weekly_forecast(
        db=db,
        zone_id=zone_id,
        days=days,
        slot_minutes=slot_minutes,
        stale_seconds=stale_seconds,
    )
    db.rollback()
    lecture_impact = _read_lecture_impact_latest(db=db, zone_id=zone_id)
    db.rollback()
    events = get_calendar_events(
        zone_id=zone_id,
        from_=datetime.now(UTC),
        to=datetime.now(UTC) + timedelta(days=days),
        db=db,
    )
    lineage = _latest_model_lineage(db=db, zone_id=zone_id, product=PRODUCT_WEEKLY)
    payload = build_weekly_explainability(
        zone_id=zone_id,
        weekly_forecast=weekly.model_dump(mode="json"),
        lecture_impact=lecture_impact,
        lineage=lineage or weekly.lineage,
        calendar_events=[event.model_dump(mode="json") for event in events],
    )
    return WeeklyExplainResponse.model_validate(payload)


@app.get("/api/v1/explain/context", response_model=ExplainContextResponse)
async def get_explain_context(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720),
    audience: Literal["ops", "executive", "enduser", "professor"] = Query(default="ops"),
    language: Literal["de", "en"] = Query(default="de"),
    query: str = Query(default=""),
    db: Session = Depends(get_db),
) -> ExplainContextResponse:
    audience_resolved = _normalize_explain_audience(audience)
    context = await _build_explainability_context(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        audience=audience_resolved,
        language=language,
        query=query,
    )
    return ExplainContextResponse(context_version="v2", context=context)


@app.get("/api/v1/models/lineage/latest")
def get_model_lineage_latest(
    zone_id: str,
    product: Literal["short_term", "weekly_slot"] = Query(default=PRODUCT_SHORT_TERM),
    horizon: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ = _get_zone_or_404(db, zone_id)
    payload = _latest_model_lineage(db=db, zone_id=zone_id, product=product, horizon=horizon)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no model lineage found for zone={zone_id} product={product}")
    return payload


@app.post("/api/v1/explain/prompt/preview")
async def explain_prompt_preview(
    request: ExplainPromptPreviewRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not EXPLAINABILITY_DEBUG_PROMPT_PREVIEW:
        raise HTTPException(status_code=403, detail="prompt preview disabled")

    audience_resolved = _normalize_explain_audience(request.audience)
    context = await _build_explainability_context(
        db=db,
        zone_id=request.zone_id,
        horizon=request.horizon,
        audience=audience_resolved,
        language=request.language,
        query=request.query,
    )
    bundle = PROMPT_REGISTRY.build_prompt(
        context=context,
        audience=audience_resolved,
        language=request.language,
        query=request.query,
    )
    context_hash = hashlib.sha256(
        json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "template_set_id": bundle.template_set_id,
        "prompt_version": bundle.prompt_version,
        "required_context_fields": bundle.required_context_fields,
        "context_hash": context_hash,
        "prompt": bundle.prompt,
        "context": context,
    }


@app.post("/api/v1/explain/narrative")
async def explain_narrative(
    request: ExplainNarrativeRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    audience_resolved = _normalize_explain_audience(request.audience)
    context = await _build_explainability_context(
        db=db,
        zone_id=request.zone_id,
        horizon=request.horizon,
        audience=audience_resolved,
        language=request.language,
        query=request.query,
    )
    try:
        result = await NARRATIVE_SERVICE.generate(
            context=context,
            audience=audience_resolved,
            language=request.language,
            query=request.query,
            response_mode=request.response_mode or EXPLAINABILITY_RESPONSE_MODE,
            ollama_model_override=request.ollama_model,
            require_ollama=request.require_ollama,
        )
    except LLMQualityGateError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "llm_quality_gate_failed",
                "message": "LLM quality gate failed after retry.",
                "min_coverage": exc.min_coverage,
                "actual_coverage": exc.actual_coverage,
                "query_intent": exc.query_intent,
            },
        ) from exc
    except LLMNarrativeUnavailableError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "llm_unavailable",
                "message": "Required Ollama/Qwen narrative is unavailable; no template fallback was returned.",
                "reason": exc.reason,
                "query_intent": exc.query_intent,
            },
        ) from exc
    return result


@app.get("/api/v1/recommendations")
async def get_recommendations(
    zone_id: str,
    horizon: int = Query(default=DEFAULT_SHORT_FORECAST_HORIZON_MINUTES, ge=1, le=720),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    zone = _get_zone_or_404(db, zone_id)
    forecast_response = await _resolve_latest_forecast(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        stale_seconds=FORECAST_STALE_THRESHOLD_SECONDS,
    )
    db.rollback()
    explanation_payload = await _resolve_adjusted_explanation(
        db=db,
        zone_id=zone_id,
        horizon=horizon,
        forecast=forecast_response,
    )
    payload = {
        "zone_id": zone_id,
        "horizon": horizon,
        "capacity": zone.capacity,
        "forecast": forecast_response.model_dump(mode="json"),
        "explanation": explanation_payload,
    }
    return await _fetch_recommendations(payload)


@app.post("/api/v1/scenarios/simulate")
async def simulate_scenario(request: ScenarioInputRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    zone = _get_zone_or_404(db, request.zone_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            forecast_resp = await client.get(
                f"{FORECAST_SERVICE_URL}/v1/forecast",
                params={"zone_id": request.zone_id, "horizon": request.horizon},
            )
            explain_resp = await client.get(
                f"{XAI_SERVICE_URL}/v1/explain",
                params={"zone_id": request.zone_id, "horizon": request.horizon},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream service unreachable: {exc}") from exc

        if forecast_resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"forecast service error: {forecast_resp.text}")
        if explain_resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"xai service error: {explain_resp.text}")

        sim_payload = {
            "zone_id": request.zone_id,
            "horizon": request.horizon,
            "persist": request.persist,
            "changes": request.changes,
            "capacity": zone.capacity,
            "forecast": forecast_resp.json(),
            "explanation": explain_resp.json(),
        }
        sim_resp = await client.post(f"{RECOMMENDATIONS_SERVICE_URL}/v1/scenarios/simulate", json=sim_payload)

    if sim_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"recommendations service error: {sim_resp.text}")

    result = sim_resp.json()
    if request.persist:
        scenario = ScenarioRun(
            scenario_id=f"scn-{uuid.uuid4()}",
            zone_id=request.zone_id,
            horizon=request.horizon,
            persist=True,
            input_payload=request.model_dump(mode="json"),
            result_payload=result,
            evidence=result.get("evidence", {}),
        )
        db.add(scenario)
        db.commit()

    return result
