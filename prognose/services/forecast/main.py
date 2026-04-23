"""FastAPI forecast service — inference, training, and model-management endpoints.

This is the main entry point for the Sitcheck occupancy-forecast microservice.
It exposes a REST API consumed by the api-gateway and the dashboard, and handles:

  - Real-time short-term forecasting via LGBM:
      * lgbm    — LightGBM quantile regression (primary and only runtime backend)
      * baseline — explicit fallback only when LGBM cannot produce a response
  - Weekly slot forecasting (deterministic, no ML model required)
  - Model training, scientific evaluation, and promotion endpoints (maintenance mode only)
  - Data lineage tracking: every forecast and model run is recorded with full provenance

Backend policy:
  Runtime inference is forced to LGBM. If a stale environment still sets
  FORECAST_MODEL_BACKEND=tf_mlp or baseline, that value is ignored. If LGBM fails
  (model not found, not promoted, inference error), the service explicitly falls
  back to baseline with LGBM_FALLBACK evidence so XAI can mark the result degraded.

Capacity fix (April 2026):
  All inference paths now clip predicted occupancy to [0, capacity] where `capacity`
  is read from the zones table.  Without clipping, the model could predict values
  above the physical room limit, which is meaningless for occupancy planning.

Key symbols:
    ForecastResponse: Pydantic schema returned by GET /v1/forecast
    WeeklyForecastResponse: Schema returned by GET /v1/forecast/weekly
    get_forecast: Main inference endpoint handler
    _forecast_lgbm / _forecast_baseline: Runtime inference logic
    _get_zone_capacity: DB → env → fallback capacity resolution chain
    train_model / train_batch / train_evaluate / train_promote: Training endpoints
"""
from __future__ import annotations

import os
import sys
import uuid
import math
import json
import threading
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, and_, create_engine, event, or_, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

CWD = Path(__file__).resolve().parent
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

from features import FEATURE_SET_VERSION, build_feature_frame, build_inference_vector
from model_store import bundle_paths, gbdt_bundle_paths, gbdt_model_status, load_bundle, load_gbdt_bundle
from scientific_eval import (
    DEFAULT_HARD_QUALITY_FLAGS,
    EvaluationConfig,
    evaluate_training_run,
    load_evaluation_report,
    load_latest_evaluation_report,
    save_evaluation_report,
)
from weekly import WEEKLY_FEATURE_SET_VERSION, WEEKLY_PRODUCT, build_weekly_forecast


# GPU is disabled by default because we run on a CPU-only inference server.
# Set TF_ENABLE_GPU=true only if a GPU device is confirmed available.
if os.getenv("TF_ENABLE_GPU", "false").lower() != "true":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


# ---------------------------------------------------------------------------
# Runtime configuration (environment variables with safe defaults)
# ---------------------------------------------------------------------------

# Default zone used for backend health checks when the endpoint has no zone query.
DEFAULT_ZONE_ID = os.getenv("DEFAULT_ZONE_ID", "default-zone")

# Database connection string.  Defaults to a local SQLite file for development.
# In production: postgresql://user:pass@host:5432/sitcheck  (TimescaleDB)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sitcheck.db")

# Active ML backend: LGBM only. If a stale environment still sets tf_mlp or
# baseline, force LGBM so old TF-MLP bundles cannot become active again.
_CONFIGURED_MODEL_BACKEND = os.getenv("FORECAST_MODEL_BACKEND", "lgbm").strip().lower()
MODEL_BACKEND = "lgbm"
MODEL_BACKEND_CONFIG_WARNING = (
    None
    if _CONFIGURED_MODEL_BACKEND == "lgbm"
    else f"ignored unsupported FORECAST_MODEL_BACKEND={_CONFIGURED_MODEL_BACKEND}; forced lgbm"
)
TF_MLP_REMOVED_DETAIL = (
    "TF-MLP has been removed from the active forecast service; only LGBM is supported."
)

# Training mode guard.  'locked' (default) blocks all /v1/train* endpoints.
# Only 'maintenance' mode allows training to prevent accidental production retraining.
TRAINING_MODE = os.getenv("FORECAST_TRAINING_MODE", "locked").strip().lower()
if TRAINING_MODE not in {"locked", "maintenance"}:
    TRAINING_MODE = "locked"

# Maximum supported horizon for the /v1/forecast endpoint (30 days = 43200 min).
MAX_FORECAST_HORIZON_MINUTES = int(os.getenv("MAX_FORECAST_HORIZON_MINUTES", "43200"))

# For horizons beyond 24 h, forecast points are spaced LONG_HORIZON_STEP_MINUTES
# apart to keep response sizes manageable (avoid thousands of 1-min points).
LONG_HORIZON_STEP_MINUTES = int(os.getenv("LONG_HORIZON_STEP_MINUTES", "60"))

# Connection pool settings for PostgreSQL/TimescaleDB (ignored for SQLite).
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "20"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "40"))
DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))

# Path where trained model bundles (.keras, .joblib, metadata.json) are stored.
TF_MODEL_DIR = os.getenv("TF_MODEL_DIR", "./models")

# Minimum number of training rows for the TF-MLP pipeline (neural networks need
# more samples than tree models to generalise).
TF_MIN_TRAIN_POINTS = int(os.getenv("TF_MIN_TRAIN_POINTS", "2000"))

# Default forecast horizon when no 'horizon' query param is provided.
TF_DEFAULT_HORIZON = int(os.getenv("TF_DEFAULT_HORIZON", "210"))

# Minimum interval between automatic retrains in the trainer daemon (minutes).
TF_RETRAIN_INTERVAL_MINUTES = int(os.getenv("TF_RETRAIN_INTERVAL_MINUTES", "360"))

# Whether to include event_active / event_impact_sum in TF feature builds.
TF_USE_CALENDAR_FEATURES = os.getenv("TF_USE_CALENDAR_FEATURES", "true").lower() == "true"

# Rows with quality_score below this are excluded from TF training data.
TF_MIN_QUALITY_SCORE = float(os.getenv("TF_MIN_QUALITY_SCORE", "0.0"))

# TF training hyperparameters (EarlyStopping usually terminates well before max epochs).
TF_TRAIN_EPOCHS = int(os.getenv("TF_TRAIN_EPOCHS", "120"))
TF_BATCH_SIZE = int(os.getenv("TF_BATCH_SIZE", "64"))

# How many hours of history to pull for TF training (default: 30 days).
TF_TRAIN_HISTORY_HOURS = int(os.getenv("TF_TRAIN_HISTORY_HOURS", str(24 * 30)))

# How many hours of recent history to use for building live inference features.
TF_INFERENCE_HISTORY_HOURS = int(os.getenv("TF_INFERENCE_HISTORY_HOURS", "72"))

# Minimum fractional improvement over baseline for automatic TF promotion (unused
# in current flow — promotion now requires explicit scientific evaluation gate).
TF_PROMOTION_MIN_IMPROVEMENT = float(os.getenv("TF_PROMOTION_MIN_IMPROVEMENT", "0.0"))

# Context quality thresholds used by _context_status() to flag data quality issues.
FORECAST_CONTEXT_MIN_POINTS = int(os.getenv("FORECAST_CONTEXT_MIN_POINTS", "120"))       # < this → CONTEXT_SPARSE
FORECAST_CONTEXT_STALE_SECONDS = int(os.getenv("FORECAST_CONTEXT_STALE_SECONDS", "900")) # > this → CONTEXT_STALE
FORECAST_CONTEXT_MAX_HISTORY_HOURS = int(os.getenv("FORECAST_CONTEXT_MAX_HISTORY_HOURS", str(24 * 30)))

# ---------------------------------------------------------------------------
# Scientific evaluation (rolling-origin CV) configuration.
# These values default to the same ones used by scientific_eval.py but can be
# overridden per-request via the TrainEvaluateRequest body.
# ---------------------------------------------------------------------------
SCIENTIFIC_EVAL_HISTORY_HOURS = int(os.getenv("SCIENTIFIC_EVAL_HISTORY_HOURS", str(24 * 120)))
SCIENTIFIC_EVAL_FOLDS = int(os.getenv("SCIENTIFIC_EVAL_FOLDS", "6"))
SCIENTIFIC_EVAL_TRAIN_DAYS = int(os.getenv("SCIENTIFIC_EVAL_TRAIN_DAYS", "30"))
SCIENTIFIC_EVAL_VAL_DAYS = int(os.getenv("SCIENTIFIC_EVAL_VAL_DAYS", "7"))
SCIENTIFIC_EVAL_TEST_DAYS = int(os.getenv("SCIENTIFIC_EVAL_TEST_DAYS", "7"))
SCIENTIFIC_EVAL_GAP_MINUTES = int(os.getenv("SCIENTIFIC_EVAL_GAP_MINUTES", "60"))        # gap prevents train→val leakage
SCIENTIFIC_EVAL_ORIGIN_STRIDE_MINUTES = int(os.getenv("SCIENTIFIC_EVAL_ORIGIN_STRIDE_MINUTES", "60"))
SCIENTIFIC_EVAL_MAX_ORIGINS_PER_SPLIT = int(os.getenv("SCIENTIFIC_EVAL_MAX_ORIGINS_PER_SPLIT", "120"))
SCIENTIFIC_EVAL_MIN_QUALITY_SCORE = float(os.getenv("SCIENTIFIC_EVAL_MIN_QUALITY_SCORE", "0.6"))
SCIENTIFIC_EVAL_EXCLUDE_FLAGS = [
    flag.strip()
    for flag in os.getenv("SCIENTIFIC_EVAL_EXCLUDE_FLAGS", ",".join(DEFAULT_HARD_QUALITY_FLAGS)).split(",")
    if flag.strip()
]
# Promotion gate: model must improve MAE by at least 8% vs Persistence H60.
SCIENTIFIC_EVAL_IMPROVEMENT_THRESHOLD = float(os.getenv("SCIENTIFIC_EVAL_IMPROVEMENT_THRESHOLD", "0.08"))
# Model must not degrade long-horizon MAE by more than 2% vs baseline.
SCIENTIFIC_EVAL_MAX_LONG_DEGRADATION = float(os.getenv("SCIENTIFIC_EVAL_MAX_LONG_DEGRADATION", "0.02"))
# Coverage of prediction interval [q03, q97] must fall within [85%, 95%].
SCIENTIFIC_EVAL_COVERAGE_LOW = float(os.getenv("SCIENTIFIC_EVAL_COVERAGE_LOW", "0.85"))
SCIENTIFIC_EVAL_COVERAGE_HIGH = float(os.getenv("SCIENTIFIC_EVAL_COVERAGE_HIGH", "0.95"))
SCIENTIFIC_EVAL_PRIMARY_HORIZON = int(os.getenv("SCIENTIFIC_EVAL_PRIMARY_HORIZON", "210"))
# Reduced epoch count for the TF-MLP used inside scientific evaluation folds
# (speed vs accuracy tradeoff; full training uses TF_TRAIN_EPOCHS).
SCIENTIFIC_EVAL_TF_EPOCHS = int(os.getenv("SCIENTIFIC_EVAL_TF_EPOCHS", "16"))
SCIENTIFIC_EVAL_TF_MAX_HORIZON_MINUTES = int(os.getenv("SCIENTIFIC_EVAL_TF_MAX_HORIZON_MINUTES", "720"))
SCIENTIFIC_EVAL_SEGMENT_MIN_SAMPLES = int(os.getenv("SCIENTIFIC_EVAL_SEGMENT_MIN_SAMPLES", "30"))

# Weekly forecast configuration.
WEEKLY_FORECAST_DAYS_DEFAULT = int(os.getenv("WEEKLY_FORECAST_DAYS_DEFAULT", "7"))
WEEKLY_FORECAST_SLOT_MINUTES = int(os.getenv("WEEKLY_FORECAST_SLOT_MINUTES", "60"))

# Maximum number of training data references to attach to lineage records.
TRAINING_DATA_REFERENCE_LIMIT = int(os.getenv("TRAINING_DATA_REFERENCE_LIMIT", "8"))

# Internal product label for short-term (real-time) forecast records.
PRODUCT_SHORT_TERM = "short_term"

# ---------------------------------------------------------------------------
# Database engine setup
# ---------------------------------------------------------------------------

if DATABASE_URL.startswith("sqlite:"):
    # SQLite path (local development and testing).
    # NullPool is used because SQLite does not support concurrent multi-threaded
    # connections from a pool; each request creates and immediately closes its
    # own connection to avoid "database is locked" errors.
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 60},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        """Configure SQLite performance pragmas on every new connection.

        These settings dramatically improve concurrent read/write throughput:
        - journal_mode=WAL: Write-Ahead Logging allows readers and writers to
          operate simultaneously instead of serialising all access.
        - synchronous=NORMAL: Reduces fsync calls; safe for non-critical workloads.
        - busy_timeout=60000: Wait up to 60 s before raising 'database is locked'.
        - temp_store=MEMORY: Keep temporary tables in RAM instead of on disk.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA temp_store=MEMORY")
        finally:
            cursor.close()
else:
    # PostgreSQL / TimescaleDB path (production).
    # Uses a connection pool for efficient reuse across requests.
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,       # verify connections before use to handle server-side timeouts
        pool_size=DATABASE_POOL_SIZE,
        max_overflow=DATABASE_MAX_OVERFLOW,
        pool_timeout=DATABASE_POOL_TIMEOUT,
    )

# Session factory — each request opens and closes its own session.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# ---------------------------------------------------------------------------
# ORM base class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Database table mappings (SQLAlchemy ORM models)
# These mirror the tables created by infra/db/migrations/*.sql
# ---------------------------------------------------------------------------

class Count(Base):
    """Sensor occupancy reading for a zone at a specific timestamp.

    Populated by the bildauswertung (image analysis) service and imported via
    the counts ingest endpoint.  Each row represents one occupancy snapshot.
    """
    __tablename__ = "counts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    occupancy: Mapped[int] = mapped_column(Integer, nullable=False)
    utilization: Mapped[float] = mapped_column(Float, nullable=False)  # occupancy / capacity
    source: Mapped[str] = mapped_column(String, nullable=False)         # e.g. "camera", "manual"
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)   # 0=poor, 1=perfect
    quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)  # e.g. ["LOW_CONFIDENCE"]
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)  # provenance chain
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class CalendarEvent(Base):
    """DHBW calendar event (e.g. exam, open day) that may affect occupancy.

    Populated by the calendar-ingest service from ICS feeds.  Used as a
    context signal in feature engineering: events boost expected occupancy.
    """
    __tablename__ = "calendar_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)  # None = affects all zones
    title: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_impact: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0..1 occupancy boost factor
    source: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class LectureActivity(Base):
    """Lecture schedule snapshot for a zone at a timestamp.

    Populated by the lecture-ingest service from the DHBW Vorlesungsplan API.
    Provides near-real-time lecture density signals for feature engineering.
    """
    __tablename__ = "lecture_activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    active_lectures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)   # lectures happening now
    active_courses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)    # distinct courses active
    starts_next_60m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)   # lectures starting soon
    ends_next_60m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)     # lectures ending soon
    source: Mapped[str] = mapped_column(String, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class ReferenceObject(Base):
    """Pointer to an external data source used to train or evaluate a model.

    Supports data lineage: when a model is trained, every input data source
    (Excel file, DB window, external API) is recorded here.  This makes it
    possible to reproduce any model run by knowing exactly what data was used.
    """
    __tablename__ = "reference_objects"

    reference_id: Mapped[str] = mapped_column(String, primary_key=True)   # deterministic hash of source identity
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reference_type: Mapped[str] = mapped_column(String, nullable=False)    # "training_data" | "external_reference"
    source_type: Mapped[str] = mapped_column(String, nullable=False)       # "excel", "csv", "counts_db_window", ...
    label: Mapped[str] = mapped_column(String, nullable=False)             # human-readable name
    uri_or_path: Mapped[str | None] = mapped_column(String, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)    # SHA-256 of file content for verification
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class ModelRun(Base):
    """Audit record for a model training or promotion event.

    Every call to /v1/train, /v1/train/batch, or /v1/train/promote creates
    a row here, providing a full audit trail of what was trained, when, with
    what data, and whether it passed the promotion gate.
    """
    __tablename__ = "model_runs"

    model_run_id: Mapped[str] = mapped_column(String, primary_key=True)
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    product: Mapped[str] = mapped_column(String, nullable=False, default=PRODUCT_SHORT_TERM)
    horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_backend: Mapped[str] = mapped_column(String, nullable=False)         # "tf_mlp" | "lgbm"
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="trained")  # "trained" | "promoted"
    scientific_status: Mapped[str] = mapped_column(String, nullable=False, default="training_only")  # "training_only" | "scientifically_validated"
    include_lecture_impact: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    feature_set_version: Mapped[str] = mapped_column(String, nullable=False, default=FEATURE_SET_VERSION)
    history_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    history_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)    # total input rows before filtering
    train_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evaluation_run_id: Mapped[str | None] = mapped_column(String, nullable=True)  # links to scientific eval report
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promotion_source: Mapped[str | None] = mapped_column(String, nullable=True)   # e.g. "scientific_evaluation_gate"
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    lineage_json: Mapped[dict[str, Any]] = mapped_column("lineage", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class ModelRunReference(Base):
    """Junction table linking a model run to its input data sources.

    Allows the lineage API to answer 'what data was this model trained on?'
    for any given model_run_id.
    """
    __tablename__ = "model_run_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(String, nullable=False)
    reference_id: Mapped[str] = mapped_column(String, nullable=False)
    relation_type: Mapped[str] = mapped_column(String, nullable=False, default="training_data")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Pydantic response / request schemas
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    """A single forecast step: timestamp, point estimate, and prediction interval.

    pi_low / pi_high form an ~80% prediction interval calibrated from
    validation-set residuals (TF) or quantile models at α=0.03/0.97 (GBDT).
    """
    timestamp: datetime
    yhat: float      # point forecast (median / q50)
    pi_low: float    # lower bound (q03 or yhat + q10 residual)
    pi_high: float   # upper bound (q97 or yhat + q90 residual)


class ForecastResponse(BaseModel):
    """Full response for GET /v1/forecast.

    Contains the model's forecast points plus evidence metadata (provenance,
    quality flags, lineage IDs) needed for explainability downstream.
    """
    zone_id: str
    horizon: int
    generated_at: datetime
    summary: str
    model_version: str
    points: list[ForecastPoint]
    evidence: dict[str, Any]    # evidence payload for ECP v2 / XAI service
    lineage: dict[str, Any] | None = None


class WeeklyForecastPoint(BaseModel):
    """One slot in the weekly forecast: day/slot identity + occupancy estimate."""
    timestamp: datetime
    yhat: float
    pi_low: float
    pi_high: float
    day_of_week: int    # 0=Monday, 6=Sunday (ISO weekday)
    slot_of_day: int    # slot index within the day (0-based)
    event_active: float | None = None
    event_impact_sum: float | None = None
    lecture_net_pull: float | None = None
    quality_score: float | None = None


class WeeklyForecastDaySummary(BaseModel):
    """Aggregated summary for one calendar day in the weekly forecast."""
    date: str
    peak_slot: str | None = None   # ISO timestamp of peak occupancy slot
    peak_yhat: float
    avg_yhat: float
    risk_level: str       # "low" | "medium" | "high" based on utilisation thresholds
    data_quality: str     # "good" | "degraded" | "poor"


class WeeklyForecastResponse(BaseModel):
    """Full response for GET /v1/forecast/weekly."""
    zone_id: str
    product: str = WEEKLY_PRODUCT
    days: int
    slot_minutes: int
    generated_at: datetime
    summary: str
    model_version: str
    points: list[WeeklyForecastPoint]
    daily_summaries: list[WeeklyForecastDaySummary]
    evidence: dict[str, Any]
    lineage: dict[str, Any]


class TrainRequest(BaseModel):
    """Request body for POST /v1/train (single horizon TF-MLP training)."""
    zone_id: str
    horizon: int = Field(default=TF_DEFAULT_HORIZON, ge=1, le=720)
    history_hours: int = Field(default=TF_TRAIN_HISTORY_HOURS, ge=24, le=24 * 365)
    full_retrain: bool = False
    include_lecture_impact: bool = True


class BatchTrainRequest(BaseModel):
    """Request body for POST /v1/train/batch (multiple horizons in one call)."""
    zone_id: str
    horizons: list[int] = Field(default_factory=lambda: [TF_DEFAULT_HORIZON, 1440])
    history_hours: int = Field(default=TF_TRAIN_HISTORY_HOURS, ge=24, le=24 * 365)
    full_retrain: bool = False
    include_lecture_impact: bool = True


class TrainEvaluateRequest(BaseModel):
    zone_id: str
    horizons: list[int] = Field(default_factory=lambda: [TF_DEFAULT_HORIZON, 1440], min_length=1)
    history_hours: int = Field(default=SCIENTIFIC_EVAL_HISTORY_HOURS, ge=24, le=24 * 365 * 2)
    folds: int = Field(default=SCIENTIFIC_EVAL_FOLDS, ge=2, le=12)
    train_days: int = Field(default=SCIENTIFIC_EVAL_TRAIN_DAYS, ge=3, le=365)
    val_days: int = Field(default=SCIENTIFIC_EVAL_VAL_DAYS, ge=1, le=90)
    test_days: int = Field(default=SCIENTIFIC_EVAL_TEST_DAYS, ge=1, le=90)
    gap_minutes: int = Field(default=SCIENTIFIC_EVAL_GAP_MINUTES, ge=0, le=1440)
    origin_stride_minutes: int = Field(default=SCIENTIFIC_EVAL_ORIGIN_STRIDE_MINUTES, ge=1, le=720)
    max_origins_per_split: int = Field(default=SCIENTIFIC_EVAL_MAX_ORIGINS_PER_SPLIT, ge=10, le=1000)
    min_quality_score: float = Field(default=SCIENTIFIC_EVAL_MIN_QUALITY_SCORE, ge=0.0, le=1.0)
    exclude_quality_flags: list[str] = Field(default_factory=lambda: list(SCIENTIFIC_EVAL_EXCLUDE_FLAGS))
    random_seed: int = Field(default=42, ge=1, le=1_000_000)
    enable_tf: bool = True
    enable_sarimax: bool = True
    save_report: bool = True
    primary_horizon: int = Field(default=SCIENTIFIC_EVAL_PRIMARY_HORIZON, ge=1, le=MAX_FORECAST_HORIZON_MINUTES)
    improvement_threshold: float = Field(default=SCIENTIFIC_EVAL_IMPROVEMENT_THRESHOLD, ge=0.0, le=1.0)
    max_long_degradation: float = Field(default=SCIENTIFIC_EVAL_MAX_LONG_DEGRADATION, ge=0.0, le=1.0)
    coverage_low: float = Field(default=SCIENTIFIC_EVAL_COVERAGE_LOW, ge=0.0, le=1.0)
    coverage_high: float = Field(default=SCIENTIFIC_EVAL_COVERAGE_HIGH, ge=0.0, le=1.0)
    include_lecture_impact: bool = True
    segment_min_samples: int = Field(default=SCIENTIFIC_EVAL_SEGMENT_MIN_SAMPLES, ge=1, le=5000)


class TrainPromoteRequest(BaseModel):
    zone_id: str
    run_id: str = Field(min_length=8, max_length=200)
    horizons: list[int] | None = None
    history_hours: int = Field(default=TF_TRAIN_HISTORY_HOURS, ge=24, le=24 * 365)
    full_retrain: bool = False


app = FastAPI(title="sitcheck-forecast", version="0.2.0")

# Thread-safe in-memory cache for loaded TF-MLP bundles.
# Avoids reloading the Keras model from disk on every request — which takes
# several seconds.  The cache is invalidated by comparing file modification
# timestamps (_bundle_signature) so stale bundles are never served after retraining.
_BUNDLE_CACHE_LOCK = threading.Lock()
_BUNDLE_CACHE: dict[tuple[str, str, int], dict[str, Any]] = {}
_GBDT_BUNDLE_CACHE_LOCK = threading.Lock()
_GBDT_BUNDLE_CACHE: dict[tuple[str, str, int], dict[str, Any]] = {}


@app.on_event("startup")
def startup() -> None:
    """Initialise the database schema and model directory on service start."""
    Base.metadata.create_all(bind=engine)
    os.makedirs(TF_MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalize_dt(value: datetime) -> datetime:
    """Ensure a datetime is UTC-aware.

    Naive datetimes (no tzinfo) are assumed to be UTC and are tagged.
    Aware datetimes in other timezones are converted to UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_json_dict(value: Any) -> dict[str, Any]:
    """Return value if it is a dict, otherwise return an empty dict.

    Defensive wrapper used when reading JSON columns from the database
    that may be None or non-dict (e.g. after a schema migration).
    """
    if isinstance(value, dict):
        return value
    return {}


def _exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _compact_bundle_status(status: dict[str, Any]) -> dict[str, Any]:
    metadata = _safe_json_dict(status.get("metadata"))
    return {
        "exists": bool(status.get("exists", False)),
        "loadable": status.get("loadable"),
        "load_error": status.get("load_error"),
        "missing": status.get("missing", []),
        "backend": status.get("backend"),
        "zone_id": status.get("zone_id"),
        "horizon": status.get("horizon"),
        "model_version": metadata.get("model_version"),
        "promoted": metadata.get("promoted"),
        "scientific_status": metadata.get("scientific_status"),
        "feature_set_version": metadata.get("feature_set_version"),
        "root": _safe_json_dict(status.get("paths")).get("root"),
    }


def _coerce_optional_dt(value: Any) -> datetime | None:
    """Parse an optional datetime from various formats (ISO string, datetime, None)."""
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return _normalize_dt(value)
    try:
        return _normalize_dt(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except Exception:
        return None


def _checksum_text(value: str) -> str:
    """Compute a SHA-256 hex digest of a UTF-8 string (for deduplication IDs)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _latest_training_references(zone_id: str, limit: int = TRAINING_DATA_REFERENCE_LIMIT) -> list[dict[str, Any]]:
    """Return the most recent training data reference records for a zone.

    Used to populate lineage when no formal model_run exists for a zone yet
    (e.g. the first time the weekly forecast is called after data import).
    """
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(ReferenceObject)
                .where(or_(ReferenceObject.zone_id == zone_id, ReferenceObject.zone_id.is_(None)))
                .where(ReferenceObject.reference_type.in_(["training_data", "external_reference"]))
                .order_by(ReferenceObject.imported_at.desc(), ReferenceObject.created_at.desc())
                .limit(max(1, limit))
            )
            .scalars()
            .all()
        )
    return [_reference_payload_to_public(row) for row in rows]


def _normalize_reference_id(reference_type: str, source_type: str, label: str, uri_or_path: str | None, checksum: str | None) -> str:
    """Create a deterministic stable ID for a data source reference.

    The ID is derived from the content identity of the reference (type, label,
    URI, checksum) so that the same source always receives the same ID,
    enabling upsert-style persistence without duplicates.
    """
    seed = "|".join(
        [
            str(reference_type or ""),
            str(source_type or ""),
            str(label or ""),
            str(uri_or_path or ""),
            str(checksum or ""),
        ]
    )
    return f"ref-{_checksum_text(seed)[:20]}"


def _dedupe_references(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate reference entries based on source identity key."""
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = "|".join(
            [
                str(item.get("reference_type") or ""),
                str(item.get("source_type") or ""),
                str(item.get("label") or ""),
                str(item.get("uri_or_path") or ""),
                str(item.get("checksum") or ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _collect_training_references(
    *,
    zone_id: str,
    history_df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    events_df: pd.DataFrame,
    history_hours: int,
) -> list[dict[str, Any]]:
    """Collect all data sources used in a training run into a reference list.

    Scans the occupancy history, lecture data, and calendar events to extract
    source metadata and produce provenance records.  Sources are deduplicated
    so that the same file/DB window doesn't appear twice.

    Args:
        zone_id: Zone identifier for the training run.
        history_df: Raw occupancy history (may contain 'evidence' column with source provenance).
        lecture_df: Lecture activity data (may contain 'metadata' with external references).
        events_df: Calendar event data.
        history_hours: How many hours of history were queried (recorded in reference metadata).

    Returns:
        List of deduplicated reference dicts, each suitable for _persist_references_for_model_run.
    """
    references: list[dict[str, Any]] = []

    time_from = None
    time_to = None
    row_count = int(len(history_df))
    if not history_df.empty and "timestamp" in history_df.columns:
        ts = pd.to_datetime(history_df["timestamp"], utc=True)
        if not ts.empty:
            time_from = _normalize_dt(ts.min().to_pydatetime())
            time_to = _normalize_dt(ts.max().to_pydatetime())

    references.append(
        {
            "zone_id": zone_id,
            "reference_type": "training_data",
            "source_type": "counts_db_window",
            "label": f"Counts DB Window {zone_id}",
            "uri_or_path": f"db://counts/{zone_id}",
            "checksum": None,
            "imported_at": None,
            "time_from": time_from,
            "time_to": time_to,
            "row_count": row_count,
            "metadata": {
                "history_hours": int(history_hours),
                "sources": sorted(
                    {
                        str(source).strip()
                        for source in history_df.get("source", pd.Series(dtype=str)).tolist()
                        if str(source).strip()
                    }
                ),
            },
            "relation_type": "training_data",
        }
    )

    if not history_df.empty and "evidence" in history_df.columns:
        evidence_refs: list[dict[str, Any]] = []
        for evidence in history_df["evidence"]:
            payload = _safe_json_dict(evidence)
            imported_at = _coerce_optional_dt(payload.get("generated_at"))
            sources = payload.get("sources", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                metadata = _safe_json_dict(source.get("metadata"))
                uri = source.get("uri")
                checksum = metadata.get("checksum")
                label = str(source.get("id") or metadata.get("label") or source.get("type") or "source").strip()
                source_type = str(source.get("type") or "unknown").strip() or "unknown"
                ref_type = "training_data" if source_type in {"csv", "excel", "xlsx"} else "data_source"
                evidence_refs.append(
                    {
                        "reference_id": metadata.get("reference_id"),
                        "zone_id": zone_id,
                        "reference_type": ref_type,
                        "source_type": source_type,
                        "label": label,
                        "uri_or_path": str(uri).strip() if uri else metadata.get("uri_or_path"),
                        "checksum": str(checksum).strip() if checksum else None,
                        "imported_at": imported_at,
                        "time_from": _coerce_optional_dt(metadata.get("time_from")),
                        "time_to": _coerce_optional_dt(metadata.get("time_to")),
                        "row_count": int(metadata.get("row_count")) if metadata.get("row_count") is not None else None,
                        "metadata": metadata,
                        "relation_type": "training_data",
                    }
                )
        references.extend(evidence_refs)

    lecture_metadata: list[dict[str, Any]] = []
    if not lecture_df.empty and "metadata" in lecture_df.columns:
        for value in lecture_df["metadata"]:
            metadata = _safe_json_dict(value)
            external_refs = metadata.get("external_references", [])
            if isinstance(external_refs, list):
                for item in external_refs:
                    if not isinstance(item, dict):
                        continue
                    lecture_metadata.append(
                        {
                            "reference_id": item.get("reference_id"),
                            "zone_id": zone_id,
                            "reference_type": "external_reference",
                            "source_type": str(item.get("source_type") or "external").strip() or "external",
                            "label": str(item.get("label") or item.get("uri") or item.get("url") or "external reference").strip(),
                            "uri_or_path": str(item.get("uri") or item.get("url") or "").strip() or None,
                            "checksum": None,
                            "imported_at": None,
                            "time_from": None,
                            "time_to": None,
                            "row_count": None,
                            "metadata": _safe_json_dict(item.get("metadata")),
                            "relation_type": "external_reference",
                        }
                    )
    references.extend(lecture_metadata)

    if not events_df.empty:
        references.append(
            {
                "zone_id": zone_id,
                "reference_type": "training_data",
                "source_type": "calendar_events",
                "label": f"Calendar Events Window {zone_id}",
                "uri_or_path": f"db://calendar_events/{zone_id}",
                "checksum": None,
                "imported_at": None,
                "time_from": _coerce_optional_dt(events_df["starts_at"].min()) if "starts_at" in events_df.columns else None,
                "time_to": _coerce_optional_dt(events_df["ends_at"].max()) if "ends_at" in events_df.columns else None,
                "row_count": int(len(events_df)),
                "metadata": {},
                "relation_type": "calendar_context",
            }
        )

    return _dedupe_references(references)


def _persist_references_for_model_run(model_run_id: str, references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write reference objects and their links to the database.

    Upserts each reference into reference_objects (same source → same row),
    then rebuilds the model_run_references junction table for this run_id.
    Returns the persisted reference records as public dicts.
    """
    if not references:
        return []
    stored: list[dict[str, Any]] = []
    with SessionLocal() as db:
        db.query(ModelRunReference).filter(ModelRunReference.model_run_id == model_run_id).delete(synchronize_session=False)
        for item in references:
            reference_type = str(item.get("reference_type") or "training_data")
            source_type = str(item.get("source_type") or "unknown")
            label = str(item.get("label") or "reference")
            uri_or_path = str(item.get("uri_or_path") or "").strip() or None
            checksum = str(item.get("checksum") or "").strip() or None
            reference_id = str(item.get("reference_id") or "").strip() or _normalize_reference_id(
                reference_type,
                source_type,
                label,
                uri_or_path,
                checksum,
            )
            row = db.get(ReferenceObject, reference_id)
            if row is None:
                row = ReferenceObject(
                    reference_id=reference_id,
                    zone_id=item.get("zone_id"),
                    reference_type=reference_type,
                    source_type=source_type,
                    label=label,
                    uri_or_path=uri_or_path,
                    checksum=checksum,
                    imported_at=_coerce_optional_dt(item.get("imported_at")),
                    time_from=_coerce_optional_dt(item.get("time_from")),
                    time_to=_coerce_optional_dt(item.get("time_to")),
                    row_count=item.get("row_count"),
                    metadata_json=_safe_json_dict(item.get("metadata")),
                )
                db.add(row)
            else:
                row.zone_id = item.get("zone_id")
                row.reference_type = reference_type
                row.source_type = source_type
                row.label = label
                row.uri_or_path = uri_or_path
                row.checksum = checksum
                row.imported_at = _coerce_optional_dt(item.get("imported_at"))
                row.time_from = _coerce_optional_dt(item.get("time_from"))
                row.time_to = _coerce_optional_dt(item.get("time_to"))
                row.row_count = item.get("row_count")
                row.metadata_json = _safe_json_dict(item.get("metadata"))

            db.add(
                ModelRunReference(
                    model_run_id=model_run_id,
                    reference_id=reference_id,
                    relation_type=str(item.get("relation_type") or "training_data"),
                )
            )
            db.flush()
            stored.append(_reference_payload_to_public(row))
        db.commit()
    return stored


def _upsert_model_run(
    *,
    model_run_id: str,
    zone_id: str,
    product: str,
    horizon: int | None,
    model_backend: str,
    model_version: str,
    include_lecture_impact: bool,
    feature_set_version: str,
    raw_rows: int | None,
    metadata: dict[str, Any],
    lineage: dict[str, Any],
    status: str,
    scientific_status: str,
    evaluation_run_id: str | None = None,
    promoted: bool = False,
    promotion_source: str | None = None,
) -> None:
    """Create or update a ModelRun audit record in the database.

    Uses insert-or-update semantics: if the model_run_id already exists
    (e.g. during a promote call that follows a train call), all fields
    are overwritten with the latest values.
    """
    metadata_payload = _safe_json_dict(metadata)
    history_window = _safe_json_dict(metadata_payload.get("history_window"))
    reference_objects = lineage.get("reference_objects", []) if isinstance(lineage, dict) else []
    with SessionLocal() as db:
        row = db.get(ModelRun, model_run_id)
        now = datetime.now(UTC)
        if row is None:
            row = ModelRun(
                model_run_id=model_run_id,
                zone_id=zone_id,
                product=product,
                horizon=horizon,
                model_backend=model_backend,
                model_version=model_version,
                include_lecture_impact=include_lecture_impact,
                feature_set_version=feature_set_version,
                raw_rows=raw_rows,
                metadata_json=metadata_payload,
                lineage_json=lineage,
                status=status,
                scientific_status=scientific_status,
                evaluation_run_id=evaluation_run_id,
                promoted=promoted,
                promoted_at=now if promoted else None,
                promotion_source=promotion_source,
            )
            db.add(row)
        else:
            row.zone_id = zone_id
            row.product = product
            row.horizon = horizon
            row.model_backend = model_backend
            row.model_version = model_version
            row.include_lecture_impact = include_lecture_impact
            row.feature_set_version = feature_set_version
            row.raw_rows = raw_rows
            row.metadata_json = metadata_payload
            row.lineage_json = lineage
            row.status = status
            row.scientific_status = scientific_status
            row.evaluation_run_id = evaluation_run_id
            row.promoted = promoted
            row.promoted_at = now if promoted else None
            row.promotion_source = promotion_source

        row.train_rows = int(metadata_payload.get("train_rows")) if metadata_payload.get("train_rows") is not None else None
        row.val_rows = int(metadata_payload.get("val_rows")) if metadata_payload.get("val_rows") is not None else None
        row.test_rows = int(metadata_payload.get("test_rows")) if metadata_payload.get("test_rows") is not None else None
        row.history_from = _coerce_optional_dt(history_window.get("from"))
        row.history_to = _coerce_optional_dt(history_window.get("to"))
        db.query(ModelRunReference).filter(ModelRunReference.model_run_id == model_run_id).delete(synchronize_session=False)
        for item in reference_objects:
            if not isinstance(item, dict):
                continue
            reference_id = item.get("reference_id")
            if not reference_id:
                continue
            existing_reference = db.get(ReferenceObject, reference_id)
            if existing_reference is None:
                continue
            db.add(
                ModelRunReference(
                    model_run_id=model_run_id,
                    reference_id=str(reference_id),
                    relation_type=str(item.get("reference_type") or "training_data"),
                )
            )
        db.commit()


def _build_model_lineage_payload(
    *,
    zone_id: str,
    product: str,
    horizon: int | None,
    model_version: str,
    model_backend: str,
    feature_set_version: str,
    include_lecture_impact: bool,
    promoted: bool,
    scientific_status: str,
    model_run_id: str,
    metadata: dict[str, Any],
    reference_objects: list[dict[str, Any]],
    evaluation_run_id: str | None = None,
    promotion_source: str | None = None,
) -> dict[str, Any]:
    """Assemble the structured lineage payload stored in model_runs.lineage.

    This dict is attached to every ForecastResponse and is the machine-readable
    provenance record consumed by the XAI service and the ECP v2 context builder.
    """
    history_window = _safe_json_dict(metadata.get("history_window"))
    metrics = _safe_json_dict(metadata.get("metrics"))
    return {
        "model_run_id": model_run_id,
        "zone_id": zone_id,
        "product": product,
        "horizon": horizon,
        "model_version": model_version,
        "backend": model_backend,
        "feature_set_version": feature_set_version,
        "include_lecture_impact": include_lecture_impact,
        "promoted": promoted,
        "scientific_status": scientific_status,
        "evaluation_run_id": evaluation_run_id,
        "promotion_source": promotion_source,
        "history_window": history_window,
        "metrics": metrics,
        "reference_objects": reference_objects,
    }


def _latest_model_lineage(
    *,
    zone_id: str,
    product: str,
    horizon: int | None = None,
) -> dict[str, Any] | None:
    """Retrieve the most recently promoted (or trained) model run for a zone/product.

    Returns None if no model run record exists yet.  Used to attach lineage
    metadata to forecast responses without requiring a fresh DB join per call.
    """
    with SessionLocal() as db:
        query = select(ModelRun).where(ModelRun.zone_id == zone_id).where(ModelRun.product == product)
        if horizon is not None:
            query = query.where(or_(ModelRun.horizon == horizon, ModelRun.horizon.is_(None)))
        row = db.execute(query.order_by(ModelRun.promoted_at.desc(), ModelRun.created_at.desc())).scalars().first()
        if row is None:
            return None
        return {
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


def _get_zone_capacity(zone_id: str) -> float:
    """Return the physical seating capacity for a zone.

    Resolution order (highest priority first):
      1. zones.capacity column in the database (authoritative)
      2. DEFAULT_ZONE_CAPACITY environment variable
      3. Hard-coded fallback: 100

    Capacity is used to clip all model predictions to [0, capacity] so that
    the service never reports occupancy above the room's physical limit.
    This fix was applied in April 2026 after the GBDT model was observed
    occasionally predicting values above capacity during high-traffic periods.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT capacity FROM zones WHERE zone_id = :zone_id LIMIT 1"),
            {"zone_id": zone_id},
        ).first()
        try:
            return float(
                row._mapping["capacity"] if row is not None else os.getenv("DEFAULT_ZONE_CAPACITY", "100")
            )
        except Exception:
            return float(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))


def _load_history_df(zone_id: str, history_hours: int = 24) -> pd.DataFrame:
    """Load recent occupancy readings from the counts table.

    Returns a DataFrame with columns: timestamp, occupancy, utilization,
    quality_score, quality_flags, source, evidence.  Rows where occupancy
    is outside [0, capacity] are removed (corrupted sensor readings).

    Args:
        zone_id: Zone to query.
        history_hours: How many hours back from now to include.

    Returns:
        Empty DataFrame with correct schema if no data exists.
    """
    from_dt = datetime.now(UTC) - timedelta(hours=history_hours)
    with SessionLocal() as db:
        capacity_row = db.execute(
            text("SELECT capacity FROM zones WHERE zone_id = :zone_id LIMIT 1"),
            {"zone_id": zone_id},
        ).first()
        try:
            zone_capacity = float(
                capacity_row._mapping["capacity"] if capacity_row is not None else os.getenv("DEFAULT_ZONE_CAPACITY", "100")
            )
        except Exception:
            zone_capacity = float(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))

        rows = db.execute(
            select(
                Count.ts.label("timestamp"),
                Count.occupancy.label("occupancy"),
                Count.utilization.label("utilization"),
                Count.quality_score.label("quality_score"),
                Count.quality_flags.label("quality_flags"),
                Count.source.label("source"),
            )
            .where(Count.zone_id == zone_id)
            .where(Count.ts >= from_dt)
            .order_by(Count.ts.asc())
        ).all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "occupancy",
                "utilization",
                "quality_score",
                "quality_flags",
                "source",
                "evidence",
            ]
        )

    history_df = pd.DataFrame(
        {
            "timestamp": [_normalize_dt(row._mapping["timestamp"]) for row in rows],
            "occupancy": [float(row._mapping["occupancy"]) for row in rows],
            "utilization": [float(row._mapping["utilization"]) for row in rows],
            "quality_score": [float(row._mapping["quality_score"]) for row in rows],
            "quality_flags": [row._mapping["quality_flags"] or [] for row in rows],
            "source": [str(row._mapping["source"]) for row in rows],
            "evidence": [{} for _ in rows],
        }
    )
    if math.isfinite(zone_capacity) and zone_capacity > 0:
        history_df = history_df[
            history_df["occupancy"].between(0.0, zone_capacity, inclusive="both")
        ].reset_index(drop=True)
    return history_df


def _load_occupancy_at_or_before(zone_id: str, target_ts: datetime, lookback_hours: int = 24) -> float | None:
    """Return the nearest occupancy reading at or before target_ts."""
    target_ts = _normalize_dt(target_ts)
    from_dt = target_ts - timedelta(hours=lookback_hours)
    capacity = _get_zone_capacity(zone_id)
    with SessionLocal() as db:
        row = db.execute(
            select(Count.occupancy)
            .where(Count.zone_id == zone_id)
            .where(Count.ts <= target_ts)
            .where(Count.ts >= from_dt)
            .where(Count.occupancy >= 0.0)
            .where(Count.occupancy <= capacity)
            .order_by(Count.ts.desc())
            .limit(1)
        ).first()
    if row is None:
        return None
    return float(row._mapping["occupancy"])


def _load_events_df(zone_id: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Load calendar events overlapping the given time window for a zone.

    Includes zone-specific events AND global events (zone_id IS NULL),
    because some events (open days, campus-wide exams) affect all zones.
    """
    with SessionLocal() as db:
        rows = db.execute(
            select(CalendarEvent)
            .where(and_(CalendarEvent.ends_at >= from_dt, CalendarEvent.starts_at <= to_dt))
            .where(or_(CalendarEvent.zone_id == zone_id, CalendarEvent.zone_id.is_(None)))
            .order_by(CalendarEvent.starts_at.asc())
        ).scalars().all()

    if not rows:
        return pd.DataFrame(columns=["starts_at", "ends_at", "expected_impact"])

    return pd.DataFrame(
        {
            "starts_at": [_normalize_dt(r.starts_at) for r in rows],
            "ends_at": [_normalize_dt(r.ends_at) for r in rows],
            "expected_impact": [float(r.expected_impact or 0.0) for r in rows],
        }
    )


def _load_lecture_df(zone_id: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Load lecture activity snapshots for a zone within the given time window.

    Provides lecture density signals (active_lectures, starts_next_60m, etc.)
    used by the feature engineering pipeline to compute lecture-impact features.
    Returns an empty DataFrame with correct schema when no lecture data exists.
    """
    with SessionLocal() as db:
        rows = db.execute(
            select(LectureActivity)
            .where(LectureActivity.zone_id == zone_id)
            .where(LectureActivity.ts >= from_dt)
            .where(LectureActivity.ts <= to_dt)
            .order_by(LectureActivity.ts.asc())
        ).scalars().all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "active_lectures",
                "active_courses",
                "starts_next_60m",
                "ends_next_60m",
                "quality_score",
                "quality_flags",
                "metadata",
            ]
        )

    return pd.DataFrame(
        {
            "timestamp": [_normalize_dt(r.ts) for r in rows],
            "active_lectures": [int(r.active_lectures) for r in rows],
            "active_courses": [int(r.active_courses) for r in rows],
            "starts_next_60m": [int(r.starts_next_60m) for r in rows],
            "ends_next_60m": [int(r.ends_next_60m) for r in rows],
            "quality_score": [float(r.quality_score) for r in rows],
            "quality_flags": [r.quality_flags or [] for r in rows],
            "metadata": [r.metadata_json or {} for r in rows],
        }
    )


def _strip_lecture_impact_metadata(lecture_df: pd.DataFrame) -> pd.DataFrame:
    """Remove large non-essential metadata fields from lecture_df before training.

    Keeps only the small fields needed for provenance (site_code, events_used,
    unique_courses) and discards the larger nested API payloads.  This reduces
    the size of lineage records stored per model run.
    """
    if lecture_df.empty or "metadata" not in lecture_df.columns:
        return lecture_df

    sanitized: list[dict[str, Any]] = []
    for metadata in lecture_df["metadata"]:
        if not isinstance(metadata, dict):
            sanitized.append({})
            continue
        sanitized.append(
            {
                key: metadata[key]
                for key in ("site_code", "events_used", "unique_courses")
                if key in metadata
            }
        )

    out = lecture_df.copy()
    out["metadata"] = sanitized
    return out


def _history_to_series(history_df: pd.DataFrame) -> pd.Series:
    """Convert occupancy history DataFrame to a 1-minute-resolution time series.

    Resamples to 1-minute intervals and linearly interpolates short gaps.
    This uniform series is the input for the baseline forecast and for
    the seasonal-naive + regression blending in _forecast_baseline.
    """
    if history_df.empty:
        return pd.Series(dtype=float)

    df = history_df[["timestamp", "occupancy"]].copy()
    df = df.set_index("timestamp").sort_index()
    series = df["occupancy"].resample("1min").mean().interpolate(limit_direction="both")
    return series


def _context_status(history_df: pd.DataFrame, now: datetime) -> dict[str, Any]:
    """Compute a quality assessment of the input context (recent sensor data).

    Returns a dict with point_count, age_seconds, stale flag, a composite
    quality score, and a list of human-readable flags.  These flags are
    attached to every ForecastResponse.evidence.quality block to inform
    the XAI service and the dashboard about data freshness.

    Score formula:
        score = recent_quality * 0.60 + density_score * 0.25 + freshness * 0.15

    Flags:
        CONTEXT_SPARSE  — fewer than FORECAST_CONTEXT_MIN_POINTS data points
        CONTEXT_STALE   — most recent reading older than FORECAST_CONTEXT_STALE_SECONDS
        CONTEXT_LOW_QUALITY — mean quality_score of recent 60 readings < 0.7
    """
    if history_df.empty:
        return {
            "point_count": 0,
            "latest_ts": None,
            "from_ts": (now - timedelta(hours=24)).isoformat(),
            "to_ts": now.isoformat(),
            "age_seconds": None,
            "stale": True,
            "score": 0.0,
            "flags": ["NO_CONTEXT_DB", "CONTEXT_STALE"],
        }

    ts = pd.to_datetime(history_df["timestamp"], utc=True)
    from_ts = _normalize_dt(ts.min().to_pydatetime())
    latest_ts = _normalize_dt(ts.max().to_pydatetime())
    age_seconds = max(0, int((now - latest_ts).total_seconds()))
    point_count = int(len(history_df))

    quality_col = history_df["quality_score"] if "quality_score" in history_df.columns else pd.Series([1.0])
    recent_quality = float(pd.to_numeric(quality_col, errors="coerce").tail(min(point_count, 60)).mean())
    if not math.isfinite(recent_quality):
        recent_quality = 0.5

    flags: list[str] = []
    if point_count < FORECAST_CONTEXT_MIN_POINTS:
        flags.append("CONTEXT_SPARSE")
    if age_seconds > FORECAST_CONTEXT_STALE_SECONDS:
        flags.append("CONTEXT_STALE")
    if recent_quality < 0.7:
        flags.append("CONTEXT_LOW_QUALITY")
    if not flags:
        flags = ["OK"]

    density_score = min(1.0, point_count / max(1, FORECAST_CONTEXT_MIN_POINTS))
    freshness_score = 1.0 if age_seconds <= FORECAST_CONTEXT_STALE_SECONDS else 0.0
    score = max(0.0, min(1.0, recent_quality * 0.6 + density_score * 0.25 + freshness_score * 0.15))
    return {
        "point_count": point_count,
        "latest_ts": latest_ts.isoformat(),
        "from_ts": from_ts.isoformat(),
        "to_ts": latest_ts.isoformat(),
        "age_seconds": age_seconds,
        "stale": age_seconds > FORECAST_CONTEXT_STALE_SECONDS,
        "score": score,
        "flags": flags,
    }


def _context_sources(zone_id: str, context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"type": "counts", "id": f"zone:{zone_id}"},
        {
            "type": "counts",
            "id": f"window:{context.get('from_ts')}..{context.get('to_ts')}",
            "note": (
                f"points={context.get('point_count', 0)} "
                f"latest={context.get('latest_ts')} age_s={context.get('age_seconds')}"
            ),
        },
    ]


def _effective_step_minutes(horizon: int) -> int:
    """Return the step size in minutes between forecast points.

    For horizons beyond 24 hours, the step grows to LONG_HORIZON_STEP_MINUTES
    (default 60 min) to avoid returning thousands of forecast points in the
    response for e.g. a 7-day horizon.  Short horizons use 1-minute steps.
    """
    if horizon > 24 * 60:
        return max(1, LONG_HORIZON_STEP_MINUTES)
    return 1


def _bundle_signature(model_dir: str, zone_id: str, horizon: int) -> tuple[int, int, int, int]:
    """Return a 4-tuple of file modification timestamps for cache invalidation.

    If any bundle file has been written (retraining happened), the signature
    changes and _load_bundle_cached will reload from disk.
    """
    paths = bundle_paths(model_dir, zone_id, horizon)
    return (
        paths["keras_model"].stat().st_mtime_ns,
        paths["scaler"].stat().st_mtime_ns,
        paths["metadata"].stat().st_mtime_ns,
        paths["residuals"].stat().st_mtime_ns,
    )


def _load_bundle_cached(model_dir: str, zone_id: str, horizon: int) -> dict[str, Any]:
    """Load a TF-MLP bundle from an in-memory cache, refreshing if files changed.

    Loading a Keras model from disk is slow (several seconds).  This cache
    keeps the loaded bundle in RAM and only reloads when the bundle files
    are newer than the cached version (checked via modification timestamps).
    A threading.Lock ensures the cache is safe under concurrent requests.
    """
    key = (model_dir, zone_id, horizon)
    signature = _bundle_signature(model_dir, zone_id, horizon)
    with _BUNDLE_CACHE_LOCK:
        cached = _BUNDLE_CACHE.get(key)
        if cached and cached.get("signature") == signature:
            return cached["bundle"]
    bundle = load_bundle(model_dir, zone_id, horizon)
    with _BUNDLE_CACHE_LOCK:
        _BUNDLE_CACHE[key] = {"signature": signature, "bundle": bundle}
    return bundle


def _gbdt_bundle_signature(model_dir: str, zone_id: str, horizon: int) -> tuple[int, int, int, int, int]:
    """Return file timestamps for GBDT cache invalidation."""
    paths = gbdt_bundle_paths(model_dir, zone_id, horizon)
    feature_importance_mtime = (
        paths["feature_importance"].stat().st_mtime_ns
        if paths["feature_importance"].exists()
        else 0
    )
    return (
        paths["gbdt_q10"].stat().st_mtime_ns,
        paths["gbdt_q50"].stat().st_mtime_ns,
        paths["gbdt_q90"].stat().st_mtime_ns,
        paths["metadata"].stat().st_mtime_ns,
        feature_importance_mtime,
    )


def _prepare_gbdt_model_for_runtime(model: Any) -> Any:
    """Limit LightGBM predict parallelism so concurrent API requests stay responsive."""
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    return model


def _load_gbdt_bundle_cached(model_dir: str, zone_id: str, horizon: int) -> dict[str, Any]:
    """Load a GBDT bundle once per process and refresh it after retraining."""
    key = (model_dir, zone_id, horizon)
    signature = _gbdt_bundle_signature(model_dir, zone_id, horizon)
    with _GBDT_BUNDLE_CACHE_LOCK:
        cached = _GBDT_BUNDLE_CACHE.get(key)
        if cached and cached.get("signature") == signature:
            return cached["bundle"]

    bundle = load_gbdt_bundle(model_dir, zone_id, horizon)
    for model_key in ("model_q10", "model_q50", "model_q90"):
        bundle[model_key] = _prepare_gbdt_model_for_runtime(bundle[model_key])

    with _GBDT_BUNDLE_CACHE_LOCK:
        _GBDT_BUNDLE_CACHE[key] = {"signature": signature, "bundle": bundle}
    return bundle


def _forecast_baseline(series: pd.Series, horizon: int, capacity: float = 100.0) -> tuple[list[ForecastPoint], dict[str, Any]]:
    """Generate a baseline forecast using seasonal-naive + linear regression blend.

    This is the ultimate fallback: it requires only a time series of recent
    occupancy values and no trained model.  It is always available.

    Algorithm:
      1. Seasonal-naive component: look back `seasonal_lag` steps to get the
         occupancy at the "same time" in the recent past.  seasonal_lag is
         calibrated to 1 hour (60 minutes) of history.
      2. Regression component: fit a linear model on (time, sin(minute), cos(minute))
         to capture a broad time-of-day trend.
      3. Blend: yhat = seasonal + regression + a damped persistence/momentum
         term, clipped to [0, capacity].
      4. Prediction interval: 10th/90th percentiles of (values[t] - values[t-lag])
         residuals, widened by a small horizon-aware floor when the residuals
         collapse to zero.

    The blend favours the seasonal component because occupancy at the same
    time one hour ago is often the strongest predictor when no ML model is
    available.  The interval floor is intentionally only a fallback guard: it
    prevents flat/interpolated history from returning pi_low == yhat == pi_high.

    Args:
        series: 1-minute-resolution occupancy time series (from _history_to_series).
        horizon: How many minutes ahead to forecast.
        capacity: Zone capacity cap for clipping (April 2026 fix).

    Returns:
        Tuple of (list of ForecastPoint, metadata dict with q10/q90/residual_count).
    """
    now = datetime.now(UTC)
    step_minutes = _effective_step_minutes(horizon)
    steps = max(1, math.ceil(horizon / step_minutes))
    step_delta = timedelta(minutes=step_minutes)

    if series.empty:
        fallback = 0.0
        points = [
            ForecastPoint(
                timestamp=now + step_delta * i,
                yhat=fallback,
                pi_low=max(0.0, fallback - 2.0),
                pi_high=fallback + 2.0,
            )
            for i in range(1, steps + 1)
        ]
        return points, {"q10": -2.0, "q90": 2.0, "residual_count": 0, "step_minutes": step_minutes}

    values = series.values.astype(float)
    last_ts = _normalize_dt(series.index[-1].to_pydatetime())

    seasonal_lag = max(1, int(round(60 / step_minutes)))
    seasonal_base = []
    for i in range(1, steps + 1):
        idx = len(values) - seasonal_lag + i - 1
        if 0 <= idx < len(values):
            seasonal_base.append(float(values[idx]))
        else:
            seasonal_base.append(float(values[-1]))

    n = len(values)
    x = np.arange(n)
    minute_of_day = np.array([ts.hour * 60 + ts.minute for ts in series.index])
    sin_t = np.sin(2 * np.pi * minute_of_day / 1440)
    cos_t = np.cos(2 * np.pi * minute_of_day / 1440)
    X = np.column_stack([x, sin_t, cos_t])

    model = LinearRegression()
    model.fit(X, values)

    future_x = np.arange(n, n + steps)
    future_idx = [last_ts + step_delta * i for i in range(1, steps + 1)]
    future_minute = np.array([ts.hour * 60 + ts.minute for ts in future_idx])
    future_X = np.column_stack(
        [
            future_x,
            np.sin(2 * np.pi * future_minute / 1440),
            np.cos(2 * np.pi * future_minute / 1440),
        ]
    )
    if len(values) > seasonal_lag + 10:
        residuals = values[seasonal_lag:] - values[:-seasonal_lag]
    else:
        residuals = np.diff(values) if len(values) > 2 else np.array([0.0])

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    diff_residuals = np.diff(values) if len(values) > 2 else np.array([], dtype=float)
    diff_residuals = diff_residuals[np.isfinite(diff_residuals)]

    def _robust_sigma(samples: np.ndarray) -> float:
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0
        median = float(np.median(finite))
        mad_sigma = float(np.median(np.abs(finite - median)) * 1.4826)
        std_sigma = float(np.std(finite))
        if finite.size >= 4:
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
        else:
            iqr_sigma = 0.0
        return max(mad_sigma, iqr_sigma, std_sigma * 0.5)

    residual_sigma = _robust_sigma(residuals)
    diff_sigma = _robust_sigma(diff_residuals)
    if capacity > 0:
        interval_floor = min(capacity * 0.25, max(1.5, capacity * 0.04))
    else:
        interval_floor = 0.0

    regression_pred = model.predict(future_X)
    seasonal_arr = np.array(seasonal_base, dtype=float)
    if len(values) > 1:
        momentum_window = min(len(values) - 1, max(5, int(round(30 / step_minutes))))
    else:
        momentum_window = 0
    if momentum_window > 0:
        future_steps = np.arange(1, steps + 1, dtype=float)
        recent_step_change = (float(values[-1]) - float(values[-1 - momentum_window])) / momentum_window
        damping = np.exp(-(future_steps * step_minutes) / 90.0)
        momentum = recent_step_change * future_steps * damping
        momentum_cap = max(interval_floor, residual_sigma, diff_sigma) * 1.5
        if capacity > 0:
            momentum_cap = min(momentum_cap, capacity * 0.12)
        momentum = np.clip(momentum, -momentum_cap, momentum_cap)
    else:
        momentum = np.zeros(steps, dtype=float)
    persistence_pred = np.full(steps, float(values[-1])) + momentum

    yhat = np.clip(0.55 * seasonal_arr + 0.35 * regression_pred + 0.10 * persistence_pred, 0.0, capacity)

    q10 = float(np.quantile(residuals, 0.10)) if residuals.size else -2.0
    q90 = float(np.quantile(residuals, 0.90)) if residuals.size else 2.0
    if q90 < q10:
        q10, q90 = q90, q10

    points = []
    for step_no, (idx, pred) in enumerate(zip(future_idx, yhat), start=1):
        horizon_minutes = max(step_minutes, step_no * step_minutes)
        growth = 1.0 + 0.20 * min(2.0, math.sqrt(horizon_minutes / 60.0))
        min_half_width = interval_floor * growth
        low_raw = float(pred + q10)
        high_raw = float(pred + q90)
        low = min(low_raw, float(pred - min_half_width))
        high = max(high_raw, float(pred + min_half_width))
        pi_low = max(0.0, min(float(pred), low))
        pi_high = min(capacity, max(float(pred), high))
        points.append(ForecastPoint(timestamp=idx, yhat=float(pred), pi_low=pi_low, pi_high=pi_high))

    return points, {
        "q10": q10,
        "q90": q90,
        "residual_count": int(residuals.size),
        "step_minutes": step_minutes,
        "interval_floor": interval_floor,
        "residual_sigma": residual_sigma,
        "diff_sigma": diff_sigma,
    }


def _forecast_lgbm(
    zone_id: str,
    horizon: int,
    history_df: pd.DataFrame,
    context: dict[str, Any],
    now: datetime,
    capacity: float = 100.0,
) -> tuple[list[ForecastPoint], dict[str, Any]]:
    """Generate a single-step quantile forecast using the LightGBM GBDT model.

    The GBDT model was trained on 15-minute Excel data (features_excel.py,
    feature version 'excel_v1') but inference uses 1-minute live sensor data
    (features.py).  There is therefore a feature mismatch that must be bridged:

    Feature mismatch bridge (LIVE_TO_EXCEL mapping):
      features.py produces occupancy_lag_1 (1-min step) while excel_v1 uses
      occupancy_lag_1 (15-min step).  The mapping in LIVE_TO_EXCEL translates
      from live column names to training column names where semantically
      equivalent features exist.  Features with no mapping are filled with 0
      and logged as a FEATURE_MISMATCH quality flag.

    Output:
      Single ForecastPoint at timestamp = now + horizon minutes, using:
        - q50 model → yhat (point forecast / median)
        - q03 model → pi_low
        - q97 model → pi_high
      All three outputs are clipped to [0, capacity] and monotonicity is
      enforced (pi_low ≤ yhat ≤ pi_high).

    Args:
        zone_id: Zone to forecast.
        horizon: Forecast horizon in minutes.
        history_df: Recent occupancy readings.
        context: Context quality status dict from _context_status().
        now: Reference datetime for evidence timestamps.
        capacity: Zone capacity cap (April 2026 fix).

    Returns:
        Tuple of ([ForecastPoint], info dict with summary/evidence/lineage).

    Raises:
        FileNotFoundError: If no GBDT bundle exists for the zone/horizon.
        ValueError: If the GBDT model is not promoted.
    """
    import logging
    logger = logging.getLogger("forecast.lgbm")

    status = gbdt_model_status(TF_MODEL_DIR, zone_id, horizon)
    if not status["exists"]:
        raise FileNotFoundError(f"no GBDT model for zone={zone_id} horizon={horizon}")

    gbdt_data = _load_gbdt_bundle_cached(TF_MODEL_DIR, zone_id, horizon)
    metadata = gbdt_data.get("metadata", {})

    if not bool(metadata.get("promoted", False)):
        raise ValueError("GBDT model is not promoted; baseline fallback enforced")

    feature_columns = metadata.get("feature_columns", [])
    model_q10 = gbdt_data["model_q10"]
    model_q50 = gbdt_data["model_q50"]
    model_q90 = gbdt_data["model_q90"]

    include_lecture_impact = True
    to_dt = now + timedelta(minutes=horizon)

    def _build_features(history_frame: pd.DataFrame) -> pd.DataFrame:
        from_dt_local = (
            _normalize_dt(history_frame["timestamp"].min().to_pydatetime())
            if not history_frame.empty
            else now - timedelta(hours=72)
        )
        events_local = _load_events_df(zone_id=zone_id, from_dt=from_dt_local, to_dt=to_dt)
        lecture_local = _load_lecture_df(zone_id=zone_id, from_dt=from_dt_local, to_dt=to_dt)
        return build_feature_frame(
            history_df=history_frame,
            events_df=events_local,
            lecture_df=lecture_local,
            use_calendar_features=TF_USE_CALENDAR_FEATURES,
            include_lecture_impact=include_lecture_impact,
        )

    feature_df = _build_features(history_df)
    if feature_df.empty:
        extended_history_df = _load_history_df(
            zone_id=zone_id,
            history_hours=FORECAST_CONTEXT_MAX_HISTORY_HOURS,
        )
        if not extended_history_df.empty and len(extended_history_df) > len(history_df):
            history_df = extended_history_df
            context = _context_status(history_df=history_df, now=now)
            feature_df = _build_features(history_df)
    if feature_df.empty:
        raise ValueError("feature_df is empty for LGBM inference after extended-history retry")

    latest_row = feature_df.tail(1)

    # Feature mapping: features.py produces different column names than excel_v1.
    # Map live features to training feature names where possible.
    LIVE_TO_EXCEL = {
        # Lag mapping (features.py uses 1-min lags, excel uses 15-min step lags)
        "occupancy_lag_1": "occupancy_lag_1",
        "occupancy_lag_2": "occupancy_lag_2",
        "occupancy_lag_3": "occupancy_lag_3",
        # features.py lag_5 ≈ excel lag_4 (close enough for tree models)
        "occupancy_lag_5": "occupancy_lag_4",
        "occupancy_lag_10": "occupancy_lag_8",
        "occupancy_lag_15": "occupancy_lag_16",
        # Rolling stats mapping
        "occupancy_roll_mean_5": "occupancy_roll_mean_4",
        "occupancy_roll_mean_15": "occupancy_roll_mean_8",
        "occupancy_roll_mean_60": "occupancy_roll_mean_16",
        "occupancy_roll_std_5": "occupancy_roll_std_4",
        "occupancy_roll_std_15": "occupancy_roll_std_8",
        # Diffs
        "occupancy_diff_1": "occupancy_diff_1",
        "occupancy_diff_5": "occupancy_diff_4",
        # Time features (same names)
        "minute_sin": "minute_sin",
        "minute_cos": "minute_cos",
        "dow_sin": "dow_sin",
        "dow_cos": "dow_cos",
        # Lecture features mapping
        "lecture_count_now": "lecture_density_proxy",
        "lecture_starts_next_60m": "lecture_starts_proxy",
        "lecture_ends_next_60m": "lecture_ends_proxy",
        "lecture_heavy_now": "lecture_heavy_proxy",
    }

    X_row = np.zeros((1, len(feature_columns)), dtype=np.float32)
    matched = 0
    missing_features = []

    for i, col in enumerate(feature_columns):
        # Try direct column match first
        if col in latest_row.columns:
            X_row[0, i] = float(latest_row[col].iloc[0])
            matched += 1
        else:
            # Try reverse mapping: find a live feature that maps to this excel feature
            live_col = None
            for live_name, excel_name in LIVE_TO_EXCEL.items():
                if excel_name == col and live_name in latest_row.columns:
                    live_col = live_name
                    break
            if live_col:
                X_row[0, i] = float(latest_row[live_col].iloc[0])
                matched += 1
            else:
                missing_features.append(col)

    # Use the request time for calendar/time factors.  Lag and rolling features
    # may come from older context when input is stale, but the forecast target is
    # still "now + horizon", so the time features must represent "now".
    ts = now
    minute_of_day = ts.hour * 60 + ts.minute

    def _set_feature(col: str, val: float) -> None:
        nonlocal matched
        if col not in feature_columns:
            return
        X_row[0, feature_columns.index(col)] = val
        if col in missing_features:
            matched += 1
            missing_features.remove(col)

    _set_feature("minute_sin", float(np.sin(2 * np.pi * minute_of_day / 1440.0)))
    _set_feature("minute_cos", float(np.cos(2 * np.pi * minute_of_day / 1440.0)))
    _set_feature("dow_sin", float(np.sin(2 * np.pi * ts.weekday() / 7.0)))
    _set_feature("dow_cos", float(np.cos(2 * np.pi * ts.weekday() / 7.0)))
    doy = ts.timetuple().tm_yday
    _set_feature("day_of_year_sin", float(np.sin(2 * np.pi * doy / 365.0)))
    _set_feature("day_of_year_cos", float(np.cos(2 * np.pi * doy / 365.0)))
    _set_feature("hour_of_day", float(ts.hour))
    _set_feature("is_weekday", float(1 if ts.weekday() < 5 else 0))

    # Derive Excel-native factors from timestamp when missing
    _MONTH_FACTOR = {1: 1.3, 2: 0.6, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.3,
                     7: 1.1, 8: 0.7, 9: 0.8, 10: 1.2, 11: 1.1, 12: 0.9}
    _WEEKDAY_FACTOR = {0: 1.08, 1: 1.05, 2: 1.0, 3: 0.95, 4: 0.65}  # Mon-Fri
    _WEATHER_FACTOR = {"sunny": 0.88, "normal": 1.0, "rainy": 1.12}

    _set_feature("f_month", _MONTH_FACTOR.get(ts.month, 1.0))
    _set_feature("f_weekday", _WEEKDAY_FACTOR.get(ts.weekday(), 1.0) if ts.weekday() < 5 else 0.65)
    # f_tod: approximate from hour using a simple curve (peak ~17h, 0 outside 10-22)
    if ts.hour < 10 or ts.hour > 22:
        _set_feature("f_tod", 0.0)
    else:
        _tod_approx = max(0.0, 0.3 * np.sin(np.pi * (ts.hour - 10) / 12.0))
        _set_feature("f_tod", float(_tod_approx))
    _set_feature("f_weather", 1.0)  # default: normal weather
    _set_feature("f_bridge", 1.0)   # default: no bridge day
    _set_feature("efficiency", 0.875)  # dataset mean
    _set_feature("capacity_effective", 73.0)  # dataset mean
    _set_feature("bridge_day", 0.0)
    _set_feature("winter_break", 0.0)
    _set_feature("weather_rainy", 0.0)
    _set_feature("weather_sunny", 0.0)
    _set_feature("is_partial_closure", 0.0)

    if "utilization" in latest_row.columns:
        utilization = float(latest_row["utilization"].iloc[0])
        _set_feature("utilization_pct", utilization * 100.0 if utilization <= 1.5 else utilization)

    latest_feature_ts = latest_row.index[-1]

    def _occupancy_at_or_before(minutes_back: int) -> float | None:
        target_ts = pd.Timestamp(latest_feature_ts) - pd.Timedelta(minutes=minutes_back)
        if "occupancy" in feature_df.columns and not feature_df.empty:
            series = feature_df["occupancy"].loc[:target_ts].dropna()
            if not series.empty:
                return float(series.iloc[-1])
        return _load_occupancy_at_or_before(
            zone_id=zone_id,
            target_ts=_normalize_dt(target_ts.to_pydatetime()),
            lookback_hours=24,
        )

    latest_occupancy = float(latest_row["occupancy"].iloc[0]) if "occupancy" in latest_row.columns else 0.0
    lag_day_value = _occupancy_at_or_before(720)
    lag_week_value = _occupancy_at_or_before(4320)
    _set_feature("occupancy_lag_day", latest_occupancy if lag_day_value is None else lag_day_value)
    _set_feature("occupancy_lag_week", latest_occupancy if lag_week_value is None else lag_week_value)

    if missing_features:
        logger.warning(
            f"LGBM inference: {len(missing_features)}/{len(feature_columns)} features "
            f"missing (set to 0): {missing_features[:10]}"
        )

    q03_pred = float(model_q10.predict(X_row)[0])
    q50_pred = float(model_q50.predict(X_row)[0])
    q97_pred = float(model_q90.predict(X_row)[0])

    # Warn if raw model output exceeds physical capacity (indicates model drift or feature issue)
    if q50_pred > capacity or q97_pred > capacity:
        logger.warning(
            f"LGBM raw output exceeds capacity={capacity:.0f}: "
            f"q50={q50_pred:.1f}, q97={q97_pred:.1f} for zone={zone_id} horizon={horizon}. "
            f"Clipping applied. Check feature drift or retraining need."
        )

    # Clip to valid range [0, capacity] and ensure monotonicity
    q03_pred = max(0.0, min(capacity, q03_pred))
    q50_pred = max(0.0, min(capacity, q50_pred))
    q97_pred = max(0.0, min(capacity, q97_pred))
    q03_pred = min(q03_pred, q50_pred)
    q97_pred = max(q97_pred, q50_pred)

    latest_ts = now

    points = [
        ForecastPoint(
            timestamp=latest_ts + timedelta(minutes=horizon),
            yhat=q50_pred,
            pi_low=q03_pred,
            pi_high=q97_pred,
        )
    ]

    model_version = metadata.get("model_version", "lgbm-v1")
    recent_quality = float(history_df["quality_score"].tail(60).mean()) if not history_df.empty else 0.5
    quality_flags = ["LOW_INPUT_QUALITY"] if recent_quality < 0.7 else []
    if missing_features:
        quality_flags.append("FEATURE_MISMATCH")
    for flag in context["flags"]:
        if flag != "OK":
            quality_flags.append(flag)
    if not quality_flags:
        quality_flags = ["OK"]

    lineage = {
        "product": PRODUCT_SHORT_TERM,
        "model_run_id": None,
        "backend": "lgbm",
        "model_version": model_version,
        "scientific_status": metadata.get("scientific_status", "walk_forward_validated"),
        "feature_set_version": metadata.get("feature_set_version", "excel_v1"),
        "include_lecture_impact": True,
        "features_matched": matched,
        "features_total": len(feature_columns),
        "features_missing": missing_features[:10],
    }
    evidence = {
        "evidence_id": f"forecast-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": context["from_ts"],
            "to": context["to_ts"],
        },
        "sources": _context_sources(zone_id=zone_id, context=context),
        "model": {
            "name": "lgbm_quantile_forecaster",
            "version": model_version,
            "backend": "lgbm",
            "product": PRODUCT_SHORT_TERM,
            "lineage": lineage,
        },
        "quality": {
            "score": max(0.0, min(1.0, recent_quality)),
            "flags": quality_flags,
        },
    }

    summary = (
        f"LGBM quantile forecast: occupancy={q50_pred:.1f} "
        f"[{q03_pred:.1f}, {q97_pred:.1f}] at +{horizon}min "
        f"(features: {matched}/{len(feature_columns)} matched)"
    )

    return points, {
        "summary": summary,
        "model_version": model_version,
        "evidence": evidence,
        "lineage": lineage,
    }


def _forecast_tf(zone_id: str, horizon: int, capacity: float = 100.0) -> tuple[list[ForecastPoint], dict[str, Any]]:
    """Disabled legacy TF-MLP inference path."""
    raise RuntimeError(TF_MLP_REMOVED_DETAIL)

    """Generate a multi-step forecast using the TF-MLP challenger model.

    Unlike the GBDT backend (single-step), the TF-MLP produces all `horizon`
    forecast steps in a single forward pass: pred[0]=t+1, ..., pred[h-1]=t+h.

    Prediction intervals are constructed from the validation residuals saved
    during training (bundle['residuals']['q10'] and q90), applied as:
        pi_low  = max(0, yhat + q10[step])
        pi_high = min(capacity, yhat + q90[step])

    This is an empirical 80% prediction interval: if the model is well-calibrated,
    approximately 80% of true values fall within [pi_low, pi_high].

    If the initial history window is too sparse to build lag features, the
    function retries with a wider window (FORECAST_CONTEXT_MAX_HISTORY_HOURS).

    Args:
        zone_id: Zone to forecast.
        horizon: Number of 1-minute steps to predict.
        capacity: Zone capacity cap for clipping (April 2026 fix).

    Returns:
        Tuple of (list of `horizon` ForecastPoints, info dict).

    Raises:
        FileNotFoundError: If no TF bundle exists.
        ValueError: If history is missing or feature frame is empty.
    """
    status = model_status(TF_MODEL_DIR, zone_id, horizon)
    if not status["exists"]:
        raise FileNotFoundError(f"no tf model for zone={zone_id} horizon={horizon}")

    bundle = _load_bundle_cached(TF_MODEL_DIR, zone_id, horizon)
    metadata = bundle.get("metadata", {})
    include_lecture_impact = bool(metadata.get("include_lecture_impact", True))

    def _build_features(history_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from_dt_local = _normalize_dt(history_frame["timestamp"].min().to_pydatetime())
        to_dt_local = datetime.now(UTC) + timedelta(minutes=horizon)
        events_local = _load_events_df(zone_id=zone_id, from_dt=from_dt_local, to_dt=to_dt_local)
        lectures_local = _load_lecture_df(zone_id=zone_id, from_dt=from_dt_local, to_dt=to_dt_local)
        features_local = build_feature_frame(
            history_df=history_frame,
            events_df=events_local,
            lecture_df=lectures_local,
            use_calendar_features=TF_USE_CALENDAR_FEATURES,
            include_lecture_impact=include_lecture_impact,
        )
        return features_local, events_local, lectures_local

    history_hours = max(24, TF_INFERENCE_HISTORY_HOURS)
    history_df = _load_history_df(zone_id=zone_id, history_hours=history_hours)
    if history_df.empty:
        raise ValueError("no history for tf inference")

    feature_df, events_df, lecture_df = _build_features(history_df)
    # If recent context is too sparse for lag/rolling features, retry with a wider window.
    if feature_df.empty and history_hours < FORECAST_CONTEXT_MAX_HISTORY_HOURS:
        extended_history = _load_history_df(
            zone_id=zone_id,
            history_hours=FORECAST_CONTEXT_MAX_HISTORY_HOURS,
        )
        if not extended_history.empty:
            history_df = extended_history
            feature_df, events_df, lecture_df = _build_features(history_df)
    if feature_df.empty:
        raise ValueError("feature_df is empty after tf feature build")

    if not bool(bundle["metadata"].get("promoted", True)):
        raise ValueError("trained TF model is not promoted; baseline fallback enforced")

    feature_columns = bundle["metadata"].get("feature_columns", [])
    vector, latest_ts = build_inference_vector(feature_df, feature_columns)

    X = vector.values.astype(np.float32)
    X_scaled = bundle["scaler"].transform(X).astype(np.float32)
    pred = bundle["model"].predict(X_scaled, verbose=0)[0]

    q10 = np.array(bundle["residuals"].get("q10", []), dtype=float)
    q90 = np.array(bundle["residuals"].get("q90", []), dtype=float)

    if q10.size < horizon:
        fill = q10[-1] if q10.size else -2.0
        q10 = np.pad(q10, (0, horizon - q10.size), constant_values=fill)
    if q90.size < horizon:
        fill = q90[-1] if q90.size else 2.0
        q90 = np.pad(q90, (0, horizon - q90.size), constant_values=fill)

    points: list[ForecastPoint] = []
    for step in range(1, horizon + 1):
        yhat = max(0.0, min(capacity, float(pred[step - 1])))
        pi_low = max(0.0, yhat + float(q10[step - 1]))
        pi_high = min(capacity, max(pi_low, yhat + float(q90[step - 1])))
        points.append(
            ForecastPoint(
                timestamp=latest_ts + timedelta(minutes=step),
                yhat=yhat,
                pi_low=pi_low,
                pi_high=pi_high,
            )
        )

    now = datetime.now(UTC)
    context = _context_status(history_df=history_df, now=now)
    recent_quality = float(history_df["quality_score"].tail(60).mean()) if not history_df.empty else 0.5
    quality_flags = ["LOW_INPUT_QUALITY"] if recent_quality < 0.7 else []
    for flag in context["flags"]:
        if flag != "OK":
            quality_flags.append(flag)
    if not quality_flags:
        quality_flags = ["OK"]

    validation_meta = metadata.get("scientific_validation", {}) if isinstance(metadata, dict) else {}
    run_id = (
        validation_meta.get("run_id")
        if isinstance(validation_meta, dict)
        else None
    ) or metadata.get("promoted_by_run_id")
    test_status = (
        validation_meta.get("test_status")
        if isinstance(validation_meta, dict)
        else None
    ) or metadata.get("test_status", "legacy")
    model_version = metadata.get("model_version", "tf-mlp-v1")
    model_version_public = (
        f"{model_version}|run={run_id}|status={test_status}"
        if run_id
        else model_version
    )

    lineage = {
        "model_run_id": metadata.get("model_run_id") or metadata.get("promoted_by_run_id"),
        "product": metadata.get("product", PRODUCT_SHORT_TERM),
        "backend": "tf_mlp",
        "model_version": model_version,
        "feature_set_version": metadata.get("feature_set_version", FEATURE_SET_VERSION),
        "scientific_status": metadata.get("scientific_status", test_status),
        "include_lecture_impact": include_lecture_impact,
        "history_window": metadata.get("history_window", {}),
        "reference_objects": metadata.get("reference_objects", []),
        "metrics": metadata.get("metrics", {}),
    }
    summary = f"TF MLP forecast for zone={zone_id}, horizon={horizon}, model={model_version_public}."

    evidence = {
        "evidence_id": f"forecast-{uuid.uuid4()}",
        "generated_at": datetime.now(UTC).isoformat(),
        "time_window": {
            "from": context["from_ts"],
            "to": context["to_ts"],
        },
        "sources": [*_context_sources(zone_id=zone_id, context=context), {"type": "events", "id": f"events:{len(events_df)}"}],
        "model": {
            "name": "tf_mlp_forecaster",
            "version": model_version,
            "run_id": run_id,
            "test_status": test_status,
            "scientific_validation": validation_meta if isinstance(validation_meta, dict) else {},
            "backend": "tf_mlp",
            "product": metadata.get("product", PRODUCT_SHORT_TERM),
            "lineage": lineage,
        },
        "quality": {
            "score": max(0.0, min(1.0, min(recent_quality, float(context["score"])))),
            "flags": quality_flags,
        },
    }

    return points, {
        "summary": summary,
        "model_version": model_version_public,
        "evidence": evidence,
        "lineage": lineage,
    }


def _set_model_promoted(paths: dict[str, Any], promoted: bool) -> None:
    """Patch the 'promoted' flag in a bundle's metadata.json file on disk.

    This is the write-side of the promotion gate: after the scientific evaluation
    passes, this function marks the on-disk bundle as deployable.
    The inference path (_forecast_tf, _forecast_lgbm) refuses to serve bundles
    where promoted=False, ensuring only validated models reach production.
    """
    metadata_path = paths.get("metadata")
    if not metadata_path:
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["promoted"] = bool(promoted)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except Exception:
        return


def _update_model_metadata(paths: dict[str, Any], patch: dict[str, Any]) -> None:
    """Merge `patch` into the bundle's metadata.json file.

    Used during promotion to enrich the on-disk metadata with scientific
    validation results, reference objects, and lineage records.
    """
    metadata_path = paths.get("metadata")
    if not metadata_path:
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(patch)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except Exception:
        return


def _filter_history_for_training(
    history_df: pd.DataFrame,
    min_quality_score: float,
    exclude_quality_flags: list[str],
) -> pd.DataFrame:
    """Remove low-quality rows from history before training or promotion.

    Applies two filters:
      1. quality_score threshold: rows with score < min_quality_score are dropped.
      2. Hard quality flags: rows tagged with any flag in exclude_quality_flags
         are dropped regardless of their numeric score.

    Hard flag exclusion is especially important for flags like SENSOR_OFFLINE
    or MANUAL_OVERRIDE where the reading cannot be trusted at all.
    """
    if history_df.empty:
        return history_df

    hard_flags = {flag.strip().upper() for flag in exclude_quality_flags if flag.strip()}
    filtered = history_df.copy()
    if "quality_score" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["quality_score"], errors="coerce").fillna(0.0) >= min_quality_score]
    if hard_flags and "quality_flags" in filtered.columns:
        filtered = filtered[
            ~filtered["quality_flags"].apply(
                lambda flags: bool(
                    {
                        str(flag).strip().upper()
                        for flag in (flags if isinstance(flags, list) else [])
                        if str(flag).strip()
                    }
                    & hard_flags
                )
            )
        ]
    return filtered.reset_index(drop=True)


def _extract_improvement(metrics: dict[str, Any]) -> float:
    raw = metrics.get("improvement_vs_baseline_mae_val", 0.0)
    try:
        value = float(raw)
    except Exception:
        return -1.0
    if not math.isfinite(value):
        return -1.0
    return value


def _require_training_maintenance_mode() -> None:
    """Raise HTTP 423 if the service is not in maintenance mode.

    Training is blocked by default (FORECAST_TRAINING_MODE=locked) to prevent
    accidental production retraining triggered by a stray API call.  Only
    FORECAST_TRAINING_MODE=maintenance allows /v1/train* endpoints to execute.
    HTTP 423 ('Locked') clearly communicates that the resource is intentionally
    unavailable, not broken.
    """
    if TRAINING_MODE == "maintenance":
        return
    raise HTTPException(
        status_code=423,
        detail="training disabled in locked mode (set FORECAST_TRAINING_MODE=maintenance for controlled maintenance runs)",
    )


def _raise_tf_mlp_removed() -> None:
    raise HTTPException(status_code=410, detail=TF_MLP_REMOVED_DETAIL)


@app.get("/health")
def health() -> dict[str, Any]:
    """Service health check endpoint.

    Returns the LGBM backend, training mode, and backend model availability.
    Used by the api-gateway liveness probe and by the dashboard health panel.
    """
    status = "ok"
    gbdt_status = gbdt_model_status(
        TF_MODEL_DIR,
        DEFAULT_ZONE_ID,
        TF_DEFAULT_HORIZON,
        verify_load=True,
    )
    backend_status: dict[str, Any] | None = _compact_bundle_status(gbdt_status)
    if (
        not backend_status["exists"]
        or backend_status["loadable"] is False
        or backend_status["promoted"] is False
    ):
        status = "degraded"

    payload = {
        "status": status,
        "service": "forecast",
        "backend": MODEL_BACKEND,
        "configured_backend": _CONFIGURED_MODEL_BACKEND,
        "backend_config_warning": MODEL_BACKEND_CONFIG_WARNING,
        "training_mode": TRAINING_MODE,
        "training_enabled": TRAINING_MODE == "maintenance",
    }
    if backend_status is not None:
        payload["backend_model_status"] = backend_status
    return payload


@app.get("/v1/forecast", response_model=ForecastResponse)
def get_forecast(
    zone_id: str = Query(...),
    horizon: int = Query(default=TF_DEFAULT_HORIZON, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
) -> ForecastResponse:
    """Primary short-term forecast endpoint.

    Attempts to generate a forecast using LGBM. If LGBM raises an exception
    (model not found, not promoted, feature error), it falls back to baseline
    with explicit LGBM_FALLBACK evidence.

    Fallback chain:
        1. lgbm → baseline if lgbm fails
        2. baseline fallback only if LGBM fails

    The response always includes evidence metadata and a lineage dict so that
    the XAI service and ECP v2 context builder can explain the forecast.

    Args:
        zone_id: Zone identifier to forecast (e.g. "default-zone").
        horizon: How many minutes ahead to forecast (1 to MAX_FORECAST_HORIZON_MINUTES).

    Returns:
        ForecastResponse with yhat, prediction interval, evidence, and lineage.
    """
    now = datetime.now(UTC)
    history_hours = max(24, min(FORECAST_CONTEXT_MAX_HISTORY_HOURS, int(math.ceil(horizon / 60 * 2))))
    history_df = _load_history_df(zone_id=zone_id, history_hours=history_hours)
    history_series = _history_to_series(history_df)
    context = _context_status(history_df=history_df, now=now)
    capacity = _get_zone_capacity(zone_id)

    if (
        MODEL_BACKEND == "lgbm"
        and history_hours < FORECAST_CONTEXT_MAX_HISTORY_HOURS
        and (history_df.empty or len(history_df) < FORECAST_CONTEXT_MIN_POINTS)
    ):
        extended_history_df = _load_history_df(
            zone_id=zone_id,
            history_hours=FORECAST_CONTEXT_MAX_HISTORY_HOURS,
        )
        if not extended_history_df.empty:
            history_df = extended_history_df
            history_series = _history_to_series(history_df)
            context = _context_status(history_df=history_df, now=now)

    if MODEL_BACKEND == "lgbm":
        try:
            points, info = _forecast_lgbm(zone_id=zone_id, horizon=horizon, history_df=history_df, context=context, now=now, capacity=capacity)
            return ForecastResponse(
                zone_id=zone_id,
                horizon=horizon,
                generated_at=now,
                summary=info["summary"],
                model_version=info["model_version"],
                points=points,
                evidence=info["evidence"],
                lineage=info.get("lineage"),
            )
        except Exception as exc:
            points, baseline_info = _forecast_baseline(history_series, horizon, capacity=capacity)
            step_minutes = int(baseline_info.get("step_minutes", 1))
            fallback_error = _exception_summary(exc)
            summary = f"LGBM backend fallback to baseline due to: {fallback_error} (step={step_minutes}m)"
            lineage = {
                "product": PRODUCT_SHORT_TERM,
                "model_run_id": None,
                "backend": "baseline",
                "model_version": "baseline-fallback-v1",
                "scientific_status": "fallback",
                "feature_set_version": FEATURE_SET_VERSION,
                "include_lecture_impact": False,
            }
            evidence = {
                "evidence_id": f"forecast-{uuid.uuid4()}",
                "generated_at": now.isoformat(),
                "time_window": {
                    "from": context["from_ts"],
                    "to": context["to_ts"],
                },
                "sources": _context_sources(zone_id=zone_id, context=context),
                "model": {
                    "name": "baseline_forecaster",
                    "version": "baseline-fallback-v1",
                    "backend": "baseline",
                    "product": PRODUCT_SHORT_TERM,
                    "fallback_error": fallback_error,
                    "lineage": lineage,
                },
                "quality": {
                    "score": max(0.0, min(1.0, min(0.6, float(context["score"])))),
                    "flags": [
                        "LGBM_FALLBACK",
                        f"LGBM_FALLBACK:{exc.__class__.__name__}",
                        *[flag for flag in context["flags"] if flag != "OK"],
                    ],
                },
            }
            return ForecastResponse(
                zone_id=zone_id,
                horizon=horizon,
                generated_at=now,
                summary=summary,
                model_version="baseline-fallback-v1",
                points=points,
                evidence=evidence,
                lineage=lineage,
            )

    points, baseline_info = _forecast_baseline(history_series, horizon, capacity=capacity)
    step_minutes = int(baseline_info.get("step_minutes", 1))
    if history_series.empty:
        summary = f"No history found. Returning neutral forecast baseline (step={step_minutes}m)."
    else:
        summary = (
            f"Baseline forecast uses seasonal-naive + regression blend over {len(history_series)} points "
            f"(step={step_minutes}m)."
        )

    lineage = {
        "product": PRODUCT_SHORT_TERM,
        "model_run_id": None,
        "backend": MODEL_BACKEND,
        "model_version": f"{MODEL_BACKEND}-v1",
        "scientific_status": "baseline_runtime",
        "feature_set_version": FEATURE_SET_VERSION,
        "include_lecture_impact": False,
    }
    evidence = {
        "evidence_id": f"forecast-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": context["from_ts"],
            "to": context["to_ts"],
        },
        "sources": _context_sources(zone_id=zone_id, context=context),
        "model": {
            "name": "baseline_forecaster",
            "version": "v1",
            "backend": MODEL_BACKEND,
            "product": PRODUCT_SHORT_TERM,
            "lineage": lineage,
        },
        "quality": {
            "score": max(0.0, min(1.0, float(context["score"]))),
            "flags": [*([flag for flag in context["flags"] if flag != "OK"]), *(["SPARSE_HISTORY"] if len(history_series) < 120 else [])]
            or ["OK"],
        },
    }

    return ForecastResponse(
        zone_id=zone_id,
        horizon=horizon,
        generated_at=now,
        summary=summary,
        model_version=f"{MODEL_BACKEND}-v1",
        points=points,
        evidence=evidence,
        lineage=lineage,
    )


@app.get("/v1/forecast/weekly", response_model=WeeklyForecastResponse)
def get_weekly_forecast(
    zone_id: str = Query(...),
    days: int = Query(default=WEEKLY_FORECAST_DAYS_DEFAULT, ge=1, le=14),
    slot_minutes: int = Query(default=WEEKLY_FORECAST_SLOT_MINUTES, ge=15, le=240),
) -> WeeklyForecastResponse:
    """Weekly deterministic slot forecast endpoint.

    Generates an occupancy forecast for the next `days` calendar days,
    aggregated into `slot_minutes`-wide time slots (e.g. hourly).

    Unlike the short-term forecast, the weekly forecast does NOT use an ML model.
    Instead it uses historical pattern aggregation (build_weekly_forecast in weekly.py):
      - Group past observations by (day_of_week, slot_of_day)
      - Compute mean occupancy per slot as the base prediction
      - Apply event and lecture delta adjustments
      - Clip to [0, capacity]

    This approach is reliable even without a trained model, making it suitable
    for planning scenarios (e.g. "will the library be busy on Thursday afternoon?").

    Args:
        zone_id: Zone identifier.
        days: Number of calendar days to forecast ahead (1–14).
        slot_minutes: Duration of each forecast slot in minutes (15–240).

    Returns:
        WeeklyForecastResponse with per-slot ForecastPoints and daily summaries.
    """
    history_hours = max(24 * 28, 24 * max(days * 2, 14))
    history_df = _load_history_df(zone_id=zone_id, history_hours=history_hours)
    capacity = _get_zone_capacity(zone_id)
    from_dt = (
        _normalize_dt(history_df["timestamp"].min().to_pydatetime())
        if not history_df.empty
        else datetime.now(UTC) - timedelta(hours=history_hours)
    )
    to_dt = datetime.now(UTC) + timedelta(days=days)
    events_df = _load_events_df(zone_id=zone_id, from_dt=from_dt, to_dt=to_dt)
    lecture_df = _load_lecture_df(zone_id=zone_id, from_dt=from_dt, to_dt=to_dt)
    payload = build_weekly_forecast(
        zone_id=zone_id,
        history_df=history_df,
        events_df=events_df,
        lecture_df=lecture_df,
        days=days,
        slot_minutes=slot_minutes,
        capacity=capacity,
    )
    latest_lineage = _latest_model_lineage(zone_id=zone_id, product=WEEKLY_PRODUCT)
    if latest_lineage:
        payload["lineage"] = {
            **payload.get("lineage", {}),
            "model_run_id": latest_lineage.get("model_run_id"),
            "scientific_status": latest_lineage.get("scientific_status"),
            "reference_objects": latest_lineage.get("lineage", {}).get("reference_objects", []),
        }
    else:
        payload["lineage"] = {
            **payload.get("lineage", {}),
            "reference_objects": _latest_training_references(zone_id=zone_id),
        }
    return WeeklyForecastResponse.model_validate(payload)


@app.post("/v1/train")
def train_model(request: TrainRequest) -> dict[str, Any]:
    """Legacy TF-MLP training endpoint.

    Removed from the active service policy: short-term runtime and model
    management are LGBM-only.
    """
    _raise_tf_mlp_removed()

    """Train a TF-MLP model for a single zone/horizon (maintenance mode only).

    Runs the full training pipeline (train_zone_model in train_tf.py) and
    saves the bundle to TF_MODEL_DIR.  The trained model is NOT automatically
    promoted: it receives scientific_status='training_only' and promoted=False.

    To make a trained model active, it must first be scientifically evaluated
    via POST /v1/train/evaluate (rolling-origin CV) and then promoted via
    POST /v1/train/promote.  This two-step process prevents untested models
    from serving live traffic.

    All training data sources (DB window, lecture data, calendar events) are
    recorded as ReferenceObject rows linked to the model run for reproducibility.

    Raises HTTP 423 if FORECAST_TRAINING_MODE != 'maintenance'.
    Raises HTTP 400 if TensorFlow is unavailable or no history exists.
    Raises HTTP 500 if training fails.
    """
    _require_training_maintenance_mode()
    tf_ok, tf_info = is_tensorflow_available()
    if not tf_ok:
        raise HTTPException(status_code=500, detail=f"tensorflow unavailable: {tf_info}")

    history_df = _load_history_df(zone_id=request.zone_id, history_hours=request.history_hours)
    if history_df.empty:
        raise HTTPException(status_code=400, detail=f"no history available for zone={request.zone_id}")

    from_dt = _normalize_dt(history_df["timestamp"].min().to_pydatetime())
    to_dt = datetime.now(UTC) + timedelta(minutes=request.horizon)
    events_df = _load_events_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    lecture_df = _load_lecture_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    reference_candidates = _collect_training_references(
        zone_id=request.zone_id,
        history_df=history_df,
        lecture_df=lecture_df if request.include_lecture_impact else pd.DataFrame(),
        events_df=events_df,
        history_hours=request.history_hours,
    )

    config = TrainingConfig(
        model_dir=TF_MODEL_DIR,
        zone_id=request.zone_id,
        horizon=request.horizon,
        product=PRODUCT_SHORT_TERM,
        min_train_points=TF_MIN_TRAIN_POINTS,
        use_calendar_features=TF_USE_CALENDAR_FEATURES,
        include_lecture_impact=request.include_lecture_impact,
        min_quality_score=TF_MIN_QUALITY_SCORE,
        epochs=TF_TRAIN_EPOCHS,
        batch_size=TF_BATCH_SIZE,
        verbose=0,
    )

    try:
        result = train_zone_model(
            history_df=history_df,
            events_df=events_df,
            lecture_df=lecture_df,
            config=config,
        )
        model_run_id = f"train-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        bundle_metadata = {}
        paths = result.get("paths", {})
        metadata_path = paths.get("metadata")
        if isinstance(metadata_path, str) and Path(metadata_path).exists():
            bundle_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        stored_reference_objects = _persist_references_for_model_run(model_run_id, reference_candidates)
        bundle_metadata["model_run_id"] = model_run_id
        bundle_metadata["reference_objects"] = stored_reference_objects
        bundle_metadata["scientific_status"] = "training_only"
        lineage = _build_model_lineage_payload(
            zone_id=request.zone_id,
            product=PRODUCT_SHORT_TERM,
            horizon=request.horizon,
            model_version=result.get("model_version", "unknown"),
            model_backend="tf_mlp",
            feature_set_version=bundle_metadata.get("feature_set_version", FEATURE_SET_VERSION),
            include_lecture_impact=request.include_lecture_impact,
            promoted=False,
            scientific_status="training_only",
            model_run_id=model_run_id,
            metadata=bundle_metadata,
            reference_objects=stored_reference_objects,
        )
        bundle_metadata["lineage"] = lineage
        _update_model_metadata(paths, bundle_metadata)
        _upsert_model_run(
            model_run_id=model_run_id,
            zone_id=request.zone_id,
            product=PRODUCT_SHORT_TERM,
            horizon=request.horizon,
            model_backend="tf_mlp",
            model_version=result.get("model_version", "unknown"),
            include_lecture_impact=request.include_lecture_impact,
            feature_set_version=bundle_metadata.get("feature_set_version", FEATURE_SET_VERSION),
            raw_rows=len(history_df),
            metadata=bundle_metadata,
            lineage=lineage,
            status="trained",
            scientific_status="training_only",
            promoted=False,
        )
        result["promoted"] = False
        result["promotion_required"] = True
        result["promotion_reason"] = (
            "automatic promotion disabled; use /v1/train/evaluate and /v1/train/promote"
        )
        result["full_retrain"] = request.full_retrain
        result["history_hours"] = request.history_hours
        result["include_lecture_impact"] = request.include_lecture_impact
        result["model_run_id"] = model_run_id
        result["lineage"] = lineage
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"training failed: {exc}") from exc


@app.get("/v1/model/status")
def get_model_status(zone_id: str = Query(...), horizon: int = Query(default=TF_DEFAULT_HORIZON, ge=1, le=720)) -> dict[str, Any]:
    """Check whether the active backend bundle exists and is usable.

    For LightGBM, this verifies that the joblib bundle can actually be loaded
    in the current Python environment; file presence alone is not enough.
    """
    status = gbdt_model_status(TF_MODEL_DIR, zone_id, horizon, verify_load=True)
    metadata = _safe_json_dict(status.get("metadata"))
    status["model_version"] = status.get("model_version") or metadata.get("model_version")
    status["active_backend"] = MODEL_BACKEND
    status["backend"] = status.get("backend") or MODEL_BACKEND
    status["configured_backend"] = _CONFIGURED_MODEL_BACKEND
    status["backend_config_warning"] = MODEL_BACKEND_CONFIG_WARNING
    status["runtime_status"] = (
        "available"
        if status.get("exists") and status.get("loadable") is not False
        else "unavailable"
    )
    status["lineage"] = _latest_model_lineage(zone_id=zone_id, product=PRODUCT_SHORT_TERM, horizon=horizon)
    return status


@app.get("/v1/model/lineage/latest")
def get_model_lineage_latest(
    zone_id: str = Query(...),
    product: str = Query(default=PRODUCT_SHORT_TERM),
    horizon: int | None = Query(default=None, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
) -> dict[str, Any]:
    """Return the latest model lineage record for a zone and product.

    Falls back gracefully when no formal model_run record exists:
    - For short_term + known horizon: reads metadata from the bundle file.
    - For weekly product or unknown horizons: returns a stub with training references.
    """
    if product not in {PRODUCT_SHORT_TERM, WEEKLY_PRODUCT}:
        raise HTTPException(status_code=400, detail=f"unsupported product={product}")

    lineage = _latest_model_lineage(zone_id=zone_id, product=product, horizon=horizon)
    if lineage is not None:
        return lineage

    if product == PRODUCT_SHORT_TERM and horizon is not None:
        status = gbdt_model_status(TF_MODEL_DIR, zone_id, horizon, verify_load=True)
        metadata = _safe_json_dict(status.get("metadata"))
        loadable = status.get("loadable")
        runtime_available = bool(status.get("exists")) and loadable is not False
        scientific_status = metadata.get("scientific_status", "unknown")
        if not runtime_available:
            scientific_status = "runtime_unavailable"
        return {
            "model_run_id": metadata.get("model_run_id"),
            "zone_id": zone_id,
            "product": PRODUCT_SHORT_TERM,
            "horizon": horizon,
            "model_backend": metadata.get("backend", status.get("backend", MODEL_BACKEND)),
            "model_version": metadata.get("model_version", f"{MODEL_BACKEND}-v1"),
            "status": "available" if runtime_available else "unavailable" if status.get("exists") else "missing",
            "scientific_status": scientific_status,
            "include_lecture_impact": metadata.get("include_lecture_impact"),
            "feature_set_version": metadata.get("feature_set_version", FEATURE_SET_VERSION),
            "history_from": _safe_json_dict(metadata.get("history_window")).get("from"),
            "history_to": _safe_json_dict(metadata.get("history_window")).get("to"),
            "raw_rows": None,
            "train_rows": metadata.get("train_rows"),
            "val_rows": metadata.get("val_rows"),
            "test_rows": metadata.get("test_rows"),
            "evaluation_run_id": metadata.get("promoted_by_run_id"),
            "promoted": metadata.get("promoted", False),
            "promoted_at": _safe_json_dict(metadata.get("scientific_validation")).get("promoted_at"),
            "promotion_source": "bundle_metadata",
            "metadata": metadata,
            "runtime": {
                "exists": bool(status.get("exists", False)),
                "loadable": loadable,
                "load_error": status.get("load_error"),
                "active_backend": MODEL_BACKEND,
            },
            "lineage": metadata.get("lineage", {}),
            "created_at": metadata.get("trained_at"),
        }

    return {
        "model_run_id": None,
        "zone_id": zone_id,
        "product": product,
        "horizon": horizon,
        "model_backend": "deterministic" if product == WEEKLY_PRODUCT else MODEL_BACKEND,
        "model_version": "weekly-slot-v1" if product == WEEKLY_PRODUCT else f"{MODEL_BACKEND}-v1",
        "status": "missing",
        "scientific_status": "unregistered",
        "include_lecture_impact": None,
        "feature_set_version": WEEKLY_FEATURE_SET_VERSION if product == WEEKLY_PRODUCT else FEATURE_SET_VERSION,
        "history_from": None,
        "history_to": None,
        "raw_rows": None,
        "train_rows": None,
        "val_rows": None,
        "test_rows": None,
        "evaluation_run_id": None,
        "promoted": False,
        "promoted_at": None,
        "promotion_source": None,
        "metadata": {},
        "lineage": {
            "reference_objects": _latest_training_references(zone_id=zone_id),
        },
        "created_at": None,
    }


@app.post("/v1/train/batch")
def train_batch(request: BatchTrainRequest) -> dict[str, Any]:
    """Legacy TF-MLP batch training endpoint.

    Removed from the active service policy: short-term runtime and model
    management are LGBM-only.
    """
    _raise_tf_mlp_removed()

    """Train TF-MLP models for multiple horizons in a single API call.

    Trains one model per horizon sequentially using the same history window.
    Partial failures are tolerated: if training fails for one horizon, the
    others continue and the failed horizon appears in results with status='error'.

    Returns a summary dict with counts of total/promoted/failed horizons.
    All trained models are set to promoted=False (same as single /v1/train).
    """
    _require_training_maintenance_mode()
    tf_ok, tf_info = is_tensorflow_available()
    if not tf_ok:
        raise HTTPException(status_code=500, detail=f"tensorflow unavailable: {tf_info}")
    if not request.horizons:
        raise HTTPException(status_code=400, detail="horizons must not be empty")

    unique_horizons = sorted({int(h) for h in request.horizons if int(h) >= 1 and int(h) <= MAX_FORECAST_HORIZON_MINUTES})
    if not unique_horizons:
        raise HTTPException(status_code=400, detail="no valid horizons in request")

    history_df = _load_history_df(zone_id=request.zone_id, history_hours=request.history_hours)
    if history_df.empty:
        raise HTTPException(status_code=400, detail=f"no history available for zone={request.zone_id}")

    from_dt = _normalize_dt(history_df["timestamp"].min().to_pydatetime())
    max_horizon = max(unique_horizons)
    to_dt = datetime.now(UTC) + timedelta(minutes=max_horizon)
    events_df = _load_events_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    lecture_df = _load_lecture_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    reference_candidates = _collect_training_references(
        zone_id=request.zone_id,
        history_df=history_df,
        lecture_df=lecture_df if request.include_lecture_impact else pd.DataFrame(),
        events_df=events_df,
        history_hours=request.history_hours,
    )

    results: list[dict[str, Any]] = []
    failed = 0

    for horizon in unique_horizons:
        config = TrainingConfig(
            model_dir=TF_MODEL_DIR,
            zone_id=request.zone_id,
            horizon=horizon,
            product=PRODUCT_SHORT_TERM,
            min_train_points=TF_MIN_TRAIN_POINTS,
            use_calendar_features=TF_USE_CALENDAR_FEATURES,
            include_lecture_impact=request.include_lecture_impact,
            min_quality_score=TF_MIN_QUALITY_SCORE,
            epochs=TF_TRAIN_EPOCHS,
            batch_size=TF_BATCH_SIZE,
            verbose=0,
        )
        try:
            result = train_zone_model(
                history_df=history_df,
                events_df=events_df,
                lecture_df=lecture_df,
                config=config,
            )
            model_run_id = f"train-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            bundle_metadata = {}
            paths = result.get("paths", {})
            metadata_path = paths.get("metadata")
            if isinstance(metadata_path, str) and Path(metadata_path).exists():
                bundle_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            stored_reference_objects = _persist_references_for_model_run(model_run_id, reference_candidates)
            bundle_metadata["model_run_id"] = model_run_id
            bundle_metadata["reference_objects"] = stored_reference_objects
            bundle_metadata["scientific_status"] = "training_only"
            lineage = _build_model_lineage_payload(
                zone_id=request.zone_id,
                product=PRODUCT_SHORT_TERM,
                horizon=horizon,
                model_version=result.get("model_version", "unknown"),
                model_backend="tf_mlp",
                feature_set_version=bundle_metadata.get("feature_set_version", FEATURE_SET_VERSION),
                include_lecture_impact=request.include_lecture_impact,
                promoted=False,
                scientific_status="training_only",
                model_run_id=model_run_id,
                metadata=bundle_metadata,
                reference_objects=stored_reference_objects,
            )
            bundle_metadata["lineage"] = lineage
            _update_model_metadata(paths, bundle_metadata)
            _upsert_model_run(
                model_run_id=model_run_id,
                zone_id=request.zone_id,
                product=PRODUCT_SHORT_TERM,
                horizon=horizon,
                model_backend="tf_mlp",
                model_version=result.get("model_version", "unknown"),
                include_lecture_impact=request.include_lecture_impact,
                feature_set_version=bundle_metadata.get("feature_set_version", FEATURE_SET_VERSION),
                raw_rows=len(history_df),
                metadata=bundle_metadata,
                lineage=lineage,
                status="trained",
                scientific_status="training_only",
                promoted=False,
            )
            result["promoted"] = False
            result["promotion_required"] = True
            result["horizon"] = horizon
            result["model_run_id"] = model_run_id
            result["lineage"] = lineage
            results.append(result)
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "status": "error",
                    "zone_id": request.zone_id,
                    "horizon": horizon,
                    "error": str(exc),
                    "promoted": False,
                }
            )

    return {
        "status": "ok" if failed == 0 else "partial",
        "zone_id": request.zone_id,
        "history_hours": request.history_hours,
        "horizons": unique_horizons,
        "include_lecture_impact": request.include_lecture_impact,
        "results": results,
        "summary": {
            "total": len(unique_horizons),
            "promoted": 0,
            "failed": failed,
            "promotion_threshold": None,
            "promotion_required": True,
        },
    }


@app.post("/v1/train/evaluate")
def train_evaluate(request: TrainEvaluateRequest) -> dict[str, Any]:
    """Run a scientific evaluation (rolling-origin CV) for the GBDT model.

    Calls evaluate_training_run() in scientific_eval.py, which trains models
    on `folds` chronological splits and evaluates each against multiple baselines.

    The result contains:
      - Per-fold and aggregate MAE, sMAPE, coverage metrics
      - Diebold-Mariano test results vs persistence baseline
      - A `decision` block with `scientific_pass` (True/False) and reasons
      - `promotion_horizons`: horizons that passed both promotion gates

    If request.save_report=True, the report JSON is saved to disk (TF_MODEL_DIR)
    so it can be retrieved later by GET /v1/model/report/latest or by the
    promote endpoint.

    Promotion gate (both conditions must be met):
      1. MAE improvement vs Persistence H60 >= improvement_threshold (default 8%)
      2. Coverage of [q03, q97] in [coverage_low, coverage_high] (default 85-95%)

    This endpoint does NOT deploy any model.  Use POST /v1/train/promote after
    this returns scientific_pass=True.
    """
    _require_training_maintenance_mode()
    if request.enable_tf:
        _raise_tf_mlp_removed()
    if request.coverage_low > request.coverage_high:
        raise HTTPException(status_code=400, detail="coverage_low must be <= coverage_high")
    if request.primary_horizon not in request.horizons:
        raise HTTPException(status_code=400, detail="primary_horizon must be included in horizons")

    unique_horizons = sorted({int(h) for h in request.horizons if 1 <= int(h) <= MAX_FORECAST_HORIZON_MINUTES})
    if not unique_horizons:
        raise HTTPException(status_code=400, detail="no valid horizons in request")

    history_df = _load_history_df(zone_id=request.zone_id, history_hours=request.history_hours)
    if history_df.empty:
        raise HTTPException(status_code=400, detail=f"no history available for zone={request.zone_id}")

    from_dt = _normalize_dt(history_df["timestamp"].min().to_pydatetime())
    max_horizon = max(unique_horizons)
    to_dt = datetime.now(UTC) + timedelta(minutes=max_horizon)
    events_df = _load_events_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    lecture_df = _load_lecture_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)

    config = EvaluationConfig(
        zone_id=request.zone_id,
        horizons=unique_horizons,
        folds=request.folds,
        train_days=request.train_days,
        val_days=request.val_days,
        test_days=request.test_days,
        gap_minutes=request.gap_minutes,
        origin_stride_minutes=request.origin_stride_minutes,
        max_origins_per_split=request.max_origins_per_split,
        min_quality_score=request.min_quality_score,
        exclude_quality_flags=request.exclude_quality_flags,
        random_seed=request.random_seed,
        tf_epochs=SCIENTIFIC_EVAL_TF_EPOCHS,
        tf_batch_size=TF_BATCH_SIZE,
        tf_min_train_points=max(300, min(TF_MIN_TRAIN_POINTS, 1200)),
        tf_max_horizon_minutes=SCIENTIFIC_EVAL_TF_MAX_HORIZON_MINUTES,
        enable_tf=False,
        enable_sarimax=request.enable_sarimax,
        improvement_threshold=request.improvement_threshold,
        max_long_degradation=request.max_long_degradation,
        coverage_low=request.coverage_low,
        coverage_high=request.coverage_high,
        primary_horizon=request.primary_horizon,
        include_lecture_impact=request.include_lecture_impact,
        segment_min_samples=request.segment_min_samples,
    )

    try:
        report = evaluate_training_run(
            history_df=history_df,
            events_df=events_df,
            lecture_df=lecture_df,
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"scientific evaluation failed: {exc}") from exc

    if request.save_report:
        report_paths = save_evaluation_report(model_dir=TF_MODEL_DIR, zone_id=request.zone_id, report=report)
        report["report_store"] = report_paths

    return report


@app.post("/v1/train/promote")
def train_promote(request: TrainPromoteRequest) -> dict[str, Any]:
    """Legacy TF-MLP promotion endpoint.

    Removed from the active service policy: short-term runtime and model
    management are LGBM-only.
    """
    _raise_tf_mlp_removed()

    """Promote a TF-MLP model to production after scientific validation.

    Validates the pre-conditions:
      1. An evaluation report for request.run_id must exist on disk.
      2. The report's decision.scientific_pass must be True.
      3. The champion model must be 'tf_mlp' (GBDT is promoted separately
         via the train_gbdt.py script).

    If all conditions pass:
      - Re-trains the model on filtered history (using quality settings from
        the evaluation config to match training and evaluation conditions).
      - Sets promoted=True and scientific_status='scientifically_validated'
        in the on-disk bundle metadata.
      - Registers the model run in the database with promotion metadata.

    The model is immediately deployable after this endpoint returns
    (the inference path checks the promoted flag before serving predictions).

    Raises:
        HTTP 404: Evaluation report not found for run_id.
        HTTP 409: Scientific pass failed or champion is not tf_mlp.
        HTTP 400: No history available after quality filtering.
    """
    _require_training_maintenance_mode()
    report = load_evaluation_report(model_dir=TF_MODEL_DIR, zone_id=request.zone_id, run_id=request.run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"evaluation report not found for run_id={request.run_id}")

    decision = report.get("decision", {}) if isinstance(report, dict) else {}
    if not bool(decision.get("scientific_pass", False)):
        raise HTTPException(
            status_code=409,
            detail=f"run {request.run_id} is not scientifically approved for promotion: {decision.get('reasons', [])}",
        )

    champion_model = str(decision.get("champion_model", ""))
    if champion_model != "tf_mlp":
        raise HTTPException(
            status_code=409,
            detail=f"run {request.run_id} champion={champion_model} is not deployable in current runtime backend",
        )

    horizons_from_report = [int(h) for h in decision.get("promotion_horizons", []) if int(h) >= 1]
    if request.horizons:
        requested = sorted({int(h) for h in request.horizons if 1 <= int(h) <= MAX_FORECAST_HORIZON_MINUTES})
        horizons = [h for h in requested if h in set(horizons_from_report)] if horizons_from_report else requested
    else:
        horizons = horizons_from_report

    if not horizons:
        raise HTTPException(status_code=400, detail="no promotable horizons resolved for this run")

    report_config = report.get("config", {}) if isinstance(report, dict) else {}
    min_quality_score = float(report_config.get("min_quality_score", TF_MIN_QUALITY_SCORE))
    include_lecture_impact = bool(report_config.get("include_lecture_impact", True))
    exclude_quality_flags = report_config.get("exclude_quality_flags", SCIENTIFIC_EVAL_EXCLUDE_FLAGS)
    if not isinstance(exclude_quality_flags, list):
        exclude_quality_flags = SCIENTIFIC_EVAL_EXCLUDE_FLAGS

    history_df = _load_history_df(zone_id=request.zone_id, history_hours=request.history_hours)
    history_df = _filter_history_for_training(
        history_df=history_df,
        min_quality_score=min_quality_score,
        exclude_quality_flags=[str(flag) for flag in exclude_quality_flags],
    )
    if history_df.empty:
        raise HTTPException(status_code=400, detail=f"no filtered history available for zone={request.zone_id}")

    from_dt = _normalize_dt(history_df["timestamp"].min().to_pydatetime())
    to_dt = datetime.now(UTC) + timedelta(minutes=max(horizons))
    events_df = _load_events_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    lecture_df = _load_lecture_df(zone_id=request.zone_id, from_dt=from_dt, to_dt=to_dt)
    reference_candidates = _collect_training_references(
        zone_id=request.zone_id,
        history_df=history_df,
        lecture_df=lecture_df if include_lecture_impact else pd.DataFrame(),
        events_df=events_df,
        history_hours=request.history_hours,
    )

    results: list[dict[str, Any]] = []
    failed = 0
    for horizon in horizons:
        config = TrainingConfig(
            model_dir=TF_MODEL_DIR,
            zone_id=request.zone_id,
            horizon=horizon,
            product=PRODUCT_SHORT_TERM,
            min_train_points=TF_MIN_TRAIN_POINTS,
            use_calendar_features=TF_USE_CALENDAR_FEATURES,
            include_lecture_impact=include_lecture_impact,
            min_quality_score=min_quality_score,
            epochs=TF_TRAIN_EPOCHS,
            batch_size=TF_BATCH_SIZE,
            verbose=0,
        )
        try:
            result = train_zone_model(
                history_df=history_df,
                events_df=events_df,
                lecture_df=lecture_df,
                config=config,
            )
            model_run_id = f"promote-{request.run_id}-h{horizon}"
            paths = result.get("paths", {})
            bundle_metadata = {}
            metadata_path = paths.get("metadata")
            if isinstance(metadata_path, str) and Path(metadata_path).exists():
                bundle_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            stored_reference_objects = _persist_references_for_model_run(model_run_id, reference_candidates)
            bundle_metadata["model_run_id"] = model_run_id
            bundle_metadata["reference_objects"] = stored_reference_objects
            bundle_metadata["scientific_status"] = "scientifically_validated"
            _set_model_promoted(paths, promoted=True)
            _update_model_metadata(
                paths,
                {
                    "promoted": True,
                    "model_run_id": model_run_id,
                    "promoted_by_run_id": request.run_id,
                    "test_status": "scientifically_validated",
                    "feature_set_version": bundle_metadata.get("feature_set_version", FEATURE_SET_VERSION),
                    "include_lecture_impact": include_lecture_impact,
                    "product": PRODUCT_SHORT_TERM,
                    "reference_objects": stored_reference_objects,
                    "scientific_status": "scientifically_validated",
                    "scientific_validation": {
                        "run_id": request.run_id,
                        "test_status": "scientifically_validated",
                        "scientific_pass": True,
                        "promoted_at": datetime.now(UTC).isoformat(),
                        "reasons": decision.get("reasons", []),
                    },
                    "full_retrain": request.full_retrain,
                },
            )
            updated_metadata = {}
            if isinstance(metadata_path, str) and Path(metadata_path).exists():
                updated_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            lineage = _build_model_lineage_payload(
                zone_id=request.zone_id,
                product=PRODUCT_SHORT_TERM,
                horizon=horizon,
                model_version=result.get("model_version", "unknown"),
                model_backend="tf_mlp",
                feature_set_version=updated_metadata.get("feature_set_version", FEATURE_SET_VERSION),
                include_lecture_impact=include_lecture_impact,
                promoted=True,
                scientific_status="scientifically_validated",
                model_run_id=model_run_id,
                metadata=updated_metadata,
                reference_objects=stored_reference_objects,
                evaluation_run_id=request.run_id,
                promotion_source="scientific_evaluation_gate",
            )
            _update_model_metadata(paths, {**updated_metadata, "lineage": lineage})
            _upsert_model_run(
                model_run_id=model_run_id,
                zone_id=request.zone_id,
                product=PRODUCT_SHORT_TERM,
                horizon=horizon,
                model_backend="tf_mlp",
                model_version=result.get("model_version", "unknown"),
                include_lecture_impact=include_lecture_impact,
                feature_set_version=updated_metadata.get("feature_set_version", FEATURE_SET_VERSION),
                raw_rows=len(history_df),
                metadata=updated_metadata,
                lineage=lineage,
                status="promoted",
                scientific_status="scientifically_validated",
                evaluation_run_id=request.run_id,
                promoted=True,
                promotion_source="scientific_evaluation_gate",
            )
            result["promoted"] = True
            result["promotion_source"] = "scientific_evaluation_gate"
            result["run_id"] = request.run_id
            result["model_run_id"] = model_run_id
            result["lineage"] = lineage
            results.append(result)
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "status": "error",
                    "zone_id": request.zone_id,
                    "horizon": horizon,
                    "error": str(exc),
                    "promoted": False,
                    "run_id": request.run_id,
                }
            )

    return {
        "status": "ok" if failed == 0 else "partial",
        "zone_id": request.zone_id,
        "run_id": request.run_id,
        "horizons": horizons,
        "include_lecture_impact": include_lecture_impact,
        "results": results,
        "summary": {
            "total": len(horizons),
            "promoted": len(horizons) - failed,
            "failed": failed,
            "source": "scientific_evaluation_gate",
        },
    }


@app.get("/v1/model/report/latest")
def get_latest_model_report(
    zone_id: str = Query(...),
    horizon: int = Query(default=TF_DEFAULT_HORIZON, ge=1, le=MAX_FORECAST_HORIZON_MINUTES),
) -> dict[str, Any]:
    """Return the most recent scientific evaluation report for a zone.

    Loads the JSON report written by the last /v1/train/evaluate call and
    augments it with a horizon_summary showing per-model metrics for the
    requested horizon.  Used by the dashboard's Model Lab tab.

    Raises HTTP 404 if no report has been saved for this zone.
    """
    report = load_latest_evaluation_report(model_dir=TF_MODEL_DIR, zone_id=zone_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"no scientific evaluation report found for zone={zone_id}")

    payload = dict(report)
    horizon_key = str(horizon)
    payload["requested_horizon"] = horizon
    payload["horizon_summary"] = {
        model_name: model_payload.get("horizons", {}).get(horizon_key)
        for model_name, model_payload in payload.get("models", {}).items()
    }
    return payload
