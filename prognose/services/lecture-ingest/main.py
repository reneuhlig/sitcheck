from __future__ import annotations

import asyncio
import os
import unicodedata
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from icalendar import Calendar
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, UniqueConstraint, create_engine, event, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sitcheck.db")
LECTURE_ZONE_ID = os.getenv("LECTURE_ZONE_ID", os.getenv("DEFAULT_ZONE_ID", "default-zone"))
DEFAULT_ZONE_CAPACITY = int(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))
LECTURE_SITE_CODE = os.getenv("LECTURE_SITE_CODE", "MA").strip().upper()
LECTURE_API_BASE_URL = os.getenv("LECTURE_API_BASE_URL", "https://api.dhbw.app").rstrip("/")
LECTURE_REFRESH_INTERVAL_SECONDS = int(os.getenv("LECTURE_REFRESH_INTERVAL_SECONDS", "1800"))
LECTURE_RUN_ONCE = os.getenv("LECTURE_RUN_ONCE", "false").lower() == "true"
LECTURE_BACKFILL_ENABLED = os.getenv("LECTURE_BACKFILL_ENABLED", "true").lower() == "true"
LECTURE_BACKFILL_MAX_COURSES = int(os.getenv("LECTURE_BACKFILL_MAX_COURSES", "0"))
LECTURE_HISTORY_DAYS = int(os.getenv("LECTURE_HISTORY_DAYS", "400"))
LECTURE_FUTURE_DAYS = int(os.getenv("LECTURE_FUTURE_DAYS", "180"))
LECTURE_REFRESH_HISTORY_DAYS = int(os.getenv("LECTURE_REFRESH_HISTORY_DAYS", "7"))
LECTURE_REFRESH_FUTURE_DAYS = int(os.getenv("LECTURE_REFRESH_FUTURE_DAYS", "14"))
LECTURE_REQUEST_TIMEOUT = float(os.getenv("LECTURE_REQUEST_TIMEOUT", "20"))
LECTURE_INGEST_PORT = int(os.getenv("LECTURE_INGEST_PORT", "8012"))
LECTURE_EFFECT_ENABLED = os.getenv("LECTURE_EFFECT_ENABLED", "true").lower() == "true"
LECTURE_AVG_ATTENDANCE = float(os.getenv("LECTURE_AVG_ATTENDANCE", "20"))
LECTURE_HEAVY_BIB_PERSONS = float(os.getenv("LECTURE_HEAVY_BIB_PERSONS", "4"))
LECTURE_HEAVY_WINDOW_MINUTES = max(1, int(os.getenv("LECTURE_HEAVY_WINDOW_MINUTES", "60")))
LECTURE_HEAVY_PHRASES = [
    item.strip().lower()
    for item in os.getenv(
        "LECTURE_HEAVY_PHRASES",
        "logik und algebra,operations research,grundlagen und logik,maschinelles lernen,machine learning,deep learning,digitale signalverarbeitung",
    ).split(",")
    if item.strip()
]
LECTURE_HEAVY_KEYWORDS = [
    item.strip().lower()
    for item in os.getenv(
        "LECTURE_HEAVY_KEYWORDS",
        "mathe,mathematik,statistik,algorithm,theorie,theoret,physik,analysis,lineare algebra,regelungstechnik",
    ).split(",")
    if item.strip()
]
LECTURE_ONSITE_TYPES = {
    item.strip().upper()
    for item in os.getenv("LECTURE_ONSITE_TYPES", "PRESENCE").split(",")
    if item.strip()
}
LECTURE_IMPACT_MODEL_VERSION = os.getenv("LECTURE_IMPACT_MODEL_VERSION", "lecture-impact-v1")
LECTURE_APP_DISPLAY_URL = os.getenv("LECTURE_APP_DISPLAY_URL", "https://www.dhbw.app/").rstrip("/") + "/"

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


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)


class LectureActivity(Base):
    __tablename__ = "lecture_activity"
    __table_args__ = (UniqueConstraint("zone_id", "ts", name="uq_lecture_activity_zone_ts"),)

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


app = FastAPI(title="sitcheck-lecture-ingest", version="0.1.0")

_state: dict[str, Any] = {
    "status": "starting",
    "last_run_at": None,
    "last_success_at": None,
    "last_error": None,
    "last_backfill_at": None,
    "last_backfill_rows": 0,
    "last_refresh_rows": 0,
    "last_course_count": 0,
    "backfill_done": False,
}


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        raise TypeError(f"unsupported datetime value: {type(value)}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


def _is_heavy_module(event: dict[str, Any]) -> bool:
    title = _normalize_text(str(event.get("name") or ""))
    if not title:
        return False
    if any(phrase in title for phrase in LECTURE_HEAVY_PHRASES):
        return True
    return any(keyword in title for keyword in LECTURE_HEAVY_KEYWORDS)


def _is_onsite_event(event: dict[str, Any]) -> bool:
    if not LECTURE_ONSITE_TYPES:
        return True
    event_type = str(event.get("type") or "").strip().upper()
    return bool(event_type and event_type in LECTURE_ONSITE_TYPES)


def _filter_onsite_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if _is_onsite_event(event)]


def _ensure_zone(db: Session) -> None:
    if db.get(Zone, LECTURE_ZONE_ID) is not None:
        return

    db.add(
        Zone(
            zone_id=LECTURE_ZONE_ID,
            name="Default Zone",
            capacity=DEFAULT_ZONE_CAPACITY,
            is_active=True,
            metadata_json={"seeded_by": "lecture-ingest"},
        )
    )
    db.commit()


def _api_get_json(path: str) -> Any:
    url = f"{LECTURE_API_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(url, timeout=LECTURE_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _fetch_courses(site_code: str) -> list[str]:
    payload = _api_get_json(f"courses/{site_code}/parsed")
    course_names: list[str] = []

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("name"):
                course_names.append(str(item["name"]).strip())
            elif isinstance(item, str):
                course_names.append(item.strip())

    course_names = sorted({course for course in course_names if course})
    if LECTURE_BACKFILL_MAX_COURSES > 0:
        course_names = course_names[:LECTURE_BACKFILL_MAX_COURSES]
    return course_names


def _parse_ics_events(ics_text: str, course_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    calendar = Calendar.from_ical(ics_text)

    for comp in calendar.walk("VEVENT"):
        try:
            start = _to_utc(comp.decoded("dtstart"))
            end_raw = comp.decoded("dtend") if comp.get("dtend") is not None else start + timedelta(hours=1)
            end = _to_utc(end_raw)
        except Exception:
            continue

        if end <= start:
            end = start + timedelta(minutes=1)

        out.append(
            {
                "start": start,
                "end": end,
                "course": course_name,
                "name": str(comp.get("summary") or course_name).strip() or course_name,
                "type": "PRESENCE",
            }
        )

    return out


def _fetch_course_ics_events(site_code: str) -> tuple[list[dict[str, Any]], int, list[str]]:
    courses = _fetch_courses(site_code)
    errors: list[str] = []
    events: list[dict[str, Any]] = []

    for course in courses:
        encoded = quote(course, safe="")
        url = f"{LECTURE_API_BASE_URL}/ics/{encoded}"
        try:
            response = requests.get(url, timeout=LECTURE_REQUEST_TIMEOUT)
            response.raise_for_status()
            events.extend(_parse_ics_events(response.text, course_name=course))
        except Exception as exc:
            errors.append(f"{course}: {exc}")

    return events, len(courses), errors


def _fetch_rapla_events(site_code: str) -> list[dict[str, Any]]:
    payload = _api_get_json(f"rapla/{site_code}/lectures")
    out: list[dict[str, Any]] = []

    if not isinstance(payload, list):
        return out

    for row in payload:
        if not isinstance(row, dict):
            continue

        start_raw = row.get("startTime")
        end_raw = row.get("endTime")
        course = str(row.get("course") or "unknown").strip() or "unknown"
        lecture_name = str(row.get("name") or row.get("title") or course).strip() or course
        lecture_type = str(row.get("type") or "").strip()

        try:
            start = _to_utc(pd.to_datetime(start_raw, utc=True).to_pydatetime())
            end = _to_utc(pd.to_datetime(end_raw, utc=True).to_pydatetime())
        except Exception:
            continue

        if end <= start:
            end = start + timedelta(minutes=1)

        out.append(
            {
                "start": start,
                "end": end,
                "course": course,
                "name": lecture_name,
                "type": lecture_type,
            }
        )

    return out


def _filter_events(events: list[dict[str, Any]], from_dt: datetime, to_dt: datetime) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["end"] >= from_dt and event["start"] <= to_dt
    ]


def _aggregate_activity(
    events: list[dict[str, Any]],
    source: str,
    quality_score: float,
    quality_flags: list[str],
) -> list[dict[str, Any]]:
    if not events:
        return []

    starts: list[tuple[pd.Timestamp, str]] = []
    ends: list[tuple[pd.Timestamp, str]] = []
    minute_starts: defaultdict[pd.Timestamp, int] = defaultdict(int)
    minute_ends: defaultdict[pd.Timestamp, int] = defaultdict(int)
    heavy_starts: list[pd.Timestamp] = []
    heavy_ends: list[pd.Timestamp] = []
    minute_heavy_ends: defaultdict[pd.Timestamp, int] = defaultdict(int)

    for event in events:
        start = pd.Timestamp(event["start"]).tz_convert("UTC").floor("min")
        end = pd.Timestamp(event["end"]).tz_convert("UTC").ceil("min")
        if end <= start:
            end = start + pd.Timedelta(minutes=1)

        course = str(event.get("course") or "unknown")
        is_heavy = _is_heavy_module(event)
        starts.append((start, course))
        ends.append((end, course))
        minute_starts[start] += 1
        minute_ends[end] += 1
        if is_heavy:
            heavy_starts.append(start)
            heavy_ends.append(end)
            minute_heavy_ends[end] += 1

    starts.sort(key=lambda item: item[0])
    ends.sort(key=lambda item: item[0])
    heavy_starts.sort()
    heavy_ends.sort()

    idx_end = max(ends[-1][0], starts[-1][0])
    if LECTURE_EFFECT_ENABLED and heavy_ends:
        idx_end = max(idx_end, heavy_ends[-1] + pd.Timedelta(minutes=LECTURE_HEAVY_WINDOW_MINUTES))
    idx = pd.date_range(start=starts[0][0], end=idx_end, freq="1min", tz="UTC")
    if len(idx) > 1_500_000:
        raise RuntimeError("lecture aggregation window too large; adjust history/future bounds")

    course_active: dict[str, int] = {}
    active_lectures = 0
    start_i = 0
    end_i = 0
    heavy_start_i = 0
    heavy_end_i = 0
    heavy_active = 0

    active_lectures_series: list[int] = []
    active_courses_series: list[int] = []
    starts_series: list[int] = []
    ends_series: list[int] = []
    heavy_active_series: list[int] = []

    for minute in idx:
        while start_i < len(starts) and starts[start_i][0] <= minute:
            _, course = starts[start_i]
            active_lectures += 1
            course_active[course] = course_active.get(course, 0) + 1
            start_i += 1

        while end_i < len(ends) and ends[end_i][0] <= minute:
            _, course = ends[end_i]
            active_lectures = max(0, active_lectures - 1)
            remaining = course_active.get(course, 0) - 1
            if remaining <= 0:
                course_active.pop(course, None)
            else:
                course_active[course] = remaining
            end_i += 1

        while heavy_start_i < len(heavy_starts) and heavy_starts[heavy_start_i] <= minute:
            heavy_active += 1
            heavy_start_i += 1

        while heavy_end_i < len(heavy_ends) and heavy_ends[heavy_end_i] <= minute:
            heavy_active = max(0, heavy_active - 1)
            heavy_end_i += 1

        active_lectures_series.append(active_lectures)
        active_courses_series.append(len(course_active))
        starts_series.append(int(minute_starts.get(minute, 0)))
        ends_series.append(int(minute_ends.get(minute, 0)))
        heavy_active_series.append(heavy_active)

    starts_forward = pd.Series(starts_series, index=idx).iloc[::-1].rolling(window=60, min_periods=1).sum().iloc[::-1]
    ends_forward = pd.Series(ends_series, index=idx).iloc[::-1].rolling(window=60, min_periods=1).sum().iloc[::-1]
    heavy_ended_series = pd.Series([int(minute_heavy_ends.get(minute, 0)) for minute in idx], index=idx)
    heavy_post_series = heavy_ended_series.rolling(window=LECTURE_HEAVY_WINDOW_MINUTES, min_periods=1).sum()

    active_arr = np.array(active_lectures_series, dtype=float)
    heavy_now_arr = np.array(heavy_active_series, dtype=float)
    heavy_post_arr = heavy_post_series.values.astype(float)
    if LECTURE_EFFECT_ENABLED:
        lecture_pull_regular = LECTURE_AVG_ATTENDANCE * active_arr
        heavy_bib_bonus = LECTURE_HEAVY_BIB_PERSONS * (heavy_now_arr + heavy_post_arr)
        lecture_net_pull = lecture_pull_regular - heavy_bib_bonus
    else:
        lecture_pull_regular = np.zeros_like(active_arr)
        heavy_bib_bonus = np.zeros_like(active_arr)
        lecture_net_pull = np.zeros_like(active_arr)

    metadata_base = {
        "site_code": LECTURE_SITE_CODE,
        "events_used": len(events),
        "unique_courses": int(len({event.get('course') for event in events})),
        "impact_model_version": LECTURE_IMPACT_MODEL_VERSION,
        "lecture_effect_enabled": LECTURE_EFFECT_ENABLED,
        "lecture_avg_attendance": LECTURE_AVG_ATTENDANCE,
        "lecture_heavy_bib_persons": LECTURE_HEAVY_BIB_PERSONS,
        "lecture_heavy_window_minutes": LECTURE_HEAVY_WINDOW_MINUTES,
        "lecture_heavy_phrases": LECTURE_HEAVY_PHRASES,
        "lecture_onsite_types": sorted(LECTURE_ONSITE_TYPES),
        "lecture_heavy_keywords": LECTURE_HEAVY_KEYWORDS,
        "external_references": [
            {
                "reference_type": "external_reference",
                "source_type": "dhbw_app",
                "label": f"DHBW App ({LECTURE_SITE_CODE})",
                "url": LECTURE_APP_DISPLAY_URL,
                "metadata": {
                    "site_code": LECTURE_SITE_CODE,
                    "courses_api_url": f"{LECTURE_API_BASE_URL}/courses/{LECTURE_SITE_CODE}/parsed",
                    "rapla_api_url": f"{LECTURE_API_BASE_URL}/rapla/{LECTURE_SITE_CODE}/lectures",
                },
            }
        ],
    }

    out: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for i, minute in enumerate(idx):
        row_metadata = {
            **metadata_base,
            "heavy_active_lectures": int(round(float(heavy_now_arr[i]))),
            "heavy_ended_last_60m": int(round(float(heavy_post_arr[i]))),
            "lecture_pull_regular": float(lecture_pull_regular[i]),
            "heavy_bib_bonus": float(heavy_bib_bonus[i]),
            "lecture_net_pull": float(lecture_net_pull[i]),
        }
        out.append(
            {
                "ts": minute.to_pydatetime(),
                "zone_id": LECTURE_ZONE_ID,
                "active_lectures": int(active_lectures_series[i]),
                "active_courses": int(active_courses_series[i]),
                "starts_next_60m": int(starts_forward.iloc[i]),
                "ends_next_60m": int(ends_forward.iloc[i]),
                "source": source,
                "quality_score": max(0.0, min(1.0, quality_score)),
                "quality_flags": quality_flags,
                "metadata": row_metadata,
                "updated_at": now,
            }
        )

    return out


def _upsert_activity(db: Session, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    dialect = db.bind.dialect.name if db.bind is not None else "unknown"
    if dialect == "postgresql":
        stmt = pg_insert(LectureActivity).values(rows)
        update_map = {
            "active_lectures": stmt.excluded.active_lectures,
            "active_courses": stmt.excluded.active_courses,
            "starts_next_60m": stmt.excluded.starts_next_60m,
            "ends_next_60m": stmt.excluded.ends_next_60m,
            "source": stmt.excluded.source,
            "quality_score": stmt.excluded.quality_score,
            "quality_flags": stmt.excluded.quality_flags,
            "metadata": stmt.excluded.metadata,
            "updated_at": stmt.excluded.updated_at,
        }
        db.execute(stmt.on_conflict_do_update(index_elements=["zone_id", "ts"], set_=update_map))
        db.commit()
        return len(rows)

    written = 0
    for row in rows:
        existing = db.execute(
            select(LectureActivity).where(
                LectureActivity.zone_id == row["zone_id"],
                LectureActivity.ts == row["ts"],
            )
        ).scalar_one_or_none()

        if existing is None:
            payload = dict(row)
            payload["metadata_json"] = payload.pop("metadata", {})
            db.add(LectureActivity(**payload))
        else:
            existing.active_lectures = row["active_lectures"]
            existing.active_courses = row["active_courses"]
            existing.starts_next_60m = row["starts_next_60m"]
            existing.ends_next_60m = row["ends_next_60m"]
            existing.source = row["source"]
            existing.quality_score = row["quality_score"]
            existing.quality_flags = row["quality_flags"]
            existing.metadata_json = row["metadata"]
            existing.updated_at = row["updated_at"]
        written += 1

    db.commit()
    return written


def run_import_cycle() -> dict[str, Any]:
    now = datetime.now(UTC)
    from_dt = now - timedelta(days=max(1, LECTURE_HISTORY_DAYS))
    to_dt = now + timedelta(days=max(1, LECTURE_FUTURE_DAYS))
    refresh_from_dt = now - timedelta(days=max(1, LECTURE_REFRESH_HISTORY_DAYS))
    refresh_to_dt = now + timedelta(days=max(1, LECTURE_REFRESH_FUTURE_DAYS))

    _state["last_run_at"] = now.isoformat()
    _state["status"] = "running"

    errors: list[str] = []
    backfill_rows = 0
    refresh_rows = 0

    with SessionLocal() as db:
        _ensure_zone(db)

        try:
            rapla_events = _fetch_rapla_events(LECTURE_SITE_CODE)
            rapla_events = _filter_onsite_events(rapla_events)
            rapla_events = _filter_events(rapla_events, from_dt=refresh_from_dt, to_dt=refresh_to_dt)
            rows = _aggregate_activity(
                events=rapla_events,
                source="rapla_refresh",
                quality_score=0.9,
                quality_flags=["LECTURE_RAPLA"],
            )
            refresh_rows = _upsert_activity(db, rows)
            _state["last_refresh_rows"] = refresh_rows
            _state["status"] = "ok"
            _state["last_success_at"] = datetime.now(UTC).isoformat()
            _state["last_error"] = None
        except Exception as exc:
            errors.append(f"rapla_refresh: {exc}")

        if LECTURE_BACKFILL_ENABLED and not _state["backfill_done"]:
            try:
                ics_events, course_count, ics_errors = _fetch_course_ics_events(LECTURE_SITE_CODE)
                _state["last_course_count"] = course_count
                errors.extend(ics_errors)
                ics_events = _filter_onsite_events(ics_events)
                ics_events = _filter_events(ics_events, from_dt=from_dt, to_dt=to_dt)
                rows = _aggregate_activity(
                    events=ics_events,
                    source="ics_backfill",
                    quality_score=0.82,
                    quality_flags=["LECTURE_ICS"],
                )
                backfill_rows = _upsert_activity(db, rows)
                _state["last_backfill_at"] = datetime.now(UTC).isoformat()
                _state["last_backfill_rows"] = backfill_rows
                _state["backfill_done"] = True
            except Exception as exc:
                errors.append(f"ics_backfill: {exc}")

    if errors:
        _state["status"] = "degraded"
        _state["last_error"] = "; ".join(errors)
    else:
        _state["status"] = "ok"
        _state["last_success_at"] = datetime.now(UTC).isoformat()
        _state["last_error"] = None

    return {
        "status": _state["status"],
        "backfill_rows": backfill_rows,
        "refresh_rows": refresh_rows,
        "errors": errors,
    }


async def _loop() -> None:
    while True:
        # run_import_cycle performs blocking network + DB IO and must not block the event loop
        await asyncio.to_thread(run_import_cycle)
        if LECTURE_RUN_ONCE:
            return
        await asyncio.sleep(max(60, LECTURE_REFRESH_INTERVAL_SECONDS))


@app.on_event("startup")
async def startup() -> None:
    Base.metadata.create_all(bind=engine)
    asyncio.create_task(_loop())


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": _state["status"],
        "service": "lecture-ingest",
        "site_code": LECTURE_SITE_CODE,
        "zone_id": LECTURE_ZONE_ID,
        "last_run_at": _state["last_run_at"],
        "last_success_at": _state["last_success_at"],
        "last_error": _state["last_error"],
        "last_backfill_at": _state["last_backfill_at"],
        "last_backfill_rows": _state["last_backfill_rows"],
        "last_refresh_rows": _state["last_refresh_rows"],
        "last_course_count": _state["last_course_count"],
        "backfill_done": _state["backfill_done"],
        "lecture_effect_enabled": LECTURE_EFFECT_ENABLED,
        "lecture_avg_attendance": LECTURE_AVG_ATTENDANCE,
        "lecture_heavy_bib_persons": LECTURE_HEAVY_BIB_PERSONS,
        "lecture_heavy_window_minutes": LECTURE_HEAVY_WINDOW_MINUTES,
        "lecture_heavy_phrases": LECTURE_HEAVY_PHRASES,
        "lecture_onsite_types": sorted(LECTURE_ONSITE_TYPES),
        "lecture_heavy_keywords": LECTURE_HEAVY_KEYWORDS,
        "lecture_impact_model_version": LECTURE_IMPACT_MODEL_VERSION,
        "lecture_refresh_history_days": LECTURE_REFRESH_HISTORY_DAYS,
        "lecture_refresh_future_days": LECTURE_REFRESH_FUTURE_DAYS,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=LECTURE_INGEST_PORT)
