from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI
from icalendar import Calendar
from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sitcheck.db")
CALENDAR_ZONE_ID = os.getenv("CALENDAR_ZONE_ID", os.getenv("DEFAULT_ZONE_ID", "default-zone"))
DEFAULT_ZONE_CAPACITY = int(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))
CALENDAR_ICS_URLS = [entry.strip() for entry in os.getenv("CALENDAR_ICS_URLS", "").split(",") if entry.strip()]
CALENDAR_IMPORT_INTERVAL_SECONDS = int(os.getenv("CALENDAR_IMPORT_INTERVAL_SECONDS", "600"))
CALENDAR_RUN_ONCE = os.getenv("CALENDAR_RUN_ONCE", "false").lower() == "true"

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


app = FastAPI(title="sitcheck-calendar-ingest", version="0.1.0")

_state: dict[str, Any] = {
    "status": "starting",
    "last_run_at": None,
    "last_success_at": None,
    "last_error": None,
    "last_imported": 0,
    "last_updated": 0,
    "last_seen_urls": len(CALENDAR_ICS_URLS),
}


def _to_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        raise TypeError(f"unsupported datetime value: {type(value)}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _load_ics_content(source_url: str) -> str:
    if source_url.startswith("file://"):
        local_path = source_url.removeprefix("file://")
        with open(local_path, "r", encoding="utf-8") as handle:
            return handle.read()

    response = requests.get(source_url, timeout=20)
    response.raise_for_status()
    return response.text


def _source_tag(source_url: str) -> str:
    parsed = urlparse(source_url)
    host = parsed.netloc or "local-file"
    return f"ics:{host}"


def _parse_events(ics_payload: str, source_url: str, zone_id: str) -> list[dict[str, Any]]:
    calendar = Calendar.from_ical(ics_payload)
    out: list[dict[str, Any]] = []

    for component in calendar.walk("VEVENT"):
        uid = str(component.get("uid") or "")
        title = str(component.get("summary") or "Untitled Event")
        starts = _to_utc_datetime(component.decoded("dtstart"))
        ends_raw = component.decoded("dtend") if component.get("dtend") is not None else starts + timedelta(hours=1)
        ends = _to_utc_datetime(ends_raw)

        categories = component.get("categories")
        if categories is None:
            category = None
        elif isinstance(categories, (list, tuple)):
            category = str(categories[0]) if categories else None
        else:
            category = str(categories)

        event_hash = hashlib.sha1(f"{source_url}|{uid}|{starts.isoformat()}".encode("utf-8")).hexdigest()[:24]
        event_id = f"ics-{event_hash}"

        out.append(
            {
                "event_id": event_id,
                "zone_id": zone_id,
                "title": title,
                "category": category,
                "starts_at": starts,
                "ends_at": ends,
                "expected_impact": None,
                "source": _source_tag(source_url),
                "metadata_json": {
                    "uid": uid,
                    "source_url": source_url,
                    "location": str(component.get("location") or ""),
                },
            }
        )

    return out


def _ensure_zone(db: Session) -> None:
    zone = db.get(Zone, CALENDAR_ZONE_ID)
    if zone is not None:
        return

    db.add(
        Zone(
            zone_id=CALENDAR_ZONE_ID,
            name="Default Zone",
            capacity=DEFAULT_ZONE_CAPACITY,
            is_active=True,
            metadata_json={"seeded_by": "calendar-ingest"},
        )
    )
    db.commit()


def _upsert_events(db: Session, events: list[dict[str, Any]]) -> tuple[int, int]:
    inserted = 0
    updated = 0

    for payload in events:
        existing = db.get(CalendarEvent, payload["event_id"])
        if existing is None:
            db.add(CalendarEvent(**payload))
            inserted += 1
            continue

        existing.zone_id = payload["zone_id"]
        existing.title = payload["title"]
        existing.category = payload["category"]
        existing.starts_at = payload["starts_at"]
        existing.ends_at = payload["ends_at"]
        existing.expected_impact = payload["expected_impact"]
        existing.source = payload["source"]
        existing.metadata_json = payload["metadata_json"]
        updated += 1

    db.commit()
    return inserted, updated


def run_import_cycle() -> dict[str, Any]:
    now = datetime.now(UTC)
    _state["last_run_at"] = now.isoformat()

    if not CALENDAR_ICS_URLS:
        _state["status"] = "idle"
        _state["last_error"] = None
        _state["last_imported"] = 0
        _state["last_updated"] = 0
        return {"status": "idle", "reason": "CALENDAR_ICS_URLS not configured", "seen_urls": 0}

    parsed_events: list[dict[str, Any]] = []
    errors: list[str] = []
    for source_url in CALENDAR_ICS_URLS:
        try:
            payload = _load_ics_content(source_url)
            parsed_events.extend(_parse_events(payload, source_url, CALENDAR_ZONE_ID))
        except Exception as exc:  # pragma: no cover
            errors.append(f"{source_url}: {exc}")

    inserted = 0
    updated = 0
    with SessionLocal() as db:
        _ensure_zone(db)
        if parsed_events:
            inserted, updated = _upsert_events(db, parsed_events)

    _state["last_imported"] = inserted
    _state["last_updated"] = updated
    _state["last_seen_urls"] = len(CALENDAR_ICS_URLS)

    if errors:
        _state["status"] = "degraded"
        _state["last_error"] = "; ".join(errors)
    else:
        _state["status"] = "ok"
        _state["last_error"] = None
        _state["last_success_at"] = datetime.now(UTC).isoformat()

    return {
        "status": _state["status"],
        "imported": inserted,
        "updated": updated,
        "parsed_events": len(parsed_events),
        "errors": errors,
    }


async def _import_loop() -> None:
    while True:
        run_import_cycle()
        if CALENDAR_RUN_ONCE:
            return
        await asyncio.sleep(max(30, CALENDAR_IMPORT_INTERVAL_SECONDS))


@app.on_event("startup")
async def startup() -> None:
    Base.metadata.create_all(bind=engine)
    asyncio.create_task(_import_loop())


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": _state["status"],
        "service": "calendar-ingest",
        "last_run_at": _state["last_run_at"],
        "last_success_at": _state["last_success_at"],
        "last_error": _state["last_error"],
        "last_imported": _state["last_imported"],
        "last_updated": _state["last_updated"],
        "last_seen_urls": _state["last_seen_urls"],
    }


@app.post("/run")
def run_now() -> dict[str, Any]:
    return run_import_cycle()


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print(run_import_cycle())
