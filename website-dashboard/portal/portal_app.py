#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
import time
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

from flask import Flask, Response, jsonify, redirect, request, send_file


CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent
STATIC_ROOT = Path(
    os.getenv(
        "SITCHECK_ORIGINAL_SITE_OUT",
        str(WORKSPACE_ROOT / "website-dashboard" / "runtime" / "original-site-out"),
    )
).resolve()
REALTIME_BASE_URL = os.getenv("SITCHECK_REALTIME_BASE_URL", "http://127.0.0.1:8080")
PROGNOSE_API_BASE_URL = os.getenv("SITCHECK_PROGNOSE_API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_ZONE_ID = os.getenv("SITCHECK_DEFAULT_ZONE_ID", "default-zone")
CC_HORIZON = int(os.getenv("SITCHECK_CC_HORIZON", "210"))
CC_FORECAST_STEP_MINUTES = int(os.getenv("SITCHECK_CC_FORECAST_STEP_MINUTES", "15"))
CC_FORECAST_HORIZON_MINUTES = int(os.getenv("SITCHECK_CC_FORECAST_HORIZON_MINUTES", "180"))
CC_HISTORY_MINUTES = int(os.getenv("SITCHECK_CC_HISTORY_MINUTES", "180"))
CC_HUB_HISTORY_MINUTES = int(os.getenv("SITCHECK_HUB_HISTORY_MINUTES", "180"))
PROXY_TIMEOUT_SECONDS = float(os.getenv("SITCHECK_PROXY_TIMEOUT_SECONDS", "15"))
EXPLAIN_NARRATIVE_TIMEOUT_SECONDS = float(
    os.getenv("SITCHECK_EXPLAIN_NARRATIVE_TIMEOUT_SECONDS", str(max(PROXY_TIMEOUT_SECONDS, 180.0)))
)
CC_TIMEOUT_SECONDS = float(os.getenv("SITCHECK_COMMAND_CENTER_TIMEOUT_SECONDS", str(PROXY_TIMEOUT_SECONDS)))
CC_HUB_COMMAND_CENTER_TIMEOUT_SECONDS = float(
    os.getenv("SITCHECK_HUB_COMMAND_CENTER_TIMEOUT_SECONDS", "2.5")
)
HUB_PROBE_COMMAND_CENTER = os.getenv("SITCHECK_HUB_PROBE_COMMAND_CENTER", "false").lower() == "true"
ANALYTICS_REDIRECT_URL = os.getenv("SITCHECK_ANALYTICS_REDIRECT_URL", "").strip()
DHBW_MANNHEIM_RAPLA_URL = os.getenv(
    "SITCHECK_DHBW_MANNHEIM_RAPLA_URL",
    "https://api.dhbw.app/rapla/MA/lectures",
).strip()
DHBW_MANNHEIM_TIMEOUT_SECONDS = float(os.getenv("SITCHECK_DHBW_MANNHEIM_TIMEOUT_SECONDS", "10"))
DHBW_MANNHEIM_CACHE_TTL_SECONDS = float(os.getenv("SITCHECK_DHBW_MANNHEIM_CACHE_TTL_SECONDS", "60"))
DHBW_LIBRARY_HOURS_SOURCE_URL = os.getenv(
    "SITCHECK_DHBW_LIBRARY_HOURS_SOURCE_URL",
    "https://www.mannheim.dhbw.de/service/bibliothek",
).strip()
LIBRARY_TIMEZONE = ZoneInfo(os.getenv("SITCHECK_LIBRARY_TIMEZONE", "Europe/Berlin"))
LIBRARY_REGULAR_HOURS = {
    0: ("10:00", "22:00"),
    1: ("10:00", "22:00"),
    2: ("10:00", "22:00"),
    3: ("10:00", "22:00"),
    4: ("10:00", "22:00"),
    5: ("10:00", "18:00"),
}
LIBRARY_WEEKDAY_LABELS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

app = Flask(__name__)
_dhbw_mannheim_context_cache: dict[str, Any] = {"expires_at": 0.0, "payload": None}


def _command_center_url(*, history_minutes: int | None = None) -> str:
    selected_history = CC_HISTORY_MINUTES if history_minutes is None else history_minutes
    query = urlparse.urlencode(
        {
            "zone_id": DEFAULT_ZONE_ID,
            "horizon": CC_HORIZON,
            "history_minutes": selected_history,
        }
    )
    return f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1/dashboard/command-center?{query}"


def _counts_url(*, history_minutes: int) -> str:
    now = datetime.now(UTC)
    start = now - timedelta(minutes=max(history_minutes, 1))
    query = urlparse.urlencode(
        {
            "zone_id": DEFAULT_ZONE_ID,
            "from": start.isoformat(),
            "to": now.isoformat(),
            "granularity": "1m",
        }
    )
    return f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1/counts?{query}"


def _forecast_latest_url(*, horizon: int | None = None) -> str:
    selected_horizon = CC_HORIZON if horizon is None else horizon
    query = urlparse.urlencode(
        {
            "zone_id": DEFAULT_ZONE_ID,
            "horizon": selected_horizon,
        }
    )
    return f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1/forecast/latest?{query}"


def _fetch_single_horizon(horizon: int, timeout_seconds: float) -> dict[str, Any] | None:
    url = _forecast_latest_url(horizon=horizon)
    try:
        return _json_get(url, timeout_seconds=timeout_seconds)
    except Exception:  # noqa: BLE001
        return None


def _fetch_multi_horizon_forecast(
    *,
    step_minutes: int = CC_FORECAST_STEP_MINUTES,
    max_minutes: int = CC_FORECAST_HORIZON_MINUTES,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    BASE_STEP = 15
    base_horizons = list(range(BASE_STEP, max_minutes + 1, BASE_STEP))  # [15,30,...,180]
    now_cutoff = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()

    def _fetch_terminal_point(h: int) -> dict[str, Any] | None:
        payload = _fetch_single_horizon(h, timeout_seconds)
        if not isinstance(payload, dict):
            return None
        raw_points = payload.get("points", [])
        if not isinstance(raw_points, list) or not raw_points:
            return None
        # Terminal value: last point = T+H for both GBDT (1 point) and baseline (1-min steps)
        point = raw_points[-1]
        if not isinstance(point, dict):
            return None
        timestamp = point.get("timestamp")
        yhat = point.get("yhat")
        if timestamp is None or yhat is None:
            return None
        if str(timestamp) < now_cutoff:
            return None
        try:
            yhat_f = float(yhat)
            low_f = float(point.get("pi_low", yhat))
            high_f = float(point.get("pi_high", yhat))
        except Exception:  # noqa: BLE001
            return None
        return {
            "timestamp": str(timestamp),
            "yhat": round(yhat_f, 2),
            "pi_low": round(min(low_f, high_f), 2),
            "pi_high": round(max(low_f, high_f), 2),
        }

    with ThreadPoolExecutor(max_workers=min(len(base_horizons), 8)) as pool:
        futures = {pool.submit(_fetch_terminal_point, h): h for h in base_horizons}
        point_map: dict[int, dict] = {}
        for future in as_completed(futures):
            h = futures[future]
            result = future.result()
            if result is not None:
                point_map[h] = result

    # Assemble base series in horizon order
    base_points = [point_map[h] for h in base_horizons if h in point_map]

    # Stride-based downsampling: pick every Nth point (preserves horizon-specific predictions)
    # 15-min view: stride=1 → all 12 points
    # 30-min view: stride=2 → base_points[1::2] = [T+30, T+60, ..., T+180] (6 points)
    # 60-min view: stride=4 → base_points[3::4] = [T+60, T+120, T+180] (3 points)
    if step_minutes <= BASE_STEP:
        points = base_points
    else:
        stride = max(1, step_minutes // BASE_STEP)
        points = base_points[stride - 1 :: stride]

    current_yhat = points[0]["yhat"] if points else None
    peak_yhat = max((p["yhat"] for p in points), default=None)

    return {
        "horizon": max_minutes,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": f"multi-horizon forecast, {len(points)} points at {step_minutes}min intervals",
        "model_version": None,
        "source": "portal-multi-horizon",
        "age_seconds": 0,
        "stale": len(points) == 0,
        "current_yhat": current_yhat,
        "peak_yhat": peak_yhat,
        "points": points,
        "long_term": [],
    }


def _target_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path.lstrip("/")
    if suffix:
        url = f"{base}/{suffix}"
    else:
        url = f"{base}/"
    query_string = request.query_string.decode("utf-8").strip()
    if query_string:
        url = f"{url}?{query_string}"
    return url


def _forward_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in HOP_BY_HOP_HEADERS:
            continue
        if lowered in {"host", "content-length"}:
            continue
        headers[key] = value
    return headers


def _copy_response_headers(target: Response, source_headers: Any) -> None:
    for key, value in source_headers.items():
        lowered = key.lower()
        if lowered in HOP_BY_HOP_HEADERS:
            continue
        if lowered == "content-length":
            continue
        target.headers[key] = value


def _proxy(base_url: str, upstream_path: str, *, timeout_seconds: float | None = None) -> Response:
    url = _target_url(base_url=base_url, path=upstream_path)
    payload = None
    if request.method not in {"GET", "HEAD"}:
        payload = request.get_data()

    req = urlrequest.Request(url=url, data=payload, method=request.method)
    for key, value in _forward_headers().items():
        req.add_header(key, value)

    timeout = PROXY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    try:
        with urlrequest.urlopen(req, timeout=timeout) as upstream:
            body = upstream.read()
            resp = Response(body, status=upstream.status)
            _copy_response_headers(resp, upstream.headers)
            return resp
    except urlerror.HTTPError as exc:
        body = exc.read()
        resp = Response(body, status=exc.code)
        _copy_response_headers(resp, exc.headers)
        return resp
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "upstream_unavailable", "detail": str(exc)}), 502


def _resolve_static_file(path_value: str, *, allow_index_fallback: bool = False) -> Path | None:
    cleaned = path_value.strip("/")
    candidates: list[str]

    if cleaned == "":
        candidates = ["index.html"]
    elif "." in Path(cleaned).name:
        candidates = [cleaned]
    else:
        candidates = [f"{cleaned}.html", f"{cleaned}/index.html"]
        if allow_index_fallback:
            candidates.append("index.html")

    for candidate in candidates:
        candidate_path = (STATIC_ROOT / candidate).resolve()
        if not str(candidate_path).startswith(str(STATIC_ROOT)):
            continue
        if candidate_path.is_file():
            return candidate_path
    return None


def _json_get(url: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
    timeout = PROXY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    req = urlrequest.Request(url=url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("Unerwartete Antwortstruktur")
        return parsed


def _parse_external_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _count_phrase(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _parse_hhmm(value: str) -> tuple[int, int]:
    hours_text, minutes_text = value.split(":", 1)
    return int(hours_text), int(minutes_text)


def _combine_local_date_with_hhmm(day_value, hhmm: str) -> datetime:
    hours, minutes = _parse_hhmm(hhmm)
    return datetime(day_value.year, day_value.month, day_value.day, hours, minutes, tzinfo=LIBRARY_TIMEZONE)


def _format_next_open_label(dt_value: datetime | None) -> str | None:
    if dt_value is None:
        return None
    localized = dt_value.astimezone(LIBRARY_TIMEZONE)
    return f"{LIBRARY_WEEKDAY_LABELS[localized.weekday()]} um {localized.strftime('%H:%M')} Uhr"


def _next_library_opening(now_local: datetime) -> datetime | None:
    for day_offset in range(0, 8):
        day_value = now_local.date() + timedelta(days=day_offset)
        hours = LIBRARY_REGULAR_HOURS.get(day_value.weekday())
        if not hours:
            continue

        opens_at = _combine_local_date_with_hhmm(day_value, hours[0])
        if day_offset == 0 and now_local >= opens_at:
            continue
        return opens_at
    return None


def _build_library_hours_context(*, horizon_minutes: int) -> dict[str, Any]:
    now_local = datetime.now(UTC).astimezone(LIBRARY_TIMEZONE)
    today_hours = LIBRARY_REGULAR_HOURS.get(now_local.weekday())
    opens_at = _combine_local_date_with_hhmm(now_local.date(), today_hours[0]) if today_hours else None
    closes_at = _combine_local_date_with_hhmm(now_local.date(), today_hours[1]) if today_hours else None
    is_open_now = bool(opens_at and closes_at and opens_at <= now_local < closes_at)
    next_open_at = None
    minutes_until_open = None
    minutes_until_close = None
    statement = ""

    if is_open_now and closes_at is not None:
        minutes_until_close = max(int((closes_at - now_local).total_seconds() // 60), 0)
        if minutes_until_close <= horizon_minutes:
            statement = (
                f"Laut den Öffnungszeiten der DHBW Mannheim schließt die Bibliothek heute um {closes_at.strftime('%H:%M')} Uhr, "
                f"also in etwa {minutes_until_close} Minuten. Damit ist bis zum Ende des Prognosehorizonts eher mit begrenzter oder sinkender Belegung zu rechnen als mit einem späten Anstieg."
            )
        else:
            statement = (
                f"Laut den Öffnungszeiten der DHBW Mannheim ist die Bibliothek heute bis {closes_at.strftime('%H:%M')} Uhr geöffnet. "
                f"Die Schließung liegt damit noch außerhalb des {horizon_minutes}-Minuten-Horizonts, bleibt aber als spätere Begrenzung relevant."
            )
    else:
        next_open_at = _next_library_opening(now_local)
        minutes_until_open = (
            max(int((next_open_at - now_local).total_seconds() // 60), 0)
            if next_open_at is not None
            else None
        )
        statement = (
            "Laut den Öffnungszeiten der DHBW Mannheim ist die Bibliothek aktuell geschlossen. "
            f"{f'Sie öffnet wieder {_format_next_open_label(next_open_at)}. ' if next_open_at is not None else ''}"
            "Für den kurzfristigen Ausblick ist das ein klarer begrenzender Faktor."
        )

    return {
        "source_url": DHBW_LIBRARY_HOURS_SOURCE_URL,
        "timezone": "Europe/Berlin",
        "is_open_now": is_open_now,
        "today_open": today_hours[0] if today_hours else None,
        "today_close": today_hours[1] if today_hours else None,
        "minutes_until_close": minutes_until_close,
        "minutes_until_open": minutes_until_open,
        "closes_within_horizon": bool(is_open_now and minutes_until_close is not None and minutes_until_close <= horizon_minutes),
        "next_open_at": next_open_at.astimezone(UTC).isoformat() if next_open_at is not None else None,
        "next_open_label": _format_next_open_label(next_open_at),
        "statement": statement,
    }


def _build_dhbw_mannheim_statement(*, active: int, starts: int, ends: int) -> str:
    changes: list[str] = []
    if starts > 0:
        changes.append(f"{_count_phrase(starts, 'Start', 'Starts')}")
    if ends > 0:
        changes.append(f"{_count_phrase(ends, 'Ende', 'Enden')}")
    changes_text = " und ".join(changes)

    if active > 0 and changes_text:
        return (
            f"Laut dhbw.app gibt es am Standort Mannheim aktuell {_count_phrase(active, 'laufende Präsenzveranstaltung', 'laufende Präsenzveranstaltungen')}. "
            f"Für die nächste Stunde zeigt der öffentliche Stundenplan eine normale Wechselphase mit {changes_text}. "
            "Das spricht eher für anhaltende Bewegung auf dem Campus als für einen einzelnen dominanten Sonderimpuls."
        )

    if active > 0:
        return (
            f"Laut dhbw.app laufen am Standort Mannheim aktuell {_count_phrase(active, 'Präsenzveranstaltung', 'Präsenzveranstaltungen')}. "
            "Damit bleibt der Campus spürbar in Bewegung, auch ohne eine einzelne Vorlesung als Haupttreiber zu überbetonen."
        )

    if changes_text:
        return (
            f"Laut dhbw.app zeigt der öffentliche Stundenplan am Standort Mannheim in der nächsten Stunde eine Wechselphase mit {changes_text}. "
            "Das kann zusätzliche Bewegung rund um die Bibliothek auslösen, ohne dass ein einzelner Termin alles dominiert."
        )

    return (
        "Laut dhbw.app ist am Standort Mannheim im öffentlichen Präsenzstundenplan für die nächste Stunde keine auffällige Wechselphase zu sehen. "
        "Vom Campus kommt aktuell also eher ein neutraler Zusatzimpuls."
    )


def _fetch_dhbw_mannheim_context() -> dict[str, Any]:
    req = urlrequest.Request(
        url=DHBW_MANNHEIM_RAPLA_URL,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Sitcheck Portal)",
        },
    )
    with urlrequest.urlopen(req, timeout=DHBW_MANNHEIM_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise ValueError("Unexpected DHBW Mannheim lecture payload")

    now = datetime.now(UTC)
    next_hour = now + timedelta(minutes=60)
    onsite_events: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("site") or "").strip().upper() != "MA":
            continue
        if str(item.get("type") or "").strip().upper() != "PRESENCE":
            continue

        start = _parse_external_datetime(item.get("startTime"))
        end = _parse_external_datetime(item.get("endTime"))
        if start is None or end is None or end <= start:
            continue

        rooms = [
            str(room).strip()
            for room in (item.get("rooms") or [])
            if isinstance(room, str) and room.strip()
        ]
        onsite_events.append(
            {
                "start": start,
                "end": end,
                "course": str(item.get("course") or "").strip(),
                "name": str(item.get("name") or "").strip(),
                "rooms": rooms,
            }
        )

    active_events = [event for event in onsite_events if event["start"] <= now < event["end"]]
    starting_events = [event for event in onsite_events if now <= event["start"] <= next_hour]
    ending_events = [event for event in onsite_events if now <= event["end"] <= next_hour]
    active_courses = sorted({event["course"] for event in active_events if event["course"]})
    active_rooms = sorted({room for event in active_events for room in event["rooms"]})

    return {
        "site_code": "MA",
        "site_name": "Mannheim",
        "source": "dhbw_app_rapla",
        "source_url": DHBW_MANNHEIM_RAPLA_URL,
        "generated_at": now.isoformat(),
        "active_onsite_lectures": len(active_events),
        "active_courses": len(active_courses),
        "starting_next_60m": len(starting_events),
        "ending_next_60m": len(ending_events),
        "sample_active_courses": active_courses[:4],
        "sample_active_rooms": active_rooms[:6],
        "statement": _build_dhbw_mannheim_statement(
            active=len(active_events),
            starts=len(starting_events),
            ends=len(ending_events),
        ),
        "library_hours": _build_library_hours_context(horizon_minutes=CC_HORIZON),
    }


def _get_dhbw_mannheim_context() -> dict[str, Any]:
    now_monotonic = time.monotonic()
    cached_payload = _dhbw_mannheim_context_cache.get("payload")
    if cached_payload and float(_dhbw_mannheim_context_cache.get("expires_at", 0.0)) > now_monotonic:
        return cached_payload

    payload = _fetch_dhbw_mannheim_context()
    _dhbw_mannheim_context_cache["payload"] = payload
    _dhbw_mannheim_context_cache["expires_at"] = now_monotonic + max(DHBW_MANNHEIM_CACHE_TTL_SECONDS, 1.0)
    return payload


def _analytics_url_from_request() -> str:
    if ANALYTICS_REDIRECT_URL:
        return ANALYTICS_REDIRECT_URL
    host = request.host.split(":")[0]
    scheme = request.scheme or "http"
    return f"{scheme}://{host}:8501"


def _not_found_response(path_value: str, *, force_json: bool = False) -> Response:
    accepts = request.headers.get("Accept", "")
    wants_json = force_json or request.path.startswith("/api/") or "application/json" in accepts
    if wants_json:
        return jsonify({"error": "not_found", "path": path_value}), 404

    static_404 = _resolve_static_file("404.html")
    if static_404 is not None:
        resp = send_file(static_404)
        resp.status_code = 404
        return resp
    return jsonify({"error": "not_found", "path": path_value}), 404


def _normalize_occupancy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    history_payload = payload.get("history", {})
    history_points = history_payload.get("points", [])
    if not isinstance(history_points, list):
        history_points = []

    normalized_history: list[dict[str, Any]] = []
    occupancy_values: list[int] = []
    for point in history_points[-20:]:
        if not isinstance(point, dict):
            continue
        timestamp = point.get("timestamp")
        occupancy = point.get("occupancy")
        if timestamp is None or occupancy is None:
            continue
        try:
            occupancy_int = int(round(float(occupancy)))
        except Exception:  # noqa: BLE001
            continue
        normalized_history.append(
            {
                "persons": occupancy_int,
                "timestamp": str(timestamp),
            }
        )
        occupancy_values.append(occupancy_int)

    live_payload = payload.get("live", {})
    meta_payload = payload.get("meta", {})

    current_raw = live_payload.get("occupancy", 0) if isinstance(live_payload, dict) else 0
    try:
        current_persons = int(round(float(current_raw)))
    except Exception:  # noqa: BLE001
        current_persons = normalized_history[-1]["persons"] if normalized_history else 0

    if occupancy_values:
        average_persons = round(sum(occupancy_values) / len(occupancy_values), 2)
    else:
        average_persons = float(current_persons)

    last_updated = None
    if isinstance(live_payload, dict):
        last_updated = live_payload.get("timestamp")
    if not last_updated and isinstance(meta_payload, dict):
        last_updated = meta_payload.get("generated_at")
    if not last_updated:
        last_updated = datetime.now(UTC).isoformat()

    return {
        "averagePersons": average_persons,
        "currentPersons": current_persons,
        "lastUpdated": last_updated,
        "history": normalized_history,
    }


def _normalize_occupancy_from_counts_payload(payload: dict[str, Any]) -> dict[str, Any]:
    points_raw = payload.get("points", [])
    if not isinstance(points_raw, list):
        points_raw = []

    normalized_history: list[dict[str, Any]] = []
    occupancy_values: list[int] = []
    for point in points_raw[-20:]:
        if not isinstance(point, dict):
            continue
        timestamp = point.get("timestamp")
        occupancy = point.get("occupancy")
        if timestamp is None or occupancy is None:
            continue
        try:
            occupancy_int = int(round(float(occupancy)))
        except Exception:  # noqa: BLE001
            continue
        normalized_history.append(
            {
                "persons": occupancy_int,
                "timestamp": str(timestamp),
            }
        )
        occupancy_values.append(occupancy_int)

    if normalized_history:
        current_persons = normalized_history[-1]["persons"]
        last_updated = normalized_history[-1]["timestamp"]
    else:
        current_persons = 0
        last_updated = datetime.now(UTC).isoformat()

    if occupancy_values:
        average_persons = round(sum(occupancy_values) / len(occupancy_values), 2)
    else:
        average_persons = float(current_persons)

    return {
        "averagePersons": average_persons,
        "currentPersons": current_persons,
        "lastUpdated": last_updated,
        "history": normalized_history,
    }


def _normalize_forecast_payload(payload: dict[str, Any]) -> dict[str, Any]:
    forecast_payload = payload.get("forecast_latest", {})
    if not isinstance(forecast_payload, dict):
        forecast_payload = {}

    points_raw = forecast_payload.get("points", [])
    if not isinstance(points_raw, list):
        points_raw = []

    points: list[dict[str, Any]] = []
    for point in points_raw:
        if not isinstance(point, dict):
            continue
        timestamp = point.get("timestamp")
        yhat = point.get("yhat")
        pi_low = point.get("pi_low")
        pi_high = point.get("pi_high")
        if timestamp is None or yhat is None:
            continue
        try:
            yhat_f = float(yhat)
            low_f = float(pi_low if pi_low is not None else yhat)
            high_f = float(pi_high if pi_high is not None else yhat)
        except Exception:  # noqa: BLE001
            continue
        points.append(
            {
                "timestamp": str(timestamp),
                "yhat": round(yhat_f, 2),
                "pi_low": round(min(low_f, high_f), 2),
                "pi_high": round(max(low_f, high_f), 2),
            }
        )

    current_forecast = points[0]["yhat"] if points else None
    peak_forecast = max((point["yhat"] for point in points), default=None)

    long_term_raw = payload.get("forecast_long_term", [])
    long_term: list[dict[str, Any]] = []
    if isinstance(long_term_raw, list):
        for item in long_term_raw:
            if not isinstance(item, dict):
                continue
            horizon = item.get("horizon")
            if horizon is None:
                continue
            first_point = None
            points_list = item.get("points", [])
            if isinstance(points_list, list) and points_list:
                first = points_list[0]
                if isinstance(first, dict):
                    first_point = {
                        "timestamp": first.get("timestamp"),
                        "yhat": first.get("yhat"),
                        "pi_low": first.get("pi_low"),
                        "pi_high": first.get("pi_high"),
                    }
            long_term.append(
                {
                    "horizon": int(horizon),
                    "model_version": item.get("model_version"),
                    "generated_at": item.get("generated_at"),
                    "first_point": first_point,
                }
            )
    long_term.sort(key=lambda entry: entry.get("horizon", 0))

    return {
        "horizon": forecast_payload.get("horizon"),
        "generated_at": forecast_payload.get("generated_at"),
        "summary": forecast_payload.get("summary"),
        "model_version": forecast_payload.get("model_version"),
        "source": forecast_payload.get("source"),
        "age_seconds": forecast_payload.get("age_seconds"),
        "stale": bool(forecast_payload.get("stale", False)),
        "current_yhat": current_forecast,
        "peak_yhat": peak_forecast,
        "points": points,
        "long_term": long_term,
    }


def _normalize_forecast_latest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _normalize_forecast_payload({"forecast_latest": payload, "forecast_long_term": []})


def _probe_json_endpoint(
    name: str,
    url: str,
    *,
    optional: bool = False,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.perf_counter()
    try:
        payload = _json_get(url, timeout_seconds=timeout_seconds)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return (
            {
                "name": name,
                "url": url,
                "status": "ok",
                "latency_ms": latency_ms,
            },
            payload,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        status = "degraded" if optional else "down"
        return (
            {
                "name": name,
                "url": url,
                "status": status,
                "latency_ms": latency_ms,
                "error": str(exc),
            },
            None,
        )


def _probe_endpoint(name: str, url: str, *, optional: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    req = urlrequest.Request(url=url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=PROXY_TIMEOUT_SECONDS) as response:
            response.read()
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "name": name,
                "url": url,
                "status": "ok",
                "latency_ms": latency_ms,
                "http_status": response.status,
            }
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        status = "degraded" if optional else "down"
        return {
            "name": name,
            "url": url,
            "status": status,
            "latency_ms": latency_ms,
            "error": str(exc),
        }


@app.get("/health")
def health() -> Response:
    return jsonify(
        {
            "ok": True,
            "service": "portal",
            "static_root": str(STATIC_ROOT),
        }
    )


@app.get("/api/test")
def api_test() -> Response:
    return jsonify({"time": datetime.now(UTC).isoformat()})


@app.get("/analytics")
def analytics_redirect() -> Response:
    return redirect(_analytics_url_from_request(), code=302)


@app.get("/api/occupancy")
def occupancy_compat() -> Response:
    try:
        counts_payload = _json_get(
            _counts_url(history_minutes=CC_HISTORY_MINUTES),
            timeout_seconds=min(CC_TIMEOUT_SECONDS, 5.0),
        )
        normalized_counts = _normalize_occupancy_from_counts_payload(counts_payload)
        if normalized_counts.get("history"):
            return jsonify(normalized_counts)
    except Exception:  # noqa: BLE001
        pass

    try:
        payload = _json_get(_command_center_url(), timeout_seconds=min(CC_TIMEOUT_SECONDS, 8.0))
        normalized = _normalize_occupancy_payload(payload)
        if normalized.get("history"):
            return jsonify(normalized)
    except Exception:  # noqa: BLE001
        pass

    return jsonify({"error": "Noch keine Auslastungsdaten verfügbar."}), 503


NATIVE_GRANULARITIES = {"1m", "5m", "15m", "60m"}
AGGREGATE_GRANULARITY_MAP = {
    "30m": ("15m", 2),
}


@app.route("/api/v1/counts", methods=["GET"])
def counts_with_aggregation() -> Response:
    granularity = request.args.get("granularity", "1m")
    if granularity in NATIVE_GRANULARITIES:
        return _proxy(
            base_url=f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1",
            upstream_path="counts",
            timeout_seconds=PROXY_TIMEOUT_SECONDS,
        )

    agg = AGGREGATE_GRANULARITY_MAP.get(granularity)
    if agg is None:
        return jsonify({"error": f"unsupported granularity: {granularity}"}), 422

    base_gran, merge_count = agg
    zone_id = request.args.get("zone_id", DEFAULT_ZONE_ID)
    from_val = request.args.get("from", "")
    to_val = request.args.get("to", "")
    url = (
        f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1/counts?"
        + urlparse.urlencode({"zone_id": zone_id, "from": from_val, "to": to_val, "granularity": base_gran})
    )
    try:
        payload = _json_get(url, timeout_seconds=PROXY_TIMEOUT_SECONDS)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list):
        raw_points = []

    aggregated: list[dict[str, Any]] = []
    for i in range(0, len(raw_points), merge_count):
        chunk = raw_points[i : i + merge_count]
        if not chunk:
            continue
        occ_values = []
        for p in chunk:
            try:
                occ_values.append(float(p.get("occupancy", 0)))
            except Exception:
                pass
        avg_occ = round(sum(occ_values) / len(occ_values), 2) if occ_values else 0
        base_point = dict(chunk[-1])
        base_point["occupancy"] = avg_occ
        aggregated.append(base_point)

    payload["points"] = aggregated
    return jsonify(payload)


@app.route("/api/v1/forecast/multi-horizon", methods=["GET"])
def forecast_multi_horizon() -> Response:
    step = int(request.args.get("step_minutes", CC_FORECAST_STEP_MINUTES))
    horizon = int(request.args.get("horizon_minutes", CC_FORECAST_HORIZON_MINUTES))
    step = max(15, min(120, step))
    horizon = max(step, min(360, horizon))
    result = _fetch_multi_horizon_forecast(
        step_minutes=step,
        max_minutes=horizon,
        timeout_seconds=min(CC_TIMEOUT_SECONDS, 10.0),
    )
    return jsonify(result)


@app.route(
    "/realtime",
    defaults={"upstream_path": ""},
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    strict_slashes=False,
)
@app.route(
    "/realtime/<path:upstream_path>",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    strict_slashes=False,
)
def proxy_realtime(upstream_path: str) -> Response:
    return _proxy(base_url=REALTIME_BASE_URL, upstream_path=upstream_path)


@app.route(
    "/api/v1",
    defaults={"upstream_path": ""},
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
@app.route(
    "/api/v1/<path:upstream_path>",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
def proxy_api_v1(upstream_path: str) -> Response:
    timeout_seconds = (
        EXPLAIN_NARRATIVE_TIMEOUT_SECONDS
        if upstream_path.strip("/") == "explain/narrative"
        else PROXY_TIMEOUT_SECONDS
    )
    return _proxy(
        base_url=f"{PROGNOSE_API_BASE_URL.rstrip('/')}/api/v1",
        upstream_path=upstream_path,
        timeout_seconds=timeout_seconds,
    )


def _build_hub_overview_payload() -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()

    portal_health = {
        "name": "portal",
        "url": "/health",
        "status": "ok",
        "latency_ms": 0.0,
    }
    realtime_health_url = f"{REALTIME_BASE_URL.rstrip('/')}/health"
    realtime_state_url = f"{REALTIME_BASE_URL.rstrip('/')}/api/state"
    api_health_url = f"{PROGNOSE_API_BASE_URL.rstrip('/')}/health"
    analytics_health_url = f"{_analytics_url_from_request().rstrip('/')}/_stcore/health"

    realtime_health = _probe_endpoint("realtime", realtime_health_url)
    realtime_state_health, realtime_state_payload = _probe_json_endpoint(
        "realtime-state", realtime_state_url, optional=True
    )
    api_health = _probe_endpoint("api-gateway", api_health_url)
    analytics_health = _probe_endpoint("analytics", analytics_health_url, optional=True)
    counts_health, counts_payload = _probe_json_endpoint(
        "counts",
        _counts_url(history_minutes=CC_HUB_HISTORY_MINUTES),
        optional=True,
        timeout_seconds=min(CC_TIMEOUT_SECONDS, 8.0),
    )
    forecast_latest_health, forecast_latest_payload = _probe_json_endpoint(
        "forecast-latest",
        _forecast_latest_url(horizon=CC_HORIZON),
        optional=True,
        timeout_seconds=min(CC_TIMEOUT_SECONDS, 8.0),
    )
    multi_horizon_forecast = _fetch_multi_horizon_forecast(
        timeout_seconds=min(CC_TIMEOUT_SECONDS, 8.0),
    )
    command_center_url = _command_center_url(history_minutes=CC_HUB_HISTORY_MINUTES)
    if HUB_PROBE_COMMAND_CENTER:
        command_center_health, command_center_payload = _probe_json_endpoint(
            "command-center",
            command_center_url,
            optional=True,
            timeout_seconds=min(CC_TIMEOUT_SECONDS, CC_HUB_COMMAND_CENTER_TIMEOUT_SECONDS),
        )
    else:
        command_center_health = {
            "name": "command-center",
            "url": command_center_url,
            "status": "ok",
            "latency_ms": 0.0,
            "detail": "not_probed",
        }
        command_center_payload = None

    occupancy_payload: dict[str, Any] | None = None
    forecast_payload: dict[str, Any] | None = None
    if command_center_payload is not None:
        occupancy_payload = _normalize_occupancy_payload(command_center_payload)
        forecast_payload = _normalize_forecast_payload(command_center_payload)

    services = {
        "portal": portal_health,
        "realtime": realtime_health,
        "realtime_state": realtime_state_health,
        "api_gateway": api_health,
        "analytics": analytics_health,
        "counts": counts_health,
        "forecast_latest": forecast_latest_health,
        "command_center": command_center_health,
    }

    errors = [
        {
            "service": entry["name"],
            "status": entry["status"],
            "error": entry.get("error", "unknown"),
        }
        for entry in services.values()
        if entry.get("status") != "ok"
    ]
    critical_service_names = {"portal", "realtime", "api-gateway", "counts", "forecast-latest"}
    degraded = any(
        entry.get("status") != "ok" and entry.get("name") in critical_service_names
        for entry in services.values()
    )

    realtime_state = {}
    if isinstance(realtime_state_payload, dict):
        realtime_state = {
            "occupancy": realtime_state_payload.get("occupancy"),
            "tracks": realtime_state_payload.get("tracks"),
            "fps": realtime_state_payload.get("fps"),
            "inference_fps": realtime_state_payload.get("inference_fps"),
            "entries_total": realtime_state_payload.get("entries_total"),
            "exits_total": realtime_state_payload.get("exits_total"),
            "dash_enabled": realtime_state_payload.get("dash_enabled"),
            "track_ok": realtime_state_payload.get("track_ok"),
            "track_error": realtime_state_payload.get("track_error"),
        }

    if occupancy_payload is None and counts_payload is not None:
        occupancy_payload = _normalize_occupancy_from_counts_payload(counts_payload)
    if forecast_payload is None and isinstance(forecast_latest_payload, dict):
        forecast_payload = _normalize_forecast_latest_payload(forecast_latest_payload)
    if multi_horizon_forecast.get("points"):
        forecast_payload = multi_horizon_forecast

    realtime_occupancy = None
    if isinstance(realtime_state_payload, dict):
        realtime_occupancy = realtime_state_payload.get("occupancy")

    if occupancy_payload is None:
        fallback_current = 0
        if realtime_occupancy is not None:
            try:
                fallback_current = int(round(float(realtime_occupancy)))
            except Exception:  # noqa: BLE001
                fallback_current = 0
        occupancy_payload = {
            "averagePersons": float(fallback_current),
            "currentPersons": fallback_current,
            "lastUpdated": now_iso,
            "history": (
                [{"persons": fallback_current, "timestamp": now_iso}]
                if realtime_occupancy is not None
                else []
            ),
        }
    elif not occupancy_payload.get("history") and realtime_occupancy is not None:
        try:
            fallback_current = int(round(float(realtime_occupancy)))
            occupancy_payload["currentPersons"] = fallback_current
            if occupancy_payload.get("averagePersons") is None:
                occupancy_payload["averagePersons"] = float(fallback_current)
            occupancy_payload["lastUpdated"] = now_iso
            occupancy_payload["history"] = [{"persons": fallback_current, "timestamp": now_iso}]
        except Exception:  # noqa: BLE001
            pass
    if forecast_payload is None:
        forecast_payload = {
            "horizon": None,
            "generated_at": None,
            "summary": None,
            "model_version": None,
            "source": None,
            "age_seconds": None,
            "stale": True,
            "current_yhat": None,
            "peak_yhat": None,
            "points": [],
            "long_term": [],
        }

    return {
        "status": "degraded" if degraded else "ok",
        "generated_at": now_iso,
        "zone_id": DEFAULT_ZONE_ID,
        "services": services,
        "occupancy": occupancy_payload,
        "forecast": forecast_payload,
        "realtime_state": realtime_state,
        "errors": errors,
    }


@app.get("/api/hub/overview")
def hub_overview() -> Response:
    return jsonify(_build_hub_overview_payload())


@app.get("/api/hub/dhbw-mannheim-context")
def dhbw_mannheim_context() -> Response:
    try:
        return jsonify(_get_dhbw_mannheim_context())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "dhbw_mannheim_context_unavailable", "detail": str(exc)}), 502


@app.route("/", defaults={"page_path": ""}, methods=["GET"])
@app.route("/<path:page_path>", methods=["GET"])
def serve_static(page_path: str) -> Response:
    cleaned = page_path.strip("/")

    if cleaned.startswith(("api/", "realtime/", "dash/")):
        return _not_found_response(cleaned, force_json=True)

    static_file = _resolve_static_file(cleaned, allow_index_fallback=False)
    if static_file is None:
        return _not_found_response(cleaned, force_json=False)
    return send_file(static_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sitcheck Portal Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
