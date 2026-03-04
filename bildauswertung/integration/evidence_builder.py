from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import uuid


def build_evidence(
    *,
    zone_id: str,
    source: str,
    model_name: str,
    model_version: str,
    quality_score: float,
    quality_flags: list[str],
    occupancy: int,
    utilization: float,
    frame_id: int | None = None,
    track_count: int = 0,
    events_in_frame: Dict[str, int] | None = None,
    time_window_seconds: float = 1.0,
    generated_at: datetime | None = None,
) -> Dict[str, Any]:
    now = _normalize_utc(generated_at)
    window_seconds = max(0.05, float(time_window_seconds))
    start = now - timedelta(seconds=window_seconds)

    source_items = []
    if frame_id is not None:
        source_items.append({"type": "frame", "id": f"frame:{int(frame_id)}"})
    source_items.append({"type": "pipeline", "id": str(source)})

    return {
        "evidence_id": f"ev-vision-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": start.isoformat(),
            "to": now.isoformat(),
        },
        "sources": source_items,
        "model": {
            "name": model_name,
            "version": model_version,
        },
        "quality": {
            "score": float(max(0.0, min(1.0, quality_score))),
            "flags": [str(flag).upper() for flag in (quality_flags or ["OK"])],
        },
        "runtime": {
            "zone_id": zone_id,
            "occupancy": int(max(0, occupancy)),
            "utilization": float(max(0.0, utilization)),
            "tracks": int(max(0, track_count)),
            "events_in_frame": {
                "entry": int((events_in_frame or {}).get("entry", 0)),
                "exit": int((events_in_frame or {}).get("exit", 0)),
            },
        },
    }


def _normalize_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

