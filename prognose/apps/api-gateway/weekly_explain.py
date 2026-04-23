from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if parsed != parsed:
        return float(default)
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def build_weekly_explainability(
    *,
    zone_id: str,
    weekly_forecast: dict[str, Any],
    lecture_impact: dict[str, Any] | None,
    lineage: dict[str, Any] | None,
    calendar_events: list[dict[str, Any]],
) -> dict[str, Any]:
    points = weekly_forecast.get("points", []) if isinstance(weekly_forecast, dict) else []
    daily_summaries = weekly_forecast.get("daily_summaries", []) if isinstance(weekly_forecast, dict) else []
    evidence = weekly_forecast.get("evidence", {}) if isinstance(weekly_forecast, dict) else {}
    generated_at = weekly_forecast.get("generated_at") or datetime.now(UTC).isoformat()
    slot_minutes = _safe_int(weekly_forecast.get("slot_minutes"), 60)
    days = _safe_int(weekly_forecast.get("days"), 7)

    if not points:
        return {
            "zone_id": zone_id,
            "days": days,
            "slot_minutes": slot_minutes,
            "generated_at": generated_at,
            "summary": "Weekly explainability unavailable because no weekly forecast slots are present.",
            "week_overview": {
                "peak_day": None,
                "peak_slot": None,
                "peak_yhat": 0.0,
                "avg_yhat": 0.0,
                "risk_level": "high",
            },
            "daily_highlights": [],
            "drivers": [
                {
                    "name": "insufficient_weekly_signal",
                    "impact": 0.0,
                    "direction": "mixed",
                    "description": "No weekly slot forecast was available for comparison.",
                }
            ],
            "uncertainty": {"score": 1.0, "level": "high", "reason": "No weekly points available."},
            "quality": {"score": 0.0, "flags": ["NO_WEEKLY_FORECAST"]},
            "references": [],
            "lineage": lineage or {},
            "evidence": evidence,
        }

    peak_point = max(points, key=lambda point: _safe_float(point.get("yhat")))
    avg_yhat = sum(_safe_float(point.get("yhat")) for point in points) / max(1, len(points))
    avg_interval = sum(
        max(0.0, _safe_float(point.get("pi_high")) - _safe_float(point.get("pi_low")))
        for point in points
    ) / max(1, len(points))
    quality_score = _safe_float(evidence.get("quality", {}).get("score"), 0.0)
    quality_flags = list(evidence.get("quality", {}).get("flags", [])) if isinstance(evidence.get("quality", {}).get("flags"), list) else []
    if not quality_flags:
        quality_flags = ["OK"]

    daily_risk_counter = Counter(str(item.get("risk_level", "low")) for item in daily_summaries)
    if daily_risk_counter.get("high", 0) > 0:
        risk_level = "high"
    elif daily_risk_counter.get("medium", 0) > 0:
        risk_level = "medium"
    else:
        risk_level = "low"

    lecture_net_pull = 0.0
    starts_next_60m = 0
    if isinstance(lecture_impact, dict):
        impact = lecture_impact.get("impact", {}) if isinstance(lecture_impact.get("impact"), dict) else {}
        lecture_net_pull = _safe_float(impact.get("lecture_net_pull"), 0.0)
        starts_next_60m = _safe_int(lecture_impact.get("starts_next_60m"), 0)

    event_density = len(calendar_events)
    drivers = [
        {
            "name": "weekly_pattern",
            "impact": round(_safe_float(peak_point.get("yhat")) - avg_yhat, 4),
            "direction": "up" if _safe_float(peak_point.get("yhat")) >= avg_yhat else "mixed",
            "description": "Difference between weekly peak slot and weekly average.",
        },
        {
            "name": "event_density",
            "impact": float(event_density),
            "direction": "up" if event_density > 0 else "mixed",
            "description": f"{event_density} calendar events overlap with the selected weekly window.",
        },
        {
            "name": "lecture_pull",
            "impact": round(abs(lecture_net_pull), 4),
            "direction": "up" if lecture_net_pull > 0 else "mixed",
            "description": f"Latest lecture net pull is {lecture_net_pull:.2f} with {starts_next_60m} starts in the next 60m.",
        },
        {
            "name": "uncertainty_band",
            "impact": round(avg_interval, 4),
            "direction": "mixed",
            "description": "Average width of weekly prediction intervals across all slots.",
        },
    ]
    drivers = sorted(drivers, key=lambda item: abs(_safe_float(item.get("impact"))), reverse=True)

    uncertainty_score = min(1.0, max(0.0, avg_interval / max(10.0, _safe_float(peak_point.get("yhat"), 1.0) + 1.0)))
    if quality_score < 0.7:
        uncertainty_score = min(1.0, uncertainty_score + 0.15)
    if uncertainty_score >= 0.7:
        uncertainty_level = "high"
    elif uncertainty_score >= 0.4:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "low"

    references = []
    if isinstance(lineage, dict):
        for item in lineage.get("reference_objects", []) or []:
            if isinstance(item, dict):
                references.append(item)
    lecture_metadata = lecture_impact.get("metadata", {}) if isinstance(lecture_impact, dict) else {}
    external_refs = lecture_metadata.get("external_references", []) if isinstance(lecture_metadata, dict) else []
    for idx, item in enumerate(external_refs, start=1):
        if not isinstance(item, dict):
            continue
        references.append(
            {
                "reference_id": f"ext-weekly-{idx}",
                "zone_id": zone_id,
                "reference_type": str(item.get("reference_type") or "external_reference"),
                "source_type": str(item.get("source_type") or "external"),
                "label": str(item.get("label") or "External reference"),
                "uri_or_path": item.get("url"),
                "checksum": None,
                "imported_at": generated_at,
                "time_from": None,
                "time_to": None,
                "row_count": None,
                "metadata": item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                "created_at": generated_at,
            }
        )

    return {
        "zone_id": zone_id,
        "days": days,
        "slot_minutes": slot_minutes,
        "generated_at": generated_at,
        "summary": (
            f"Weekly outlook peaks on {str(peak_point.get('timestamp'))} with "
            f"{_safe_float(peak_point.get('yhat')):.1f} expected occupancy."
        ),
        "week_overview": {
            "peak_day": str(peak_point.get("timestamp"))[:10],
            "peak_slot": peak_point.get("timestamp"),
            "peak_yhat": round(_safe_float(peak_point.get("yhat")), 4),
            "avg_yhat": round(avg_yhat, 4),
            "risk_level": risk_level,
        },
        "daily_highlights": daily_summaries,
        "drivers": drivers,
        "uncertainty": {
            "score": round(uncertainty_score, 4),
            "level": uncertainty_level,
            "reason": "Derived from average weekly interval width and data quality.",
        },
        "quality": {
            "score": round(quality_score, 6),
            "flags": quality_flags,
        },
        "references": references,
        "lineage": lineage or {},
        "evidence": evidence,
    }
