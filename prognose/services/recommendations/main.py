from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    zone_id: str
    horizon: int = Field(ge=1, le=720)
    capacity: int = Field(ge=1)
    forecast: dict[str, Any]
    explanation: dict[str, Any]


class ScenarioRequest(BaseModel):
    zone_id: str
    horizon: int = Field(ge=1, le=720)
    persist: bool = False
    changes: dict[str, Any]
    capacity: int = Field(ge=1)
    forecast: dict[str, Any]
    explanation: dict[str, Any] | None = None


app = FastAPI(title="sitcheck-recommendations", version="0.1.0")


def _extract_peak(forecast: dict[str, Any], capacity: int) -> tuple[float, float]:
    points = forecast.get("points", [])
    if not points:
        return 0.0, 0.0

    peak_occupancy = max(float(p.get("yhat", 0.0)) for p in points)
    peak_utilization = peak_occupancy / max(capacity, 1)
    return peak_occupancy, peak_utilization


def _build_evidence(zone_id: str, horizon: int, quality_score: float, quality_flags: list[str]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "evidence_id": f"rec-{uuid.uuid4()}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": (now - timedelta(hours=24)).isoformat(),
            "to": (now + timedelta(minutes=horizon)).isoformat(),
        },
        "sources": [
            {"type": "forecast", "id": f"zone:{zone_id}"},
            {"type": "xai", "id": f"zone:{zone_id}"},
        ],
        "model": {"name": "recommendation-rules", "version": "v1"},
        "quality": {"score": quality_score, "flags": quality_flags},
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "recommendations"}


@app.post("/v1/recommendations")
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    peak_occupancy, peak_utilization = _extract_peak(payload.forecast, payload.capacity)

    uncertainty_score = float(payload.explanation.get("uncertainty", {}).get("score", 0.8))
    xai_quality = float(payload.explanation.get("evidence", {}).get("quality", {}).get("score", 0.5))

    quality_ok = xai_quality >= 0.55
    uncertainty_ok = uncertainty_score <= 0.7
    gates = {
        "quality_ok": quality_ok,
        "uncertainty_ok": uncertainty_ok,
        "notes": []
    }

    if not quality_ok:
        gates["notes"].append("Quality gate blocked recommendations")
    if not uncertainty_ok:
        gates["notes"].append("Uncertainty gate blocked recommendations")

    actions: list[dict[str, Any]] = []

    if quality_ok and uncertainty_ok:
        if peak_utilization >= 0.85:
            actions.append(
                {
                    "action_type": "open_room",
                    "priority": 1,
                    "rationale": "Forecast peak utilization exceeds 85%",
                    "expected_impact": {
                        "delta_occupancy": -0.12 * peak_occupancy,
                        "delta_utilization": -0.12,
                    },
                }
            )
        if peak_utilization >= 0.70:
            actions.append(
                {
                    "action_type": "push_time",
                    "priority": 2,
                    "rationale": "Peak risk within horizon; shift arrivals to flatten peak",
                    "expected_impact": {
                        "delta_occupancy": -0.07 * peak_occupancy,
                        "delta_utilization": -0.07,
                    },
                }
            )
        if not actions:
            actions.append(
                {
                    "action_type": "monitor",
                    "priority": 3,
                    "rationale": "No critical peak expected",
                    "expected_impact": {
                        "delta_occupancy": -0.02 * peak_occupancy,
                        "delta_utilization": -0.02,
                    },
                }
            )

    summary = (
        "Recommendations generated from forecast peak, uncertainty, and data quality."
        if actions
        else "No recommendation emitted because quality/uncertainty gates failed."
    )

    quality_flags = ["OK"] if quality_ok and uncertainty_ok else ["GATED"]

    return {
        "zone_id": payload.zone_id,
        "horizon": payload.horizon,
        "summary": summary,
        "actions": actions,
        "gates": gates,
        "evidence": _build_evidence(payload.zone_id, payload.horizon, xai_quality, quality_flags),
    }


@app.post("/v1/scenarios/simulate")
def simulate(payload: ScenarioRequest) -> dict[str, Any]:
    baseline_peak_occupancy, baseline_peak_utilization = _extract_peak(payload.forecast, payload.capacity)

    capacity = float(payload.capacity)
    counter_peak = baseline_peak_occupancy
    counter_capacity = capacity

    capacity_delta = float(payload.changes.get("capacity_delta", 0) or 0)
    if capacity_delta:
        counter_capacity = max(1.0, counter_capacity + capacity_delta)

    if bool(payload.changes.get("open_room", False)):
        counter_peak *= 0.88

    push_time_minutes = float(payload.changes.get("push_time_minutes", 0) or 0)
    if push_time_minutes:
        counter_peak *= max(0.5, 1.0 - min(30.0, push_time_minutes) * 0.008)

    staff_delta = float(payload.changes.get("staff_delta", 0) or 0)
    if staff_delta:
        counter_peak = max(0.0, counter_peak - staff_delta * 2.0)

    baseline_util = baseline_peak_utilization
    counter_util = counter_peak / max(counter_capacity, 1.0)

    summary = "Counterfactual scenario simulated with deterministic policy deltas."
    quality_score = float(payload.explanation.get("evidence", {}).get("quality", {}).get("score", 0.6)) if payload.explanation else 0.6

    result = {
        "zone_id": payload.zone_id,
        "horizon": payload.horizon,
        "summary": summary,
        "baseline": {
            "peak_occupancy": round(baseline_peak_occupancy, 3),
            "peak_utilization": round(baseline_util, 3),
        },
        "counterfactual": {
            "peak_occupancy": round(counter_peak, 3),
            "peak_utilization": round(counter_util, 3),
        },
        "delta": {
            "peak_occupancy": round(counter_peak - baseline_peak_occupancy, 3),
            "peak_utilization": round(counter_util - baseline_util, 3),
        },
        "evidence": _build_evidence(payload.zone_id, payload.horizon, quality_score, ["SIMULATED"]),
    }

    return result
