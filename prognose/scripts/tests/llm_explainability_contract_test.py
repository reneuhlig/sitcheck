#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "shared" / "schemas"
DASHBOARD_DIR = ROOT / "apps" / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from explainability import build_explainability_context, render_template_fallback  # noqa: E402


def load_schemas() -> dict[str, dict]:
    store: dict[str, dict] = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        store[schema["$id"]] = schema
    return store


def validate(schema_id: str, payload: dict, store: dict[str, dict]) -> None:
    schema = store[schema_id]
    resolver = RefResolver.from_schema(schema, store=store)
    validator = Draft202012Validator(schema, resolver=resolver)
    validator.validate(payload)


def sample_evidence() -> dict:
    return {
        "evidence_id": "ev-1",
        "generated_at": "2026-02-24T10:00:00Z",
        "time_window": {"from": "2026-02-24T09:00:00Z", "to": "2026-02-24T10:00:00Z"},
        "sources": [{"type": "counts", "id": "window-1"}],
        "model": {"name": "baseline", "version": "v1", "backend": "baseline"},
        "quality": {"score": 0.9, "flags": ["OK"]},
    }


def main() -> int:
    store = load_schemas()

    forecast_latest = {
        "zone_id": "default-zone",
        "horizon": 60,
        "generated_at": "2026-02-24T10:00:00Z",
        "age_seconds": 10,
        "stale": False,
        "summary": "Stable trend",
        "model_version": "baseline-v1",
        "points": [
            {"timestamp": "2026-02-24T10:01:00Z", "yhat": 35, "pi_low": 30, "pi_high": 40}
        ],
        "evidence": sample_evidence(),
        "source": "snapshot",
    }
    explanation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "Momentum drives the change",
        "drivers": [
            {"name": "momentum", "impact": 0.4, "direction": "up", "description": "Recent rise"},
            {"name": "event_context", "impact": 0.2, "direction": "up", "description": "Class switch"},
        ],
        "uncertainty": {"score": 0.3, "level": "medium", "reason": "Moderate interval width"},
        "evidence": sample_evidence(),
    }
    recommendation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "Open overflow room if needed",
        "actions": [
            {
                "action_type": "open_room",
                "priority": 1,
                "rationale": "Peak expected",
                "expected_impact": {"delta_occupancy": -10},
            }
        ],
        "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
        "evidence": sample_evidence(),
    }
    history = {
        "zone_id": "default-zone",
        "from": "2026-02-24T09:00:00Z",
        "to": "2026-02-24T10:00:00Z",
        "granularity": "1m",
        "points": [
            {
                "timestamp": "2026-02-24T09:59:00Z",
                "zone_id": "default-zone",
                "occupancy": 33,
                "utilization": 0.33,
                "source": "demo",
                "quality_score": 0.9,
                "quality_flags": ["OK"],
                "evidence": sample_evidence(),
            },
            {
                "timestamp": "2026-02-24T10:00:00Z",
                "zone_id": "default-zone",
                "occupancy": 35,
                "utilization": 0.35,
                "source": "demo",
                "quality_score": 0.9,
                "quality_flags": ["OK"],
                "evidence": sample_evidence(),
            },
        ],
    }

    context = build_explainability_context(
        forecast_latest=forecast_latest,
        explanation=explanation,
        recommendation=recommendation,
        history=history,
        query="Warum steigt die Prognose?",
        audience="ops",
        language="de",
        timezone="UTC",
    )

    validate("https://sitcheck.dev/schemas/llm-explainability-context.schema.json", context, store)

    out1 = render_template_fallback(context)
    out2 = render_template_fallback(context)
    if out1 != out2:
        raise AssertionError("Template fallback is not deterministic")

    validate("https://sitcheck.dev/schemas/llm-explanation-response.schema.json", out1, store)

    structured = out1["structured"]
    if not structured.get("evidence_refs"):
        raise AssertionError("evidence_refs missing")
    for driver in structured.get("top_drivers", []):
        if not driver.get("evidence_ref"):
            raise AssertionError("driver missing evidence_ref")
    for action in structured.get("recommended_actions", []):
        if not action.get("evidence_ref"):
            raise AssertionError("action missing evidence_ref")
    if not structured.get("uncertainty", {}).get("evidence_ref"):
        raise AssertionError("uncertainty missing evidence_ref")

    print("llm explainability contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
