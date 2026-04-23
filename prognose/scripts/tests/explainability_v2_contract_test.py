#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, RefResolver
except Exception:  # pragma: no cover - optional dependency in local no-docker mode
    Draft202012Validator = None  # type: ignore[assignment]
    RefResolver = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "shared" / "schemas"
API_GATEWAY_DIR = ROOT / "apps" / "api-gateway"
if str(API_GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(API_GATEWAY_DIR))

from explainability.context_builder import build_explainability_context_v2  # noqa: E402
from explainability.narrative_service import render_template_fallback, validate_dual_response_shape  # noqa: E402
from explainability.prompt_registry import PromptRegistry  # noqa: E402


def load_schemas() -> dict[str, dict]:
    store: dict[str, dict] = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        store[schema["$id"]] = schema
    return store


def validate(schema_id: str, payload: dict, store: dict[str, dict]) -> None:
    if Draft202012Validator is None or RefResolver is None:
        if schema_id.endswith("llm-explainability-context-v2.schema.json"):
            required = {
                "request_meta",
                "zone_capacity",
                "utilization_now_pct",
                "occupancy_explainer",
                "improvement_candidates",
                "forecast_snapshot",
                "history_digest",
                "driver_summary",
                "uncertainty",
                "recommendation_digest",
                "lecture_impact_digest",
                "quality_digest",
                "citation_map",
                "policy_block",
            }
            missing = sorted(required - set(payload.keys()))
            if missing:
                raise AssertionError(f"missing required context keys without jsonschema: {missing}")
        elif schema_id.endswith("llm-explanation-response.schema.json"):
            if not isinstance(payload.get("narrative"), dict) or not isinstance(payload.get("structured"), dict):
                raise AssertionError("missing narrative/structured in fallback payload")
        return

    schema = store[schema_id]
    resolver = RefResolver.from_schema(schema, store=store)
    validator = Draft202012Validator(schema, resolver=resolver)
    validator.validate(payload)


def sample_evidence(source_type: str, source_id: str) -> dict:
    return {
        "evidence_id": f"ev-{source_type}",
        "generated_at": "2026-02-26T11:00:00Z",
        "time_window": {"from": "2026-02-26T10:00:00Z", "to": "2026-02-26T11:00:00Z"},
        "sources": [{"type": source_type, "id": source_id}],
        "model": {"name": "baseline", "version": "v1"},
        "quality": {"score": 0.9, "flags": ["OK"]},
    }


def main() -> int:
    store = load_schemas()

    forecast_latest = {
        "zone_id": "default-zone",
        "horizon": 60,
        "generated_at": "2026-02-26T11:00:00Z",
        "age_seconds": 12,
        "stale": False,
        "summary": "stable",
        "model_version": "baseline-v1",
        "source": "snapshot",
        "points": [{"timestamp": "2026-02-26T11:01:00Z", "yhat": 28.0, "pi_low": 24.0, "pi_high": 31.0}],
        "evidence": sample_evidence("forecast", "fc-1"),
    }
    explanation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "momentum-driven",
        "drivers": [{"name": "momentum", "impact": 0.3, "direction": "up", "description": "recent rise"}],
        "uncertainty": {"score": 0.2, "level": "low", "reason": "enough history"},
        "evidence": sample_evidence("xai", "x-1"),
    }
    recommendation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "monitor",
        "actions": [{"action_type": "monitor", "priority": 1, "rationale": "no risk"}],
        "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
        "evidence": sample_evidence("recommendation", "r-1"),
    }
    history_points = [
        {
            "timestamp": "2026-02-26T10:59:00Z",
            "zone_id": "default-zone",
            "occupancy": 25,
            "utilization": 0.25,
            "quality_score": 0.9,
            "quality_flags": ["OK"],
            "evidence": sample_evidence("counts", "hist-1"),
        },
        {
            "timestamp": "2026-02-26T11:00:00Z",
            "zone_id": "default-zone",
            "occupancy": 28,
            "utilization": 0.28,
            "quality_score": 0.9,
            "quality_flags": ["OK"],
            "evidence": sample_evidence("counts", "hist-2"),
        },
    ]

    context = build_explainability_context_v2(
        zone_id="default-zone",
        zone_capacity=100,
        horizon=60,
        audience="professor",
        language="de",
        query="Warum steigt die Prognose?",
        forecast_latest=forecast_latest,
        explanation=explanation,
        recommendation=recommendation,
        history_points=history_points,
        live_state={
            "timestamp": "2026-02-26T11:00:00Z",
            "occupancy": 28,
            "utilization": 0.28,
            "quality_score": 0.9,
            "quality_flags": ["OK"],
            "point_count": 120,
        },
        alerts=[{"code": "ALL_CLEAR", "level": "ok", "message": "none"}],
        lecture_impact={
            "timestamp": "2026-02-26T11:00:00Z",
            "source": "rapla_refresh",
            "quality_score": 0.9,
            "quality_flags": ["LECTURE_RAPLA"],
            "active_lectures": 14,
            "active_courses": 10,
            "starts_next_60m": 20,
            "ends_next_60m": 15,
            "impact": {
                "heavy_active_lectures": 2,
                "heavy_ended_last_60m": 1,
                "lecture_pull_regular": 280,
                "heavy_bib_bonus": 12,
                "lecture_net_pull": 268,
                "impact_model_version": "lecture-impact-v1",
            },
        },
        template_set_id="explainability-de-v2",
        prompt_version="2.2.0",
    )

    validate("https://sitcheck.dev/schemas/llm-explainability-context-v2.schema.json", context, store)

    registry = PromptRegistry(template_set_id="explainability-de-v2")
    prompt1 = registry.build_prompt(context=context, audience="professor", language="de", query="Warum steigt die Prognose?")
    prompt2 = registry.build_prompt(context=context, audience="professor", language="de", query="Warum steigt die Prognose?")
    if prompt1.prompt != prompt2.prompt:
        raise AssertionError("Prompt builder is not deterministic")
    fast_prompt = registry.build_fast_narrative_prompt(
        context=context,
        audience="professor",
        language="de",
        query="Warum steigt die Prognose?",
    ).prompt
    required_terms = ("narrative", "structured", "top_drivers", "recommended_actions", "evidence_refs")
    for term in required_terms:
        if term not in fast_prompt:
            raise AssertionError(f"fast prompt missing required contract term: {term}")
    if "genau 5 Schluesseln" in fast_prompt:
        raise AssertionError("fast prompt still asks for five top-level fields")

    fallback = render_template_fallback(context)
    validate("https://sitcheck.dev/schemas/llm-explanation-response.schema.json", fallback, store)
    ok, reason = validate_dual_response_shape(fallback)
    if not ok:
        raise AssertionError(f"template fallback shape invalid: {reason}")

    print("explainability v2 contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
