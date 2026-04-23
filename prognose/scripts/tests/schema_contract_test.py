#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "packages" / "shared" / "schemas"


def load_schemas() -> dict:
    schemas = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        schemas[schema["$id"]] = schema
    return schemas


def validate_instance(schema_id: str, instance: dict, store: dict):
    schema = store[schema_id]
    resolver = RefResolver.from_schema(schema, store=store)
    validator = Draft202012Validator(schema, resolver=resolver)
    validator.validate(instance)


def sample_evidence() -> dict:
    return {
        "evidence_id": "ev-1",
        "generated_at": "2026-02-18T19:00:00Z",
        "time_window": {"from": "2026-02-18T18:00:00Z", "to": "2026-02-18T19:00:00Z"},
        "sources": [{"type": "counts", "id": "window-1"}],
        "model": {"name": "baseline", "version": "v1"},
        "quality": {"score": 0.88, "flags": ["OK"]},
    }


def main() -> None:
    store = load_schemas()

    zone = {
        "zone_id": "default-zone",
        "name": "Main Room",
        "capacity": 100,
        "is_active": True,
        "metadata": {"campus": "MA"},
    }
    count_point = {
        "timestamp": "2026-02-18T19:00:00Z",
        "zone_id": "default-zone",
        "occupancy": 40,
        "utilization": 0.4,
        "source": "demo-generator",
        "quality_score": 0.9,
        "quality_flags": ["OK"],
        "evidence": sample_evidence(),
    }
    forecast = {
        "zone_id": "default-zone",
        "horizon": 60,
        "generated_at": "2026-02-18T19:00:00Z",
        "summary": "Stable occupancy",
        "model_version": "baseline-v1",
        "points": [{"timestamp": "2026-02-18T19:01:00Z", "yhat": 40.0, "pi_low": 35.0, "pi_high": 45.0}],
        "evidence": sample_evidence(),
        "lineage": {
            "product": "short_term",
            "model_run_id": "run-1",
            "model_backend": "tf_mlp",
            "model_version": "baseline-v1",
            "scientific_status": "training_only",
            "include_lecture_impact": True,
            "feature_set_version": "short_term_v2",
            "reference_objects": [],
        },
    }
    forecast_latest = {
        "zone_id": "default-zone",
        "horizon": 60,
        "generated_at": "2026-02-18T19:00:00Z",
        "age_seconds": 12,
        "stale": False,
        "summary": "Stable occupancy",
        "model_version": "baseline-v1",
        "points": [{"timestamp": "2026-02-18T19:01:00Z", "yhat": 40.0, "pi_low": 35.0, "pi_high": 45.0}],
        "evidence": sample_evidence(),
        "lineage": {
            "product": "short_term",
            "model_run_id": "run-1",
            "model_backend": "tf_mlp",
            "model_version": "baseline-v1",
            "scientific_status": "training_only",
            "include_lecture_impact": True,
            "feature_set_version": "short_term_v2",
            "reference_objects": [],
        },
        "source": "snapshot",
    }
    reference_object = {
        "reference_id": "refobj-1",
        "zone_id": "default-zone",
        "reference_type": "training_data",
        "source_type": "excel",
        "label": "historical.xlsx",
        "uri_or_path": "/data/historical.xlsx",
        "checksum": "abc12345abc12345",
        "imported_at": "2026-02-18T18:00:00Z",
        "time_from": "2025-02-18T00:00:00Z",
        "time_to": "2026-02-18T00:00:00Z",
        "row_count": 1000,
        "metadata": {"sheet": 0},
        "created_at": "2026-02-18T18:01:00Z",
    }
    weekly_forecast = {
        "zone_id": "default-zone",
        "product": "weekly_slot",
        "days": 7,
        "slot_minutes": 60,
        "generated_at": "2026-02-18T19:00:00Z",
        "age_seconds": 12,
        "stale": False,
        "summary": "Weekly slot outlook",
        "model_version": "weekly-slot-v1",
        "points": [
            {
                "timestamp": "2026-02-19T08:00:00Z",
                "yhat": 40.0,
                "pi_low": 34.0,
                "pi_high": 46.0,
                "day_of_week": 3,
                "slot_of_day": 480,
                "event_active": 0.0,
                "event_impact_sum": 0.0,
                "lecture_net_pull": 10.0,
                "quality_score": 0.9,
            }
        ],
        "daily_summaries": [
            {
                "date": "2026-02-19",
                "peak_slot": "2026-02-19T08:00:00Z",
                "peak_yhat": 40.0,
                "avg_yhat": 36.0,
                "risk_level": "medium",
                "data_quality": "good",
            }
        ],
        "evidence": sample_evidence(),
        "lineage": {
            "product": "weekly_slot",
            "model_run_id": "weekly-run-1",
            "model_backend": "deterministic",
            "model_version": "weekly-slot-v1",
            "scientific_status": "deterministic_reference",
            "include_lecture_impact": True,
            "feature_set_version": "weekly-slot-v1",
            "reference_objects": [reference_object],
        },
        "source": "snapshot",
    }
    weekly_explanation = {
        "zone_id": "default-zone",
        "days": 7,
        "slot_minutes": 60,
        "generated_at": "2026-02-18T19:00:00Z",
        "summary": "Weekly peak expected on Thursday morning.",
        "week_overview": {
            "peak_day": "2026-02-19",
            "peak_slot": "2026-02-19T08:00:00Z",
            "peak_yhat": 40.0,
            "avg_yhat": 36.0,
            "risk_level": "medium",
        },
        "daily_highlights": weekly_forecast["daily_summaries"],
        "drivers": [
            {
                "name": "weekly_pattern",
                "impact": 4.0,
                "direction": "up",
                "description": "Peak exceeds weekly average.",
            }
        ],
        "uncertainty": {"score": 0.3, "level": "low", "reason": "Stable weekly interval width."},
        "quality": {"score": 0.9, "flags": ["OK"]},
        "references": [reference_object],
        "lineage": weekly_forecast["lineage"],
        "evidence": sample_evidence(),
    }
    explanation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "Demand is driven by recent momentum.",
        "drivers": [{"name": "momentum", "impact": 0.34, "direction": "up", "description": "Last 15 minutes trend"}],
        "uncertainty": {"score": 0.2, "level": "low", "reason": "Sufficient history"},
        "evidence": sample_evidence(),
    }
    recommendation = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "Open overflow room if peak exceeds threshold.",
        "actions": [{"action_type": "open_room", "priority": 1, "rationale": "Peak expected", "expected_impact": {"delta_occupancy": -10}}],
        "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
        "evidence": sample_evidence(),
    }
    llm_context = {
        "forecast_latest": forecast_latest,
        "explanation": explanation,
        "recommendation": recommendation,
        "history_digest": {
            "last_occupancy": 42.0,
            "last_utilization": 0.42,
            "trend_15m": 1.5,
            "trend_60m": 3.1,
            "quality_score_avg": 0.9,
            "quality_flags_top": ["OK"],
            "point_count": 120,
        },
        "scenario_digest": None,
        "context_meta": {
            "request_id": "ctx-1",
            "generated_at": "2026-02-18T19:00:00Z",
            "audience": "ops",
            "language": "de",
            "timezone": "UTC",
            "guardrails": ["api_tools_only", "no_invented_facts"],
            "query": "Warum steigt die Auslastung?",
        },
        "citation_map": [
            {
                "ref_id": "ref-1",
                "evidence_id": "ev-1",
                "source_type": "counts",
                "source_id": "window-1",
                "time_window": {"from": "2026-02-18T18:00:00Z", "to": "2026-02-18T19:00:00Z"},
                "model_version": "v1",
                "quality_score": 0.9,
                "quality_flags": ["OK"],
            }
        ],
    }
    llm_context_v2 = {
        "request_meta": {
            "request_id": "ctx-v2-1",
            "generated_at": "2026-02-18T19:00:00Z",
            "zone_id": "default-zone",
            "horizon": 60,
            "audience": "professor",
            "language": "de",
            "query": "Warum steigt die Auslastung?",
            "guardrails": ["api_tools_only", "no_invented_facts"],
            "template_set_id": "explainability-de-v2",
            "prompt_version": "2.0.0",
        },
        "zone_capacity": 100,
        "utilization_now_pct": 42.0,
        "occupancy_explainer": {
            "current_occupancy": 42,
            "avg_60m": 39.8,
            "trend_15m": 1.5,
            "forecast_next_60m_peak": 48.0,
            "risk_level": "medium",
        },
        "improvement_candidates": [
            {
                "measure_id": "capacity-buffer-next-slot",
                "measure_text": "Pufferbereich fuer Peak vorbereiten.",
                "expected_effect": "Spitzenlast abfedern.",
                "effort": "low",
                "owner_hint": "Bibliotheksteam",
                "evidence_ref": "ref-1",
            },
            {
                "measure_id": "lecture-transition-communication",
                "measure_text": "Vorlesungswechsel kommunizieren.",
                "expected_effect": "Zufluss verteilen.",
                "effort": "low",
                "owner_hint": "Service Desk",
                "evidence_ref": "ref-1",
            },
            {
                "measure_id": "quality-hardening",
                "measure_text": "Datenqualitaet im Betrieb pruefen.",
                "expected_effect": "Belastbarere Prognosen.",
                "effort": "medium",
                "owner_hint": "Data/IT Betrieb",
                "evidence_ref": "ref-1",
            },
        ],
        "forecast_snapshot": {
            "zone_id": "default-zone",
            "horizon": 60,
            "generated_at": "2026-02-18T19:00:00Z",
            "age_seconds": 12,
            "stale": False,
            "summary": "Stable occupancy",
            "model_version": "baseline-v1",
            "source": "snapshot",
            "next_point": {"timestamp": "2026-02-18T19:01:00Z", "yhat": 40.0, "pi_low": 35.0, "pi_high": 45.0},
            "peak": {"value": 40.0, "timestamp": "2026-02-18T19:01:00Z"},
            "point_count": 1,
            "evidence_ref": "ref-1",
        },
        "history_digest": {
            "last_occupancy": 42.0,
            "last_utilization": 0.42,
            "trend_15m": 1.5,
            "trend_60m": 3.1,
            "quality_score_avg": 0.9,
            "quality_flags_top": ["OK"],
            "point_count": 120,
            "similar_pattern": {"found": False, "reason": "insufficient_history"},
        },
        "driver_summary": {
            "top_drivers": [
                {
                    "name": "momentum",
                    "impact": 0.34,
                    "direction": "up",
                    "description": "Last 15 minutes trend",
                    "evidence_ref": "ref-1",
                }
            ],
            "evidence_ref": "ref-1",
        },
        "uncertainty": {"score": 0.2, "level": "low", "reason": "Sufficient history", "evidence_ref": "ref-1"},
        "recommendation_digest": {
            "summary": "Open overflow room if peak exceeds threshold.",
            "quality_ok": True,
            "uncertainty_ok": True,
            "actions": [
                {
                    "action_type": "open_room",
                    "priority": 1,
                    "rationale": "Peak expected",
                    "evidence_ref": "ref-1",
                }
            ],
            "evidence_ref": "ref-1",
        },
        "lecture_impact_digest": {
            "active_lectures": 14,
            "active_courses": 10,
            "starts_next_60m": 22,
            "ends_next_60m": 17,
            "heavy_active_lectures": 2,
            "heavy_ended_last_60m": 1,
            "lecture_pull_regular": 280.0,
            "heavy_bib_bonus": 12.0,
            "lecture_net_pull": 268.0,
            "impact_model_version": "lecture-impact-v1",
        },
        "quality_digest": {
            "live_quality_score": 0.9,
            "live_quality_flags": ["OK"],
            "history_quality_score_avg": 0.9,
            "live_point_count": 180,
            "alerts": [{"code": "ALL_CLEAR", "level": "ok", "message": "Keine kritischen Warnungen."}],
        },
        "citation_map": [
            {
                "ref_id": "ref-1",
                "evidence_id": "ev-1",
                "source_type": "counts",
                "source_id": "window-1",
                "time_window": {"from": "2026-02-18T18:00:00Z", "to": "2026-02-18T19:00:00Z"},
                "model_version": "v1",
                "quality_score": 0.9,
                "quality_flags": ["OK"],
            }
        ],
        "policy_block": {
            "abstain_rules": ["Nur monitor bei hoher Unsicherheit."],
            "decision_gates": {"quality_ok": True, "uncertainty_ok": True, "forecast_stale": False},
            "output_contract": {
                "narrative_keys": ["one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis"],
                "structured_required": [
                    "audience",
                    "zone_id",
                    "horizon",
                    "verdict",
                    "top_drivers",
                    "uncertainty",
                    "recommended_actions",
                    "evidence_refs",
                    "confidence_statement",
                    "limitations",
                ],
            },
        },
    }
    llm_response = {
        "narrative": {
            "one_liner": "Prognose bleibt stabil.",
            "warum": "Momentum ist positiv [ref:ref-1].",
            "unsicherheit": "Unsicherheit ist niedrig [ref:ref-1].",
            "empfehlung": "Weiter beobachten [ref:ref-1].",
            "evidence_hinweis": "[ref:ref-1]",
        },
        "structured": {
            "audience": "professor",
            "zone_id": "default-zone",
            "horizon": 60,
            "verdict": "monitor",
            "top_drivers": [
                {"name": "momentum", "impact": 0.3, "direction": "up", "evidence_ref": "ref-1"}
            ],
            "uncertainty": {
                "score": 0.2,
                "level": "low",
                "reason": "Sufficient history",
                "evidence_ref": "ref-1",
            },
            "recommended_actions": [
                {
                    "action_type": "monitor",
                    "priority": 1,
                    "rationale": "No severe risk",
                    "evidence_ref": "ref-1",
                }
            ],
            "evidence_refs": ["ref-1"],
            "confidence_statement": "High confidence",
            "limitations": ["None"],
        },
    }
    scenario_input = {
        "zone_id": "default-zone",
        "horizon": 60,
        "persist": False,
        "changes": {"open_room": True},
    }
    scenario_result = {
        "zone_id": "default-zone",
        "horizon": 60,
        "summary": "Opening an overflow room lowers peak load.",
        "baseline": {"peak_occupancy": 90, "peak_utilization": 0.9},
        "counterfactual": {"peak_occupancy": 80, "peak_utilization": 0.8},
        "delta": {"peak_occupancy": -10, "peak_utilization": -0.1},
        "evidence": sample_evidence(),
    }
    command_center = {
        "meta": {
            "generated_at": "2026-02-18T19:00:00Z",
            "zone_id": "default-zone",
            "horizon": 60,
            "history_minutes": 180,
            "long_term_days": 14,
            "stale_seconds": 900,
            "environment": "dev",
        },
        "service_health": [
            {"service": "api-gateway", "status": "ok", "latency_ms": 0, "detail": "local"},
            {"service": "forecast", "status": "ok", "latency_ms": 10, "detail": "ok"},
        ],
        "live": {
            "timestamp": "2026-02-18T19:00:00Z",
            "occupancy": 44,
            "utilization": 0.44,
            "quality_score": 0.9,
            "quality_flags": ["OK"],
            "point_count": 180,
        },
        "history": {
            "zone_id": "default-zone",
            "from": "2026-02-18T16:00:00Z",
            "to": "2026-02-18T19:00:00Z",
            "granularity": "1m",
            "points": [count_point],
        },
        "forecast_latest": forecast_latest,
        "forecast_long_term": [forecast_latest],
        "weekly_forecast": weekly_forecast,
        "weekly_explanation": weekly_explanation,
        "explanation": explanation,
        "recommendations": recommendation,
        "model_lineage": {
            "short_term": forecast_latest["lineage"],
            "weekly_slot": weekly_forecast["lineage"],
        },
        "reference_objects": [reference_object],
        "calendar_events": [
            {
                "event_id": "evt-1",
                "zone_id": "default-zone",
                "title": "Vorlesungswechsel",
                "starts_at": "2026-02-18T20:00:00Z",
                "ends_at": "2026-02-18T21:00:00Z",
                "source": "mock",
                "metadata": {},
            }
        ],
        "alerts": [{"code": "ALL_CLEAR", "level": "ok", "message": "Keine kritischen Warnungen."}],
    }
    lecture_activity = {
        "zone_id": "default-zone",
        "from": "2026-02-18T16:00:00Z",
        "to": "2026-02-18T19:00:00Z",
        "granularity": "1m",
        "points": [
            {
                "timestamp": "2026-02-18T19:00:00Z",
                "zone_id": "default-zone",
                "active_lectures": 14,
                "active_courses": 10,
                "starts_next_60m": 22,
                "ends_next_60m": 17,
                "source": "rapla_refresh",
                "quality_score": 0.9,
                "quality_flags": ["LECTURE_RAPLA"],
                "metadata": {"site_code": "MA"},
            }
        ],
    }

    validate_instance("https://sitcheck.dev/schemas/zone.schema.json", zone, store)
    validate_instance("https://sitcheck.dev/schemas/count-point.schema.json", count_point, store)
    validate_instance("https://sitcheck.dev/schemas/forecast-response.schema.json", forecast, store)
    validate_instance("https://sitcheck.dev/schemas/forecast-latest-response.schema.json", forecast_latest, store)
    validate_instance("https://sitcheck.dev/schemas/reference-object.schema.json", reference_object, store)
    validate_instance("https://sitcheck.dev/schemas/model-lineage.schema.json", forecast_latest["lineage"], store)
    validate_instance("https://sitcheck.dev/schemas/weekly-forecast-response.schema.json", weekly_forecast, store)
    validate_instance("https://sitcheck.dev/schemas/weekly-explain-response.schema.json", weekly_explanation, store)
    validate_instance("https://sitcheck.dev/schemas/explanation.schema.json", explanation, store)
    validate_instance("https://sitcheck.dev/schemas/recommendation.schema.json", recommendation, store)
    validate_instance("https://sitcheck.dev/schemas/scenario-input.schema.json", scenario_input, store)
    validate_instance("https://sitcheck.dev/schemas/scenario-result.schema.json", scenario_result, store)
    validate_instance("https://sitcheck.dev/schemas/dashboard-command-center.schema.json", command_center, store)
    validate_instance("https://sitcheck.dev/schemas/lecture-activity-response.schema.json", lecture_activity, store)
    validate_instance("https://sitcheck.dev/schemas/llm-explainability-context.schema.json", llm_context, store)
    validate_instance("https://sitcheck.dev/schemas/llm-explainability-context-v2.schema.json", llm_context_v2, store)
    validate_instance("https://sitcheck.dev/schemas/llm-explanation-response.schema.json", llm_response, store)

    invalid_zone = {"zone_id": "broken", "name": "X", "is_active": True}
    failed = False
    try:
        validate_instance("https://sitcheck.dev/schemas/zone.schema.json", invalid_zone, store)
    except Exception:
        failed = True
    if not failed:
        raise AssertionError("Invalid zone did not fail schema validation")

    print("python schema contract tests passed")


if __name__ == "__main__":
    main()
