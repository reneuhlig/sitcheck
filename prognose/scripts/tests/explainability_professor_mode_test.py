#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_GATEWAY_DIR = ROOT / "apps" / "api-gateway"
if str(API_GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(API_GATEWAY_DIR))

from explainability.context_builder import build_explainability_context_v2  # noqa: E402
from explainability.narrative_service import (  # noqa: E402
    render_dual_output_markdown,
    render_template_fallback,
    validate_dual_response_shape,
)


def sample_evidence(source_type: str, source_id: str) -> dict:
    return {
        "evidence_id": f"ev-{source_type}",
        "generated_at": "2026-02-27T08:00:00Z",
        "time_window": {"from": "2026-02-27T07:00:00Z", "to": "2026-02-27T08:00:00Z"},
        "sources": [{"type": source_type, "id": source_id}],
        "model": {"name": "baseline", "version": "v1"},
        "quality": {"score": 0.92, "flags": ["OK"]},
    }


def main() -> int:
    context = build_explainability_context_v2(
        zone_id="default-zone",
        zone_capacity=120,
        horizon=60,
        audience="professor",
        language="de",
        query="Bitte professorentauglich erklaeren",
        forecast_latest={
            "zone_id": "default-zone",
            "horizon": 60,
            "generated_at": "2026-02-27T08:00:00Z",
            "age_seconds": 20,
            "stale": False,
            "summary": "leichter Anstieg",
            "model_version": "tf-mlp-v2",
            "source": "snapshot",
            "points": [
                {"timestamp": "2026-02-27T08:01:00Z", "yhat": 66.0, "pi_low": 58.0, "pi_high": 72.0},
                {"timestamp": "2026-02-27T08:30:00Z", "yhat": 74.0, "pi_low": 64.0, "pi_high": 82.0},
            ],
            "evidence": sample_evidence("forecast", "fc-1"),
        },
        explanation={
            "zone_id": "default-zone",
            "horizon": 60,
            "summary": "trend + lecture",
            "drivers": [
                {"name": "trend", "impact": 0.41, "direction": "up", "description": "kurzfristiger Anstieg"},
                {"name": "lecture_net_pull", "impact": -0.21, "direction": "down", "description": "aktive Vorlesungen"},
            ],
            "uncertainty": {"score": 0.33, "level": "medium", "reason": "moderate data drift"},
            "evidence": sample_evidence("xai", "x-1"),
        },
        recommendation={
            "zone_id": "default-zone",
            "horizon": 60,
            "summary": "fruehzeitig steuern",
            "actions": [
                {"action_type": "open_room", "priority": 1, "rationale": "peak erwartbar"},
            ],
            "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
            "evidence": sample_evidence("recommendation", "r-1"),
        },
        history_points=[
            {
                "timestamp": "2026-02-27T07:59:00Z",
                "zone_id": "default-zone",
                "occupancy": 60,
                "utilization": 0.5,
                "quality_score": 0.93,
                "quality_flags": ["OK"],
                "evidence": sample_evidence("counts", "hist-1"),
            },
            {
                "timestamp": "2026-02-27T08:00:00Z",
                "zone_id": "default-zone",
                "occupancy": 64,
                "utilization": 0.533,
                "quality_score": 0.93,
                "quality_flags": ["OK"],
                "evidence": sample_evidence("counts", "hist-2"),
            },
        ],
        live_state={
            "timestamp": "2026-02-27T08:00:00Z",
            "occupancy": 64,
            "utilization": 0.533,
            "quality_score": 0.93,
            "quality_flags": ["OK"],
            "point_count": 180,
        },
        alerts=[{"code": "ALL_CLEAR", "level": "ok", "message": "none"}],
        lecture_impact={
            "timestamp": "2026-02-27T08:00:00Z",
            "source": "rapla_refresh",
            "quality_score": 0.9,
            "quality_flags": ["LECTURE_RAPLA"],
            "active_lectures": 12,
            "active_courses": 9,
            "starts_next_60m": 16,
            "ends_next_60m": 14,
            "impact": {
                "heavy_active_lectures": 3,
                "heavy_ended_last_60m": 1,
                "lecture_pull_regular": 240,
                "heavy_bib_bonus": 16,
                "lecture_net_pull": 224,
                "impact_model_version": "lecture-impact-v1",
            },
        },
        template_set_id="explainability-de-v2",
        prompt_version="2.2.0",
    )

    fallback = render_template_fallback(context)
    ok, reason = validate_dual_response_shape(fallback)
    if not ok:
        raise AssertionError(f"fallback shape invalid: {reason}")

    actions = fallback.get("structured", {}).get("recommended_actions", [])
    if len(actions) < 1:
        raise AssertionError("expected at least one recommended action in fallback response")

    markdown = render_dual_output_markdown(
        fallback,
        context=context,
        answer_profile="forecast_full",
        include_kpi_block=False,
        query_intent="forecast",
    )
    required_blocks = ["**Kurzfazit**", "**Warum ist die Auslastung so?**", "**Was ist unsicher?**", "**Was verbessern wir als Naechstes?**"]
    for block in required_blocks:
        if block not in markdown:
            raise AssertionError(f"missing forecast block: {block}")

    if "```" in markdown:
        raise AssertionError("markdown contains JSON/code fence in main text")

    if "**Werte kompakt**" in markdown:
        raise AssertionError("did not expect KPI block for compact-disabled render")

    if markdown.count("[ref:") < 1:
        raise AssertionError("expected evidence refs in measures")

    off_topic_markdown = render_dual_output_markdown(
        fallback,
        context=context,
        answer_profile="offtopic_redirect",
        include_kpi_block=False,
        query_intent="off_topic",
    )
    for block in ["**Kurzantwort**", "**Hinweis zum Scope**", "**Naechste sinnvolle Forecast-Frage**"]:
        if block not in off_topic_markdown:
            raise AssertionError(f"missing off-topic redirect block: {block}")
    if "**Werte kompakt**" in off_topic_markdown:
        raise AssertionError("off-topic render must not include KPI block")

    print("explainability professor mode tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
