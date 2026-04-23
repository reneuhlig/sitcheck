#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "apps" / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

from agents import AssistantOrchestrator


class FakeClient:
    timeout = 15.0

    def get_command_center(
        self,
        zone_id: str,
        horizon: int = 60,
        history_minutes: int = 180,
        stale_seconds: int = 900,
        long_term_days: int = 14,
    ):
        return {
            "meta": {
                "generated_at": "2026-02-24T12:00:00Z",
                "zone_id": zone_id,
                "horizon": horizon,
                "history_minutes": history_minutes,
                "long_term_days": long_term_days,
                "stale_seconds": stale_seconds,
                "environment": "test",
            },
            "service_health": [{"service": "api-gateway", "status": "ok", "latency_ms": 0, "detail": "ok"}],
            "live": {
                "timestamp": "2026-02-24T11:59:00Z",
                "occupancy": 10,
                "utilization": 0.1,
                "quality_score": 0.9,
                "quality_flags": ["OK"],
                "point_count": 120,
            },
            "history": self.tool_get_history(zone_id=zone_id, minutes=history_minutes, granularity="1m"),
            "forecast_latest": self.get_forecast_latest(zone_id=zone_id, horizon=horizon),
            "forecast_long_term": [],
            "explanation": self.tool_explain_forecast(zone_id=zone_id, horizon=horizon),
            "recommendations": self.tool_recommend_actions(zone_id=zone_id, horizon=horizon),
            "calendar_events": [],
            "alerts": [{"code": "ALL_CLEAR", "level": "ok", "message": "ok"}],
        }

    def tool_get_history(self, zone_id: str, minutes: int = 180, granularity: str = "1m"):
        return {
            "zone_id": zone_id,
            "from": "2026-02-24T10:00:00Z",
            "to": "2026-02-24T12:00:00Z",
            "granularity": granularity,
            "points": [
                {
                    "timestamp": "2026-02-24T11:59:00Z",
                    "zone_id": zone_id,
                    "occupancy": 10,
                    "utilization": 0.1,
                    "quality_score": 0.9,
                    "quality_flags": ["OK"],
                    "evidence": {
                        "evidence_id": "ev-test",
                        "generated_at": "2026-02-24T11:59:00Z",
                        "time_window": {"from": "2026-02-24T11:00:00Z", "to": "2026-02-24T11:59:00Z"},
                        "sources": [{"type": "counts", "id": "hist-1"}],
                        "model": {"name": "seed", "version": "v1"},
                        "quality": {"score": 0.9, "flags": ["OK"]},
                    },
                }
            ],
        }

    def get_forecast_latest(self, zone_id: str, horizon: int = 60):
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "generated_at": "2026-02-24T12:00:00Z",
            "age_seconds": 10,
            "stale": False,
            "summary": "stable",
            "model_version": "baseline-v1",
            "source": "snapshot",
            "points": [{"timestamp": "2026-02-24T12:01:00Z", "yhat": 12, "pi_low": 9, "pi_high": 15}],
            "evidence": {
                "evidence_id": "ev-forecast",
                "generated_at": "2026-02-24T12:00:00Z",
                "time_window": {"from": "2026-02-24T11:00:00Z", "to": "2026-02-24T12:00:00Z"},
                "sources": [{"type": "forecast", "id": "fc-1"}],
                "model": {"name": "baseline", "version": "v1"},
                "quality": {"score": 0.9, "flags": ["OK"]},
            },
        }

    def tool_get_forecast(self, zone_id: str, horizon: int = 60):
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "generated_at": "2026-02-24T12:00:00Z",
            "summary": "stable",
            "model_version": "baseline-v1",
            "points": [{"timestamp": "2026-02-24T12:01:00Z", "yhat": 12, "pi_low": 9, "pi_high": 15}],
            "evidence": self.get_forecast_latest(zone_id, horizon).get("evidence"),
        }

    def tool_explain_forecast(self, zone_id: str, horizon: int = 60):
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "summary": "momentum up",
            "drivers": [{"name": "trend", "impact": 0.4, "direction": "up", "description": "recent trend"}],
            "uncertainty": {"score": 0.2, "level": "low", "reason": "enough history"},
            "evidence": {
                "evidence_id": "ev-xai",
                "generated_at": "2026-02-24T12:00:00Z",
                "time_window": {"from": "2026-02-24T11:00:00Z", "to": "2026-02-24T12:00:00Z"},
                "sources": [{"type": "xai", "id": "x-1"}],
                "model": {"name": "xai", "version": "v1"},
                "quality": {"score": 0.9, "flags": ["OK"]},
            },
        }

    def tool_recommend_actions(self, zone_id: str, horizon: int = 60):
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "summary": "monitor",
            "actions": [{"action_type": "monitor", "priority": 1, "rationale": "ok", "expected_impact": {"delta_occupancy": -1}}],
            "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
            "evidence": {
                "evidence_id": "ev-rec",
                "generated_at": "2026-02-24T12:00:00Z",
                "time_window": {"from": "2026-02-24T11:00:00Z", "to": "2026-02-24T12:00:00Z"},
                "sources": [{"type": "recommendation", "id": "r-1"}],
                "model": {"name": "rules", "version": "v1"},
                "quality": {"score": 0.9, "flags": ["OK"]},
            },
        }

    def generate_explain_narrative(
        self,
        zone_id: str,
        horizon: int = 60,
        audience: str = "ops",
        query: str = "",
        language: str = "de",
        response_mode: str = "free",
        ollama_model: str | None = None,
        require_ollama: bool = False,
    ):
        _ = (query, language, response_mode, ollama_model, require_ollama)
        return {
            "mode": "template",
            "narrative_markdown": "Prognose bleibt stabil.",
            "response": {
                "narrative": {
                    "one_liner": "Prognose bleibt stabil.",
                    "warum": "Momentum ist positiv [ref:ref-1].",
                    "unsicherheit": "Unsicherheit ist niedrig [ref:ref-1].",
                    "empfehlung": "Weiter beobachten [ref:ref-1].",
                    "evidence_hinweis": "[ref:ref-1]",
                },
                "structured": {
                    "audience": audience,
                    "zone_id": zone_id,
                    "horizon": horizon,
                    "verdict": "monitor",
                    "top_drivers": [
                        {"name": "trend", "impact": 0.4, "direction": "up", "evidence_ref": "ref-1"}
                    ],
                    "uncertainty": {
                        "score": 0.2,
                        "level": "low",
                        "reason": "enough history",
                        "evidence_ref": "ref-1",
                    },
                    "recommended_actions": [
                        {"action_type": "monitor", "priority": 1, "rationale": "ok", "evidence_ref": "ref-1"}
                    ],
                    "evidence_refs": ["ref-1"],
                    "confidence_statement": "High confidence",
                    "limitations": [],
                },
            },
            "context": {
                "request_meta": {
                    "request_id": "ctx-test",
                    "generated_at": "2026-02-24T12:00:00Z",
                    "zone_id": zone_id,
                    "horizon": horizon,
                    "audience": audience,
                    "language": "de",
                    "query": query,
                    "guardrails": ["api_tools_only"],
                    "template_set_id": "explainability-de-v2",
                    "prompt_version": "2.0.0",
                }
            },
            "meta": {
                "template_set_id": "explainability-de-v2",
                "prompt_version": "2.0.0",
            },
            "warnings": [],
            "structured": {
                "audience": audience,
                "zone_id": zone_id,
                "horizon": horizon,
                "verdict": "monitor",
                "top_drivers": [{"name": "trend", "impact": 0.4, "direction": "up", "evidence_ref": "ref-1"}],
                "uncertainty": {"score": 0.2, "level": "low", "reason": "enough history", "evidence_ref": "ref-1"},
                "recommended_actions": [{"action_type": "monitor", "priority": 1, "rationale": "ok", "evidence_ref": "ref-1"}],
                "evidence_refs": ["ref-1"],
                "confidence_statement": "High confidence",
                "limitations": [],
            },
        }


assistant = AssistantOrchestrator(FakeClient(), ollama_enabled=False)

text, payload = assistant.run("show me history", "default-zone", 60)
assert text
assert payload["route"] in {"query", "plot", "analysis", "rag"}
if payload["route"] != "rag":
    assert payload.get("ecp") is not None
    assert payload.get("llm_response") is not None
    assert payload.get("narrative", {}).get("meta", {}).get("template_set_id") == "explainability-de-v2"

text2, payload2 = assistant.run("explain and recommend", "default-zone", 60)
assert payload2["route"] == "analysis"
assert "forecast" in payload2["data"]
assert payload2.get("llm_response", {}).get("structured", {}).get("audience") in {"ops", "executive", "enduser", "professor"}


class TimeoutClient(FakeClient):
    timeout = 15.0

    def get_command_center(
        self,
        zone_id: str,
        horizon: int = 60,
        history_minutes: int = 180,
        stale_seconds: int = 900,
        long_term_days: int = 14,
        timeout: float | None = None,
    ):
        _ = (zone_id, horizon, history_minutes, stale_seconds, long_term_days, timeout)
        raise TimeoutError("command-center timeout")

    def tool_get_history(self, zone_id: str, minutes: int = 180, granularity: str = "1m", timeout: float | None = None):
        _ = (zone_id, minutes, granularity, timeout)
        raise TimeoutError("history timeout")

    def get_forecast_latest(
        self,
        zone_id: str,
        horizon: int = 60,
        stale_seconds: int = 900,
        timeout: float | None = None,
    ):
        _ = (zone_id, horizon, stale_seconds, timeout)
        raise TimeoutError("forecast timeout")

    def tool_explain_forecast(self, zone_id: str, horizon: int = 60, timeout: float | None = None):
        _ = (zone_id, horizon, timeout)
        raise TimeoutError("explain timeout")

    def tool_recommend_actions(self, zone_id: str, horizon: int = 60, timeout: float | None = None):
        _ = (zone_id, horizon, timeout)
        raise TimeoutError("recommend timeout")

    def generate_explain_narrative(
        self,
        zone_id: str,
        horizon: int = 60,
        audience: str = "ops",
        query: str = "",
        language: str = "de",
        response_mode: str = "free",
        ollama_model: str | None = None,
        require_ollama: bool = False,
    ):
        _ = (zone_id, horizon, audience, query, language, response_mode, ollama_model, require_ollama)
        raise TimeoutError("narrative timeout")


timeout_assistant = AssistantOrchestrator(TimeoutClient(), ollama_enabled=False)
text3, payload3 = timeout_assistant.run("explain and recommend", "default-zone", 60, audience_override="professor")
assert text3
assert payload3["mode"] == "api_error_no_local_fallback"
assert payload3.get("narrative", {}).get("meta", {}).get("fallback_reason") == "local_fallback_disabled"
assert "nur echte Qwen-Antworten" in text3

print("phase6 dashboard assistant smoke test passed")
