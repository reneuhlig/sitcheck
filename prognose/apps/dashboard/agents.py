from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api_client import SitcheckApiClient
from explainability import build_explainability_context, render_dual_output_markdown, render_template_fallback


@dataclass
class QueryAgent:
    client: SitcheckApiClient

    def run(self, zone_id: str, horizon: int) -> dict[str, Any]:
        command_center: dict[str, Any] = {}
        try:
            command_center = self.client.get_command_center(
                zone_id=zone_id,
                horizon=horizon,
                history_minutes=max(horizon * 3, 120),
                stale_seconds=900,
                long_term_days=14,
            )
        except Exception:
            command_center = {}
        try:
            history = self.client.tool_get_history(
                zone_id=zone_id,
                minutes=max(horizon * 3, 120),
                granularity="1m",
            )
        except Exception:
            history = {"zone_id": zone_id, "points": [], "granularity": "1m"}
        return {
            "history": history,
            "latest_forecast": command_center.get("forecast_latest", {}),
            "command_center": command_center,
        }


@dataclass
class PlotAgent:
    client: SitcheckApiClient

    def run(self, zone_id: str, horizon: int) -> dict[str, Any]:
        try:
            return self.client.tool_get_history(zone_id=zone_id, minutes=max(horizon * 4, 180), granularity="1m")
        except Exception:
            return {"zone_id": zone_id, "points": [], "granularity": "1m"}


@dataclass
class AnalysisAgent:
    client: SitcheckApiClient

    def run(self, zone_id: str, horizon: int) -> dict[str, Any]:
        command_center: dict[str, Any] = {}
        try:
            command_center = self.client.get_command_center(
                zone_id=zone_id,
                horizon=horizon,
                history_minutes=max(horizon * 3, 120),
                stale_seconds=900,
                long_term_days=14,
            )
        except Exception:
            command_center = {}
        return {
            "forecast": command_center.get("forecast_latest", {}),
            "explain": command_center.get("explanation", {}),
            "recommend": command_center.get("recommendations", {}),
            "command_center": command_center,
        }


@dataclass
class RAGAgent:
    def run(self, query: str) -> dict[str, Any]:
        notes = [
            "No free SQL is used. Assistant only calls fixed REST tools.",
            "Progressive disclosure: summary -> drivers -> evidence -> counterfactual.",
            "MCP mode is read-only and should enforce persist=false for scenario simulation.",
        ]
        return {"query": query, "knowledge": notes}


class AssistantOrchestrator:
    def __init__(
        self,
        client: SitcheckApiClient,
        ollama_enabled: bool = False,
        ollama_model: str = "qwen2.5:0.5b",
        ollama_base_url: str = "http://localhost:11434",
        local_template_fallback_enabled: bool = False,
    ):
        self.client = client
        self.ollama_enabled = bool(ollama_enabled)
        self.ollama_model = str(ollama_model)
        self.ollama_base_url = str(ollama_base_url)
        self.local_template_fallback_enabled = bool(local_template_fallback_enabled)
        self.query_agent = QueryAgent(client)
        self.plot_agent = PlotAgent(client)
        self.analysis_agent = AnalysisAgent(client)
        self.rag_agent = RAGAgent()

    @staticmethod
    def _safe_call(fn, default):
        try:
            return fn()
        except Exception:
            return default

    @staticmethod
    def _default_history(zone_id: str) -> dict[str, Any]:
        return {
            "zone_id": zone_id,
            "points": [],
            "granularity": "1m",
        }

    @staticmethod
    def _default_forecast(zone_id: str, horizon: int) -> dict[str, Any]:
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "summary": "Fallback: forecast unavailable.",
            "model_version": "unavailable",
            "points": [],
            "evidence": {"quality": {"score": 0.0, "flags": ["UPSTREAM_UNAVAILABLE"]}},
        }

    @staticmethod
    def _default_explanation(zone_id: str, horizon: int) -> dict[str, Any]:
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "summary": "Fallback: explainability service unavailable.",
            "drivers": [
                {
                    "name": "upstream_unavailable",
                    "impact": 0.0,
                    "direction": "mixed",
                    "description": "Explainability endpoint timed out or was unreachable.",
                }
            ],
            "uncertainty": {"score": 1.0, "level": "high", "reason": "No explainability payload available."},
            "evidence": {"quality": {"score": 0.0, "flags": ["UPSTREAM_UNAVAILABLE"]}},
        }

    @staticmethod
    def _default_recommendation(zone_id: str, horizon: int) -> dict[str, Any]:
        return {
            "zone_id": zone_id,
            "horizon": horizon,
            "summary": "Fallback: recommendations unavailable.",
            "actions": [],
            "gates": {"quality_ok": False, "uncertainty_ok": False, "notes": ["UPSTREAM_UNAVAILABLE"]},
            "evidence": {"quality": {"score": 0.0, "flags": ["UPSTREAM_UNAVAILABLE"]}},
        }

    @staticmethod
    def _classify(query: str) -> str:
        q = query.lower()
        tokens = set(q.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split())
        if any(token in q for token in ["plot", "chart", "graph", "kurve"]):
            return "plot"
        if any(token in q for token in ["explain", "why", "recommend", "simulate", "scenario", "insight", "warum"]):
            return "analysis"
        if "usable xai" in q or any(token in tokens for token in ["policy", "guardrail", "mcp", "how"]):
            return "rag"
        return "query"

    @staticmethod
    def _classify_audience(query: str) -> str:
        q = query.lower()
        if any(
            token in q
            for token in [
                "professor",
                "prof.",
                "kolloquium",
                "bewertung",
                "auslastung erklaeren",
                "auslastung erklären",
            ]
        ):
            return "professor"
        if any(token in q for token in ["management", "vorstand", "executive", "leitung", "business"]):
            return "executive"
        if any(token in q for token in ["student", "nutzer", "enduser", "laie", "einfach"]):
            return "enduser"
        return "ops"

    def _build_local_fallback_context(
        self,
        user_query: str,
        zone_id: str,
        horizon: int,
        audience: str,
        route_data: dict[str, Any],
        fallback_timeout: float | None = None,
        allow_network_fetch: bool = True,
    ) -> dict[str, Any]:
        history = None
        command_center = route_data.get("command_center") if isinstance(route_data, dict) else None
        if isinstance(command_center, dict):
            history = command_center.get("history")
        if not isinstance(history, dict):
            history = route_data.get("history") if isinstance(route_data, dict) else None
        if not isinstance(history, dict):
            if allow_network_fetch:
                history = self._safe_call(
                    lambda: self.client.tool_get_history(
                        zone_id=zone_id,
                        minutes=max(horizon * 4, 180),
                        granularity="1m",
                        timeout=fallback_timeout,
                    ),
                    self._default_history(zone_id),
                )
            else:
                history = self._default_history(zone_id)

        forecast_latest = None
        explanation = None
        recommendation = None
        if isinstance(command_center, dict):
            forecast_latest = command_center.get("forecast_latest")
            explanation = command_center.get("explanation")
            recommendation = command_center.get("recommendations")
        if not isinstance(forecast_latest, dict):
            forecast_latest = route_data.get("latest_forecast") if isinstance(route_data, dict) else None
        if not isinstance(forecast_latest, dict):
            forecast_latest = route_data.get("forecast") if isinstance(route_data, dict) else None
        if not isinstance(explanation, dict):
            explanation = route_data.get("explain") if isinstance(route_data, dict) else None
        if not isinstance(recommendation, dict):
            recommendation = route_data.get("recommend") if isinstance(route_data, dict) else None

        if not isinstance(forecast_latest, dict):
            if allow_network_fetch:
                forecast_latest = self._safe_call(
                    lambda: self.client.get_forecast_latest(
                        zone_id=zone_id,
                        horizon=horizon,
                        timeout=fallback_timeout,
                    ),
                    self._default_forecast(zone_id, horizon),
                )
            else:
                forecast_latest = self._default_forecast(zone_id, horizon)
        if not isinstance(explanation, dict):
            if allow_network_fetch:
                explanation = self._safe_call(
                    lambda: self.client.tool_explain_forecast(
                        zone_id=zone_id,
                        horizon=horizon,
                        timeout=fallback_timeout,
                    ),
                    self._default_explanation(zone_id, horizon),
                )
            else:
                explanation = self._default_explanation(zone_id, horizon)
        if not isinstance(recommendation, dict):
            if allow_network_fetch:
                recommendation = self._safe_call(
                    lambda: self.client.tool_recommend_actions(
                        zone_id=zone_id,
                        horizon=horizon,
                        timeout=fallback_timeout,
                    ),
                    self._default_recommendation(zone_id, horizon),
                )
            else:
                recommendation = self._default_recommendation(zone_id, horizon)

        return build_explainability_context(
            forecast_latest=forecast_latest,
            explanation=explanation,
            recommendation=recommendation,
            history=history,
            query=user_query,
            audience=audience,
            language="de",
            timezone="UTC",
            scenario_digest=None,
        )

    def _narrate_via_api(
        self,
        *,
        user_query: str,
        zone_id: str,
        horizon: int,
        audience: str,
        route_data: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        try:
            response = self.client.generate_explain_narrative(
                zone_id=zone_id,
                horizon=horizon,
                audience=audience,
                query=user_query,
                language="de",
                response_mode="free",
                ollama_model=self.ollama_model if self.ollama_enabled else None,
                require_ollama=self.ollama_enabled,
            )
            markdown = str(response.get("narrative_markdown") or "")
            if not markdown and isinstance(response.get("response"), dict):
                markdown = render_dual_output_markdown(response["response"])
            if not markdown:
                markdown = "Keine Narrative verfügbar."
            return markdown, response, str(response.get("mode", "api"))
        except Exception as exc:
            if not self.local_template_fallback_enabled:
                message = (
                    "Kurzfazit: Qwen/Ollama konnte gerade keine gültige Explainability-Antwort liefern.\n\n"
                    f"Fehler: {exc}\n\n"
                    "Hinweis: Der lokale Template-Fallback ist deaktiviert, damit im Assistant nur echte Qwen-Antworten angezeigt werden."
                )
                payload = {
                    "mode": "api_error_no_local_fallback",
                    "response": {"narrative_markdown": message},
                    "context": {},
                    "warnings": [f"central_api_error: {exc}"],
                    "meta": {
                        "template_set_id": "none",
                        "prompt_version": "none",
                        "model": self.ollama_model if self.ollama_enabled else "none",
                        "fallback_reason": "local_fallback_disabled",
                    },
                    "structured": {},
                }
                return message, payload, "api_error_no_local_fallback"
            try:
                reduced_timeout = max(2.0, min(6.0, self.client.timeout / 3.0))
                fallback_context = self._build_local_fallback_context(
                    user_query=user_query,
                    zone_id=zone_id,
                    horizon=horizon,
                    audience=audience,
                    route_data=route_data,
                    fallback_timeout=reduced_timeout,
                    allow_network_fetch=False,
                )
                fallback_response = render_template_fallback(fallback_context)
                if isinstance(fallback_response.get("structured"), dict):
                    limitations = fallback_response["structured"].setdefault("limitations", [])
                    if isinstance(limitations, list):
                        limitations.append(f"Central narrative API unavailable: {exc}")
                        limitations.append("Local fallback used without additional upstream calls.")
                payload = {
                    "mode": "local_template_fallback",
                    "response": fallback_response,
                    "context": fallback_context,
                    "warnings": [
                        f"central_api_error: {exc}",
                        "local_fallback_network_enrichment=disabled",
                    ],
                    "meta": {
                        "template_set_id": "legacy-local",
                        "prompt_version": "v1-fallback",
                        "model": "template",
                        "fallback_reason": "central_api_unreachable",
                    },
                    "structured": fallback_response.get("structured", {}),
                }
                return render_dual_output_markdown(fallback_response), payload, "local_template_fallback"
            except Exception as nested_exc:
                minimal_text = (
                    "Kurzfazit: Explainability ist aktuell nur eingeschränkt verfügbar.\n\n"
                    "Warum: API/LLM-Requests sind in Timeout gelaufen.\n\n"
                    "Nächster Schritt: Erneut versuchen oder nur Command-Center-Metriken nutzen."
                )
                payload = {
                    "mode": "minimal_fallback",
                    "response": {"narrative_markdown": minimal_text},
                    "context": {},
                    "warnings": [f"central_api_error: {exc}", f"local_fallback_error: {nested_exc}"],
                    "meta": {
                        "template_set_id": "legacy-local",
                        "prompt_version": "v1-minimal-fallback",
                        "model": "template",
                        "fallback_reason": "both_api_and_local_fallback_failed",
                    },
                    "structured": {},
                }
                return minimal_text, payload, "minimal_fallback"

    def run(
        self,
        user_query: str,
        zone_id: str,
        horizon: int,
        audience_override: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        route = self._classify(user_query)
        audience = audience_override or self._classify_audience(user_query)

        if route == "plot":
            route_data = self.plot_agent.run(zone_id, horizon)
            payload: dict[str, Any] = {"route": route, "audience": audience, "data": route_data}
            return "Plot data bereitgestellt.", payload

        if route == "analysis":
            route_data = self.analysis_agent.run(zone_id, horizon)
        elif route == "rag":
            route_data = self.rag_agent.run(user_query)
            payload = {"route": route, "audience": audience, "data": route_data}
            return (
                "RAG-Hinweis: Assistant nutzt feste REST-Tools ohne freie SQL-Abfragen. "
                "Für Explainability bitte eine Analysefrage zur Prognose stellen.",
                payload,
            )
        else:
            route_data = self.query_agent.run(zone_id, horizon)

        narrative_text, narrative_payload, mode = self._narrate_via_api(
            user_query=user_query,
            zone_id=zone_id,
            horizon=horizon,
            audience=audience,
            route_data=route_data,
        )

        payload = {
            "route": route,
            "audience": audience,
            "mode": mode,
            "data": route_data,
            "narrative": narrative_payload,
            "llm_response": narrative_payload.get("response", {}),
            "ecp": narrative_payload.get("context", {}),
        }
        return narrative_text, payload
