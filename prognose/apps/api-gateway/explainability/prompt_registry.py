from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptBundle:
    prompt: str
    template_set_id: str
    prompt_version: str
    required_context_fields: list[str]


class PromptRegistry:
    def __init__(
        self,
        template_dir: Path | None = None,
        template_set_id: str = "explainability-de-v2",
    ):
        root = Path(__file__).resolve().parents[3]
        self.template_dir = template_dir or (root / "packages" / "shared" / "prompts" / "explainability")
        self.template_set_id = template_set_id
        self.manifest = self._load_manifest()
        self.prompt_version = str(self.manifest.get("version", "2.0.0"))

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.template_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"prompt manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifest_id = str(manifest.get("template_set_id", ""))
        if manifest_id != self.template_set_id:
            raise RuntimeError(
                f"template_set_id mismatch: expected {self.template_set_id}, got {manifest_id}"
            )
        return manifest

    def _read_template(self, rel_path: str) -> str:
        path = self.template_dir / rel_path
        if not path.exists():
            raise RuntimeError(f"template missing: {path}")
        return path.read_text().strip()

    @staticmethod
    def _render(text: str, variables: dict[str, str]) -> str:
        rendered = text
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
        return rendered

    @staticmethod
    def _truncate_text(value: Any, max_chars: int = 280) -> str:
        text = str(value or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    def _compact_context_for_prompt(self, context: dict[str, Any]) -> dict[str, Any]:
        request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
        occupancy = context.get("occupancy_explainer", {}) if isinstance(context.get("occupancy_explainer"), dict) else {}
        uncertainty = context.get("uncertainty", {}) if isinstance(context.get("uncertainty"), dict) else {}
        forecast = context.get("forecast_snapshot", {}) if isinstance(context.get("forecast_snapshot"), dict) else {}
        history = context.get("history_digest", {}) if isinstance(context.get("history_digest"), dict) else {}
        driver_summary = context.get("driver_summary", {}) if isinstance(context.get("driver_summary"), dict) else {}
        recommendation = (
            context.get("recommendation_digest", {})
            if isinstance(context.get("recommendation_digest"), dict)
            else {}
        )
        quality = context.get("quality_digest", {}) if isinstance(context.get("quality_digest"), dict) else {}
        policy = context.get("policy_block", {}) if isinstance(context.get("policy_block"), dict) else {}

        top_drivers_raw = (
            driver_summary.get("top_drivers", [])
            if isinstance(driver_summary.get("top_drivers"), list)
            else []
        )
        top_drivers = []
        for item in top_drivers_raw[:3]:
            if not isinstance(item, dict):
                continue
            top_drivers.append(
                {
                    "name": item.get("name"),
                    "impact": item.get("impact"),
                    "direction": item.get("direction"),
                    "description": self._truncate_text(item.get("description"), max_chars=180),
                    "evidence_ref": item.get("evidence_ref"),
                }
            )

        rec_actions_raw = recommendation.get("actions", []) if isinstance(recommendation.get("actions"), list) else []
        rec_actions = []
        for item in rec_actions_raw[:3]:
            if not isinstance(item, dict):
                continue
            rec_actions.append(
                {
                    "action_type": item.get("action_type"),
                    "priority": item.get("priority"),
                    "rationale": self._truncate_text(item.get("rationale"), max_chars=220),
                    "evidence_ref": item.get("evidence_ref"),
                }
            )

        candidates_raw = (
            context.get("improvement_candidates", [])
            if isinstance(context.get("improvement_candidates"), list)
            else []
        )
        candidates = []
        for item in candidates_raw[:3]:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    "measure_id": item.get("measure_id"),
                    "measure_text": self._truncate_text(item.get("measure_text"), max_chars=180),
                    "expected_effect": self._truncate_text(item.get("expected_effect"), max_chars=180),
                    "effort": item.get("effort"),
                    "owner_hint": item.get("owner_hint"),
                    "evidence_ref": item.get("evidence_ref"),
                }
            )

        alerts_raw = quality.get("alerts", []) if isinstance(quality.get("alerts"), list) else []
        alerts = []
        for item in alerts_raw[:4]:
            if not isinstance(item, dict):
                continue
            alerts.append(
                {
                    "code": item.get("code"),
                    "level": item.get("level"),
                    "message": self._truncate_text(item.get("message"), max_chars=180),
                }
            )

        citation_raw = context.get("citation_map", []) if isinstance(context.get("citation_map"), list) else []
        citations = []
        for item in citation_raw[:12]:
            if not isinstance(item, dict):
                continue
            window = item.get("time_window", {}) if isinstance(item.get("time_window"), dict) else {}
            citations.append(
                {
                    "ref_id": item.get("ref_id"),
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                    "quality_score": item.get("quality_score"),
                    "time_window": {"from": window.get("from"), "to": window.get("to")},
                }
            )

        compact_context = {
            "request_meta": {
                "zone_id": request_meta.get("zone_id"),
                "horizon": request_meta.get("horizon"),
                "audience": request_meta.get("audience"),
                "language": request_meta.get("language"),
                "query": self._truncate_text(request_meta.get("query"), max_chars=240),
                "generated_at": request_meta.get("generated_at"),
                "guardrails": request_meta.get("guardrails", []),
                "template_set_id": request_meta.get("template_set_id"),
                "prompt_version": request_meta.get("prompt_version"),
            },
            "zone_capacity": context.get("zone_capacity"),
            "utilization_now_pct": context.get("utilization_now_pct"),
            "occupancy_explainer": {
                "current_occupancy": occupancy.get("current_occupancy"),
                "avg_60m": occupancy.get("avg_60m"),
                "trend_15m": occupancy.get("trend_15m"),
                "forecast_next_60m_peak": occupancy.get("forecast_next_60m_peak"),
                "risk_level": occupancy.get("risk_level"),
            },
            "forecast_snapshot": {
                "zone_id": forecast.get("zone_id"),
                "horizon": forecast.get("horizon"),
                "generated_at": forecast.get("generated_at"),
                "age_seconds": forecast.get("age_seconds"),
                "stale": forecast.get("stale"),
                "summary": self._truncate_text(forecast.get("summary"), max_chars=220),
                "model_version": forecast.get("model_version"),
                "source": forecast.get("source"),
                "peak": forecast.get("peak"),
                "evidence_ref": forecast.get("evidence_ref"),
            },
            "history_digest": {
                "last_occupancy": history.get("last_occupancy"),
                "last_utilization": history.get("last_utilization"),
                "trend_15m": history.get("trend_15m"),
                "trend_60m": history.get("trend_60m"),
                "quality_score_avg": history.get("quality_score_avg"),
                "quality_flags_top": history.get("quality_flags_top", []),
                "point_count": history.get("point_count"),
            },
            "driver_summary": {"top_drivers": top_drivers, "evidence_ref": driver_summary.get("evidence_ref")},
            "uncertainty": uncertainty,
            "recommendation_digest": {
                "summary": self._truncate_text(recommendation.get("summary"), max_chars=220),
                "quality_ok": recommendation.get("quality_ok"),
                "uncertainty_ok": recommendation.get("uncertainty_ok"),
                "actions": rec_actions,
                "evidence_ref": recommendation.get("evidence_ref"),
            },
            "improvement_candidates": candidates,
            "quality_digest": {
                "live_quality_score": quality.get("live_quality_score"),
                "live_quality_flags": quality.get("live_quality_flags", []),
                "history_quality_score_avg": quality.get("history_quality_score_avg"),
                "live_point_count": quality.get("live_point_count"),
                "alerts": alerts,
            },
            "citation_map": citations,
            "policy_block": {
                "abstain_rules": (policy.get("abstain_rules", []) if isinstance(policy.get("abstain_rules"), list) else [])[:3],
                "decision_gates": policy.get("decision_gates", {}),
                "output_contract": policy.get("output_contract", {}),
            },
        }
        return compact_context

    def _build_context_brief(self, compact_context: dict[str, Any]) -> str:
        request_meta = compact_context.get("request_meta", {}) if isinstance(compact_context.get("request_meta"), dict) else {}
        occupancy = (
            compact_context.get("occupancy_explainer", {})
            if isinstance(compact_context.get("occupancy_explainer"), dict)
            else {}
        )
        uncertainty = compact_context.get("uncertainty", {}) if isinstance(compact_context.get("uncertainty"), dict) else {}
        recommendation = (
            compact_context.get("recommendation_digest", {})
            if isinstance(compact_context.get("recommendation_digest"), dict)
            else {}
        )
        forecast = (
            compact_context.get("forecast_snapshot", {})
            if isinstance(compact_context.get("forecast_snapshot"), dict)
            else {}
        )
        drivers = (
            compact_context.get("driver_summary", {}).get("top_drivers", [])
            if isinstance(compact_context.get("driver_summary"), dict)
            else []
        )
        candidates = (
            compact_context.get("improvement_candidates", [])
            if isinstance(compact_context.get("improvement_candidates"), list)
            else []
        )
        citations = (
            compact_context.get("citation_map", [])
            if isinstance(compact_context.get("citation_map"), list)
            else []
        )
        quality = compact_context.get("quality_digest", {}) if isinstance(compact_context.get("quality_digest"), dict) else {}

        lines: list[str] = []
        lines.append(
            "Meta: "
            f"zone={request_meta.get('zone_id')} horizon={request_meta.get('horizon')} "
            f"audience={request_meta.get('audience')} language={request_meta.get('language')}"
        )
        lines.append(
            "Live: "
            f"occupancy={occupancy.get('current_occupancy')} avg60={occupancy.get('avg_60m')} "
            f"trend15={occupancy.get('trend_15m')} peak60={occupancy.get('forecast_next_60m_peak')} "
            f"risk={occupancy.get('risk_level')} util_now_pct={compact_context.get('utilization_now_pct')}"
        )
        lines.append(
            "Forecast: "
            f"stale={forecast.get('stale')} age_s={forecast.get('age_seconds')} "
            f"model={forecast.get('model_version')} source={forecast.get('source')} "
            f"evidence_ref={forecast.get('evidence_ref')}"
        )
        lines.append(
            "Uncertainty: "
            f"level={uncertainty.get('level')} score={uncertainty.get('score')} "
            f"reason={self._truncate_text(uncertainty.get('reason'), max_chars=160)} "
            f"evidence_ref={uncertainty.get('evidence_ref')}"
        )
        lines.append(
            "Gates: "
            f"quality_ok={recommendation.get('quality_ok')} uncertainty_ok={recommendation.get('uncertainty_ok')} "
            f"live_quality_score={quality.get('live_quality_score')} "
            f"flags={quality.get('live_quality_flags')}"
        )

        if drivers:
            lines.append("TopDrivers:")
            for idx, item in enumerate(drivers[:3], start=1):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- d{idx}: name={item.get('name')} impact={item.get('impact')} direction={item.get('direction')} "
                    f"evidence_ref={item.get('evidence_ref')}"
                )

        if candidates:
            lines.append("ImprovementCandidates:")
            for idx, item in enumerate(candidates[:3], start=1):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- m{idx}: id={item.get('measure_id')} effort={item.get('effort')} "
                    f"text={self._truncate_text(item.get('measure_text'), max_chars=130)} "
                    f"effect={self._truncate_text(item.get('expected_effect'), max_chars=130)} "
                    f"owner={item.get('owner_hint')} evidence_ref={item.get('evidence_ref')}"
                )

        if citations:
            lines.append("EvidenceRefs:")
            refs = [str(item.get("ref_id")) for item in citations[:10] if isinstance(item, dict)]
            lines.append(", ".join(refs))

        return "\n".join(lines)

    def validate_context(self, context: dict[str, Any]) -> None:
        required = list(self.manifest.get("required_context_fields", []))
        missing = [field for field in required if field not in context]
        if missing:
            raise ValueError(f"context missing required fields: {', '.join(missing)}")

    def build_prompt(
        self,
        *,
        context: dict[str, Any],
        audience: str = "ops",
        language: str = "de",
        query: str = "",
    ) -> PromptBundle:
        self.validate_context(context)

        audiences = list(self.manifest.get("audiences", []))
        if audience not in audiences:
            audience = str(self.manifest.get("default_audience", "ops"))

        files = self.manifest.get("files", {})
        audience_files = files.get("audience", {}) if isinstance(files.get("audience"), dict) else {}

        system_section = self._read_template(str(files.get("system", "system.de.md")))
        task_section = self._read_template(str(files.get("task", "task.forecast_explain.de.md")))
        output_rules_section = self._read_template(str(files.get("output_rules", "output_rules.de.md")))
        audience_section = self._read_template(str(audience_files.get(audience, "audience.ops.de.md")))

        variables = {
            "audience": audience,
            "language": language,
            "query": query,
            "template_set_id": self.template_set_id,
            "prompt_version": self.prompt_version,
        }

        compact_context = self._compact_context_for_prompt(context)
        context_brief = self._build_context_brief(compact_context)
        sections = [
            self._render(system_section, variables),
            self._render(audience_section, variables),
            self._render(task_section, variables),
            self._render(output_rules_section, variables),
            "ECP v2 Kontext (kompakt):\n" + context_brief,
        ]

        return PromptBundle(
            prompt="\n\n".join(section for section in sections if section),
            template_set_id=self.template_set_id,
            prompt_version=self.prompt_version,
            required_context_fields=list(self.manifest.get("required_context_fields", [])),
        )

    def build_fast_narrative_prompt(
        self,
        *,
        context: dict[str, Any],
        audience: str = "ops",
        language: str = "de",
        query: str = "",
        query_intent: str = "forecast",
        query_focus: str = "general",
        answer_profile: str = "forecast_full",
        include_kpi_block: bool = True,
        missing_fields: list[str] | None = None,
        retry_attempt: int = 0,
    ) -> PromptBundle:
        self.validate_context(context)

        audiences = list(self.manifest.get("audiences", []))
        if audience not in audiences:
            audience = str(self.manifest.get("default_audience", "ops"))

        compact_context = self._compact_context_for_prompt(context)
        context_brief = self._build_context_brief(compact_context)
        request_meta = compact_context.get("request_meta", {}) if isinstance(compact_context.get("request_meta"), dict) else {}
        citations = compact_context.get("citation_map", []) if isinstance(compact_context.get("citation_map"), list) else []
        citation_refs = [str(item.get("ref_id")) for item in citations[:8] if isinstance(item, dict) and item.get("ref_id")]
        refs_hint = ", ".join(citation_refs) if citation_refs else "ref-1"
        retry_fields = [field for field in (missing_fields or []) if field]
        query_clean = self._truncate_text(query, max_chars=220)

        common_lines = [
            "Du bist der Sitcheck Explainability Assistant.",
            "Nutze ausschliesslich die bereitgestellten Fakten.",
            "Erzeuge NUR ein gueltiges JSON-Objekt mit genau zwei Top-Level-Schluesseln: narrative und structured.",
            "narrative muss enthalten: one_liner, warum, unsicherheit, empfehlung, evidence_hinweis.",
            "structured muss enthalten: audience, zone_id, horizon, verdict, top_drivers, uncertainty, recommended_actions, evidence_refs, confidence_statement, limitations.",
            "verdict ist eines von: monitor, attention, action_needed, blocked.",
            "Wenn Modell/Fallback, CONTEXT_STALE, NO_CONTEXT_DB oder schlechte Datenqualitaet erkennbar sind: benenne 'degradiert/nicht normal belastbar'.",
            "Wenn verdict=blocked: recommended_actions nur monitor oder quality-hardening, keine Staffing-/Peak-Hartmassnahme.",
            "Jeder top_drivers-, uncertainty- und recommended_actions-Eintrag braucht evidence_ref aus den erlaubten refs.",
            "Kein Markdown und keine zusaetzlichen Top-Level-Schluessel.",
            "Schreibe narrative.* in 1-2 klaren Saetzen, maximal 35 Woerter je Feld.",
            "Wiederhole die Nutzerfrage NICHT woertlich.",
            "Beginne one_liner nicht mit Imperativen wie 'Erklaere', 'Schreibe', 'Welche' oder 'Was sind'.",
            f"Sprache={language}, Zielgruppe={audience}.",
            (
                "Query-Intent="
                f"{query_intent}, Query-Focus={query_focus}, "
                f"Answer-Profile={answer_profile}, Include-KPI={str(include_kpi_block).lower()}"
            ),
            f"Nutzerfrage: {query_clean}",
            f"Zone={request_meta.get('zone_id')} Horizon={request_meta.get('horizon')}",
            f"Erlaubte evidence refs: {refs_hint}",
        ]

        if retry_attempt > 0 and retry_fields:
            common_lines.extend(
                [
                    f"REPARATURVERSUCH #{retry_attempt}:",
                    "Die letzte Antwort hatte zu wenig narrativen Inhalt.",
                    "Fuelle diese Felder zwingend sinnvoll aus: " + ", ".join(retry_fields),
                ]
            )

        if query_intent == "forecast":
            intent_lines = [
                "Antwortstil: evidenzbasiert, handlungsorientiert, nicht schablonenhaft.",
                "Beantworte die konkrete Nutzerfrage zuerst im one_liner.",
                "Nutze KPIs nur wenn die Frage danach verlangt oder Include-KPI=true ist.",
                "Wenn Unsicherheit hoch ist, benenne Grenzen klar und priorisiere vorsichtige Handlungshinweise.",
            ]
            if query_focus == "student":
                intent_lines.extend(
                    [
                        "Fokus=Student: Formuliere in einfacher deutscher Sprache (A2/B1), kurze Saetze.",
                        "Vermeide Fachbegriffe; wenn noetig, erklaere sie in einem Halbsatz.",
                        "one_liner muss aktuelle Lage + Ausblick fuer naechste 60 Minuten enthalten.",
                    ]
                )
            elif query_focus == "improvement":
                intent_lines.extend(
                    [
                        "Fokus=Verbesserung: formuliere im one_liner die wichtigsten Massnahmen.",
                        "warum erklaert die Priorisierung der Massnahmen.",
                        "empfehlung beschreibt den konkret naechsten Umsetzungsschritt.",
                    ]
                )
            elif query_focus == "risk":
                intent_lines.extend(
                    [
                        "Fokus=Risiko: one_liner nennt das zentrale Prognoserisiko.",
                        "unsicherheit beschreibt Auswirkungen und Grenzen klar.",
                        "empfehlung nennt Mitigation/Monitoring als konkrete Reaktion.",
                    ]
                )
            elif query_focus == "briefing":
                intent_lines.extend(
                    [
                        "Fokus=Briefing: formuliere management-tauglich und adressatengerecht.",
                        "one_liner muss eine klare Entscheidungsaussage enthalten.",
                    ]
                )
            elif query_focus == "status":
                intent_lines.extend(
                    [
                        "Fokus=Status: kurze Lageeinordnung fuer den aktuellen Zustand.",
                        "one_liner muss eine konkrete Lageaussage mit mindestens einer Zahl aus dem Kontext enthalten.",
                        "Vermeide generische Handlungspakete, wenn nicht explizit gefragt.",
                    ]
                )
        elif query_intent == "meta":
            intent_lines = [
                "Meta-Modus: Beantworte die Modell-/Systemfrage direkt und knapp.",
                "Keine erzwungenen Forecast-Abschnitte, keine erfundenen Prognosezahlen.",
                "Nutze empfehlung als kurze Weiterleitung auf eine sinnvolle Forecast-Frage.",
            ]
        else:
            intent_lines = [
                "Off-Topic-Modus: Gib eine kurze freundliche Absage fuer die Fremddomaene.",
                "Bleibe transparent beim Scope: Sitcheck-Prognosen statt allgemeinem Chat.",
                "Nutze empfehlung fuer eine naechste sinnvolle Forecast-Frage.",
            ]

        prompt = "\n".join(common_lines + intent_lines + ["Kontext:", context_brief])

        return PromptBundle(
            prompt=prompt,
            template_set_id=self.template_set_id,
            prompt_version=self.prompt_version,
            required_context_fields=list(self.manifest.get("required_context_fields", [])),
        )
