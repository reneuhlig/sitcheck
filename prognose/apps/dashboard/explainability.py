from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any


SYSTEM_PROMPT_DE = (
    "Du bist Sitcheck Explainability Assistant. "
    "Nutze ausschließlich die gelieferten ECP-Daten. "
    "Wenn Evidenz fehlt, sage das explizit. "
    "Erfinde keine Ursachen, Quellen oder Zahlen. "
    "Jede Kernaussage braucht evidence_refs. "
    "Ausgabeformat: JSON mit Feldern narrative und structured gemäß Vertrag."
)

AUDIENCE_PROMPTS_DE = {
    "ops": (
        "Zielgruppe: Betriebsteam. Fokus: kurzfristige Peaks, Quality Flags, "
        "konkrete nächste Schritte in den nächsten Minuten. "
        "Erkläre Top-Treiber technisch knapp."
    ),
    "executive": (
        "Zielgruppe: Management. Fokus: Risiko, Auswirkung, Entscheidungsempfehlung, Evidenz-ID. "
        "Weniger technische Details, hohe Entscheidungsklarheit."
    ),
    "enduser": (
        "Zielgruppe: nicht-technische Nutzer. Fokus: einfache Sprache, "
        "Warum steigt/fällt es, wie sicher ist das, kurze Evidenzhinweise."
    ),
    "professor": (
        "Zielgruppe: Professorinnen und Professoren. Fokus: Auslastung verstaendlich erklaeren, "
        "Unsicherheit transparent machen und konkrete Verbesserungsmassnahmen fuer den Betrieb benennen."
    ),
}

GUARDRAILS = [
    "no_free_sql",
    "api_tools_only",
    "no_invented_facts",
    "each_claim_needs_evidence_ref",
    "no_hard_action_if_uncertainty_high",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _first_nonempty_ref(citation_map: list[dict[str, Any]], preferred_types: list[str]) -> str:
    for source_type in preferred_types:
        for item in citation_map:
            if item.get("source_type") == source_type:
                return str(item.get("ref_id"))
    if citation_map:
        return str(citation_map[0].get("ref_id"))
    return "ref-missing"


def build_history_digest(history_payload: dict[str, Any]) -> dict[str, Any]:
    points = list(history_payload.get("points", []))
    if not points:
        return {
            "last_occupancy": 0.0,
            "last_utilization": 0.0,
            "trend_15m": 0.0,
            "trend_60m": 0.0,
            "quality_score_avg": 0.5,
            "quality_flags_top": ["NO_HISTORY"],
            "point_count": 0,
        }

    occupancies = [_safe_float(point.get("occupancy", 0.0)) for point in points]
    utilizations = [_safe_float(point.get("utilization", 0.0)) for point in points]
    quality_scores = [_safe_float(point.get("quality_score", 0.5), 0.5) for point in points]

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    last_occupancy = occupancies[-1]
    last_utilization = utilizations[-1] if utilizations else 0.0

    mean_15 = mean(occupancies[-15:])
    mean_60 = mean(occupancies[-60:])

    trend_15m = last_occupancy - mean_15
    trend_60m = last_occupancy - mean_60

    flag_counter: Counter[str] = Counter()
    for point in points:
        for flag in point.get("quality_flags", []) or []:
            if isinstance(flag, str):
                flag_counter[flag] += 1

    if not flag_counter:
        flag_counter["OK"] += 1

    return {
        "last_occupancy": round(last_occupancy, 3),
        "last_utilization": round(last_utilization, 6),
        "trend_15m": round(trend_15m, 3),
        "trend_60m": round(trend_60m, 3),
        "quality_score_avg": max(0.0, min(1.0, round(mean(quality_scores), 6))),
        "quality_flags_top": [flag for flag, _ in flag_counter.most_common(3)],
        "point_count": len(points),
    }


def build_citation_map(*evidence_objects: dict[str, Any] | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for evidence in evidence_objects:
        if not isinstance(evidence, dict):
            continue

        evidence_id = str(evidence.get("evidence_id", "ev-missing"))
        time_window = evidence.get("time_window", {}) if isinstance(evidence.get("time_window"), dict) else {}
        model = evidence.get("model", {}) if isinstance(evidence.get("model"), dict) else {}
        quality = evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}

        sources = evidence.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        for source in sources:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("type", "api"))
            source_id = str(source.get("id", "unknown"))
            key = (evidence_id, source_type, source_id)
            if key in seen:
                continue
            seen.add(key)

            refs.append(
                {
                    "ref_id": f"ref-{len(refs) + 1}",
                    "evidence_id": evidence_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "time_window": {
                        "from": str(time_window.get("from", _now_iso())),
                        "to": str(time_window.get("to", _now_iso())),
                    },
                    "model_version": str(model.get("version", "unknown")),
                    "quality_score": max(0.0, min(1.0, _safe_float(quality.get("score", 0.5), 0.5))),
                    "quality_flags": list(quality.get("flags", [])) if isinstance(quality.get("flags"), list) else [],
                }
            )

    if not refs:
        refs.append(
            {
                "ref_id": "ref-1",
                "evidence_id": "ev-missing",
                "source_type": "api",
                "source_id": "none",
                "time_window": {"from": _now_iso(), "to": _now_iso()},
                "model_version": "unknown",
                "quality_score": 0.0,
                "quality_flags": ["NO_EVIDENCE"],
            }
        )

    return refs


def build_explainability_context(
    *,
    forecast_latest: dict[str, Any],
    explanation: dict[str, Any],
    recommendation: dict[str, Any],
    history: dict[str, Any],
    query: str,
    audience: str,
    language: str = "de",
    timezone: str = "UTC",
    scenario_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history_digest = build_history_digest(history)
    citation_map = build_citation_map(
        forecast_latest.get("evidence", {}),
        explanation.get("evidence", {}),
        recommendation.get("evidence", {}),
        scenario_digest.get("evidence", {}) if scenario_digest else None,
    )

    context = {
        "forecast_latest": forecast_latest,
        "explanation": explanation,
        "recommendation": recommendation,
        "history_digest": history_digest,
        "scenario_digest": scenario_digest,
        "context_meta": {
            "request_id": f"ctx-{uuid.uuid4()}",
            "generated_at": _now_iso(),
            "audience": audience,
            "language": language,
            "timezone": timezone,
            "guardrails": GUARDRAILS,
            "query": query,
        },
        "citation_map": citation_map,
    }
    return context


def _confidence_statement(level: str, score: float) -> str:
    if level == "high":
        return f"Niedrige Verlässlichkeit (Unsicherheit={score:.2f}). Aussagen nur als Tendenz nutzen."
    if level == "medium":
        return f"Mittlere Verlässlichkeit (Unsicherheit={score:.2f}). Entscheidungen mit Monitoring absichern."
    return f"Hohe Verlässlichkeit (Unsicherheit={score:.2f})."


def render_template_fallback(context: dict[str, Any]) -> dict[str, Any]:
    forecast = context.get("forecast_latest", {})
    explain = context.get("explanation", {})
    rec = context.get("recommendation", {})
    audience = str(context.get("context_meta", {}).get("audience", "ops"))
    citation_map = list(context.get("citation_map", []))

    zone_id = str(forecast.get("zone_id", "unknown-zone"))
    horizon = _safe_int(forecast.get("horizon", 60), 60)

    stale = bool(forecast.get("stale", False))
    uncertainty = explain.get("uncertainty", {}) if isinstance(explain.get("uncertainty"), dict) else {}
    uncertainty_level = str(uncertainty.get("level", "high"))
    uncertainty_score = max(0.0, min(1.0, _safe_float(uncertainty.get("score", 0.8), 0.8)))

    gates = rec.get("gates", {}) if isinstance(rec.get("gates"), dict) else {}
    quality_ok = bool(gates.get("quality_ok", True))
    uncertainty_ok = bool(gates.get("uncertainty_ok", True))

    if stale or not quality_ok:
        verdict = "blocked"
    elif uncertainty_level == "high" or not uncertainty_ok:
        verdict = "attention"
    elif any((a.get("action_type") != "monitor") for a in rec.get("actions", []) if isinstance(a, dict)):
        verdict = "action_needed"
    else:
        verdict = "monitor"

    ref_counts = _first_nonempty_ref(citation_map, ["counts", "forecast", "api"])
    ref_events = _first_nonempty_ref(citation_map, ["events", "counts", "api"])
    ref_xai = _first_nonempty_ref(citation_map, ["xai", "forecast", "counts"])
    ref_rec = _first_nonempty_ref(citation_map, ["recommendation", "forecast", "counts"])

    top_drivers: list[dict[str, Any]] = []
    for driver in (explain.get("drivers", []) if isinstance(explain.get("drivers"), list) else [])[:3]:
        if not isinstance(driver, dict):
            continue
        ref = ref_events if str(driver.get("name")) == "event_context" else ref_counts
        top_drivers.append(
            {
                "name": str(driver.get("name", "unknown")),
                "impact": _safe_float(driver.get("impact", 0.0)),
                "direction": str(driver.get("direction", "mixed")),
                "evidence_ref": ref,
            }
        )

    recommended_actions: list[dict[str, Any]] = []
    actions = rec.get("actions", []) if isinstance(rec.get("actions"), list) else []
    if uncertainty_level == "high":
        actions = [
            {
                "action_type": "monitor",
                "priority": 1,
                "rationale": "Unsicherheit ist hoch, daher nur überwachen.",
            }
        ]

    if not actions:
        actions = [{"action_type": "monitor", "priority": 1, "rationale": "Keine belastbare Aktion abgeleitet."}]

    for action in actions:
        if not isinstance(action, dict):
            continue
        recommended_actions.append(
            {
                "action_type": str(action.get("action_type", "monitor")),
                "priority": _safe_int(action.get("priority", 1), 1),
                "rationale": str(action.get("rationale", "")),
                "evidence_ref": ref_rec,
            }
        )

    evidence_refs = list({
        *(driver.get("evidence_ref") for driver in top_drivers),
        *(action.get("evidence_ref") for action in recommended_actions),
        ref_xai,
    })
    evidence_refs = [ref for ref in evidence_refs if isinstance(ref, str) and ref]
    if not evidence_refs:
        evidence_refs = ["ref-1"]

    limitations: list[str] = []
    if stale:
        limitations.append("Forecast-Snapshot ist veraltet.")
    if not quality_ok:
        limitations.append("Datenqualitäts-Gate blockiert harte Empfehlungen.")
    if uncertainty_level == "high":
        limitations.append("Hohe Unsicherheit reduziert Entscheidungssicherheit.")
    if not limitations:
        limitations.append("Keine kritischen Einschränkungen erkannt.")

    one_liner = f"Prognose für Zone {zone_id}: Urteil = {verdict}."
    warum = (
        f"Haupttreiber: {', '.join(d['name'] for d in top_drivers) or 'keine'}. "
        f"Basis ist Forecast/History-Evidenz [ref:{ref_counts}]."
    )
    unsicherheit_text = (
        f"Unsicherheit ist {uncertainty_level} (Score {uncertainty_score:.2f}) [ref:{ref_xai}]."
    )
    empfehlung_text = (
        f"Empfohlene Aktion: {recommended_actions[0]['action_type']} [ref:{recommended_actions[0]['evidence_ref']}]."
    )
    evidence_hint = (
        "Evidenzreferenzen: " + ", ".join(f"[ref:{ref}]" for ref in evidence_refs)
    )

    response = {
        "narrative": {
            "one_liner": one_liner,
            "warum": warum,
            "unsicherheit": unsicherheit_text,
            "empfehlung": empfehlung_text,
            "evidence_hinweis": evidence_hint,
        },
        "structured": {
            "audience": audience,
            "zone_id": zone_id,
            "horizon": horizon,
            "verdict": verdict,
            "top_drivers": top_drivers,
            "uncertainty": {
                "score": uncertainty_score,
                "level": uncertainty_level,
                "reason": str(uncertainty.get("reason", "unspecified")),
                "evidence_ref": ref_xai,
            },
            "recommended_actions": recommended_actions,
            "evidence_refs": evidence_refs,
            "confidence_statement": _confidence_statement(uncertainty_level, uncertainty_score),
            "limitations": limitations,
        },
    }
    return response


def build_llm_prompt(context: dict[str, Any], audience: str) -> str:
    audience_prompt = AUDIENCE_PROMPTS_DE.get(audience, AUDIENCE_PROMPTS_DE["ops"])
    context_json = json.dumps(context, ensure_ascii=False)
    return (
        f"{SYSTEM_PROMPT_DE}\n\n"
        f"{audience_prompt}\n\n"
        "Antworte ausschließlich als JSON-Objekt mit den Schlüsseln 'narrative' und 'structured'.\n"
        f"Kontext (ECP v1): {context_json}"
    )


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    for idx, char in enumerate(raw_text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end < 0:
        return None

    try:
        return json.loads(raw_text[start : end + 1])
    except Exception:
        return None


def validate_dual_response_shape(response: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(response, dict):
        return False, "response is not object"

    narrative = response.get("narrative")
    structured = response.get("structured")
    if not isinstance(narrative, dict) or not isinstance(structured, dict):
        return False, "missing narrative/structured"

    narrative_keys = {"one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis"}
    if not narrative_keys.issubset(narrative.keys()):
        return False, "narrative keys missing"

    structured_keys = {
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
    }
    if not structured_keys.issubset(structured.keys()):
        return False, "structured keys missing"

    evidence_refs = structured.get("evidence_refs", [])
    if not isinstance(evidence_refs, list) or not evidence_refs:
        return False, "evidence_refs missing"

    for driver in structured.get("top_drivers", []):
        if not isinstance(driver, dict) or not driver.get("evidence_ref"):
            return False, "driver missing evidence_ref"

    for action in structured.get("recommended_actions", []):
        if not isinstance(action, dict) or not action.get("evidence_ref"):
            return False, "action missing evidence_ref"

    uncertainty = structured.get("uncertainty", {})
    if not isinstance(uncertainty, dict) or not uncertainty.get("evidence_ref"):
        return False, "uncertainty missing evidence_ref"

    if structured.get("audience") == "professor":
        actions = structured.get("recommended_actions", [])
        if not isinstance(actions, list) or len(actions) < 3:
            return False, "professor audience requires at least 3 actions"

    return True, None


def render_dual_output_markdown(response: dict[str, Any]) -> str:
    narrative = response.get("narrative", {}) if isinstance(response.get("narrative"), dict) else {}
    structured = response.get("structured", {}) if isinstance(response.get("structured"), dict) else {}

    lines: list[str] = []
    one_liner = str(narrative.get("one_liner", "")).strip() or "Keine Kurzbewertung verfuegbar."
    lines.append("**Kurzfazit**")
    lines.append(one_liner)

    warum = str(narrative.get("warum", "")).strip() or "Keine belastbare Ursachenbeschreibung verfuegbar."
    lines.append("**Warum ist die Auslastung so?**")
    lines.append(warum)

    unsicherheit = str(narrative.get("unsicherheit", "")).strip() or "Unsicherheit konnte nicht klar bestimmt werden."
    lines.append("**Was ist unsicher?**")
    lines.append(unsicherheit)

    lines.append("**Was verbessern wir als Naechstes?**")
    actions = structured.get("recommended_actions", [])
    if isinstance(actions, list) and actions:
        for idx, action in enumerate(actions[:3], start=1):
            if not isinstance(action, dict):
                continue
            lines.append(
                f"{idx}. {str(action.get('action_type', 'monitor'))}: "
                f"{str(action.get('rationale') or 'Keine Begruendung')}"
            )
    else:
        lines.append("1. Weiter beobachten und bei Aenderungen neu bewerten.")

    verdict = str(structured.get("verdict", "")).strip() or "n/a"
    confidence = str(structured.get("confidence_statement", "")).strip() or "n/a"
    lines.append("**Werte kompakt**")
    lines.append(f"- Urteil: {verdict}")
    lines.append(f"- Verlaesslichkeit: {confidence}")
    evidence = str(narrative.get("evidence_hinweis", "")).strip()
    if evidence:
        lines.append(f"- Evidenz: {evidence}")

    if not lines:
        return "Keine verständliche Narrative verfügbar."
    return "\n\n".join(lines)
