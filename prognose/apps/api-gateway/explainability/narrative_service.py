from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from .prompt_registry import PromptRegistry

logger = logging.getLogger("sitcheck.explainability")


NARRATIVE_FIELDS = ("one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis")
FORECAST_INTENT_KEYWORDS = (
    "zone",
    "auslast",
    "prognose",
    "forecast",
    "horizon",
    "horizont",
    "peak",
    "trend",
    "risiko",
    "belegung",
    "occupancy",
    "unsicherheit",
    "kapaz",
    "massnahme",
    "maßnahme",
    "driver",
    "treiber",
)
META_INTENT_KEYWORDS = (
    "wer bist du",
    "als modell",
    "als model",
    "systemprompt",
    "system prompt",
    "template",
    "vorlage",
    "prompt",
    "welches modell",
    "welches model",
    "ollama",
    "gemma",
    "llm",
)
OFF_TOPIC_INTENT_KEYWORDS = (
    "kuchen",
    "rezept",
    "koch",
    "wetter",
    "urlaub",
    "witz",
    "song",
    "lyrics",
    "poem",
    "gedicht",
)
COMPACT_REQUEST_KEYWORDS = (
    "kurz",
    "kompakt",
    "tl;dr",
    "stichpunkt",
    "knapp",
    "in 2 saetzen",
    "in 2 sätzen",
)
KPI_REQUEST_KEYWORDS = (
    "kpi",
    "zahl",
    "werte",
    "wert",
    "trend",
    "peak",
    "auslast",
    "prognose",
    "forecast",
    "horizon",
    "horizont",
    "risiko",
    "personen",
    "belegung",
    "occupancy",
)
IMPROVEMENT_FOCUS_KEYWORDS = (
    "verbesserung",
    "massnahme",
    "maßnahme",
    "naechster schritt",
    "nächster schritt",
    "aufwand",
    "wirkung",
)
RISK_FOCUS_KEYWORDS = (
    "risiko",
    "unsicher",
    "uncertainty",
    "umgehen",
    "mitigation",
)
BRIEFING_FOCUS_KEYWORDS = (
    "brief",
    "briefing",
    "executive",
    "professor",
    "student",
    "studierende",
)
STATUS_FOCUS_KEYWORDS = (
    "aktuell",
    "status",
    "2 minuten",
    "lage",
    "kurz",
)
STUDENT_FOCUS_KEYWORDS = (
    "studierende",
    "studier",
    "studentensicht",
    "studenten",
    "student",
    "einfacher sprache",
    "einfach erklären",
)


class LLMQualityGateError(RuntimeError):
    def __init__(self, *, min_coverage: int, actual_coverage: int, query_intent: str):
        self.min_coverage = int(min_coverage)
        self.actual_coverage = int(actual_coverage)
        self.query_intent = str(query_intent)
        super().__init__(
            "llm_quality_gate_failed: "
            f"actual_coverage={self.actual_coverage} < min_coverage={self.min_coverage}"
        )


class LLMNarrativeUnavailableError(RuntimeError):
    def __init__(self, *, reason: str | None, query_intent: str):
        self.reason = str(reason or "unknown")
        self.query_intent = str(query_intent)
        super().__init__(f"llm_unavailable:{self.reason}")


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


def _first_nonempty_ref(citation_map: list[dict[str, Any]], preferred_types: list[str]) -> str:
    for source_type in preferred_types:
        for item in citation_map:
            if str(item.get("source_type", "")) == source_type:
                return str(item.get("ref_id"))
    if citation_map:
        return str(citation_map[0].get("ref_id"))
    return "ref-1"


def _confidence_statement(level: str, score: float) -> str:
    if level == "high":
        return f"Niedrige Verlässlichkeit (Unsicherheit={score:.2f}). Aussagen nur als Tendenz nutzen."
    if level == "medium":
        return f"Mittlere Verlässlichkeit (Unsicherheit={score:.2f}). Entscheidungen mit Monitoring absichern."
    return f"Hohe Verlässlichkeit (Unsicherheit={score:.2f})."


def _forecast_quality_flags(context: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    for item in context.get("citation_map", []) if isinstance(context.get("citation_map"), list) else []:
        if not isinstance(item, dict):
            continue
        for flag in item.get("quality_flags", []) if isinstance(item.get("quality_flags"), list) else []:
            if isinstance(flag, str):
                flags.add(flag.upper())
    return flags


def _forecast_alert_codes(context: dict[str, Any]) -> set[str]:
    quality = context.get("quality_digest", {}) if isinstance(context.get("quality_digest"), dict) else {}
    codes: set[str] = set()
    for alert in quality.get("alerts", []) if isinstance(quality.get("alerts"), list) else []:
        if isinstance(alert, dict) and isinstance(alert.get("code"), str):
            codes.add(str(alert["code"]).upper())
    return codes


def forecast_degradation_status(context: dict[str, Any]) -> dict[str, Any]:
    forecast = context.get("forecast_snapshot", {}) if isinstance(context.get("forecast_snapshot"), dict) else {}
    quality = context.get("quality_digest", {}) if isinstance(context.get("quality_digest"), dict) else {}
    model_version = str(forecast.get("model_version") or "unknown")
    summary = str(forecast.get("summary") or "")
    flags = _forecast_quality_flags(context)
    alert_codes = _forecast_alert_codes(context)

    model_fallback = (
        "baseline" in model_version.lower()
        or "fallback" in summary.lower()
        or any(flag == "LGBM_FALLBACK" or flag.startswith("LGBM_FALLBACK:") for flag in flags)
        or any(flag == "TF_FALLBACK" or flag.startswith("TF_FALLBACK:") for flag in flags)
        or "BASELINE_FALLBACK" in alert_codes
    )
    context_degraded = (
        bool(forecast.get("stale"))
        or "CONTEXT_STALE" in flags
        or "NO_CONTEXT_DB" in flags
        or "FORECAST_CONTEXT_DEGRADED" in alert_codes
    )
    quality_blocked = "QUALITY_RISK" in alert_codes or _safe_float(quality.get("history_quality_score_avg", 1.0), 1.0) < 0.7

    reasons: list[str] = []
    if model_fallback:
        reasons.append(f"Baseline/Fallback aktiv ({model_version})")
    if context_degraded:
        reasons.append("Forecast-Kontext veraltet oder fehlt")
    if quality_blocked:
        reasons.append("Datenqualitaet blockiert harte Massnahmen")

    return {
        "degraded": bool(reasons),
        "model_fallback": model_fallback,
        "context_degraded": context_degraded,
        "quality_blocked": quality_blocked,
        "model_version": model_version,
        "flags": sorted(flags),
        "alert_codes": sorted(alert_codes),
        "reasons": reasons,
    }


def _append_degradation_block(lines: list[str], context: dict[str, Any]) -> None:
    status = forecast_degradation_status(context)
    if not status["degraded"]:
        return
    lines.append("**Belastbarkeit**")
    lines.append(
        "Die Prognose ist degradiert und nicht normal belastbar: "
        + "; ".join(status["reasons"])
        + "."
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

    return True, None


def _canonical_key(key: Any) -> str:
    raw = str(key or "").lower()
    return "".join(ch for ch in raw if ch.isalnum())


def _trim_text(value: Any, max_chars: int = 260) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _keyword_hits(query: str, keywords: tuple[str, ...]) -> int:
    if not query:
        return 0
    return sum(1 for keyword in keywords if keyword in query)


def _word_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9äöüÄÖÜß]+", _normalize_query(text))
        if len(token) >= 3
    }


def _looks_like_prompt_echo(candidate: str, query: str) -> bool:
    cand = _normalize_query(candidate)
    qry = _normalize_query(query)
    if not cand or not qry:
        return False
    if cand == qry or cand in qry or qry in cand:
        return True
    if cand.startswith(("erklaere ", "erkläre ", "schreibe ", "welche ", "was sind ", "ich bereite ")):
        return True

    cand_tokens = _word_tokens(cand)
    qry_tokens = _word_tokens(qry)
    if len(cand_tokens) < 4 or len(qry_tokens) < 4:
        return False
    overlap = len(cand_tokens & qry_tokens) / max(1, len(cand_tokens))
    return overlap >= 0.7


def classify_query_intent(query: str) -> str:
    normalized = _normalize_query(query)
    if not normalized:
        return "forecast"

    forecast_hits = _keyword_hits(normalized, FORECAST_INTENT_KEYWORDS)
    meta_hits = _keyword_hits(normalized, META_INTENT_KEYWORDS)
    off_topic_hits = _keyword_hits(normalized, OFF_TOPIC_INTENT_KEYWORDS)

    # Product decision: forecast keywords always win over meta/off-topic markers.
    if forecast_hits > 0:
        return "forecast"
    if meta_hits > 0 and off_topic_hits == 0:
        return "meta"
    if off_topic_hits > 0:
        return "off_topic"
    if meta_hits > 0:
        return "meta"
    return "forecast"


def classify_query_focus(query: str) -> str:
    normalized = _normalize_query(query)
    if not normalized:
        return "general"

    scores = {
        "student": _keyword_hits(normalized, STUDENT_FOCUS_KEYWORDS),
        "improvement": _keyword_hits(normalized, IMPROVEMENT_FOCUS_KEYWORDS),
        "risk": _keyword_hits(normalized, RISK_FOCUS_KEYWORDS),
        "briefing": _keyword_hits(normalized, BRIEFING_FOCUS_KEYWORDS),
        "status": _keyword_hits(normalized, STATUS_FOCUS_KEYWORDS),
    }
    best_label = max(scores, key=scores.get)
    if scores.get(best_label, 0) <= 0:
        return "general"

    # Deterministic tie-break for mixed queries.
    priority = ("student", "improvement", "risk", "briefing", "status")
    top_score = scores[best_label]
    tied = [label for label, value in scores.items() if value == top_score]
    for label in priority:
        if label in tied:
            return label
    return best_label


def derive_answer_profile(query: str, query_intent: str, query_focus: str = "general") -> str:
    normalized = _normalize_query(query)
    if query_intent == "off_topic":
        return "offtopic_redirect"
    if query_intent == "meta":
        return "forecast_compact"
    if query_focus == "student":
        return "forecast_student"
    if query_focus == "status":
        return "forecast_compact"
    if query_focus == "improvement":
        return "forecast_actions"
    if query_focus == "risk":
        return "forecast_risk"
    if query_focus == "briefing":
        return "forecast_brief"
    if any(keyword in normalized for keyword in COMPACT_REQUEST_KEYWORDS):
        return "forecast_compact"
    return "forecast_full"


def should_include_kpi_block(query: str, query_intent: str, query_focus: str = "general") -> bool:
    if query_intent != "forecast":
        return False
    normalized = _normalize_query(query)
    if not normalized:
        return True
    if query_focus == "student":
        # Keep student answers simpler unless numbers are explicitly requested.
        return any(
            keyword in normalized
            for keyword in ("werte", "zahl", "zahlen", "kpi", "prozent", "%", "personen", "trend", "peak")
        )
    if query_focus in {"status", "risk"}:
        return True
    if re.search(r"\b\d+\s*(min|minute|minuten|h|std|stunde|stunden|hour|hours)\b", normalized):
        return True
    if re.search(r"\b(horizon|horizont)\s*\d+\b", normalized):
        return True
    return any(keyword in normalized for keyword in KPI_REQUEST_KEYWORDS)


def llm_coverage_fields(source_map: dict[str, str]) -> list[str]:
    return [field for field in NARRATIVE_FIELDS if source_map.get(field) == "llm"]


def missing_llm_fields(source_map: dict[str, str]) -> list[str]:
    return [field for field in NARRATIVE_FIELDS if source_map.get(field) != "llm"]


def build_hybrid_narrative_payload(
    *,
    response: dict[str, Any] | None,
    raw_text: str,
    fallback_narrative: dict[str, Any],
    evidence_refs: list[str] | None = None,
) -> tuple[dict[str, str] | None, dict[str, str]]:
    base = {
        "one_liner": _trim_text(fallback_narrative.get("one_liner"), max_chars=220),
        "warum": _trim_text(fallback_narrative.get("warum"), max_chars=260),
        "unsicherheit": _trim_text(fallback_narrative.get("unsicherheit"), max_chars=220),
        "empfehlung": _trim_text(fallback_narrative.get("empfehlung"), max_chars=220),
        "evidence_hinweis": _trim_text(fallback_narrative.get("evidence_hinweis"), max_chars=220),
    }

    if isinstance(response, dict) and isinstance(response.get("narrative"), dict):
        candidate = response.get("narrative", {})
    elif isinstance(response, dict):
        candidate = response
    else:
        candidate = {}

    text_by_key: dict[str, str] = {}
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                text_by_key[_canonical_key(key)] = _trim_text(value)

    aliases = {
        "one_liner": {"oneliner", "summary", "kurzfazit", "fazit"},
        "warum": {"warum", "why", "reason", "treiber"},
        "unsicherheit": {"unsicherheit", "uncertainty", "risiko", "risk"},
        "empfehlung": {"empfehlung", "recommendation", "action", "naechsterschritt", "nextstep"},
        "evidence_hinweis": {"evidencehinweis", "evidencehint", "evidence", "refs", "referenzen"},
    }

    source_map = {
        "one_liner": "fallback",
        "warum": "fallback",
        "unsicherheit": "fallback",
        "empfehlung": "fallback",
        "evidence_hinweis": "fallback",
    }
    merged = dict(base)
    for field, keys in aliases.items():
        for key in keys:
            if key in text_by_key:
                merged[field] = text_by_key[key]
                source_map[field] = "llm"
                break

    if raw_text.strip():
        regex_aliases = {
            "one_liner": ("one_liner", "summary", "kurzfazit", "fazit"),
            "warum": ("warum", "why", "reason", "treiber"),
            "unsicherheit": ("unsicherheit", "uncertainty", "risiko", "risk"),
            "empfehlung": ("empfehlung", "recommendation", "action", "nextstep"),
            "evidence_hinweis": ("evidence_hinweis", "evidence_hint", "evidence", "refs", "referenzen"),
        }
        for field, keys in regex_aliases.items():
            if source_map.get(field) == "llm":
                continue
            for key in keys:
                match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', raw_text, flags=re.IGNORECASE)
                if match:
                    merged[field] = _trim_text(match.group(1))
                    source_map[field] = "llm"
                    break

    if raw_text.strip() and source_map.get("one_liner") != "llm":
        merged["one_liner"] = _trim_text(raw_text.replace("\n", " "), max_chars=220)
        source_map["one_liner"] = "llm"

    refs = [ref for ref in (evidence_refs or []) if isinstance(ref, str) and ref]
    if refs and "ref:" not in merged["evidence_hinweis"]:
        merged["evidence_hinweis"] = f"{merged['evidence_hinweis']} [ref:{refs[0]}]"

    if all(str(merged.get(key, "")).strip() for key in ("one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis")):
        return merged, source_map
    return None, source_map


def _as_percent(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.1f}%"


def _as_number(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.1f}"


def apply_degradation_guardrail(
    *,
    response_payload: dict[str, Any],
    context: dict[str, Any],
    narrative_source_map: dict[str, str],
) -> bool:
    status = forecast_degradation_status(context)
    if not status["degraded"]:
        return False

    request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
    occupancy_explainer = (
        context.get("occupancy_explainer", {})
        if isinstance(context.get("occupancy_explainer"), dict)
        else {}
    )
    uncertainty = context.get("uncertainty", {}) if isinstance(context.get("uncertainty"), dict) else {}
    driver_summary = context.get("driver_summary", {}) if isinstance(context.get("driver_summary"), dict) else {}
    citation_map = list(context.get("citation_map", [])) if isinstance(context.get("citation_map"), list) else []
    zone_id = str(request_meta.get("zone_id") or "unknown-zone")
    current_occ = _safe_int(occupancy_explainer.get("current_occupancy", 0), 0)
    peak_60 = _as_number(occupancy_explainer.get("forecast_next_60m_peak", 0.0))
    risk_level = str(occupancy_explainer.get("risk_level") or "unknown")
    uncertainty_level = str(uncertainty.get("level") or "unknown")
    uncertainty_score = _safe_float(uncertainty.get("score", 0.0), 0.0)
    evidence_ref = _first_nonempty_ref(citation_map, ["forecast", "counts", "xai", "api"])
    uncertainty_ref = str(uncertainty.get("evidence_ref") or _first_nonempty_ref(citation_map, ["xai", "forecast", "counts"]))
    drivers = [
        str(driver.get("name"))
        for driver in driver_summary.get("top_drivers", [])
        if isinstance(driver, dict) and driver.get("name")
    ][:3]
    reason_text = "; ".join(status["reasons"])

    response_payload["narrative"] = {
        "one_liner": (
            f"Zone {zone_id}: aktuell {current_occ} Personen, erwarteter 60-Minuten-Peak {peak_60}, "
            f"Auslastungsrisiko {risk_level}; die Prognose ist degradiert und nicht normal belastbar."
        ),
        "warum": (
            f"Wichtige Treiber sind {', '.join(drivers) if drivers else 'keine stabilen Treiber'}. "
            f"Die Entscheidbarkeit ist eingeschraenkt: {reason_text} [ref:{evidence_ref}]."
        ),
        "unsicherheit": (
            f"Die Unsicherheit ist {uncertainty_level} (Score {uncertainty_score:.2f}); "
            f"die Aussage darf nur als Tendenz genutzt werden [ref:{uncertainty_ref}]."
        ),
        "empfehlung": (
            "Keine harte operative Massnahme aus der Prognose ableiten; zuerst Datenqualitaet, Sensorik und "
            f"Ingest-Pipeline pruefen und eng monitoren [ref:{evidence_ref}]."
        ),
        "evidence_hinweis": f"Evidenz: [ref:{evidence_ref}], [ref:{uncertainty_ref}]. Modell: {status['model_version']}.",
    }

    structured = response_payload.get("structured", {}) if isinstance(response_payload.get("structured"), dict) else {}
    structured["verdict"] = "blocked"
    structured["recommended_actions"] = [
        {
            "action_type": "quality-hardening" if status["quality_blocked"] else "monitor",
            "priority": 1,
            "rationale": "Degradierte Prognose: Datenqualitaet/Modellpfad pruefen und keine harte Massnahme ableiten.",
            "evidence_ref": evidence_ref,
        }
    ]
    limitations = [str(item) for item in structured.get("limitations", []) if isinstance(item, str)]
    for reason in status["reasons"]:
        if reason not in limitations:
            limitations.append(reason)
    structured["limitations"] = limitations or ["Prognose ist degradiert und nicht normal belastbar."]
    refs = [ref for ref in structured.get("evidence_refs", []) if isinstance(ref, str)]
    for ref in (evidence_ref, uncertainty_ref):
        if ref and ref not in refs:
            refs.append(ref)
    structured["evidence_refs"] = refs or [evidence_ref]
    structured["confidence_statement"] = _confidence_statement(uncertainty_level, uncertainty_score)
    response_payload["structured"] = structured

    for field in NARRATIVE_FIELDS:
        narrative_source_map[field] = "guardrail"
    return True


def apply_non_degraded_action_guardrail(
    *,
    response_payload: dict[str, Any],
    context: dict[str, Any],
    narrative_source_map: dict[str, str],
) -> bool:
    status = forecast_degradation_status(context)
    if status["degraded"]:
        return False

    structured = response_payload.get("structured", {}) if isinstance(response_payload.get("structured"), dict) else {}
    actions = structured.get("recommended_actions", []) if isinstance(structured.get("recommended_actions"), list) else []

    def _is_quality_action(action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        text = (str(action.get("action_type") or "") + " " + str(action.get("rationale") or "")).lower()
        return any(token in text for token in ("quality", "datenqual", "sensor", "ingest", "pipeline"))

    limitations = [str(item) for item in structured.get("limitations", []) if isinstance(item, str)]
    filtered_limitations = [
        item
        for item in limitations
        if not any(token in item.lower() for token in ("quality", "datenqual", "sensor", "ingest", "pipeline"))
    ]
    changed = filtered_limitations != limitations
    if not filtered_limitations:
        filtered_limitations = ["Keine kritischen Einschränkungen erkannt."]
    structured["limitations"] = filtered_limitations

    if any(_is_quality_action(action) for action in actions):
        recommendation = context.get("recommendation_digest", {}) if isinstance(context.get("recommendation_digest"), dict) else {}
        citation_map = list(context.get("citation_map", [])) if isinstance(context.get("citation_map"), list) else []
        evidence_ref = str(
            recommendation.get("evidence_ref")
            or _first_nonempty_ref(citation_map, ["recommendation", "forecast", "counts", "api"])
        )
        structured["recommended_actions"] = [
            {
                "action_type": "monitor",
                "priority": 1,
                "rationale": "H15-LGBM ist aktiv; Lage weiter beobachten und den nächsten Snapshot gegen Ist-Daten prüfen.",
                "evidence_ref": evidence_ref,
            }
        ]
        narrative = response_payload.get("narrative", {}) if isinstance(response_payload.get("narrative"), dict) else {}
        narrative["empfehlung"] = (
            f"H15-LGBM ist aktiv; Lage weiter beobachten und den nächsten Snapshot gegen Ist-Daten prüfen [ref:{evidence_ref}]."
        )
        response_payload["narrative"] = narrative
        narrative_source_map["empfehlung"] = "guardrail"
        changed = True

    narrative = response_payload.get("narrative", {}) if isinstance(response_payload.get("narrative"), dict) else {}
    bad_tokens = ("stale", "degrad", "datenqual", "sensor", "ingest", "pipeline", "nicht normal belastbar")
    if any(
        any(token in str(narrative.get(field) or "").lower() for token in bad_tokens)
        for field in NARRATIVE_FIELDS
    ):
        request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
        forecast = context.get("forecast_snapshot", {}) if isinstance(context.get("forecast_snapshot"), dict) else {}
        recommendation = context.get("recommendation_digest", {}) if isinstance(context.get("recommendation_digest"), dict) else {}
        citation_map = list(context.get("citation_map", [])) if isinstance(context.get("citation_map"), list) else []
        evidence_ref = str(
            recommendation.get("evidence_ref")
            or _first_nonempty_ref(citation_map, ["recommendation", "forecast", "counts", "api"])
        )
        zone_id = str(request_meta.get("zone_id") or "unknown-zone")
        horizon = _safe_int(request_meta.get("horizon", 15), 15)
        model_version = str(forecast.get("model_version") or status["model_version"])
        narrative.update(
            {
                "one_liner": f"Zone {zone_id}: H{horizon}-LGBM ist aktiv und die Prognose ist verwendbar.",
                "warum": f"Der Forecast nutzt das validierte Modell {model_version}; es liegen keine kritischen Einschränkungen vor [ref:{evidence_ref}].",
                "unsicherheit": "Die Unsicherheit ist niedrig bis moderat; die Aussage ist als operative Tendenz nutzbar.",
                "empfehlung": f"Lage weiter beobachten und den nächsten H{horizon}-Snapshot gegen Ist-Daten prüfen [ref:{evidence_ref}].",
                "evidence_hinweis": f"Evidenz: [ref:{evidence_ref}]. Modell: {model_version}.",
            }
        )
        response_payload["narrative"] = narrative
        for field in NARRATIVE_FIELDS:
            narrative_source_map[field] = "guardrail"
        changed = True

    if structured.get("verdict") == "blocked":
        structured["verdict"] = "monitor"
        changed = True

    response_payload["structured"] = structured
    return changed


def _build_context_one_liner(context: dict[str, Any], *, style: str = "standard") -> str:
    request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
    occupancy_explainer = (
        context.get("occupancy_explainer", {})
        if isinstance(context.get("occupancy_explainer"), dict)
        else {}
    )
    zone_id = str(request_meta.get("zone_id") or "unknown-zone")
    current_occ = _safe_int(occupancy_explainer.get("current_occupancy", 0), 0)
    peak_60 = _as_number(occupancy_explainer.get("forecast_next_60m_peak", 0.0))
    risk = str(occupancy_explainer.get("risk_level") or "unknown")
    util_now = _as_percent(context.get("utilization_now_pct", 0.0))

    if style == "student":
        return (
            f"Aktuell sind in Zone {zone_id} {current_occ} Personen. "
            f"In den naechsten 60 Minuten erwarten wir maximal {peak_60} Personen. "
            f"Das Risiko ist {risk}."
        )
    return (
        f"Zone {zone_id}: aktuell {current_occ} Personen ({util_now} Auslastung), "
        f"naechster 60m-Peak {peak_60}, Risiko {risk}."
    )


def _build_student_reason(context: dict[str, Any]) -> str:
    occupancy_explainer = (
        context.get("occupancy_explainer", {})
        if isinstance(context.get("occupancy_explainer"), dict)
        else {}
    )
    current_occ = _safe_int(occupancy_explainer.get("current_occupancy", 0), 0)
    peak_60 = _as_number(occupancy_explainer.get("forecast_next_60m_peak", 0.0))
    trend_15m = _as_number(occupancy_explainer.get("trend_15m", 0.0))
    return (
        f"Aktuell sind es {current_occ} Personen. "
        f"Der Trend der letzten 15 Minuten liegt bei {trend_15m} und der erwartete Peak bei {peak_60}."
    )


def _build_student_next_step(context: dict[str, Any]) -> str:
    improvements = (
        context.get("improvement_candidates", [])
        if isinstance(context.get("improvement_candidates"), list)
        else []
    )
    for item in improvements:
        if isinstance(item, dict) and str(item.get("measure_text") or "").strip():
            return str(item.get("measure_text")).strip()
    return "Behalte die Lage im Blick und pruefe in 15 Minuten erneut."


def _build_student_uncertainty(context: dict[str, Any], structured: dict[str, Any]) -> str:
    uncertainty = structured.get("uncertainty", {}) if isinstance(structured.get("uncertainty"), dict) else {}
    level = str(uncertainty.get("level") or "high")
    score = _safe_float(uncertainty.get("score", 0.0), 0.0)
    return f"Die Unsicherheit ist {level} (Score {score:.2f}). Nutze die Aussage als Tendenz, nicht als fixe Zusage."


def _build_measure_lines(
    *,
    structured: dict[str, Any],
    improvements: list[dict[str, Any]],
) -> list[str]:
    measure_lines: list[str] = []
    if improvements:
        for idx, item in enumerate(improvements[:3], start=1):
            if not isinstance(item, dict):
                continue
            measure_lines.append(
                f"{idx}. {str(item.get('measure_text') or 'Massnahme')} "
                f"(Aufwand: {str(item.get('effort') or 'low')}, "
                f"Erwartete Wirkung: {str(item.get('expected_effect') or 'n/a')}, "
                f"Naechster Schritt: {str(item.get('owner_hint') or 'Team abstimmen')}, "
                f"Evidenz: [ref:{str(item.get('evidence_ref') or 'ref-1')}])."
            )
    if measure_lines:
        return measure_lines

    actions = structured.get("recommended_actions", []) if isinstance(structured.get("recommended_actions"), list) else []
    for idx, action in enumerate(actions[:3], start=1):
        if not isinstance(action, dict):
            continue
        measure_lines.append(
            f"{idx}. {str(action.get('action_type', 'monitor'))}: "
            f"{str(action.get('rationale') or 'Keine Begruendung verfuegbar')} "
            f"[ref:{str(action.get('evidence_ref') or 'ref-1')}]."
        )
    if measure_lines:
        return measure_lines

    return ["1. Lage weiter beobachten und in 15 Minuten erneut bewerten."]


def _append_kpi_block(lines: list[str], *, context: dict[str, Any], structured: dict[str, Any]) -> None:
    occupancy_explainer = (
        context.get("occupancy_explainer", {})
        if isinstance(context.get("occupancy_explainer"), dict)
        else {}
    )
    forecast_snapshot = (
        context.get("forecast_snapshot", {})
        if isinstance(context.get("forecast_snapshot"), dict)
        else {}
    )

    verdict = str(structured.get("verdict", "")).strip()
    confidence = str(structured.get("confidence_statement", "")).strip()
    lines.append("**Werte kompakt**")
    lines.append(f"- Personen aktuell: {_safe_int(occupancy_explainer.get('current_occupancy', 0), 0)}")
    lines.append(f"- Auslastung aktuell: {_as_percent(context.get('utilization_now_pct', 0.0))}")
    lines.append(f"- 60m-Mittel: {_as_number(occupancy_explainer.get('avg_60m', 0.0))}")
    lines.append(f"- Trend 15m: {_as_number(occupancy_explainer.get('trend_15m', 0.0))}")
    lines.append(f"- Forecast-Peak 60m: {_as_number(occupancy_explainer.get('forecast_next_60m_peak', 0.0))}")
    lines.append(f"- Risiko: {str(occupancy_explainer.get('risk_level') or 'unknown')}")
    lines.append(f"- Urteil: {verdict or 'n/a'}")
    lines.append(f"- Modell: {str(forecast_snapshot.get('model_version') or 'unknown')}")
    lines.append(f"- Quelle: {str(forecast_snapshot.get('source') or 'unknown')}")
    lines.append(f"- Verlaesslichkeit: {confidence or 'n/a'}")


def render_dual_output_markdown(
    response: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    answer_profile: str = "forecast_full",
    include_kpi_block: bool = True,
    query_intent: str = "forecast",
    query_focus: str = "general",
) -> str:
    narrative = response.get("narrative", {}) if isinstance(response.get("narrative"), dict) else {}
    structured = response.get("structured", {}) if isinstance(response.get("structured"), dict) else {}
    context = context or {}
    occupancy_explainer = context.get("occupancy_explainer", {}) if isinstance(context.get("occupancy_explainer"), dict) else {}
    improvements = context.get("improvement_candidates", []) if isinstance(context.get("improvement_candidates"), list) else []

    lines: list[str] = []
    one_liner = str(narrative.get("one_liner", "")).strip()
    if not one_liner:
        verdict = str(structured.get("verdict", "monitor"))
        one_liner = f"Aktuelle Lageeinschaetzung: {verdict}."

    warum = str(narrative.get("warum", "")).strip()
    if not warum:
        drivers = structured.get("top_drivers", [])
        driver_names = ", ".join(
            str(driver.get("name", "unknown"))
            for driver in drivers[:3]
            if isinstance(driver, dict)
        )
        current_occ = _safe_int(occupancy_explainer.get("current_occupancy", 0), 0)
        trend_15m = _safe_float(occupancy_explainer.get("trend_15m", 0.0), 0.0)
        forecast_peak = _safe_float(occupancy_explainer.get("forecast_next_60m_peak", 0.0), 0.0)
        warum = (
            f"Aktuell liegen wir bei {current_occ} Personen. "
            f"Der 15-Minuten-Trend liegt bei {trend_15m:+.1f} und der naechste 60-Minuten-Peak bei {forecast_peak:.1f}. "
            f"Wesentliche Treiber: {driver_names or 'keine belastbaren Treiber'}."
        )

    unsicherheit = str(narrative.get("unsicherheit", "")).strip()
    if not unsicherheit:
        uncertainty = structured.get("uncertainty", {}) if isinstance(structured.get("uncertainty"), dict) else {}
        level = str(uncertainty.get("level", "high"))
        score = _safe_float(uncertainty.get("score", 0.8), 0.8)
        limitations = structured.get("limitations", []) if isinstance(structured.get("limitations"), list) else []
        unsicherheit = (
            f"Unsicherheit ist {level} (Score {score:.2f}). "
            f"Beachte insbesondere: {', '.join(str(item) for item in limitations[:2]) or 'keine besonderen Einschraenkungen'}."
        )

    empfehlung = str(narrative.get("empfehlung", "")).strip()
    if not empfehlung:
        action = None
        for item in structured.get("recommended_actions", []) if isinstance(structured.get("recommended_actions"), list) else []:
            if isinstance(item, dict):
                action = item
                break
        if isinstance(action, dict):
            empfehlung = (
                f"{str(action.get('action_type') or 'monitor')}: "
                f"{str(action.get('rationale') or 'keine Begruendung verfuegbar')} "
                f"[ref:{str(action.get('evidence_ref') or 'ref-1')}]."
            )
        else:
            empfehlung = "Lage weiter beobachten und kurzfristig neu bewerten."

    if answer_profile == "offtopic_redirect":
        lines.append("**Kurzantwort**")
        lines.append(one_liner)
        lines.append("**Hinweis zum Scope**")
        lines.append(warum or "Ich beantworte belastbar Fragen zu Prognose, Auslastung und Risiken je Zone.")
        lines.append("**Naechste sinnvolle Forecast-Frage**")
        lines.append(empfehlung or "Wie entwickelt sich die Auslastung fuer diese Zone in den naechsten 60 Minuten?")
        evidence_hint = str(narrative.get("evidence_hinweis", "")).strip()
        if evidence_hint:
            lines.append("**Evidenzhinweis**")
            lines.append(evidence_hint)
    elif answer_profile == "forecast_compact":
        lines.append("**Kurzfazit**" if query_intent == "forecast" else "**Antwort**")
        lines.append(one_liner)
        if query_intent == "forecast":
            _append_degradation_block(lines, context)
            if warum:
                lines.append("**Einordnung**")
                lines.append(warum)
            if unsicherheit:
                lines.append("**Unsicherheit**")
                lines.append(unsicherheit)
            lines.append("**Naechster Schritt**")
            lines.append(empfehlung)
        else:
            lines.append("**Scope**")
            lines.append(warum)
            lines.append("**Forecast-Weiterleitung**")
            lines.append(empfehlung)
        if include_kpi_block:
            _append_kpi_block(lines, context=context, structured=structured)
    elif answer_profile == "forecast_student":
        lines.append("**Kurzfazit (einfach)**")
        lines.append(one_liner)
        _append_degradation_block(lines, context)
        lines.append("**Warum ist das so?**")
        lines.append(warum)
        lines.append("**Was solltest du als Naechstes tun?**")
        lines.append(empfehlung)
        if unsicherheit:
            lines.append("**Unsicherheit (kurz)**")
            lines.append(unsicherheit)
        if include_kpi_block:
            occupancy_explainer_local = (
                context.get("occupancy_explainer", {})
                if isinstance(context.get("occupancy_explainer"), dict)
                else {}
            )
            lines.append("**Werte einfach**")
            lines.append(f"- Aktuell: {_safe_int(occupancy_explainer_local.get('current_occupancy', 0), 0)} Personen")
            lines.append(f"- Naechste 60 Minuten (Peak): {_as_number(occupancy_explainer_local.get('forecast_next_60m_peak', 0.0))}")
            lines.append(f"- Risiko: {str(occupancy_explainer_local.get('risk_level') or 'unknown')}")
    elif answer_profile == "forecast_actions":
        lines.append("**Kurzfazit**")
        lines.append(one_liner)
        _append_degradation_block(lines, context)
        lines.append("**Priorisierte Massnahmen**")
        lines.extend(_build_measure_lines(structured=structured, improvements=improvements))
        if warum:
            lines.append("**Warum diese Priorisierung?**")
            lines.append(warum)
        if unsicherheit:
            lines.append("**Umsetzungsrisiken**")
            lines.append(unsicherheit)
        if include_kpi_block:
            _append_kpi_block(lines, context=context, structured=structured)
    elif answer_profile == "forecast_risk":
        lines.append("**Kurzfazit**")
        lines.append(one_liner)
        _append_degradation_block(lines, context)
        lines.append("**Zentrale Risiken/Unsicherheit**")
        lines.append(unsicherheit)
        lines.append("**Empfohlener Umgang**")
        lines.append(empfehlung)
        limitations = structured.get("limitations", []) if isinstance(structured.get("limitations"), list) else []
        if limitations:
            lines.append("**Worauf wir achten sollten**")
            for item in limitations[:3]:
                lines.append(f"- {str(item)}")
        if include_kpi_block:
            _append_kpi_block(lines, context=context, structured=structured)
    elif answer_profile == "forecast_brief":
        heading = "**Briefing**" if query_focus == "briefing" else "**Kurzfazit**"
        lines.append(heading)
        lines.append(one_liner)
        _append_degradation_block(lines, context)
        lines.append("**Einordnung**")
        lines.append(warum)
        lines.append("**Risiko/Unsicherheit**")
        lines.append(unsicherheit)
        lines.append("**Konkreter naechster Schritt**")
        lines.append(empfehlung)
        if include_kpi_block:
            _append_kpi_block(lines, context=context, structured=structured)
    else:
        lines.append("**Kurzfazit**")
        lines.append(one_liner)
        _append_degradation_block(lines, context)
        lines.append("**Warum ist die Auslastung so?**")
        lines.append(warum)
        lines.append("**Was ist unsicher?**")
        lines.append(unsicherheit)
        if query_intent == "forecast":
            lines.append("**Was verbessern wir als Naechstes?**")
            lines.extend(_build_measure_lines(structured=structured, improvements=improvements))
        else:
            lines.append("**Naechster Schritt**")
            lines.append(empfehlung)
        if include_kpi_block:
            _append_kpi_block(lines, context=context, structured=structured)

    if not lines:
        return "Keine verständliche Narrative verfügbar."
    return "\n\n".join(lines)


def render_template_fallback(context: dict[str, Any]) -> dict[str, Any]:
    request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
    forecast = context.get("forecast_snapshot", {}) if isinstance(context.get("forecast_snapshot"), dict) else {}
    driver_summary = context.get("driver_summary", {}) if isinstance(context.get("driver_summary"), dict) else {}
    uncertainty = context.get("uncertainty", {}) if isinstance(context.get("uncertainty"), dict) else {}
    recommendation = context.get("recommendation_digest", {}) if isinstance(context.get("recommendation_digest"), dict) else {}
    quality = context.get("quality_digest", {}) if isinstance(context.get("quality_digest"), dict) else {}
    citation_map = list(context.get("citation_map", [])) if isinstance(context.get("citation_map"), list) else []
    improvement_candidates = (
        context.get("improvement_candidates", [])
        if isinstance(context.get("improvement_candidates"), list)
        else []
    )
    occupancy_explainer = (
        context.get("occupancy_explainer", {})
        if isinstance(context.get("occupancy_explainer"), dict)
        else {}
    )

    audience = str(request_meta.get("audience", "ops"))
    zone_id = str(request_meta.get("zone_id", "unknown-zone"))
    horizon = _safe_int(request_meta.get("horizon", 60), 60)

    uncertainty_level = str(uncertainty.get("level", "high"))
    uncertainty_score = max(0.0, min(1.0, _safe_float(uncertainty.get("score", 0.8), 0.8)))
    degradation = forecast_degradation_status(context)

    uncertainty_ok = bool(recommendation.get("uncertainty_ok", True))
    stale = bool(forecast.get("stale", False))

    if stale or degradation["degraded"]:
        verdict = "blocked"
    elif uncertainty_level == "high" or not uncertainty_ok:
        verdict = "attention"
    elif any(
        str(action.get("action_type", "monitor")) != "monitor"
        for action in recommendation.get("actions", [])
        if isinstance(action, dict)
    ):
        verdict = "action_needed"
    else:
        verdict = "monitor"

    top_drivers: list[dict[str, Any]] = []
    for driver in (driver_summary.get("top_drivers", []) if isinstance(driver_summary.get("top_drivers"), list) else [])[:3]:
        if not isinstance(driver, dict):
            continue
        top_drivers.append(
            {
                "name": str(driver.get("name", "unknown")),
                "impact": _safe_float(driver.get("impact", 0.0)),
                "direction": str(driver.get("direction", "mixed")),
                "evidence_ref": str(driver.get("evidence_ref") or _first_nonempty_ref(citation_map, ["counts", "forecast", "api"])),
            }
        )

    recommended_actions = []
    actions = recommendation.get("actions", []) if isinstance(recommendation.get("actions"), list) else []
    default_action_ref = _first_nonempty_ref(citation_map, ["recommendation", "forecast", "counts", "api"])
    if verdict == "blocked":
        quality_candidate = None
        for candidate in improvement_candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_text = (
                str(candidate.get("measure_id") or "")
                + " "
                + str(candidate.get("measure_text") or "")
            ).lower()
            if any(token in candidate_text for token in ("quality", "datenqual", "sensor", "ingest", "pipeline")):
                quality_candidate = candidate
                break
        actions = [
            {
                "action_type": "quality-hardening" if quality_candidate else "monitor",
                "priority": 1,
                "rationale": (
                    str(quality_candidate.get("measure_text"))
                    if quality_candidate
                    else "Prognose ist degradiert; nur beobachten und Daten-/Modellpfad pruefen."
                ),
                "evidence_ref": str(quality_candidate.get("evidence_ref") if quality_candidate else default_action_ref),
            }
        ]
    elif improvement_candidates:
        actions = []
        for idx, candidate in enumerate(improvement_candidates[:3], start=1):
            if not isinstance(candidate, dict):
                continue
            actions.append(
                {
                    "action_type": str(candidate.get("measure_id") or "measure"),
                    "priority": idx,
                    "rationale": (
                        f"{str(candidate.get('measure_text') or 'Massnahme')} "
                        f"(Aufwand: {str(candidate.get('effort') or 'low')}, "
                        f"Wirkung: {str(candidate.get('expected_effect') or 'n/a')}, "
                        f"Naechster Schritt: {str(candidate.get('owner_hint') or 'Team abstimmen')})"
                    ),
                    "evidence_ref": str(candidate.get("evidence_ref") or ""),
                }
            )
    if uncertainty_level == "high" and not actions:
        actions = [{"action_type": "monitor", "priority": 1, "rationale": "Unsicherheit ist hoch, daher nur überwachen."}]
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
                "evidence_ref": str(action.get("evidence_ref") or default_action_ref),
            }
        )

    uncertainty_ref = str(uncertainty.get("evidence_ref") or _first_nonempty_ref(citation_map, ["xai", "forecast", "counts", "api"]))
    evidence_refs = list(
        {
            *(driver.get("evidence_ref") for driver in top_drivers),
            *(action.get("evidence_ref") for action in recommended_actions),
            uncertainty_ref,
        }
    )
    evidence_refs = [ref for ref in evidence_refs if isinstance(ref, str) and ref]
    if not evidence_refs:
        evidence_refs = ["ref-1"]

    limitations: list[str] = []
    if degradation["model_fallback"]:
        limitations.append("Forecast nutzt Baseline/Fallback statt primärem Modell.")
    if degradation["context_degraded"]:
        limitations.append("Forecast-Kontext ist veraltet oder unvollständig.")
    if stale:
        limitations.append("Forecast-Snapshot ist veraltet.")
    if degradation["quality_blocked"]:
        limitations.append("Datenqualitäts-Gate blockiert harte Empfehlungen.")
    if uncertainty_level == "high":
        limitations.append("Hohe Unsicherheit reduziert Entscheidungssicherheit.")
    if not limitations:
        limitations.append("Keine kritischen Einschränkungen erkannt.")

    if degradation["degraded"]:
        one_liner = (
            f"Zone {zone_id}: aktuell {_safe_int(occupancy_explainer.get('current_occupancy', 0), 0)} Personen, "
            "aber die Prognose ist degradiert und nicht normal belastbar."
        )
    else:
        one_liner = (
            f"Zone {zone_id}: aktuell {_safe_int(occupancy_explainer.get('current_occupancy', 0), 0)} Personen, "
            f"Risikostufe {str(occupancy_explainer.get('risk_level') or 'unknown')}."
        )
    warum = (
        f"Haupttreiber: {', '.join(d['name'] for d in top_drivers) or 'keine'}. "
        f"Basis ist Forecast/History-Evidenz [ref:{_first_nonempty_ref(citation_map, ['counts', 'forecast', 'api'])}]."
    )
    unsicherheit_text = (
        f"Unsicherheit ist {uncertainty_level} (Score {uncertainty_score:.2f}) [ref:{uncertainty_ref}]."
    )
    empfehlung_text = (
        f"Empfohlene Aktion: {recommended_actions[0]['action_type']} [ref:{recommended_actions[0]['evidence_ref']}]."
    )
    evidence_hint = "Evidenzreferenzen: " + ", ".join(f"[ref:{ref}]" for ref in evidence_refs)

    return {
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
                "evidence_ref": uncertainty_ref,
            },
            "recommended_actions": recommended_actions,
            "evidence_refs": evidence_refs,
            "confidence_statement": _confidence_statement(uncertainty_level, uncertainty_score),
            "limitations": limitations,
        },
    }


def apply_scope_redirect_narrative(
    *,
    response_payload: dict[str, Any],
    query_intent: str,
    query: str,
    zone_id: str,
    horizon: int,
) -> None:
    if query_intent not in {"meta", "off_topic"}:
        return
    narrative = response_payload.get("narrative", {}) if isinstance(response_payload.get("narrative"), dict) else {}
    if query_intent == "meta":
        narrative["one_liner"] = "Ich bin der Sitcheck Explainability-Assistent fuer Prognosefragen."
        narrative["warum"] = "Ich erklaere Auslastung, Treiber, Unsicherheit und naechste Betriebsschritte anhand Evidenz."
        narrative["unsicherheit"] = "Abseits des Prognosekontexts sind Antworten nur eingeschraenkt sinnvoll."
        narrative["empfehlung"] = (
            f"Stelle eine Forecast-Frage, z.B.: Wie entwickelt sich Zone {zone_id} in {horizon} Minuten?"
        )
        narrative["evidence_hinweis"] = "Antwort basiert auf Service-Scope und aktuellen Forecast-Kontextdaten."
    else:
        cleaned_query = _trim_text(query, max_chars=120) or "deine Anfrage"
        narrative["one_liner"] = f"Ich kann {cleaned_query} nicht fachlich beantworten, da ich auf Sitcheck-Prognosen fokussiert bin."
        narrative["warum"] = "Meine Aufgabe ist Explainability fuer Auslastung, Forecast-Treiber, Risiken und Massnahmen."
        narrative["unsicherheit"] = "Fremddomaenen ohne Forecast-Bezug wuerden zu unzuverlaessigen Aussagen fuehren."
        narrative["empfehlung"] = (
            f"Sinnvolle Folgefrage: Welche Risiken zeigt die Prognose fuer Zone {zone_id} in {horizon} Minuten?"
        )
        narrative["evidence_hinweis"] = "Keine Fremdquellen verwendet; Antwort folgt dem konfigurierten Service-Scope."
    response_payload["narrative"] = narrative


class NarrativeService:
    def __init__(
        self,
        *,
        prompt_registry: PromptRegistry,
        ollama_enabled: bool,
        ollama_base_url: str,
        ollama_model: str,
        ollama_timeout_seconds: float = 30.0,
        fallback_enabled: bool = True,
        intent_router_enabled: bool = True,
        dynamic_render_enabled: bool = True,
        llm_min_field_coverage: int = 3,
        llm_retry_on_low_coverage: int = 1,
        strict_llm_gate: bool = True,
    ):
        self.prompt_registry = prompt_registry
        self.ollama_enabled = ollama_enabled
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.fallback_enabled = fallback_enabled
        self.intent_router_enabled = intent_router_enabled
        self.dynamic_render_enabled = dynamic_render_enabled
        self.llm_min_field_coverage = max(1, min(5, int(llm_min_field_coverage)))
        self.llm_retry_on_low_coverage = max(0, int(llm_retry_on_low_coverage))
        self.strict_llm_gate = strict_llm_gate

    async def _call_ollama(self, *, prompt: str, model: str) -> tuple[dict[str, Any], str]:
        async with httpx.AsyncClient(timeout=self.ollama_timeout_seconds) as client:
            ollama_resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "format": "json",
                    "stream": False,
                    "prompt": prompt,
                    "options": {
                        "num_predict": 1536,
                        "temperature": 0.0,
                    },
                },
            )
        if ollama_resp.status_code >= 400:
            raise RuntimeError(f"ollama http {ollama_resp.status_code}")
        raw_payload = ollama_resp.json()
        raw_text = str(raw_payload.get("response", ""))
        return raw_payload, raw_text

    async def generate(
        self,
        *,
        context: dict[str, Any],
        audience: str,
        language: str,
        query: str,
        response_mode: str,
        ollama_model_override: str | None = None,
        require_ollama: bool = False,
    ) -> dict[str, Any]:
        explain_run_id = f"xai-run-{uuid.uuid4()}"
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
        context_hash = hashlib.sha256(context_json.encode("utf-8")).hexdigest()

        query_intent = classify_query_intent(query) if self.intent_router_enabled else "forecast"
        query_focus = classify_query_focus(query) if self.intent_router_enabled else "general"
        answer_profile = (
            derive_answer_profile(query, query_intent, query_focus)
            if self.dynamic_render_enabled
            else "forecast_full"
        )
        include_kpi_block = (
            should_include_kpi_block(query, query_intent, query_focus)
            if self.dynamic_render_enabled
            else True
        )

        prompt_bundle = self.prompt_registry.build_fast_narrative_prompt(
            context=context,
            audience=audience,
            language=language,
            query=query,
            query_intent=query_intent,
            query_focus=query_focus,
            answer_profile=answer_profile,
            include_kpi_block=include_kpi_block,
        )
        active_ollama_model = str(ollama_model_override or self.ollama_model).strip() or self.ollama_model

        fallback = render_template_fallback(context)
        warnings: list[str] = []

        start = time.perf_counter()
        mode = "template"
        response_payload = fallback
        model_name = "template"
        fallback_reason: str | None = None
        parse_success = False
        narrative_source_map: dict[str, str] = {
            "one_liner": "fallback",
            "warum": "fallback",
            "unsicherheit": "fallback",
            "empfehlung": "fallback",
            "evidence_hinweis": "fallback",
        }
        llm_gate_passed = False
        llm_gate_actual_coverage = 0
        llm_retry_count = 0

        fallback_structured = (
            fallback.get("structured", {})
            if isinstance(fallback.get("structured"), dict)
            else {}
        )
        fallback_narrative = (
            fallback.get("narrative", {})
            if isinstance(fallback.get("narrative"), dict)
            else {}
        )

        def _resolve_candidate_from_raw(raw_text: str) -> tuple[
            dict[str, Any] | None,
            str | None,
            dict[str, str],
            bool,
            str | None,
            list[str],
        ]:
            local_warnings: list[str] = []
            parsed = extract_json_object(raw_text)

            if isinstance(parsed, dict):
                ok, reason = validate_dual_response_shape(parsed)
                if ok:
                    source_map = {field: "llm" for field in NARRATIVE_FIELDS}
                    return parsed, "ollama", source_map, True, None, local_warnings

                hybrid_narrative, source_map = build_hybrid_narrative_payload(
                    response=parsed,
                    raw_text=raw_text,
                    fallback_narrative=fallback_narrative,
                    evidence_refs=[
                        ref
                        for ref in fallback_structured.get("evidence_refs", [])
                        if isinstance(ref, str)
                    ],
                )
                if hybrid_narrative is not None:
                    local_warnings.append("LLM output adapted to hybrid response shape")
                    return (
                        {"narrative": hybrid_narrative, "structured": fallback_structured},
                        "ollama_hybrid",
                        source_map,
                        True,
                        None,
                        local_warnings,
                    )
                return None, None, narrative_source_map, False, f"invalid_shape:{reason}", [
                    f"LLM output invalid: {reason}"
                ]

            hybrid_narrative, source_map = build_hybrid_narrative_payload(
                response=None,
                raw_text=raw_text,
                fallback_narrative=fallback_narrative,
                evidence_refs=[
                    ref
                    for ref in fallback_structured.get("evidence_refs", [])
                    if isinstance(ref, str)
                ],
            )
            if hybrid_narrative is not None:
                local_warnings.append("LLM output was not valid JSON; hybrid narrative salvage applied")
                return (
                    {"narrative": hybrid_narrative, "structured": fallback_structured},
                    "ollama_hybrid",
                    source_map,
                    any(source == "llm" for source in source_map.values()),
                    None,
                    local_warnings,
                )
            return None, None, narrative_source_map, False, "unparseable_json", [
                "LLM output could not be parsed as JSON"
            ]

        if self.ollama_enabled:
            try:
                retry_fields: list[str] = []
                max_attempts = 1 + self.llm_retry_on_low_coverage
                for attempt in range(max_attempts):
                    llm_retry_count = attempt
                    if attempt == 0:
                        active_bundle = prompt_bundle
                    else:
                        active_bundle = self.prompt_registry.build_fast_narrative_prompt(
                            context=context,
                            audience=audience,
                            language=language,
                            query=query,
                            query_intent=query_intent,
                            query_focus=query_focus,
                            answer_profile=answer_profile,
                            include_kpi_block=include_kpi_block,
                            missing_fields=retry_fields,
                            retry_attempt=attempt,
                        )

                    _raw_payload, raw_text = await self._call_ollama(
                        prompt=active_bundle.prompt,
                        model=active_ollama_model,
                    )
                    candidate_payload, candidate_mode, candidate_map, candidate_parse_success, candidate_reason, candidate_warnings = (
                        _resolve_candidate_from_raw(raw_text)
                    )
                    warnings.extend(candidate_warnings)
                    parse_success = parse_success or candidate_parse_success

                    if candidate_payload is None or candidate_mode is None:
                        fallback_reason = candidate_reason or "invalid_llm_output"
                        continue

                    llm_gate_actual_coverage = len(llm_coverage_fields(candidate_map))
                    if llm_gate_actual_coverage >= self.llm_min_field_coverage:
                        response_payload = candidate_payload
                        mode = candidate_mode
                        model_name = active_ollama_model
                        narrative_source_map = candidate_map
                        llm_gate_passed = True
                        fallback_reason = None
                        break

                    retry_fields = missing_llm_fields(candidate_map)
                    fallback_reason = f"low_coverage:{llm_gate_actual_coverage}"
                    warnings.append(
                        "LLM output below coverage gate "
                        f"({llm_gate_actual_coverage}/{self.llm_min_field_coverage})"
                    )
                    if attempt < max_attempts - 1:
                        warnings.append(
                            "Retrying LLM call for missing narrative fields: "
                            + ", ".join(retry_fields)
                        )
                        continue

                    if self.strict_llm_gate:
                        raise LLMQualityGateError(
                            min_coverage=self.llm_min_field_coverage,
                            actual_coverage=llm_gate_actual_coverage,
                            query_intent=query_intent,
                        )

                    # Non-strict mode keeps best candidate rather than hard failing.
                    response_payload = candidate_payload
                    mode = candidate_mode
                    model_name = active_ollama_model
                    narrative_source_map = candidate_map
                    break
            except Exception as exc:
                if isinstance(exc, LLMQualityGateError):
                    raise
                err_detail = str(exc).strip() or exc.__class__.__name__
                fallback_reason = f"ollama_error:{err_detail}"
                warnings.append(f"Ollama call failed: {err_detail}")

        if require_ollama and mode not in {"ollama", "ollama_hybrid"}:
            raise LLMNarrativeUnavailableError(
                reason=fallback_reason or ("ollama_required_but_disabled" if not self.ollama_enabled else "invalid_llm_output"),
                query_intent=query_intent,
            )

        if mode not in {"ollama", "ollama_hybrid"}:
            if not self.fallback_enabled:
                raise LLMNarrativeUnavailableError(
                    reason=fallback_reason or "unknown",
                    query_intent=query_intent,
                )
            mode = "template_fallback" if self.ollama_enabled else "template"
            model_name = "template"

        if mode in {"template", "template_fallback"} and query_intent in {"meta", "off_topic"}:
            request_meta = context.get("request_meta", {}) if isinstance(context.get("request_meta"), dict) else {}
            apply_scope_redirect_narrative(
                response_payload=response_payload,
                query_intent=query_intent,
                query=query,
                zone_id=str(request_meta.get("zone_id", "unknown-zone")),
                horizon=_safe_int(request_meta.get("horizon", 60), 60),
            )

        # Guardrail: avoid prompt-echo responses (especially with small local models).
        narrative = response_payload.get("narrative", {}) if isinstance(response_payload.get("narrative"), dict) else {}
        if query_intent == "forecast" and isinstance(narrative, dict):
            one_liner = str(narrative.get("one_liner") or "")
            one_liner_norm = _normalize_query(one_liner)
            needs_replacement = query_focus in {"status", "student"} or _looks_like_prompt_echo(one_liner, query)

            if query_focus == "status" and not re.search(r"\d", one_liner_norm):
                needs_replacement = True

            if query_focus == "student":
                technical_markers = ("stale", "horizon", "a2-b1", "indicates", "context")
                if any(marker in one_liner_norm for marker in technical_markers):
                    needs_replacement = True

            if needs_replacement:
                fallback_one_liner = _trim_text(
                    _build_context_one_liner(
                        context,
                        style="student" if query_focus == "student" else "standard",
                    ),
                    max_chars=220,
                )
                if fallback_one_liner:
                    narrative["one_liner"] = fallback_one_liner
                    response_payload["narrative"] = narrative
                    narrative_source_map["one_liner"] = "fallback"
                    warnings.append("one_liner replaced due to status_student_stability_guard")

            if query_focus == "student":
                technical_markers = ("a2-b1", "a2/b1", "horizon", "context", "indicates", "stale")
                warum_text = str(narrative.get("warum") or "")
                empfehlung_text = str(narrative.get("empfehlung") or "")
                unsicherheit_text = str(narrative.get("unsicherheit") or "")
                structured_payload = (
                    response_payload.get("structured", {})
                    if isinstance(response_payload.get("structured"), dict)
                    else {}
                )

                if any(marker in _normalize_query(warum_text) for marker in technical_markers):
                    narrative["warum"] = _build_student_reason(context)
                    narrative_source_map["warum"] = "fallback"
                    warnings.append("student_warum replaced by simple_context_reason")

                if any(marker in _normalize_query(empfehlung_text) for marker in technical_markers):
                    narrative["empfehlung"] = _build_student_next_step(context)
                    narrative_source_map["empfehlung"] = "fallback"
                    warnings.append("student_empfehlung replaced by simple_context_step")

                if any(marker in _normalize_query(unsicherheit_text) for marker in technical_markers):
                    narrative["unsicherheit"] = _build_student_uncertainty(context, structured_payload)
                    narrative_source_map["unsicherheit"] = "fallback"
                    warnings.append("student_unsicherheit replaced by simple_context_uncertainty")

                response_payload["narrative"] = narrative

        if query_intent == "forecast" and apply_degradation_guardrail(
            response_payload=response_payload,
            context=context,
            narrative_source_map=narrative_source_map,
        ):
            warnings.append("degradation_guardrail_applied")
        if query_intent == "forecast" and apply_non_degraded_action_guardrail(
            response_payload=response_payload,
            context=context,
            narrative_source_map=narrative_source_map,
        ):
            warnings.append("non_degraded_action_guardrail_applied")

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        evidence_refs = []
        structured = response_payload.get("structured", {}) if isinstance(response_payload, dict) else {}
        if isinstance(structured, dict):
            evidence_refs = [ref for ref in structured.get("evidence_refs", []) if isinstance(ref, str)]
        evidence_coverage = 1.0 if evidence_refs else 0.0

        logger.info(
            "xai_narrative_metrics explain_run_id=%s mode=%s xai_narrative_latency_ms=%d "
            "xai_narrative_fallback_rate=%d xai_parse_success_rate=%d xai_evidence_ref_coverage=%.2f "
            "fallback_reason=%s template_set=%s prompt_version=%s "
            "query_intent=%s query_focus=%s answer_profile=%s include_kpi_block=%d "
            "llm_gate_passed=%d llm_gate_actual_coverage=%d llm_retry_count=%d",
            explain_run_id,
            mode,
            elapsed_ms,
            1 if mode not in {"ollama", "ollama_hybrid"} else 0,
            1 if parse_success else 0,
            evidence_coverage,
            fallback_reason or "none",
            prompt_bundle.template_set_id,
            prompt_bundle.prompt_version,
            query_intent,
            query_focus,
            answer_profile,
            1 if include_kpi_block else 0,
            1 if llm_gate_passed else 0,
            llm_gate_actual_coverage,
            llm_retry_count,
        )

        render_profile = answer_profile if self.dynamic_render_enabled else "forecast_full"
        render_kpi = include_kpi_block if self.dynamic_render_enabled else True
        degradation = forecast_degradation_status(context)

        return {
            "mode": mode,
            "narrative_markdown": render_dual_output_markdown(
                response_payload,
                context=context,
                answer_profile=render_profile,
                include_kpi_block=render_kpi,
                query_intent=query_intent,
                query_focus=query_focus,
            ),
            "structured": response_payload.get("structured", {}),
            "response": response_payload,
            "context": context,
            "warnings": warnings,
            "meta": {
                "explain_run_id": explain_run_id,
                "template_set_id": prompt_bundle.template_set_id,
                "prompt_version": prompt_bundle.prompt_version,
                "context_hash": context_hash,
                "context_generated_at": str(context.get("request_meta", {}).get("generated_at", datetime.now(UTC).isoformat())),
                "model": model_name,
                "latency_ms": elapsed_ms,
                "response_mode": response_mode,
                "fallback_reason": fallback_reason,
                "narrative_source_map": narrative_source_map,
                "llm_field_coverage": len(llm_coverage_fields(narrative_source_map)),
                "query_intent": query_intent,
                "query_focus": query_focus,
                "answer_profile": answer_profile,
                "include_kpi_block": include_kpi_block,
                "llm_gate_passed": llm_gate_passed,
                "llm_gate_min_coverage": self.llm_min_field_coverage,
                "llm_gate_actual_coverage": llm_gate_actual_coverage,
                "llm_retry_count": llm_retry_count,
                "forecast_model_version": degradation["model_version"],
                "forecast_degraded": degradation["degraded"],
                "forecast_degradation_reasons": degradation["reasons"],
            },
        }
