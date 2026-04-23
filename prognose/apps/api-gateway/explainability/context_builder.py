from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any


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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _first_nonempty_ref(citation_map: list[dict[str, Any]], preferred_types: list[str]) -> str:
    for source_type in preferred_types:
        for item in citation_map:
            if str(item.get("source_type", "")) == source_type:
                return str(item.get("ref_id"))
    if citation_map:
        return str(citation_map[0].get("ref_id"))
    return "ref-missing"


def build_history_digest(history_points: list[dict[str, Any]]) -> dict[str, Any]:
    if not history_points:
        return {
            "last_occupancy": 0.0,
            "last_utilization": 0.0,
            "trend_15m": 0.0,
            "trend_60m": 0.0,
            "quality_score_avg": 0.0,
            "quality_flags_top": ["NO_HISTORY"],
            "point_count": 0,
            "similar_pattern": {
                "found": False,
                "reason": "insufficient_history",
            },
        }

    occupancies = [_safe_float(point.get("occupancy", 0.0)) for point in history_points]
    utilizations = [_safe_float(point.get("utilization", 0.0)) for point in history_points]
    quality_scores = [_safe_float(point.get("quality_score", 0.0), 0.0) for point in history_points]

    last_occupancy = occupancies[-1]
    last_utilization = utilizations[-1] if utilizations else 0.0
    mean_15 = _mean(occupancies[-15:])
    mean_60 = _mean(occupancies[-60:])

    flags = Counter()
    for point in history_points:
        for flag in point.get("quality_flags", []) or []:
            if isinstance(flag, str):
                flags[flag] += 1
    if not flags:
        flags["OK"] += 1

    similar_pattern: dict[str, Any]
    if len(occupancies) >= 120:
        current = occupancies[-60:]
        best_mae: float | None = None
        best_start = -1
        # Compare with older 60m windows and skip latest region.
        for start_idx in range(0, len(occupancies) - 120 + 1, 5):
            window = occupancies[start_idx : start_idx + 60]
            mae = _mean([abs(a - b) for a, b in zip(current, window)])
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_start = start_idx

        if best_mae is None or best_start < 0:
            similar_pattern = {
                "found": False,
                "reason": "no_candidate_window",
            }
        else:
            from_ts = history_points[best_start].get("timestamp")
            to_ts = history_points[best_start + 59].get("timestamp")
            similar_pattern = {
                "found": True,
                "distance_mae": round(float(best_mae), 4),
                "window_from": str(from_ts),
                "window_to": str(to_ts),
                "current_mean": round(_mean(current), 4),
                "matched_mean": round(_mean(occupancies[best_start : best_start + 60]), 4),
            }
    else:
        similar_pattern = {
            "found": False,
            "reason": "insufficient_history",
        }

    return {
        "last_occupancy": round(last_occupancy, 4),
        "last_utilization": round(last_utilization, 6),
        "trend_15m": round(last_occupancy - mean_15, 4),
        "trend_60m": round(last_occupancy - mean_60, 4),
        "quality_score_avg": round(max(0.0, min(1.0, _mean(quality_scores))), 6),
        "quality_flags_top": [flag for flag, _ in flags.most_common(5)],
        "point_count": len(history_points),
        "similar_pattern": similar_pattern,
    }


def build_citation_map(*evidence_objects: dict[str, Any] | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for evidence in evidence_objects:
        if not isinstance(evidence, dict):
            continue

        evidence_id = str(evidence.get("evidence_id") or "ev-missing")
        time_window = evidence.get("time_window") if isinstance(evidence.get("time_window"), dict) else {}
        model = evidence.get("model") if isinstance(evidence.get("model"), dict) else {}
        quality = evidence.get("quality") if isinstance(evidence.get("quality"), dict) else {}

        sources = evidence.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        for source in sources:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("type") or "api")
            source_id = str(source.get("id") or "unknown")
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
                        "from": str(time_window.get("from") or _now_iso()),
                        "to": str(time_window.get("to") or _now_iso()),
                    },
                    "model_version": str(model.get("version") or "unknown"),
                    "quality_score": round(max(0.0, min(1.0, _safe_float(quality.get("score", 0.5), 0.5))), 6),
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


def _derive_risk_level(
    *,
    utilization_now_pct: float,
    forecast_peak_pct: float,
    uncertainty_level: str,
    forecast_stale: bool,
    quality_ok: bool,
) -> str:
    score = 0
    if utilization_now_pct >= 95.0:
        score += 2
    elif utilization_now_pct >= 85.0:
        score += 1

    if forecast_peak_pct >= 95.0:
        score += 2
    elif forecast_peak_pct >= 85.0:
        score += 1

    if uncertainty_level == "high":
        score += 2
    elif uncertainty_level == "medium":
        score += 1

    if forecast_stale:
        score += 1
    if not quality_ok:
        score += 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _build_improvement_candidates(
    *,
    zone_capacity: int,
    utilization_now_pct: float,
    trend_15m: float,
    forecast_peak: float,
    quality_ok: bool,
    uncertainty_level: str,
    forecast_stale: bool,
    lecture_digest: dict[str, Any],
    ref_recommendation: str,
    ref_counts: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    peak_ratio = 0.0
    if zone_capacity > 0:
        peak_ratio = max(0.0, forecast_peak / float(zone_capacity))

    if utilization_now_pct >= 80.0 or peak_ratio >= 0.85:
        candidates.append(
            {
                "measure_id": "capacity-buffer-next-slot",
                "measure_text": "Für die nächste Stunde einen Pufferplatz- oder Ausweichbereich vorbereiten.",
                "expected_effect": "Spitzenlast kurzfristig glätten und Überfüllung vermeiden.",
                "effort": "low",
                "owner_hint": "Bibliotheksteam + Standortkoordination",
                "evidence_ref": ref_counts,
            }
        )

    starts_next = _safe_int(lecture_digest.get("starts_next_60m", 0), 0)
    lecture_pull = _safe_float(lecture_digest.get("lecture_net_pull", 0.0), 0.0)
    if starts_next > 0 or lecture_pull > 0.0:
        candidates.append(
            {
                "measure_id": "lecture-transition-communication",
                "measure_text": "Vor Vorlesungswechseln aktiv auf alternative Lernzonen hinweisen.",
                "expected_effect": "Zufluss in kritischen 30-60 Minuten besser verteilen.",
                "effort": "low",
                "owner_hint": "Service Desk + Kommunikation",
                "evidence_ref": ref_counts,
            }
        )

    if uncertainty_level == "high" or not quality_ok or forecast_stale:
        candidates.append(
            {
                "measure_id": "quality-hardening",
                "measure_text": "Datenqualität und Sensor-/Ingest-Pipeline im Tagesbetrieb priorisiert prüfen.",
                "expected_effect": "Stabilere Prognosen und belastbarere Maßnahmenentscheidungen.",
                "effort": "medium",
                "owner_hint": "Data/IT Betrieb",
                "evidence_ref": ref_recommendation,
            }
        )

    if trend_15m >= 5.0:
        candidates.append(
            {
                "measure_id": "staffing-peak-window",
                "measure_text": "In den nächsten 60 Minuten eine zusätzliche Vor-Ort-Unterstützung einplanen.",
                "expected_effect": "Schnellere Entlastung bei Auslastungsanstieg.",
                "effort": "medium",
                "owner_hint": "Schichtplanung",
                "evidence_ref": ref_counts,
            }
        )

    if not candidates:
        candidates.append(
            {
                "measure_id": "monitor-only",
                "measure_text": "Weiter beobachten und bei Abweichungen innerhalb von 15 Minuten neu bewerten.",
                "expected_effect": "Fehlsteuerung vermeiden, solange keine klare Lastspitze erkennbar ist.",
                "effort": "low",
                "owner_hint": "Operations",
                "evidence_ref": ref_recommendation,
            }
        )

    # Ensure stable but non-duplicated fallback measures.
    deduped_candidates: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = str(candidate.get("measure_text") or "").strip().lower()
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        deduped_candidates.append(candidate)

    fallback_pool = [
        {
            "measure_id": "monitor-followup-1",
            "measure_text": "Trend weiter beobachten und Statusupdate an das Team geben.",
            "expected_effect": "Gemeinsames Lagebild sichern.",
            "effort": "low",
            "owner_hint": "Operations",
            "evidence_ref": ref_counts,
        },
        {
            "measure_id": "monitor-followup-2",
            "measure_text": "Nächsten Snapshot in 15 Minuten gegen Ist-Daten validieren.",
            "expected_effect": "Frühe Abweichungen erkennen und Fehlreaktionen vermeiden.",
            "effort": "low",
            "owner_hint": "Operations",
            "evidence_ref": ref_counts,
        },
        {
            "measure_id": "monitor-followup-3",
            "measure_text": "Unsicherheits- und Qualitätsstatus im Team-Update explizit ausweisen.",
            "expected_effect": "Entscheidungen bleiben trotz hoher Unsicherheit nachvollziehbar.",
            "effort": "low",
            "owner_hint": "Operations",
            "evidence_ref": ref_recommendation,
        },
    ]

    for fallback_candidate in fallback_pool:
        if len(deduped_candidates) >= 3:
            break
        key = str(fallback_candidate.get("measure_text") or "").strip().lower()
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        deduped_candidates.append(fallback_candidate)

    while len(deduped_candidates) < 3:
        idx = len(deduped_candidates) + 1
        deduped_candidates.append(
            {
                "measure_id": f"monitor-followup-extra-{idx}",
                "measure_text": f"Lagebild weiter monitoren (Folgepunkt {idx}).",
                "expected_effect": "Kontinuierliche Transparenz im Betrieb.",
                "effort": "low",
                "owner_hint": "Operations",
                "evidence_ref": ref_counts,
            }
        )

    return deduped_candidates[:3]


def build_explainability_context_v2(
    *,
    zone_id: str,
    zone_capacity: int,
    horizon: int,
    audience: str,
    language: str,
    query: str,
    forecast_latest: dict[str, Any],
    explanation: dict[str, Any],
    recommendation: dict[str, Any],
    history_points: list[dict[str, Any]],
    live_state: dict[str, Any],
    alerts: list[dict[str, Any]],
    lecture_impact: dict[str, Any] | None,
    template_set_id: str,
    prompt_version: str,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    history_digest = build_history_digest(history_points)

    history_evidence = history_points[-1].get("evidence") if history_points else None
    lecture_evidence = None
    if lecture_impact and isinstance(lecture_impact, dict):
        lecture_evidence = {
            "evidence_id": f"lecture-{uuid.uuid4()}",
            "generated_at": now.isoformat(),
            "time_window": {
                "from": str(lecture_impact.get("timestamp") or now.isoformat()),
                "to": str(lecture_impact.get("timestamp") or now.isoformat()),
            },
            "sources": [{"type": "lecture_activity", "id": str(lecture_impact.get("source") or "unknown")}],
            "model": {"name": "lecture-impact", "version": str(lecture_impact.get("impact", {}).get("impact_model_version") or "v1")},
            "quality": {
                "score": _safe_float(lecture_impact.get("quality_score", 0.5), 0.5),
                "flags": list(lecture_impact.get("quality_flags", []) or []),
            },
        }

    citation_map = build_citation_map(
        forecast_latest.get("evidence", {}),
        explanation.get("evidence", {}),
        recommendation.get("evidence", {}),
        history_evidence if isinstance(history_evidence, dict) else None,
        lecture_evidence,
    )

    ref_counts = _first_nonempty_ref(citation_map, ["counts", "forecast", "api"])
    ref_xai = _first_nonempty_ref(citation_map, ["xai", "forecast", "counts", "api"])
    ref_recommendation = _first_nonempty_ref(citation_map, ["recommendation", "forecast", "counts", "api"])

    forecast_points = list(forecast_latest.get("points", [])) if isinstance(forecast_latest.get("points"), list) else []
    forecast_yhats = [_safe_float(point.get("yhat", 0.0)) for point in forecast_points]
    forecast_peak = max(forecast_yhats) if forecast_yhats else 0.0

    uncertainty_payload = explanation.get("uncertainty") if isinstance(explanation.get("uncertainty"), dict) else {}

    driver_summary = []
    for driver in (explanation.get("drivers") if isinstance(explanation.get("drivers"), list) else [])[:5]:
        if not isinstance(driver, dict):
            continue
        evidence_ref = ref_counts
        if str(driver.get("name")) == "event_context":
            evidence_ref = _first_nonempty_ref(citation_map, ["events", "counts", "api"])
        driver_summary.append(
            {
                "name": str(driver.get("name") or "unknown"),
                "impact": _safe_float(driver.get("impact", 0.0)),
                "direction": str(driver.get("direction") or "mixed"),
                "description": str(driver.get("description") or ""),
                "evidence_ref": evidence_ref,
            }
        )

    gates = recommendation.get("gates") if isinstance(recommendation.get("gates"), dict) else {}
    recommendation_actions = []
    for action in (recommendation.get("actions") if isinstance(recommendation.get("actions"), list) else []):
        if not isinstance(action, dict):
            continue
        recommendation_actions.append(
            {
                "action_type": str(action.get("action_type") or "monitor"),
                "priority": _safe_int(action.get("priority", 1), 1),
                "rationale": str(action.get("rationale") or ""),
                "evidence_ref": ref_recommendation,
            }
        )
    if not recommendation_actions:
        recommendation_actions.append(
            {
                "action_type": "monitor",
                "priority": 1,
                "rationale": "Keine belastbare Aktion ableitbar.",
                "evidence_ref": ref_recommendation,
            }
        )

    lecture_payload = lecture_impact or {}
    lecture_digest = {
        "active_lectures": _safe_int(lecture_payload.get("active_lectures", 0), 0),
        "active_courses": _safe_int(lecture_payload.get("active_courses", 0), 0),
        "starts_next_60m": _safe_int(lecture_payload.get("starts_next_60m", 0), 0),
        "ends_next_60m": _safe_int(lecture_payload.get("ends_next_60m", 0), 0),
        "heavy_active_lectures": _safe_int(lecture_payload.get("impact", {}).get("heavy_active_lectures", 0), 0)
        if isinstance(lecture_payload.get("impact"), dict)
        else 0,
        "heavy_ended_last_60m": _safe_int(lecture_payload.get("impact", {}).get("heavy_ended_last_60m", 0), 0)
        if isinstance(lecture_payload.get("impact"), dict)
        else 0,
        "lecture_pull_regular": _safe_float(lecture_payload.get("impact", {}).get("lecture_pull_regular", 0.0), 0.0)
        if isinstance(lecture_payload.get("impact"), dict)
        else 0.0,
        "heavy_bib_bonus": _safe_float(lecture_payload.get("impact", {}).get("heavy_bib_bonus", 0.0), 0.0)
        if isinstance(lecture_payload.get("impact"), dict)
        else 0.0,
        "lecture_net_pull": _safe_float(lecture_payload.get("impact", {}).get("lecture_net_pull", 0.0), 0.0)
        if isinstance(lecture_payload.get("impact"), dict)
        else 0.0,
        "impact_model_version": str(lecture_payload.get("impact", {}).get("impact_model_version") or "")
        if isinstance(lecture_payload.get("impact"), dict)
        else "",
    }

    quality_digest = {
        "live_quality_score": _safe_float(live_state.get("quality_score", 0.0), 0.0),
        "live_quality_flags": list(live_state.get("quality_flags", []) or []),
        "history_quality_score_avg": _safe_float(history_digest.get("quality_score_avg", 0.0), 0.0),
        "live_point_count": _safe_int(live_state.get("point_count", 0), 0),
        "alerts": [
            {
                "code": str(alert.get("code") or "UNKNOWN"),
                "level": str(alert.get("level") or "info"),
                "message": str(alert.get("message") or ""),
            }
            for alert in alerts
            if isinstance(alert, dict)
        ],
    }

    zone_capacity_value = max(0, _safe_int(zone_capacity, 0))
    current_occupancy = _safe_int(live_state.get("occupancy", history_digest.get("last_occupancy", 0.0)), 0)
    current_utilization = _safe_float(
        live_state.get(
            "utilization",
            (current_occupancy / max(zone_capacity_value, 1)) if zone_capacity_value > 0 else history_digest.get("last_utilization", 0.0),
        ),
        0.0,
    )
    utilization_now_pct = max(0.0, current_utilization * 100.0)
    history_occupancies = [_safe_float(point.get("occupancy", 0.0)) for point in history_points]
    avg_60m = _mean(history_occupancies[-60:])
    uncertainty_level = str(uncertainty_payload.get("level") or "high")
    quality_ok = bool(gates.get("quality_ok", True))
    forecast_stale = bool(forecast_latest.get("stale", False))
    peak_pct = (forecast_peak / max(zone_capacity_value, 1)) * 100.0 if zone_capacity_value > 0 else 0.0
    risk_level = _derive_risk_level(
        utilization_now_pct=utilization_now_pct,
        forecast_peak_pct=peak_pct,
        uncertainty_level=uncertainty_level,
        forecast_stale=forecast_stale,
        quality_ok=quality_ok,
    )
    improvement_candidates = _build_improvement_candidates(
        zone_capacity=zone_capacity_value,
        utilization_now_pct=utilization_now_pct,
        trend_15m=_safe_float(history_digest.get("trend_15m", 0.0), 0.0),
        forecast_peak=forecast_peak,
        quality_ok=quality_ok,
        uncertainty_level=uncertainty_level,
        forecast_stale=forecast_stale,
        lecture_digest=lecture_digest,
        ref_recommendation=ref_recommendation,
        ref_counts=ref_counts,
    )

    context = {
        "request_meta": {
            "request_id": f"ctx-{uuid.uuid4()}",
            "generated_at": now.isoformat(),
            "zone_id": zone_id,
            "horizon": horizon,
            "audience": audience,
            "language": language,
            "query": query,
            "guardrails": [
                "no_free_sql",
                "api_tools_only",
                "no_invented_facts",
                "each_claim_needs_evidence_ref",
                "no_hard_action_if_uncertainty_high",
            ],
            "template_set_id": template_set_id,
            "prompt_version": prompt_version,
        },
        "zone_capacity": zone_capacity_value,
        "utilization_now_pct": round(utilization_now_pct, 2),
        "occupancy_explainer": {
            "current_occupancy": current_occupancy,
            "avg_60m": round(avg_60m, 2),
            "trend_15m": round(_safe_float(history_digest.get("trend_15m", 0.0), 0.0), 2),
            "forecast_next_60m_peak": round(forecast_peak, 2),
            "risk_level": risk_level,
        },
        "improvement_candidates": improvement_candidates,
        "forecast_snapshot": {
            "zone_id": str(forecast_latest.get("zone_id") or zone_id),
            "horizon": _safe_int(forecast_latest.get("horizon", horizon), horizon),
            "generated_at": str(forecast_latest.get("generated_at") or now.isoformat()),
            "age_seconds": _safe_int(forecast_latest.get("age_seconds", 0), 0),
            "stale": bool(forecast_latest.get("stale", False)),
            "summary": str(forecast_latest.get("summary") or ""),
            "model_version": str(forecast_latest.get("model_version") or "unknown"),
            "source": str(forecast_latest.get("source") or "snapshot"),
            "next_point": forecast_points[0] if forecast_points else None,
            "peak": {
                "value": round(forecast_peak, 4),
                "timestamp": str(forecast_points[forecast_yhats.index(forecast_peak)].get("timestamp")) if forecast_points and forecast_yhats else None,
            },
            "point_count": len(forecast_points),
            "evidence_ref": _first_nonempty_ref(citation_map, ["forecast", "counts", "api"]),
        },
        "history_digest": history_digest,
        "driver_summary": {
            "top_drivers": driver_summary,
            "evidence_ref": ref_xai,
        },
        "uncertainty": {
            "score": round(max(0.0, min(1.0, _safe_float(uncertainty_payload.get("score", 0.8), 0.8))), 6),
            "level": uncertainty_level,
            "reason": str(uncertainty_payload.get("reason") or "unspecified"),
            "evidence_ref": ref_xai,
        },
        "recommendation_digest": {
            "summary": str(recommendation.get("summary") or ""),
            "quality_ok": bool(gates.get("quality_ok", True)),
            "uncertainty_ok": bool(gates.get("uncertainty_ok", True)),
            "actions": recommendation_actions,
            "evidence_ref": ref_recommendation,
        },
        "lecture_impact_digest": lecture_digest,
        "quality_digest": quality_digest,
        "citation_map": citation_map,
        "policy_block": {
            "abstain_rules": [
                "Wenn uncertainty.level=high, nur monitor empfehlen.",
                "Bei quality_ok=false keine harte Maßnahme ausgeben.",
                "Keine Aussage ohne evidence_ref.",
            ],
            "decision_gates": {
                "quality_ok": bool(gates.get("quality_ok", True)),
                "uncertainty_ok": bool(gates.get("uncertainty_ok", True)),
                "forecast_stale": bool(forecast_latest.get("stale", False)),
            },
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

    return context
