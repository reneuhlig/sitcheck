from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from agents import AssistantOrchestrator
from api_client import SitcheckApiClient
from ui.panels import (
    render_alert_rail,
    render_calendar,
    render_drivers_and_recommendations,
    render_evidence,
    render_forecast_chart,
    render_forecast_lab,
    render_header,
    render_horizon_panel,
    render_kpis,
    render_weekly_outlook,
    render_scenario_panel,
)
from ui.state import DashboardState, sidebar_state, try_auto_refresh
from ui.theme import apply_theme


st.set_page_config(page_title="Sitcheck Command Center", page_icon=":bar_chart:", layout="wide")
apply_theme()


@st.cache_data(ttl=5, show_spinner=False)
def cached_health(base_url: str) -> dict[str, Any]:
    return SitcheckApiClient(base_url).health()


@st.cache_data(ttl=30, show_spinner=False)
def cached_zones(base_url: str) -> list[dict[str, Any]]:
    return SitcheckApiClient(base_url).get_zones()


@st.cache_data(ttl=5, show_spinner=False)
def cached_command_center(
    base_url: str,
    zone_id: str,
    horizon: int,
    history_minutes: int,
    stale_seconds: int,
    long_term_days: int,
) -> dict[str, Any]:
    return SitcheckApiClient(base_url).get_command_center(
        zone_id=zone_id,
        horizon=horizon,
        history_minutes=history_minutes,
        stale_seconds=stale_seconds,
        long_term_days=long_term_days,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_weekly_outlook(
    base_url: str,
    zone_id: str,
    weekly_days: int,
    weekly_slot_minutes: int,
    stale_seconds: int,
    history_minutes: int,
) -> dict[str, Any]:
    client = SitcheckApiClient(base_url)
    return build_weekly_outlook_payload(
        client=client,
        zone_id=zone_id,
        weekly_days=weekly_days,
        weekly_slot_minutes=weekly_slot_minutes,
        stale_seconds=stale_seconds,
        history_minutes=history_minutes,
    )


def build_command_center_fallback(client: SitcheckApiClient, state: DashboardState) -> dict[str, Any]:
    now = datetime.now(UTC)
    try:
        history = client.tool_get_history(zone_id=state.zone_id, minutes=state.history_minutes, granularity="1m")
    except Exception:
        history = {"zone_id": state.zone_id, "points": [], "granularity": "1m"}
    points = history.get("points", [])
    latest_point = points[-1] if points else {}
    try:
        forecast = client.get_forecast_latest(zone_id=state.zone_id, horizon=state.horizon, stale_seconds=state.stale_threshold_sec)
    except Exception:
        forecast = {"zone_id": state.zone_id, "horizon": state.horizon, "stale": True, "points": [], "source": "fallback"}
    try:
        explanation = client.tool_explain_forecast(zone_id=state.zone_id, horizon=state.horizon)
    except Exception:
        explanation = {
            "zone_id": state.zone_id,
            "horizon": state.horizon,
            "summary": "Fallback: explainability unavailable",
            "drivers": [],
            "uncertainty": {"score": 1.0, "level": "high", "reason": "upstream timeout"},
            "evidence": {"quality": {"score": 0.0, "flags": ["UPSTREAM_UNAVAILABLE"]}},
        }
    try:
        recommendations = client.tool_recommend_actions(zone_id=state.zone_id, horizon=state.horizon)
    except Exception:
        recommendations = {"zone_id": state.zone_id, "horizon": state.horizon, "summary": "Fallback: recommendations unavailable", "actions": []}
    try:
        calendar_events = client.list_calendar_events(zone_id=state.zone_id, hours=state.long_term_days * 24)
    except Exception:
        calendar_events = []

    alerts = []
    if not points:
        alerts.append({"code": "NO_DATA", "level": "warn", "message": "No history points available."})
    if forecast.get("stale"):
        alerts.append({"code": "STALE", "level": "risk", "message": "Forecast snapshot is stale."})
    if not alerts:
        alerts.append({"code": "ALL_CLEAR", "level": "ok", "message": "No critical warnings."})

    return {
        "meta": {
            "generated_at": now.isoformat(),
            "zone_id": state.zone_id,
            "horizon": state.horizon,
            "history_minutes": state.history_minutes,
            "long_term_days": state.long_term_days,
            "stale_seconds": state.stale_threshold_sec,
            "environment": "fallback",
        },
        "service_health": [{"service": "api-gateway", "status": "ok", "latency_ms": 0, "detail": "fallback mode"}],
        "live": {
            "timestamp": latest_point.get("timestamp"),
            "occupancy": int(latest_point.get("occupancy", 0) or 0),
            "utilization": float(latest_point.get("utilization", 0.0) or 0.0),
            "quality_score": float(latest_point.get("quality_score", 0.0) or 0.0),
            "quality_flags": latest_point.get("quality_flags", ["NO_DATA"]),
            "point_count": len(points),
        },
        "history": history,
        "forecast_latest": forecast,
        "forecast_long_term": [],
        "explanation": explanation,
        "recommendations": recommendations,
        "calendar_events": calendar_events,
        "alerts": alerts,
    }


def _build_weekly_reference_cards(
    forecast: dict[str, Any],
    lineage: dict[str, Any],
    weekly_context: dict[str, Any],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []

    evidence = forecast.get("evidence", {}) if isinstance(forecast.get("evidence"), dict) else {}
    for source in evidence.get("sources", []) if isinstance(evidence.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        references.append(
            {
                "label": f"{source.get('type', 'source')}: {source.get('id', 'unknown')}",
                "kind": "evidence",
                "note": str(source.get("note") or "Forecast evidence source"),
                "href": str(source.get("uri") or ""),
            }
        )

    training = lineage.get("training", {}) if isinstance(lineage.get("training"), dict) else {}
    for ref in training.get("data_references", []) if isinstance(training.get("data_references"), list) else []:
        if not isinstance(ref, dict):
            continue
        references.append(
            {
                "label": str(ref.get("label") or "Training reference"),
                "kind": "training",
                "note": str(ref.get("note") or ref.get("source_id") or ""),
                "href": str(ref.get("href") or ref.get("uri") or ""),
            }
        )

    context = weekly_context.get("context", {}) if isinstance(weekly_context, dict) else {}
    for citation in context.get("citation_map", []) if isinstance(context.get("citation_map"), list) else []:
        if not isinstance(citation, dict):
            continue
        references.append(
            {
                "label": f"{citation.get('source_type', 'source')}: {citation.get('source_id', 'unknown')}",
                "kind": "context",
                "note": f"quality={citation.get('quality_score', 'n/a')} window={citation.get('time_window', {})}",
                "href": str(citation.get("uri") or ""),
            }
        )

    references.extend(
        [
            {
                "label": "DHBW lecture API",
                "kind": "external",
                "note": "Lecture feed used by lecture-ingest and weekly context.",
                "href": "https://api.dhbw.app/rapla/MA/lectures",
            },
            {
                "label": "DHBW ICS course feeds",
                "kind": "external",
                "note": "Historical course backfill source used by lecture-ingest.",
                "href": "https://api.dhbw.app/ics/{course}",
            },
            {
                "label": "Historical Excel backfill",
                "kind": "dataset",
                "note": "Local import path for historical occupancy data.",
                "href": "scripts/data/import_excel_counts.py",
            },
        ]
    )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in references:
        key = (str(item.get("label", "")).strip().lower(), str(item.get("href", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_weekly_outlook_payload(
    *,
    client: SitcheckApiClient,
    zone_id: str,
    weekly_days: int,
    weekly_slot_minutes: int,
    stale_seconds: int,
    history_minutes: int,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    horizon_minutes = max(1, int(weekly_days) * 24 * 60)

    weekly_forecast = client.get_weekly_forecast_latest(
        zone_id=zone_id,
        days=weekly_days,
        slot_minutes=weekly_slot_minutes,
        stale_seconds=stale_seconds,
    )
    if not weekly_forecast.get("points"):
        weekly_forecast = client.get_forecast_latest(
            zone_id=zone_id,
            horizon=horizon_minutes,
            stale_seconds=stale_seconds,
        )

    weekly_numeric_explainability: dict[str, Any]
    try:
        weekly_numeric_explainability = client.get_weekly_explainability(
            zone_id=zone_id,
            days=weekly_days,
            slot_minutes=weekly_slot_minutes,
            stale_seconds=stale_seconds,
        )
    except Exception as exc:
        try:
            weekly_numeric_explainability = client.tool_explain_forecast(zone_id=zone_id, horizon=horizon_minutes)
        except Exception:
            weekly_numeric_explainability = {
                "zone_id": zone_id,
                "horizon": horizon_minutes,
                "summary": f"Weekly numeric explainability unavailable: {exc.__class__.__name__}",
                "drivers": [],
                "uncertainty": {"score": 1.0, "level": "high", "reason": "upstream timeout"},
                "evidence": {"quality": {"score": 0.0, "flags": ["WEEKLY_XAI_UNAVAILABLE"]}},
            }

    try:
        weekly_context = client.get_explain_context(
            zone_id=zone_id,
            horizon=horizon_minutes,
            audience="ops",
            language="de",
            query="Weekly Outlook",
        )
    except Exception as exc:
        weekly_context = {
            "context_version": "fallback",
            "context": {
                "zone_id": zone_id,
                "horizon": horizon_minutes,
                "summary": f"Weekly explainability context unavailable: {exc.__class__.__name__}",
                "citation_map": [],
            },
        }

    lineage = client.get_model_lineage(zone_id=zone_id, product="weekly_slot")
    references = _build_weekly_reference_cards(
        forecast=weekly_forecast,
        lineage=lineage,
        weekly_context=weekly_context if isinstance(weekly_context, dict) else {},
    )

    return {
        "meta": {
            "generated_at": now.isoformat(),
            "zone_id": zone_id,
            "weekly_days": weekly_days,
            "weekly_slot_minutes": weekly_slot_minutes,
            "horizon_minutes": horizon_minutes,
            "history_minutes": history_minutes,
            "stale_seconds": stale_seconds,
            "environment": "dashboard",
        },
        "mode": weekly_forecast.get("source", "live"),
        "weekly_forecast": weekly_forecast,
        "weekly_numeric_explainability": weekly_numeric_explainability,
        "weekly_explainability": weekly_context,
        "lineage": lineage,
        "references": references,
    }


def render_command_center(client: SitcheckApiClient, state: DashboardState) -> None:
    try:
        payload = cached_command_center(
            state.api_base_url,
            state.zone_id,
            state.horizon,
            state.history_minutes,
            state.stale_threshold_sec,
            state.long_term_days,
        )
    except Exception as exc:
        st.warning(f"Command center endpoint unavailable, using fallback reads. reason={exc}")
        payload = build_command_center_fallback(client=client, state=state)

    render_header(payload)
    render_alert_rail(payload)
    render_kpis(payload)
    render_forecast_chart(payload)
    render_drivers_and_recommendations(payload)
    render_evidence(payload)
    render_scenario_panel(client=client, zone_id=state.zone_id, horizon=state.horizon)
    render_horizon_panel(payload)
    render_calendar(payload)

    with st.expander("Compact Data Drawer", expanded=False):
        history_points = payload.get("history", {}).get("points", [])
        if history_points:
            df = pd.DataFrame(history_points).tail(30)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No history points available.")


def _render_llm_widget(
    assistant: AssistantOrchestrator,
    state: DashboardState,
    audience: str,
    widget_key: str,
    default_prompt: str,
) -> None:
    st.markdown("---")
    st.markdown("### 💬 Frage an den Assistenten")

    role_config = [
        (
            "OPS",
            "ops",
            f"Gibt es aktuell Service-Probleme oder Qualitätsauffälligkeiten für Zone {state.zone_id}? "
            "Fasse kurz zusammen was sofort geprüft werden muss.",
        ),
        (
            "Executive",
            "executive",
            f"Schreibe einen Executive Brief für Zone {state.zone_id}: "
            "Aktuelle Lage, Risiko und eine klare Handlungsempfehlung.",
        ),
        (
            "Enduser",
            "enduser",
            f"Ist jetzt ein guter Zeitpunkt zum Lernen in der Bibliothek (Zone {state.zone_id})? "
            f"Wie entwickelt sich die Auslastung in den nächsten {state.horizon} Minuten?",
        ),
        (
            "Professor",
            "professor",
            f"Erkläre die wichtigsten Prognosetreiber und die Modellqualität für Zone {state.zone_id} heute. "
            "Welche Features dominieren und wie valide ist die aktuelle Unsicherheitsschätzung?",
        ),
        (
            "Freie Frage",
            None,
            "",
        ),
    ]

    tab_labels = [r[0] for r in role_config]
    tabs = st.tabs(tab_labels)

    for tab, (label, role, template) in zip(tabs, role_config):
        with tab:
            tab_key = f"{widget_key}_{label.lower().replace(' ', '_')}"
            resp_key = f"{tab_key}_response"

            user_prompt = st.text_area(
                "Frage",
                value=template,
                key=f"{tab_key}_text",
                height=100,
                label_visibility="collapsed",
                placeholder="Deine Frage an das LLM …",
            )

            if st.button("Senden", key=f"{tab_key}_btn", type="primary"):
                if not user_prompt.strip():
                    st.warning("Bitte zuerst eine Frage eingeben.")
                else:
                    with st.spinner("Analysiere …"):
                        try:
                            narrative, _ = assistant.run(
                                user_prompt,
                                state.zone_id,
                                state.horizon,
                                audience_override=role,
                            )
                            st.session_state[resp_key] = narrative
                        except Exception as exc:
                            st.session_state[resp_key] = f"Fehler: {exc}"

            if resp_key in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state[resp_key])


def render_nutzersicht(client: SitcheckApiClient, state: DashboardState, assistant: AssistantOrchestrator | None = None) -> None:
    try:
        payload = cached_command_center(
            state.api_base_url,
            state.zone_id,
            state.horizon,
            state.history_minutes,
            state.stale_threshold_sec,
            state.long_term_days,
        )
    except Exception as exc:
        st.error(f"Daten nicht verfügbar: {exc}")
        return

    live = payload.get("live") or {}
    occupancy = live.get("occupancy", 0) or 0
    capacity = live.get("capacity") or None
    utilization = live.get("utilization") or None

    forecast_points = (payload.get("forecast_latest") or {}).get("points") or []
    def _yhat(offset_min: int) -> str:
        for p in forecast_points:
            try:
                ts = datetime.fromisoformat(str(p.get("timestamp", "")).replace("Z", "+00:00"))
                delta = (ts - datetime.now(UTC)).total_seconds() / 60
                if abs(delta - offset_min) < 8:
                    v = p.get("yhat")
                    return str(round(v)) if v is not None else "–"
            except Exception:
                pass
        return "–"

    if utilization is not None:
        pct = round(utilization * 100) if utilization <= 1.0 else round(utilization)
        if pct >= 85:
            ampel = "🔴 Sehr voll"
        elif pct >= 60:
            ampel = "🟡 Mäßig belegt"
        else:
            ampel = "🟢 Gut zum Kommen"
        ampel_label = f"{ampel} ({pct}% Auslastung)"
    else:
        ampel_label = "Auslastung wird geladen …"

    st.markdown(f"## {ampel_label}")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Personen jetzt", occupancy)
    col2.metric("Kapazität", capacity if capacity else "–")
    col3.metric("Prognose +30 min", _yhat(30))
    col4.metric("Prognose +210 min", _yhat(210))

    alerts = payload.get("alerts") or []
    for alert in alerts:
        level = alert.get("level", "info")
        msg = f"**{alert.get('title', '')}** — {alert.get('message', '')}"
        if level == "critical":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)

    render_forecast_chart(payload)

    st.markdown("---")
    st.caption("Buchungen und Reservierungen sind über das Portal verfügbar.")

    if assistant:
        _render_llm_widget(
            assistant=assistant,
            state=state,
            audience="enduser",
            widget_key="llm_nutzersicht",
            default_prompt=f"Ist jetzt ein guter Zeitpunkt zum Lernen in der Bibliothek (Zone {state.zone_id})?",
        )


def render_betrieb(client: SitcheckApiClient, state: DashboardState, assistant: AssistantOrchestrator | None = None) -> None:
    try:
        payload = cached_command_center(
            state.api_base_url,
            state.zone_id,
            state.horizon,
            state.history_minutes,
            state.stale_threshold_sec,
            state.long_term_days,
        )
    except Exception as exc:
        st.error(f"Daten nicht verfügbar: {exc}")
        return

    render_header(payload)
    render_alert_rail(payload)

    st.markdown("### Service-Status")
    service_health = payload.get("service_health") or []
    if service_health:
        rows = []
        for svc in service_health:
            rows.append({
                "Service": svc.get("service", "–"),
                "Status": svc.get("status", "–"),
                "Latenz (ms)": svc.get("latency_ms", "–"),
                "Detail": svc.get("detail", "–"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Keine Service-Health-Daten verfügbar.")

    st.markdown("### Datenqualität & Modell")
    live = payload.get("live") or {}
    forecast_meta = payload.get("forecast_latest") or {}
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Score", round(live.get("quality_score", 0), 2))
    col2.metric("Modell-Version", forecast_meta.get("model_version", "–"))
    col3.metric("Forecast veraltet", "Ja" if forecast_meta.get("stale") else "Nein")

    flags = live.get("quality_flags") or []
    if flags:
        st.warning("Quality-Flags aktiv: " + ", ".join(flags))

    render_evidence(payload)

    if assistant:
        _render_llm_widget(
            assistant=assistant,
            state=state,
            audience="ops",
            widget_key="llm_betrieb",
            default_prompt=f"Gibt es aktuell Service-Probleme oder Qualitätsauffälligkeiten für Zone {state.zone_id}?",
        )


def render_technisch(client: SitcheckApiClient, state: DashboardState, assistant: AssistantOrchestrator | None = None) -> None:
    try:
        payload = cached_command_center(
            state.api_base_url,
            state.zone_id,
            state.horizon,
            state.history_minutes,
            state.stale_threshold_sec,
            state.long_term_days,
        )
    except Exception as exc:
        st.error(f"Daten nicht verfügbar: {exc}")
        return

    render_header(payload)
    render_drivers_and_recommendations(payload)

    st.markdown("### Feature-Treiber (Tabelle)")
    drivers = (payload.get("explanation") or {}).get("drivers") or []
    if drivers:
        df = pd.DataFrame([
            {
                "Feature": d.get("name", "–"),
                "Richtung": d.get("direction", "–"),
                "Gewicht": round(d.get("importance", 0), 3),
            }
            for d in drivers
        ]).sort_values("Gewicht", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Keine Feature-Treiber verfügbar.")

    st.markdown("### Unsicherheit")
    uncertainty = (payload.get("explanation") or {}).get("uncertainty") or {}
    if uncertainty:
        col1, col2 = st.columns(2)
        col1.metric("Niveau", uncertainty.get("level", "–"))
        score = uncertainty.get("score")
        col2.metric("Score", f"{round(score * 100)}%" if score is not None else "–")
    else:
        st.info("Keine Unsicherheitsdaten verfügbar.")

    try:
        weekly = cached_weekly_outlook(
            state.api_base_url,
            state.zone_id,
            state.weekly_days,
            state.weekly_slot_minutes,
            state.stale_threshold_sec,
        )
        lineage = weekly.get("lineage") or {}
        if lineage:
            st.markdown("### Modell-Lineage")
            col1, col2, col3 = st.columns(3)
            col1.metric("Backend", lineage.get("model_backend", "–"))
            col2.metric("Promoted", "Ja" if lineage.get("promoted") else "Nein")
            col3.metric("Version", str(lineage.get("model_version", "–")))
    except Exception:
        pass

    if assistant:
        _render_llm_widget(
            assistant=assistant,
            state=state,
            audience="professor",
            widget_key="llm_technisch",
            default_prompt=f"Erkläre die wichtigsten Prognosetreiber und die Modellqualität für Zone {state.zone_id} heute.",
        )


def render_weekly_page(client: SitcheckApiClient, state: DashboardState) -> None:
    try:
        payload = cached_weekly_outlook(
            state.api_base_url,
            state.zone_id,
            state.weekly_days,
            state.weekly_slot_minutes,
            state.stale_threshold_sec,
            state.history_minutes,
        )
    except Exception as exc:
        st.warning(f"Weekly endpoint unavailable, using graceful fallback. reason={exc}")
        payload = build_weekly_outlook_payload(
            client=client,
            zone_id=state.zone_id,
            weekly_days=state.weekly_days,
            weekly_slot_minutes=state.weekly_slot_minutes,
            stale_seconds=state.stale_threshold_sec,
            history_minutes=state.history_minutes,
        )

    render_weekly_outlook(payload)


def render_assistant_page(assistant: AssistantOrchestrator, state: DashboardState) -> None:
    st.markdown("## Assistant")
    st.caption("Allowed tools: history, forecast, explain, recommendations, simulate_scenario")

    templates = [
        {
            "label": "Auslastung in 2 Minuten",
            "prompt": (
                f"Erklaere die Auslastung fuer Zone {state.zone_id} in 2 Minuten. "
                "Bitte erst verstaendlicher Text, dann Werte kompakt."
            ),
        },
        {
            "label": "Top 3 Verbesserungen",
            "prompt": (
                f"Was sind die 3 wichtigsten Verbesserungsmassnahmen fuer Zone {state.zone_id} "
                "mit Aufwand, erwarteter Wirkung und naechstem Schritt?"
            ),
        },
        {
            "label": "Prognose-Risiken",
            "prompt": (
                f"Welche Risiken hat die Prognose fuer Zone {state.zone_id} "
                f"(Horizont {state.horizon}) und wie sollten wir damit umgehen?"
            ),
        },
        {
            "label": "Professoren-Briefing",
            "prompt": (
                f"Ich bereite ein Gespraech mit Professoren vor. "
                f"Erklaere die Auslastung fuer Zone {state.zone_id} mit klaren Handlungsvorschlaegen fuer morgen."
            ),
        },
        {
            "label": "Executive Brief",
            "prompt": (
                f"Schreibe einen Executive Brief für Zone {state.zone_id} "
                "mit Zusammenfassung, Risiko und klarer Handlungsempfehlung."
            ),
        },
        {
            "label": "Studentensicht",
            "prompt": (
                f"Erkläre die aktuelle Lage für Studierende in einfacher Sprache für Zone {state.zone_id} "
                f"und die nächsten {state.horizon} Minuten."
            ),
        },
    ]

    st.markdown("### Vorlagen")
    audience_override = st.selectbox(
        "Audience (Override)",
        options=["auto", "ops", "executive", "enduser", "professor"],
        index=0,
        help="Bei 'auto' wird die Zielgruppe aus der Anfrage erkannt.",
    )
    selected_template_prompt = ""
    template_cols = st.columns(3)
    for idx, item in enumerate(templates):
        with template_cols[idx % 3]:
            if st.button(item["label"], key=f"assistant_template_{idx}", use_container_width=True):
                selected_template_prompt = item["prompt"]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("payload") is not None:
                narrative = msg["payload"].get("narrative", {})
                meta = narrative.get("meta", {}) if isinstance(narrative, dict) else {}
                structured = narrative.get("structured", {}) if isinstance(narrative, dict) else {}
                if meta:
                    st.caption(
                        "Explainability Meta: "
                        f"template={meta.get('template_set_id', '-')} | "
                        f"version={meta.get('prompt_version', '-')} | "
                        f"model={meta.get('model', '-')} | "
                        f"mode={msg['payload'].get('mode', '-')} | "
                        f"fallback={meta.get('fallback_reason') or 'none'}"
                    )
                if isinstance(structured, dict) and structured:
                    verdict = structured.get("verdict", "-")
                    confidence = structured.get("confidence_statement", "-")
                    st.caption(f"Urteil: {verdict} | Confidence: {confidence}")
                with st.expander("Technische Details", expanded=False):
                    st.json(msg["payload"])

    prompt = st.chat_input("Ask for operations status, explainability, or scenario simulation")
    if selected_template_prompt:
        prompt = selected_template_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        selected_audience = None if audience_override == "auto" else audience_override
        try:
            narrative, payload = assistant.run(
                prompt,
                state.zone_id,
                state.horizon,
                audience_override=selected_audience,
            )
        except Exception as exc:
            narrative = (
                "Kurzfazit: Der Assistant konnte gerade keine vollständige Analyse laden.\n\n"
                f"Fehler: {exc}\n\n"
                "Bitte erneut versuchen oder zuerst die Command-Center-Daten prüfen."
            )
            payload = {
                "mode": "assistant_runtime_fallback",
                "narrative": {
                    "meta": {
                        "template_set_id": "runtime-fallback",
                        "prompt_version": "v1",
                        "model": "none",
                        "fallback_reason": "assistant_exception",
                    },
                    "structured": {},
                },
                "error": str(exc),
            }
        st.session_state.messages.append({"role": "assistant", "content": narrative, "payload": payload})
        with st.chat_message("assistant"):
            st.markdown(narrative)
            narrative_meta = payload.get("narrative", {}).get("meta", {})
            if narrative_meta:
                st.caption(
                    "Explainability Meta: "
                    f"template={narrative_meta.get('template_set_id', '-')} | "
                    f"version={narrative_meta.get('prompt_version', '-')} | "
                    f"model={narrative_meta.get('model', '-')} | "
                    f"mode={payload.get('mode', '-')} | "
                    f"fallback={narrative_meta.get('fallback_reason') or 'none'}"
                )
            structured = payload.get("narrative", {}).get("structured", {})
            if isinstance(structured, dict) and structured:
                verdict = structured.get("verdict", "-")
                confidence = structured.get("confidence_statement", "-")
                st.caption(f"Urteil: {verdict} | Confidence: {confidence}")
            with st.expander("Technische Details", expanded=False):
                st.json(payload)


state = sidebar_state(zone_candidates=[])
try_auto_refresh(state.auto_refresh, state.refresh_interval_sec)

client = SitcheckApiClient(
    state.api_base_url,
    timeout=float(os.getenv("DASHBOARD_API_TIMEOUT_SECONDS", "30")),
)
assistant = AssistantOrchestrator(
    client=client,
    ollama_enabled=state.ollama_enabled,
    ollama_model=state.ollama_model,
    ollama_base_url=state.ollama_base_url,
    local_template_fallback_enabled=(
        os.getenv("ASSISTANT_LOCAL_TEMPLATE_FALLBACK", "false").lower() == "true"
    ),
)

try:
    health = cached_health(state.api_base_url)
    st.success(f"API reachable: {health.get('service')} ({health.get('status')})")
    try:
        zones = cached_zones(state.api_base_url)
        if zones and not any(z.get("zone_id") == state.zone_id for z in zones):
            known = ", ".join(str(z.get("zone_id")) for z in zones)
            st.info(f"Known zones: {known}")
    except Exception:
        pass
except Exception as exc:
    st.error(f"API not reachable: {exc}")

if state.view == "Command Center":
    render_command_center(client=client, state=state)
elif state.view == "Nutzersicht":
    render_nutzersicht(client=client, state=state, assistant=assistant)
elif state.view == "Betrieb":
    render_betrieb(client=client, state=state, assistant=assistant)
elif state.view == "Technisch":
    render_technisch(client=client, state=state, assistant=assistant)
elif state.view == "Weekly Outlook":
    render_weekly_page(client=client, state=state)
elif state.view == "Forecast Lab":
    render_forecast_lab(
        client=client,
        zone_id=state.zone_id,
        horizon=state.horizon,
        long_term_days=state.long_term_days,
        stale_threshold_sec=state.stale_threshold_sec,
    )
else:
    render_assistant_page(assistant=assistant, state=state)
