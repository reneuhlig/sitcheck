from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import SitcheckApiClient
from ui.state import HORIZON_PANEL_PRESETS


def _badge(label: str, level: str) -> str:
    css = {
        "ok": "cc-badge-ok",
        "info": "cc-badge-info",
        "warn": "cc-badge-warn",
        "risk": "cc-badge-risk",
    }.get(level, "cc-badge-info")
    return f'<span class="cc-badge {css}">{label}</span>'


def _service_level(status: str) -> str:
    if status == "ok":
        return "ok"
    if status == "degraded":
        return "warn"
    return "risk"


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


def _points_frame(points: list[dict[str, Any]]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["timestamp", "yhat", "pi_low", "pi_high"])
    df = pd.DataFrame(points).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("yhat", "pi_low", "pi_high"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp"])
    return df


def _forecast_quality_flags(forecast: dict[str, Any]) -> list[str]:
    evidence = forecast.get("evidence", {}) if isinstance(forecast.get("evidence"), dict) else {}
    quality = evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}
    flags = quality.get("flags", []) if isinstance(quality.get("flags"), list) else []
    return [str(flag).upper() for flag in flags if str(flag).strip()]


def _is_degraded_forecast(forecast: dict[str, Any]) -> bool:
    model_version = str(forecast.get("model_version", "")).lower()
    flags = _forecast_quality_flags(forecast)
    return (
        bool(forecast.get("stale", False))
        or "baseline" in model_version
        or any(flag in {"CONTEXT_STALE", "NO_CONTEXT_DB", "TF_FALLBACK", "LGBM_FALLBACK"} for flag in flags)
        or any(flag.startswith("TF_FALLBACK:") or flag.startswith("LGBM_FALLBACK:") for flag in flags)
    )


def _render_reference_cards(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("No references available.")
        return

    for item in items:
        label = str(item.get("label", "Reference"))
        href = str(item.get("href", "")).strip()
        note = str(item.get("note", "")).strip()
        detail = str(item.get("detail", "")).strip()
        kind = str(item.get("kind", "source"))
        lines = [f"**{label}**", f"`{kind}`"]
        if note:
            lines.append(note)
        if detail:
            lines.append(detail)
        if href.startswith("http://") or href.startswith("https://"):
            lines.append(f"[Open source]({href})")
        elif href:
            lines.append(f"`{href}`")
        st.markdown("\n\n".join(lines))


def _lineage_badges(lineage: dict[str, Any]) -> str:
    model = lineage.get("model", {}) if isinstance(lineage.get("model"), dict) else {}
    training = lineage.get("training", {}) if isinstance(lineage.get("training"), dict) else {}
    parts = [
        _badge(f"backend={model.get('backend', 'n/a')}", "info"),
        _badge(f"version={model.get('version', 'n/a')}", "info"),
        _badge("promoted" if model.get("promoted") else "not promoted", "ok" if model.get("promoted") else "warn"),
    ]
    if training.get("run_id"):
        parts.append(_badge(f"run={training.get('run_id')}", "info"))
    return "".join(parts)


def _default_weekly_references(payload: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    lineage = payload.get("lineage", {}) if isinstance(payload.get("lineage"), dict) else {}
    forecast = payload.get("weekly_forecast", {}) if isinstance(payload.get("weekly_forecast"), dict) else {}
    evidence = forecast.get("evidence", {}) if isinstance(forecast.get("evidence"), dict) else {}

    for source in evidence.get("sources", []) if isinstance(evidence.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        references.append(
            {
                "label": f"{source.get('type', 'source')}: {source.get('id', 'unknown')}",
                "kind": "evidence",
                "note": source.get("note") or "From forecast evidence",
                "href": str(source.get("uri") or ""),
            }
        )

    training = lineage.get("training", {}) if isinstance(lineage.get("training"), dict) else {}
    data_refs = training.get("data_references", []) if isinstance(training.get("data_references"), list) else []
    for ref in data_refs:
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


def render_header(payload: dict[str, Any]) -> None:
    meta = payload.get("meta", {})
    service_health = payload.get("service_health", [])
    zone_id = str(meta.get("zone_id", "unknown"))
    generated_at = str(meta.get("generated_at", "n/a"))
    environment = str(meta.get("environment", "dev"))

    st.markdown(
        f"""
        <div class="cc-panel cc-fade-in">
          <h1 style="margin:0 0 .4rem 0;">Sitcheck Command Center</h1>
          <div style="color:#bcd0ec;">Zone <b>{zone_id}</b> | Last refresh <span class="mono">{generated_at}</span> | Env <b>{environment}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    badge_html = "".join(
        _badge(f"{item.get('service', 'service')}: {item.get('status', 'unknown')}", _service_level(str(item.get("status", "down"))))
        for item in service_health
    )
    st.markdown(f"<div style='margin-top:.5rem'>{badge_html}</div>", unsafe_allow_html=True)


def render_alert_rail(payload: dict[str, Any]) -> None:
    alerts = payload.get("alerts", [])
    if not alerts:
        return
    badges = "".join(
        _badge(str(item.get("code", "ALERT")), str(item.get("level", "info")))
        for item in alerts
    )
    st.markdown(
        f"""
        <div class="cc-panel cc-fade-in" style="margin-top:.4rem;">
          <div style="font-weight:700; margin-bottom:.3rem;">Alert Rail</div>
          <div>{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(payload: dict[str, Any]) -> None:
    live = payload.get("live", {})
    forecast = payload.get("forecast_latest", {})
    points = forecast.get("points", [])
    uncertainty_avg = 0.0
    if points:
        uncertainty_avg = sum(
            max(0.0, float(p.get("pi_high", 0.0)) - float(p.get("pi_low", 0.0)))
            for p in points
        ) / len(points)
    peak = max((float(p.get("yhat", 0.0)) for p in points), default=0.0)

    age_seconds = forecast.get("age_seconds")
    if not isinstance(age_seconds, int):
        generated_at = str(forecast.get("generated_at", ""))
        try:
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            age_seconds = max(0, int((datetime.now(UTC) - dt.astimezone(UTC)).total_seconds()))
        except Exception:
            age_seconds = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Live Occupancy", int(live.get("occupancy", 0)))
    c2.metric("Utilization", f"{float(live.get('utilization', 0.0)):.2f}")
    c3.metric("Snapshot Age", f"{age_seconds if age_seconds is not None else 'n/a'} sec")
    c4.metric("Model Version", str(forecast.get("model_version", "n/a")))
    c5.metric("Peak Horizon", f"{peak:.1f}")
    c6.metric("Avg Uncertainty", f"{uncertainty_avg:.2f}")


def render_forecast_chart(payload: dict[str, Any]) -> None:
    history = payload.get("history", {})
    history_points = history.get("points", [])
    forecast = payload.get("forecast_latest", {})
    forecast_points = forecast.get("points", [])

    st.markdown("### Forecast + History")
    if _is_degraded_forecast(forecast):
        flags = ", ".join(_forecast_quality_flags(forecast)[:5]) or "STALE"
        st.warning(f"Forecast degraded: {flags}")
    fig = go.Figure()
    if history_points:
        hdf = pd.DataFrame(history_points)
        hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], utc=True)
        fig.add_trace(go.Scatter(x=hdf["timestamp"], y=hdf["occupancy"], mode="lines", name="history"))

    if forecast_points:
        fdf = pd.DataFrame(forecast_points)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        fig.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["yhat"], mode="lines", name="yhat"))
        fig.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["pi_low"], mode="lines", name="pi_low"))
        fig.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["pi_high"], mode="lines", name="pi_high"))
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_drivers_and_recommendations(payload: dict[str, Any]) -> None:
    explanation = payload.get("explanation", {})
    recommendations = payload.get("recommendations", {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Top Drivers")
        st.write(explanation.get("summary", "n/a"))
        drivers = explanation.get("drivers", [])
        if drivers:
            ddf = pd.DataFrame(drivers)[:5]
            st.dataframe(ddf, use_container_width=True)
            fig = go.Figure(
                go.Bar(
                    x=ddf["impact"],
                    y=ddf["name"],
                    orientation="h",
                    marker_color="#2ea3ff",
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drivers available.")

    with col_b:
        st.markdown("### Recommendations")
        st.write(recommendations.get("summary", "n/a"))
        gates = recommendations.get("gates", {})
        gate_badges = "".join(
            [
                _badge(f"quality_ok={gates.get('quality_ok', False)}", "ok" if gates.get("quality_ok") else "warn"),
                _badge(
                    f"uncertainty_ok={gates.get('uncertainty_ok', False)}",
                    "ok" if gates.get("uncertainty_ok") else "warn",
                ),
            ]
        )
        st.markdown(gate_badges, unsafe_allow_html=True)

        actions = recommendations.get("actions", [])
        if actions:
            for action in actions:
                st.markdown(
                    f"""
                    <div class="cc-panel" style="margin-bottom:.4rem;">
                      <div style="font-weight:700;">{action.get("action_type", "action")} (P{action.get("priority", "n/a")})</div>
                      <div>{action.get("rationale", "")}</div>
                      <div class="mono" style="margin-top:.2rem; color:#b6c9e5;">
                        impact: {action.get("expected_impact", {})}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No actions generated.")


def render_evidence(payload: dict[str, Any]) -> None:
    with st.expander("Evidence / Citations", expanded=False):
        st.json(
            {
                "forecast": payload.get("forecast_latest", {}).get("evidence", {}),
                "explanation": payload.get("explanation", {}).get("evidence", {}),
                "recommendations": payload.get("recommendations", {}).get("evidence", {}),
            }
        )


def render_references_and_lineage(payload: dict[str, Any]) -> None:
    st.markdown("### References & Lineage")
    lineage = payload.get("lineage", {}) if isinstance(payload.get("lineage"), dict) else {}
    references = payload.get("references", [])
    if not isinstance(references, list) or not references:
        references = _default_weekly_references(payload)

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        if lineage:
            st.markdown(_lineage_badges(lineage), unsafe_allow_html=True)
            st.json(lineage)
        else:
            st.info("No lineage metadata available yet.")
    with c2:
        _render_reference_cards(references)


def render_weekly_outlook(payload: dict[str, Any]) -> None:
    meta = payload.get("meta", {})
    weekly_forecast = payload.get("weekly_forecast", {})
    weekly_numeric = payload.get("weekly_numeric_explainability", {})
    weekly_context = payload.get("weekly_explainability", {})
    lineage = payload.get("lineage", {})
    references = payload.get("references", [])
    points = _points_frame(list(weekly_forecast.get("points", [])))

    st.markdown("## Weekly Outlook")
    st.caption(
        "7-day slot view with numeric-first explainability, weekly references and lineage."
    )

    mode = str(payload.get("mode", "fallback"))
    source = str(weekly_forecast.get("source", "unknown"))
    badges = "".join(
        [
            _badge(f"mode={mode}", "info"),
            _badge(f"source={source}", "info"),
            _badge(f"days={meta.get('weekly_days', 7)}", "info"),
            _badge(f"slot={meta.get('weekly_slot_minutes', 60)}m", "info"),
        ]
    )
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("### Week Overview")
    horizon_minutes = _safe_int(weekly_forecast.get("horizon"), _safe_int(meta.get("weekly_days", 7)) * 24 * 60)
    generated_at = str(weekly_forecast.get("generated_at", meta.get("generated_at", "n/a")))
    model_version = str(weekly_forecast.get("model_version", "n/a"))
    peak = float(points["yhat"].max()) if not points.empty else 0.0
    avg = float(points["yhat"].mean()) if not points.empty else 0.0
    width = float((points["pi_high"] - points["pi_low"]).mean()) if not points.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon", f"{horizon_minutes} min")
    c2.metric("Peak yhat", f"{peak:.1f}")
    c3.metric("Avg yhat", f"{avg:.1f}")
    c4.metric("Avg uncertainty", f"{width:.2f}")
    st.caption(f"Generated at {generated_at} | Model {model_version}")

    if not points.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points["timestamp"], y=points["yhat"], mode="lines", name="yhat"))
        fig.add_trace(go.Scatter(x=points["timestamp"], y=points["pi_low"], mode="lines", name="pi_low"))
        fig.add_trace(go.Scatter(x=points["timestamp"], y=points["pi_high"], mode="lines", name="pi_high"))
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weekly forecast points available yet. Fallback view is shown.")

    st.markdown("### Daily Slot Grid")
    if not points.empty:
        slot_minutes = max(1, _safe_int(meta.get("weekly_slot_minutes"), 60))
        slot_index = ((points["timestamp"].dt.hour * 60 + points["timestamp"].dt.minute) // slot_minutes).astype(int)
        grid = points.copy()
        grid["day"] = points["timestamp"].dt.strftime("%a %d.%m")
        grid["slot_label"] = slot_index.map(lambda idx: f"S{int(idx) + 1:02d}")
        heat = grid.pivot_table(index="day", columns="slot_label", values="yhat", aggfunc="mean")
        heat = heat.reindex(
            sorted(heat.columns, key=lambda label: int(str(label).lstrip("S")) if str(label).lstrip("S").isdigit() else 0),
            axis=1,
        )
        if len(heat.index) > 0 and len(heat.columns) > 0:
            fig = go.Figure(
                data=go.Heatmap(
                    z=heat.values,
                    x=heat.columns.tolist(),
                    y=heat.index.tolist(),
                    colorscale="Blues",
                    colorbar=dict(title="yhat"),
                )
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        daily = grid.groupby("day", as_index=False).agg(
            peak_yhat=("yhat", "max"),
            avg_yhat=("yhat", "mean"),
        )
        daily_widths = (
            grid.assign(width=grid["pi_high"] - grid["pi_low"])
            .groupby("day", as_index=False)
            .agg(avg_width=("width", "mean"))
        )
        daily = daily.merge(daily_widths, on="day", how="left")
        st.dataframe(daily, use_container_width=True)
    else:
        st.info("Daily slot grid unavailable without forecast points.")

    st.markdown("### Drivers & Context")
    weekly_context_summary = (
        weekly_context.get("context", {}) if isinstance(weekly_context, dict) else {}
    )
    summary = str(
        weekly_numeric.get("summary")
        or weekly_context_summary.get("summary")
        or weekly_context.get("summary")
        or "n/a"
    )
    st.write(summary)
    drivers = weekly_numeric.get("drivers", []) if isinstance(weekly_numeric.get("drivers"), list) else []
    if drivers:
        driver_df = pd.DataFrame(drivers).head(8)
        st.dataframe(driver_df, use_container_width=True)
        if {"impact", "name"}.issubset(driver_df.columns):
            fig = go.Figure(
                go.Bar(
                    x=driver_df["impact"],
                    y=driver_df["name"],
                    orientation="h",
                    marker_color="#2ea3ff",
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weekly drivers available yet.")

    if isinstance(weekly_context, dict) and weekly_context:
        with st.expander("Weekly explainability context", expanded=False):
            st.json(weekly_context)

    st.markdown("### Uncertainty & Quality")
    uncertainty = weekly_numeric.get("uncertainty", {}) if isinstance(weekly_numeric.get("uncertainty"), dict) else {}
    quality = weekly_forecast.get("evidence", {}).get("quality", {}) if isinstance(weekly_forecast.get("evidence"), dict) else {}
    cols = st.columns(3)
    cols[0].metric("Uncertainty level", str(uncertainty.get("level", "n/a")))
    cols[1].metric("Quality score", f"{_safe_float(quality.get('score', 0.0)):.2f}")
    cols[2].metric("Quality flags", str(len(quality.get("flags", []) if isinstance(quality.get("flags"), list) else [])))
    if uncertainty:
        st.caption(str(uncertainty.get("reason", "No uncertainty detail available.")))

    render_references_and_lineage({"lineage": lineage, "references": references, "weekly_forecast": weekly_forecast})


def render_horizon_panel(payload: dict[str, Any]) -> None:
    st.markdown("### Horizon Snapshots")
    items = payload.get("forecast_long_term", [])
    if not items:
        st.info("No long-term snapshots available.")
        return

    rows = []
    for item in items:
        rows.append(
            {
                "horizon_min": item.get("horizon"),
                "age_seconds": item.get("age_seconds"),
                "stale": item.get("stale"),
                "model_version": item.get("model_version"),
                "source": item.get("source"),
                "generated_at": item.get("generated_at"),
            }
        )
    df = pd.DataFrame(rows).sort_values("horizon_min")
    st.dataframe(df, use_container_width=True)

    selected_horizon = st.selectbox("Inspect horizon", options=sorted(set(df["horizon_min"].tolist()) | set(HORIZON_PANEL_PRESETS)))
    selected = next((item for item in items if int(item.get("horizon", 0)) == int(selected_horizon)), None)
    if selected:
        points = selected.get("points", [])
        if points:
            sdf = pd.DataFrame(points)
            sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sdf["timestamp"], y=sdf["yhat"], mode="lines", name="yhat"))
            fig.add_trace(go.Scatter(x=sdf["timestamp"], y=sdf["pi_low"], mode="lines", name="pi_low"))
            fig.add_trace(go.Scatter(x=sdf["timestamp"], y=sdf["pi_high"], mode="lines", name="pi_high"))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)


def render_calendar(payload: dict[str, Any]) -> None:
    st.markdown("### Calendar Context")
    events = payload.get("calendar_events", [])
    if not events:
        st.info("No calendar events in selected window.")
        return
    df = pd.DataFrame(events)
    for col in ("starts_at", "ends_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    keep = [c for c in ["title", "category", "starts_at", "ends_at", "expected_impact", "source"] if c in df.columns]
    st.dataframe(df[keep], use_container_width=True)


def render_scenario_panel(client: SitcheckApiClient, zone_id: str, horizon: int) -> None:
    st.markdown("### Counterfactual Simulation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        capacity_delta = st.number_input("capacity_delta", value=0, step=5)
    with c2:
        open_room = st.checkbox("open_room", value=False)
    with c3:
        push_time = st.number_input("push_time_minutes", value=0, step=5)
    with c4:
        staff_delta = st.number_input("staff_delta", value=0, step=1)

    if st.button("Simulate scenario", use_container_width=True):
        result = client.tool_simulate_scenario(
            zone_id=zone_id,
            horizon=horizon,
            changes={
                "capacity_delta": int(capacity_delta),
                "open_room": bool(open_room),
                "push_time_minutes": int(push_time),
                "staff_delta": int(staff_delta),
            },
            persist=False,
        )
        st.json(result)


def render_forecast_lab(
    *,
    client: SitcheckApiClient,
    zone_id: str,
    horizon: int,
    long_term_days: int,
    stale_threshold_sec: int,
) -> None:
    st.markdown("## Forecast Lab")
    st.caption("Deep-dive for long-range snapshots, history and scenario checks.")

    horizon_options = sorted(set(HORIZON_PANEL_PRESETS + [horizon, long_term_days * 24 * 60]))
    selected_horizon = st.selectbox("Horizon (minutes)", options=horizon_options, index=horizon_options.index(horizon) if horizon in horizon_options else 0)

    latest = client.get_forecast_latest(zone_id=zone_id, horizon=int(selected_horizon), stale_seconds=stale_threshold_sec)
    st.write(latest.get("summary", "n/a"))
    st.caption(
        f"horizon={latest.get('horizon')} | source={latest.get('source')} | model={latest.get('model_version')} | age={latest.get('age_seconds')}s"
    )

    points = latest.get("points", [])
    if points:
        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["yhat"], mode="lines", name="yhat"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pi_low"], mode="lines", name="pi_low"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pi_high"], mode="lines", name="pi_high"))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No forecast points available for selected horizon.")

    now = datetime.now(UTC)
    history = client.get_forecast_history(
        zone_id=zone_id,
        horizon=int(selected_horizon),
        from_iso=(now - timedelta(days=7)).isoformat(),
        to_iso=now.isoformat(),
        limit=200,
        stale_seconds=stale_threshold_sec,
    )
    items = history.get("items", [])
    st.markdown("### Snapshot History (7 days)")
    if items:
        hdf = pd.DataFrame(items)
        if "generated_at" in hdf.columns:
            hdf["generated_at"] = pd.to_datetime(hdf["generated_at"], utc=True)
        keep = [c for c in ["generated_at", "model_version", "source", "age_seconds", "stale"] if c in hdf.columns]
        st.dataframe(hdf[keep].head(100), use_container_width=True)
    else:
        st.info("No snapshot history available yet.")
