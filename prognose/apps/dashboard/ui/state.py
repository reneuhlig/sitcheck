from __future__ import annotations

import os
from dataclasses import dataclass

import requests
import streamlit as st


HORIZON_PANEL_PRESETS = [60, 1440, 10080, 20160]
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_MODEL_FALLBACK_OPTIONS = [
    "qwen2.5:0.5b",
    "gemma3:4b",
    "gemma3:12b",
]


@dataclass
class DashboardState:
    view: str
    api_base_url: str
    zone_id: str
    horizon: int
    history_minutes: int
    long_term_days: int
    weekly_days: int
    weekly_slot_minutes: int
    auto_refresh: bool
    refresh_interval_sec: int
    stale_threshold_sec: int
    ollama_enabled: bool
    ollama_model: str
    ollama_base_url: str


def try_auto_refresh(enabled: bool, refresh_interval_sec: int) -> None:
    if not enabled:
        return
    try:
        st.autorefresh(interval=refresh_interval_sec * 1000, key="command-center-refresh")
    except Exception:
        st.sidebar.warning("Auto refresh is not available in this Streamlit build.")


@st.cache_data(ttl=20, show_spinner=False)
def fetch_ollama_models(ollama_base_url: str) -> list[str]:
    base = str(ollama_base_url or "").strip().rstrip("/")
    if not base:
        return []
    try:
        response = requests.get(f"{base}/api/tags", timeout=(2.0, 5.0))
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)

    # Preserve order, remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def sidebar_state(zone_candidates: list[str] | None = None) -> DashboardState:
    candidates = zone_candidates or []
    default_zone = os.getenv("DEFAULT_ZONE_ID", "default-zone")
    default_horizon = max(15, min(240, int(os.getenv("DEFAULT_FORECAST_HORIZON_MINUTES", "210"))))
    weekly_days = int(os.getenv("WEEKLY_OUTLOOK_DAYS", "7"))
    weekly_slot_minutes = int(os.getenv("WEEKLY_OUTLOOK_SLOT_MINUTES", "60"))

    st.sidebar.header("Sitcheck Command Center")
    view = st.sidebar.radio(
        "View",
        options=["Command Center", "Nutzersicht", "Betrieb", "Technisch", "Weekly Outlook", "Forecast Lab", "Assistant"],
        index=0,
    )
    api_base_url = st.sidebar.text_input("API Base URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))

    if candidates:
        default_idx = candidates.index(default_zone) if default_zone in candidates else 0
        zone_id = st.sidebar.selectbox("Zone ID", options=candidates, index=default_idx)
    else:
        zone_id = st.sidebar.text_input("Zone ID", value=default_zone)

    horizon = st.sidebar.slider("Forecast Horizon (minutes)", min_value=15, max_value=240, value=default_horizon, step=15)
    long_term_days = st.sidebar.slider("Long-term Horizon (days)", min_value=7, max_value=42, value=14, step=1)
    history_minutes = st.sidebar.slider("History Window (minutes)", min_value=30, max_value=720, value=180, step=30)

    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval_sec = st.sidebar.slider(
        "Refresh Interval (sec)",
        min_value=5,
        max_value=120,
        value=int(os.getenv("DASHBOARD_AUTO_REFRESH_SECONDS", "15")),
        step=5,
    )
    stale_threshold_sec = st.sidebar.slider(
        "Stale Threshold (sec)",
        min_value=60,
        max_value=3600,
        value=int(os.getenv("FORECAST_STALE_THRESHOLD_SECONDS", "900")),
        step=30,
    )

    ollama_enabled = st.sidebar.toggle("Enable Ollama Narrative", value=os.getenv("OLLAMA_ENABLED", "false").lower() == "true")
    ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    discovered_models = fetch_ollama_models(ollama_base_url) if ollama_enabled else []
    model_options = discovered_models or OLLAMA_MODEL_FALLBACK_OPTIONS.copy()
    if DEFAULT_OLLAMA_MODEL not in model_options:
        model_options.insert(0, DEFAULT_OLLAMA_MODEL)
    default_index = model_options.index(DEFAULT_OLLAMA_MODEL) if DEFAULT_OLLAMA_MODEL in model_options else 0
    ollama_model = st.sidebar.selectbox("Ollama Model", options=model_options, index=default_index)
    if ollama_enabled and not discovered_models:
        st.sidebar.caption("Hinweis: Konnte Modelle nicht live laden, zeige Fallback-Liste.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Agent pattern inspired by the Medium blueprint: Query/Plot/Analysis/RAG")

    return DashboardState(
        view=view,
        api_base_url=api_base_url,
        zone_id=zone_id,
        horizon=horizon,
        history_minutes=history_minutes,
        long_term_days=long_term_days,
        weekly_days=weekly_days,
        weekly_slot_minutes=weekly_slot_minutes,
        auto_refresh=auto_refresh,
        refresh_interval_sec=refresh_interval_sec,
        stale_threshold_sec=stale_threshold_sec,
        ollama_enabled=ollama_enabled,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )
