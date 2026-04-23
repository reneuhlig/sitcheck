from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --surface: #ffffff;
            --surface-soft: #f9fbfe;
            --border: #d9e0ea;
            --border-strong: #c3cedd;
            --text: #111827;
            --muted: #5d6b82;
            --muted-2: #7b8798;
            --ok: #15803d;
            --ok-bg: #eaf7ef;
            --ok-border: #b9e2c6;
            --info: #2563eb;
            --info-bg: #eef5ff;
            --info-border: #bdd4ff;
            --warn: #b45309;
            --warn-bg: #fff7e6;
            --warn-border: #f4d18c;
            --risk: #b91c1c;
            --risk-bg: #fff1f0;
            --risk-border: #ffc2bd;
            --shadow: 0 16px 42px rgba(30, 41, 59, 0.08);
            --accent: #2563eb;
        }

        /* ── Global ──────────────────────────────────────── */
        .stApp {
            background:
                radial-gradient(circle at 12% 14%, rgba(188, 228, 254, 0.4), transparent 28%),
                radial-gradient(circle at 82% 18%, rgba(139, 198, 236, 0.22), transparent 25%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(244, 250, 255, 0.96)),
                #f4faff !important;
            color: var(--text);
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.94) !important;
            backdrop-filter: blur(10px);
        }

        /* ── Sidebar ─────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label {
            color: var(--text) !important;
        }

        /* ── Panels ──────────────────────────────────────── */
        .cc-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 16px;
            margin-bottom: 14px;
        }

        /* ── Status pills (badges) ───────────────────────── */
        .cc-badge {
            display: inline-flex;
            min-height: 28px;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 800;
            white-space: nowrap;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .cc-badge-ok   { border-color: var(--ok-border);   color: var(--ok);   background: var(--ok-bg); }
        .cc-badge-info { border-color: var(--info-border); color: var(--info); background: var(--info-bg); }
        .cc-badge-warn { border-color: var(--warn-border); color: var(--warn); background: var(--warn-bg); }
        .cc-badge-risk { border-color: var(--risk-border); color: var(--risk); background: var(--risk-bg); }

        /* ── Decision tiles ──────────────────────────────── */
        .sc-tile-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        .sc-tile {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 16px;
            min-height: 110px;
            border-top: 4px solid #94a3b8;
        }
        .sc-tile-ok   { border-top-color: var(--ok); }
        .sc-tile-info { border-top-color: var(--info); }
        .sc-tile-warn { border-top-color: var(--warn); }
        .sc-tile-risk { border-top-color: var(--risk); }
        .sc-tile-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin: 0;
        }
        .sc-tile-value {
            color: #0f172a;
            font-size: clamp(1.04rem, 1.55vw, 1.34rem);
            font-weight: 850;
            line-height: 1.2;
            margin-top: 10px;
        }
        .sc-tile-detail {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 6px;
            line-height: 1.38;
        }

        /* ── Topbar / Header ─────────────────────────────── */
        .sc-topbar {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            margin-bottom: 14px;
        }
        .sc-brand-mark {
            display: grid;
            width: 44px;
            height: 44px;
            place-items: center;
            color: #ffffff;
            background: #1f2937;
            border-radius: 8px;
            font-weight: 800;
            font-size: 1.1rem;
            flex-shrink: 0;
        }
        .sc-topbar h1 {
            margin: 0;
            font-size: clamp(1.25rem, 2vw, 1.6rem);
            line-height: 1.1;
            color: var(--text);
            font-weight: 800;
        }
        .sc-brand-meta {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 2px;
        }

        /* ── Alerts row ──────────────────────────────────── */
        .sc-alerts-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
        }

        /* ── Driver rows ─────────────────────────────────── */
        .sc-drivers {
            display: grid;
            gap: 2px;
            margin-top: 10px;
        }
        .sc-driver-row {
            display: grid;
            grid-template-columns: minmax(90px, 1fr) minmax(80px, 1.2fr) 52px;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .sc-driver-row:last-child { border-bottom: none; }
        .sc-driver-name {
            color: #182235;
            font-weight: 800;
            font-size: 0.88rem;
        }
        .sc-driver-bar {
            height: 8px;
            background: #e8eef7;
            border-radius: 999px;
            overflow: hidden;
        }
        .sc-driver-bar-fill {
            display: block;
            height: 100%;
            background: var(--accent);
            border-radius: 999px;
            transition: width 300ms ease-out;
        }
        .sc-driver-impact {
            color: var(--muted);
            font-variant-numeric: tabular-nums;
            text-align: right;
            font-size: 0.85rem;
        }

        /* ── Action items ────────────────────────────────── */
        .sc-action {
            padding: 12px;
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .sc-action strong {
            color: #182235;
            font-weight: 800;
        }
        .sc-action-meta {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .sc-action p {
            margin: 8px 0 0;
            color: var(--muted);
            line-height: 1.42;
            font-size: 0.9rem;
        }

        /* ── Technical strip ─────────────────────────────── */
        .sc-tech-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 8px;
        }
        .sc-tech-strip span {
            padding: 4px 7px;
            background: #f1f5f9;
            border-radius: 6px;
            color: var(--muted-2);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.76rem;
        }

        /* ── Section headers ─────────────────────────────── */
        .sc-section-title {
            color: #172033;
            font-size: 1.02rem;
            font-weight: 800;
            margin: 0 0 14px;
        }
        .sc-text {
            color: #273244;
            font-size: 0.96rem;
            line-height: 1.58;
        }

        /* ── Animations ──────────────────────────────────── */
        .cc-fade-in, .sc-fade-in {
            animation: scFadeIn 360ms ease-out;
        }
        @keyframes scFadeIn {
            from { opacity: 0; transform: translateY(3px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Streamlit overrides ─────────────────────────── */
        code, pre, .mono {
            font-family: "IBM Plex Mono", ui-monospace, monospace !important;
        }

        [data-testid="stMetric"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            box-shadow: var(--shadow) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            font-size: 0.78rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 850 !important;
        }

        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 160ms ease !important;
        }

        [data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            box-shadow: var(--shadow) !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 8px !important;
            overflow: hidden;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #172033 !important;
            font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        }

        .stAlert {
            border-radius: 8px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            font-weight: 600 !important;
        }

        footer {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
