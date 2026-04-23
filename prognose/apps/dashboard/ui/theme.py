from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --bg-0: #061122;
            --bg-1: #0b1a33;
            --bg-2: #102544;
            --surface: rgba(15, 31, 58, 0.72);
            --surface-border: rgba(90, 137, 206, 0.24);
            --text-main: #eaf1ff;
            --text-muted: #a8bddc;
            --ok: #21c07a;
            --info: #2ea3ff;
            --warn: #f5b83d;
            --risk: #ff6c67;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at -10% -20%, rgba(46,163,255,0.20), rgba(0,0,0,0)),
                radial-gradient(900px 400px at 110% 10%, rgba(33,192,122,0.14), rgba(0,0,0,0)),
                linear-gradient(145deg, var(--bg-0), var(--bg-1) 55%, var(--bg-2));
            color: var(--text-main);
            font-family: "Manrope", sans-serif;
        }

        section[data-testid="stSidebar"] {
            background: rgba(7, 17, 34, 0.92);
            border-right: 1px solid rgba(120, 154, 214, 0.18);
        }

        .cc-panel {
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            backdrop-filter: blur(6px);
        }

        .cc-badge {
            display: inline-block;
            padding: 0.18rem 0.62rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            margin-right: 0.42rem;
            margin-bottom: 0.42rem;
            border: 1px solid rgba(255,255,255,0.16);
        }
        .cc-badge-ok { background: rgba(33,192,122,0.2); color: #a6f7cf; }
        .cc-badge-info { background: rgba(46,163,255,0.18); color: #b7e4ff; }
        .cc-badge-warn { background: rgba(245,184,61,0.20); color: #ffe8b2; }
        .cc-badge-risk { background: rgba(255,108,103,0.22); color: #ffd2cf; }

        code, pre, .mono {
            font-family: "IBM Plex Mono", monospace !important;
        }

        .cc-fade-in {
            animation: fadeIn 360ms ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(3px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

