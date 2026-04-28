import streamlit as st


def aqi_meta(aqi: int | None) -> dict:
    if aqi is None:
        return {"label": "Unknown",      "color": "#94a3b8", "bg": "rgba(148,163,184,0.14)"}
    if aqi <= 50:
        return {"label": "Good",         "color": "#22c55e", "bg": "rgba(34,197,94,0.14)"}
    if aqi <= 100:
        return {"label": "Satisfactory", "color": "#84cc16", "bg": "rgba(132,204,22,0.14)"}
    if aqi <= 200:
        return {"label": "Moderate",     "color": "#f59e0b", "bg": "rgba(245,158,11,0.14)"}
    if aqi <= 300:
        return {"label": "Poor",         "color": "#f97316", "bg": "rgba(249,115,22,0.14)"}
    if aqi <= 400:
        return {"label": "Very Poor",    "color": "#ef4444", "bg": "rgba(239,68,68,0.16)"}
    return     {"label": "Severe",       "color": "#dc2626", "bg": "rgba(220,38,38,0.20)"}


CHART_LINE_COLOR   = "#38bdf8"
CHART_FILL_COLOR   = "rgba(56,189,248,0.10)"
CHART_WEEKLY_COLOR = "#818cf8"
CHART_GRID_COLOR   = "rgba(255,255,255,0.05)"
CHART_TICK_COLOR   = "#64748b"
CHART_PAPER_BG     = "rgba(0,0,0,0)"
CHART_PLOT_BG      = "rgba(15,23,42,0.55)"

POLLUTANT_COLORS = {
    "PM2_5": "#f87171",
    "PM10":  "#fb923c",
    "NO2":   "#facc15",
    "SO2":   "#a3e635",
    "CO":    "#34d399",
    "O3":    "#22d3ee",
}


def apply_theme() -> None:
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left,  rgba(59,130,246,0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(14,165,233,0.07), transparent 22%),
            linear-gradient(180deg, #07111f 0%, #040816 100%);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1525 0%, #09101c 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .block-container { padding-top: 0.6rem; padding-bottom: 0.8rem; }
    .stApp header    { background: transparent; }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px 14px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .glass {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 16px 18px 14px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
    }
    .topbar {
        background: rgba(8,15,30,0.88);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 9px 14px;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-number {
        font-size: 5rem; line-height: 0.9;
        font-weight: 800; letter-spacing: -0.06em; margin: 0;
    }
    .hero-label  { font-size: 0.92rem; color: #cbd5e1; margin-top: 2px; }
    .section-ttl { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; }
    .subtle      { color: #94a3b8; font-size: 0.88rem; }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.10);
        margin-left: 10px; vertical-align: middle;
    }
    .chip {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.78rem; border: 1px solid rgba(255,255,255,0.08);
        margin-right: 5px; margin-bottom: 5px;
    }
    .chip-ok   { background: rgba(34,197,94,0.14);  color: #bbf7d0; }
    .chip-warn { background: rgba(245,158,11,0.14); color: #fde68a; }
    .chip-bad  { background: rgba(239,68,68,0.14);  color: #fecaca; }

    .pollutant-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 10px; margin-top: 8px;
    }
    .pollutant-pill {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 10px 12px; text-align: center;
    }
    .pill-label { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
    .pill-value { font-size: 1.1rem; font-weight: 700; }

    .error-banner {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.30);
        border-radius: 14px; padding: 14px 18px; margin: 10px 0;
        display: flex; align-items: flex-start; gap: 12px;
    }
    .error-icon  { font-size: 1.3rem; line-height: 1.2; }
    .error-title { color: #fca5a5; font-weight: 700; font-size: 0.93rem; }
    .error-body  { color: #fecaca; font-size: 0.86rem; margin-top: 3px; line-height: 1.5; }

    .warn-banner {
        background: rgba(245,158,11,0.09);
        border: 1px solid rgba(245,158,11,0.28);
        border-radius: 14px; padding: 13px 18px; margin: 10px 0;
    }
    .warn-title { color: #fde68a; font-weight: 700; font-size: 0.93rem; }
    .warn-body  { color: #fef3c7; font-size: 0.86rem; margin-top: 3px; }

    .info-banner {
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.22);
        border-radius: 14px; padding: 13px 18px; margin: 10px 0;
    }
    .info-title { color: #7dd3fc; font-weight: 700; font-size: 0.93rem; }
    .info-body  { color: #bae6fd; font-size: 0.86rem; margin-top: 3px; }

    .down-state {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.18);
        border-radius: 20px; padding: 48px 32px;
        text-align: center; margin-top: 24px;
    }
    .empty-state {
        background: rgba(15,23,42,0.55);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 56px 32px;
        text-align: center; margin-top: 20px;
    }
    .empty-icon  { font-size: 3.5rem; margin-bottom: 16px; }
    .empty-title { color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin-bottom: 8px; }
    .empty-sub   { color: #94a3b8; font-size: 0.9rem; max-width: 380px; margin: 0 auto; }

    .advisory-card {
        background: rgba(56,189,248,0.07);
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 14px; padding: 13px 16px;
        font-size: 0.9rem; color: #bae6fd; margin-top: 6px; line-height: 1.5;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(15,23,42,0.88) !important;
        border-color: rgba(255,255,255,0.08) !important;
        color: #e5e7eb !important;
    }
    .sidebar-label {
        font-size: 0.78rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)