import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ui_styles import SIDEBAR_NAV_CSS

st.markdown(SIDEBAR_NAV_CSS, unsafe_allow_html=True)

load_dotenv()

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL      = os.getenv("BACKEND_URL", "http://localhost:8000")
STATIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "working_stations.json")
DOCS_PATH     = Path(__file__).parent / "docs" / "screenshots"
IST           = timezone(timedelta(hours=5, minutes=30))

POLLUTANT_COLORS = {
    "PM2_5": "#f87171", "PM10": "#fb923c",
    "NO2":   "#facc15", "SO2":  "#a3e635",
    "CO":    "#34d399", "O3":   "#22d3ee",
}

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(14,165,233,0.08), transparent 24%),
            linear-gradient(180deg, #07111f 0%, #040816 100%);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #09101b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .block-container { padding-top: 0.7rem; padding-bottom: 0.8rem; }
    .stApp header { background: rgba(4,8,22,0.0); }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px 14px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.16);
    }
    .glass {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.20);
    }
    .topbar {
        background: rgba(8,15,30,0.85);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 0.75rem;
    }
    .hero-number {
        font-size: 4.8rem; line-height: 0.95;
        font-weight: 800; letter-spacing: -0.06em; margin: 0;
    }
    .hero-label  { font-size: 0.95rem; color: #cbd5e1; margin-top: -4px; }
    .subtle      { color: #94a3b8; font-size: 0.9rem; }
    .section-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.55rem; }
    .sidebar-title { font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.3rem; }

    /* Error banner */
    .error-banner {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.30);
        border-radius: 16px;
        padding: 18px 22px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .error-banner .error-icon  { font-size: 1.6rem; line-height: 1; margin-top: 2px; flex-shrink: 0; }
    .error-banner .error-title { color: #fca5a5; font-weight: 700; font-size: 1rem; }
    .error-banner .error-body  { color: #fecaca; font-size: 0.88rem; margin-top: 5px; line-height: 1.6; }
    .error-banner .error-hint  {
        color: #94a3b8; font-size: 0.82rem; margin-top: 8px;
        border-top: 1px solid rgba(239,68,68,0.15); padding-top: 8px;
    }

    /* Warning banner */
    .warn-banner {
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.28);
        border-radius: 16px;
        padding: 16px 20px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .warn-banner .warn-icon  { font-size: 1.4rem; flex-shrink: 0; }
    .warn-banner .warn-title { color: #fde68a; font-weight: 700; font-size: 0.95rem; }
    .warn-banner .warn-body  { color: #fef3c7; font-size: 0.86rem; margin-top: 4px; line-height: 1.55; }

    /* Backend down full state */
    .down-state {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.18);
        border-radius: 20px;
        padding: 50px 40px;
        text-align: center;
        margin-top: 30px;
    }

    /* IST badge */
    .ist-badge {
        display: inline-block; padding: 2px 7px; border-radius: 6px;
        background: rgba(56,189,248,0.12); color: #7dd3fc;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
        border: 1px solid rgba(56,189,248,0.18); margin-left: 6px;
        vertical-align: middle;
    }

    /* Info icon button next to AQI label */
    .info-btn {
        display: inline-flex; align-items: center; justify-content: center;
        width: 20px; height: 20px; border-radius: 50%;
        background: rgba(148,163,184,0.15); color: #94a3b8;
        font-size: 0.72rem; font-weight: 700; cursor: pointer;
        border: 1px solid rgba(148,163,184,0.25);
        margin-left: 6px; vertical-align: middle;
        text-decoration: none;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(15,23,42,0.85) !important;
        border-color: rgba(255,255,255,0.08) !important;
        color: #e5e7eb !important;
    }
    /* ── Multipage nav links (Home / User Manual) ── */
    div[data-testid="stSidebarNav"] {
        background: rgba(14,165,233,0.06);
        border: 1px solid rgba(14,165,233,0.12);
        border-radius: 14px;
        padding: 6px 4px;
        margin-bottom: 14px;
    }

    div[data-testid="stSidebarNav"] a {
        display: flex;
        align-items: center;
        padding: 9px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #94a3b8 !important;
        text-decoration: none !important;
        transition: all 0.18s ease;
        margin: 2px 0;
    }

    div[data-testid="stSidebarNav"] a:hover {
        background: rgba(56,189,248,0.10);
        color: #7dd3fc !important;
    }

    /* Active / current page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(14,165,233,0.15);
        color: #38bdf8 !important;
        border: 1px solid rgba(14,165,233,0.20);
    }

    /* Add icons via ::before pseudo-element */
    div[data-testid="stSidebarNav"] a[href="/"] span::before,
    div[data-testid="stSidebarNav"] a:first-child span::before {
        content: "";
    }

    div[data-testid="stSidebarNav"] a:not(:first-child) span::before {
        content: "📖 ";
    }

    /* Hide the default Streamlit logo/header above nav */
    div[data-testid="stSidebarNav"]::before {
        content: "NAVIGATE";
        display: block;
        color: #475569;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 4px 14px 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def aqi_theme(aqi):
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
    return     {"label": "Severe",       "color": "#dc2626", "bg": "rgba(220,38,38,0.18)"}


def station_label(s):
    return "{} · {} ({})".format(s["city"], s["station_name"], s["openaq_location_id"])


def to_ist(ts_str):
    if not ts_str:
        return "—"
    try:
        dt = pd.to_datetime(ts_str, utc=True)
        return dt.tz_convert(IST).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return ts_str[:19].replace("T", " ")


def show_error(title, body, hint=None):
    hint_html = (
        "<div class='error-hint'>💡 " + hint + "</div>" if hint else ""
    )
    st.markdown(
        "<div class='error-banner'>"
        "<div class='error-icon'>⚠️</div>"
        "<div style='flex:1'>"
        "<div class='error-title'>" + title + "</div>"
        "<div class='error-body'>" + body + "</div>"
        + hint_html +
        "</div></div>",
        unsafe_allow_html=True,
    )


def show_warning(title, body):
    st.markdown(
        "<div class='warn-banner'>"
        "<div class='warn-icon'>💬</div>"
        "<div>"
        "<div class='warn-title'>" + title + "</div>"
        "<div class='warn-body'>" + body + "</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def parse_error(exc):
    """Returns (title, body, hint) — all human-friendly, zero technical jargon."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "Unable to reach the prediction service",
            "The air quality prediction service is currently offline. "
            "Please try again in a few minutes.",
            "If the problem persists, contact the system administrator.",
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return (
            "This is taking longer than expected",
            "The system is fetching the latest air quality readings — "
            "this can sometimes take up to a minute.",
            "Please click Predict AQI again to retry.",
        )
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = ""

        if "Not enough live rows" in detail or "live rows" in detail:
            return (
                "Not enough recent data for this station",
                "This monitoring station hasn't reported enough hourly readings yet. "
                "The forecast model needs at least 7 days of sensor history.",
                "Try selecting a different station from the same city, or check back tomorrow.",
            )
        if "not supported" in detail:
            return (
                "This city is not available yet",
                "Predictions for this city are not supported in the current model version.",
                "Please select a different station from the list.",
            )
        if "No station registry" in detail:
            return (
                "Station not found",
                "This monitoring station could not be located in our database.",
                "Please select a different station from the dropdown.",
            )
        if "Live fetch failed" in detail or "fetch" in detail.lower():
            return (
                "Could not get latest sensor readings",
                "We were unable to fetch fresh data from the air quality network right now. "
                "The external data provider may be temporarily unavailable.",
                "Please wait a few minutes and try again.",
            )
        return (
            "Something went wrong",
            "The prediction service returned an unexpected response. Please try again.",
            None,
        )
    return (
        "Something went wrong",
        "An unexpected error occurred. Please try again in a moment.",
        None,
    )


# ── Data fetchers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_health():
    try:
        return requests.get("{}/health".format(BASE_URL), timeout=5).json()
    except Exception:
        return {"status": "down"}


@st.cache_data(ttl=30)
def fetch_ready():
    try:
        return requests.get("{}/ready".format(BASE_URL), timeout=5).json()
    except Exception:
        return {"status": "down", "model_loaded": False}


@st.cache_data(ttl=300)
def load_workable_stations():
    with open(STATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=90)
def fetch_prediction(city, location_id):
    resp = requests.get(
        "{}/predict".format(BASE_URL),
        params={"city": city, "location_id": location_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ── Session state ──────────────────────────────────────────────────────────────
for _key, _default in [
    ("prediction_data",  None),
    ("selected_station", None),
    ("last_error",       None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Fetch system status ────────────────────────────────────────────────────────
stations   = load_workable_stations()
health     = fetch_health()
ready      = fetch_ready()
backend_up = health.get("status") == "ok"
model_up   = ready.get("model_loaded", False)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>🌫️ AQI Predictor</div>"
        "<div style='color:#64748b;font-size:1rem;margin-bottom:14px;'>"
        "Real-time air quality prediction · India</div>",
        unsafe_allow_html=True,
    )

    if not backend_up:
        show_error(
            "Service Unavailable",
            "The prediction service is currently offline.",
            "Please start the backend service and refresh the page.",
        )

    st.divider()

    city_filter = st.selectbox(
        "Filter by city",
        ["All"] + sorted(set(s["city"] for s in stations)),
        index=0,
        help="Narrow the station list to a specific city.",
    )
    filtered = stations if city_filter == "All" else [
        s for s in stations if s["city"] == city_filter
    ]
    options = [station_label(s) for s in filtered]

    selected_option = st.selectbox(
        "Select station",
        options,
        index=0 if options else None,
        disabled=not backend_up,
        help="Choose the air quality monitoring station to forecast.",
    )
    if selected_option:
        st.session_state.selected_station = next(
            (s for s in filtered if station_label(s) == selected_option), None
        )

    st.write("")
    predict_clicked = st.button(
        "Predict AQI",
        use_container_width=True,
        disabled=not backend_up or not model_up,
        help="Fetch live sensor data and run the 24-hour AQI forecast.",
    )

    st.divider()

 

# ── Handle predict click ───────────────────────────────────────────────────────
if predict_clicked and st.session_state.selected_station:
    station = st.session_state.selected_station
    st.session_state.last_error = None
    fetch_prediction.clear()
    with st.spinner("Fetching live air quality data for {} …".format(station["city"])):
        try:
            st.session_state.prediction_data = fetch_prediction(
                station["city"],
                station["openaq_location_id"],
            )
        except Exception as exc:
            st.session_state.last_error      = parse_error(exc)
            st.session_state.prediction_data = None

elif predict_clicked and not st.session_state.selected_station:
    show_warning(
        "No station selected",
        "Please choose a city and monitoring station from the sidebar first.",
    )


# ── Top bar ────────────────────────────────────────────────────────────────────
now_ist      = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
station_name = (
    st.session_state.selected_station["station_name"]
    if st.session_state.selected_station
    else "Select a station to begin"
)

st.markdown(
    "<div class='topbar'>"
    "<div style='display:flex;justify-content:space-between;align-items:center;"
    "gap:12px;flex-wrap:wrap;'>"
    "<div>"
    "<div style='font-size:1.15rem;font-weight:800;color:#f8fafc;'>" + station_name + "</div>"
    "<div class='subtle'>" + now_ist + "</div>"
    "</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

# ── Error display ──────────────────────────────────────────────────────────────
if st.session_state.last_error:
    title, body, hint = st.session_state.last_error
    show_error(title, body, hint)

# ── Backend down full state ────────────────────────────────────────────────────
if not backend_up:
    st.markdown(
        "<div class='down-state'>"
        "<div style='font-size:3.5rem;'>🔌</div>"
        "<div style='font-size:1.4rem;font-weight:800;color:#fca5a5;margin-top:16px;'>"
        "Prediction service is offline"
        "</div>"
        "<div class='subtle' style='margin-top:10px;max-width:440px;"
        "margin-left:auto;margin-right:auto;line-height:1.7;'>"
        "The air quality prediction service is not reachable right now.<br>"
        "Please start the backend service and refresh this page."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── AQI Reference popup content (shown via st.popover) ────────────────────────
AQI_REF_ROWS = [
    ("0–50",   "Good",         "#22c55e", "rgba(34,197,94,0.12)",   "Minimal impact — safe for all"),
    ("51–100",  "Satisfactory", "#84cc16", "rgba(132,204,22,0.12)", "Minor discomfort to sensitive people"),
    ("101–200", "Moderate",     "#f59e0b", "rgba(245,158,11,0.12)", "Discomfort for lung/heart patients"),
    ("201–300", "Poor",         "#f97316", "rgba(249,115,22,0.12)", "Discomfort for most on prolonged exposure"),
    ("301–400", "Very Poor",    "#ef4444", "rgba(239,68,68,0.14)",  "Respiratory illness on prolonged exposure"),
    ("401–500", "Severe",       "#dc2626", "rgba(220,38,38,0.16)",  "Serious health risk for everyone"),
]


# ── Main dashboard ─────────────────────────────────────────────────────────────
data = st.session_state.prediction_data

if data:
    theme    = aqi_theme(data.get("current_aqi"))
    advisory = data.get("advisory", "")
    last_updated_ist = to_ist(data.get("last_updated", ""))
    raw_station_id   = (data.get("station_id", "") or "").replace("openaq:", "")

    # ── Row 1: AQI hero + advisory + station ──────────────────────────────────
    c1, c2, c3 = st.columns([2.2, 1, 1])

    with c1:
        # AQI hero card
        st.markdown(
            "<div class='glass'>"
            "<div class='subtle'>" + data.get("city", "") + " · " +
            data.get("station_name", "") + "</div>"
            "<div style='display:flex;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-top:6px;'>"
            "<div>"
            "<div class='hero-number' style='color:" + theme["color"] + ";'>" +
            str(data.get("current_aqi", "NA")) + "</div>"
            "<div class='hero-label'>AQI · " + theme["label"] + "</div>"
            "</div>"
            "<div class='glass' style='min-width:170px;padding:12px 14px;'>"
            "<div class='subtle'>Primary Pollutant</div>"
            "<div style='font-size:1.4rem;font-weight:800;color:#f8fafc;'>" +
            str(data.get("primary_pollutant", "NA")) + "</div>"
            "</div>"
            "<div class='glass' style='min-width:230px;padding:12px 14px;'>"
            "<div class='subtle'>Last Updated <span class='ist-badge'>IST</span></div>"
            "<div style='font-size:0.96rem;font-weight:700;color:#f8fafc;'>" +
            last_updated_ist + "</div>"
            "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # ⓘ AQI Reference popover — sits just below the hero card
        with st.popover("ⓘ  What does this AQI value mean?", use_container_width=False):
            st.markdown(
                "<div style='font-size:0.9rem;font-weight:700;color:#e2e8f0;"
                "margin-bottom:10px;'>AQI Scale — CPCB National Air Quality Index</div>",
                unsafe_allow_html=True,
            )
            for rng, label, color, bg, tip in AQI_REF_ROWS:
                is_current = (label == theme["label"])
                border = "2px solid " + color if is_current else "1px solid " + color + "33"
                st.markdown(
                    "<div style='background:" + bg + ";border:" + border + ";"
                    "border-radius:10px;padding:8px 12px;margin-bottom:6px;"
                    "display:flex;justify-content:space-between;align-items:center;'>"
                    "<div>"
                    "<span style='color:" + color + ";font-weight:700;font-size:0.85rem;'>"
                    + label + "</span>"
                    "<span style='color:#64748b;font-size:0.78rem;margin-left:8px;'>" + rng + "</span>"
                    + (" <span style='color:" + color + ";font-size:0.72rem;font-weight:600;"
                       "background:" + bg + ";border:1px solid " + color + "44;"
                       "padding:1px 6px;border-radius:999px;margin-left:4px;'>● Current</span>"
                       if is_current else "") +
                    "</div>"
                    "<div style='color:#94a3b8;font-size:0.8rem;max-width:200px;"
                    "text-align:right;'>" + tip + "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div style='color:#475569;font-size:0.74rem;margin-top:8px;'>"
                "Source: CPCB · airquality.cpcb.gov.in</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown(
            "<div class='glass' style='min-height:176px;'>"
            "<div class='section-title'>Health Advisory</div>"
            "<div class='subtle' style='line-height:1.65;'>" + advisory + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            "<div class='glass' style='min-height:176px;'>"
            "<div class='section-title'>Station</div>"
            "<div style='font-weight:700;color:#f8fafc;'>" + data.get("station_name", "") + "</div>"
            "<div class='subtle'>" + data.get("city", "") + "</div>"
            "<div class='subtle' style='margin-top:10px;'>Location ID: " +
            str(data.get("location_id", "NA")) + "</div>"
            "<div class='subtle'>Station: " + raw_station_id + "</div>"
            "<div class='subtle'>Source: OpenAQ v3</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Row 2: 24h Forecast + Pollutants ──────────────────────────────────────
    chart1, chart2 = st.columns([2, 1])

    with chart1:
        st.markdown("<div class='section-title'>24-Hour AQI Forecast</div>",
                    unsafe_allow_html=True)
        forecast = data.get("forecast_24h", [])
        if forecast:
            hours  = ["H+{}".format(i) for i in range(1, len(forecast) + 1)]
            colors = [aqi_theme(v)["color"] for v in forecast]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours, y=forecast,
                mode="lines+markers",
                line=dict(color="#ef4444", width=3),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.16)",
                marker=dict(size=6, color=colors),
                hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>",
            ))
            fig.add_hline(
                y=data.get("current_aqi", 0),
                line_dash="dot",
                line_color="rgba(255,255,255,0.20)",
                annotation_text="Now",
                annotation_font_color="#94a3b8",
                annotation_font_size=11,
            )
            fig.update_layout(
                height=310,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#cbd5e1"),
                xaxis=dict(showgrid=False, title="Hour"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="AQI"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            show_warning(
                "Forecast not available",
                "The model did not return forecast values for this station.",
            )

    with chart2:
        st.markdown("<div class='section-title'>Current Pollutant Levels</div>",
                    unsafe_allow_html=True)
        pollutants = data.get("pollutants", {})
        if pollutants:
            p_df = (
                pd.DataFrame(list(pollutants.items()), columns=["pollutant", "value"])
                .sort_values("value", ascending=True)
            )
            bar_colors = [POLLUTANT_COLORS.get(p, "#60a5fa") for p in p_df["pollutant"]]
            fig = go.Figure(go.Bar(
                x=p_df["value"], y=p_df["pollutant"],
                orientation="h",
                marker=dict(color=bar_colors),
                hovertemplate="<b>%{y}</b>: %{x:.2f}<extra></extra>",
            ))
            fig.update_layout(
                height=310,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(color="#cbd5e1"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            show_warning(
                "Pollutant data unavailable",
                "Pollutant readings were not available for this station.",
            )

    # ── Row 3: 7-Day trend + Summary metrics ──────────────────────────────────
    bottom1, bottom2 = st.columns([1.5, 1])

    with bottom1:
        st.markdown("<div class='section-title'>7-Day AQI Trend</div>",
                    unsafe_allow_html=True)
        weekly = data.get("weekly_trend", [])
        if weekly:
            today = datetime.now(IST)
            days  = [
                (today - pd.Timedelta(days=len(weekly) - 1 - i)).strftime("%a %d %b")
                for i in range(len(weekly))
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days, y=weekly,
                mode="lines+markers",
                line=dict(color="#a78bfa", width=2.5),
                marker=dict(size=7, color=[aqi_theme(v)["color"] for v in weekly]),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.08)",
                hovertemplate="<b>%{x}</b><br>Avg AQI: %{y}<extra></extra>",
            ))
            fig.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#cbd5e1"),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            show_warning(
                "Trend data unavailable",
                "7-day historical data was not available for this station.",
            )

    with bottom2:
        st.markdown("<div class='section-title'>Forecast Summary</div>",
                    unsafe_allow_html=True)
        forecast_list = data.get("forecast_24h", [])
        forecast_peak = max(forecast_list) if forecast_list else None
        forecast_avg  = round(sum(forecast_list) / len(forecast_list)) if forecast_list else None
        peak_theme    = aqi_theme(forecast_peak)

        st.markdown(
            "<div class='glass' style='min-height:250px;'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>"
            "<div>"
            "<div class='subtle'>Current AQI</div>"
            "<div style='font-size:1.6rem;font-weight:800;color:" + theme["color"] + ";'>" +
            str(data.get("current_aqi", "NA")) + "</div>"
            "<div class='subtle' style='font-size:0.78rem;'>" + theme["label"] + "</div>"
            "</div>"
            "<div>"
            "<div class='subtle'>24h Peak</div>"
            "<div style='font-size:1.6rem;font-weight:800;color:" + peak_theme["color"] + ";'>" +
            (str(forecast_peak) if forecast_peak else "—") + "</div>"
            "<div class='subtle' style='font-size:0.78rem;'>" + peak_theme["label"] + "</div>"
            "</div>"
            "</div>"
            "<div style='margin-top:16px;'>"
            "<div class='subtle'>24h Average AQI</div>"
            "<div style='font-weight:700;color:#f8fafc;font-size:1.1rem;'>" +
            (str(forecast_avg) if forecast_avg else "—") + "</div>"
            "</div>"
            "<div style='margin-top:14px;'>"
            "<div class='subtle'>Primary Pollutant</div>"
            "<div style='font-weight:700;color:#f8fafc;'>" +
            str(data.get("primary_pollutant", "NA")) + "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── AQI Reference strip at bottom ─────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title' style='margin-bottom:10px;'>"
        "📊 AQI Reference — CPCB Standards</div>",
        unsafe_allow_html=True,
    )
    ref_cols = st.columns(6)
    for i, (rng, label, color, bg, tip) in enumerate(AQI_REF_ROWS):
        is_current = (label == theme["label"])
        with ref_cols[i]:
            ring = "2px solid " + color if is_current else "1px solid " + color + "33"
            st.markdown(
                "<div style='background:" + bg + ";border:" + ring + ";"
                "border-radius:14px;padding:14px 10px;text-align:center;'>"
                + ("<div style='font-size:0.68rem;color:" + color + ";font-weight:700;"
                   "letter-spacing:0.05em;margin-bottom:4px;'>● YOU ARE HERE</div>"
                   if is_current else "") +
                "<div style='color:" + color + ";font-size:1.1rem;font-weight:800;'>" + rng + "</div>"
                "<div style='color:" + color + ";font-size:0.8rem;font-weight:700;"
                "margin-top:4px;padding-top:4px;border-top:1px solid " + color + "22;'>" + label + "</div>"
                "<div style='color:#94a3b8;font-size:0.74rem;margin-top:6px;line-height:1.4;'>" + tip + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div style='color:#334155;font-size:0.74rem;text-align:center;margin-top:6px;'>"
        "CPCB National Air Quality Index · airquality.cpcb.gov.in</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='glass' style='text-align:center;padding:50px 30px;'>"
        "<div style='font-size:3rem;'>🌫️</div>"
        "<div style='font-size:1.45rem;font-weight:800;color:#f8fafc;margin-top:12px;'>"
        "Select a station and click Predict AQI"
        "</div>"
        "<div class='subtle' style='margin-top:8px;max-width:420px;"
        "margin-left:auto;margin-right:auto;line-height:1.7;'>"
        "Choose a city and monitoring station from the sidebar.<br>"
        "The dashboard will show live AQI, 24-hour forecast, "
        "pollutant levels, and a health advisory."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Also show reference strip on empty state so it's always visible
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title' style='margin-bottom:10px;'>"
        "📊 AQI Reference — CPCB Standards</div>",
        unsafe_allow_html=True,
    )
    ref_cols2 = st.columns(6)
    for i, (rng, label, color, bg, tip) in enumerate(AQI_REF_ROWS):
        with ref_cols2[i]:
            st.markdown(
                "<div style='background:" + bg + ";border:1px solid " + color + "33;"
                "border-radius:14px;padding:14px 10px;text-align:center;'>"
                "<div style='color:" + color + ";font-size:1.1rem;font-weight:800;'>" + rng + "</div>"
                "<div style='color:" + color + ";font-size:0.8rem;font-weight:700;"
                "margin-top:4px;padding-top:4px;border-top:1px solid " + color + "22;'>" + label + "</div>"
                "<div style='color:#94a3b8;font-size:0.74rem;margin-top:6px;line-height:1.4;'>" + tip + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<div style='color:#334155;font-size:0.74rem;text-align:center;margin-top:6px;'>"
        "CPCB National Air Quality Index · airquality.cpcb.gov.in</div>",
        unsafe_allow_html=True,
    )