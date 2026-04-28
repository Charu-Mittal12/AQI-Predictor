import streamlit as st
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ui_styles import SIDEBAR_NAV_CSS

# ── MUST be first Streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="User Manual – AQI Forecast",
    page_icon="📖",
    layout="wide",
)

st.markdown(SIDEBAR_NAV_CSS, unsafe_allow_html=True)

DOCS = Path(__file__).parent.parent / "docs" / "screenshots"

def img(filename):
    """Returns path string if screenshot exists, else None."""
    p = DOCS / filename
    return str(p) if p.exists() else None

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #040816 100%);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #09101b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .glass {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 20px 24px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.20);
        margin-bottom: 16px;
    }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 50%;
        background: rgba(14,165,233,0.18); color: #38bdf8;
        font-weight: 800; font-size: 0.95rem;
        border: 1px solid rgba(14,165,233,0.30);
        flex-shrink: 0;
    }
    .step-row {
        display: flex; align-items: flex-start;
        gap: 14px; margin-bottom: 14px;
    }
    .step-content {
        flex: 1;
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 14px 18px;
    }
    .step-title { font-weight: 700; color: #f8fafc; font-size: 0.95rem; }
    .step-desc  { color: #94a3b8; font-size: 0.87rem; margin-top: 4px; line-height: 1.6; }
    .subtle     { color: #94a3b8; font-size: 0.9rem; }
    .card-label { font-weight: 700; color: #f8fafc; margin-bottom: 8px; font-size: 0.95rem; }
    h1, h2, h3  { color: #f8fafc !important; }

    /* Tab styling */
    div[data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15,23,42,0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 8px 16px !important;
        font-size: 0.85rem !important;
        color: #64748b !important;
    }
    div[aria-selected="true"][data-baseweb="tab"] {
        background: rgba(14,165,233,0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(14,165,233,0.25) !important;
    }

    /* Screenshot frame */
    .img-frame {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin: 12px 0;
    }
    div[data-baseweb="tab-list"] {
    gap: 8px;          /* ← increase this value for more spacing */
    background: rgba(15,23,42,0.5);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 8px 22px !important;   /* ← increase horizontal padding */
    font-size: 0.85rem !important;
    color: #64748b !important;
     }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px;'>📖 User Manual</h1>"
    "<div class='subtle' style='margin-bottom:20px;'>"
    "AQI Forecast Dashboard · Powered by LightGBM + OpenAQ v3 + CPCB Standards"
    "</div>",
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Getting Started",
    " Dashboard Guide",
    " AQI Scale",
    " Pollutants",
    " Troubleshooting",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Getting Started
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🌫️ What is this dashboard?")
    st.markdown("""
<div class="glass">
This dashboard predicts the <strong style="color:#38bdf8;">Air Quality Index (AQI)</strong>
for Indian cities using live sensor data from CPCB-registered monitoring stations
(via OpenAQ v3) and a LightGBM machine learning model trained on 25 cities.
<br><br>
<strong style="color:#e2e8f0;">You get:</strong>
<ul style="color:#94a3b8;line-height:2;margin-top:8px;">
  <li>Current AQI with health category</li>
  <li>24-hour AQI forecast</li>
  <li>Live pollutant levels — PM2.5, PM10, NO₂, SO₂, CO, O₃</li>
  <li>7-day historical trend</li>
  <li>Plain-language health advisory</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # Dashboard overview screenshot
    overview = img("dashboard_prediction.png")
    if overview:
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(overview, caption="AQI Forecast Dashboard — full view after a prediction",
                 use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📸 Screenshot will appear here — add dashboard_prediction.png to docs/screenshots/")

    st.markdown("---")
    st.markdown("###  Step-by-Step: Get Your First Prediction")

    steps = [
        ("Open the dashboard",
         "Go to the dashboard URL in your browser (e.g. <code style='color:#7dd3fc;'>http://localhost:8501</code>)."),
        ("Filter by city",
         "Use the <b>Filter by city</b> dropdown in the left sidebar to narrow the station list."),
        ("Select a station",
         "Choose a monitoring station from <b>Select station</b>. "
         "Format: <code style='color:#7dd3fc;'>City · Station Name (ID)</code>"),
        ("Click Predict AQI",
         "Click the <b> Predict AQI</b> button. "
         "The system fetches live data — allow 10–60 seconds on first run."),
        ("Read your results",
         "The dashboard shows current AQI, 24h forecast, pollutant chart, "
         "7-day trend, and a health advisory."),
    ]

    left, right = st.columns([1.2, 1])
    with left:
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(
                "<div class='step-row'>"
                "<div class='step-num'>" + str(i) + "</div>"
                "<div class='step-content'>"
                "<div class='step-title'>" + title + "</div>"
                "<div class='step-desc'>" + desc + "</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

    with right:
        # Sidebar screenshot on the right of steps
        sidebar = img("sidebar_select.png")
        if sidebar:
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(sidebar, caption="Sidebar: city filter + station selector",
                     width=200)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='glass' style='text-align:center;padding:40px 20px;color:#475569;'>"
                "📸 sidebar_select.png<br>"
                "<span style='font-size:0.78rem;'>Add to docs/screenshots/</span>"
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dashboard Guide
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Understanding Each Section")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
<div class="glass">
<div class="card-label">🔢 AQI Hero Card</div>
<ul style="color:#94a3b8;line-height:1.9;margin:0;padding-left:18px;">
  <li>Large coloured number = <strong style="color:#e2e8f0;">current AQI</strong></li>
  <li>Colour shifts green → yellow → red with severity</li>
  <li>Click <strong style="color:#38bdf8;">ⓘ What does this AQI mean?</strong> for the full scale popup</li>
  <li><strong style="color:#e2e8f0;">Primary Pollutant</strong> = the gas most responsible for the AQI</li>
  <li><strong style="color:#e2e8f0;">Last Updated</strong> shows sensor reading time in IST</li>
</ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="glass">
<div class="card-label">📉 7-Day Trend</div>
<ul style="color:#94a3b8;line-height:1.9;margin:0;padding-left:18px;">
  <li>Shows daily average AQI over the past 7 days</li>
  <li>Each point is coloured by its AQI category</li>
  <li>Rising trend = worsening air quality this week</li>
  <li>Dates are shown in IST format</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="glass">
<div class="card-label">📈 24-Hour Forecast Chart</div>
<ul style="color:#94a3b8;line-height:1.9;margin:0;padding-left:18px;">
  <li>X-axis: H+1 to H+24 = next 24 hours</li>
  <li>Y-axis: predicted AQI value</li>
  <li>Dotted horizontal line = current AQI (your baseline)</li>
  <li>Forecast above the line → air quality is worsening</li>
  <li>Each dot coloured by AQI severity category</li>
</ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="glass">
<div class="card-label">🧪 Pollutant Levels Chart</div>
<ul style="color:#94a3b8;line-height:1.9;margin:0;padding-left:18px;">
  <li>Horizontal bar chart of all measured gases</li>
  <li>Each pollutant has a distinct colour</li>
  <li>Longer bar = higher concentration</li>
  <li>Units: µg/m³ (PM, NO₂, SO₂, O₃) · mg/m³ (CO)</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # Forecast chart screenshot — full width below
    forecast_ss = img("forecast_chart.png")
    if forecast_ss:
        st.markdown("**24-Hour Forecast Chart — example:**")
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(forecast_ss, caption="24-hour forecast — each dot coloured by AQI severity",
                 use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###  Forecast Summary Card")
    st.markdown("""
<div class="glass" style="max-width:600px;">
<ul style="color:#94a3b8;line-height:2;margin:0;padding-left:18px;">
  <li><strong style="color:#e2e8f0;">Current AQI</strong> — value from the latest sensor reading</li>
  <li><strong style="color:#e2e8f0;">24h Peak</strong> — highest predicted AQI in the next 24 hours</li>
  <li><strong style="color:#e2e8f0;">24h Average</strong> — mean predicted AQI over the next 24 hours</li>
  <li><strong style="color:#e2e8f0;">Primary Pollutant</strong> — the dominant gas at this station</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AQI Scale
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### AQI Colour Scale — CPCB National Standards")
    st.markdown(
        "<div class='subtle' style='margin-bottom:16px;'>"
        "India uses the CPCB (Central Pollution Control Board) AQI scale. "
        "Each category has a colour, a range, and a health guideline."
        "</div>",
        unsafe_allow_html=True,
    )

    aqi_rows = [
        ("0–50",   "Good",         "#22c55e", "rgba(34,197,94,0.12)",
         "Minimal health impact.", "Safe for all outdoor activities."),
        ("51–100",  "Satisfactory", "#84cc16", "rgba(132,204,22,0.12)",
         "Minor discomfort to sensitive people.", "Generally safe. Sensitive groups take care."),
        ("101–200", "Moderate",     "#f59e0b", "rgba(245,158,11,0.12)",
         "Discomfort for lung/heart patients.", "Reduce prolonged outdoor activity."),
        ("201–300", "Poor",         "#f97316", "rgba(249,115,22,0.12)",
         "Discomfort for most people.", "Avoid prolonged outdoor exposure."),
        ("301–400", "Very Poor",    "#ef4444", "rgba(239,68,68,0.14)",
         "Respiratory illness risk.", "Stay indoors if possible."),
        ("401–500", "Severe",       "#dc2626", "rgba(220,38,38,0.16)",
         "Serious risk for everyone.", "Avoid all outdoor activity."),
    ]

    cols = st.columns(6)
    for i, (rng, label, color, bg, impact, action) in enumerate(aqi_rows):
        with cols[i]:
            st.markdown(
                "<div style='background:" + bg + ";border:1px solid " + color + "44;"
                "border-radius:16px;padding:16px 10px;text-align:center;height:100%;'>"
                "<div style='color:" + color + ";font-size:1.15rem;font-weight:800;'>" + rng + "</div>"
                "<div style='color:" + color + ";font-size:0.82rem;font-weight:700;"
                "margin:6px 0;padding:4px 0;border-top:1px solid " + color + "22;"
                "border-bottom:1px solid " + color + "22;'>" + label + "</div>"
                "<div style='color:#94a3b8;font-size:0.73rem;margin-top:6px;line-height:1.5;'>"
                + impact + "</div>"
                "<div style='color:#64748b;font-size:0.7rem;margin-top:5px;font-style:italic;'>"
                + action + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div style='color:#475569;font-size:0.75rem;text-align:center;margin-top:10px;'>"
        "Source: CPCB National Air Quality Index · airquality.cpcb.gov.in"
        "</div>",
        unsafe_allow_html=True,
    )

    # AQI reference screenshot
    aqi_ss = img("aqi_reference.png")
    if aqi_ss:
        st.markdown("---")
        st.markdown("**AQI reference as shown at the bottom of the dashboard:**")
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(aqi_ss, caption="CPCB AQI reference strip on the dashboard",
                 use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 How to read the ⓘ popup on the dashboard")
    st.markdown("""
<div class="glass" style="max-width:640px;">
On the main dashboard, below the AQI hero number, click the button:<br><br>
<span style="background:rgba(14,165,233,0.12);border:1px solid rgba(14,165,233,0.2);
padding:6px 14px;border-radius:8px;color:#38bdf8;font-size:0.88rem;">
ⓘ  What does this AQI value mean?
</span><br><br>
<span style="color:#94a3b8;">
A popup opens showing the full scale with your <strong style="color:#e2e8f0;">current 
category highlighted</strong> with a "● Current" badge so you can immediately see 
where your station's air quality stands.
</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Pollutants
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🧪 Pollutant Reference Guide")
    st.markdown(
        "<div class='subtle' style='margin-bottom:16px;'>"
        "These are the six pollutants measured by CPCB stations and displayed on the dashboard."
        "</div>",
        unsafe_allow_html=True,
    )

    pollutants = [
        ("PM2.5", "#f87171", "Fine Particulate Matter", "µg/m³",
         "Vehicles, industry, biomass burning",
         "Penetrates deep into lungs and bloodstream. Most harmful pollutant for health."),
        ("PM10",  "#fb923c", "Coarse Particulate Matter", "µg/m³",
         "Road dust, construction, wind",
         "Affects nose, throat, and upper airways. Less penetrating than PM2.5."),
        ("NO₂",   "#facc15", "Nitrogen Dioxide", "µg/m³",
         "Vehicle exhaust, power plants",
         "Irritates airways, worsens asthma and respiratory disease."),
        ("SO₂",   "#a3e635", "Sulphur Dioxide", "µg/m³",
         "Coal burning, industrial processes",
         "Causes breathing difficulty and eye irritation. Reacts to form acid rain."),
        ("CO",    "#34d399", "Carbon Monoxide", "mg/m³",
         "Incomplete combustion, vehicles",
         "Reduces oxygen delivery in blood. High levels can be fatal."),
        ("O₃",    "#22d3ee", "Ground-level Ozone", "µg/m³",
         "Sunlight reacting with NOx and VOCs",
         "Irritates airways, damages lung tissue. Worse on hot sunny days."),
    ]

    for name, color, full_name, unit, source, concern in pollutants:
        st.markdown(
            "<div style='display:flex;gap:16px;align-items:flex-start;"
            "background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.05);"
            "border-radius:14px;padding:14px 18px;margin-bottom:10px;'>"
            "<div style='min-width:64px;text-align:center;'>"
            "<div style='color:" + color + ";font-size:1.3rem;font-weight:800;'>" + name + "</div>"
            "<div style='color:#475569;font-size:0.72rem;margin-top:2px;'>" + unit + "</div>"
            "</div>"
            "<div style='flex:1;border-left:1px solid rgba(255,255,255,0.06);padding-left:16px;'>"
            "<div style='font-weight:700;color:#e2e8f0;font-size:0.9rem;'>" + full_name + "</div>"
            "<div style='color:#64748b;font-size:0.8rem;margin-top:2px;'>Source: " + source + "</div>"
            "<div style='color:#94a3b8;font-size:0.84rem;margin-top:6px;line-height:1.6;'>"
            + concern + "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Troubleshooting
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ❓ Common Issues & Fixes")

    trouble_items = [
        ("⚠️ Not enough recent data for this station",
         "rgba(245,158,11,0.08)", "rgba(245,158,11,0.25)", "#fde68a", "#fef3c7",
         "This station hasn't reported 7 days of readings yet. "
         "The forecast model needs at least 168 hours of sensor history.",
         "Try a different station in the same city, or check back the next day."),
        ("🔌 Service is currently offline",
         "rgba(239,68,68,0.08)", "rgba(239,68,68,0.25)", "#fca5a5", "#fecaca",
         "The backend prediction service is not running or unreachable.",
         "If you are an admin: run docker compose up -d and refresh the page."),
        ("⏱️ Prediction is taking very long (> 60 seconds)",
         "rgba(56,189,248,0.08)", "rgba(56,189,248,0.20)", "#7dd3fc", "#bae6fd",
         "The system fetches fresh live data from OpenAQ on every prediction. "
         "First predictions for a station can take up to 60 seconds.",
         "Click Predict AQI again if it times out. Subsequent runs are faster."),
        ("🚫 This city is not available yet",
         "rgba(245,158,11,0.08)", "rgba(245,158,11,0.25)", "#fde68a", "#fef3c7",
         "This city was not included in the model's training dataset.",
         "Please select a different station from the available list."),
        ("📉 Charts are not showing",
         "rgba(56,189,248,0.08)", "rgba(56,189,248,0.20)", "#7dd3fc", "#bae6fd",
         "Forecast or trend data was not available for this station.",
         "Try a different station that has more historical sensor readings."),
    ]

    for title, bg, border_c, title_c, body_c, cause, fix in trouble_items:
        st.markdown(
            "<div style='background:" + bg + ";border:1px solid " + border_c + ";"
            "border-radius:14px;padding:16px 20px;margin-bottom:12px;'>"
            "<div style='color:" + title_c + ";font-weight:700;font-size:0.93rem;"
            "margin-bottom:8px;'>" + title + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>"
            "<div>"
            "<div style='color:#475569;font-size:0.73rem;font-weight:700;"
            "letter-spacing:0.06em;margin-bottom:4px;'>CAUSE</div>"
            "<div style='color:" + body_c + ";font-size:0.85rem;line-height:1.6;'>" + cause + "</div>"
            "</div>"
            "<div>"
            "<div style='color:#475569;font-size:0.73rem;font-weight:700;"
            "letter-spacing:0.06em;margin-bottom:4px;'>FIX</div>"
            "<div style='color:" + body_c + ";font-size:0.85rem;line-height:1.6;'>" + fix + "</div>"
            "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📡 Data Sources")
    sources = [
        ("OpenAQ v3 API",    "#38bdf8", "Live hourly pollutant readings from CPCB-registered monitoring stations across India"),
        ("Open-Meteo API",   "#a78bfa", "Weather features — temperature, humidity, wind speed, and atmospheric pressure"),
        ("CPCB Standards",   "#34d399", "AQI calculation methodology and health category thresholds"),
        ("LightGBM Model",   "#f59e0b", "Gradient boosting ML model trained on 25 Indian cities · 24-hour forecast horizon"),
        ("Apache Airflow",   "#60a5fa", "Hourly automated data pipeline that fetches, cleans, and stores sensor readings"),
    ]
    for name, color, desc in sources:
        st.markdown(
            "<div style='display:flex;gap:14px;align-items:center;"
            "background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.05);"
            "border-radius:12px;padding:12px 16px;margin-bottom:8px;'>"
            "<div style='color:" + color + ";font-weight:700;min-width:160px;font-size:0.88rem;'>"
            + name + "</div>"
            "<div style='color:#94a3b8;font-size:0.85rem;'>" + desc + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='color:#334155;font-size:0.75rem;text-align:center;margin-top:20px;'>"
        "AQI Forecast Dashboard · MLOps Project · OpenAQ v3 + CPCB"
        "</div>",
        unsafe_allow_html=True,
    )