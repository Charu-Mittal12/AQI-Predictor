import streamlit as st
from ui.theme import aqi_meta, POLLUTANT_COLORS


# ── Banners ────────────────────────────────────────────────────────────────────


def show_error(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="error-banner">
            <div class="error-icon">⚠️</div>
            <div>
                <div class="error-title">{title}</div>
                <div class="error-body">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )


def show_warning(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="warn-banner">
            <div class="warn-title">⚡ {title}</div>
            <div class="warn-body">{detail}</div>
        </div>
        """, unsafe_allow_html=True,
    )


def show_info(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="info-banner">
            <div class="info-title">ℹ️ {title}</div>
            <div class="info-body">{detail}</div>
        </div>
        """, unsafe_allow_html=True,
    )


# ── Status / top bar ───────────────────────────────────────────────────────────

def backend_offline_screen(reason: str) -> None:
    st.markdown(
        f"""
        <div class="down-state">
            <div style="font-size:3rem;margin-bottom:16px;">🔴</div>
            <div style="color:#fca5a5;font-size:1.3rem;font-weight:700;margin-bottom:10px;">
                Backend Unreachable
            </div>
            <div style="color:#fecaca;font-size:0.9rem;max-width:400px;margin:0 auto;line-height:1.6;">
                {reason}
            </div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:18px;">
                Make sure FastAPI backend is running at <code>BACKEND_URL</code>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )


def status_chip(status: str) -> str:
    if status in ("ok", "ready"):
        css, dot = "chip chip-ok", "●"
    elif status == "degraded":
        css, dot = "chip chip-warn", "●"
    else:
        css, dot = "chip chip-bad", "●"
    return f'<span class="{css}">{dot} {status.upper()}</span>'


def topbar(backend_status: str, model_ready: bool, city: str, last_updated: str) -> None:
    model_chip = status_chip("ready" if model_ready else "degraded")
    be_chip    = status_chip(backend_status)
    lu = last_updated
    st.markdown(
        f"""
        <div class="topbar">
            <span class="subtle">🌫️ <strong style="color:#e2e8f0;">AQI Forecast</strong></span>
            <span style="flex:1"></span>
            {be_chip} {model_chip}
            <span class="subtle" style="font-size:0.78rem;">📍 {city}</span>
            <span class="subtle" style="font-size:0.78rem;">🕐 {lu}</span>
        </div>
        """, unsafe_allow_html=True,
    )


# ── Hero AQI card ──────────────────────────────────────────────────────────────

def aqi_hero(city: str, aqi: int, primary_pollutant: str) -> None:
    meta = aqi_meta(aqi)
    badge_style = (
        f"background:{meta['bg']};color:{meta['color']};"
        f"border-color:{meta['color']}33;"
    )
    st.markdown(
        f"""
        <div class="glass">
            <div class="subtle" style="margin-bottom:8px;">📍 {city}</div>
            <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
                <span class="hero-number" style="color:{meta['color']};">{aqi}</span>
                <span class="badge" style="{badge_style}">{meta['label']}</span>
            </div>
            <div class="hero-label" style="margin-top:10px;">
                AQI Index &nbsp;·&nbsp; Primary pollutant:
                <strong style="color:#e2e8f0;">{primary_pollutant}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )


# ── Advisory ──────────────────────────────────────────────────────────────────

def advisory_card(text: str) -> None:
    st.markdown(
        f'<div class="advisory-card">🩺 <strong>Health Advisory:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


# ── Pollutants grid ────────────────────────────────────────────────────────────

def pollutants_grid(pollutants: dict) -> None:
    UNIT_MAP = {
        "PM2_5": "µg/m³ PM2.5",
        "PM10":  "µg/m³ PM10",
        "NO2":   "µg/m³ NO₂",
        "SO2":   "µg/m³ SO₂",
        "CO":    "mg/m³ CO",
        "O3":    "µg/m³ O₃",
    }
    pills = ""
    for key, val in pollutants.items():
        color = POLLUTANT_COLORS.get(key, "#94a3b8")
        label = UNIT_MAP.get(key, key)
        v_str = f"{val:.1f}" if isinstance(val, float) else str(val)
        pills += f"""
        <div class="pollutant-pill">
            <div class="pill-label">{label}</div>
            <div class="pill-value" style="color:{color};">{v_str}</div>
        </div>
        """
    st.markdown(f'<div class="pollutant-grid">{pills}</div>', unsafe_allow_html=True)


# ── Empty / landing state ──────────────────────────────────────────────────────

def empty_state(title: str, subtitle: str, icon: str = "🌤️") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-sub">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True,
    )


# ── Model info sidebar card ────────────────────────────────────────────────────

def model_info_sidebar(info: dict) -> None:
    name    = info.get("model_name",             "—")
    family  = info.get("model_family",           "—")
    version = info.get("app_version",            "—")
    cities  = info.get("supported_cities_count", "—")
    horizon = info.get("forecast_horizon_hours", 24)
    commit  = (info.get("git_commit") or "—")[:7]
    metric  = info.get("primary_metric",         "—")
    m_val   = info.get("primary_metric_value")
    m_str   = f"{m_val:.3f}" if isinstance(m_val, float) else "—"

    st.markdown(
        f"""
        <div class="glass" style="font-size:0.83rem;line-height:1.8;">
            <div style="color:#e2e8f0;font-weight:700;margin-bottom:8px;">🤖 Model Info</div>
            <div><span class="subtle">Name</span><br/>
                 <strong style="color:#e2e8f0;">{name}</strong></div>
            <div><span class="subtle">Family · version</span><br/>
                 <strong style="color:#e2e8f0;">{family}</strong>
                 <span class="subtle"> v{version}</span></div>
            <div><span class="subtle">Cities · horizon</span><br/>
                 <strong style="color:#e2e8f0;">{cities} cities</strong>
                 <span class="subtle"> · {horizon}h</span></div>
            <div><span class="subtle">{metric}</span><br/>
                 <strong style="color:#22c55e;">{m_str}</strong></div>
            <div><span class="subtle">Git commit</span><br/>
                 <code style="color:#7dd3fc;">{commit}</code></div>
        </div>
        """, unsafe_allow_html=True,
    )