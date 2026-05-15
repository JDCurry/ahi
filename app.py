"""
AHI — Adaptive Hazard Intelligence Platform
Multi-state demonstration dashboard.
Resilience Analytics Lab, LLC

AHI v2.5 — multi-hazard risk prediction with regional model architecture.
State-aware: select state from sidebar; UI content + calibration loaded
from states/<XX>/config.yaml and states/registry.yaml.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import warnings
import base64
warnings.filterwarnings('ignore')

# Optional imports (folium only)
try:
    import folium
    from folium.features import GeoJsonTooltip
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# State-aware ONNX inference
try:
    from inference_onnx import predict_county_risks_simple, predict_from_ahi_v2
    AHI_V2_AVAILABLE = True
except Exception as e:
    print(f"[IMPORT] inference_onnx import failed: {e}")
    predict_county_risks_simple = None
    predict_from_ahi_v2 = None
    AHI_V2_AVAILABLE = False

# State context loader (registry + per-state config)
from state_context import StateContext, load_registry, deployed_states

get_batch_adjacency = None

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Adaptive Hazard Intelligence",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEVICE = 'cpu'
MAX_FORECAST_DAYS = 14

# ---- State selection ----
# Sidebar dropdown picks the active state. All downstream content
# (counties, calibration, utility names, NWS offices, climate language)
# loads from states/<state_code>/config.yaml.
_REGISTRY = load_registry()
_DEPLOYED = deployed_states()
if not _DEPLOYED:
    st.error("No states are marked `deployed: true` in states/registry.yaml.")
    st.stop()

# Default to first state alphabetically by name
_DEFAULT_STATE = sorted(_DEPLOYED, key=lambda c: _REGISTRY[c]['name'])[0]

with st.sidebar:
    st.markdown("### State")
    _state_options = sorted(_DEPLOYED, key=lambda c: _REGISTRY[c]['name'])
    _state_labels = {c: f"{_REGISTRY[c]['name']} ({c})" for c in _state_options}
    selected_state = st.selectbox(
        "Active state",
        options=_state_options,
        index=_state_options.index(_DEFAULT_STATE) if _DEFAULT_STATE in _state_options else 0,
        format_func=lambda c: _state_labels[c],
        label_visibility='collapsed',
        key='active_state',
    )

@st.cache_resource(show_spinner=False)
def _load_state_context(state_code: str) -> StateContext:
    return StateContext.load(state_code)

ctx = _load_state_context(selected_state)

# Backwards-compatible aliases — preserves the rest of the file's variable names
# so the bulk of the rendering code below didn't have to be rewritten.
CO_COUNTY_COORDS = {k: tuple(v) for k, v in ctx.county_coords.items()}
COUNTIES         = ctx.counties
COUNTY_UTILITY   = ctx.county_utility
HAZARD_GUIDANCE  = ctx.hazard_guidance
MODEL_VERSION    = ctx.model_version
DATA_VERSION     = ctx.data_version

# County coordinates, COUNTIES list, and utility map all come from ctx (above).

# =============================================================================
# COLOR THEME — Resilience Analytics Lab (sage green / institutional)
# =============================================================================

COLORS = {
    'app_bg':        '#24282D',
    'card_bg':       '#161b22',
    'sidebar_bg':    '#0d1117',
    'elevated_bg':   '#1c2128',
    'primary':       '#4a7c59',
    'primary_light': '#6b9e7a',
    'primary_dark':  '#2d5a3a',
    'accent':        '#8fbc8f',
    'border':        '#30363d',
    'text_primary':  '#e6edf3',
    'text_secondary':'#8b949e',
    'text_tertiary': '#6e7681',
    # Per-hazard
    'fire':    '#e05252',
    'flood':   '#4a90d9',
    'wind':    '#9b59b6',
    'winter':  '#2ec4b6',
    'seismic': '#e67e22',
}

# Unified 5-tier sequential risk palette (absolute — used for national overview)
RISK_TIERS = [
    (0.00, 0.10, '#2d5a3a', 'Low',      '< 10%',   'Baseline conditions — routine monitoring'),
    (0.10, 0.20, '#6b9e7a', 'Elevated', '10–20%',  'Above baseline — increased awareness recommended'),
    (0.20, 0.35, '#f59e0b', 'Moderate', '20–35%',  'Notable risk — review preparedness plans'),
    (0.35, 0.50, '#f97316', 'High',     '35–50%',  'Significant risk — consider pre-positioning resources'),
    (0.50, 1.01, '#dc2626', 'Severe',   '> 50%',   'Activate operations — emergency response posture'),
]

def risk_color(prob):
    for lo, hi, c, *_ in RISK_TIERS:
        if lo <= prob < hi:
            return c
    return RISK_TIERS[-1][2]

def risk_level(prob):
    for lo, hi, _, level, _, interp in RISK_TIERS:
        if lo <= prob < hi:
            return level, interp
    return RISK_TIERS[-1][3], RISK_TIERS[-1][5]


# ---------------------------------------------------------------------------
# Relative risk tiers — contextualised against state historical base rates
# ---------------------------------------------------------------------------
# Instead of fixed thresholds, tiers are based on ratio of predicted
# probability to the state's historical base rate for that hazard+month.
# This prevents "normal weather" (e.g. 63% wind in Alabama May) from
# showing as "Severe" while still flagging genuine anomalies.

RELATIVE_TIER_DEFS = [
    # (max_ratio, color, label, description)
    (0.50, '#2d5a3a', 'Low',      'Well below historical norm — routine monitoring'),
    (1.00, '#6b9e7a', 'Normal',   'Near historical baseline — standard operations'),
    (1.50, '#f59e0b', 'Elevated', 'Above historical norm — increased awareness'),
    (2.00, '#f97316', 'High',     'Significantly above baseline — review preparedness'),
    (99.0, '#dc2626', 'Severe',   'Far exceeds historical norm — activate response'),
]


def _load_base_rates(state_code: str) -> dict:
    """Load historical monthly base rates from state's seasonal_bias.json."""
    p = Path(f'states/{state_code}/seasonal_bias.json')
    if not p.exists():
        return {}
    try:
        with open(p, encoding='utf-8-sig') as f:
            doc = json.load(f)
        return doc.get('base_rates', {})
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_base_rate(state_code: str, hazard: str, month: int) -> float:
    """Return historical base rate for a hazard in a given state-month."""
    rates = _load_base_rates(state_code)
    return rates.get(hazard, {}).get(str(month), 0.0)


def risk_level_relative(prob: float, base_rate: float):
    """Assign risk tier relative to historical base rate.

    Returns (label, color, description, ratio_str).
    """
    if base_rate < 0.005:
        # Very rare hazard — use absolute tiers (any signal is notable)
        level, interp = risk_level(prob)
        color = risk_color(prob)
        return level, color, interp, ''

    ratio = prob / base_rate
    for max_r, color, label, desc in RELATIVE_TIER_DEFS:
        if ratio <= max_r:
            if ratio >= 1.0:
                ratio_str = f"{(ratio - 1) * 100:.0f}% above historical"
            else:
                ratio_str = f"{(1 - ratio) * 100:.0f}% below historical"
            return label, color, desc, ratio_str

    # Fallback (ratio > 99)
    t = RELATIVE_TIER_DEFS[-1]
    return t[2], t[1], t[3], f"{(ratio - 1) * 100:.0f}% above historical"


def compute_relative_tiers(df, state_code: str, month: int):
    """Compute tier counts for a state DataFrame using relative thresholds.

    For each county, the dominant hazard's probability is compared against
    that hazard's historical base rate for the state+month.
    """
    counts = {'Severe': 0, 'High': 0, 'Elevated': 0, 'Normal': 0, 'Low': 0}
    base_rates = _load_base_rates(state_code)

    for _, row in df.iterrows():
        max_h = row.get('max_hazard', 'wind')
        max_p = row.get('max_p', 0.0)
        br = base_rates.get(max_h, {}).get(str(month), 0.0)
        level, _, _, _ = risk_level_relative(max_p, br)
        counts[level] = counts.get(level, 0) + 1
    return counts

HAZARD_NAMES = {
    'fire': 'Fire', 'flood': 'Flood', 'wind': 'Wind',
    'winter': 'Winter Storm', 'seismic': 'Seismic'
}

# Hazards shown in the dashboard UI. Seismic is kept in the model but hidden
# from the dashboard until real-time seismic feeds (USGS ShakeAlert) are
# integrated.  The inference pipeline still produces seismic predictions —
# they are simply excluded from display, tier counts, and primary-hazard
# ranking so practitioners see only actionable intelligence.
DISPLAY_HAZARDS = ['fire', 'flood', 'wind', 'winter']

# HAZARD_GUIDANCE, COUNTY_UTILITY, MODEL_VERSION, DATA_VERSION
# all sourced from ctx (StateContext loaded at top of file).

_MONTH_TO_SEASON = {
    12: 'winter', 1: 'winter',  2: 'winter',
     3: 'spring', 4: 'spring',  5: 'spring',
     6: 'summer', 7: 'summer',  8: 'summer',
     9: 'fall',  10: 'fall',   11: 'fall',
}


def generate_audit_report(county: str, forecast_date_str: str,
                          risks: dict, horizon_days: int,
                          state_ctx=None) -> dict:
    """Build a structured audit record for a single county prediction."""
    sctx = state_ctx or ctx
    from datetime import datetime as _dt
    try:
        fdate = _dt.fromisoformat(forecast_date_str)
        month = fdate.month
        season = _MONTH_TO_SEASON.get(month, 'spring')
        date_display = fdate.strftime('%B %d, %Y')
        month_name = fdate.strftime('%B')
    except Exception:
        month, season, date_display, month_name = 4, 'spring', forecast_date_str, 'April'

    # Build audit factors from the correct state context
    audit_factors = {
        (h, s): text
        for h, by_season in sctx.audit_factors.items()
        for s, text in by_season.items()
    }

    nws_codes = sctx.nws_office_codes()
    nws_label = ', '.join(nws_codes) if nws_codes else 'your local NWS offices'

    # Filter to display hazards only (excludes seismic for now)
    display_risks = {h: risks[h] for h in DISPLAY_HAZARDS if h in risks}
    ranked = sorted(display_risks.items(), key=lambda x: x[1], reverse=True)
    primary_key, primary_score = ranked[0]
    second_key,  second_score  = ranked[1] if len(ranked) > 1 else ('', 0.0)
    margin = primary_score - second_score
    level, _ = risk_level(primary_score)

    primary_name = HAZARD_NAMES.get(primary_key, primary_key)
    second_name  = HAZARD_NAMES.get(second_key,  second_key)

    def _pp(m: float) -> str:
        pct = m * 100
        if pct < 1.0:
            return "less than 1 percentage point"
        elif round(pct, 1) == 1.0:
            return "1 percentage point"
        else:
            return f"{pct:.1f} percentage points"

    if margin < 0.02:
        ranking_note = (
            f"{primary_name} leads the ranking by only {_pp(margin)} over "
            f"{second_name}. Treat both as comparable elevated concerns — do not treat this as "
            f"a single definitive priority."
        )
    elif margin < 0.05:
        ranking_note = (
            f"{primary_name} leads by {_pp(margin)}. "
            f"{second_name} should be monitored as a close secondary concern."
        )
    else:
        ranking_note = (
            f"{primary_name} is the clear primary hazard, leading by {_pp(margin)}."
        )

    seasonal_text = audit_factors.get((primary_key, season), '') or \
        f"Historical {month_name} patterns for {primary_name.lower()} risk in {sctx.state_name} informed this prediction."

    factors = [
        {"factor": "Seasonal pattern",
         "explanation": seasonal_text},
        {"factor": "Geographic context",
         "explanation": f"{county} County's location, elevation, and land-cover profile "
                        f"contribute to its baseline {primary_name.lower()} risk relative to "
                        f"other {sctx.state_name} counties. These static features are baked into the model's learned weights."},
        {"factor": "Regional spatial signal",
         "explanation": "Neighboring county patterns are incorporated via the spatial attention "
                        "mesh. Cross-county phenomena — fire spread, downstream flooding, storm tracks "
                        "across the region — influence the regional ranking even for the focal county."},
    ]

    limitations = [
        f"AHI uses historical pattern detection ({sctx.data_version.split(',')[1].strip() if ',' in sctx.data_version else '2000–2025'}), "
        "not live weather feeds. Results reflect seasonal and geographic baselines.",
        "This output is a decision-support tool, not an official forecast. Cross-reference with "
        f"current NWS watches/warnings ({nws_label}) and local situational awareness before operational action.",
        f"Risk probability is a calibrated point-in-time estimate for {date_display} — "
        f"not a cumulative probability across {horizon_days} days.",
    ]

    if margin < 0.02:
        ranking_stability = "Low separation — multiple hazards are within close range and should all be monitored."
    elif margin < 0.05:
        ranking_stability = "Moderate separation — primary hazard leads, but the secondary hazard warrants active attention."
    else:
        ranking_stability = "Clear separation — primary hazard is dominant in this forecast."

    from datetime import datetime as _dt2
    generated_ts = _dt2.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    return {
        "model_version":    sctx.model_version,
        "data_version":     sctx.data_version,
        "forecast_type":    "Point-in-time calibrated risk estimate",
        "generated":        generated_ts,
        "county":           county,
        "forecast_date":    forecast_date_str,
        "horizon_days":     horizon_days,
        "season":           season,
        "primary_hazard":   primary_name,
        "risk_probability": round(primary_score, 4),
        "risk_level":       level,
        "hazard_ranking": [
            {"hazard": HAZARD_NAMES.get(h, h), "probability": round(s, 4),
             "percent": f"{s*100:.1f}%"}
            for h, s in ranked
        ],
        "ranking_note":      ranking_note,
        "ranking_stability": ranking_stability,
        "top_factors":       factors,
        "limitations":       limitations,
    }


def render_decision_audit(audit: dict, state_ctx=None):
    """Render the audit record as a collapsible UI section with JSON export."""
    import json as _json
    primary = audit['primary_hazard']
    level   = audit['risk_level']
    county  = audit['county']
    prob    = audit['risk_probability'] * 100

    with st.expander("Decision Audit", expanded=False):
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']}; border-left:3px solid {COLORS['text_tertiary']};
             padding:12px 16px; border-radius:4px; margin-bottom:12px;">
          <div style="color:{COLORS['text_tertiary']}; font-size:0.75em; text-transform:uppercase;
               letter-spacing:0.05em;">Primary result</div>
          <div style="color:{COLORS['text_primary']}; font-size:1em; margin-top:4px;">
            <strong>{primary}</strong> is the top-ranked hazard for {county} County on
            {audit['forecast_date']} with an estimated probability of
            <strong>{prob:.1f}%</strong> ({level}).
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-bottom:8px;">
          <div style="color:{COLORS['text_tertiary']}; font-size:0.8em; font-weight:600;
               text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
            Ranking context
          </div>
          <div style="color:{COLORS['text_secondary']}; font-size:0.9em;">
            {audit['ranking_note']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        stability = audit.get('ranking_stability', '')
        if stability:
            st.markdown(f"""
            <div style="margin-bottom:12px; padding:6px 10px;
                 background:{COLORS['elevated_bg']}; border-radius:4px;
                 border-left:2px solid {COLORS['text_tertiary']};">
              <span style="color:{COLORS['text_tertiary']}; font-size:0.75em;
                    font-weight:600; text-transform:uppercase;
                    letter-spacing:0.05em;">Ranking stability &nbsp;·&nbsp; </span>
              <span style="color:{COLORS['text_secondary']}; font-size:0.82em;">
                {stability}
              </span>
            </div>
            """, unsafe_allow_html=True)

        cols = st.columns(len(audit['hazard_ranking']))
        for col, entry in zip(cols, audit['hazard_ranking']):
            col.metric(entry['hazard'], entry['percent'])

        st.markdown("<hr style='border-color:#333; margin:12px 0;'>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="color:{COLORS['text_tertiary']}; font-size:0.8em; font-weight:600;
             text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
          Audit Factors
        </div>
        """, unsafe_allow_html=True)
        for f in audit['top_factors']:
            st.markdown(f"""
            <div style="margin-bottom:8px; padding-left:12px;
                 border-left:2px solid {COLORS['text_tertiary']};">
              <span style="color:{COLORS['text_primary']}; font-size:0.85em;
                    font-weight:600;">{f['factor']}</span>
              <span style="color:{COLORS['text_secondary']}; font-size:0.85em;">
                — {f['explanation']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#333; margin:12px 0;'>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="color:{COLORS['text_tertiary']}; font-size:0.8em; font-weight:600;
             text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
          Limitations
        </div>
        """, unsafe_allow_html=True)
        for lim in audit['limitations']:
            st.markdown(
                f'<div style="color:{COLORS["text_secondary"]}; font-size:0.82em; '
                f'margin-bottom:6px;">• {lim}</div>',
                unsafe_allow_html=True
            )

        st.markdown(f"""
        <div style="margin-top:10px; padding:8px 12px;
             background:{COLORS['elevated_bg']}; border-radius:4px;
             border-left:3px solid {COLORS['primary']};">
          <span style="color:{COLORS['text_secondary']}; font-size:0.84em;">
            <strong style="color:{COLORS['primary_light']};">Operational caveat:</strong>
            Use this output alongside current NWS watches/warnings ({', '.join((state_ctx or ctx).nws_office_codes()) or 'your local NWS offices'}),
            local observations, and agency-specific thresholds. It does not replace official
            forecasts or on-the-ground situational awareness.
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:14px; padding:8px 12px;
             background:{COLORS['card_bg']}; border-radius:4px;
             color:{COLORS['text_tertiary']}; font-size:0.74em; font-style:italic;
             line-height:1.7;">
          <strong style="font-style:normal;">Model version:</strong> {audit['model_version']}<br>
          <strong style="font-style:normal;">Data basis:</strong> {audit['data_version']}<br>
          <strong style="font-style:normal;">Forecast type:</strong> {audit.get('forecast_type', 'Point-in-time calibrated risk estimate')}<br>
          <strong style="font-style:normal;">Generated:</strong> {audit.get('generated', '—')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#333; margin:12px 0;'>", unsafe_allow_html=True)

        audit_json = _json.dumps(audit, indent=2)
        st.download_button(
            label="⬇ Download audit record (JSON)",
            data=audit_json,
            file_name=f"ahi_audit_{county.lower().replace(' ', '_')}_{audit['forecast_date']}.json",
            mime="application/json",
            use_container_width=False,
        )


# =============================================================================
# CSS
# =============================================================================

def get_logo_base64():
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background: {COLORS['app_bg']} !important;
        color: {COLORS['text_secondary']} !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }}
    h1, h2, h3 {{
        color: {COLORS['text_primary']} !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }}
    h1 {{
        font-weight: 600 !important;
        border-bottom: 2px solid {COLORS['primary']} !important;
        padding-bottom: 8px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        background: {COLORS['card_bg']};
        border-radius: 8px 8px 0 0;
        padding: 4px 4px 0 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']} !important;
        font-weight: 500;
        padding: 10px 24px;
        border-radius: 6px 6px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary_dark']} !important;
        color: {COLORS['text_primary']} !important;
    }}

    .hazard-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 20px 16px;
        text-align: center;
        transition: border-color 0.2s;
    }}
    .hazard-card:hover {{ border-color: {COLORS['primary']}; }}
    .hazard-card .label {{ font-weight: 700; font-size: 1.1em; margin-bottom: 4px; }}
    .hazard-card .value {{ font-size: 1.6em; font-weight: 600; color: {COLORS['text_primary']}; }}

    .primary-risk-card {{
        background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['elevated_bg']} 100%);
        border: 1px solid {COLORS['border']};
        border-left: 6px solid var(--accent-color, {COLORS['primary']});
        border-radius: 10px;
        padding: 24px 28px;
        margin: 16px 0;
    }}
    .primary-risk-card .eyebrow {{
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75em;
        color: {COLORS['text_tertiary']};
        margin: 0 0 6px 0;
    }}
    .primary-risk-card .headline {{
        font-size: 1.8em;
        font-weight: 700;
        color: {COLORS['text_primary']};
        margin: 0 0 4px 0;
    }}
    .primary-risk-card .percent {{
        font-size: 2.4em;
        font-weight: 700;
        color: var(--accent-color, {COLORS['primary_light']});
        line-height: 1;
    }}
    .primary-risk-card .interp {{
        color: {COLORS['text_secondary']};
        font-style: italic;
        margin: 8px 0 0 0;
    }}

    .stButton > button {{
        background: {COLORS['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.6em 1.5em !important;
    }}
    .stButton > button:hover {{ background: {COLORS['primary_light']} !important; }}

    .stSelectbox {{
        width: 100%;
    }}
    .stSelectbox [data-baseweb="select"] {{
        background: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background: {COLORS['card_bg']} !important;
        color: {COLORS['text_primary']} !important;
    }}
    .stSelectbox [data-baseweb="select"] input {{
        background: {COLORS['card_bg']} !important;
        color: {COLORS['text_primary']} !important;
    }}
    .stSelectbox [data-baseweb="select"]:hover {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 4px 12px rgba(74, 124, 89, 0.2) !important;
    }}
    .stSelectbox [role="listbox"] {{
        background: {COLORS['card_bg']} !important;
    }}
    .stSelectbox [role="option"] {{
        color: {COLORS['text_primary']} !important;
    }}
    .stSlider {{
        width: 100%;
    }}
    .stSlider [data-baseweb="slider"] {{
        padding: 8px 0;
    }}
    .stSlider > div > div > div > span {{
        color: {COLORS['text_primary']} !important;
    }}
    .stDataFrame {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px;
    }}

    .ahi-header-text .title {{
        font-weight: 600 !important;
        color: {COLORS['text_primary']};
        margin: 0 !important;
        padding: 0 !important;
    }}
    .ahi-header-text .subtitle {{
        color: {COLORS['text_secondary']};
        font-size: 0.95em;
        margin: 4px 0 0 0;
        padding: 0 !important;
    }}

    .risk-section {{
        background: {COLORS['card_bg']};
        border-left: 3px solid {COLORS['primary']};
        padding: 16px 20px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="stStatusWidget"] {{display: none;}}
    .viewerBadge_container__r5tak {{display: none !important;}}
    [data-testid="stDecoration"] {{display: none !important;}}
    .stDeployButton {{display: none !important;}}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA & MODEL
# =============================================================================

@st.cache_resource
def load_v2_model(region: str):
    """Check whether the regional ONNX model exists for the active state.

    The model itself is loaded lazily by inference_onnx.py; this function only
    confirms the file is present so the UI can show an honest "model loaded"
    status without paying the session-init cost on every page render.
    """
    if not AHI_V2_AVAILABLE:
        return None, None, False
    for p in [Path(f"models/{region}/model.onnx"),
              Path(f"/mount/src/ahi-platform/models/{region}/model.onnx")]:
        if p.exists():
            print(f"[AHI] Regional ONNX present: {p} "
                  f"({p.stat().st_size / 1024 / 1024:.1f} MB)")
            return "onnx", None, True
    return None, None, False


@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_hazard_data(state_code: str):
    """Load the active state's inference parquet (states/<XX>/inference_data.parquet)."""
    path = Path(f'states/{state_code}/inference_data.parquet')
    if path.exists():
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        # Downcast float64 → float32 to save ~50% RAM per DataFrame
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        return df
    return None


@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_geojson(state_code: str):
    """Load the active state's county GeoJSON (states/<XX>/counties.geojson)."""
    path = Path(f'states/{state_code}/counties.geojson')
    if path.exists():
        with open(path, encoding='utf-8-sig') as f:
            return json.load(f)
    return None


def _normalize_geojson_names(geojson):
    """Add a normalized 'NAME_NORM' property for consistent lookup.

    Uses NAMELSAD (e.g. "Fairfax County", "Alexandria city") when available
    so Virginia independent cities and Louisiana parishes match correctly.
    Falls back to NAME. Strips " County" and upper-cases for case-insensitive
    matching. This means:
      - "Fairfax County" → "FAIRFAX"      (matches prediction "Fairfax")
      - "Alexandria city" → "ALEXANDRIA CITY" (matches prediction "Alexandria City")
      - "Acadia Parish"  → "ACADIA PARISH"   (matches prediction "Acadia Parish")
    """
    if geojson is None:
        return None
    out = json.loads(json.dumps(geojson))
    for feat in out.get('features', []):
        props = feat.get('properties', {})
        # Prefer NAMELSAD (includes designation), fall back to NAME
        name = props.get('NAMELSAD') or props.get('NAME') or props.get('name')
        if name:
            props['NAME_NORM'] = name.replace(' County', '').strip().upper()
    return out


def _get_geojson_features(geojson_data):
    if geojson_data is None:
        return []
    features = geojson_data.get('features', [])
    result = []
    for feat in features:
        props = feat.get('properties', {})
        name = None
        for f in ['NAME', 'name', 'COUNTY', 'county_name']:
            if f in props:
                name = props[f]
                break
        if name:
            name_norm = name.replace(' County', '').strip()
            result.append({'name': name_norm, 'geometry': feat.get('geometry'), 'properties': props})
    return result


def _geometry_bounds(geom):
    coords = []
    def _extract(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)):
                coords.append((obj[0], obj[1]))
            else:
                for item in obj:
                    _extract(item)
    _extract(geom.get('coordinates', []))
    if not coords:
        return (-105.5, 39.0, -104.5, 40.0)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_county(county_name, target_date):
    _, _, ok = load_v2_model(ctx.region)
    if not ok:
        return None, f"Model not loaded — check models/{ctx.region}/model.onnx"
    hazard_df = load_hazard_data(ctx.state_code)
    if hazard_df is None or len(hazard_df) == 0:
        return None, "Hazard dataset not found"
    try:
        from inference_onnx import predict_county_risks_simple as _predict
        return _predict(ctx.state_code, ctx.region, county_name, hazard_df, target_date), None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Prediction failed: {e}"


def predict_all_counties(target_date, progress_callback=None):
    rows = []
    for i, county in enumerate(COUNTIES):
        if progress_callback:
            progress_callback(i, len(COUNTIES), county)
        risks, err = predict_single_county(county, target_date)
        if risks:
            row = {'county': county, 'date': str(target_date)}
            for h in DISPLAY_HAZARDS:
                row[f'{h}_p'] = risks.get(h, 0.0)
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# National-view helpers (used by page_national)
# ---------------------------------------------------------------------------

import unicodedata as _ud

def _ascii_normalize_county(s: str) -> str:
    """Match the geojson builder's ID convention. Handles diacritics + mojibake.
    'Doña Ana' -> 'DONA ANA', 'DoÃ±a Ana' -> 'DONA ANA', 'DONA ANA' -> 'DONA ANA'.
    """
    try:
        s = s.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return _ud.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').upper()


@st.cache_data(show_spinner=False)
def load_national_geojson():
    """Cached load of the national counties geojson (3,109 CONUS + 35 AK/HI)."""
    p = Path('data/national_counties.geojson')
    if not p.exists():
        return None
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# National predictions — loaded from precomputed CSVs for instant startup.
# CSVs are generated offline via:  python scripts/precompute_national.py
# One CSV per month: data/national_predictions_month05.csv, etc.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def predict_all_national(month: int):
    """Load precomputed national predictions for a given month.

    Falls back to live inference if no precomputed file exists (slow).
    Returns DataFrame with columns:
        state, county, county_id, fire_p, flood_p, wind_p, winter_p,
        max_p, max_hazard
    """
    csv_path = Path(f'data/national_predictions_month{month:02d}.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Recompute max from DISPLAY_HAZARDS only (excludes seismic)
        p_cols = [f'{h}_p' for h in DISPLAY_HAZARDS]
        df['max_p'] = df[p_cols].max(axis=1)
        df['max_hazard'] = df[p_cols].idxmax(axis=1).str.replace('_p', '')
        print(f"[NATIONAL] Loaded precomputed month {month}: "
              f"{len(df)} counties from {csv_path}")
        return df

    # Fallback: live inference (slow, memory-heavy — for dev only)
    print(f"[NATIONAL] No precomputed CSV for month {month} — running live inference")
    return _predict_all_national_live(month)


def _predict_all_national_live(month: int):
    """Live inference fallback — runs ONNX for every county. Slow."""
    from datetime import datetime
    from inference_onnx import predict_county_risks_simple as _predict, _build_maps

    target_date = datetime.now().date().replace(day=15) if month == datetime.now().month \
                    else datetime(2026, month, 15).date()

    registry = load_registry()
    deployed = [code for code, m in registry.items() if m.get('deployed', False)]
    rows = []

    for state_code in deployed:
        try:
            state_ctx = StateContext.load(state_code)
        except Exception:
            continue
        hazard_df = pd.read_parquet(state_ctx.parquet_path) if state_ctx.parquet_path.exists() else None
        if hazard_df is None or len(hazard_df) == 0:
            continue
        _build_maps(hazard_df)

        for county in state_ctx.counties:
            try:
                risks = _predict(state_ctx.state_code, state_ctx.region,
                                  county, hazard_df, target_date)
            except Exception:
                continue
            if not risks:
                continue
            rows.append({
                'state':     state_code,
                'county':    county,
                'county_id': _ascii_normalize_county(county.strip()),
                'fire_p':    risks.get('fire', 0.0),
                'flood_p':   risks.get('flood', 0.0),
                'wind_p':    risks.get('wind', 0.0),
                'winter_p':  risks.get('winter', 0.0),
                'seismic_p': risks.get('seismic', 0.0),
            })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    p_cols = [f'{h}_p' for h in DISPLAY_HAZARDS]
    df['max_p'] = df[p_cols].max(axis=1)
    df['max_hazard'] = df[p_cols].idxmax(axis=1).str.replace('_p', '')
    return df


# =============================================================================
# UI HELPERS
# =============================================================================

def render_primary_risk_callout(risks):
    """Hero card highlighting the top-ranked hazard."""
    display_risks = {h: v for h, v in risks.items() if h in DISPLAY_HAZARDS}
    sorted_risks = sorted(display_risks.items(), key=lambda x: x[1], reverse=True)
    top_hazard, top_prob = sorted_risks[0]
    level, interp = risk_level(top_prob)
    color = COLORS.get(top_hazard, COLORS['primary_light'])

    st.markdown(f"""
    <div class="primary-risk-card" style="--accent-color: {color};">
        <p class="eyebrow">Primary Risk — This Forecast Window</p>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px;">
            <div>
                <div class="headline">{HAZARD_NAMES.get(top_hazard, top_hazard.title())}</div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.95em;">Risk Level: <strong style="color: {color};">{level}</strong></div>
            </div>
            <div class="percent">{top_prob*100:.1f}%</div>
        </div>
        <p class="interp">{interp}</p>
    </div>
    """, unsafe_allow_html=True)


def render_hazard_cards(risks):
    """Ranked hazard probability cards."""
    display_risks = {h: risks[h] for h in DISPLAY_HAZARDS if h in risks}
    sorted_hazards = sorted(display_risks.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(len(sorted_hazards))
    for i, (col, (hazard, prob)) in enumerate(zip(cols, sorted_hazards)):
        pct = f"{prob * 100:.1f}%"
        color = COLORS.get(hazard, COLORS['primary'])
        rank_label = "#1 Primary" if i == 0 else f"#{i+1}"
        with col:
            st.markdown(f"""
            <div class="hazard-card">
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.7em; letter-spacing: 0.1em; text-transform: uppercase;">{rank_label}</div>
                <div class="label" style="color: {color};">{HAZARD_NAMES.get(hazard, hazard.title())}</div>
                <div class="value">{pct}</div>
            </div>
            """, unsafe_allow_html=True)


_DEFAULT_GUIDANCE = {
    'fire': {
        'Low':      'Routine wildfire awareness. Confirm defensible-space compliance and evacuation route familiarity.',
        'Elevated': 'Monitor local fire-weather forecasts (Red Flag warnings). Review pre-attack plans and water-supply accessibility.',
        'Moderate': 'Increase readiness posture. Coordinate with local fire districts on mutual-aid availability and staging.',
        'High':     'Pre-position suppression resources. Issue community preparedness advisories and confirm evacuation notification systems.',
        'Severe':   'Activate wildfire response operations. Coordinate with state forestry and federal partners on suppression and evacuation.',
    },
    'flood': {
        'Low':      'Routine flood awareness. Confirm storm-drain maintenance and floodplain status.',
        'Elevated': 'Monitor NWS river gauges and flash-flood watches. Review flood-response plans and sandbag inventory.',
        'Moderate': 'Increase monitoring of upstream conditions. Pre-stage flood barriers and coordinate with public works.',
        'High':     'Pre-position flood-response assets. Issue public advisories for low-lying areas and confirm shelter availability.',
        'Severe':   'Activate flood-response operations. Coordinate evacuations for flood-prone zones and request mutual-aid support.',
    },
    'wind': {
        'Low':      'Routine severe-weather awareness. Confirm tree-trimming schedules and backup power availability.',
        'Elevated': 'Monitor severe-weather outlooks. Review debris-management plans and utility coordination contacts.',
        'Moderate': 'Increase readiness for wind-related impacts. Coordinate with utility providers on outage-response priorities.',
        'High':     'Pre-position damage-assessment teams. Issue public advisories on securing loose objects and shelter-in-place procedures.',
        'Severe':   'Activate severe-wind response operations. Coordinate with utilities on restoration priorities and mutual-aid deployment.',
    },
    'winter': {
        'Low':      'Routine winter-weather awareness. Confirm road-treatment supplies and cold-weather shelter capacity.',
        'Elevated': 'Monitor winter-storm watches. Review snow-removal priorities and coordinate with transportation agencies.',
        'Moderate': 'Increase readiness for winter impacts. Pre-stage road-treatment materials and confirm warming-center availability.',
        'High':     'Pre-position winter-response resources. Issue travel advisories and coordinate with utilities on outage preparedness.',
        'Severe':   'Activate winter-storm response operations. Coordinate road closures, warming shelters, and utility restoration.',
    },
}


def render_risk_summary(risks, county=''):
    display_risks = {h: risks[h] for h in DISPLAY_HAZARDS if h in risks}
    sorted_risks = sorted(display_risks.items(), key=lambda x: x[1], reverse=True)
    st.markdown("#### Top Hazards — Recommended Actions")
    for hazard, prob in sorted_risks[:3]:
        level, interpretation = risk_level(prob)
        # Use state-specific guidance if available, otherwise generic defaults
        guidance = HAZARD_GUIDANCE.get(hazard, {}).get(level, '')
        if not guidance:
            guidance = _DEFAULT_GUIDANCE.get(hazard, {}).get(level, '')
        # Append county utility contact for wind/winter at Elevated tier and above
        if hazard in ('wind', 'winter') and level != 'Low' and county:
            utility = COUNTY_UTILITY.get(county, 'contact local utility provider')
            guidance = f"{guidance} Primary utility: {utility}."
        color = COLORS.get(hazard, COLORS['text_primary'])
        st.markdown(f"""
        <div class="risk-section">
            <h4 style="color: {color}; margin: 0 0 4px 0;">{HAZARD_NAMES.get(hazard, hazard.title())} — {prob*100:.1f}% ({level})</h4>
            <p style="color: {COLORS['text_secondary']}; margin: 2px 0; font-style: italic;">{interpretation}</p>
            <p style="color: {COLORS['text_primary']}; margin: 6px 0 0 0; font-size: 0.9em;"><strong>Suggested actions:</strong> {guidance}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(
        f"""<p style="color: {COLORS['text_tertiary']}; font-size: 0.75em;
        font-style: italic; margin-top: 8px;">
        Guidance reflects hazard-tier operational doctrine. Fire, flood, and wind
        county-specific resource integration (local fire districts, flood authorities,
        critical facility registry) is a Phase I Aim 3 deliverable developed in
        partnership with pilot sites.
        </p>""",
        unsafe_allow_html=True
    )


def render_interpretation_guide(forecast_days, state_ctx=None):
    """Expandable guide using the actual forecast horizon."""
    sctx = state_ctx or ctx
    nws = ', '.join(sctx.nws_office_codes()) or 'your local NWS offices'
    with st.expander("How to interpret these numbers", expanded=False):
        st.markdown(f"""
        **What the percentages mean:**
        - These are **calibrated risk probabilities for a single point-in-time**: conditions on the forecast date ({forecast_days} days from today), not an average across the window
        - Changing the forecast date changes the target date for inference — different dates have different seasonal weights, so a 7-day and 14-day run can produce different rankings if the window crosses a seasonal transition
        - Probabilities are based on **25 years of historical patterns** (2000–2025) across all {len(sctx.counties)} {sctx.state_name} counties
        - A county with few historical events can still show elevated risk if current seasonal/geographic conditions match patterns that preceded events elsewhere

        **Risk thresholds:**
        | Level | Range | Suggested Response |
        |-------|-------|--------------------|
        | Low | < 10% | Routine monitoring |
        | Elevated | 10–20% | Increased awareness |
        | Moderate | 20–35% | Review preparedness |
        | High | 35–50% | Pre-position resources |
        | Severe | > 50% | Activate operations |

        **Important:** AHI uses historical pattern detection, not live weather feeds.
        Predictions reflect seasonal and geographic baselines — always cross-reference with
        current NWS watches/warnings ({nws}) for operational decisions.
        """)


# =============================================================================
# MAP: Plotly choropleth
# =============================================================================

def _auto_zoom_from_coords(county_coords):
    """Compute Mapbox zoom + center from county coordinate dict."""
    coords = list(county_coords.values()) if county_coords else [(39.0, -105.5)]
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    center = {'lat': sum(lats) / len(lats), 'lon': sum(lons) / len(lons)}
    max_range = max(max(lats) - min(lats), max(lons) - min(lons), 0.01)
    if max_range > 15:    zoom = 3.5
    elif max_range > 10:  zoom = 4.0
    elif max_range > 6:   zoom = 5.0
    elif max_range > 3:   zoom = 5.8
    elif max_range > 1.5: zoom = 6.5
    elif max_range > 0.5: zoom = 7.5
    elif max_range > 0.1: zoom = 8.5
    else:                 zoom = 10.5   # very small areas (DC, single county)
    return center, zoom


def render_statewide_choropleth(df, hazard_key, hazard_label,
                                 state_code, county_coords,
                                 map_style='Dark'):
    """Statewide risk map using Choroplethmapbox (same style as national tab)."""
    geojson_data = load_geojson(state_code)
    if geojson_data is None:
        st.info(f"GeoJSON not found for {state_code}. County map unavailable.")
        return

    geojson_norm = _normalize_geojson_names(geojson_data)
    col_name = f"{hazard_key}_p"

    plot_df = df.copy()
    plot_df['county_norm'] = plot_df['county'].str.replace(' County', '').str.strip().str.upper()
    plot_df['_display'] = plot_df['county'].apply(_county_display_name)
    plot_df['pct'] = plot_df[col_name] * 100

    style = _NATIONAL_TILE_STYLES.get(map_style, _NATIONAL_TILE_STYLES['Dark'])

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_norm,
        locations=plot_df['county_norm'],
        z=plot_df['pct'],
        featureidkey="properties.NAME_NORM",
        colorscale=[
            [0.00, '#2d5a3a'], [0.10, '#2d5a3a'],
            [0.10, '#6b9e7a'], [0.20, '#6b9e7a'],
            [0.20, '#f59e0b'], [0.35, '#f59e0b'],
            [0.35, '#f97316'], [0.50, '#f97316'],
            [0.50, '#dc2626'], [1.00, '#dc2626'],
        ],
        zmin=0, zmax=100,
        marker_line_width=0.6,
        marker_line_color=style['border'],
        marker_opacity=style['opacity'],
        colorbar=dict(
            title_text=f"{hazard_label} Risk (%)",
            title_font_color=COLORS['text_secondary'],
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            bgcolor=COLORS['card_bg'],
            thickness=12, len=0.6,
        ),
        customdata=plot_df[['_display']].values,
        hovertemplate="<b>%{customdata[0]}</b><br>" + hazard_label + ": %{z:.1f}%<extra></extra>",
    ))

    center, zoom = _auto_zoom_from_coords(county_coords)
    fig.update_layout(
        mapbox_style=style['mapbox_style'],
        mapbox_layers=style['mapbox_layers'],
        mapbox_zoom=zoom,
        mapbox_center=center,
        paper_bgcolor=COLORS['card_bg'],
        margin=dict(l=0, r=0, t=10, b=10),
        height=540,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_county_spotlight_map(selected_county, risks, target_date,
                                 state_code, county_coords):
    """State map with selected county highlighted, all others dimmed.
    Uses Choroplethmapbox for consistent styling with national/state tabs."""
    geojson_data = load_geojson(state_code)
    geojson_norm = _normalize_geojson_names(geojson_data)
    if geojson_norm is None:
        st.info(f"GeoJSON not available for {state_code}.")
        return

    selected_norm = selected_county.replace(' County', '').strip().upper()
    selected_display = _county_display_name(selected_county.replace(' County', '').strip())

    ordered_hazard_keys = sorted(
        DISPLAY_HAZARDS,
        key=lambda h: risks.get(h, 0.0),
        reverse=True
    )

    hazard_choice = st.selectbox(
        "Overlay hazard (ranked by this county's risk)",
        ordered_hazard_keys,
        format_func=lambda h: HAZARD_NAMES.get(h, h.title()),
        key='county_hazard_select'
    )
    hkey = hazard_choice
    sel_prob = risks.get(hkey, 0.0) * 100

    all_names = []
    for feat in geojson_norm.get('features', []):
        n = feat.get('properties', {}).get('NAME_NORM')
        if n:
            all_names.append(n)

    risk_colorscale = [
        [0.00, '#2d5a3a'], [0.10, '#2d5a3a'],
        [0.10, '#6b9e7a'], [0.20, '#6b9e7a'],
        [0.20, '#f59e0b'], [0.35, '#f59e0b'],
        [0.35, '#f97316'], [0.50, '#f97316'],
        [0.50, '#dc2626'], [1.00, '#dc2626'],
    ]

    background_names = [n for n in all_names if n != selected_norm]

    # Background counties (dimmed)
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_norm,
        locations=background_names,
        z=[0] * len(background_names),
        featureidkey="properties.NAME_NORM",
        colorscale=[[0, '#2a3140'], [1, '#2a3140']],
        zmin=0, zmax=100,
        showscale=False,
        marker_line_color='#4a5568',
        marker_line_width=0.5,
        marker_opacity=0.5,
        hovertemplate="<b>%{location}</b><extra></extra>",
    ))

    # Selected county (highlighted)
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_norm,
        locations=[selected_norm],
        z=[sel_prob],
        featureidkey="properties.NAME_NORM",
        colorscale=risk_colorscale,
        zmin=0, zmax=100,
        showscale=False,
        marker_line_color='#fbbf24',
        marker_line_width=2,
        marker_opacity=0.9,
        hovertemplate=f"<b>{selected_display}</b><br>{HAZARD_NAMES.get(hazard_choice, hazard_choice.title())}: %{{z:.1f}}%<extra></extra>",
    ))

    center, zoom = _auto_zoom_from_coords(county_coords)
    fig.update_layout(
        mapbox_style='carto-darkmatter',
        mapbox_zoom=zoom,
        mapbox_center=center,
        paper_bgcolor=COLORS['card_bg'],
        margin=dict(l=0, r=0, t=10, b=10),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: COUNTY RISK ASSESSMENT
# =============================================================================

def page_quick_predict():
    st.markdown("## County Risk Assessment")
    st.caption("Analyze hazard risk for a single county. Assessment based on 25 years of historical hazard patterns.")

    # State + county + horizon selectors
    col_st, col_county, col_hz = st.columns([1.2, 2, 1])
    with col_st:
        _cr_state_options = sorted(_DEPLOYED, key=lambda c: _REGISTRY[c]['name'])
        _cr_state_labels = {c: f"{_REGISTRY[c]['name']} ({c})" for c in _cr_state_options}
        # Default to whatever was selected on the State tab, or sidebar
        _cr_default = st.session_state.get('state_tab_state', selected_state)
        cr_state = st.selectbox(
            "Select State",
            options=_cr_state_options,
            index=_cr_state_options.index(_cr_default) if _cr_default in _cr_state_options else 0,
            format_func=lambda c: _cr_state_labels[c],
            key='county_tab_state',
        )
    cr_ctx = _load_state_context(cr_state)
    with col_county:
        selected_county = st.selectbox("Select County", cr_ctx.counties, index=0)
    with col_hz:
        forecast_horizon = st.selectbox("Forecast Window",
                                         options=[14, 30], index=0,
                                         format_func=lambda d: f"{d} days",
                                         key='county_tab_horizon')

    days = forecast_horizon
    today = datetime.now().date()
    target_date = today + timedelta(days=days)

    lat, lon = cr_ctx.county_coords.get(selected_county, (39.0, -105.5))
    month_name = target_date.strftime('%B')
    month = target_date.month

    # Generic season notes (no longer CO-specific)
    if month in [3, 4, 5]:
        season_note = "Spring — snowmelt and convective flooding; transitional fire risk"
    elif month in [6, 7, 8]:
        season_note = "Summer — peak fire and convective storm season"
    elif month in [9, 10, 11]:
        season_note = "Fall — wind events; early-season winter storms"
    else:
        season_note = "Winter — snow, ice, and wind events"

    st.markdown(f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 20px; margin: 12px 0;">
        <div style="display: flex; gap: 32px; flex-wrap: wrap;">
            <div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">Location</div>
                <div style="color: {COLORS['text_primary']}; font-size: 1.1em; font-weight: 600;">{_county_display_name(selected_county)}, {cr_ctx.state_name}</div>
            </div>
            <div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">Forecast Date</div>
                <div style="color: {COLORS['text_primary']}; font-size: 1.1em;">{target_date.strftime('%B %d, %Y')} <span style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">({days}-day outlook)</span></div>
            </div>
            <div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">Season</div>
                <div style="color: {COLORS['text_primary']}; font-size: 1.1em;">{month_name} — {season_note}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        status = st.empty()
        status.info("Loading ONNX model…")
        _, _, ok = load_v2_model(cr_ctx.region)
        if not ok:
            status.error(f"Model unavailable — check models/{cr_ctx.region}/model.onnx")
            return
        status.info("Extracting county features…")
        hazard_df = load_hazard_data(cr_state)
        status.info("Running model inference…")
        risks, err = None, None
        try:
            from inference_onnx import predict_county_risks_simple as _predict
            risks = _predict(cr_state, cr_ctx.region, selected_county,
                              hazard_df, target_date)
        except Exception as e:
            err = str(e)
        if risks is None:
            status.error(f"Prediction failed: {err}")
        else:
            status.info("Applying calibration (temperature scaling + seasonal bias)…")
            time.sleep(0.15)
            status.empty()
            audit = generate_audit_report(
                selected_county, str(target_date), risks, days,
                state_ctx=cr_ctx,
            )
            st.session_state['last_prediction'] = {
                'county':  selected_county,
                'date':    str(target_date),
                'risks':   risks,
                'horizon': days,
                'audit':   audit,
            }

    if 'last_prediction' in st.session_state:
        last = st.session_state['last_prediction']
        if last.get('county') == selected_county:
            st.markdown("---")
            render_primary_risk_callout(last['risks'])
            render_hazard_cards(last['risks'])
            st.markdown("")
            render_risk_summary(last['risks'], county=last.get('county', ''))
            st.markdown("---")
            with st.expander("County Spotlight Map", expanded=False):
                render_county_spotlight_map(selected_county, last['risks'], last.get('date'),
                                             state_code=cr_state, county_coords=cr_ctx.county_coords)
            if 'audit' in last:
                render_decision_audit(last['audit'], state_ctx=cr_ctx)
            render_interpretation_guide(last.get('horizon', days), state_ctx=cr_ctx)


# =============================================================================
# PAGE: STATE OVERVIEW  (National → State drill-down)
# =============================================================================

def page_state_overview():
    """State-level situational awareness: select a state, see aggregated hazard
    risk across all counties, a ranked county table, and a choropleth map.
    Serves as the bridge between the National tab and the County Risk Assessment."""

    # ---- State selector (no horizon — state view is a monthly overview) ----
    state_options = sorted(_DEPLOYED, key=lambda c: _REGISTRY[c]['name'])
    state_labels = {c: f"{_REGISTRY[c]['name']} ({c})" for c in state_options}
    sel_state = st.selectbox(
        "Select State",
        options=state_options,
        index=state_options.index(selected_state) if selected_state in state_options else 0,
        format_func=lambda c: state_labels[c],
        key='state_tab_state',
    )

    # Load the selected state's context
    state_ctx = _load_state_context(sel_state)
    target_date = datetime.now().date() + timedelta(days=14)
    month_label = target_date.strftime('%B %Y')

    st.markdown(f"## {state_ctx.state_name} — Statewide Risk Assessment")
    st.caption(f"{len(state_ctx.counties)} counties · {month_label} outlook · "
               f"Based on historical patterns for {target_date.strftime('%B')}. "
               f"Use the **County Risk Assessment** tab for specific date forecasts.")

    # ---- Run predictions for all counties ----
    # Keep only one state overview at a time to limit memory
    cache_key = 'state_overview_df'
    if st.session_state.get('state_overview_state') != sel_state:
        with st.spinner(f"Running predictions for {len(state_ctx.counties)} "
                         f"{state_ctx.state_name} counties…"):
            hazard_df = load_hazard_data(sel_state)
            if hazard_df is None:
                st.error(f"No inference data for {sel_state}. "
                         f"Check states/{sel_state}/inference_data.parquet.")
                return
            from inference_onnx import predict_county_risks_simple as _predict
            rows = []
            for county in state_ctx.counties:
                try:
                    risks = _predict(sel_state, state_ctx.region,
                                      county, hazard_df, target_date)
                except Exception:
                    continue
                if risks:
                    row = {'county': county, 'date': str(target_date)}
                    for h in DISPLAY_HAZARDS:
                        row[f'{h}_p'] = risks.get(h, 0.0)
                    rows.append(row)
            if rows:
                st.session_state[cache_key] = pd.DataFrame(rows)
                st.session_state['state_overview_state'] = sel_state
            else:
                st.error("No predictions generated.")
                return
    df = st.session_state[cache_key]

    hazards = DISPLAY_HAZARDS
    df['max_p'] = df[[f'{h}_p' for h in hazards]].max(axis=1)
    df['max_hazard'] = df[[f'{h}_p' for h in hazards]].idxmax(axis=1).str.replace('_p', '')

    # ---- Statewide situational awareness hero ----
    # Aggregate: mean and max risk per hazard across all counties
    st.markdown("### Situational Awareness")
    month = target_date.month
    mean_risks = {h: df[f'{h}_p'].mean() for h in hazards}
    primary_hazard = max(mean_risks, key=mean_risks.get)
    primary_mean = mean_risks[primary_hazard]
    primary_br = get_base_rate(sel_state, primary_hazard, month)
    level, rel_color, interp, ratio_str = risk_level_relative(primary_mean, primary_br)
    color = COLORS.get(primary_hazard, COLORS['primary_light'])

    context_line = f"Mean risk: <strong style='color: {rel_color};'>{level}</strong>"
    if ratio_str:
        context_line += f" · {ratio_str}"
    context_line += f" · {(df['max_hazard'] == primary_hazard).sum()} of {len(df)} counties led by this hazard"
    if primary_br >= 0.005:
        context_line += f" · Historical avg: {primary_br*100:.1f}%"

    st.markdown(f"""
    <div class="primary-risk-card" style="--accent-color: {color};">
        <p class="eyebrow">Primary Statewide Risk — {state_ctx.state_name}</p>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px;">
            <div>
                <div class="headline">{HAZARD_NAMES.get(primary_hazard, primary_hazard.title())}</div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.95em;">
                    {context_line}
                </div>
            </div>
            <div class="percent">{primary_mean*100:.1f}%</div>
        </div>
        <p class="interp">{interp}</p>
    </div>
    """, unsafe_allow_html=True)

    # Hazard summary cards (mean across state, with historical context)
    sorted_hazards = sorted(mean_risks.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(len(sorted_hazards))
    for i, (col, (hazard, mean_p)) in enumerate(zip(cols, sorted_hazards)):
        rank_label = "#1 Primary" if i == 0 else f"#{i+1}"
        hcolor = COLORS.get(hazard, COLORS['primary'])
        with col:
            st.markdown(f"""
            <div class="hazard-card">
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.7em; letter-spacing: 0.1em; text-transform: uppercase;">{rank_label}</div>
                <div class="label" style="color: {hcolor};">{HAZARD_NAMES.get(hazard, hazard.title())}</div>
                <div class="value">{mean_p*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Risk tier distribution (relative to historical base rates)
    st.markdown("---")
    rel_tiers = compute_relative_tiers(df, sel_state, month)
    tier_cols = st.columns(5)
    tier_cols[0].metric("Severe",   rel_tiers.get('Severe', 0))
    tier_cols[1].metric("High",     rel_tiers.get('High', 0))
    tier_cols[2].metric("Elevated", rel_tiers.get('Elevated', 0))
    tier_cols[3].metric("Normal",   rel_tiers.get('Normal', 0))
    tier_cols[4].metric("Low",      rel_tiers.get('Low', 0))

    # ---- County table ----
    st.markdown("---")
    st.markdown("### All Counties")
    base_rates = _load_base_rates(sel_state)
    display = df.copy()
    for h in hazards:
        display[HAZARD_NAMES[h]] = (display[f'{h}_p'] * 100).round(1)

    def _county_status(row):
        max_h = row.get('max_hazard', 'wind')
        max_p = row.get('max_p', 0.0)
        br = base_rates.get(max_h, {}).get(str(month), 0.0)
        level, _, _, _ = risk_level_relative(max_p, br)
        return level

    display['Status'] = display.apply(_county_status, axis=1)
    show = display[['county', 'Status'] + [HAZARD_NAMES[h] for h in hazards]].rename(
        columns={'county': 'County'})
    col_config = {
        HAZARD_NAMES[h]: st.column_config.NumberColumn(
            HAZARD_NAMES[h], format="%.1f%%")
        for h in hazards
    }
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config=col_config,
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "Download Predictions (CSV)", data=csv,
        file_name=f"ahi_{sel_state.lower()}_statewide_{target_date}.csv",
        mime="text/csv"
    )

    # ---- State choropleth (Choroplethmapbox — same style as national tab) ----
    st.markdown("---")
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        hazard_choice = st.selectbox(
            "Hazard layer",
            DISPLAY_HAZARDS, index=0,
            format_func=lambda h: HAZARD_NAMES.get(h, h.title()),
            key='state_overview_hazard_map',
        )
    with mc2:
        state_map_style = st.selectbox(
            "Map style",
            options=list(_NATIONAL_TILE_STYLES.keys()),
            index=0,
            key='state_map_style',
        )
    render_statewide_choropleth(
        df, hazard_choice, HAZARD_NAMES.get(hazard_choice, hazard_choice.title()),
        state_code=sel_state,
        county_coords=state_ctx.county_coords,
        map_style=state_map_style,
    )

    map_hazard_br = get_base_rate(sel_state, hazard_choice, month)
    map_hazard_name = HAZARD_NAMES.get(hazard_choice, hazard_choice.title())
    if map_hazard_br >= 0.005:
        st.caption(
            f"Map colors show absolute risk. {state_ctx.state_name} historically "
            f"averages **{map_hazard_br*100:.1f}%** {map_hazard_name.lower()} risk "
            f"in month {month} — counties near that baseline are operating normally. "
            f"Risk tiers above compare each county to this baseline, not to the "
            f"absolute scale."
        )
    else:
        st.caption(
            f"Map colors show absolute risk. {map_hazard_name} has a very low "
            f"historical base rate in {state_ctx.state_name}, so absolute and "
            f"relative tiers align closely."
        )


# =============================================================================
# PAGE: STATEWIDE (legacy — kept for reference, no longer in tab list)
# =============================================================================

def page_statewide():
    st.markdown("## Statewide Predictions")
    st.caption(f"Run AHI v2.5 for all {len(COUNTIES)} {ctx.state_name} counties. Results include an interactive risk map.")

    target_date = datetime.now().date() + timedelta(days=MAX_FORECAST_DAYS)

    if st.button("Run Statewide Predictions", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def callback(i, total, county):
            progress.progress((i + 1) / total)
            status.text(f"Inferring {county}… ({i+1}/{total})")

        df = predict_all_counties(target_date, progress_callback=callback)
        progress.progress(1.0)
        status.text("Complete.")

        if df is not None and len(df) > 0:
            st.session_state['statewide'] = df
            st.success(f"Predictions complete for {len(df)} counties.")
        else:
            st.error("No predictions generated. Check model availability.")

    if 'statewide' not in st.session_state:
        st.info("Click **Run Statewide Predictions** to generate results.")
        return

    df = st.session_state['statewide']
    hazards = DISPLAY_HAZARDS

    # Keep as floats (0–100 range) so column-header click sorts numerically.
    # st.column_config.NumberColumn handles the "%" suffix at render time.
    display = df.copy()
    for h in hazards:
        col = f'{h}_p'
        if col in display.columns:
            display[h.title()] = (display[col] * 100).round(1)
    st.dataframe(
        display[['county'] + [h.title() for h in hazards]].rename(columns={'county': 'County'}),
        use_container_width=True, hide_index=True,
        column_config={
            h.title(): st.column_config.NumberColumn(
                h.title(), format="%.1f%%",
            )
            for h in hazards
        },
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "Download Predictions (CSV)", data=csv,
        file_name=f"ahi_{ctx.state_code.lower()}_statewide_{target_date}.csv", mime="text/csv"
    )

    st.markdown("---")

    hazard_choice = st.selectbox(
        "Select hazard to display on map",
        DISPLAY_HAZARDS, index=0,
        format_func=lambda h: HAZARD_NAMES.get(h, h.title()),
    )
    render_statewide_choropleth(df, hazard_choice, HAZARD_NAMES.get(hazard_choice, hazard_choice.title()),
                                state_code=ctx.state_code,
                                county_coords=ctx.county_coords)


# =============================================================================
# PAGE: RISK ASSESSMENT (portfolio view)
# =============================================================================

def page_risk_assessment():
    st.markdown("## Comprehensive Risk Assessment")
    st.caption(f"Portfolio-level view of model predictions across all {len(COUNTIES)} {ctx.state_name} counties.")

    if 'statewide' not in st.session_state:
        st.info("No statewide predictions yet. Run them now to populate this view.")
        if st.button("Run Statewide Predictions ▶", type="primary"):
            target_date = datetime.now().date() + timedelta(days=MAX_FORECAST_DAYS)
            progress = st.progress(0)
            status = st.empty()

            def _callback(i, total, county):
                progress.progress((i + 1) / total)
                status.text(f"Inferring {county}… ({i+1}/{total})")

            df_sw = predict_all_counties(target_date, progress_callback=_callback)
            progress.progress(1.0)
            status.text("Complete.")

            if df_sw is not None and len(df_sw) > 0:
                st.session_state['statewide'] = df_sw
                st.rerun()
            else:
                st.error("No predictions generated. Check model availability.")
        return

    df = st.session_state['statewide'].copy()
    hazards = DISPLAY_HAZARDS

    df['max_p'] = df[[f'{h}_p' for h in hazards]].max(axis=1)
    df['max_hazard'] = df[[f'{h}_p' for h in hazards]].idxmax(axis=1).str.replace('_p', '').map(HAZARD_NAMES)

    severe   = int((df['max_p'] >= 0.50).sum())
    high     = int(((df['max_p'] >= 0.35) & (df['max_p'] < 0.50)).sum())
    moderate = int(((df['max_p'] >= 0.20) & (df['max_p'] < 0.35)).sum())
    elevated = int(((df['max_p'] >= 0.10) & (df['max_p'] < 0.20)).sum())
    low      = int((df['max_p'] < 0.10).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Severe (>50%)", severe)
    c2.metric("High (35–50%)", high)
    c3.metric("Moderate (20–35%)", moderate)
    c4.metric("Elevated (10–20%)", elevated)
    c5.metric("Low (<10%)", low)

    st.markdown("---")

    st.markdown("### Per-Hazard Statewide Summary")
    hazard_rows = []
    for h in hazards:
        col = f'{h}_p'
        idx_max = df[col].idxmax()
        hazard_rows.append({
            'Hazard': HAZARD_NAMES[h],
            # Floats so column-header sort works numerically; column_config below
            # handles the "%" suffix display.
            'Statewide Mean': round(df[col].mean() * 100, 1),
            'Median':         round(df[col].median() * 100, 1),
            'Max':            round(df[col].max() * 100, 1),
            'Highest County': df.loc[idx_max, 'county'],
            'Counties ≥ 20%': int((df[col] >= 0.20).sum()),
        })
    st.dataframe(
        pd.DataFrame(hazard_rows),
        use_container_width=True, hide_index=True,
        column_config={
            'Statewide Mean': st.column_config.NumberColumn('Statewide Mean', format="%.1f%%"),
            'Median':         st.column_config.NumberColumn('Median',         format="%.1f%%"),
            'Max':            st.column_config.NumberColumn('Max',            format="%.1f%%"),
        },
    )

    st.markdown("---")

    st.markdown("### County Ranking by Hazard")
    rank_col1, rank_col2 = st.columns([0.9, 3.1], gap="large")
    with rank_col1:
        st.markdown("")
        rank_hazard = st.selectbox("Hazard", [HAZARD_NAMES[h] for h in hazards], index=0, key='rank_hazard', label_visibility="visible")
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        top_n = st.slider("Counties to show", 5, 64, 10, key='rank_topn',
                          help="Highest-risk counties for the selected hazard, in descending order.")
    with rank_col2:
        rank_key = {v: k for k, v in HAZARD_NAMES.items()}[rank_hazard]
        col = f'{rank_key}_p'
        top_df = df.nlargest(top_n, col)[['county', col]].copy()
        top_df['pct'] = top_df[col] * 100
        top_df['tier_color'] = top_df[col].apply(risk_color)

        fig = go.Figure(go.Bar(
            x=top_df['pct'][::-1],
            y=top_df['county'][::-1],
            orientation='h',
            marker=dict(color=top_df['tier_color'][::-1]),
            hovertemplate="<b>%{y}</b><br>" + rank_hazard + ": %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=COLORS['card_bg'],
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text_secondary'], family='Inter'),
            xaxis=dict(title=f"{rank_hazard} Risk (%)", gridcolor=COLORS['border'],
                       range=[0, max(top_df['pct'].max() * 1.15, 10)]),
            yaxis=dict(gridcolor=COLORS['border']),
            height=max(320, 28 * top_n),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Detailed County Table")
    detail = df.copy()
    detail['Max Risk'] = (detail['max_p'] * 100).round(1)
    detail['Primary Hazard'] = detail['max_hazard']
    for h in hazards:
        detail[HAZARD_NAMES[h]] = (detail[f'{h}_p'] * 100).round(1)
    detail = detail[['county', 'Primary Hazard', 'Max Risk'] + [HAZARD_NAMES[h] for h in hazards]]
    detail = detail.rename(columns={'county': 'County'}).sort_values('Max Risk', ascending=False)
    st.dataframe(
        detail.style.format({
            'Max Risk': '{:.1f}%',
            **{HAZARD_NAMES[h]: '{:.1f}%' for h in hazards}
        }),
        use_container_width=True, hide_index=True
    )


# =============================================================================
# PAGE: MODEL DIAGNOSTICS
# =============================================================================

def page_model_info():
    st.markdown("## AHI v2.5 — CONUS Model Diagnostics")
    st.caption("9 regional models · 48 states + DC · 3,109 counties")

    st.markdown("""
    ### What is AHI v2.5?

    **AHI v2.5** is the Adaptive Hazard Intelligence model powering this dashboard. It predicts the
    likelihood of four natural hazard types — wildfire, flood, wind, and winter storm — at the
    county level across the contiguous United States.

    **The core problem it solves:** Weather sequences (temperature, wind, precipitation) evolve on a
    fast timescale (days), while spatial correlations (fire spread, downstream flooding, storm tracks)
    operate on a slow timescale (weeks/seasons). AHI uses a proprietary multi-mesh architecture to
    decompose these timescales and learn hazard-specific patterns from 25 years of historical data.

    **v2.5** introduces a learned seasonal bias module that allows the model to independently discover
    regional seasonal structure from data, enabling scalable deployment across all CONUS states without
    manual per-state configuration.
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "AHI v2.5")
    col2.metric("Parameters", "~1.3M per region")
    col3.metric("Architecture", "Proprietary")
    col4.metric("Status", "Online")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Regional Models", "9")
    col6.metric("States + DC", "48 + DC")
    col7.metric("CONUS Counties", "3,109")
    col8.metric("Reference AUC (CO)", "0.883")

    # ---- Regional Model Overview ----
    st.markdown("---")
    st.markdown("### Regional Model Deployment")
    st.markdown("""
    AHI deploys **9 regional models**, each trained on states with similar climate, geography, and
    hazard profiles. Every state within a region shares the same model weights but receives
    **per-state calibration** to match local historical hazard frequencies.
    """)

    _region_info = {
        'colorado':        {'states': ['CO'], 'desc': 'Elevation gradient benchmark (64 counties)'},
        'great_lakes':     {'states': ['IL', 'IN', 'KY', 'MI', 'OH', 'TN', 'WV'], 'desc': 'Lake-effect weather, tornado alley fringe, Ohio River flooding'},
        'mountain_west':   {'states': ['AZ', 'ID', 'MT', 'NM', 'NV', 'UT', 'WY'], 'desc': 'Arid/semi-arid fire, monsoon, mountain winter storms'},
        'northeast':       {'states': ['CT', 'DC', 'DE', 'MA', 'MD', 'ME', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VA', 'VT'], 'desc': "Nor'easters, coastal flooding, ice storms"},
        'northern_plains': {'states': ['IA', 'MN', 'MO', 'ND', 'SD', 'WI'], 'desc': 'Blizzards, prairie fire, spring flooding'},
        'pacific':         {'states': ['CA'], 'desc': 'Wildfire, atmospheric rivers, coastal flooding'},
        'pnw':             {'states': ['OR', 'WA'], 'desc': 'Atmospheric rivers, Cascadia subduction, PNW wildfire'},
        'southeast_gulf':  {'states': ['AL', 'AR', 'FL', 'GA', 'LA', 'MS', 'NC', 'SC'], 'desc': 'Hurricanes, Gulf moisture, severe convective storms'},
        'southern_plains': {'states': ['KS', 'NE', 'OK', 'TX'], 'desc': 'Tornado alley, prairie fire, flash flooding'},
    }
    region_rows = []
    for region, info in _region_info.items():
        region_rows.append({
            'Region': region.replace('_', ' ').title(),
            'States': len(info['states']),
            'Coverage': ', '.join(info['states']),
            'Hazard Profile': info['desc'],
        })
    st.dataframe(pd.DataFrame(region_rows), use_container_width=True, hide_index=True)

    st.markdown("### Model Capabilities")
    st.markdown("""
    | Capability | Description |
    |-----------|-------------|
    | **Multi-modal input** | Ingests weather, geography, and land cover features into a unified representation |
    | **Temporal learning** | Learns hazard-specific memory horizons from historical sequences |
    | **Spatial awareness** | Captures cross-county correlations (fire spread, downstream flooding, storm tracks) |
    | **Hazard specialization** | Per-hazard adaptation without duplicating the full model |
    | **Cross-hazard modeling** | Models physical dependencies between correlated hazard types |
    | **Calibrated output** | Per-hazard prediction heads with post-hoc calibration per state |
    """)

    # ---- Reference Performance: Colorado ----
    st.markdown("---")
    st.markdown("### Reference Performance — Colorado (Held-out Test Set)")
    st.caption("Colorado serves as the primary benchmark: 64 counties with a clear elevation gradient.")

    co_data = [
        {"Hazard": "Winter",  "AUC": 0.963, "Quality": "Excellent", "Notes": "Best performer — strong elevation-driven seasonal signal."},
        {"Hazard": "Flood",   "AUC": 0.891, "Quality": "Excellent", "Notes": "Bimodal seasonal pattern (snowmelt + monsoon) well-captured."},
        {"Hazard": "Fire",    "AUC": 0.857, "Quality": "Excellent", "Notes": "Western Slope + foothills fire patterns learned well."},
        {"Hazard": "Wind",    "AUC": 0.817, "Quality": "Excellent", "Notes": "Chinook corridor captured; diffuse plains wind harder to localize."},
    ]
    st.dataframe(pd.DataFrame(co_data), use_container_width=True, hide_index=True)

    hazards_co   = ["Winter", "Flood", "Fire", "Wind"]
    aucs_co      = [0.963, 0.891, 0.857, 0.817]
    bar_colors   = [COLORS['winter'], COLORS['flood'], COLORS['fire'], COLORS['wind']]

    # WA reference performance
    hazards_wa   = ["Winter", "Fire", "Wind", "Flood"]
    aucs_wa      = [0.908, 0.851, 0.837, 0.830]

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(x=hazards_co, y=aucs_co, marker_color=bar_colors,
                               name="Colorado (reference)", opacity=0.9))
    fig_perf.add_trace(go.Bar(x=hazards_wa, y=aucs_wa, marker_color=bar_colors,
                               name="Washington (PNW)", opacity=0.5))
    fig_perf.add_hline(y=0.8, line_dash="dash", line_color="#6b9e7a", annotation_text="Excellent (0.8)")
    fig_perf.add_hline(y=0.5, line_dash="dash", line_color="#dc2626", annotation_text="Random (0.5)")
    fig_perf.update_layout(
        title="AHI v2.5 — AUC by Hazard Type (validated regions)",
        barmode='group',
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(color=COLORS['text_secondary'], family='Inter'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color=COLORS['text_secondary'])),
        xaxis=dict(gridcolor=COLORS['border']),
        yaxis=dict(title="AUC-ROC", range=[0, 1], gridcolor=COLORS['border']),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    mc1, mc2 = st.columns(2)
    mc1.success("**Colorado Mean AUC: 0.882** — All 4 hazards Excellent (AUC > 0.8)")
    mc2.info("**Washington Mean AUC: 0.857** — All 4 hazards Excellent")

    # ---- Calibration ----
    st.markdown("---")
    st.markdown("### Calibration Pipeline")
    st.markdown("""
    **Calibration** means predicted probabilities match real-world frequencies. If the model says 10% fire risk,
    fires should occur roughly 10% of the time in those conditions. AHI v2.5 uses a **per-state calibration
    pipeline** to ensure locally meaningful predictions:

    - **Temperature scaling** — Per-hazard confidence adjustment fitted on each state's validation set
    - **Seasonal bias** — Regional climatology adjustments (fire season, hurricane season, etc.)
    - **Severity-weighted base rates** — County-level rates weighted by event magnitude (wind speed, fire acreage, winter event type)
    - **Base-rate ceilings** — Caps predictions at historical plausibility limits

    Each state receives its own calibration parameters, ensuring predictions are locally meaningful.
    """)

    st.markdown("---")
    st.markdown("### Updates & Roadmap")

    st.markdown("**Current (AHI v2.5 — CONUS Deployment)**")
    st.markdown("""
    - 9 regional models serving 48 states + DC (3,109 counties)
    - Proprietary multi-mesh architecture with per-state calibration
    - Severity-weighted calibration for fire (acreage), wind (gust speed), and winter (event type)
    - Relative risk tiers contextualize predictions against historical base rates per state
    - Precomputed national predictions for instant page load
    """)

    st.markdown("**Planned / Future Work**")
    st.markdown("""
    - Integrate real-time weather feeds (NWS/NOAA APIs) for operational nowcasts
    - Uncertainty quantification for prediction intervals
    - Alaska and Hawaii coverage (non-CONUS)
    - Continual learning pipeline with scheduled model retraining
    """)

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    | Source | Dataset | Usage |
    |--------|---------|-------|
    | **NOAA Storm Events** | Historical storm records across all CONUS states (2000–2025) | Flood, wind, winter storm labels (strict county + 3-day window matching) |
    | **WFIGS** | Wildland Fire Locations Full History (all CONUS) | Wildfire labels (geocoded to county boundaries) |
    | **USGS Earthquakes** | National seismic catalog (M ≥ 2.0, 2000–2025) | Seismic event labels (model trained; dashboard display coming soon) |
    | **FEMA** | Disaster declarations (all states) | Supplementary validation labels |
    | **GridMET** | Daily gridded weather — CONUS (lat 25–49, lon –125 to –67) | Temperature, precipitation, humidity, wind speed, fire weather (ERC) |
    | **US Census (TIGER)** | County-level population density | Static demographic feature for exposure weighting |
    | **NLCD / Land Cover** | Forest & urban fractions, elevation proxy by terrain class | Static geographic features for terrain-aware inference |
    """)

    st.caption(
        "Note: AHI v2.5 uses population density as its only demographic feature. "
        "CDC Social Vulnerability Index (SVI) data was evaluated but not incorporated into the "
        "training pipeline; it is reserved for future fairness/equity analysis."
    )


# =============================================================================
# PAGE: NATIONAL (CONUS overview — first/default tab)
# =============================================================================

# Tile-server presets for the national choropleth basemap selector.
# Esri free / USGS free / built-in styles. No tokens required.
_NATIONAL_TILE_STYLES = {
    'Satellite':  {
        'mapbox_style': 'white-bg',
        'mapbox_layers': [{
            'below': 'traces', 'sourcetype': 'raster',
            'sourceattribution':
                'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, '
                'USDA FSA, USGS, AeroGRID, IGN',
            'source': ['https://services.arcgisonline.com/ArcGIS/rest/services/'
                       'World_Imagery/MapServer/tile/{z}/{y}/{x}'],
        }],
        'border': '#fbbf24', 'opacity': 0.65,
    },
    'Dark':       {'mapbox_style': 'carto-darkmatter', 'mapbox_layers': [],
                   'border': '#30363d', 'opacity': 0.85},
    'Light':      {'mapbox_style': 'carto-positron',   'mapbox_layers': [],
                   'border': '#888',    'opacity': 0.85},
    # USGS satellite — kept in reserve; uncomment if Esri throttles
    # 'Satellite (USGS)':  {
    #     'mapbox_style': 'white-bg',
    #     'mapbox_layers': [{
    #         'below': 'traces', 'sourcetype': 'raster',
    #         'sourceattribution':
    #             'Tiles &copy; U.S. Geological Survey — National Map',
    #         'source': ['https://basemap.nationalmap.gov/arcgis/rest/services/'
    #                    'USGSImageryOnly/MapServer/tile/{z}/{y}/{x}'],
    #     }],
    #     'border': '#fbbf24', 'opacity': 0.65,
    # },
}


def _county_display_name(name: str) -> str:
    """Return full display name: 'King' → 'King County', but 'Iberia Parish' stays as-is."""
    for suffix in (' Parish', ' City', ' Borough', ' Census Area', ' Municipality'):
        if name.endswith(suffix):
            return name
    return f'{name} County'


def render_national_choropleth(df: pd.DataFrame, geojson: dict, hazard: str,
                                map_style: str = 'Dark', height: int = 620):
    """Plotly choropleth of CONUS counties colored by selected hazard."""
    col = f'{hazard}_p' if hazard != 'max' else 'max_p'
    df = df.copy()
    df['_id'] = df['state'] + '|' + df['county_id']
    df['pct'] = df[col] * 100
    df['_display'] = df['county'].apply(_county_display_name)

    style = _NATIONAL_TILE_STYLES.get(map_style, _NATIONAL_TILE_STYLES['Dark'])

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=df['_id'],
        z=df['pct'],
        featureidkey='properties._id',
        colorscale=[
            [0.00, RISK_TIERS[0][2]], [0.10, RISK_TIERS[1][2]],
            [0.20, RISK_TIERS[2][2]], [0.35, RISK_TIERS[3][2]],
            [0.50, RISK_TIERS[4][2]], [1.00, RISK_TIERS[4][2]],
        ],
        zmin=0, zmax=50,
        marker_line_width=0.4,
        marker_line_color=style['border'],
        marker_opacity=style['opacity'],
        customdata=df[['state', '_display']].values,
        hovertemplate=('<b>%{customdata[1]}, %{customdata[0]}</b><br>' +
                        f'{HAZARD_NAMES.get(hazard, hazard.title())}: ' +
                        '%{z:.1f}%<extra></extra>'),
        showscale=False,  # colorbar commented out — felt out of place in executive layout
    ))
    fig.update_layout(
        mapbox_style=style['mapbox_style'],
        mapbox_layers=style['mapbox_layers'],
        mapbox_zoom=3.2,
        mapbox_center={'lat': 39.5, 'lon': -98.5},
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor=COLORS['app_bg'],
    )
    return fig


def _render_hazard_bars_vertical(row):
    """Vertical hazard bars matching the executive mockup layout."""
    html = ""
    for h in DISPLAY_HAZARDS:
        p = row[f'{h}_p'] * 100
        bar_pct = min(p, 100.0)
        html += (
            f"<div style='margin-bottom:10px;'>"
            f"<div style='display:flex; justify-content:space-between; "
            f"color:{COLORS['text_primary']}; font-size:0.95em;'>"
            f"<span>{HAZARD_NAMES[h]}</span>"
            f"<span style='font-weight:600;'>{p:.1f}%</span>"
            f"</div>"
            f"<div style='height:6px; background:{COLORS['border']}; "
            f"border-radius:3px; overflow:hidden; margin-top:4px;'>"
            f"<div style='width:{bar_pct}%; height:100%; "
            f"background:{COLORS[h]};'></div></div></div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _render_weather_drivers(wx_data):
    """Weather driver cards matching the executive mockup layout.
    GridMET units: tmmx/tmmn in Kelvin, rmin in %, vs in m/s, pr in mm,
    erc dimensionless, vpd in kPa."""
    def _k_to_f(v):
        return (v - 273.15) * 9.0 / 5.0 + 32.0 if v is not None and v > 100 else v

    drivers = [
        ('ERC',        wx_data.get('erc'),  '(energy release)'),
        ('Wind Speed', wx_data.get('vs'),   'm/s'),
        ('Min RH',     wx_data.get('rmin'), '%'),
        ('Precip',     wx_data.get('pr'),   'mm'),
        ('Max Temp',   _k_to_f(wx_data.get('tmmx')), '°F'),
        ('Min Temp',   _k_to_f(wx_data.get('tmmn')), '°F'),
        ('VPD',        wx_data.get('vpd'),  'kPa'),
    ]
    drivers = [(l, v, u) for l, v, u in drivers if v is not None]
    if not drivers:
        return

    st.markdown(
        f"<div style='color:{COLORS['text_secondary']}; font-size:0.85em; "
        f"margin-top:8px;'>**Weather drivers (sample for this month):**</div>",
        unsafe_allow_html=True)
    cols = st.columns(min(len(drivers), 4))
    for i, (label, val, unit) in enumerate(drivers):
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:{COLORS['card_bg']}; padding:8px 10px; "
                f"border-radius:4px; margin-bottom:6px;'>"
                f"<div style='color:{COLORS['text_tertiary']}; font-size:0.7em; "
                f"text-transform:uppercase;'>{label}</div>"
                f"<div style='color:{COLORS['text_primary']}; font-size:1.05em; "
                f"font-weight:500;'>{val:.1f}<span style='color:"
                f"{COLORS['text_tertiary']}; font-size:0.75em;'> {unit}</span></div>"
                f"</div>",
                unsafe_allow_html=True)


def page_national():
    st.caption("3,109 CONUS counties · Click anywhere on the map to drill into "
               "county detail without leaving the page.")

    geojson = load_national_geojson()
    if geojson is None:
        st.error(
            "National geojson not found at `data/national_counties.geojson`. "
            "Build it once from the platform's per-state geojsons (see "
            "scripts/build_national_geojson.py in the mockup folder)."
        )
        return

    now = datetime.now().date()
    cur_month = now.month
    month_label = now.strftime('%B %Y')

    # ---- Top controls ----
    c1, c2 = st.columns([1.5, 1.5])
    with c1:
        hazard = st.selectbox(
            "Hazard layer",
            options=['max'] + DISPLAY_HAZARDS,
            index=0,
            format_func=lambda h: ('Max risk (any hazard)' if h == 'max'
                                     else HAZARD_NAMES.get(h, h.title())),
            key='national_hazard',
        )
    with c2:
        map_style = st.selectbox(
            "Map style",
            options=list(_NATIONAL_TILE_STYLES.keys()),
            index=0,
            key='national_map_style',
        )

    df = predict_all_national(cur_month)
    if df is None or len(df) == 0:
        st.error("No precomputed predictions found for this month. "
                 "Run `python scripts/precompute_national.py` to generate them.")
        return

    st.caption(f"**{month_label}** · {len(df):,} counties · "
               f"Predictions calibrated from historical patterns for {now.strftime('%B')}. "
               f"Drill into the **State** tab for county-level detail.")

    # ---- Extract selection BEFORE rendering columns ----
    sel_id = None
    if 'national_map' in st.session_state:
        sel = st.session_state.get('national_map')
        if sel and isinstance(sel, dict) and 'selection' in sel:
            pts = sel['selection'].get('points', [])
            if pts:
                sel_id = pts[0].get('location')

    # ---- Executive layout: map (left) + detail panel (right) ----
    map_col, detail_col = st.columns([3, 2])

    with map_col:
        fig = render_national_choropleth(df, geojson, hazard, map_style=map_style,
                                          height=560)
        st.plotly_chart(fig, use_container_width=True,
                        on_select="rerun", selection_mode='points',
                        key='national_map')

    with detail_col:
        # Re-check selection after map render (on_select triggers rerun)
        if sel_id is None and 'national_map' in st.session_state:
            sel = st.session_state.get('national_map')
            if sel and isinstance(sel, dict) and 'selection' in sel:
                pts = sel['selection'].get('points', [])
                if pts:
                    sel_id = pts[0].get('location')

        if sel_id:
            match = df[df['state'] + '|' + df['county_id'] == sel_id]
            if len(match) > 0:
                row = match.iloc[0]
                primary = row['max_hazard']
                primary_pct = row['max_p'] * 100
                level, _ = risk_level(row['max_p'])
                pcolor = COLORS.get(primary, COLORS['primary_light'])

                display_name = _county_display_name(row['county'].title())
                st.markdown(
                    f"<h2 style='margin:0 0 4px 0; color:{COLORS['text_primary']};'>"
                    f"{display_name}, {row['state']}</h2>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:{COLORS['text_secondary']}; margin-bottom:16px;'>"
                    f"Primary: <strong style='color:{pcolor};'>"
                    f"{HAZARD_NAMES.get(primary, primary.title())}</strong> · "
                    f"{primary_pct:.1f}% · {level}</div>",
                    unsafe_allow_html=True)

                st.markdown("---")

                # Hazard bars (vertical layout from mockup)
                _render_hazard_bars_vertical(row)

                # Weather drivers from parquet (if available)
                weather_key = f'_weather_{row["state"]}_{row["county"]}'
                if weather_key not in st.session_state:
                    try:
                        hdf = load_hazard_data(row['state'])
                        if hdf is not None:
                            county_upper = row['county'].upper().strip()
                            cmask = (hdf['county'].str.upper()
                                     .str.replace(' COUNTY', '', regex=False)
                                     .str.strip() == county_upper)
                            crows = hdf[cmask]
                            if len(crows) > 0 and 'month' in crows.columns:
                                sm = crows[crows['month'] == cur_month]
                                wx = sm.iloc[0] if len(sm) > 0 else crows.iloc[0]
                                st.session_state[weather_key] = {
                                    'erc':  float(wx.get('erc', 0)),
                                    'vs':   float(wx.get('vs', 0)),
                                    'rmin': float(wx.get('rmin', 0)),
                                    'pr':   float(wx.get('pr', 0)),
                                    'tmmx': float(wx.get('tmmx', 0)),
                                    'tmmn': float(wx.get('tmmn', 0)),
                                    'vpd':  float(wx.get('vpd', 0)),
                                }
                            else:
                                st.session_state[weather_key] = None
                        else:
                            st.session_state[weather_key] = None
                    except Exception:
                        st.session_state[weather_key] = None

                wx = st.session_state.get(weather_key)
                if wx:
                    _render_weather_drivers(wx)

                st.caption(
                    f"Click the **State** tab and pick **{row['state']}** to "
                    f"drill into county-level audit reports and hazard guidance.")
            else:
                st.info(f"County not found in predictions.")
        else:
            st.markdown(
                f"<div style='display:flex; align-items:center; justify-content:center; "
                f"height:400px; color:{COLORS['text_tertiary']}; "
                f"border:1px dashed {COLORS['border']}; border-radius:8px; "
                f"text-align:center; padding:24px;'>"
                f"<div>Click any county on the map<br>to see hazard breakdown</div></div>",
                unsafe_allow_html=True)

    # ---- National stats footer ----
    severe   = int((df['max_p'] >= 0.50).sum())
    high     = int(((df['max_p'] >= 0.35) & (df['max_p'] < 0.50)).sum())
    moderate = int(((df['max_p'] >= 0.20) & (df['max_p'] < 0.35)).sum())
    elevated = int(((df['max_p'] >= 0.10) & (df['max_p'] < 0.20)).sum())
    low      = int((df['max_p'] < 0.10).sum())
    cols = st.columns(5)
    cols[0].metric("Severe (>50%)",     severe)
    cols[1].metric("High (35–50%)",     high)
    cols[2].metric("Moderate (20–35%)", moderate)
    cols[3].metric("Elevated (10–20%)", elevated)
    cols[4].metric("Low (<10%)",        low)


# =============================================================================
# PAGE: ABOUT
# =============================================================================

def page_about():
    st.markdown("## About AHI")

    st.markdown(f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: {COLORS['primary_light']}; margin-top: 0;">Adaptive Hazard Intelligence</h3>
        <p style="color: {COLORS['text_primary']}; line-height: 1.7;">
        AHI is a calibrated, multi-hazard risk prediction system deployed across the contiguous United States.
        It predicts the likelihood of four natural hazard types — wildfire, flood, wind, and winter storm
        — at the county level using a proprietary deep learning architecture trained on 25 years of
        historical data. AHI currently covers <strong>3,109 counties</strong> across <strong>48 states and DC</strong>
        through 9 regional models with per-state calibration.
        </p>
        <p style="color: {COLORS['text_secondary']}; line-height: 1.7;">
        AHI is developed by Resilience Analytics Lab, LLC. The platform is being prepared for
        DHS SBIR Phase I to integrate real-time NWS/NOAA weather feeds for operational nowcasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Published Research")
    st.markdown("""
    - Curry, J.D. (2025). *Heat Kernel Attention: Diffusion-Based Attention for Transformer Architectures.* SSRN 5959898.
    - Curry, J.D. (2026). *Simplicial Computation: Topology as Control in Heterogeneous Attention.* SSRN 6037977.
    """)

    st.markdown("### Key Capabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Multi-Hazard Prediction**
        - Five hazard types from a single unified architecture
        - 25 years of historical training data (2000–2025)
        - Severity-weighted calibration reflects actual event impact

        **Regional Model Strategy**
        - 9 climate-coherent regions serving 48 states + DC
        - Region-specific weights with per-state calibration
        - Locally meaningful predictions for every county
        """)
    with col2:
        st.markdown("""
        **Per-State Calibration**
        - Per-hazard confidence adjustment for each state
        - Severity-weighted base rates (wind speed, fire acreage, winter event type)
        - Historical plausibility ceilings prevent overconfident predictions

        **Clean Label Engineering**
        - Strict county-level geographic matching across all CONUS
        - Multiple source cross-validation (NOAA, WFIGS, USGS, FEMA)
        - Severity normalization: routine events weighted differently than catastrophic ones
        """)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {COLORS['text_tertiary']}; padding: 20px;">
        <div style="font-weight: 600; color: {COLORS['text_secondary']};">Resilience Analytics Lab, LLC</div>
        <div style="margin-top: 4px;"><a href="https://www.resilienceanalyticslab.com/" target="_blank" style="color: {COLORS['accent']}; text-decoration: none;">www.resilienceanalyticslab.com</a></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_css()

    logo_b64 = get_logo_base64()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 40px; height: 40px; flex-shrink: 0; margin-top: 2px;">' if logo_b64 else ""

    st.markdown(f"""
    <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;">
        {logo_html}
        <div class="ahi-header-text">
            <h2 class="title">Adaptive Hazard Intelligence</h2>
            <div class="subtitle">Calibrated hazard risk for defensible decisions · 3,109 CONUS counties · AHI v2.5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_national, tab_state, tab_county, tab_diag, tab_about = st.tabs([
        "National",
        "State",
        "County Risk Assessment",
        "Model Diagnostics",
        "About"
    ])

    with tab_national:
        page_national()
    with tab_state:
        page_state_overview()
    with tab_county:
        page_quick_predict()
    with tab_diag:
        page_model_info()
    with tab_about:
        page_about()


if __name__ == '__main__':
    main()
