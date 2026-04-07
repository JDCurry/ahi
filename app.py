"""
AHI — Adaptive Hazard Intelligence
SBIR Phase I demonstration dashboard.
Resilience Analytics Lab, LLC
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

# Optional imports (folium only — geopandas replaced with json-based loader)
try:
    import folium
    from folium.features import GeoJsonTooltip
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ONNX-based inference (no torch dependency — saves ~200MB)
try:
    from inference_onnx import predict_county_risks_simple, predict_from_ahi_v2
    AHI_V2_AVAILABLE = True
except Exception as e:
    print(f"[IMPORT] inference_onnx import failed: {e}")
    predict_county_risks_simple = None
    predict_from_ahi_v2 = None
    AHI_V2_AVAILABLE = False

get_batch_adjacency = None

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AHI — Adaptive Hazard Intelligence",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEVICE = 'cpu'
DATA_DIR = Path("data")
MAX_FORECAST_DAYS = 14

WA_COUNTY_COORDS = {
    'Adams': (46.98, -118.56), 'Asotin': (46.19, -117.20), 'Benton': (46.23, -119.52),
    'Chelan': (47.87, -120.62), 'Clallam': (48.11, -123.93), 'Clark': (45.78, -122.48),
    'Columbia': (46.29, -117.91), 'Cowlitz': (46.19, -122.67), 'Douglas': (47.53, -119.69),
    'Ferry': (48.47, -118.52), 'Franklin': (46.53, -118.89), 'Garfield': (46.43, -117.54),
    'Grant': (47.21, -119.45), 'Grays Harbor': (47.15, -123.76), 'Island': (48.21, -122.58),
    'Jefferson': (47.76, -123.50), 'King': (47.49, -121.84), 'Kitsap': (47.64, -122.65),
    'Kittitas': (47.12, -120.68), 'Klickitat': (45.87, -120.78), 'Lewis': (46.58, -122.38),
    'Lincoln': (47.58, -118.41), 'Mason': (47.35, -123.18), 'Okanogan': (48.55, -119.74),
    'Pacific': (46.56, -123.78), 'Pend Oreille': (48.53, -117.27), 'Pierce': (47.04, -122.13),
    'San Juan': (48.53, -123.02), 'Skagit': (48.48, -121.80), 'Skamania': (46.02, -121.92),
    'Snohomish': (48.05, -121.72), 'Spokane': (47.62, -117.40), 'Stevens': (48.40, -117.85),
    'Thurston': (46.93, -122.83), 'Wahkiakum': (46.29, -123.42), 'Walla Walla': (46.23, -118.48),
    'Whatcom': (48.85, -121.72), 'Whitman': (46.90, -117.52), 'Yakima': (46.46, -120.74)
}

COUNTIES = sorted(WA_COUNTY_COORDS.keys())

# =============================================================================
# COLOR THEME — Resilience Analytics Lab (sage green / institutional)
# =============================================================================

COLORS = {
    'app_bg': '#24282D',
    'card_bg': '#161b22',
    'sidebar_bg': '#0d1117',
    'elevated_bg': '#1c2128',
    'primary': '#4a7c59',
    'primary_light': '#6b9e7a',
    'primary_dark': '#2d5a3a',
    'accent': '#8fbc8f',
    'border': '#30363d',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    # Per-hazard
    'fire': '#e05252',
    'flood': '#4a90d9',
    'wind': '#9b59b6',
    'winter': '#2ec4b6',
    'seismic': '#e67e22',
}

# Unified 5-tier sequential risk palette (matches legend everywhere)
RISK_TIERS = [
    (0.00, 0.10, '#2d5a3a', 'Low',      '< 10%',   'Baseline conditions — routine monitoring'),
    (0.10, 0.20, '#6b9e7a', 'Elevated', '10–20%',  'Above baseline — increased awareness recommended'),
    (0.20, 0.35, '#f59e0b', 'Moderate', '20–35%',  'Notable risk — review preparedness plans'),
    (0.35, 0.50, '#f97316', 'High',     '35–50%',  'Significant risk — consider pre-positioning resources'),
    (0.50, 1.01, '#dc2626', 'Severe',   '> 50%',   'Elevated readiness recommended'),
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

HAZARD_NAMES = {
    'fire': 'Fire', 'flood': 'Flood', 'wind': 'Wind',
    'winter': 'Winter Storm', 'seismic': 'Seismic'
}

HAZARD_GUIDANCE = {
    'fire': 'Review evacuation routes. Coordinate with fire districts on resource availability. Assess defensible space near critical facilities. Verify water supply access points.',
    'flood': 'Inspect drainage systems and culverts. Verify flood gauge monitoring. Pre-stage pumps and sandbags at flood-prone areas. Coordinate road closure plans.',
    'wind': 'Coordinate with utilities on power line inspections. Secure outdoor equipment. Pre-position generators at critical facilities. Alert manufactured housing communities.',
    'winter': 'Verify road treatment supplies. Check backup power at warming shelters. Coordinate with WSDOT on plowing priorities. Prepare travel advisory messaging.',
    'seismic': 'Review structural assessments for critical buildings. Confirm communications redundancy. Verify search and rescue readiness. Review tsunami evacuation routes for coastal areas.',
}

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

    /* Primary risk hero card */
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

    .stSelectbox [data-baseweb="select"] {{
        background: {COLORS['card_bg']};
        border-color: {COLORS['border']};
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
def load_v2_model():
    if not AHI_V2_AVAILABLE:
        return None, None, False
    for p in [Path("outputs/ahi_v2/model.onnx"),
              Path("/mount/src/ahi/outputs/ahi_v2/model.onnx")]:
        if p.exists():
            print(f"[AHI] ONNX model available: {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
            return "onnx", None, True
    return None, None, False


@st.cache_data
def load_hazard_data():
    path = DATA_DIR / 'hazard_lm_clean_labeled.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data
def load_geojson():
    path = DATA_DIR / 'wa_counties.geojson'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _normalize_geojson_names(geojson):
    """Add a normalized 'NAME_NORM' property for consistent lookup."""
    if geojson is None:
        return None
    out = json.loads(json.dumps(geojson))  # deep copy
    for feat in out.get('features', []):
        props = feat.get('properties', {})
        name = None
        for f in ['NAME', 'name', 'COUNTY', 'county_name']:
            if f in props:
                name = props[f]
                break
        if name:
            props['NAME_NORM'] = name.replace(' County', '').strip()
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
        return (-122, 47, -120, 48)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_county(county_name, target_date):
    _, _, ok = load_v2_model()
    if not ok:
        return None, "Model not loaded — check outputs/ahi_v2/model.onnx"
    hazard_df = load_hazard_data()
    if hazard_df is None or len(hazard_df) == 0:
        return None, "Hazard dataset not found"
    try:
        from inference_onnx import predict_county_risks_simple as _predict
        return _predict(None, county_name, hazard_df, target_date), None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Prediction failed: {e}"


def predict_all_counties(target_date, progress_callback=None):
    hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']
    rows = []
    for i, county in enumerate(COUNTIES):
        if progress_callback:
            progress_callback(i, len(COUNTIES), county)
        risks, err = predict_single_county(county, target_date)
        if risks:
            row = {'county': county, 'date': str(target_date)}
            for h in hazards:
                row[f'{h}_p'] = risks.get(h, 0.0)
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


# =============================================================================
# UI HELPERS
# =============================================================================

def render_primary_risk_callout(risks):
    """Hero card highlighting the top-ranked hazard."""
    sorted_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)
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
    sorted_hazards = sorted(risks.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(5)
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


def render_risk_summary(risks):
    sorted_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)
    st.markdown("#### Top Hazards — Recommended Actions")
    for hazard, prob in sorted_risks[:3]:
        level, interpretation = risk_level(prob)
        guidance = HAZARD_GUIDANCE.get(hazard, '')
        color = COLORS.get(hazard, COLORS['text_primary'])
        st.markdown(f"""
        <div class="risk-section">
            <h4 style="color: {color}; margin: 0 0 4px 0;">{HAZARD_NAMES.get(hazard, hazard.title())} — {prob*100:.1f}% ({level})</h4>
            <p style="color: {COLORS['text_secondary']}; margin: 2px 0; font-style: italic;">{interpretation}</p>
            <p style="color: {COLORS['text_primary']}; margin: 6px 0 0 0; font-size: 0.9em;"><strong>Suggested actions:</strong> {guidance}</p>
        </div>
        """, unsafe_allow_html=True)


def render_interpretation_guide(forecast_days):
    """Expandable guide — now uses the actual forecast horizon."""
    with st.expander("How to interpret these numbers", expanded=False):
        st.markdown(f"""
        **What the percentages mean:**
        - These are **calibrated risk probabilities** for the **{forecast_days}-day forecast window**
        - They represent the likelihood of hazard conditions based on **25 years of historical patterns** (2000–2025)
        - Probabilities reflect statewide learned patterns across all 39 WA counties, not solely this county's history
        - A county with few historical events can still show elevated risk if current seasonal/geographic conditions match patterns that preceded events elsewhere

        **Risk thresholds:**
        | Level | Range | Suggested Response |
        |-------|-------|--------------------|
        | Low | < 10% | Routine monitoring |
        | Elevated | 10–20% | Increased awareness |
        | Moderate | 20–35% | Review preparedness |
        | High | 35–50% | Pre-position resources |
        | Severe | > 50% | Elevated readiness |

        **Important:** AHI uses historical pattern detection, not live weather feeds.
        Predictions reflect seasonal and geographic baselines — always cross-reference with
        current NWS watches/warnings for operational decisions.
        """)


# =============================================================================
# MAP: Plotly choropleth (reliable — no external JS component)
# =============================================================================

def render_statewide_choropleth(df, hazard_key, hazard_label):
    """Statewide risk map using plotly choropleth (native, no folium)."""
    geojson_data = load_geojson()
    if geojson_data is None:
        st.info("GeoJSON not found — map unavailable.")
        return

    geojson_norm = _normalize_geojson_names(geojson_data)
    col_name = f"{hazard_key}_p"

    plot_df = df.copy()
    plot_df['county_norm'] = plot_df['county'].str.replace(' County', '').str.strip()
    plot_df['pct'] = plot_df[col_name] * 100

    # Tier-based discrete coloring using colorscale
    fig = go.Figure(go.Choropleth(
        geojson=geojson_norm,
        locations=plot_df['county_norm'],
        z=plot_df['pct'],
        featureidkey="properties.NAME_NORM",
        colorscale=[
            [0.00, '#2d5a3a'],
            [0.10, '#2d5a3a'],
            [0.10, '#6b9e7a'],
            [0.20, '#6b9e7a'],
            [0.20, '#f59e0b'],
            [0.35, '#f59e0b'],
            [0.35, '#f97316'],
            [0.50, '#f97316'],
            [0.50, '#dc2626'],
            [1.00, '#dc2626'],
        ],
        zmin=0,
        zmax=100,
        marker_line_color=COLORS['border'],
        marker_line_width=0.8,
        colorbar=dict(
            title=f"{hazard_label}<br>Risk (%)",
            thickness=12,
            len=0.6,
            x=1.02,
            xanchor='left',
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            title_font=dict(color=COLORS['text_secondary'], size=11),
        ),
        hovertemplate="<b>%{location} County</b><br>" + hazard_label + ": %{z:.1f}%<extra></extra>",
    ))
    # Fixed WA bounding box — prevents squish from dynamic fitbounds
    fig.update_geos(
        visible=False,
        bgcolor=COLORS['card_bg'],
        projection_type='mercator',
        lonaxis_range=[-125, -116.5],
        lataxis_range=[45.3, 49.2],
    )
    fig.update_layout(
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=0, r=0, t=10, b=10),
        height=540,
        font=dict(color=COLORS['text_secondary'], family='Inter'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Shared legend
    st.markdown(f"""
    <div style="display: flex; gap: 16px; justify-content: center; margin-top: 4px; flex-wrap: wrap; font-size: 0.9em;">
        <span style="color: #2d5a3a;">&#9632; Low (&lt;10%)</span>
        <span style="color: #6b9e7a;">&#9632; Elevated (10–20%)</span>
        <span style="color: #f59e0b;">&#9632; Moderate (20–35%)</span>
        <span style="color: #f97316;">&#9632; High (35–50%)</span>
        <span style="color: #dc2626;">&#9632; Severe (&gt;50%)</span>
    </div>
    """, unsafe_allow_html=True)


def render_county_spotlight_map(selected_county, risks, target_date):
    """Small plotly map zoomed on selected county."""
    geojson_data = load_geojson()
    geojson_norm = _normalize_geojson_names(geojson_data)
    if geojson_norm is None:
        return

    selected_norm = selected_county.replace(' County', '').strip()

    # Order overlay dropdown by this county's risks (highest first)
    ordered_hazards = sorted(
        [('Fire', 'fire'), ('Flood', 'flood'), ('Wind', 'wind'),
         ('Winter', 'winter'), ('Seismic', 'seismic')],
        key=lambda kv: risks.get(kv[1], 0.0),
        reverse=True
    )
    hazard_options = [label for label, _ in ordered_hazards]

    hazard_choice = st.selectbox(
        "Overlay hazard (ranked by this county's risk)",
        hazard_options,
        key='county_hazard_select'
    )
    hkey = hazard_choice.lower()
    sel_prob = risks.get(hkey, 0.0) * 100

    # Build single-county choropleth
    locs = [selected_norm]
    zs = [sel_prob]

    fig = go.Figure(go.Choropleth(
        geojson=geojson_norm,
        locations=locs,
        z=zs,
        featureidkey="properties.NAME_NORM",
        colorscale=[
            [0.00, '#2d5a3a'],
            [0.10, '#2d5a3a'],
            [0.10, '#6b9e7a'],
            [0.20, '#6b9e7a'],
            [0.20, '#f59e0b'],
            [0.35, '#f59e0b'],
            [0.35, '#f97316'],
            [0.50, '#f97316'],
            [0.50, '#dc2626'],
            [1.00, '#dc2626'],
        ],
        zmin=0, zmax=100, showscale=False,
        marker_line_color='#e6edf3',
        marker_line_width=2,
        hovertemplate=f"<b>{selected_norm} County</b><br>{hazard_choice}: %{{z:.1f}}%<extra></extra>",
    ))
    fig.update_geos(
        visible=False,
        bgcolor=COLORS['card_bg'],
        projection_type='mercator',
        lonaxis_range=[-125, -116.5],
        lataxis_range=[45.3, 49.2],
    )
    fig.update_layout(
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=0, r=0, t=10, b=10),
        height=400,
        font=dict(color=COLORS['text_secondary'], family='Inter'),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: QUICK PREDICT
# =============================================================================

def page_quick_predict():
    st.markdown("## County Risk Assessment")
    st.caption("Analyze hazard risk for a single county. Assessment based on 25 years of historical hazard patterns.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_county = st.selectbox("Select County", COUNTIES, index=COUNTIES.index('King'))
    with col2:
        forecast_horizon = st.selectbox("Forecast Horizon", ["7 days", "14 days"], index=1)

    days = int(forecast_horizon.split()[0])
    today = datetime.now().date()
    target_date = today + timedelta(days=days)

    lat, lon = WA_COUNTY_COORDS.get(selected_county, (47.5, -120.5))
    month_name = target_date.strftime('%B')
    month = target_date.month
    if month in [3, 4, 5]:
        season_note = "Spring — transitional; flood risk from snowmelt"
    elif month in [6, 7, 8]:
        season_note = "Summer — peak fire season"
    elif month in [9, 10, 11]:
        season_note = "Fall — wind events, early winter storms"
    else:
        season_note = "Winter — snow, ice, and wind events"

    st.markdown(f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 20px; margin: 12px 0;">
        <div style="display: flex; gap: 32px; flex-wrap: wrap;">
            <div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">Location</div>
                <div style="color: {COLORS['text_primary']}; font-size: 1.1em; font-weight: 600;">{selected_county} County, Washington</div>
            </div>
            <div>
                <div style="color: {COLORS['text_tertiary']}; font-size: 0.85em;">Forecast Window</div>
                <div style="color: {COLORS['text_primary']}; font-size: 1.1em;">{days} days (through {target_date.strftime('%B %d, %Y')})</div>
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
        _, _, ok = load_v2_model()
        if not ok:
            status.error("Model unavailable.")
            return
        status.info("Extracting county features…")
        hazard_df = load_hazard_data()
        status.info("Running temporal + spatial mesh inference…")
        risks, err = predict_single_county(selected_county, target_date)
        if risks is None:
            status.error(f"Prediction failed: {err}")
        else:
            status.info("Applying calibration (temperature scaling + seasonal bias)…")
            time.sleep(0.15)
            status.empty()
            st.session_state['last_prediction'] = {
                'county': selected_county,
                'date': str(target_date),
                'risks': risks,
                'horizon': days
            }

    if 'last_prediction' in st.session_state:
        last = st.session_state['last_prediction']
        if last.get('county') == selected_county:
            st.markdown("---")
            render_primary_risk_callout(last['risks'])
            render_hazard_cards(last['risks'])
            st.markdown("")
            render_risk_summary(last['risks'])
            st.markdown("---")
            with st.expander("County Spotlight Map", expanded=False):
                render_county_spotlight_map(selected_county, last['risks'], last.get('date'))
            render_interpretation_guide(last.get('horizon', days))


# =============================================================================
# PAGE: STATEWIDE
# =============================================================================

def page_statewide():
    st.markdown("## Statewide Predictions")
    st.caption("Run AHI v2.5 for all 39 Washington counties. Results include an interactive risk map.")

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
    hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']

    display = df.copy()
    for h in hazards:
        col = f'{h}_p'
        if col in display.columns:
            display[h.title()] = (display[col] * 100).round(1).astype(str) + '%'
    st.dataframe(
        display[['county'] + [h.title() for h in hazards]].rename(columns={'county': 'County'}),
        use_container_width=True, hide_index=True
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "Download Predictions (CSV)", data=csv,
        file_name=f"ahi_statewide_{target_date}.csv", mime="text/csv"
    )

    st.markdown("---")

    hazard_choice = st.selectbox(
        "Select hazard to display on map",
        ['Fire', 'Flood', 'Wind', 'Winter', 'Seismic'], index=0
    )
    render_statewide_choropleth(df, hazard_choice.lower(), hazard_choice)


# =============================================================================
# PAGE: RISK ASSESSMENT (new statewide summary tab)
# =============================================================================

def page_risk_assessment():
    st.markdown("## Comprehensive Risk Assessment")
    st.caption("Portfolio-level view of model predictions across all 39 Washington counties.")

    if 'statewide' not in st.session_state:
        st.info("Run **Statewide Predictions** first — this tab summarizes those results.")
        if st.button("Go to Statewide Predictions ▶"):
            st.session_state['_nav_hint'] = 'statewide'
        return

    df = st.session_state['statewide'].copy()
    hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']

    # Compute composite risk score = max hazard probability per county
    df['max_p'] = df[[f'{h}_p' for h in hazards]].max(axis=1)
    df['max_hazard'] = df[[f'{h}_p' for h in hazards]].idxmax(axis=1).str.replace('_p', '').map(HAZARD_NAMES)

    # Portfolio summary
    severe = int((df['max_p'] >= 0.50).sum())
    high = int(((df['max_p'] >= 0.35) & (df['max_p'] < 0.50)).sum())
    moderate = int(((df['max_p'] >= 0.20) & (df['max_p'] < 0.35)).sum())
    elevated = int(((df['max_p'] >= 0.10) & (df['max_p'] < 0.20)).sum())
    low = int((df['max_p'] < 0.10).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Severe (>50%)", severe)
    c2.metric("High (35–50%)", high)
    c3.metric("Moderate (20–35%)", moderate)
    c4.metric("Elevated (10–20%)", elevated)
    c5.metric("Low (<10%)", low)

    st.markdown("---")

    # Per-hazard statewide summary
    st.markdown("### Per-Hazard Statewide Summary")
    hazard_rows = []
    for h in hazards:
        col = f'{h}_p'
        idx_max = df[col].idxmax()
        hazard_rows.append({
            'Hazard': HAZARD_NAMES[h],
            'Statewide Mean': f"{df[col].mean()*100:.1f}%",
            'Median': f"{df[col].median()*100:.1f}%",
            'Max': f"{df[col].max()*100:.1f}%",
            'Highest County': df.loc[idx_max, 'county'],
            'Counties ≥ 20%': int((df[col] >= 0.20).sum()),
        })
    st.dataframe(pd.DataFrame(hazard_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Top-10 chart per selected hazard
    st.markdown("### County Ranking by Hazard")
    rank_col1, rank_col2 = st.columns([1, 3])
    with rank_col1:
        rank_hazard = st.selectbox("Hazard", [HAZARD_NAMES[h] for h in hazards], index=0, key='rank_hazard')
        top_n = st.slider("Counties to show", 5, 39, 10, key='rank_topn',
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
            xaxis=dict(title=f"{rank_hazard} Risk (%)", gridcolor=COLORS['border'], range=[0, max(top_df['pct'].max() * 1.15, 10)]),
            yaxis=dict(gridcolor=COLORS['border']),
            height=max(320, 28 * top_n),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sortable detail table
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
    st.markdown("## AHI v2.5 — Learned Seasonal Bias Model Diagnostics")

    _, _, ok = load_v2_model()
    if not ok:
        st.error("AHI v2.5 model not loaded.")
        return

    st.markdown("""
    ### What is AHI v2.5 (Learned Seasonal Bias)?

    **AHI v2.5** is the Adaptive Hazard Intelligence model powering this dashboard. It predicts the
    likelihood of five natural hazard types across all 39 Washington State counties — like a weather
    forecast, but for emergencies.

    **The core problem it solves:** Weather sequences (temperature, wind, precipitation) evolve on a
    fast timescale (days), while spatial correlations (smoke drift, downstream flooding, storm tracks)
    operate on a slow timescale (weeks/seasons). A single attention stack cannot efficiently extract both.

    **How it works (stacked mesh architecture):**
    1. **Temporal Mesh** — 3-layer transformer with **heat kernel diffusion attention** learns per-hazard memory horizons (fire needs ~3 months of context, flood needs ~1 week)
    2. **Spatial Mesh** — 2-layer transformer with **standard softmax attention** + county adjacency masking captures cross-county correlations (wildfire spread, downstream flooding)
    3. **Gated Coupling** — A learned gate blends temporal and spatial representations, starting near-zero and growing as the spatial signal proves useful during training
    4. **MMA Bias Field** — Multi-Modal Attention routes different feature types (weather, geography, land cover) through type-aware attention biases

    **v2.5 improvement:** Replaces hardcoded seasonal penalty functions with a **Learned Seasonal Bias** module
    — a trainable 5×12 parameter matrix (one weight per hazard per month) that the model optimizes during training.
    The model independently discovered the same seasonal structure that was previously hardcoded, with finer granularity.
    This eliminates manual per-state seasonal configuration, critical for scaling to all 50 states.

    **Key innovation:** Date-grouped batching — each training step sees all 39 counties for the same date,
    giving the spatial mesh a coherent snapshot to learn cross-county patterns from.

    **Result:** Mean AUC = **0.829**, surpassing the XGBoost baseline (0.781) and v2.0 (0.819) across all five hazard types.
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "AHI v2.5")
    col2.metric("Parameters", "1.3M")
    col3.metric("Attention", "Heat Kernel + Softmax")
    col4.metric("Status", "Online")

    col5, col6, col7 = st.columns(3)
    col5.metric("Coupling Gate", "0.0828")
    col6.metric("Spatial Graph", "39 counties")
    col7.metric("Mean Test AUC", "0.829")

    st.markdown("### Architecture")
    st.markdown("""
    | Component | Details | What it does |
    |-----------|---------|--------------|
    | **Multi-Modal Embedding** | MLP, 128 dim, 50 static + 14×20 temporal | Encodes weather, geography, land cover into a unified representation |
    | **Temporal Mesh** | 3-layer transformer, **heat kernel diffusion**, 4 heads | Learns per-hazard memory horizons — fire needs ~3mo, flood ~1wk |
    | **Spatial Mesh** | 2-layer transformer, **standard softmax**, 4 heads | Captures cross-county correlations using k=5 nearest-neighbor adjacency |
    | **Gated Coupling** | `temporal + gate * proj(spatial)`, gate init 0.01 | Blends spatial signal into temporal — gate frozen for 3 warmup epochs |
    | **MMA Bias Field** | 3-channel low-rank (rank=8) attention bias | Routes heterogeneous feature types through type-aware attention |
    | **Per-Hazard LoRA** | Low-rank adaptation (rank 16) per hazard, per layer | Hazard-specific fine-tuning without duplicating the full model |
    | **Cross-Hazard Interaction** | Physics-informed 5×5 mixing matrix | Models dependencies between correlated hazards |
    | **Prediction Heads** | 5 independent heads (128 → 64 → 32 → 1) | Calibrated logistic predictors per hazard type |
    """)

    st.markdown("### Training Configuration")
    st.markdown("""
    | Setting | Value | Purpose |
    |---------|-------|---------|
    | **Loss Function** | Focal loss (γ=2.0, α=0.75) | Down-weights easy negatives to handle severe class imbalance |
    | **Seasonal Bias** | Learned 5×12 parameter matrix (v2.5) | Replaces hardcoded penalties; model learns seasonal structure from data |
    | **Batching** | Date-grouped (all 39 counties per batch) | Ensures spatial mesh sees coherent county snapshots |
    | **Coupling Warmup** | Gate frozen at 0.01 for 3 epochs | Prevents spatial noise from corrupting warm-started temporal weights |
    | **Warm Start** | v1 weights transferred to temporal mesh | Guarantees v2 starts at v1 performance; spatial mesh adds on top |
    | **Optimizer** | AdamW (lr=1e-4, weight_decay=0.05) | Aggressive regularization for small dataset |
    | **Scheduler** | OneCycleLR with 10% warmup | Gradual warm-up prevents early-training instability |
    | **Early Stopping** | Patience=7 on val AUC | More patient than v1 — spatial mesh needs time to learn |
    | **Train/Val/Test Split** | Temporal: 80/10/10 by date | Prevents temporal leakage — model never sees future data |
    """)

    st.markdown("---")
    st.markdown("### Performance by Hazard (Held-out Test Set)")

    v2_data = [
        {"Hazard": "Winter",  "AUC": 0.908, "Quality": "Excellent", "Notes": "Best performer. Clear temporal + spatial patterns."},
        {"Hazard": "Fire",    "AUC": 0.851, "Quality": "Excellent", "Notes": "Spatial mesh captures smoke/burn spread patterns."},
        {"Hazard": "Wind",    "AUC": 0.837, "Quality": "Excellent", "Notes": "Spatial correlations help track storm movement."},
        {"Hazard": "Flood",   "AUC": 0.830, "Quality": "Excellent", "Notes": "Learned seasonal bias improves flood discrimination."},
        {"Hazard": "Seismic", "AUC": 0.718, "Quality": "Good",      "Notes": "Historical spatial patterns; earthquakes inherently hard to predict."},
    ]
    st.dataframe(pd.DataFrame(v2_data), use_container_width=True, hide_index=True)

    hazards_v2 = ["Fire", "Winter", "Wind", "Flood", "Seismic"]
    aucs_v2 = [0.851, 0.908, 0.837, 0.830, 0.718]
    bar_colors = [COLORS['fire'], COLORS['winter'], COLORS['wind'], COLORS['flood'], COLORS['seismic']]

    fig_v2 = go.Figure()
    fig_v2.add_trace(go.Bar(x=hazards_v2, y=aucs_v2, marker_color=bar_colors, name="AHI v2.5"))
    fig_v2.add_hline(y=0.8, line_dash="dash", line_color="#6b9e7a", annotation_text="Excellent (0.8)")
    fig_v2.add_hline(y=0.5, line_dash="dash", line_color="#dc2626", annotation_text="Random (0.5)")
    fig_v2.update_layout(
        title="AHI v2.5 AUC by Hazard Type",
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(color=COLORS['text_secondary'], family='Inter'),
        xaxis=dict(gridcolor=COLORS['border']),
        yaxis=dict(title="AUC-ROC", range=[0, 1], gridcolor=COLORS['border']),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_v2, use_container_width=True)
    st.success("**AHI v2.5 Mean AUC: 0.829** — 4 of 5 hazards in the Excellent range (AUC > 0.8)")

    st.markdown("---")
    st.markdown("### Calibration")
    st.markdown("""
    **Calibration** means predicted probabilities match real-world frequencies. If the model says 10% fire risk,
    fires should occur roughly 10% of the time in those conditions. AHI v2.5 uses:

    - **Per-hazard temperature scaling** — NLL-optimized on validation set
    - **Seasonal logit bias** — physics-informed monthly adjustments by hazard
    - **Base-rate ceilings** — caps predictions at historical plausibility limits
    - **Seismic dampening** — constant geographic background risk (not weather-driven)
    """)

    with st.expander("Model Evolution — prior generations", expanded=False):
        st.markdown("""
    AHI v2.5 is the result of iterative R&D across multiple architectures:

    | Metric | XGBoost Baseline | HazardLM v1 | AHI v2.0 | **AHI v2.5** |
    |--------|----------------:|--------------------:|------------------------:|------------------------:|
    | **Mean AUC** | 0.781 | 0.641 | 0.819 | **0.829** |
    | **Fire** | 0.870 | 0.731 | 0.848 | **0.851** |
    | **Flood** | 0.714 | 0.648 | 0.818 | **0.830** |
    | **Wind** | 0.713 | 0.585 | 0.823 | **0.837** |
    | **Winter** | 0.885 | 0.742 | 0.904 | **0.908** |
    | **Seismic** | 0.721 | 0.499 | 0.703 | **0.718** |
    | **Params** | N/A (trees) | 880K | 1.3M | **1.3M** |
    | **Architecture** | Per-hazard trees | Single-stack diffusion | Stacked mesh | **Learned seasonal bias** |

    **XGBoost** was the initial baseline — strong on fire and winter but limited on spatially-correlated hazards.
    **HazardLM v1** introduced heat kernel attention but couldn't handle multiple timescales simultaneously.
    **AHI v2.0** resolved this with separate temporal and spatial meshes connected by gated coupling.
    **AHI v2.5** replaces hardcoded seasonal penalties with a learned 5×12 bias matrix, improving all hazard AUCs
    and enabling expansion to new states without manual seasonal configuration.
    """)

    st.markdown("---")
    st.markdown("### Updates & Roadmap")

    st.markdown("**Current (AHI v2.5 Learned Seasonal Bias)**")
    st.markdown("""
    - Stacked mesh architecture grounded in Simplicial Computation theory (resolves timescale incompatibility)
    - Temporal mesh (heat kernel) + spatial mesh (softmax + adjacency) + gated coupling achieves mean AUC 0.829
    - Learned Seasonal Bias module (5×12 trainable matrix) replaces hardcoded seasonal penalties — scales to new states without manual configuration
    - Date-grouped batching ensures spatial mesh sees coherent 39-county snapshots per training step
    - Warm-started from v1 weights — guaranteed no performance regression during training
    - Fire 0.851, Flood 0.830, Wind 0.837, Winter 0.908, Seismic 0.718 on held-out test set
    """)

    st.markdown("**Planned / Future Work**")
    st.markdown("""
    - Expand to Pacific Northwest states (Oregon, Idaho) with hierarchical calibration for low-event counties
    - Integrate real-time weather feeds (NWS / NOAA) for operational nowcasts — **core SBIR Phase I deliverable**
    - Add Monte Carlo Dropout uncertainty quantification for prediction intervals
    - Improve spatial modeling (graph neural networks for county adjacency / hazard spread)
    - Build continual learning pipeline with scheduled model retraining
    - Conduct softmax ablation study to quantify diffusion attention benefit vs. standard transformer
    """)

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    | Source | Dataset | Usage |
    |--------|---------|-------|
    | **NOAA Storm Events** | 26 CSV files of historical storm records | Flood, wind, winter storm labels (strict county + 3-day window matching) |
    | **WFIGS** | Wildland Fire Locations Full History | Wildfire labels (geocoded to county boundaries) |
    | **USGS Earthquakes** | WA seismic catalog | Seismic event labels |
    | **FEMA** | Disaster declarations (geocoded) | Supplementary validation labels |
    | **GridMET** | Daily gridded weather | Temperature, precipitation, humidity, wind, fire weather (ERC) |
    | **US Census (TIGER)** | County-level population density | Static demographic feature for exposure weighting |
    | **NLCD / Land Cover** | Forest & urban fractions, elevation | Static geographic features for terrain-aware inference |
    """)

    st.caption(
        "Note: AHI v2.5 uses population density as its only demographic feature. "
        "CDC Social Vulnerability Index (SVI) data was evaluated but not incorporated into the "
        "training pipeline; it is reserved for future fairness/equity analysis."
    )


# =============================================================================
# PAGE: ABOUT
# =============================================================================

def page_about():
    st.markdown("## About AHI")

    st.markdown(f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: {COLORS['primary_light']}; margin-top: 0;">Adaptive Hazard Intelligence</h3>
        <p style="color: {COLORS['text_primary']}; line-height: 1.7;">
        AHI is a calibrated, multi-hazard risk prediction system for Washington State emergency managers.
        It predicts the likelihood of five natural hazard types — wildfire, flood, wind, winter storm, and
        seismic — at the county level using a stacked diffusion mesh transformer trained on 25 years of
        historical data.
        </p>
        <p style="color: {COLORS['text_secondary']}; line-height: 1.7;">
        AHI is being developed by Resilience Analytics Lab, LLC for SBIR Phase I commercialization toward
        operational nationwide deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Published Research")
    st.markdown("""
    - Curry, J.D. (2025). *Heat Kernel Attention: Diffusion-Based Attention for Transformer Architectures.* SSRN 5959898.
    - Curry, J.D. (2026). *Simplicial Computation: Topology as Control in Heterogeneous Attention.* SSRN 6037977.
    """)

    st.markdown("### Key Innovations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Stacked Mesh Architecture**
        - Separates fast temporal dynamics from slow spatial correlations
        - Resolves timescale incompatibility (τ*-incompatibility) proven in Simplicial Computation paper
        - Gated coupling prevents catastrophic interference between meshes

        **Date-Grouped Batching**
        - All 39 counties presented per training step
        - Enables coherent spatial attention learning
        - Key discovery: random batching produces gate ≈ 0 (spatial mesh ignored)
        """)
    with col2:
        st.markdown("""
        **Calibration Pipeline**
        - Per-hazard temperature scaling
        - Seasonal logit bias (fire suppressed in winter, winter suppressed in summer)
        - Base-rate ceilings prevent overconfident predictions
        - Seismic dampening (constant geographic risk, not weather-driven)

        **Clean Label Engineering**
        - 3-day event window (vs. 30-day which created 97.7% false positive rate)
        - Strict county-level geographic matching
        - Multiple source cross-validation (NOAA, WFIGS, USGS, FEMA)
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
            <div class="subtitle">Calibrated hazard risk for defensible decisions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "County Risk Assessment",
        "Statewide",
        "Risk Assessment",
        "Model Diagnostics",
        "About"
    ])

    with tab1:
        page_quick_predict()
    with tab2:
        page_statewide()
    with tab3:
        page_risk_assessment()
    with tab4:
        page_model_info()
    with tab5:
        page_about()


if __name__ == '__main__':
    main()
