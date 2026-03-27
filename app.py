"""
AHI — Adaptive Hazard Intelligence
Focused model prediction dashboard for SBIR Phase I demonstration.
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
import torch
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import folium
    from folium.features import GeoJsonTooltip
    from streamlit_folium import st_folium
    import geopandas as gpd
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Model imports
try:
    from ahi_v2_model import AHIv2Model, AHIv2Config
    from ahi_v2_graph import build_adjacency_graph
    AHI_V2_AVAILABLE = True
except ImportError as e:
    print(f"[IMPORT] AHI v2 import failed: {e}")
    AHI_V2_AVAILABLE = False

try:
    from inference_core import predict_county_risks_simple, predict_from_ahi_v2
except Exception as e:
    print(f"[IMPORT] inference_core import failed: {e}")
    predict_county_risks_simple = None
    predict_from_ahi_v2 = None

try:
    from ahi_v2_graph import get_batch_adjacency
except ImportError:
    get_batch_adjacency = None

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AHI — Adaptive Hazard Intelligence",
    page_icon="assets/logo-icon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path("data")
MAX_FORECAST_DAYS = 14

# Model paths
V2_MODEL_PATH_LOCAL = Path("outputs/ahi_v2/best_model.pt")
V2_MODEL_PATH_CLOUD = Path("/mount/src/ahi/outputs/ahi_v2/best_model.pt")
MIN_MODEL_SIZE = 5_000_000

# County coordinates
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
# COLOR THEME — Resilience Analytics Lab green palette
# =============================================================================

COLORS = {
    'app_bg': '#3B4A3B',
    'card_bg': '#161b22',
    'sidebar_bg': '#0d1117',
    'elevated_bg': '#1c2128',
    'primary': '#4a7c59',        # Sage green (from LLC brand)
    'primary_light': '#6b9e7a',
    'primary_dark': '#2d5a3a',
    'accent': '#8fbc8f',         # Dark sea green
    'border': '#30363d',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    # Hazard-specific
    'fire': '#e05252',
    'flood': '#4a90d9',
    'wind': '#9b59b6',
    'winter': '#2ec4b6',
    'seismic': '#e67e22',
}

HAZARD_NAMES = {
    'fire': 'Fire', 'flood': 'Flood', 'wind': 'Wind',
    'winter': 'Winter Storm', 'seismic': 'Seismic'
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

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

    /* Tabs */
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

    /* Cards */
    .hazard-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 20px 16px;
        text-align: center;
        transition: border-color 0.2s;
    }}
    .hazard-card:hover {{
        border-color: {COLORS['primary']};
    }}
    .hazard-card .label {{
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 4px;
    }}
    .hazard-card .value {{
        font-size: 1.6em;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    /* Buttons */
    .stButton > button {{
        background: {COLORS['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.6em 1.5em !important;
    }}
    .stButton > button:hover {{
        background: {COLORS['primary_light']} !important;
    }}

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {{
        background: {COLORS['card_bg']};
        border-color: {COLORS['border']};
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px;
    }}

    /* Header branding */
    .ahi-header {{
        margin-bottom: 8px;
    }}
    .ahi-header .title {{
        font-weight: 600 !important;
        color: {COLORS['text_primary']};
    }}
    .ahi-header .subtitle {{
        color: {COLORS['text_secondary']};
        font-size: 0.95em;
    }}

    /* Risk interpretation */
    .risk-section {{
        background: {COLORS['card_bg']};
        border-left: 3px solid {COLORS['primary']};
        padding: 16px 20px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
    }}

    /* Hide hamburger + footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA & MODEL LOADING
# =============================================================================

@st.cache_resource
def load_v2_model():
    """Load AHI v2 stacked mesh model + adjacency graph."""
    if not AHI_V2_AVAILABLE:
        return None, None, False

    v2_path = None
    for p in [V2_MODEL_PATH_LOCAL, V2_MODEL_PATH_CLOUD]:
        if p.exists() and p.stat().st_size > MIN_MODEL_SIZE:
            v2_path = p
            break

    if v2_path is None:
        return None, None, False

    try:
        config = AHIv2Config()
        model = AHIv2Model(config)
        state = torch.load(str(v2_path), map_location='cpu', weights_only=False)
        sd = state.get('model_state_dict', state.get('state_dict', state))
        model.load_state_dict(sd, strict=False)
        model.eval()

        adjacency, _, _, county_names = build_adjacency_graph()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[AHI] Model loaded: {total_params:,} params, gate={model.coupling.gate.item():.4f}")

        return model, adjacency, True
    except Exception as e:
        import traceback
        print(f"[AHI] Load failed: {e}\n{traceback.format_exc()}")
        return None, None, False


@st.cache_data
def load_hazard_data():
    """Load the canonical hazard dataset."""
    path = DATA_DIR / 'hazard_lm_clean_labeled.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data
def load_geojson():
    """Load WA county boundaries."""
    path = DATA_DIR / 'wa_counties.geojson'
    if path.exists() and FOLIUM_AVAILABLE:
        return gpd.read_file(str(path))
    return None


@st.cache_data
def load_county_metadata():
    """Load county population/SVI metadata."""
    for name in ['WA_County_Master_Table__Final_with_Schools_.cleaned.csv',
                 'WA_County_Master_Table__Final_with_Schools_.csv',
                 'wa_county_master.csv']:
        path = DATA_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return None


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def predict_single_county(county_name, target_date):
    """Run AHI v2 inference for one county. Returns (risks_dict, error_msg)."""
    model, adjacency, ok = load_v2_model()
    if not ok or model is None:
        return None, "Model not loaded — check outputs/ahi_v2/best_model.pt"

    hazard_df = load_hazard_data()
    if hazard_df is None or len(hazard_df) == 0:
        return None, "Hazard dataset not found"

    try:
        import inference_core as _ic

        if not _ic._COUNTY_MAP:
            _ic._build_maps(hazard_df)

        county_upper = county_name.upper().replace(' COUNTY', '').strip()
        mask = hazard_df['county'].str.upper().str.replace(' COUNTY', '').str.strip() == county_upper
        county_rows = hazard_df[mask]

        if len(county_rows) == 0:
            return None, f"No data for county: {county_name}"

        if 'date' in county_rows.columns:
            county_rows = county_rows.sort_values('date', ascending=False)
        county_row = county_rows.iloc[0]
        actual_county = county_row.get('county', county_name)

        static_cont, temporal, region_ids, state_ids, nlcd_ids = \
            _ic.build_tensors_from_county_data(county_row, actual_county, target_date)

        num_nodes = adjacency.size(0)
        spatial_mask = get_batch_adjacency(adjacency, region_ids, num_nodes)

        month = target_date.month
        risks = predict_from_ahi_v2(
            model, static_cont, temporal, region_ids, state_ids, nlcd_ids,
            adjacency_mask=spatial_mask, month=month,
        )
        return risks, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Prediction failed: {e}"


def predict_all_counties(target_date, progress_callback=None):
    """Run predictions for all 39 counties. Returns DataFrame."""
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

def risk_level(prob):
    """Map probability to risk level and interpretation."""
    if prob < 0.10:
        return "Low", "Baseline conditions — routine monitoring"
    elif prob < 0.20:
        return "Elevated", "Above baseline — increased awareness recommended"
    elif prob < 0.35:
        return "Moderate", "Notable risk — review preparedness plans"
    elif prob < 0.50:
        return "High", "Significant risk — consider pre-positioning resources"
    else:
        return "Severe", "Elevated readiness recommended"


HAZARD_GUIDANCE = {
    'fire': 'Review evacuation routes. Coordinate with fire districts on resource availability. Assess defensible space near critical facilities. Verify water supply access points.',
    'flood': 'Inspect drainage systems and culverts. Verify flood gauge monitoring. Pre-stage pumps and sandbags at flood-prone areas. Coordinate road closure plans.',
    'wind': 'Coordinate with utilities on power line inspections. Secure outdoor equipment. Pre-position generators at critical facilities. Alert manufactured housing communities.',
    'winter': 'Verify road treatment supplies. Check backup power at warming shelters. Coordinate with WSDOT on plowing priorities. Prepare travel advisory messaging.',
    'seismic': 'Review structural assessments for critical buildings. Confirm communications redundancy. Verify search and rescue readiness. Review tsunami evacuation routes for coastal areas.',
}


def render_hazard_cards(risks):
    """Render sorted hazard probability cards."""
    sorted_hazards = sorted(risks.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(5)
    for col, (hazard, prob) in zip(cols, sorted_hazards):
        pct = f"{prob * 100:.1f}%"
        color = COLORS.get(hazard, COLORS['primary'])
        with col:
            st.markdown(f"""
            <div class="hazard-card">
                <div class="label" style="color: {color};">{hazard.title()}</div>
                <div class="value">{pct}</div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_summary(risks):
    """Render sorted risk interpretation with guidance."""
    sorted_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)

    st.markdown(f"#### Top Hazards for This Period")

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


def render_interpretation_guide():
    """Expandable guide for interpreting predictions."""
    with st.expander("How to interpret these numbers", expanded=False):
        st.markdown(f"""
        **What the percentages mean:**
        - These are **calibrated risk probabilities** for the {MAX_FORECAST_DAYS}-day forecast window
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
# PAGE: QUICK PREDICT
# =============================================================================

def page_quick_predict():
    st.markdown("## Quick Predict")
    st.caption("Run the AHI v2 model for a single county. Predictions based on 25 years of historical hazard patterns.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_county = st.selectbox("Select County", COUNTIES, index=COUNTIES.index('King'))
    with col2:
        forecast_horizon = st.selectbox("Forecast Horizon", ["7 days", "14 days"], index=1)

    days = int(forecast_horizon.split()[0])
    today = datetime.now().date()
    target_date = today + timedelta(days=days)

    # County info card
    lat, lon = WA_COUNTY_COORDS.get(selected_county, (47.5, -120.5))
    month_name = target_date.strftime('%B')

    # Season context
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

    # Predict button
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running AHI v2 inference..."):
            risks, err = predict_single_county(selected_county, target_date)
            if risks is None:
                st.error(f"Prediction failed: {err}")
            else:
                st.session_state['last_prediction'] = {
                    'county': selected_county,
                    'date': str(target_date),
                    'risks': risks,
                    'horizon': days
                }

    # Render stored results
    if 'last_prediction' in st.session_state:
        last = st.session_state['last_prediction']
        if last.get('county') == selected_county:
            st.markdown("---")
            render_hazard_cards(last['risks'])
            st.markdown("")
            render_risk_summary(last['risks'])
            render_interpretation_guide()


# =============================================================================
# PAGE: STATEWIDE PREDICTIONS
# =============================================================================

def page_statewide():
    st.markdown("## Statewide Predictions")
    st.caption("Run AHI v2 for all 39 Washington counties. Results include an interactive risk map.")

    target_date = datetime.now().date() + timedelta(days=MAX_FORECAST_DAYS)

    if st.button("Run Statewide Predictions", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def callback(i, total, county):
            progress.progress((i + 1) / total)
            status.text(f"Processing {county}... ({i+1}/{total})")

        with st.spinner(""):
            df = predict_all_counties(target_date, progress_callback=callback)

        progress.progress(1.0)
        status.text("Complete!")

        if df is not None and len(df) > 0:
            st.session_state['statewide'] = df
            st.success(f"Predictions complete for {len(df)} counties.")
        else:
            st.error("No predictions generated. Check model availability.")

    # Display cached results
    if 'statewide' not in st.session_state:
        st.info("Click **Run Statewide Predictions** to generate results.")
        return

    df = st.session_state['statewide']
    hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']

    # Summary table
    display = df.copy()
    for h in hazards:
        col = f'{h}_p'
        if col in display.columns:
            display[h.title()] = (display[col] * 100).round(1).astype(str) + '%'
    st.dataframe(
        display[['county'] + [h.title() for h in hazards]].rename(columns={'county': 'County'}),
        use_container_width=True, hide_index=True
    )

    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Predictions (CSV)", data=csv,
        file_name=f"ahi_statewide_{target_date}.csv", mime="text/csv"
    )

    st.markdown("---")

    # Interactive map
    hazard_choice = st.selectbox(
        "Select hazard to display on map",
        ['Fire', 'Flood', 'Wind', 'Winter', 'Seismic'], index=0
    )
    col_name = hazard_choice.lower() + '_p'

    gdf = load_geojson()

    if FOLIUM_AVAILABLE and gdf is not None:
        try:
            gdf_copy = gdf.copy()
            name_field = None
            for f in ['NAME', 'name', 'COUNTY', 'county_name']:
                if f in gdf_copy.columns:
                    name_field = f
                    break

            if name_field:
                gdf_copy['county_norm'] = gdf_copy[name_field].str.replace(' County', '').str.strip()
                df_copy = df.copy()
                df_copy['county_norm'] = df_copy['county'].str.replace(' County', '').str.strip()
                merged = gdf_copy.merge(df_copy, on='county_norm', how='left')

                # Build choropleth-style map with county polygons
                m = folium.Map(
                    location=[47.4, -120.5], zoom_start=7,
                    tiles='CartoDB dark_matter'
                )

                # Color scale
                def risk_color(prob):
                    if prob > 0.50:
                        return '#e05252'
                    elif prob > 0.35:
                        return '#e67e22'
                    elif prob > 0.20:
                        return '#f1c40f'
                    elif prob > 0.10:
                        return '#6b9e7a'
                    else:
                        return '#2d5a3a'

                # County polygons
                for _, row in merged.iterrows():
                    try:
                        prob = float(row.get(col_name, 0.0) or 0.0)
                        county = row.get('county_norm', 'Unknown')
                        color = risk_color(prob)
                        geom = row.get('geometry')

                        if geom is not None:
                            geo_json = folium.GeoJson(
                                geom.__geo_interface__,
                                style_function=lambda x, c=color, p=prob: {
                                    'fillColor': c,
                                    'color': '#30363d',
                                    'weight': 1,
                                    'fillOpacity': 0.55 + min(p, 0.4),
                                },
                            )
                            popup_html = f"""
                            <div style="font-family: Inter, sans-serif; min-width: 180px;">
                                <div style="font-weight: 700; font-size: 14px; margin-bottom: 4px;">{county} County</div>
                                <div style="font-size: 13px;">{hazard_choice}: <strong>{prob*100:.1f}%</strong></div>
                            </div>
                            """
                            geo_json.add_child(folium.Popup(popup_html, max_width=250))
                            geo_json.add_child(folium.Tooltip(f"{county}: {prob*100:.1f}%"))
                            geo_json.add_to(m)
                    except Exception:
                        continue

                st_folium(m, width=900, height=520, returned_objects=[])

                # Legend
                st.markdown(f"""
                <div style="display: flex; gap: 16px; justify-content: center; margin-top: 8px; flex-wrap: wrap;">
                    <span style="color: #2d5a3a;">&#9632; Low (&lt;10%)</span>
                    <span style="color: #6b9e7a;">&#9632; Elevated (10-20%)</span>
                    <span style="color: #f1c40f;">&#9632; Moderate (20-35%)</span>
                    <span style="color: #e67e22;">&#9632; High (35-50%)</span>
                    <span style="color: #e05252;">&#9632; Severe (&gt;50%)</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("GeoJSON missing county name field.")
        except Exception as e:
            st.warning(f"Map rendering failed: {e}")
    else:
        # Fallback: marker-based map without geopandas
        if FOLIUM_AVAILABLE:
            m = folium.Map(location=[47.4, -120.5], zoom_start=7, tiles='CartoDB dark_matter')
            for _, row in df.iterrows():
                county = row['county']
                prob = row.get(col_name, 0.0)
                lat, lon = WA_COUNTY_COORDS.get(county, (47.5, -120.5))
                color = 'red' if prob > 0.35 else 'orange' if prob > 0.20 else 'yellow' if prob > 0.10 else 'green'
                folium.CircleMarker(
                    location=[lat, lon], radius=8 + prob * 15,
                    color=color, fill=True, fill_opacity=0.7,
                    tooltip=f"{county}: {prob*100:.1f}%"
                ).add_to(m)
            st_folium(m, width=900, height=520)
        else:
            st.info("Install `folium`, `streamlit-folium`, and `geopandas` for map visualization.")


# =============================================================================
# PAGE: MODEL INFO
# =============================================================================

def page_model_info():
    st.markdown("## Model Diagnostics")

    model, adjacency, ok = load_v2_model()

    if not ok:
        st.error("AHI v2 model not loaded.")
        return

    total_params = sum(p.numel() for p in model.parameters())
    gate_val = model.coupling.gate.item()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Parameters", f"{total_params:,}")
    with col2:
        st.metric("Architecture", "Stacked Mesh")
    with col3:
        st.metric("Coupling Gate", f"{gate_val:.4f}")
    with col4:
        st.metric("Counties", "39")

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 24px;">
        <p style="color: {COLORS['text_primary']}; line-height: 1.7;">
        <strong>AHI v2</strong> uses a dual-mesh transformer that separates temporal and spatial processing:</p>
        <ul style="color: {COLORS['text_primary']}; line-height: 1.8;">
            <li><strong>Temporal Mesh</strong> (3 layers) — Heat kernel diffusion attention processes 14-day weather sequences,
            learning per-hazard memory horizons</li>
            <li><strong>Spatial Mesh</strong> (2 layers) — Standard softmax attention with k-nearest-neighbor county adjacency
            masking captures cross-county correlations</li>
            <li><strong>Gated Coupling</strong> — Learned gate (g ≈ {gate_val:.3f}) controls spatial contribution</li>
            <li><strong>5 Prediction Heads</strong> — Per-hazard LoRA adapters with cross-hazard interaction layer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Performance (AUC Scores)")

    perf_data = {
        'Hazard': ['Fire', 'Flood', 'Wind', 'Winter', 'Seismic', 'Mean'],
        'AHI v2': [0.848, 0.818, 0.823, 0.904, 0.703, 0.819],
        'XGBoost': [0.872, 0.714, 0.711, 0.890, 0.719, 0.781],
    }
    perf_df = pd.DataFrame(perf_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=perf_data['Hazard'], y=perf_data['AHI v2'],
        name='AHI v2', marker_color=COLORS['primary']
    ))
    fig.add_trace(go.Bar(
        x=perf_data['Hazard'], y=perf_data['XGBoost'],
        name='XGBoost', marker_color=COLORS['text_tertiary']
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color=COLORS['accent'],
                  annotation_text="Excellent (0.8)")
    fig.update_layout(
        barmode='group', height=380,
        paper_bgcolor=COLORS['card_bg'], plot_bgcolor=COLORS['card_bg'],
        font={'color': COLORS['text_secondary'], 'family': 'Inter'},
        xaxis={'gridcolor': COLORS['border']},
        yaxis={'gridcolor': COLORS['border'], 'range': [0.5, 1.0], 'title': 'AUC-ROC'},
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(perf_df.style.format({
        'AHI v2': '{:.3f}', 'XGBoost': '{:.3f}'
    }), use_container_width=True, hide_index=True)

    st.markdown("### Training Data")
    st.markdown(f"""
    - **Observations:** 370,000+ county-day records
    - **Counties:** All 39 Washington State counties
    - **Time span:** 2000–2025
    - **Sources:** NOAA Storm Events, WFIGS Wildfires, USGS Earthquakes, GridMET weather, Census/SVI demographics
    - **Label window:** 3-day event attribution (strict county matching)
    """)


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
        The system was developed as a BAS-EM capstone project at Pierce College and is being extended
        through Resilience Analytics Lab, LLC for SBIR Phase I commercialization toward operational
        nationwide deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Published Research")
    st.markdown("""
    - Curry, J.D. (2025). *Heat Kernel Attention: Diffusion-Based Attention for Transformer Architectures.* SSRN.
    - Curry, J.D. (2026). *Meta-Meta Attention: Content-Aware Bias Fields for Heterogeneous Feature Routing.* SSRN 6316718.
    - Curry, J.D. (2026). *Simplicial Computation: Topology as Control in Heterogeneous Attention.* SSRN 6037977.
    """)

    st.markdown("### Key Innovations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
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
        st.markdown(f"""
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
        <div>Everett, Washington</div>
        <div style="margin-top: 8px;">Joshua D. Curry — Founder & Principal Investigator</div>
        <div>Pierce College Fort Steilacoom — Emergency Management Department</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    inject_css()

    # Header
    st.markdown(f"""
    <div class="ahi-header">
        <h2 class="title">Adaptive Hazard Intelligence</h2>
        <div class="subtitle">Calibrated hazard risk for defensible decisions</div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quick Predict",
        "Statewide",
        "Model",
        "About"
    ])

    with tab1:
        page_quick_predict()
    with tab2:
        page_statewide()
    with tab3:
        page_model_info()
    with tab4:
        page_about()


if __name__ == '__main__':
    main()
