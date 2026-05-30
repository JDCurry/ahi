"""
State-aware ONNX inference engine for AHI v2.5.

Single deployment serves multiple states. Each state has its own:
  - calibration JSONs (temperature_scales, seasonal_bias, base_rate_ceiling)
  - inference parquet (states/XX/inference_data.parquet)
  - GeoJSON (states/XX/counties.geojson)
  - config.yaml (UI content)

Regional ONNX models are stored under models/<region>/model.onnx and shared
across all states in the same region. State -> region mapping lives in
states/registry.yaml.

Calibration pipeline (per state):
  1. (raw_logit + hazard_bias) / T  # per-hazard additive bias + temperature scaling
  2. + seasonal_bias[h][m]          # state's seasonal_bias.json
  3. sigmoid                        # convert to probability
  4. min(p, ceiling[h][m])          # state's base_rate_ceiling.json

Author: Joshua D. Curry
"""
import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

HAZARD_TYPES = ['fire', 'flood', 'wind', 'winter', 'seismic']

STATIC_FEATURE_COLS = [
    # [0-20] GridMET core (in parquets)
    'latitude', 'longitude', 'day_of_year', 'month', 'year',
    'tmmx', 'tmmn', 'rmin', 'rmax', 'vs', 'erc', 'pr', 'vpd',
    'red_flag_active', 'tmmx_3d_mean', 'pr_3d_mean', 'vs_3d_mean',
    'elevation', 'forest_fraction', 'urban_fraction', 'pop_density',
    # [21-24] ERA5 water vapor transport (daily — zero until live pipeline)
    'era5_ivt_max', 'era5_tcwv_max', 'era5_ivt_mean', 'era5_tcwv_mean',
    # [25-29] NFHL flood zones (static, merged into inference parquets)
    'nfhl_sfha_frac', 'nfhl_v_frac', 'nfhl_x_frac', 'nfhl_sfha_km2', 'nfhl_v_km2',
    # [30-31] ERA5 temperature/pressure (daily — zero until live pipeline)
    'era5_t2m_min', 'era5_msl_min',
    # [32-35] MODIS vegetation indices (daily — zero until live pipeline)
    'modis_ndvi', 'modis_evi', 'modis_ndvi_anom', 'modis_evi_anom',
    # [36-43] ERA5 extended (daily — zero until live pipeline)
    'era5_tp_sum', 'era5_tp_max', 'era5_msl_mean', 'era5_gust_max',
    'era5_ws_max', 'era5_t2m_mean', 'era5_t2m_max', 'era5_ws_mean',
    # [44-49] WUI (static, merged into inference parquets)
    'wui_frac', 'wui_intermix_frac', 'wui_interface_frac',
    'wui_veg_frac', 'wui_veg_cover_mean', 'wui_huden_log',
]

# Round 2: climate-region routing for the per-region prediction heads.
# Each state maps to a region ID 0-8 (canonical order). The ONNX model
# uses this index to gather the correct per-region head per row.
STATE_TO_REGION_ID = {
    'CO': 0,
    'IL': 1, 'IN': 1, 'KY': 1, 'MI': 1, 'OH': 1, 'TN': 1, 'WV': 1,
    'AZ': 2, 'ID': 2, 'MT': 2, 'NM': 2, 'NV': 2, 'UT': 2, 'WY': 2,
    'CT': 3, 'DC': 3, 'DE': 3, 'MA': 3, 'MD': 3, 'ME': 3,
    'NH': 3, 'NJ': 3, 'NY': 3, 'PA': 3, 'RI': 3, 'VA': 3, 'VT': 3,
    'IA': 4, 'MN': 4, 'MO': 4, 'ND': 4, 'SD': 4, 'WI': 4,
    'CA': 5,
    'OR': 6, 'WA': 6,
    'AL': 7, 'AR': 7, 'FL': 7, 'GA': 7, 'LA': 7, 'MS': 7, 'NC': 7, 'SC': 7,
    'KS': 8, 'NE': 8, 'OK': 8, 'TX': 8,
}

# ---------------------------------------------------------------------------
# Per-state calibration cache: {state_code: {temperatures, biases, ceilings}}
# ---------------------------------------------------------------------------
_STATE_CALIBRATION_CACHE: Dict[str, Dict] = {}
_COUNTY_BIAS_CACHE:       Dict[str, Dict] = {}   # state_code -> county_seasonal_bias.json
_REGIONAL_ONNX_CACHE:     Dict[str, object] = {}   # region -> ort.InferenceSession
_COUNTY_MAP: Dict[str, int] = {}
_STATE_MAP:  Dict[str, int] = {}

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# State calibration loader
# ---------------------------------------------------------------------------

def _load_state_calibration(state_code: str) -> Dict:
    """Load (and cache) all three calibration JSONs for a state."""
    if state_code in _STATE_CALIBRATION_CACHE:
        return _STATE_CALIBRATION_CACHE[state_code]

    state_dir = ROOT / 'states' / state_code

    # Temperature scales + per-hazard logit biases
    t_path = state_dir / 'temperature_scales.json'
    hazard_biases = {h: 0.0 for h in HAZARD_TYPES}
    if t_path.exists():
        with open(t_path, encoding="utf-8-sig") as f:
            t_doc = json.load(f)
        temperatures = t_doc.get('temperatures', t_doc)
        temperatures = {h: float(temperatures[h]) for h in HAZARD_TYPES if h in temperatures}
        # Per-hazard biases (new format: "biases" dict)
        if 'biases' in t_doc:
            for h, v in t_doc['biases'].items():
                if h in HAZARD_TYPES:
                    hazard_biases[h] = float(v)
        # Legacy: fire_bias field (single-hazard format)
        elif 'fire_bias' in t_doc:
            hazard_biases['fire'] = float(t_doc['fire_bias'])
    else:
        print(f"[CALIBRATION] {state_code}: no temperature_scales.json — using T=1.0")
        temperatures = {h: 1.0 for h in HAZARD_TYPES}

    # Seasonal bias
    sb_path = state_dir / 'seasonal_bias.json'
    biases = {h: {m: 0.0 for m in range(1, 13)} for h in HAZARD_TYPES}
    if sb_path.exists():
        with open(sb_path, encoding="utf-8-sig") as f:
            sb_doc = json.load(f)
        for h, monthly in sb_doc.get('biases', {}).items():
            if h in HAZARD_TYPES:
                biases[h] = {int(m): float(v) for m, v in monthly.items()}
    else:
        print(f"[CALIBRATION] {state_code}: no seasonal_bias.json — using zero biases")

    # Ceilings
    bc_path = state_dir / 'base_rate_ceiling.json'
    base_ceiling = {h: 1.0 for h in HAZARD_TYPES}
    seasonal_ceiling = {}
    if bc_path.exists():
        with open(bc_path, encoding="utf-8-sig") as f:
            bc_doc = json.load(f)
        for h, v in bc_doc.get('base_rate_ceiling', {}).items():
            if h in HAZARD_TYPES:
                base_ceiling[h] = float(v)
        for h, monthly in bc_doc.get('seasonal_ceiling', {}).items():
            if h in HAZARD_TYPES:
                seasonal_ceiling[h] = {int(m): float(v) for m, v in monthly.items()}
    else:
        print(f"[CALIBRATION] {state_code}: no base_rate_ceiling.json — using p=1.0")

    cal = {
        'temperatures':     temperatures,
        'hazard_biases':    hazard_biases,
        'biases':           biases,
        'base_ceiling':     base_ceiling,
        'seasonal_ceiling': seasonal_ceiling,
    }
    _STATE_CALIBRATION_CACHE[state_code] = cal
    active_biases = {h: v for h, v in hazard_biases.items() if v != 0.0}
    bias_str = ', '.join(f'{h}_bias={v:+.3f}' for h, v in active_biases.items())
    if bias_str:
        bias_str = f", {bias_str}"
    print(f"[CALIBRATION] Loaded {state_code} calibration "
          f"(T fire={temperatures.get('fire', 1.0):.3f}{bias_str})")
    return cal


def _get_ceiling(state_code: str, hazard: str, month: int) -> float:
    cal = _load_state_calibration(state_code)
    sc = cal['seasonal_ceiling']
    if month and 1 <= month <= 12 and hazard in sc:
        return sc[hazard].get(month, cal['base_ceiling'].get(hazard, 1.0))
    return cal['base_ceiling'].get(hazard, 1.0)


def _load_county_bias(state_code: str) -> Optional[Dict]:
    """Load county-level seasonal biases if available."""
    if state_code in _COUNTY_BIAS_CACHE:
        return _COUNTY_BIAS_CACHE[state_code]

    cb_path = ROOT / 'states' / state_code / 'county_seasonal_bias.json'
    if cb_path.exists():
        with open(cb_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        _COUNTY_BIAS_CACHE[state_code] = data
        print(f"[CALIBRATION] Loaded {state_code} county-level biases "
              f"({data.get('n_counties', '?')} counties)")
        return data
    else:
        _COUNTY_BIAS_CACHE[state_code] = None
        return None


def _get_county_bias(state_code: str, hazard: str, month: int,
                      county_name: Optional[str] = None) -> float:
    """Get seasonal bias, preferring county-level over state-level.

    Lookup order:
      1. county_seasonal_bias.json[hazard][county][month]  (exact match)
      2. county_seasonal_bias.json[hazard][county][month]  (case-insensitive)
      3. seasonal_bias.json[hazard][month]                 (state fallback)
    """
    if county_name:
        cb = _load_county_bias(state_code)
        if cb and 'biases' in cb:
            hazard_biases = cb['biases'].get(hazard, {})
            m_str = str(month)

            # Exact match
            if county_name in hazard_biases:
                county_data = hazard_biases[county_name]
                if m_str in county_data:
                    return float(county_data[m_str])

            # Case-insensitive fallback
            county_upper = county_name.upper().replace(' COUNTY', '').strip()
            for stored_name, monthly in hazard_biases.items():
                if stored_name.upper().replace(' COUNTY', '').strip() == county_upper:
                    if m_str in monthly:
                        return float(monthly[m_str])

    # Fall back to state-level bias
    cal = _load_state_calibration(state_code)
    return cal['biases'].get(hazard, {}).get(month, 0.0)


def _apply_calibration(state_code: str, raw_logit: float,
                        hazard: str, month: int,
                        county_name: Optional[str] = None) -> float:
    """Apply calibration to a single raw logit.

    Calibration pipeline:
      1. (logit + hazard_bias) / T — per-state additive bias shifts logit
                                      before temperature scaling (all hazards)
      2. + seasonal_bias           — county-level or state-level monthly bias
      3. sigmoid                   — convert to probability
      4. min(p, ceiling)           — cap at base rate ceiling

    Uses county-level bias when available (county_seasonal_bias.json),
    otherwise falls back to statewide bias (seasonal_bias.json).
    """
    cal = _load_state_calibration(state_code)

    T = max(cal['temperatures'].get(hazard, 1.0), 0.01)

    # Per-state additive bias: shifts logit before T-scaling
    # prob = sigmoid((logit + bias) / T + seasonal_bias)
    logit = raw_logit + cal['hazard_biases'].get(hazard, 0.0)

    scaled = logit / T

    bias = _get_county_bias(state_code, hazard, month, county_name)
    scaled += bias

    prob = 1.0 / (1.0 + math.exp(-scaled))
    ceiling = _get_ceiling(state_code, hazard, month)
    return max(0.0, min(prob, ceiling))


# ---------------------------------------------------------------------------
# Regional ONNX session
# ---------------------------------------------------------------------------

def _get_onnx_session(region: str):
    """Lazy-load (and cache) the regional ONNX session."""
    if region in _REGIONAL_ONNX_CACHE:
        return _REGIONAL_ONNX_CACHE[region]

    try:
        import onnxruntime as ort
    except ImportError:
        print("[AHI] onnxruntime not installed — ONNX inference unavailable")
        return None

    candidates = [
        ROOT / 'models' / region / 'model.onnx',
        Path(f'/mount/src/ahi-platform/models/{region}/model.onnx'),  # Render
    ]
    for p in candidates:
        if p.exists():
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.enable_cpu_mem_arena = False   # reduce retained memory on Render
            opts.enable_mem_pattern = False
            session = ort.InferenceSession(
                str(p), sess_options=opts, providers=['CPUExecutionProvider']
            )
            _REGIONAL_ONNX_CACHE[region] = session
            print(f"[AHI] Loaded regional model {region} from {p} "
                  f"({p.stat().st_size / 1024 / 1024:.1f} MB)")
            return session

    print(f"[AHI] No ONNX model found for region '{region}' — checked {candidates}")
    _REGIONAL_ONNX_CACHE[region] = None   # cache the negative result so we don't spam the log
    return None


def model_available(region: str) -> bool:
    """Cheap, side-effect-free check for whether a regional model exists on disk.

    Used by validation tooling to decide whether to run model-dependent checks
    (extreme events, ceiling lock-in, logit traces) or just data-only checks.
    """
    candidates = [
        ROOT / 'models' / region / 'model.onnx',
        Path(f'/mount/src/ahi-platform/models/{region}/model.onnx'),
    ]
    return any(p.exists() for p in candidates)


def _run_onnx_inference(region: str, static_cont: np.ndarray, temporal: np.ndarray,
                        region_ids: np.ndarray, state_ids: np.ndarray,
                        nlcd_ids: np.ndarray,
                        climate_region_ids: Optional[np.ndarray] = None,
                        state_code: Optional[str] = None) -> Optional[Dict[str, float]]:
    session = _get_onnx_session(region)
    if session is None:
        return None
    feeds = {
        'static_cont': static_cont.astype(np.float32),
        'temporal':    temporal.astype(np.float32),
        'region_ids':  region_ids.astype(np.int64),
        'state_ids':   state_ids.astype(np.int64),
        'nlcd_ids':    nlcd_ids.astype(np.int64),
    }
    # Round 2 ONNX has a sixth input. Pass it if the model expects it.
    input_names = {i.name for i in session.get_inputs()}
    if 'climate_region_ids' in input_names:
        if climate_region_ids is None:
            # Fallback: derive from state_code (single-row inference)
            crid = STATE_TO_REGION_ID.get(state_code, 0) if state_code else 0
            climate_region_ids = np.array([crid] * static_cont.shape[0], dtype=np.int64)
        feeds['climate_region_ids'] = climate_region_ids.astype(np.int64)
    outputs = session.run(None, feeds)
    return {h: float(outputs[i].flatten()[0]) for i, h in enumerate(HAZARD_TYPES)}


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def _build_maps(hazard_df: pd.DataFrame) -> None:
    global _COUNTY_MAP, _STATE_MAP
    if 'county' in hazard_df.columns:
        counties = sorted(hazard_df['county'].unique())
        _COUNTY_MAP = {c: i % 250 for i, c in enumerate(counties)}
    if 'state' in hazard_df.columns:
        states = sorted(hazard_df['state'].unique())
        _STATE_MAP = {s: i for i, s in enumerate(states)}


# ---------------------------------------------------------------------------
# Tensor builder
# ---------------------------------------------------------------------------

def build_tensors_from_county_data(
    county_row: pd.Series,
    county_name: str = '',
    target_date: date = None,
    static_pad_dim: int = 50,
    temporal_seq_len: int = 14,
    temporal_feat_dim: int = 20,
    default_state: str = 'CO',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    static_values = []
    for col in STATIC_FEATURE_COLS:
        if col in county_row.index:
            val = county_row[col]
            try:
                static_values.append(float(val) if pd.notna(val) else 0.0)
            except (ValueError, TypeError):
                static_values.append(0.0)
        else:
            if col == 'day_of_year' and target_date is not None:
                static_values.append(float(target_date.timetuple().tm_yday))
            elif col == 'month' and target_date is not None:
                static_values.append(float(target_date.month))
            elif col == 'year' and target_date is not None:
                static_values.append(float(target_date.year))
            else:
                static_values.append(0.0)

    if target_date is not None:
        for i, col in enumerate(STATIC_FEATURE_COLS):
            if col == 'day_of_year':
                static_values[i] = float(target_date.timetuple().tm_yday)
            elif col == 'month':
                static_values[i] = float(target_date.month)
            elif col == 'year':
                static_values[i] = float(target_date.year)

    while len(static_values) < static_pad_dim:
        static_values.append(0.0)

    static_cont = np.array(static_values[:static_pad_dim], dtype=np.float32).reshape(1, -1)
    temporal = np.zeros((1, temporal_seq_len, temporal_feat_dim), dtype=np.float32)

    county_id = _COUNTY_MAP.get(county_name, 0) if county_name else 0
    state_name = (county_row.get('state', default_state)
                  if 'state' in county_row.index else default_state)
    state_id = _STATE_MAP.get(state_name, 0)

    region_ids = np.array([county_id], dtype=np.int64)
    state_ids  = np.array([state_id], dtype=np.int64)
    nlcd_ids   = np.array([0],         dtype=np.int64)
    return static_cont, temporal, region_ids, state_ids, nlcd_ids


# ---------------------------------------------------------------------------
# Public API: predict_county_risks_simple
# ---------------------------------------------------------------------------

def predict_county_risks_simple(
    state_code: str,
    region: str,
    county_name: str,
    hazard_df: pd.DataFrame,
    target_date: Optional[date] = None,
) -> Dict[str, float]:
    """Predict calibrated risk for a single county.

    Args:
        state_code: Two-letter state code, used to look up calibration JSONs.
        region:     Region folder name, used to look up the ONNX model.
        county_name: County to predict for.
        hazard_df:  State's inference parquet (states/XX/inference_data.parquet).
        target_date: Date for prediction (sets month/day_of_year features).

    Returns:
        {hazard: calibrated_probability} for all 5 hazards.
    """
    if hazard_df is not None and len(hazard_df) > 0:
        _build_maps(hazard_df)

    month = target_date.month if target_date is not None else 0
    county_upper = county_name.upper().replace(' COUNTY', '').strip()

    if hazard_df is not None and len(hazard_df) > 0 and 'county' in hazard_df.columns:
        mask = (
            hazard_df['county'].str.upper()
                               .str.replace(' COUNTY', '', regex=False)
                               .str.strip()
            == county_upper
        )
        rows = hazard_df[mask]
    else:
        rows = pd.DataFrame()

    if len(rows) == 0:
        print(f"[INFERENCE] {state_code}: county not found: {county_name}")
        return _generate_fallback_risks(county_name)

    if 'date' in rows.columns:
        rows = rows.sort_values('date', ascending=False)

    # Prefer same-month row for seasonally consistent weather features
    if target_date is not None and 'month' in rows.columns:
        same_month = rows[rows['month'] == target_date.month]
        county_row = same_month.iloc[0] if len(same_month) > 0 else rows.iloc[0]
    else:
        county_row = rows.iloc[0]
    actual_county = county_row.get('county', county_name)

    try:
        static_cont, temporal, region_ids, state_ids, nlcd_ids = \
            build_tensors_from_county_data(county_row, actual_county, target_date,
                                            default_state=state_code)
        logits = _run_onnx_inference(region, static_cont, temporal,
                                      region_ids, state_ids, nlcd_ids,
                                      state_code=state_code)
        if logits is None:
            return _generate_fallback_risks(county_name)
        return {h: _apply_calibration(state_code, logits[h], h, month,
                                       county_name=county_name)
                for h in HAZARD_TYPES}
    except Exception as e:
        print(f"[INFERENCE] {state_code}: error predicting for {county_name}: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_risks(county_name)


def predict_from_ahi_v2(
    state_code: str,
    region: str,
    static_cont, temporal, region_ids, state_ids, nlcd_ids,
    hazard_types=None,
    month: int = 0,
) -> Dict[str, float]:
    """Batch-friendly inference + calibration for a state."""
    hazard_types = hazard_types or HAZARD_TYPES

    for arr in (static_cont, temporal, region_ids, state_ids, nlcd_ids):
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()

    if hasattr(static_cont, 'numpy'):
        static_cont = static_cont.numpy()
        temporal    = temporal.numpy()
        region_ids  = region_ids.numpy()
        state_ids   = state_ids.numpy()
        nlcd_ids    = nlcd_ids.numpy()

    batch_size = static_cont.shape[0]
    session = _get_onnx_session(region)
    if session is None:
        if batch_size == 1:
            return {h: 0.0 for h in hazard_types}
        return {h: [0.0] * batch_size for h in hazard_types}

    feeds = {
        'static_cont': static_cont.astype(np.float32),
        'temporal':    temporal.astype(np.float32),
        'region_ids':  region_ids.astype(np.int64),
        'state_ids':   state_ids.astype(np.int64),
        'nlcd_ids':    nlcd_ids.astype(np.int64),
    }
    # Round 2 ONNX has climate_region_ids — derived from state_code (one per row)
    input_names = {i.name for i in session.get_inputs()}
    if 'climate_region_ids' in input_names:
        crid = STATE_TO_REGION_ID.get(state_code, 0)
        feeds['climate_region_ids'] = np.array([crid] * batch_size, dtype=np.int64)
    outputs = session.run(None, feeds)

    if batch_size == 1:
        return {h: _apply_calibration(state_code, float(outputs[i].flatten()[0]), h, month)
                for i, h in enumerate(HAZARD_TYPES) if h in hazard_types}

    risks = {h: [] for h in hazard_types}
    for i, h in enumerate(HAZARD_TYPES):
        if h in hazard_types:
            for raw in outputs[i].flatten():
                risks[h].append(_apply_calibration(state_code, float(raw), h, month))
    return risks


# ---------------------------------------------------------------------------
# Fallback (county not in dataset)
# ---------------------------------------------------------------------------

def _generate_fallback_risks(county_name: str) -> Dict[str, float]:
    try:
        seed = abs(hash(county_name)) % 10000
    except Exception:
        seed = 42
    rng = np.random.RandomState(seed)
    return {
        'fire':    float(rng.beta(2, 5)),
        'flood':   float(rng.beta(2, 8)),
        'wind':    float(rng.beta(2, 6)),
        'winter':  float(rng.beta(2, 5)),
        'seismic': float(rng.beta(1, 15)),
    }
