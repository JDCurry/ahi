"""
Precompute national predictions using AHI v5.0 hybrid engine.

Generates one CSV per month: data/national_predictions_month{MM}.csv
The app loads these at startup — no live inference on Render.

Usage:
    python scripts/precompute_v5.py                    # current + next month
    python scripts/precompute_v5.py --months 1 2 3 4   # specific months
    python scripts/precompute_v5.py --all               # all 12 months
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_v5 import V5Engine, HAZARDS

DISPLAY_HAZARDS = ['fire', 'flood', 'wind', 'winter']


def load_state_parquet(state_code: str) -> pd.DataFrame:
    single = ROOT / 'states' / state_code / 'inference_data.parquet'
    parts = ROOT / 'states' / state_code / 'inference_data'
    if single.exists():
        return pd.read_parquet(single)
    elif parts.exists() and parts.is_dir():
        return pd.read_parquet(parts)
    return pd.DataFrame()


def _normalize_county(name: str) -> str:
    """Normalize parquet county name to match county_order convention."""
    s = name.upper().strip()
    for suffix in (' COUNTY', ' PARISH', ' PLANNING REGION'):
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
    return s


def _build_parquet_to_order_map(engine: V5Engine, state_code: str,
                                parquet_names: list) -> dict:
    """Map parquet county names to county_order indices, handling suffixes."""
    result = {}
    order_names = {
        c: engine.county_index(state_code, c)
        for s, c in engine.county_order if s == state_code
    }

    for pname in parquet_names:
        norm = _normalize_county(pname)

        # Exact match after normalization
        if norm in order_names:
            result[pname] = order_names[norm]
            continue

        # VA independent cities: parquet has "ALEXANDRIA CITY",
        # order may have "ALEXANDRIA" (no suffix) or "ALEXANDRIA CITY" (kept)
        if norm.endswith(' CITY'):
            without = norm[:-5].strip()
            if without in order_names:
                result[pname] = order_names[without]
                continue

    return result


def build_feature_matrix(engine: V5Engine, month: int) -> np.ndarray:
    """Build (n_counties, n_features) matrix for a target month."""
    feature_cols = engine.feature_cols
    n_c = engine.n_counties
    X = np.zeros((n_c, engine.n_features), dtype=np.float32)

    states_needed = sorted(set(s for s, c in engine.county_order))
    filled = 0

    for state_code in states_needed:
        df = load_state_parquet(state_code)
        if len(df) == 0:
            continue

        if 'month' in df.columns:
            month_rows = df[df['month'] == month]
            if len(month_rows) == 0:
                month_rows = df
        else:
            month_rows = df

        parquet_names = list(month_rows['county'].unique())
        name_map = _build_parquet_to_order_map(engine, state_code, parquet_names)
        county_groups = month_rows.groupby('county')

        for county_name, group in county_groups:
            idx = name_map.get(county_name)
            if idx is None:
                continue

            row = group.iloc[0]
            for j, feat in enumerate(feature_cols):
                if feat in row.index:
                    val = row[feat]
                    X[idx, j] = float(val) if pd.notna(val) else 0.0
                elif feat == 'day_of_year':
                    X[idx, j] = 15.0 + (month - 1) * 30.4
                elif feat == 'month':
                    X[idx, j] = float(month)
                elif feat == 'year':
                    X[idx, j] = float(datetime.now().year)
            filled += 1

    print(f"  Month {month:02d}: {filled}/{n_c} counties populated "
          f"({n_c - filled} zero-filled)")
    return X


def precompute_month(engine: V5Engine, month: int, out_dir: Path):
    print(f"Precomputing month {month}...")
    X = build_feature_matrix(engine, month)
    probs = engine.predict(X)  # (3109, 4) calibrated

    rows = []
    for i, (state, county) in enumerate(engine.county_order):
        row = {
            'state': state,
            'county': county.title(),
            'county_id': county,
        }
        for j, h in enumerate(HAZARDS):
            row[f'{h}_p'] = round(float(probs[i, j]), 4)

        p_vals = [row[f'{h}_p'] for h in DISPLAY_HAZARDS]
        row['max_p'] = max(p_vals)
        row['max_hazard'] = DISPLAY_HAZARDS[p_vals.index(max(p_vals))]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / f'national_predictions_month{month:02d}.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path} ({len(df)} counties)")

    for h in DISPLAY_HAZARDS:
        mean_p = df[f'{h}_p'].mean()
        print(f"    {h:7s}: mean={mean_p:.4f}  max={df[f'{h}_p'].max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    if args.all:
        months = list(range(1, 13))
    elif args.months:
        months = args.months
    else:
        now = datetime.now()
        cur = now.month
        nxt = cur + 1 if cur < 12 else 1
        months = sorted(set([cur, nxt]))

    out_dir = ROOT / 'data'
    out_dir.mkdir(exist_ok=True)

    print("Loading v5 engine...")
    engine = V5Engine()
    print(f"  {engine.n_counties} counties, {engine.n_features} features")
    print(f"  Calibrators: {list(engine._cal.keys())}")

    for m in months:
        precompute_month(engine, m, out_dir)

    print("\nDone. Precomputed CSVs ready for deployment.")


if __name__ == '__main__':
    main()
