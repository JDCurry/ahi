#!/usr/bin/env python3
"""Precompute national predictions for the current and next month.

Generates data/national_predictions_monthNN.csv for instant loading
on the National tab (avoids running ONNX inference on every page load).

Usage:
    cd ahi-platform
    python scripts/precompute_national.py
"""
import pandas as pd
import unicodedata
import yaml
import time
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_onnx import predict_county_risks_simple


def _normalize_county_id(state: str, county: str) -> str:
    """Normalize county name to match geojson _id conventions."""
    cid = county.upper().replace(' ', '_')
    # CT planning regions: geojson omits the _PLANNING_REGION suffix
    if state == 'CT' and cid.endswith('_PLANNING_REGION'):
        cid = cid.replace('_PLANNING_REGION', '')
    # Handle unicode (e.g., NM Doña Ana → DONA_ANA)
    cid = unicodedata.normalize('NFKD', cid)
    cid = ''.join(c for c in cid if ord(c) < 128)
    return cid


def _normalize_display_name(county: str) -> str:
    """Normalize county display name to safe ASCII for CSV/web rendering."""
    name = unicodedata.normalize('NFKD', county)
    return ''.join(c for c in name if ord(c) < 128)


def main():
    with open(ROOT / 'states' / 'registry.yaml') as f:
        registry = yaml.safe_load(f)
    deployed = sorted([s for s, info in registry.items() if info.get('deployed')])
    print(f"{len(deployed)} deployed states")

    today = date.today()
    cur_month = today.month
    next_month = cur_month + 1 if cur_month < 12 else 1
    next_year = today.year if next_month > cur_month else today.year + 1

    months = [
        (today.year, cur_month),
        (next_year, next_month),
    ]

    for year, month in months:
        print(f"\n=== {year}-{month:02d} ===")
        t0 = time.time()
        rows = []
        target = date(year, month, 15)

        for sc in deployed:
            region = registry[sc]['region']
            cfg_path = ROOT / 'states' / sc / 'config.yaml'
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            counties = sorted(cfg.get('county_coords', {}).keys())

            pq_path = ROOT / 'states' / sc / 'inference_data.parquet'
            parts_dir = ROOT / 'states' / sc / 'inference_data'
            if pq_path.exists():
                hdf = pd.read_parquet(pq_path)
            elif parts_dir.exists() and parts_dir.is_dir():
                hdf = pd.read_parquet(parts_dir)
            else:
                print(f"  {sc}: SKIP (no parquet)")
                continue

            ok = 0
            for county in counties:
                try:
                    risks = predict_county_risks_simple(sc, region, county, hdf, target)
                    if risks:
                        rows.append({
                            'state': sc,
                            'county': _normalize_display_name(county),
                            'county_id': _normalize_county_id(sc, county),
                            'fire_p': round(risks.get('fire', 0.0), 4),
                            'flood_p': round(risks.get('flood', 0.0), 4),
                            'wind_p': round(risks.get('wind', 0.0), 4),
                            'winter_p': round(risks.get('winter', 0.0), 4),
                            'seismic_p': round(risks.get('seismic', 0.0), 4),
                        })
                        ok += 1
                except Exception:
                    pass
            print(f"  {sc}: {ok}/{len(counties)}")

        df = pd.DataFrame(rows)
        hcols = [f'{h}_p' for h in ['fire', 'flood', 'wind', 'winter', 'seismic']]
        df['max_p'] = df[hcols].max(axis=1).round(4)
        df['max_hazard'] = df[hcols].idxmax(axis=1).str.replace('_p', '')

        out = ROOT / 'data' / f'national_predictions_month{month:02d}.csv'
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        elapsed = time.time() - t0
        print(f"\nMonth {month}: {len(df)} counties, "
              f"{out.stat().st_size / 1024:.0f} KB, {elapsed:.0f}s")


if __name__ == '__main__':
    main()
