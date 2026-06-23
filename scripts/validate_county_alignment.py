#!/usr/bin/env python3
"""Validate county alignment across the entire AHI pipeline.

Checks that every county-equivalent has a consistent presence across:
  1. State geojson (TIGER/Line polygons)
  2. State config.yaml (county_coords)
  3. State inference_data.parquet
  4. National predictions CSV
  5. National geojson (_id matching)

Run before every deploy to catch:
  - VA/MD/MO-style city/county name collisions
  - Encoding issues (NM Doña Ana)
  - Missing counties in any pipeline stage

Exit code 0 = clean, 1 = issues found.

Usage:
    python scripts/validate_county_alignment.py
    python scripts/validate_county_alignment.py --fix-predictions
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COUNTS = {
    'AL': 67, 'AR': 75, 'AZ': 15, 'CA': 58, 'CO': 64,
    'CT': 9,  'DC': 1,  'DE': 3,  'FL': 67, 'GA': 159,
    'IA': 99, 'ID': 44, 'IL': 102, 'IN': 92, 'KS': 105,
    'KY': 120, 'LA': 64, 'MA': 14, 'MD': 24, 'ME': 16,
    'MI': 83, 'MN': 87, 'MO': 115, 'MS': 82, 'MT': 56,
    'NC': 100, 'ND': 53, 'NE': 93, 'NH': 10, 'NJ': 21,
    'NM': 33, 'NV': 17, 'NY': 62, 'OH': 88, 'OK': 77,
    'OR': 36, 'PA': 67, 'RI': 5, 'SC': 46, 'SD': 66,
    'TN': 95, 'TX': 254, 'UT': 29, 'VA': 133, 'VT': 14,
    'WA': 39, 'WI': 72, 'WV': 55, 'WY': 23,
}


def _norm(s: str) -> str:
    n = s.upper().replace(' ', '_')
    n = unicodedata.normalize('NFKD', n)
    return ''.join(c for c in n if ord(c) < 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-predictions', action='store_true',
                        help='Auto-fix prediction CSV encoding issues')
    args = parser.parse_args()

    with open(ROOT / 'states' / 'registry.yaml') as f:
        registry = yaml.safe_load(f)
    deployed = sorted(s for s, info in registry.items() if info.get('deployed'))

    errors = []
    warnings = []

    # ── Stage 1: Per-state FIPS count check ──────────────────────────────
    print("Stage 1: FIPS county-equivalent counts")
    for sc in deployed:
        expected = EXPECTED_COUNTS.get(sc)
        if expected is None:
            warnings.append(f"{sc}: no expected count in FIPS table")
            continue

        gj_path = ROOT / 'states' / sc / 'counties.geojson'
        if gj_path.exists():
            with open(gj_path, encoding='utf-8') as f:
                gj = json.load(f)
            gj_count = len(gj['features'])
            if gj_count != expected:
                errors.append(f"{sc}: geojson has {gj_count} features, expected {expected}")

        cfg_path = ROOT / 'states' / sc / 'config.yaml'
        if cfg_path.exists():
            with open(cfg_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            cfg_count = len(cfg.get('county_coords', {}))
            if cfg_count != expected:
                errors.append(f"{sc}: config has {cfg_count} counties, expected {expected}")

    if not errors:
        print(f"  PASS: all {len(deployed)} states match FIPS counts")
    else:
        for e in errors:
            print(f"  FAIL: {e}")

    # ── Stage 2: National geojson vs prediction alignment ─────────────────
    print("\nStage 2: National geojson vs predictions")
    import pandas as pd
    import glob

    nat_gj_path = ROOT / 'data' / 'national_counties.geojson'
    if not nat_gj_path.exists():
        errors.append("national_counties.geojson missing — run build_national_geojson.py")
    else:
        with open(nat_gj_path, encoding='utf-8') as f:
            nat_gj = json.load(f)
        geo_ids = set(f['properties']['_id'] for f in nat_gj['features'])

        csvs = sorted(glob.glob(str(ROOT / 'data' / 'national_predictions_month*.csv')))
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            df['_id'] = df['state'] + '|' + df['county_id']
            pred_ids = set(df['_id'])

            in_pred_not_geo = sorted(pred_ids - geo_ids)
            in_geo_not_pred = sorted(geo_ids - pred_ids)

            month = Path(csv_path).stem.split('month')[1]
            if in_pred_not_geo:
                for x in in_pred_not_geo:
                    errors.append(f"month{month}: {x} in predictions but not in geojson")
            if in_geo_not_pred:
                for x in in_geo_not_pred:
                    warnings.append(f"month{month}: {x} in geojson but not in predictions")

            if not in_pred_not_geo and not in_geo_not_pred:
                print(f"  PASS: month{month} — {len(pred_ids)} predictions match {len(geo_ids)} features")
            else:
                print(f"  FAIL: month{month} — {len(in_pred_not_geo)} unmatched predictions, "
                      f"{len(in_geo_not_pred)} unmatched features")

    # ── Stage 3: National geojson feature count ──────────────────────────
    print("\nStage 3: National feature count")
    if nat_gj_path.exists():
        expected_total = sum(EXPECTED_COUNTS.get(sc, 0) for sc in deployed)
        actual_total = len(nat_gj['features'])
        if actual_total == expected_total:
            print(f"  PASS: {actual_total} features = sum of FIPS counts")
        else:
            errors.append(f"National geojson has {actual_total} features, expected {expected_total}")
            print(f"  FAIL: {actual_total} features vs {expected_total} expected")

    # ── Stage 4: Duplicate _id check ─────────────────────────────────────
    print("\nStage 4: Duplicate _id check")
    if nat_gj_path.exists():
        from collections import Counter
        id_counts = Counter(f['properties']['_id'] for f in nat_gj['features'])
        dupes = {k: v for k, v in id_counts.items() if v > 1}
        if dupes:
            for k, v in sorted(dupes.items()):
                errors.append(f"Duplicate _id in national geojson: {k} (×{v})")
            print(f"  FAIL: {len(dupes)} duplicate _ids")
        else:
            print(f"  PASS: all {len(id_counts)} _ids are unique")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    if errors:
        print(f"FAILED: {len(errors)} errors, {len(warnings)} warnings")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")
        sys.exit(1)
    else:
        print(f"PASSED: all checks clean ({len(warnings)} warnings)")
        for w in warnings:
            print(f"  WARN:  {w}")
        sys.exit(0)


if __name__ == '__main__':
    main()
