#!/usr/bin/env python3
"""Rebuild national_counties.geojson from per-state geojsons.

Fixes:
  - VA/MD/MO county-vs-city disambiguation (LSAD 25 = independent city)
  - CT planning region suffix stripping
  - LA parish suffix handling
  - NM Doña Ana encoding normalization

Each feature gets a `_id` property: STATE|COUNTY_NAME or STATE|NAME_CITY.
The `_id` must match what precompute_national.py produces in `county_id`.

Usage:
    python scripts/build_national_geojson.py
"""
import json
import sys
import unicodedata
import yaml
from pathlib import Path
from shapely.geometry import shape, mapping

ROOT = Path(__file__).resolve().parents[1]


def _normalize_name(name: str) -> str:
    """Uppercase, underscored, ASCII-safe county name."""
    n = name.upper().replace(' ', '_')
    n = unicodedata.normalize('NFKD', n)
    n = ''.join(c for c in n if ord(c) < 128)
    return n


def _feature_id(state: str, feat: dict) -> str:
    """Compute the canonical _id for a geojson feature."""
    props = feat['properties']
    name = props.get('NAME', '')
    namelsad = props.get('NAMELSAD', '')
    lsad = props.get('LSAD', '')
    classfp = props.get('CLASSFP', '')

    base = _normalize_name(name)

    # Independent cities (VA, MD Baltimore city, MO St. Louis city)
    # Use LSAD=25 only — CLASSFP=C7 catches Carson City NV where
    # "City" is part of the actual name, not a disambiguation suffix
    if lsad == '25':
        base += '_CITY'

    # CT planning regions
    if state == 'CT' and base.endswith('_PLANNING_REGION'):
        base = base.replace('_PLANNING_REGION', '')

    # LA parishes: NAMELSAD = "Acadia Parish", LSAD = '15'
    if state == 'LA' and lsad == '15':
        base += '_PARISH'

    return f'{state}|{base}'


SIMPLIFY_TOLERANCE = 0.001  # ~111 m — visually lossless at choropleth zoom levels


def _simplify_geometry(geom):
    """Simplify polygon via Douglas-Peucker for smaller file size."""
    s = shape(geom).simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    return mapping(s)


def main():
    with open(ROOT / 'states' / 'registry.yaml') as f:
        registry = yaml.safe_load(f)
    deployed = sorted(s for s, info in registry.items() if info.get('deployed'))

    all_features = []
    id_counts = {}

    for sc in deployed:
        gj_path = ROOT / 'states' / sc / 'counties.geojson'
        if not gj_path.exists():
            print(f"  {sc}: SKIP (no geojson)")
            continue

        with open(gj_path, encoding='utf-8') as f:
            gj = json.load(f)

        count = 0
        for feat in gj['features']:
            _id = _feature_id(sc, feat)

            if _id in id_counts:
                print(f"  WARNING: duplicate _id {_id} in {sc}")
                id_counts[_id] += 1
                continue

            id_counts[_id] = 1

            slim = {
                'type': 'Feature',
                'properties': {
                    'NAME': feat['properties'].get('NAME', ''),
                    'STATE': sc,
                    '_id': _id,
                },
                'geometry': _simplify_geometry(feat['geometry']),
            }
            all_features.append(slim)
            count += 1

        print(f"  {sc}: {count} features")

    national = {
        'type': 'FeatureCollection',
        'features': all_features,
    }

    out_path = ROOT / 'data' / 'national_counties.geojson'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(national, f, separators=(',', ':'))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {len(all_features)} features to {out_path} ({size_mb:.1f} MB)")

    # FIPS count sanity check
    from validate_county_alignment import EXPECTED_COUNTS
    expected_total = sum(EXPECTED_COUNTS.get(sc, 0) for sc in deployed)
    if len(all_features) != expected_total:
        print(f"\nERROR: {len(all_features)} features != {expected_total} expected (FIPS)")
        sys.exit(1)
    print(f"FIPS check: {len(all_features)} features = expected total")

    # Verify against predictions
    try:
        import pandas as pd
        csv_path = ROOT / 'data' / 'national_predictions_month06.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['_id'] = df['state'] + '|' + df['county_id']
            pred_ids = set(df['_id'])
            geo_ids = set(id_counts.keys())
            missing_geo = sorted(pred_ids - geo_ids)
            missing_pred = sorted(geo_ids - pred_ids)
            if missing_geo:
                print(f"\nWARNING: {len(missing_geo)} prediction IDs not in geojson:")
                for x in missing_geo:
                    print(f"  {x}")
            if missing_pred:
                print(f"\nINFO: {len(missing_pred)} geojson IDs not in predictions:")
                for x in missing_pred:
                    print(f"  {x}")
            if not missing_geo and not missing_pred:
                print(f"\nAll {len(pred_ids)} prediction IDs match geojson features.")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
