#!/usr/bin/env python3
"""Populate county_coords and map_bbox in every state's config.yaml
from the inference_data.parquet.

Reads each state's parquet, extracts unique counties with mean lat/lon,
and patches the config.yaml in place.

Usage:
    cd ahi-platform
    python scripts/populate_county_coords.py
"""
from pathlib import Path
import yaml
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = ROOT / 'states'


def title_case_county(name: str) -> str:
    """APACHE -> Apache, DE KALB -> De Kalb, O'BRIEN -> O'Brien."""
    parts = name.strip().split()
    result = []
    for p in parts:
        if "'" in p:
            idx = p.index("'")
            result.append(p[:idx+1] + p[idx+1:].capitalize() if idx+1 < len(p) else p.capitalize())
        else:
            result.append(p.capitalize())
    return ' '.join(result)


def main():
    updated = 0
    skipped = []

    for state_dir in sorted(STATES_DIR.iterdir()):
        if not state_dir.is_dir():
            continue
        config_path = state_dir / 'config.yaml'
        parquet_path = state_dir / 'inference_data.parquet'

        if state_dir.name.startswith('_'):
            continue
        if not config_path.exists():
            continue
        if not parquet_path.exists():
            skipped.append(f"{state_dir.name}: no parquet")
            continue

        # Read config
        with open(config_path) as f:
            raw = f.read()

        # Load parquet
        df = pd.read_parquet(parquet_path)
        if 'county' not in df.columns or 'latitude' not in df.columns:
            skipped.append(f"{state_dir.name}: missing county/lat/lon columns")
            continue

        # Build county_coords: {Title Case Name: [lat, lon]}
        grouped = df.groupby('county').agg(
            lat=('latitude', 'mean'),
            lon=('longitude', 'mean'),
        )
        county_coords = {}
        for county_upper, row in grouped.iterrows():
            name = title_case_county(county_upper)
            county_coords[name] = [float(round(row['lat'], 4)), float(round(row['lon'], 4))]

        # Build map_bbox
        all_lats = [v[0] for v in county_coords.values()]
        all_lons = [v[1] for v in county_coords.values()]
        lat_pad = max(0.3, (max(all_lats) - min(all_lats)) * 0.08)
        lon_pad = max(0.3, (max(all_lons) - min(all_lons)) * 0.08)
        bbox = {
            'lon': [float(round(min(all_lons) - lon_pad, 2)), float(round(max(all_lons) + lon_pad, 2))],
            'lat': [float(round(min(all_lats) - lat_pad, 2)), float(round(max(all_lats) + lat_pad, 2))],
        }

        # Check if already populated (use unsafe_load to handle numpy artifacts from prior run)
        with open(config_path) as f:
            cfg = yaml.unsafe_load(f)
        existing = cfg.get('county_coords', {})
        # Check if already populated with CLEAN data (not numpy binary artifacts)
        if existing and len(existing) == len(county_coords):
            # Check raw YAML text for numpy artifacts
            if 'numpy' not in raw and 'python/object' not in raw:
                print(f"  {state_dir.name}: already has {len(existing)} counties, skipping")
                continue
            else:
                print(f"  {state_dir.name}: rewriting to clean numpy artifacts")

        # Patch the config: replace county_coords and map_bbox
        # Use yaml.dump for clean formatting
        cfg['county_coords'] = county_coords
        cfg['map_bbox'] = bbox

        # Also set default_county if it's still TODO or empty
        if not cfg.get('default_county') or cfg['default_county'] == 'TODO':
            # Pick the most populous or first alphabetically
            cfg['default_county'] = sorted(county_coords.keys())[0]

        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=120)

        print(f"  {state_dir.name}: {len(county_coords)} counties, "
              f"bbox lat=[{bbox['lat'][0]},{bbox['lat'][1]}] "
              f"lon=[{bbox['lon'][0]},{bbox['lon'][1]}]")
        updated += 1

    print(f"\nUpdated {updated} states.")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == '__main__':
    main()
