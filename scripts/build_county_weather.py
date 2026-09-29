"""
Build data/county_weather_by_month.parquet: the weather-driver panel's inputs
for every county and month, about 1.5 MB.

The National tab shows seven weather drivers when a county is clicked. Each is
the average of that county's daily values over every day of that month from
2000 through 2025 (735-806 days per county-month), read from the same dense
CONUS grid that scripts/precompute_v5.py predicts from, so the panel shows the
conditions behind the numbers above it. (The per-state inference files are not
used: Colorado's temperatures there are in C, which the panel would print as F.)

History: the panel used to show the first row of the month, the 1st of the
month in 2000, and loaded the state's whole inference parquet to get it
(Virginia: 1.26M rows, +822 MB), which is what pushed the Render service past
its 2 GB limit.

Re-run whenever the dense grid changes:

    python scripts/build_county_weather.py [--dense PATH]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'county_weather_by_month.parquet'
WEATHER = ['erc', 'vs', 'rmin', 'pr', 'tmmx', 'tmmn', 'vpd']
DENSE_DEFAULT = Path(os.environ.get(
    'AHI_DENSE_PARQUET',
    r'C:\Users\JDC\Desktop\hazard-lm\data\dense_CONUS_61feat_v2.parquet'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', type=Path, default=DENSE_DEFAULT)
    args = ap.parse_args()
    if not args.dense.exists():
        raise SystemExit(f'dense grid not found: {args.dense}\n'
                         'Pass --dense PATH or set AHI_DENSE_PARQUET.')

    df = pq.read_table(args.dense, columns=['state', 'county', 'date'] + WEATHER).to_pandas()
    df = df.drop_duplicates(['state', 'county', 'date'], keep='first')  # the grid has 138 duplicate keys
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    grouped = df.groupby(['state', 'county', 'month'], sort=True)
    out = grouped[WEATHER].mean().reset_index()
    span = grouped['year'].agg(['min', 'max', 'size']).reset_index(drop=True)
    out['first_year'], out['last_year'], out['n_days'] = span['min'], span['max'], span['size']

    for c in WEATHER:
        out[c] = out[c].astype('float32')
    out['month'] = out['month'].astype('int8')
    for c in ('first_year', 'last_year', 'n_days'):
        out[c] = out[c].astype('int16')
    out.to_parquet(OUT, index=False)
    print(f'wrote {OUT.relative_to(ROOT)}: {len(out):,} county-months, {out.state.nunique()} states, '
          f'{out.county.nunique():,} county names, {int(out.first_year.min())}-{int(out.last_year.max())}, '
          f'{int(out.n_days.min())}-{int(out.n_days.max())} days each, {OUT.stat().st_size / 1e6:.2f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
