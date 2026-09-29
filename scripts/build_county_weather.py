"""
Build data/county_weather_by_month.parquet: the weather-driver panel's inputs
for every county and month, about 1 MB.

The National tab shows seven weather drivers when a county is clicked. The app
used to read them by loading that state's whole inference parquet (Virginia:
1.26M rows, +822 MB), which is what pushed the Render service past its 2 GB
limit. Reading only the needed columns cut that to roughly 60 MB per new state,
but the process heap still kept it. This table removes the parquet read from
the request path altogether.

Semantics are unchanged: for each county and month it keeps the FIRST row in
file order, exactly as the old code picked `rows[month == m].iloc[0]`. It does
not average, so the numbers on screen stay identical.

Re-run whenever states/*/inference_data.parquet changes:

    python scripts/build_county_weather.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'county_weather_by_month.parquet'
WEATHER = ['erc', 'vs', 'rmin', 'pr', 'tmmx', 'tmmn', 'vpd']


def state_source(state_dir: Path):
    """A state's inference data: one file for most states, a folder of parts for
    the largest (Georgia, Texas). load_hazard_data() accepts both, so must this."""
    single = state_dir / 'inference_data.parquet'
    folder = state_dir / 'inference_data'
    if single.exists():
        return single
    if folder.is_dir() and any(folder.glob('*.parquet')):
        return folder
    return None


def part_files(path: Path):
    return [path] if path.is_file() else sorted(path.glob('*.parquet'))


def main() -> int:
    parts = []
    for state_dir in sorted(d for d in (ROOT / 'states').iterdir()
                            if d.is_dir() and not d.name.startswith('_')):
        state = state_dir.name
        path = state_source(state_dir)
        if path is None:
            continue
        have = set(pq.read_schema(part_files(path)[0]).names)
        cols = ['county', 'month'] + [c for c in WEATHER if c in have]
        df = pq.read_table(path, columns=cols).to_pandas()
        # first row per (county, month) in file order == the old .iloc[0]
        first = df.drop_duplicates(['county', 'month'], keep='first').copy()
        first.insert(0, 'state', state)
        parts.append(first)
        print(f'  {state}: {first.county.nunique():>4} counties, {len(first):>5} county-months')
    out = pd.concat(parts, ignore_index=True)
    for c in WEATHER:
        if c in out:
            out[c] = out[c].astype('float32')
    out['month'] = out['month'].astype('int8')
    out.to_parquet(OUT, index=False)
    print(f'\nwrote {OUT.relative_to(ROOT)}: {len(out):,} rows, '
          f'{out.state.nunique()} states, {OUT.stat().st_size / 1e6:.2f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
