"""
Rewrite base_rates in states/<XX>/seasonal_bias.json from the v5 training
labels: for each state, hazard, and month, the share of county-days from 2000
through 2025 on which that hazard's event label is set.

The State tab shows these as "Historical avg" and divides each county's
probability by them to pick its relative tier ("3x historical avg"). The
values they replace were one annual number per hazard, computed for AHI 4.0
from round4_national.parquet under different label definitions: Kansas flood
read 14.8% against 0.28% in the v5 labels, Kansas wind 0.95% against 8.0%, and
Colorado wind 0.0%. Monthly values also follow the seasons (Kansas wind: 1.6%
in January, 14.8% in July), which one annual number cannot.

Only base_rates changes. seismic keeps its stored value (the dense grid has no
seismic label, and seismic is hidden in the app); the rest of each file is
left as it is.

    python scripts/build_base_rates.py [--dense PATH]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HAZARDS = ['fire', 'flood', 'wind', 'winter']
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

    labels = [f'{h}_label' for h in HAZARDS]
    df = pq.read_table(args.dense, columns=['state', 'county', 'date'] + labels).to_pandas()
    df = df.drop_duplicates(['state', 'county', 'date'], keep='first')  # the grid has 138 duplicate keys
    df['month'] = df['date'].dt.month
    rates = df.groupby(['state', 'month'])[labels].mean()

    updated = []
    for state in sorted(rates.index.get_level_values('state').unique()):
        path = ROOT / 'states' / state / 'seasonal_bias.json'
        if not path.exists():
            print(f'  {state}: no seasonal_bias.json, skipped')
            continue
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        base = doc.get('base_rates', {})
        for h in HAZARDS:
            base[h] = {str(m): round(float(rates.loc[(state, m), f'{h}_label']), 6) for m in range(1, 13)}
        doc['base_rates'] = base
        doc['base_rates_source'] = (f'{args.dense.name} labels: share of county-days with an event, '
                                    'per month, 2000-2025 (scripts/build_base_rates.py)')
        path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
        updated.append(state)
    print(f'updated base_rates in {len(updated)} states')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
