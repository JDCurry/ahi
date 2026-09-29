"""
Build data/county_labels.csv: every county's name as the Census writes it,
keyed by its ID in the national map file. The National tab uses it for hover
labels and the detail panel's heading.

The predictions CSVs carry short, title-cased names (Acadia, Alexandria,
Capitol, Mcdonald, Prince George'S), and adding " County" to those mislabels
every Louisiana parish, every Connecticut planning region and 34 Virginia
independent cities. The Census long name (NAMELSAD in each state's
counties.geojson) is authoritative: Acadia Parish, Alexandria City, Capitol
Planning Region, James City County, McDonald County, Doña Ana County.

Keyed by national map ID rather than by CSV name because the CSVs do not agree
with each other: June's says 'Acadia Parish', October's says 'Acadia'. Every
month resolves to the same map ID.

Re-run whenever states/*/counties.geojson or data/national_counties.geojson
changes:

    python scripts/build_county_labels.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from county_names import build_index, census_label, display_name, resolve  # noqa: E402

OUT = ROOT / 'data' / 'county_labels.csv'


def main() -> int:
    national: dict[str, list[str]] = {}
    geojson = json.load(open(ROOT / 'data/national_counties.geojson', encoding='utf-8-sig'))
    for feat in geojson['features']:
        state, _, name = feat['properties']['_id'].partition('|')
        national.setdefault(state, []).append(name)

    rows, problems = [], []
    for state in sorted(national):
        path = ROOT / f'states/{state}/counties.geojson'
        if not path.exists():
            problems.append(f'{state}: no states/{state}/counties.geojson')
            continue
        index = build_index(national[state])
        for feat in json.load(open(path, encoding='utf-8-sig'))['features']:
            p = feat['properties']
            long_name, short = p.get('NAMELSAD'), p.get('NAME') or p.get('name')
            key = resolve(long_name or short, index)
            if key is None:
                problems.append(f'{state}: {long_name or short!r} has no shape in the national map')
                continue
            # Colorado's file has no NAMELSAD; all of its units are plain counties
            label = census_label(long_name) if long_name else display_name(short)
            rows.append({'_id': f'{state}|{key}', 'label': label})

    out = pd.DataFrame(rows)
    dup = [k for k, n in Counter(out['_id']).items() if n > 1]
    problems += [f'{k} is claimed by {out[out._id == k].label.tolist()}' for k in dup]
    missing = {f'{s}|{n}' for s, names in national.items() for n in names} - set(out['_id'])
    problems += [f'{k} got no label' for k in sorted(missing)]
    if problems:
        print(f'FAIL: {len(problems)} problems')
        for p in problems[:40]:
            print('  ', p)
        return 1
    out.sort_values('_id').to_csv(OUT, index=False, encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT)}: {len(out):,} labels, one per national map shape')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
