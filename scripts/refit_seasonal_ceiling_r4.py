#!/usr/bin/env python3
"""
Refit seasonal_bias.json and base_rate_ceiling.json for Round 4 labels.

Round 4 cleaned fire labels (9.8%->1.5%) and wind labels (6.4%->0.7%).
The old seasonal biases and ceilings were calibrated for the old rates
and will distort predictions if not updated.

This script:
1. Loads the Round 4 training parquet
2. Computes per-state, per-hazard, per-month base rates
3. Converts to logit-space seasonal biases
4. Computes per-state per-hazard ceilings (max monthly rate + margin)
5. Writes updated JSON files to each state folder

Author: Joshua D. Curry
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

AHI_ROOT = Path(r'C:\Users\JDC\Documents\GitHub\ahi')
PARQUET = Path(r'C:\Users\JDC\Desktop\round35\round35_training\data\round4_national.parquet')

HAZARDS = ['fire', 'flood', 'wind', 'winter', 'seismic']
DEPLOY_HAZARDS = ['fire', 'flood', 'wind', 'winter']
CEILING_MARGIN = 1.5  # ceiling = max_monthly_rate * margin


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    print("Loading Round 4 parquet...")
    df = pd.read_parquet(PARQUET, columns=['state', 'date'] +
                         [f'{h}_label' for h in HAZARDS])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    print(f"  {len(df):,} rows, {df['state'].nunique()} states")

    states = sorted(df['state'].unique())
    updated = 0

    for sc in states:
        state_dir = AHI_ROOT / 'states' / sc
        if not state_dir.exists():
            continue

        state_df = df[df['state'] == sc]
        n_total = len(state_df)

        # --- Seasonal bias ---
        biases = {}
        base_rates = {}

        for h in HAZARDS:
            col = f'{h}_label'
            overall_rate = state_df[col].mean()
            base_rates[h] = round(float(overall_rate), 6)

            monthly_rates = {}
            for m in range(1, 13):
                month_df = state_df[state_df['month'] == m]
                if len(month_df) > 0:
                    rate = month_df[col].mean()
                else:
                    rate = overall_rate
                monthly_rates[str(m)] = round(float(logit(rate)), 4)

            biases[h] = monthly_rates

        sb = {
            'state': sc,
            'description': 'Per-hazard monthly seasonal bias (logit space). Round 4 labels.',
            'method': 'logit(monthly_base_rate) from round4_national.parquet',
            'n_samples': int(n_total),
            'base_rates': base_rates,
            'biases': biases,
        }

        sb_path = state_dir / 'seasonal_bias.json'
        with open(sb_path, 'w') as f:
            json.dump(sb, f, indent=4)

        # --- Base rate ceiling ---
        ceilings = {}
        seasonal_ceilings = {}

        for h in DEPLOY_HAZARDS:
            col = f'{h}_label'
            monthly = state_df.groupby('month')[col].mean()
            max_monthly = float(monthly.max())
            ceiling = min(max_monthly * CEILING_MARGIN, 0.95)
            ceiling = max(ceiling, 0.01)  # minimum floor
            ceilings[h] = round(ceiling, 4)
            seasonal_ceilings[h] = {
                str(m): round(float(r * CEILING_MARGIN), 4)
                for m, r in monthly.items()
            }

        brc = {
            'state': sc,
            'description': 'Base rate ceiling per hazard. Round 4 labels.',
            'base_rate_ceiling': ceilings,
            'seasonal_ceiling': seasonal_ceilings,
        }

        brc_path = state_dir / 'base_rate_ceiling.json'
        with open(brc_path, 'w') as f:
            json.dump(brc, f, indent=4)

        # Summary
        fire_ceil = ceilings.get('fire', 0)
        wind_ceil = ceilings.get('wind', 0)
        flood_ceil = ceilings.get('flood', 0)
        print(f"  {sc}: fire_ceil={fire_ceil:.3f} wind_ceil={wind_ceil:.3f} "
              f"flood_ceil={flood_ceil:.3f} "
              f"fire_rate={base_rates.get('fire',0):.4f} "
              f"wind_rate={base_rates.get('wind',0):.4f}")
        updated += 1

    print(f"\nUpdated {updated} states")


if __name__ == '__main__':
    main()
