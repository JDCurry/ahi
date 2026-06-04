#!/usr/bin/env python3
"""
Populate Round 4 lagged features (FIRMS/USGS/SPC) into state inference parquets.

For each state's inference_data.parquet, computes trailing-window aggregates
from the source county-daily files and z-scores them using train-set stats.

Adds 11 new columns (positions 50-60):
  firms_count_7d, firms_count_3d, firms_frp_max_7d, firms_frp_mean_7d,
  spc_severe_days_3d, spc_max_wind_3d,
  usgs_log_q_3d, usgs_log_q_delta, usgs_discharge_max_3d,
  usgs_n_stations, usgs_discharge_cv_3d

Usage:
    python scripts/populate_round4_features.py
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

AHI_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r'C:\Users\JDC\Desktop\round35\round35_training\data')

FIRMS_PATH = DATA_DIR / 'firms_county_daily.parquet'
USGS_PATH = DATA_DIR / 'usgs_county_daily.parquet'
SPC_PATH = DATA_DIR / 'spc_wind_county_daily.parquet'
FIPS_PATH = DATA_DIR / 'fips_to_county.csv'
NORM_STATS_PATH = Path(r'C:\Users\JDC\Desktop\round4\round4_norm_stats.json')

NEW_COLS = [
    'firms_count_7d', 'firms_count_3d', 'firms_frp_max_7d', 'firms_frp_mean_7d',
    'spc_severe_days_3d', 'spc_max_wind_3d',
    'usgs_log_q_3d', 'usgs_log_q_delta', 'usgs_discharge_max_3d',
    'usgs_n_stations', 'usgs_discharge_cv_3d',
]


def compute_trailing_agg(source_df: pd.DataFrame, target_dates: pd.DataFrame,
                         state: str, value_cols: dict, windows: list[int]) -> pd.DataFrame:
    """Compute trailing window aggregates from source county-daily data.

    value_cols: {source_col: {window: (output_col, agg_func)}}
    """
    src = source_df[source_df['state'] == state].copy()
    if len(src) == 0:
        return pd.DataFrame()

    src['date'] = pd.to_datetime(src['date'])
    results = target_dates[['county', 'date']].copy()

    for window in windows:
        lagged = []
        for lag in range(1, window + 1):
            shifted = src.copy()
            shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)
            lagged.append(shifted)
        all_lags = pd.concat(lagged, ignore_index=True)

        for src_col, (out_col, agg_func) in value_cols.items():
            if f'_{window}d' not in out_col:
                continue
            if src_col not in all_lags.columns:
                continue
            agg = all_lags.groupby(['county', 'date'])[src_col].agg(agg_func).reset_index()
            agg.columns = ['county', 'date', out_col]
            results = results.merge(agg, on=['county', 'date'], how='left')

    return results


def main():
    # Load normalization stats
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    print(f"Loaded normalization stats: {len(norm_stats)} features")

    # Load source data
    print("Loading source data...")
    firms = pd.read_parquet(FIRMS_PATH)
    firms['date'] = pd.to_datetime(firms['date'])
    firms['county'] = firms['county'].str.upper()
    print(f"  FIRMS: {len(firms):,} county-days")

    usgs = pd.read_parquet(USGS_PATH)
    usgs['date'] = pd.to_datetime(usgs['date'])
    usgs['county'] = usgs['county'].str.upper()
    print(f"  USGS: {len(usgs):,} county-days")

    spc = pd.read_parquet(SPC_PATH)
    spc['date'] = pd.to_datetime(spc['date'])
    fips_map = pd.read_csv(FIPS_PATH)
    fips_lookup = dict(zip(fips_map['county_fips'].astype(str).str.zfill(5),
                           fips_map['county_name']))
    spc['county'] = spc['county_fips'].map(fips_lookup)
    spc = spc.dropna(subset=['county'])
    spc['county'] = spc['county'].str.upper()
    # Rename SPC columns
    spc.rename(columns={'max_wind_kt': 'spc_max_wind', 'has_severe': 'spc_severe'}, inplace=True)
    print(f"  SPC: {len(spc):,} county-days")

    # Process each state
    states_dir = AHI_ROOT / 'states'
    state_dirs = sorted([d for d in states_dir.iterdir() if d.is_dir() and len(d.name) == 2])
    print(f"\nProcessing {len(state_dirs)} states...")

    for state_dir in state_dirs:
        sc = state_dir.name
        parquet_path = state_dir / 'inference_data.parquet'
        parts_dir = state_dir / 'inference_data'

        # Load state parquet (handle partitioned TX)
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif parts_dir.exists() and parts_dir.is_dir():
            df = pd.read_parquet(parts_dir)
        else:
            print(f"  {sc}: SKIP (no parquet)")
            continue

        df['date'] = pd.to_datetime(df['date'])
        df['county_upper'] = df['county'].str.upper()
        n_rows = len(df)

        # Drop existing new columns if re-running
        for c in NEW_COLS:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        # --- FIRMS trailing features ---
        firms_state = firms[firms['state'] == sc].copy()
        if len(firms_state) > 0:
            for window, suffix in [(7, '7d'), (3, '3d')]:
                lagged = []
                for lag in range(1, window + 1):
                    shifted = firms_state.copy()
                    shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)
                    lagged.append(shifted)
                all_lags = pd.concat(lagged, ignore_index=True)

                count_agg = all_lags.groupby(['county', 'date'])['firms_fire_count'].sum().reset_index()
                count_agg.columns = ['county', 'date', f'firms_count_{suffix}']
                df = df.merge(count_agg, left_on=['county_upper', 'date'],
                              right_on=['county', 'date'], how='left', suffixes=('', f'_f{suffix}'))
                if f'county_f{suffix}' in df.columns:
                    df.drop(columns=[f'county_f{suffix}'], inplace=True)

                if suffix == '7d':
                    frp_max = all_lags.groupby(['county', 'date'])['firms_frp_max'].max().reset_index()
                    frp_max.columns = ['county', 'date', 'firms_frp_max_7d']
                    df = df.merge(frp_max, left_on=['county_upper', 'date'],
                                  right_on=['county', 'date'], how='left', suffixes=('', '_fm7'))
                    if 'county_fm7' in df.columns:
                        df.drop(columns=['county_fm7'], inplace=True)

                    frp_mean = all_lags.groupby(['county', 'date'])['firms_frp_mean'].mean().reset_index()
                    frp_mean.columns = ['county', 'date', 'firms_frp_mean_7d']
                    df = df.merge(frp_mean, left_on=['county_upper', 'date'],
                                  right_on=['county', 'date'], how='left', suffixes=('', '_fp7'))
                    if 'county_fp7' in df.columns:
                        df.drop(columns=['county_fp7'], inplace=True)

        # --- SPC trailing features ---
        spc_state = spc[spc['state'] == sc].copy()
        if len(spc_state) > 0:
            lagged = []
            for lag in range(1, 4):
                shifted = spc_state.copy()
                shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)
                lagged.append(shifted)
            all_lags = pd.concat(lagged, ignore_index=True)

            severe_agg = all_lags.groupby(['county', 'date'])['spc_severe'].sum().reset_index()
            severe_agg.columns = ['county', 'date', 'spc_severe_days_3d']
            df = df.merge(severe_agg, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_ss'))
            if 'county_ss' in df.columns:
                df.drop(columns=['county_ss'], inplace=True)

            wind_agg = all_lags.groupby(['county', 'date'])['spc_max_wind'].max().reset_index()
            wind_agg.columns = ['county', 'date', 'spc_max_wind_3d']
            df = df.merge(wind_agg, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_sw'))
            if 'county_sw' in df.columns:
                df.drop(columns=['county_sw'], inplace=True)

        # --- USGS trailing features ---
        usgs_state = usgs[usgs['state'] == sc].copy()
        if len(usgs_state) > 0:
            # 3-day trailing
            lagged_3d = []
            for lag in range(1, 4):
                shifted = usgs_state.copy()
                shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)
                lagged_3d.append(shifted)
            all_3d = pd.concat(lagged_3d, ignore_index=True)

            q_mean = all_3d.groupby(['county', 'date'])['usgs_log_q_mean'].mean().reset_index()
            q_mean.columns = ['county', 'date', 'usgs_log_q_3d']
            df = df.merge(q_mean, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_uq'))
            if 'county_uq' in df.columns:
                df.drop(columns=['county_uq'], inplace=True)

            q_max = all_3d.groupby(['county', 'date'])['usgs_discharge_max'].max().reset_index()
            q_max.columns = ['county', 'date', 'usgs_discharge_max_3d']
            df = df.merge(q_max, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_um'))
            if 'county_um' in df.columns:
                df.drop(columns=['county_um'], inplace=True)

            n_st = all_3d.groupby(['county', 'date'])['usgs_n_stations'].max().reset_index()
            n_st.columns = ['county', 'date', 'usgs_n_stations']
            df = df.merge(n_st, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_un'))
            if 'county_un' in df.columns:
                df.drop(columns=['county_un'], inplace=True)

            cv = all_3d.groupby(['county', 'date'])['usgs_discharge_cv'].mean().reset_index()
            cv.columns = ['county', 'date', 'usgs_discharge_cv_3d']
            df = df.merge(cv, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_uc'))
            if 'county_uc' in df.columns:
                df.drop(columns=['county_uc'], inplace=True)

            # 7-day trailing for delta
            lagged_7d = []
            for lag in range(1, 8):
                shifted = usgs_state[['county', 'date', 'usgs_log_q_mean']].copy()
                shifted['date'] = shifted['date'] + pd.Timedelta(days=lag)
                lagged_7d.append(shifted)
            all_7d = pd.concat(lagged_7d, ignore_index=True)
            q7 = all_7d.groupby(['county', 'date'])['usgs_log_q_mean'].mean().reset_index()
            q7.columns = ['county', 'date', '_usgs_log_q_7d']
            df = df.merge(q7, left_on=['county_upper', 'date'],
                          right_on=['county', 'date'], how='left', suffixes=('', '_u7'))
            if 'county_u7' in df.columns:
                df.drop(columns=['county_u7'], inplace=True)

            df['usgs_log_q_delta'] = df['usgs_log_q_3d'].fillna(0) - df['_usgs_log_q_7d'].fillna(0)
            df.drop(columns=['_usgs_log_q_7d'], inplace=True, errors='ignore')

        df.drop(columns=['county_upper'], inplace=True)

        # Fill NaN and z-score normalize
        for c in NEW_COLS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)
            if c in norm_stats:
                mu = norm_stats[c]['mean']
                sigma = norm_stats[c]['std']
                if sigma > 1e-8:
                    df[c] = ((df[c] - mu) / sigma).astype(np.float32)
                else:
                    df[c] = np.float32(0.0)
            else:
                df[c] = df[c].astype(np.float32)

        assert len(df) == n_rows, f"{sc}: row count changed {n_rows} -> {len(df)}"

        # Save
        if parts_dir.exists() and parts_dir.is_dir():
            # TX partitioned — save as single file now (compressed)
            df.to_parquet(parquet_path, index=False, compression='zstd')
            print(f"  {sc}: {n_rows:,} rows, {parquet_path.stat().st_size/1024/1024:.1f} MB (merged from parts)")
        else:
            df.to_parquet(parquet_path, index=False, compression='zstd')
            print(f"  {sc}: {n_rows:,} rows, {parquet_path.stat().st_size/1024/1024:.1f} MB")

    print("\nDone! All state inference parquets now have 11 lagged trailing features.")


if __name__ == '__main__':
    main()
