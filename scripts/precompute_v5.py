"""
Precompute national predictions using AHI v5.0 hybrid engine.

Generates one CSV per month: data/national_predictions_month{MM}.csv
The app loads these at startup — no live inference on Render.

Each county's value for a month is the average of the model's daily
predictions over every day of that month from 2000 through 2025 (about 800
days per county): the risk on a typical day in that month.

Input is the dense CONUS grid the v5 model was built from (hazard-lm's
dense_CONUS_61feat_v2.parquet: 3,109 counties x 9,497 days), not the
per-state states/*/inference_data.parquet files. Those were assembled by an
earlier pipeline and do not match what the model was trained on: Colorado's
temperatures are in C instead of K, streamflow inputs are scaled differently,
Connecticut and Louisiana have no wildland-urban-interface values, and no
state has era5_gust_mean. The model reads each of those as a real signal.

History: until 2026-09-28 this script predicted one day per county, the
first row of the month in those per-state files (the 1st of the month in
2000), so one day's conditions stood in for 26 years.

The attention model sees all 3,109 counties at once, so each day is predicted
as one CONUS-wide matrix, every county on the same date. A full 12-month run
is about 9,700 of those passes: roughly 25 minutes with --jobs 4 on a 20-core
machine, using about 5 GB of RAM per job.

Usage:
    python scripts/precompute_v5.py                    # current + next month
    python scripts/precompute_v5.py --months 1 2 3 4   # specific months
    python scripts/precompute_v5.py --all --jobs 4     # all 12 months, 4 at a time
    python scripts/precompute_v5.py --dense PATH ...   # grid elsewhere (or set AHI_DENSE_PARQUET)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_v5 import V5Engine, HAZARDS

DISPLAY_HAZARDS = ['fire', 'flood', 'wind', 'winter']
DENSE_DEFAULT = Path(os.environ.get(
    'AHI_DENSE_PARQUET',
    r'C:\Users\JDC\Desktop\hazard-lm\data\dense_CONUS_61feat_v2.parquet'))


def load_month(engine: V5Engine, month: int, dense: Path) -> pd.DataFrame:
    """Every county-day in `month`, across all years, with its county's index
    in the engine's canonical order. month and year come from the date, as in
    training."""
    derived = ('month', 'year')
    cols = ['state', 'county', 'date'] + [f for f in engine.feature_cols if f not in derived]
    table = ds.dataset(str(dense), format='parquet').to_table(
        columns=cols, filter=pc.equal(pc.month(ds.field('date')), month))
    df = table.to_pandas()
    df = df.drop_duplicates(['state', 'county', 'date'], keep='first')  # the grid has 138 duplicate keys
    df['month'] = df['date'].dt.month.astype(np.float32)
    df['year'] = df['date'].dt.year.astype(np.float32)
    df['idx'] = [engine.county_index(s, c) for s, c in zip(df['state'], df['county'])]
    if df['idx'].isna().any():
        missing = df.loc[df['idx'].isna(), ['state', 'county']].drop_duplicates()
        raise SystemExit(f'no county_order match for {missing.head().values.tolist()}')
    return df.drop(columns=['state', 'county'])


def predict_month(engine: V5Engine, onnx, month: int, dense: Path):
    """Mean calibrated daily probability per county, (n_counties, 4), plus the
    number of days averaged and the first and last date."""
    df = load_month(engine, month, dense)
    dates = np.sort(df['date'].unique())
    n_c, n_f, n_d = engine.n_counties, engine.n_features, len(dates)
    day = pd.Series(np.arange(n_d), index=dates)[df['date']].to_numpy()
    county = df['idx'].to_numpy().astype(int)

    # The attention model needs every county on every date
    present = np.zeros((n_d, n_c), dtype=bool)
    present[day, county] = True
    if not present.all():
        raise SystemExit(f'month {month}: {(~present).sum():,} county-days missing')

    X = np.zeros((n_d, n_c, n_f), dtype=np.float32)
    for j, feat in enumerate(engine.feature_cols):
        if feat not in df:
            raise SystemExit(f'model input {feat} is not in {dense.name}')
        X[day, county, j] = df[feat].fillna(0).to_numpy(np.float32)
    del df

    import xgboost as xgb
    raw = np.zeros((n_d, n_c, 4), dtype=np.float64)
    dm = xgb.DMatrix(X.reshape(-1, n_f))
    raw[:, :, 0] = engine._xgb['fire'].predict(dm).reshape(n_d, n_c)
    raw[:, :, 1] = engine._xgb['flood'].predict(dm).reshape(n_d, n_c)
    del dm
    for d in range(n_d):
        attn = onnx.run(['probs'], {'raw_features': X[d]})[0]
        raw[d, :, 2] = attn[:, 2]  # wind
        raw[d, :, 3] = attn[:, 3]  # winter
    total = np.zeros((n_c, 4), dtype=np.float64)
    for i, h in enumerate(HAZARDS):
        total[:, i] = engine._cal[h](raw[:, :, i]).sum(axis=0)
    return total / n_d, n_d, pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])


def write_month(engine: V5Engine, month: int, probs: np.ndarray, out_dir: Path) -> Path:
    rows = []
    for i, (state, county) in enumerate(engine.county_order):
        row = {
            'state': state,
            'county': county.title(),
            'county_id': county,
        }
        for j, h in enumerate(HAZARDS):
            row[f'{h}_p'] = round(float(probs[i, j]), 4)

        p_vals = [row[f'{h}_p'] for h in DISPLAY_HAZARDS]
        row['max_p'] = max(p_vals)
        row['max_hazard'] = DISPLAY_HAZARDS[p_vals.index(max(p_vals))]
        rows.append(row)

    csv_path = out_dir / f'national_predictions_month{month:02d}.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def run_month(month: int, out_dir: Path, threads: int, dense: Path) -> str:
    """One month start to finish; a separate process when --jobs > 1."""
    import onnxruntime as ort
    t0 = time.time()
    engine = V5Engine()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    onnx = ort.InferenceSession(str(ROOT / 'models' / 'v5' / 'ahi_v5_attention.onnx'),
                                sess_options=opts, providers=['CPUExecutionProvider'])
    for booster in engine._xgb.values():
        booster.set_param({'nthread': threads})
    probs, n_days, first, last = predict_month(engine, onnx, month, dense)
    csv_path = write_month(engine, month, probs, out_dir)
    means = '  '.join(f'{h} {probs[:, j].mean():.4f}' for j, h in enumerate(DISPLAY_HAZARDS))
    return (f'  Month {month:02d}: {n_days} days ({first:%Y-%m-%d} to {last:%Y-%m-%d}) x '
            f'{engine.n_counties} counties -> {csv_path.name} in {time.time() - t0:.0f}s\n'
            f'    mean daily probability: {means}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=1,
                    help='months to run at once (about 5 GB of RAM each)')
    ap.add_argument('--dense', type=Path, default=DENSE_DEFAULT,
                    help='dense CONUS grid (default: $AHI_DENSE_PARQUET or hazard-lm/data)')
    ap.add_argument('--out', type=Path, default=ROOT / 'data')
    args = ap.parse_args()
    if not args.dense.exists():
        raise SystemExit(f'dense grid not found: {args.dense}\n'
                         'Pass --dense PATH or set AHI_DENSE_PARQUET.')

    if args.all:
        months = list(range(1, 13))
    elif args.months:
        months = args.months
    else:
        now = datetime.now()
        cur = now.month
        nxt = cur + 1 if cur < 12 else 1
        months = sorted(set([cur, nxt]))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(args.jobs, len(months)))
    threads = max(1, (os.cpu_count() or 2) // jobs)
    print(f'Averaging daily predictions for months {months} from {args.dense.name}: '
          f'{jobs} at a time, {threads} threads each')

    if jobs == 1:
        for m in months:
            print(run_month(m, out_dir, threads, args.dense), flush=True)
    else:
        n = len(months)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for summary in pool.map(run_month, months, [out_dir] * n, [threads] * n, [args.dense] * n):
                print(summary, flush=True)

    print("\nDone. Precomputed CSVs ready for deployment.")


if __name__ == '__main__':
    main()
