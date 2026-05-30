#!/usr/bin/env python3
"""
Compress inference_data.parquet files to fit under GitHub's 100 MB limit.
- Downcast float64 → float32
- Use zstd compression
- Report before/after sizes
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

STATES_DIR = Path(r'C:\Users\JDC\Documents\GitHub\ahi\states')
SIZE_WARN = 50  # MB
SIZE_LIMIT = 95  # MB target (leave 5 MB headroom under GitHub's 100 MB)

results = []

for state_dir in sorted(STATES_DIR.iterdir()):
    pq = state_dir / 'inference_data.parquet'
    if not pq.exists():
        continue

    st = state_dir.name
    orig_size = pq.stat().st_size / 1024 / 1024

    df = pd.read_parquet(pq)
    n_f64 = sum(df.dtypes == 'float64')

    # Downcast float64 → float32
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    # Also downcast int64 → int32 where safe
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype('int32')

    # Write with zstd compression
    df.to_parquet(pq, index=False, compression='zstd', compression_level=9)

    new_size = pq.stat().st_size / 1024 / 1024
    saved = orig_size - new_size
    pct = (saved / orig_size * 100) if orig_size > 0 else 0

    flag = ''
    if new_size > 100:
        flag = ' *** STILL OVER 100 MB ***'
    elif new_size > SIZE_WARN:
        flag = ' (warning >50MB)'

    results.append((st, orig_size, new_size, saved, pct, n_f64, len(df), flag))

    if orig_size > SIZE_WARN or n_f64 > 0:
        print(f"  {st}: {orig_size:.1f} -> {new_size:.1f} MB ({pct:.0f}% saved, {n_f64} f64->f32, {len(df):,} rows){flag}")

print(f"\n{'='*60}")
total_orig = sum(r[1] for r in results)
total_new = sum(r[2] for r in results)
print(f"Total: {total_orig:.0f} -> {total_new:.0f} MB ({total_orig-total_new:.0f} MB saved)")
over_100 = [r for r in results if r[2] > 100]
if over_100:
    print(f"\nSTILL OVER 100 MB: {', '.join(r[0] for r in over_100)}")
else:
    print(f"\nAll files under 100 MB — ready to push!")
