#!/usr/bin/env python3
"""
Split inference_data.parquet files that exceed GitHub's 100 MB limit
into multiple parts. The load functions in app.py and precompute_national.py
handle automatic reassembly via pd.read_parquet on the directory.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

STATES_DIR = Path(r'C:\Users\JDC\Documents\GitHub\ahi\states')
MAX_SIZE_MB = 95  # target max per file

for state_dir in sorted(STATES_DIR.iterdir()):
    pq = state_dir / 'inference_data.parquet'
    if not pq.exists():
        continue

    st = state_dir.name
    size_mb = pq.stat().st_size / 1024 / 1024

    if size_mb <= MAX_SIZE_MB:
        continue

    print(f"\n{st}: {size_mb:.1f} MB - needs splitting")
    df = pd.read_parquet(pq)
    print(f"  {len(df):,} rows x {len(df.columns)} cols")

    # Calculate number of parts needed
    n_parts = int(np.ceil(size_mb / MAX_SIZE_MB)) + 1  # extra margin
    rows_per_part = len(df) // n_parts

    # Sort by date for clean splits
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    # Create parts directory
    parts_dir = state_dir / 'inference_data'
    parts_dir.mkdir(exist_ok=True)

    for i in range(n_parts):
        start = i * rows_per_part
        end = (i + 1) * rows_per_part if i < n_parts - 1 else len(df)
        part = df.iloc[start:end]
        part_path = parts_dir / f'part_{i}.parquet'
        part.to_parquet(part_path, index=False, compression='zstd', compression_level=9)
        part_size = part_path.stat().st_size / 1024 / 1024
        date_range = ''
        if 'date' in part.columns:
            date_range = f" ({part['date'].min()} to {part['date'].max()})"
        print(f"  Part {i}: {len(part):,} rows, {part_size:.1f} MB{date_range}")

    # Remove the single large file
    pq.unlink()
    print(f"  Removed single file, created {n_parts} parts in {parts_dir}")

    # Verify read-back
    df_back = pd.read_parquet(parts_dir)
    print(f"  Verified: pd.read_parquet(directory) returns {len(df_back):,} rows")
    assert len(df_back) == len(df), f"Row count mismatch: {len(df_back)} vs {len(df)}"

print("\nDone!")
