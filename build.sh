#!/usr/bin/env bash
# Render build script — installs deps, then checks that the committed
# national predictions were built from the current model files.
#
# Render Build Command:  ./build.sh
#
# How it works:
#   data/.model_hash stores the MD5 of all ONNX model files from the last
#   precompute run. On each build, we recompute the hash and compare:
#     - Hash matches  → deploy the committed CSVs
#     - Hash differs  → warn and still deploy the committed CSVs; they must be
#                       regenerated locally (see below), not on Render
#
# After a model update, run scripts/precompute_v5.py locally and commit the
# new CSVs + .model_hash.
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

# --- Smart precompute: only rebuild CSVs if models changed ---
HASH_FILE="data/.model_hash"

# Compute hash of all ONNX models (uses Python for cross-platform consistency)
CURRENT_HASH=$(python -c "
import hashlib, pathlib
h = hashlib.md5()
for f in sorted(pathlib.Path('models').rglob('*.onnx')):
    h.update(f.read_bytes())
print(h.hexdigest())
")

SAVED_HASH=""
if [ -f "$HASH_FILE" ]; then
    SAVED_HASH=$(cat "$HASH_FILE" | tr -d '[:space:]')
fi

if [ "$CURRENT_HASH" = "$SAVED_HASH" ]; then
    echo "=== Models unchanged ($CURRENT_HASH) — using committed predictions ==="
else
    # Predictions can't be rebuilt here: scripts/precompute_v5.py averages every
    # day of 2000-2025 from the dense CONUS grid, which lives in hazard-lm, not
    # in this repo. (This branch used to run precompute_national.py, the v4
    # engine, which overwrote the v5 files with the old model's numbers.)
    echo "=== WARNING: model files changed (was: $SAVED_HASH, now: $CURRENT_HASH) ==="
    echo "=== Serving the committed predictions. Regenerate them locally with"
    echo "===   python scripts/precompute_v5.py --all --jobs 4"
    echo "=== then commit data/national_predictions_month*.csv and data/.model_hash ==="
fi
