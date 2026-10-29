#!/usr/bin/env bash
# Render build: install deps, then check that the committed monthly
# predictions were built from the model artifacts in this commit.
#
# render.yaml runs this as `bash build.sh`.
#
# data/.model_hash is written by scripts/precompute_v5.py (one MD5 over the
# ONNX, XGBoost and calibrator files under models/v5/). On each build we
# recompute it and compare:
#   - match   -> the committed CSVs are current
#   - differ  -> warn and deploy the committed CSVs anyway. They cannot be
#                rebuilt here: precompute reads the dense CONUS grid, which
#                lives in hazard-lm, not in this repo.
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

HASH_FILE="data/.model_hash"
CURRENT_HASH=$(python scripts/model_hash.py)
SAVED_HASH=""
if [ -f "$HASH_FILE" ]; then
    SAVED_HASH=$(tr -d '[:space:]' < "$HASH_FILE")
fi

if [ "$CURRENT_HASH" = "$SAVED_HASH" ]; then
    echo "=== Model artifacts unchanged ($CURRENT_HASH) — committed predictions are current ==="
else
    echo "=== WARNING: model artifacts changed (was: $SAVED_HASH, now: $CURRENT_HASH) ==="
    echo "=== Serving the committed predictions. Regenerate them locally with"
    echo "===   python scripts/precompute_v5.py --all --jobs 4"
    echo "=== then commit data/national_predictions_month*.csv and data/.model_hash ==="
fi
