#!/usr/bin/env bash
# Render build script — installs deps, then conditionally precomputes
# national predictions if models or calibration files changed.
#
# Render Build Command:  ./build.sh
#
# How it works:
#   data/.model_hash stores the MD5 of all ONNX model files from the last
#   precompute run. On each build, we recompute the hash and compare:
#     - Hash matches  → skip precompute (fast deploy, ~2 min)
#     - Hash differs  → run precompute (rebuilds CSVs, ~30 min)
#
# After a model update, commit the new CSVs + .model_hash locally for
# fast deploys. If you forget, the build step catches it as a fallback.
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
    echo "=== Models unchanged ($CURRENT_HASH) — skipping precompute ==="
else
    echo "=== Models changed (was: $SAVED_HASH, now: $CURRENT_HASH) ==="
    echo "=== Rebuilding national predictions... ==="
    python scripts/precompute_national.py
    echo "=== Precompute complete ==="
fi
