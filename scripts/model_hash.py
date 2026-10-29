"""
One MD5 over every artifact that changes the predictions: the attention ONNX,
the XGBoost boosters and the isotonic calibrators under models/v5/.

scripts/precompute_v5.py writes it to data/.model_hash after a run; build.sh
recomputes it on Render and warns when the committed CSVs were built from
different artifacts. Run directly to print the current hash.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ('models/v5/*.onnx', 'models/v5/xgb_*.json', 'models/v5/calibrator_*.json')


def compute(root: Path = ROOT) -> str:
    h = hashlib.md5()
    for pattern in PATTERNS:
        for f in sorted(root.glob(pattern)):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def write_model_hash(out_dir: Path) -> Path:
    path = out_dir / '.model_hash'
    path.write_text(compute() + '\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    sys.stdout.write(compute() + '\n')
