"""
AHI v5.0 — CONUS-wide hybrid inference engine.

Two engines serve four hazards:
  Attention ONNX  → wind, winter  (input: 3109×62 raw features → 3109×4 probs)
  XGBoost         → fire, flood   (input: 3109×62 raw features → 3109 probs each)

Calibration: isotonic regression via np.interp (portable JSON maps).

Usage:
    from inference_v5 import V5Engine
    engine = V5Engine()
    probs = engine.predict(feature_matrix)  # (3109, 4) calibrated
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HAZARDS = ['fire', 'flood', 'wind', 'winter']
ENGINE_MAP = {'fire': 'xgb', 'flood': 'xgb', 'wind': 'attn', 'winter': 'attn'}

ROOT = Path(__file__).resolve().parent
V5_DIR = ROOT / 'models' / 'v5'


class IsotonicCalibrator:
    __slots__ = ('x', 'y')

    def __init__(self, path: Path):
        with open(path) as f:
            d = json.load(f)
        self.x = np.array(d['x_thresholds'], dtype=np.float64)
        self.y = np.array(d['y_thresholds'], dtype=np.float64)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self.x, self.y)


class V5Engine:
    """Loads all v5 artifacts once; provides CONUS-wide batch prediction."""

    def __init__(self, model_dir: Optional[Path] = None):
        d = model_dir or V5_DIR
        self._load_contract(d)
        self._load_onnx(d)
        self._load_xgb(d)
        self._load_calibrators(d)

    def _load_contract(self, d: Path):
        with open(d / 'county_order.json') as f:
            contract = json.load(f)
        self.county_order: List[List[str]] = contract['counties']
        self.feature_cols: List[str] = contract['feature_cols']
        self.n_counties = len(self.county_order)
        self.n_features = len(self.feature_cols)
        self._county_lookup = {
            (s, c): i for i, (s, c) in enumerate(self.county_order)
        }

    def _load_onnx(self, d: Path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self._onnx = ort.InferenceSession(
            str(d / 'ahi_v5_attention.onnx'),
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )

    def _load_xgb(self, d: Path):
        import xgboost as xgb
        self._xgb = {}
        for h in ('fire', 'flood'):
            booster = xgb.Booster()
            booster.load_model(str(d / f'xgb_CONUS_{h}.json'))
            self._xgb[h] = booster

    def _load_calibrators(self, d: Path):
        self._cal = {
            h: IsotonicCalibrator(d / f'calibrator_{h}.json')
            for h in HAZARDS
        }

    def county_index(self, state: str, county: str) -> Optional[int]:
        return self._county_lookup.get((state, county.upper()))

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        """Run full hybrid inference + calibration.

        Args:
            raw_features: (n_counties, n_features) float32, canonical county order.

        Returns:
            (n_counties, 4) float64 calibrated probabilities [fire, flood, wind, winter].
        """
        X = raw_features.astype(np.float32)
        C = X.shape[0]
        out = np.zeros((C, 4), dtype=np.float64)

        # Attention ONNX → wind (col 2), winter (col 3)
        attn_probs = self._onnx.run(
            ['probs'], {'raw_features': X}
        )[0]  # (C, 4) — columns are [fire, flood, wind, winter]

        # XGBoost → fire (col 0), flood (col 1)
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        xgb_fire = self._xgb['fire'].predict(dm)
        xgb_flood = self._xgb['flood'].predict(dm)

        # Assemble pre-calibration: pick winning engine per hazard
        out[:, 0] = xgb_fire
        out[:, 1] = xgb_flood
        out[:, 2] = attn_probs[:, 2]  # wind
        out[:, 3] = attn_probs[:, 3]  # winter

        # Isotonic calibration per hazard
        for i, h in enumerate(HAZARDS):
            out[:, i] = self._cal[h](out[:, i])

        return out

    def predict_county(self, raw_features: np.ndarray,
                       state: str, county: str) -> Optional[Dict[str, float]]:
        """Predict for a single county from a full CONUS feature matrix."""
        idx = self.county_index(state, county)
        if idx is None:
            return None
        full = self.predict(raw_features)
        return {h: float(full[idx, i]) for i, h in enumerate(HAZARDS)}

    def predict_dict(self, raw_features: np.ndarray) -> List[Dict[str, float]]:
        """Predict all counties, return list of dicts."""
        full = self.predict(raw_features)
        return [
            {h: float(full[i, j]) for j, h in enumerate(HAZARDS)}
            for i in range(full.shape[0])
        ]
