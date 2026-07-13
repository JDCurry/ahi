# Adaptive Hazard Intelligence (AHI)

**Resilience Analytics Lab, LLC**

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-ahi.run-4caf7d?style=flat-square)](https://ahi.run)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46e3b7?style=flat-square)](https://render.com)

[![AHI Dashboard](assets/dashboard.png)](https://ahi.run)

---

## Overview

AHI is a calibrated, multi-hazard risk prediction system for the contiguous United States. It produces daily county-level probabilities for four natural hazards — wildfire, flood, wind, and winter storm — using a hybrid engine (spatial attention + gradient boosting) trained on 25 years of weather, satellite, and hazard event data across 3,109 counties.

The system is designed for emergency management decision-making at the county and regional level, providing calibrated risk tiers, audit-ready outputs, and structured decision support. Probabilities are isotonic-calibrated so they mean what they say — not just relative rankings.

**Key properties:**
- Calibrated probabilities (not just rankings) via per-hazard isotonic regression, verified to reproduce evaluated AUCs within 0.001
- 62-feature input vector spanning weather, reanalysis, vegetation, terrain, flood zones, wildland-urban interface, satellite fire detections, streamflow, and severe wind reports
- Single CONUS-wide hybrid engine: spatial attention for wind/winter, XGBoost for fire/flood — engine selection by paired bootstrap on 2.95M test county-days
- FIRMS/SPC-validated labels: satellite fire detections and severe wind reports filter training labels to remove noise
- Mean AUC 0.898 across all four hazards (fire 0.878, flood 0.903, wind 0.869, winter 0.942)
- Runs on CPU via ONNX Runtime + XGBoost — no GPU required
- Live at [ahi.run](https://ahi.run)

---

## Architecture

AHI v5.0 uses a **hybrid engine** that combines two model families, with the best engine per hazard selected by paired bootstrap on 2,950,441 identical test county-days:

### Attention Engine (wind, winter)
- CONUS-wide spatial attention model with learned gated coupling between temporal and spatial signal
- Baked-in scaler, k-NN adjacency, and county/state embeddings
- Single ONNX graph processes all 3,109 counties in one forward pass
- Grounded in heat kernel attention, where attention weights follow the heat equation solution

### XGBoost Engine (fire, flood)
- Gradient-boosted trees on the same 62-feature matrix
- One model per hazard, no scaler needed (tree-based)

### Calibration
- Per-hazard isotonic regression maps raw probabilities to calibrated outputs
- Portable JSON lookup tables: `np.interp(p_raw, x_thresholds, y_thresholds)`
- Verified to reproduce evaluated AUCs within 0.001 tolerance

| Hazard | Engine | AUC |
|---|---|---|
| Fire | XGBoost | 0.878 |
| Flood | XGBoost | 0.903 |
| Wind | Attention | 0.869 |
| Winter | Attention | 0.942 |
| **Mean** | **Hybrid** | **0.898** |

---

## Data Sources

| Source | Features | Coverage |
|---|---|---|
| NOAA GridMET | Temperature, humidity, wind, precipitation, ERC, VPD, fire weather | Daily, 2000-2025 |
| ECMWF ERA5 | Integrated vapor transport, total column water vapor, surface pressure, wind gusts, 2m temperature | Daily, 2000-2025 |
| NASA MODIS (Terra/Aqua) | NDVI, EVI, vegetation anomalies | Monthly, 2000-2025 |
| NASA FIRMS | Satellite fire detections — trailing 3-day and 7-day fire count, FRP (lagged features + label validation) | Daily, 2000-2025 |
| USGS NWIS | Daily streamflow discharge — trailing 3-day mean, delta, max (lagged features) | Daily, 2000-2025 |
| NOAA SPC | Severe wind reports — trailing 3-day severe days, max wind speed (lagged features + label validation) | Daily, 2000-2025 |
| FEMA NFHL | Special flood hazard areas, V-zones, X-zones | Static |
| SILVIS WUI | Wildland-urban interface fractions, vegetation cover, housing density | Static |
| NOAA Storm Events | Flood, wind, winter storm, tornado records | 2000-2025 |
| WFIGS | Historical wildfire locations and perimeters (10+ acre threshold) | All CONUS states |
| FEMA Disaster Declarations | County-level disaster records | 2000-2025 |
| USGS Earthquake Catalog | Seismic events (model trained but hidden from UI) | 2000-2025 |

---

## Repository Contents

This repository contains the deployed dashboard and inference pipeline. The model architecture, training pipeline, and training data preparation are maintained as proprietary assets.

| Path | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `inference_v5.py` | v5 hybrid inference engine (ONNX + XGBoost + isotonic calibration) |
| `inference_onnx.py` | Legacy v4 ONNX inference engine (per-state calibration) |
| `state_context.py` | Per-state configuration loader |
| `models/v5/` | CONUS-wide v5 model artifacts (ONNX, XGBoost, calibrators, county order) |
| `states/<XX>/` | Per-state inference data, GeoJSON, and config |
| `states/registry.yaml` | State deployment status and region mapping |
| `data/` | National precomputed prediction CSVs |
| `scripts/` | Utility scripts (precompute_v5, state onboarding) |

---

## Tech Stack

- Python 3.11
- Streamlit — dashboard framework
- ONNX Runtime — attention model inference (CPU-only, no GPU required)
- XGBoost — gradient boosting inference
- Plotly — interactive mapping and visualization
- Pandas / NumPy — data processing
- Deployed on Render

---

## Research

AHI's architecture is grounded in published research on attention mechanisms and topological computation by Joshua D. Curry, available on SSRN:

- [Diffusion Attention: Replacing Softmax with Heat Kernel Dynamics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5953096)
- [Heat Kernel Attention: Provable Sparsity via Diffusion Dynamics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5959898)
- [Simplicial Computation: Topology as a Control Variable](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6037977)

---

## License

MIT License. See `LICENSE` for details.

The model architecture, training pipeline, and trained weights are proprietary assets of Resilience Analytics Lab, LLC and are not included in this repository.
