# Adaptive Hazard Intelligence (AHI)

**Resilience Analytics Lab, LLC**

---

## Overview

AHI is a calibrated, multi-hazard risk prediction system deployed across the contiguous United States. It produces daily risk probabilities for five hazard types — wildfire, flood, wind, winter storm, and seismic — at the county level using a proprietary deep learning architecture trained on 25 years of historical weather, climate, and hazard event data.

The system supports emergency management decision-making at the county and regional level, providing actionable risk tiers, audit-ready outputs, and a structured decision support interface.

---

## Live Dashboard

**[https://ahi.run](https://ahi.run)**

- County-level risk predictions across 3,109 CONUS counties
- Five-tier risk classification with operational guidance per hazard
- National overview with state and county drill-down
- Severity-weighted calibration reflecting actual event impact
- 9 regional models serving 48 states + DC

---

## Data Sources

| Source | Coverage |
|---|---|
| NOAA GridMET | Daily weather (temperature, humidity, wind, precipitation, fire weather), 2000–2025 |
| WFIGS Wildland Fire Locations | Historical wildfire records, all CONUS states |
| USGS Earthquake Catalog | Seismic events, national catalog (M ≥ 2.0) |
| NOAA Storm Events Database | Flood, wind, winter storm records |
| FEMA Disaster Declarations | County-level disaster records |

---

## Repository Contents

This repository contains the deployed dashboard and inference pipeline only. The model architecture, training pipeline, and training data preparation are maintained as proprietary assets.

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `inference_onnx.py` | Inference engine with per-state calibration |
| `models/` | Regional ONNX deployment models |
| `states/` | Per-state calibration and reference data |
| `data/` | National county reference data |

---

## Tech Stack

- Python 3.11
- Streamlit — dashboard framework
- ONNX Runtime — model inference
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
