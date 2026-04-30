# Adaptive Hazard Intelligence (AHI)

**Resilience Analytics Lab, LLC**
SBIR Phase I — Multi-Hazard Risk Intelligence for Emergency Management

---

## Overview

AHI is a machine learning system for multi-hazard risk prediction across Washington State counties. It produces calibrated daily risk probabilities for five hazard classes — wildfire, flood, wind, winter storm, and seismic — using a custom stacked mesh transformer architecture trained on historical weather, climate, and hazard event data.

The system is designed to support emergency management decision-making at the county and regional level, providing actionable risk tiers, audit-ready outputs, and a structured decision support interface.

---

## Live Dashboard

The AHI dashboard is deployed and accessible at:

**[https://ahi.onrender.com](https://ahi.onrender.com)**

The dashboard provides:

- County-level risk forecasts at 7-day and 14-day horizons
- Five-tier risk classification with operational guidance per hazard
- Statewide risk map across all 39 Washington State counties
- Decision audit panel with ranking context, contributing factors, and model trace metadata
- JSON export of audit records for downstream reporting

---

## Model Performance (v2.5)

| Hazard | AUC |
|---|---|
| Wildfire | 0.851 |
| Flood | 0.830 |
| Wind | 0.837 |
| Winter Storm | 0.908 |
| Seismic | 0.718 |
| **Mean** | **0.829** |

AHI v2.5 introduces a learned seasonal bias module (`LearnedSeasonalBias`) that replaces hardcoded seasonal penalty functions. The model independently discovers seasonal structure from data, improving mean AUC by 1.0 point over v2.0 and eliminating the need for manual per-hazard, per-state penalty configuration as the system scales.

---

## Architecture

AHI v2.5 uses a dual-branch stacked mesh architecture:

- **Temporal mesh** — three heat-kernel attention layers process recent weather history per county
- **Spatial mesh** — two standard attention layers process cross-county patterns using a geographic adjacency graph
- **Gated coupling** — a learned scalar gate weights the contribution of each branch at inference time
- **Calibration pipeline** — per-hazard temperature scaling, seasonal logit priors, and base-rate ceilings applied post-model

Total parameters: 1,294,547. Deployed via ONNX runtime — no PyTorch dependency in production.

Training uses date-grouped batching, where all 39 counties for a given date are processed together, enabling the spatial mesh to learn coherent cross-county signals.

---

## Data Sources

| Source | Coverage |
|---|---|
| NOAA GridMET | Daily weather (tmmx, tmmn, rmin, rmax, vs, erc, pr, vpd), 2000-2025 |
| WFIGS Wildland Fire Locations | Historical ignition records, WA state |
| USGS Earthquake Catalog | Seismic events, WA and Cascadia region |
| NOAA Storm Events Database | Flood, wind, winter storm declarations |
| FEMA Disaster Declarations | County-level disaster records |

Training dataset: 370,000 county-day records, 39 counties, 9,497 unique dates.

---

## Repository Contents

This repository contains the deployed dashboard and inference pipeline. The training pipeline, model architecture source, and PyTorch weights are maintained offline as proprietary assets.

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `inference_onnx.py` | ONNX runtime inference wrapper |
| `inference_core.py` | Calibration pipeline (temperature scaling, seasonal priors) |
| `outputs/ahi_v2/model.onnx` | Compiled deployment model |
| `temperature_scales.json` | Per-hazard calibration parameters |
| `data/` | County reference tables and WA GeoJSON |

---

## Tech Stack

- Python 3.11
- Streamlit — dashboard framework
- ONNX Runtime — model inference
- Plotly — interactive mapping and visualization
- Pandas / NumPy — data processing
- Deployed on Render (always-on web service)

---

## Project Context

AHI is a capstone research project developed at Pierce College under the Bachelor of Applied Science in Emergency Management program, in partnership with Resilience Analytics Lab, LLC. The system is being developed toward a DHS SBIR Phase I submission targeting operational nowcasting capabilities for multi-hazard situational awareness.

Phase I goal: integrate live NOAA/NWS weather feeds to transition from historical pattern detection to real-time risk inference across Washington State, with a pathway to nationwide deployment.

---

## License

MIT License. See `LICENSE` for details.

The model architecture, training pipeline, and trained weights are proprietary assets of Resilience Analytics Lab, LLC and are not included in this repository.
