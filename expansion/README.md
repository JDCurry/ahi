# AHI State Expansion Pipeline

This directory contains everything needed to train, calibrate, and deploy AHI models
for new states beyond Washington. Each state gets its own model checkpoint (`.pt`) and
feature dataset (`.parquet`), following the same architecture and training pipeline.

## Directory Structure

```
expansion/
├── README.md                  ← You are here
├── docs/
│   ├── data_sources.md        ← Federal vs. state data inventory
│   ├── calibration_guide.md   ← Temperature scaling, seasonal bias, ceilings
│   ├── feature_spec.md        ← Model input specification (21 features + embeddings)
│   └── expansion_checklist.md ← Step-by-step for adding a new state
├── templates/
│   ├── build_state_labels.py  ← Template label pipeline (adapt per state)
│   ├── state_config.yaml      ← Per-state configuration template
│   └── calibrate_state.py     ← Temperature scaling fitting script
```

## Per-State Deployment Structure

Each trained state produces:
```
states/{state_code}/
├── best_model.pt              ← Trained AHI v2 checkpoint
├── {state_code}_labeled.parquet  ← Feature dataset (inference + training)
├── temperature_scales.json    ← Fitted temperature scales
├── state_config.yaml          ← Seasonal penalties, ceilings, metadata
└── county_centroids.csv       ← County lat/lon for adjacency graph
```

## Quick Start

1. Read `docs/expansion_checklist.md` for the full process
2. Copy `templates/state_config.yaml` and fill in state-specific values
3. Run `templates/build_state_labels.py` to build the labeled dataset
4. Train using the private training pipeline (not included in public repo)
5. Run `templates/calibrate_state.py` to fit temperature scales
6. Deploy: add state folder to `states/` and update the dashboard

## Architecture

All states share the same AHI v2 Stacked Mesh architecture (1.29M parameters).
Per-state models are trained independently — the architecture is fixed, only the
weights and calibration parameters change per state.
