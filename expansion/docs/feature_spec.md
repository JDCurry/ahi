# AHI v2 Feature Specification

## Model Input Dimensions

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| `static_cont` | (batch, 50) | float32 | 21 continuous features, zero-padded to 50 |
| `temporal` | (batch, 14, 20) | float32 | 14 timesteps × 20 features (currently zero-filled) |
| `region_ids` | (batch,) | int64 | County embedding index (mod 250) |
| `state_ids` | (batch,) | int64 | State embedding index (0–4) |
| `nlcd_ids` | (batch,) | int64 | Land cover class (currently unused, set to 0) |

## 21 Static Features (STATIC_FEATURE_COLS)

### Geographic/Temporal (5 features)
| # | Feature | Source | Description | Range |
|---|---------|--------|-------------|-------|
| 1 | `latitude` | County centroid | Decimal degrees | 45–49 (WA) |
| 2 | `longitude` | County centroid | Decimal degrees | -124 to -117 (WA) |
| 3 | `day_of_year` | Target date | Day 1–366 | 1–366 |
| 4 | `month` | Target date | Month 1–12 | 1–12 |
| 5 | `year` | Target date | Calendar year | 2000–2026 |

### Weather (8 features from NOAA GridMET)
| # | Feature | Source | Description | Units |
|---|---------|--------|-------------|-------|
| 6 | `tmmx` | GridMET | Daily max temperature | °C |
| 7 | `tmmn` | GridMET | Daily min temperature | °C |
| 8 | `rmin` | GridMET | Daily min relative humidity | % |
| 9 | `rmax` | GridMET | Daily max relative humidity | % |
| 10 | `vs` | GridMET | Daily mean wind speed | m/s |
| 11 | `erc` | GridMET | Energy Release Component (fire weather) | index |
| 12 | `pr` | GridMET | Daily total precipitation | mm |
| 13 | `vpd` | GridMET | Vapor Pressure Deficit (drought stress) | kPa |

### Derived Weather (4 features)
| # | Feature | Source | Description | Units |
|---|---------|--------|-------------|-------|
| 14 | `red_flag_active` | Derived | Fire weather danger flag (high temp + low humidity + wind) | 0/1 |
| 15 | `tmmx_3d_mean` | Derived | 3-day rolling mean of max temperature | °C |
| 16 | `pr_3d_mean` | Derived | 3-day rolling mean of precipitation | mm |
| 17 | `vs_3d_mean` | Derived | 3-day rolling mean of wind speed | m/s |

### Static Geography (4 features)
| # | Feature | Source | Description | Range |
|---|---------|--------|-------------|-------|
| 18 | `elevation` | USGS NED | Mean county elevation | meters |
| 19 | `forest_fraction` | USGS NLCD | Fraction of county area that is forest | 0.0–1.0 |
| 20 | `urban_fraction` | USGS NLCD | Fraction of county area that is urban/developed | 0.0–1.0 |
| 21 | `pop_density` | US Census | Population per square mile | varies |

## Temporal Features (Currently Dormant)

The model architecture accepts 14-timestep sequences of 20 features each (280 total).
These are designed for lagged weather sequences (`lag_1_tmmx`, `roll_7_pr`, `delta_3_vpd`, etc.)
but the current clean labeled dataset does not contain these columns.

**Current behavior:** Zero-filled at inference time. The temporal mesh processes these
zeros but the heat kernel attention learns to effectively ignore them.

**Phase I SBIR goal:** Populate with real-time NOAA/NWS weather feeds to activate the
temporal mesh for true multi-day forecasting.

## Categorical Embeddings

| Feature | Embedding Dim | Vocabulary | Notes |
|---------|--------------|------------|-------|
| `county_id` | 64 | 250 (mod) | Unique per county across all states |
| `state_id` | 32 | 5 | WA=0, neighbors reserved for multi-state |
| `nlcd_id` | 32 | 20 | NLCD land cover class (currently unused) |

### County ID Assignment for New States
County IDs are assigned sequentially and taken mod 250 for the embedding lookup.
When adding new states, assign IDs starting from the last WA county ID + 1.
The mod operation means IDs wrap at 250 — this is by design (embedding sharing
for counties with similar characteristics).

### State ID Assignment
| ID | State | Status |
|----|-------|--------|
| 0 | WA | Trained |
| 1 | OR | Reserved |
| 2 | CA | Reserved |
| 3 | ID | Reserved |
| 4 | AK | Reserved |

For states beyond the initial 5, expand the state embedding vocabulary in the model config.

## Label Specification

5 binary labels, one per hazard type:

| Label | Column | Positive Criteria | WA Base Rate (3-day window) |
|-------|--------|-------------------|-----------------------------|
| Fire | `fire_label` | NOAA fire event OR WFIGS wildfire within ±3 days | ~1.1% |
| Flood | `flood_label` | NOAA flood/flash flood event within ±3 days | ~0.9% |
| Wind | `wind_label` | NOAA high wind/thunderstorm wind within ±3 days | ~2.8% |
| Winter | `winter_label` | NOAA winter storm/snow/ice within ±3 days | ~4.3% |
| Seismic | `seismic_label` | USGS earthquake ≥M2.0 within ±3 days of county | ~3.0% |

### Critical: 3-Day Label Window
The label window determines how many days around an event count as "positive."

- **30-day window (WRONG):** Creates ~97.7% positive rate — massive halo effect
- **3-day window (CORRECT):** Produces realistic base rates matching operational relevance
- This was the most consequential data engineering decision in the entire project

## Parquet Schema

The training/inference parquet has exactly 32 columns:

```
Metadata:     date, county, state, state_id, county_id
Features:     latitude, longitude, day_of_year, month, year,
              tmmx, tmmn, rmin, rmax, vs, erc, pr, vpd,
              red_flag_active, tmmx_3d_mean, pr_3d_mean, vs_3d_mean,
              elevation, forest_fraction, urban_fraction, pop_density
Labels:       any_hazard, fire_label, flood_label, wind_label,
              winter_label, seismic_label
```

Row count = (number of counties) × (number of dates in training period).
WA: 39 counties × ~9,497 dates ≈ 370,383 rows.
