# AHI Data Sources — Federal vs. State-Level

## Overview

AHI v2 uses **21 continuous features** plus county/state embeddings. The model does NOT
use Census demographics, Social Vulnerability Index (SVI), or any state-specific agency data.
All training features come from federal sources that cover all 50 states.

---

## Federal Data Sources (Reusable Nationwide)

These sources require no state-specific adaptation. The same download and processing
pipeline works for any US state.

### 1. NOAA Storm Events Database
- **URL:** https://www.ncdc.noaa.gov/stormevents/
- **Used for:** Hazard labels (fire, flood, wind, winter storm events)
- **Format:** CSV files (`StormEvents_details_*.csv`), one per year
- **Resolution:** County or NWS Forecast Zone
- **Coverage:** All US states, 1950–present
- **WA files:** 26 CSVs in `data/noaa_storms/`
- **State adaptation:** Filter `STATE` column. Zone-to-county mapping must be
  rebuilt per state (NWS forecast zones differ by state).
- **Label extraction:** Regex matching on EVENT_TYPE column:
  - Fire: `fire|wildfire|brush fire`
  - Flood: `flood|flash flood|coastal flood`
  - Wind: `thunderstorm wind|high wind|strong wind|tornado`
  - Winter: `winter storm|heavy snow|ice storm|blizzard|freezing`

### 2. NOAA GridMET (Weather)
- **URL:** https://www.climatologylab.org/gridmet.html
- **Used for:** 8 weather features + 3-day rolling means + red flag indicator
- **Features from GridMET:**

  | Feature | Description | Units |
  |---------|-------------|-------|
  | `tmmx` | Max temperature | °C |
  | `tmmn` | Min temperature | °C |
  | `rmin` | Min relative humidity | % |
  | `rmax` | Max relative humidity | % |
  | `vs` | Wind speed | m/s |
  | `erc` | Energy Release Component | index |
  | `pr` | Precipitation | mm |
  | `vpd` | Vapor Pressure Deficit | kPa |
  | `tmmx_3d_mean` | 3-day rolling mean of tmmx | °C |
  | `pr_3d_mean` | 3-day rolling mean of pr | mm |
  | `vs_3d_mean` | 3-day rolling mean of vs | m/s |
  | `red_flag_active` | Fire weather danger flag | 0/1 |

- **Resolution:** 4km grid, aggregated to county level
- **Coverage:** CONUS, 1979–present
- **State adaptation:** None — download by bounding box, aggregate to county centroids

### 3. WFIGS Wildland Fire Locations (USGS/NIFC)
- **URL:** https://data-nifc.opendata.arcgis.com/
- **Used for:** Fire event labels (supplements NOAA Storm Events)
- **Format:** CSV with POOState, POOCounty, discovery date
- **Coverage:** All US states, 2000–present
- **State adaptation:** Filter `POOState`. County field (`POOCounty`) may need
  normalization per state (naming conventions vary).

### 4. USGS Earthquake Catalog
- **URL:** https://earthquake.usgs.gov/fdsnws/event/1/
- **Used for:** Seismic event labels
- **Format:** CSV with magnitude, lat/lon, timestamp
- **Coverage:** Global (filter by state bounding box)
- **State adaptation:** Download by bounding box. Assign to nearest county by
  centroid distance. Minimum magnitude threshold (default: M2.0).
- **Note:** Seismic relevance varies enormously by state. States without significant
  seismic activity (e.g., most of the Midwest) may want to exclude this hazard entirely.

### 5. FEMA Disaster Declarations
- **URL:** https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries
- **Used for:** Supplemental hazard labels (all types)
- **Format:** JSON/CSV with disaster type, state, county, dates
- **Coverage:** All US states, 1953–present
- **State adaptation:** Filter by state. County names may need normalization.

### 6. Geographic/Land Cover Data
- **Used for:** Static features (`elevation`, `forest_fraction`, `urban_fraction`)
- **Sources:**
  - USGS National Elevation Dataset (NED): Elevation per county
  - USGS National Land Cover Database (NLCD): Forest/urban fractions
- **Coverage:** CONUS
- **State adaptation:** Extract by county boundaries. Available for all states.

### 7. Population Density
- **Used for:** `pop_density` static feature
- **Source:** US Census (county-level population / area)
- **Coverage:** All US states
- **State adaptation:** Simple lookup from Census county population tables

---

## Data NOT Used in Model Training

These sources appear in the AHI dashboard but are **not features in the v2 model**.
They are display-only for emergency manager context.

| Source | Dashboard Use | Model Use |
|--------|--------------|-----------|
| CDC/ATSDR Social Vulnerability Index (SVI) | County vulnerability scores, equity context | **NONE** |
| US Census (demographics) | Population, income, demographics display | **Only pop_density** |
| FEMA NFIP (flood insurance) | Claims history display | **NONE** |
| WUI (Wildland-Urban Interface) | Fire exposure display | **NONE** |
| NAIP Satellite Imagery | County thumbnail images | **NONE** |
| County school/facility data | Infrastructure display | **NONE** |

### Why SVI/Census Are Not Model Features

The AHI v2 model predicts **hazard occurrence** (will a fire/flood/wind event happen?),
not **hazard impact** (how many people will be affected?). Social vulnerability affects
impact severity but not whether an event occurs. Weather and geography drive event occurrence.

SVI and Census data could be integrated in a future "impact estimation" module (Phase II)
that combines AHI hazard probabilities with vulnerability overlays.

---

## State-Level Data That May Vary

When expanding to new states, these items need state-specific attention:

### NWS Forecast Zone Mapping
- NOAA Storm Events reports some events by NWS zone, not county
- WA uses 48 zone→county mappings (hardcoded in `build_clean_labels.py`)
- **Each state needs its own zone-to-county mapping table**
- Source: NWS zone shapefiles at https://www.weather.gov/gis/PublicZones

### County Naming Conventions
- NOAA, FEMA, WFIGS, and USGS all use slightly different county name formats
- Examples: "St. Louis" vs "Saint Louis", "DeKalb" vs "De Kalb"
- **Build a canonical county name mapping per state**

### Number of Counties
- Varies dramatically: 3 (Delaware) to 254 (Texas)
- Affects adjacency graph (k-NN parameter may need adjustment)
- Affects DateGroupedSampler batch size

### Hazard Relevance
- Not all 5 hazards apply to every state:
  - **Seismic:** Relevant in WA, CA, AK, HI, OK, SC. Minimal in most interior states.
  - **Winter:** Minimal in FL, HI, southern TX
  - **Fire:** Lower relevance in wet northeastern states
- States may need 3–4 hazard heads instead of 5
- Consider adding state-relevant hazards: hurricane (Gulf/Atlantic), tornado (Midwest)

---

## Data Volume Estimates Per State

Based on WA (39 counties, 25 years):

| Component | WA Size | Per State Estimate |
|-----------|---------|-------------------|
| Labeled parquet | 16 MB | 5–50 MB (scales with counties × years) |
| NOAA Storm Events | ~200 MB | ~200 MB (same national files) |
| GridMET weather | ~2 GB | ~2 GB per state (download once) |
| WFIGS fires | ~500 MB | ~500 MB (same national file) |
| USGS earthquakes | ~50 MB | ~50 MB per state (API query) |
| FEMA declarations | ~10 MB | ~10 MB (same national file) |
| County GeoJSON | 3.6 MB | 1–10 MB per state |
| **Total per state** | | **~3 GB raw → 16 MB trained** |

The raw federal files (NOAA Storm Events, WFIGS, FEMA) are shared across all states.
Only GridMET and the final labeled parquet are state-specific.
