# AHI State Expansion Checklist

## Prerequisites
- [ ] Private training pipeline (`train_ahi_v2.py`) available locally
- [ ] GPU with ≥8GB VRAM (training) or CPU (inference only)
- [ ] Python environment with torch, pandas, scikit-learn, xgboost

---

## Phase 1: Data Collection

### Federal Sources (Download Once, Filter Per State)

These datasets are national. Download the full files and filter by state code.

- [ ] **NOAA Storm Events** — already nationwide in yearly CSVs
  - Files: `StormEvents_details_d{YYYY}.csv.gz`
  - URL: https://www.ncdc.noaa.gov/stormevents/ftp.jsp
  - Filter: `STATE == '{STATE_NAME}'`

- [ ] **WFIGS Wildland Fire Locations** — single nationwide file
  - File: `WFIGS_Wildland_Fire_Locations_Full_History.csv`
  - URL: https://data-nifc.opendata.arcgis.com/
  - Filter: `POOState == 'US-{STATE_CODE}'`

- [ ] **USGS Earthquakes** — query by state bounding box
  - API: `https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minlatitude=...`
  - Filter: magnitude ≥ 2.0, within state bounding box

- [ ] **FEMA Disaster Declarations** — single nationwide file
  - API: `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries`
  - Filter: `state == '{STATE_CODE}'`

- [ ] **NOAA GridMET** — download by state bounding box
  - URL: https://www.climatologylab.org/gridmet.html
  - Variables: tmmx, tmmn, rmin, rmax, vs, erc, pr, vpd
  - Aggregate 4km grid to county centroids

### State-Specific Data
- [ ] **County boundaries GeoJSON** — US Census TIGER/Line
  - URL: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
  - Or use `geopandas` with Census API

- [ ] **County centroids** — derive from GeoJSON (centroid lat/lon per county)

- [ ] **NWS Forecast Zone → County mapping**
  - Download zone shapefile for the state
  - Build ZONE_TO_COUNTIES dict (manual mapping required)
  - URL: https://www.weather.gov/gis/PublicZones

- [ ] **County population** — Census Bureau county population estimates
  - Used for `pop_density` feature only

- [ ] **Elevation per county** — USGS NED, aggregate to county mean

- [ ] **NLCD land cover per county** — forest_fraction, urban_fraction from NLCD

---

## Phase 2: Label Engineering

- [ ] Copy `templates/build_state_labels.py` to `states/{state_code}/`
- [ ] Update state filter (STATE name, POOState code)
- [ ] Build NWS zone-to-county mapping for the state
- [ ] Build county name normalization table
- [ ] Run label builder with **3-day window** (critical — do NOT use 30-day)
- [ ] Verify positive rates per hazard (should be 1–5% per hazard, not 90%+)
- [ ] Decide which hazards apply:
  - All states: fire, flood, wind
  - Cold states: + winter
  - Seismic states: + seismic (WA, CA, AK, HI, OK, SC, TN/MO, UT, NV)
  - Hurricane states: consider adding hurricane hazard head

---

## Phase 3: Feature Engineering

- [ ] Aggregate GridMET to county-day level (county centroid nearest grid point)
- [ ] Compute derived features: tmmx_3d_mean, pr_3d_mean, vs_3d_mean, red_flag_active
- [ ] Add static features: elevation, forest_fraction, urban_fraction, pop_density
- [ ] Assign county_id (sequential, starting after last used ID)
- [ ] Assign state_id (see feature_spec.md for reserved IDs)
- [ ] Merge labels with features on (county, date)
- [ ] Verify: every county present for every date (complete panel)
- [ ] Save as `states/{state_code}/{state_code}_labeled.parquet`

---

## Phase 4: Training

- [ ] Copy `state_config.yaml` template, fill in state-specific values
- [ ] Train AHI v2 with DateGroupedSampler (batch = all counties per date)
- [ ] Monitor: gate value should rise above 0.03 (spatial mesh contributing)
- [ ] If gate stays near 0: check that DateGroupedSampler is working correctly
- [ ] Early stopping on validation AUC (patience=7)
- [ ] Save best checkpoint to `states/{state_code}/best_model.pt`
- [ ] Record per-hazard AUC on test set

---

## Phase 5: Calibration

- [ ] Run inference on validation split → collect raw logits per hazard
- [ ] Run `templates/calibrate_state.py` to fit temperature T per hazard
- [ ] Manually verify predictions during known historical events
- [ ] If T crushes a good head (AUC > 0.75) → override to T=1.0
- [ ] Set seasonal bias from monthly event distributions
- [ ] Set base-rate ceilings from AUC performance
- [ ] Set weak-head bias for any head with AUC < 0.70
- [ ] Save `temperature_scales.json` and update `state_config.yaml`
- [ ] Run statewide predictions → sanity check all counties
- [ ] Document ALL manual overrides with justification

---

## Phase 6: Deployment

- [ ] Copy state folder to `states/{state_code}/` in deployment repo
- [ ] Update dashboard state selector to include new state
- [ ] Add county coordinates to WA_COUNTY_COORDS equivalent
- [ ] Test Quick Predict for 3+ counties in the new state
- [ ] Test Statewide Predictions
- [ ] Verify map renders correctly with state GeoJSON
- [ ] Push to deployment repo

---

## Validation Checklist (Before Going Live)

- [ ] No county shows > 50% risk for any hazard in normal conditions
- [ ] Seismic risk is < 5% for non-seismic states
- [ ] Seasonal penalties correctly suppress out-of-season hazards
- [ ] Known historical events produce elevated predictions when backtested
- [ ] All 5 (or fewer) hazard columns present in statewide export
- [ ] County names match official Census/FIPS names
- [ ] Download CSV works and contains all counties
