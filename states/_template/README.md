# State Onboarding Template

Each state directory under `states/` must contain:

| File | Source | Required |
|---|---|---|
| `config.yaml` | This template, hand-edited | Yes |
| `temperature_scales.json` | `scripts/temperature_scale_v2.py` (needs trained model) | Yes |
| `seasonal_bias.json` | `scripts/compute_calibration.py --state XX` | Yes |
| `base_rate_ceiling.json` | `scripts/compute_calibration.py --state XX` | Yes |
| `counties.geojson` | TIGER/Line, manually filtered to state | Yes |
| `inference_data.parquet` | State data pipeline | Yes |

## Workflow

```bash
# 1. Build state's clean labeled parquet (data pipeline lives in private hazard-lm repo)
python /path/to/hazard-lm/scripts/build_clean_labels.py --state XX

# 2. Bootstrap the state folder
python scripts/onboard_state.py \
  --state XX \
  --parquet /path/to/xx_clean_labeled.parquet \
  --geojson /path/to/xx_counties.geojson \
  --region <region>

# 3. Train or fine-tune the regional model (private repo)
#    Output: models/<region>/model.onnx

# 4. Fit temperature scales for the state
python /path/to/hazard-lm/scripts/temperature_scale_v2.py \
  --state XX \
  --output states/XX/temperature_scales.json

# 5. Hand-fill state-specific UI content in states/XX/config.yaml:
#    - county_utility (lookup state PUC service territory map)
#    - nws_offices (NWS county warning area map)
#    - state_agencies
#    - hazard_guidance (5 tiers x 5 hazards)
#    - season_notes / audit_factors

# 6. Flip deployment flag in states/registry.yaml
#    XX: { deployed: true }

# 7. Test locally:
streamlit run app.py
```
