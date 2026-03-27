# AHI Calibration Guide

## Why Calibration Exists

The AHI v2 model outputs raw logits that are then converted to probabilities via sigmoid.
These raw probabilities are **not well-calibrated** — they do not match observed base rates.
For example, a county might show 40% flood probability when the historical base rate is 1.7%.

Calibration corrects this through four sequential transformations applied per hazard:

```
raw_logit
  → temperature scaling      (sharpen/soften confidence)
  → weak-head bias           (penalize unreliable hazard heads)
  → seasonal logit bias      (suppress out-of-season hazards)
  → sigmoid → probability
  → base-rate ceiling         (cap maximum output)
```

Each transformation is documented below with its rationale, current WA values,
and guidance for adapting to new states.

---

## 1. Temperature Scaling

### What It Does
Divides the raw logit by a learned temperature parameter T before applying sigmoid:
```
calibrated_logit = raw_logit / T
```
- **T < 1.0:** Sharpens predictions (pushes probabilities toward 0 or 1)
- **T = 1.0:** No change (raw model output)
- **T > 1.0:** Softens predictions (pushes toward 50%)

### How It's Fitted
Temperature T is optimized per hazard on a held-out validation set to minimize
Negative Log-Likelihood (NLL). This is standard Platt scaling with a single parameter.

### Current WA Values (`temperature_scales.json`)
| Hazard | T | Effect | Justification |
|--------|---|--------|---------------|
| Fire | 0.436 | Moderate sharpening | Model overestimates fire broadly; sharpening concentrates on true fire-prone counties |
| Flood | 1.000 | **No sharpening** | Raw logits already negative for most counties; sharpening crushes to 0%. T=1.0 preserves signal. |
| Wind | 0.446 | Moderate sharpening | Similar to fire — model spreads probability too broadly |
| Winter | 0.660 | Mild sharpening | Best-calibrated head; needs minimal correction |
| Seismic | 0.372 | Strong sharpening | Model outputs are diffuse; needs concentration |

### Why Flood T=1.0 (Not Fitted)
When T was fitted on the v2 validation set, the optimal flood T was 0.272. However,
this crushed all flood predictions to near-zero because the model's raw flood logits
are clustered near or below zero. With T=0.272, even slightly negative logits produce
probabilities < 0.1%. During an active flood warning in Snohomish County (March 2026),
the dashboard showed 0.1% flood risk — clearly wrong.

**Decision (2026-03-27):** Set flood T=1.0 to preserve the model's discrimination signal.
The v2 flood head has AUC 0.818 — it ranks counties correctly, but the absolute
probabilities need to pass through without aggressive sharpening.

### Guidance for New States
- **Always re-fit T per hazard on the new state's validation set**
- After fitting, manually verify predictions during known events
- If a fitted T crushes a well-performing head (AUC > 0.75), override to T=1.0
- Document any overrides in the state's `state_config.yaml`

---

## 2. Weak-Head Bias

### What It Does
Applies a fixed negative logit bias to hazard heads where the model has poor discrimination
(low AUC), pulling overconfident predictions toward realistic base rates.

```
adjusted_logit = calibrated_logit + bias  (bias < 0)
```

### Current WA Values
| Hazard | Bias | v2 AUC | Justification |
|--------|------|--------|---------------|
| Fire | 0.0 | 0.848 | Strong head — no penalty |
| Flood | 0.0 | 0.818 | Strong head — no penalty |
| Wind | 0.0 | 0.823 | Strong head — no penalty |
| Winter | 0.0 | 0.904 | Best head — no penalty |
| Seismic | -2.5 | 0.703 | Weakest head; geographic-only signal. Bias prevents constant elevated predictions near fault lines |

### Why Seismic Gets Penalized
The seismic head's "discrimination" is mostly geographic (counties near the Cascadia
subduction zone always score higher). It doesn't vary day-to-day because no weather
features predict earthquakes. Without the bias, every western WA county shows 10-15%
seismic risk permanently, which:
- Competes with active weather hazards for attention
- Undermines emergency manager trust in the tool
- Doesn't represent actionable daily risk

### Guidance for New States
- **Re-evaluate per hazard after training:** Only apply bias where AUC < 0.70
- Seismic will need bias in most states (model can't predict earthquakes from weather)
- States without seismic relevance should either exclude the head or set ceiling to 2%
- If a hazard head has AUC < 0.55, consider excluding it entirely rather than biasing

---

## 3. Seasonal Logit Bias

### What It Does
Adds a month-specific logit offset to suppress out-of-season hazards. This encodes
domain knowledge that certain hazards are physically impossible or extremely rare in
certain months, regardless of what the model outputs.

```
seasonal_logit = calibrated_logit + seasonal_bias[hazard][month]
```

### Current WA Values

**Fire** (peak: June–September):
| Month | Bias | Rationale |
|-------|------|-----------|
| Jan | -3.0 | Snow cover, no ignition sources |
| Feb | -2.5 | Still winter |
| Mar | -2.0 | Occasional early grass fires |
| Apr | -1.0 | Drying begins |
| May | -0.5 | Pre-season |
| Jun–Sep | 0.0 | **Peak fire season** |
| Oct | -0.5 | Season winding down |
| Nov | -2.0 | Wet season begins |
| Dec | -3.0 | Snow cover |

**Winter Storm** (peak: November–March):
| Month | Bias | Rationale |
|-------|------|-----------|
| Jan–Mar | 0.0 | **Peak winter** |
| Apr | -0.5 | Transitioning |
| May | -1.5 | Rare |
| Jun–Aug | -3.0 | **Physically impossible** |
| Sep | -2.0 | Very rare |
| Oct | -0.5 | Season starting |
| Nov–Dec | 0.0 | **Peak winter** |

**Wind** (year-round, peak October–March):
| Month | Bias | Rationale |
|-------|------|-----------|
| Jan–Mar | 0.0 | Storm season |
| Apr | -0.3 | Slightly less |
| May–Aug | -0.5 | Convective winds still occur but major events rare |
| Sep–Dec | 0.0 | Storm season |

**Flood:** 0.0 all months (WA floods year-round: snowmelt spring, atmospheric rivers fall/winter)

**Seismic:** 0.0 all months (no seasonal pattern)

### Adapting for New States

Seasonal penalties are the most state-specific calibration parameter. Examples:

| State | Fire Season | Winter Season | Notes |
|-------|------------|---------------|-------|
| WA | Jun–Sep | Nov–Mar | Pacific NW marine climate |
| CA | May–Nov | Dec–Feb | Extended fire season, Santa Ana winds Oct–Dec |
| FL | Jan–May | N/A (rare) | Winter is dry season = fire. Add: hurricane Jun–Nov |
| CO | Jun–Sep | Oct–Apr | High-altitude winter, afternoon summer thunderstorms |
| TX | Year-round | Dec–Feb | Fire risk in drought periods across all seasons |

**Process for setting seasonal bias:**
1. Compute monthly event rates from the state's labeled dataset
2. Identify months with < 1% of annual events → set bias -2.0 to -3.0
3. Identify months with < 5% of annual events → set bias -0.5 to -1.5
4. Peak season months → bias 0.0
5. Validate against known event history (major fires, floods, etc.)

---

## 4. Base-Rate Ceiling

### What It Does
Caps the maximum probability output per hazard. Prevents the model from ever showing
(for example) 80% flood risk, which would undermine credibility even if the model is
confident.

```
final_prob = min(calibrated_prob, ceiling)
```

### Current WA Values
| Hazard | Ceiling | Justification |
|--------|---------|---------------|
| Fire | 0.35 | Even in peak fire season, individual county-day risk < 35% |
| Flood | 0.35 | Strong v2 head (AUC 0.818), can trust higher predictions |
| Wind | 0.25 | Good head (AUC 0.823), moderate ceiling |
| Winter | 0.35 | Best head (AUC 0.904), trustworthy |
| Seismic | 0.05 | Constant background risk — cap at 5% to prevent dominating display |

### Guidance for New States
- Set ceiling based on the trained head's AUC:
  - AUC > 0.85: ceiling 0.35
  - AUC 0.75–0.85: ceiling 0.25
  - AUC 0.65–0.75: ceiling 0.15
  - AUC < 0.65: ceiling 0.08 (or exclude the head)
- Seismic ceiling should be 0.05 unless the state has frequent significant seismic activity (CA, AK)
- Always verify: run statewide predictions and confirm no single county shows risk that
  would alarm an EM without justification

---

## Calibration Workflow for New States

```
1. Train model on state dataset
2. Run inference on validation split → collect raw logits
3. Fit temperature T per hazard (minimize NLL)
4. Manually verify: predict during known events
   - If T crushes a good head → override to T=1.0
   - If T leaves a bad head overconfident → add weak-head bias
5. Set seasonal bias from monthly event distributions
6. Set base-rate ceilings from AUC performance
7. Save to state_config.yaml and temperature_scales.json
8. Run statewide predictions → sanity check all counties
9. Document all manual overrides with justification
```

---

## Transparency for SBIR Reviewers

The calibration pipeline is a **post-hoc correction layer**, not a modification of the
model itself. The AHI v2 model weights are frozen after training — calibration only
transforms the output probabilities.

**Why this matters:**
- The model's **discrimination** (ranking counties correctly) is fixed at training time
- Calibration only adjusts **absolute probability values** to match operational expectations
- This is standard practice in deployed ML systems (weather forecasting, medical diagnostics)
- All calibration parameters are documented, reproducible, and version-controlled

**What the model CANNOT do without live data:**
- Detect that there is currently an active flood warning
- Respond to today's actual weather conditions
- Predict specific events (it estimates probability windows)

**What it CAN do with historical patterns:**
- Identify which counties are most flood/fire/wind-prone for a given month
- Rank relative risk across all 39 counties
- Flag seasonal transitions (e.g., spring snowmelt → flood risk)
- Provide a defensible baseline for resource pre-positioning
