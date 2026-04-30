# AHI — Private / Proprietary Files

**Resilience Analytics Lab, LLC**
**Do not commit, share, or publish any file in this folder.**

This folder is gitignored and will never appear in the public repository.
All files here constitute confidential intellectual property under active
SBIR Phase I development.

---

## Contents

| File | Description | Why Private |
|------|-------------|-------------|
| `ahi_v2_model.py` | AHI v2.5 full model architecture | Core IP — stacked temporal/spatial mesh, heat-kernel attention, `LearnedSeasonalBias(5,12)`, gated coupling. This is the differentiating design. |
| `ahi_v2_graph.py` | Spatial graph construction | Defines county adjacency graph used by the spatial mesh layer. |
| `hazard_lm_diffusion.py` | HazardLM Diffusion architecture (v1) | Older architecture using heat kernel attention replacing softmax. Historical reference. |
| `export_onnx.py` | ONNX export pipeline | Converts trained `.pt` → `model.onnx` for deployment. Exposes training config. |
| `best_model.pt` | AHI v2.5 trained weights (Experiment D) | 15MB PyTorch state dict. Combined with the architecture above = fully reproducible model. |

---

## What IS public (in the repo)

- `model.onnx` — compiled deployment artifact, required for Render inference
- `inference_onnx.py` — ONNX runtime wrapper, no architecture details
- `inference_core.py` — calibration pipeline (temperature scaling, seasonal priors)
- `app.py` — Streamlit dashboard UI

The public repo demonstrates the product. The files above protect the method.

---

## Offline backup location

Full training workspace (including all epoch checkpoints, training scripts,
and experiment outputs) is stored at:

```
C:\Users\JDC\Desktop\hazard-lm\
C:\Users\JDC\Desktop\hazard-lm\ahi snapshot 3-12-26\
C:\Users\JDC\Desktop\hazard-lm\experiments\
```

`train_ahi_v2.py` and all training pipeline scripts are in the snapshot
folder and are NOT tracked in any GitHub repository.

---

## Git history note

These files were removed from public tracking in commit `0b0f541` (2026-04-30).
They may still exist in earlier git history. If formal IP protection requires
full history purge, use BFG Repo Cleaner or contact a git administrator to
rewrite history and force-push.

---

*Last updated: 2026-04-30*
