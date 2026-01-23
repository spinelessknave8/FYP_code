# NTU FYP: Open-Set Defect Detection (Image-Level Only)

This repository provides a **minimal, reproducible** PyTorch codebase to compare two locked approaches for open-set defect detection at **image level only**:

- **Approach 1: Integrated Open-Set Recognition (One-Stage)**
  - ResNet50 classifier trained on **known defect classes**
  - Open-set detection by **Mahalanobis distance** on penultimate-layer embeddings
- **Approach 2: Cascaded Anomaly–Classification (Two-Stage)**
  - Stage 1: PatchCore anomaly detection trained on **MVTec AD good images** only
  - Stage 2: ResNet50 classifier on known defect classes

**What this project is NOT:**
- No bounding boxes, no segmentation, no pixel-level outputs
- No extra datasets, no synthetic data, no novel losses or architectures

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: if you need OpenCV, install it separately. Note that recent OpenCV wheels may require NumPy>=2.
```bash
pip install opencv-python
```

## Data Placement

See `data/README_DATA.md` for dataset layout and mapping.

## Configs and Splits

- `configs/default.yaml` contains global settings.
- `configs/neu_split_a.yaml`, `configs/neu_split_b.yaml`, `configs/neu_split_c.yaml` define known/unknown splits.

Each split has **known classes** (train/val/test) and **unknown classes** (test-only).

## How Thresholds are Calibrated

- **OSR distance threshold (tau)**: set on **known validation** embeddings so **95% are accepted**.
- **Classifier confidence threshold (kappa)**: optional; set on **known validation** logits so **95% are accepted**.
- **PatchCore anomaly threshold**: set on **MVTec good validation** so **95% are accepted**.

No test data is used for threshold calibration.

## Running Full Experiment

```bash
python run_all.py --config configs/default.yaml
```

This will:
- Train PatchCore once on MVTec good images (cached)
- For each NEU split (A/B/C):
  - Train ResNet50
  - Run OSR pipeline
  - Run cascade pipeline
  - Benchmark runtime
  - Write metrics and plots

## Running Individual Steps

```bash
python -m src.pipelines.train_classifier --config configs/neu_split_a.yaml
python -m src.pipelines.extract_embeddings --config configs/neu_split_a.yaml
python -m src.pipelines.run_osr --config configs/neu_split_a.yaml
python -m src.pipelines.train_patchcore --config configs/default.yaml
python -m src.pipelines.run_cascade --config configs/neu_split_a.yaml
python -m src.pipelines.benchmark_runtime --config configs/neu_split_a.yaml
```

## Outputs

Outputs are stored under `outputs/`:

```
outputs/
  split_a/
    classifier/
    embeddings/
    osr/
    cascade/
    plots/
    metrics.json
  split_b/
  split_c/
  summary.csv
```

## Known Limitations

- PatchCore is trained on **MVTec AD** and evaluated on **NEU** defects. This is a cross-domain anomaly detector by design.
- NEU contains only defective images; there are **no normal/no-defect images**. Cascade evaluation reports a mode that **ignores the `no_defect` outcome**.

## Notes on Reproducibility

- Fixed random seeds
- CPU-first design (MPS/CUDA optional if available)
- Minimal dependencies
