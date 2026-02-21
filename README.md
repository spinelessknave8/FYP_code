# NTU FYP: Open-Set Defect Detection (Image-Level Only)

This repository provides a **minimal, reproducible** PyTorch codebase to compare two locked approaches for open-set defect detection at **image level only**:

- **Approach 1: Integrated Open-Set Recognition (One-Stage)**
  - ResNet50 classifier trained on **known defect classes**
  - Open-set detection by **Mahalanobis distance** on penultimate-layer embeddings
- **Approach 2: Cascaded Anomaly–Classification (Two-Stage)**
  - Stage 1: PatchCore anomaly detection trained on **Severstal normal (no-defect) images**
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
By default, Stage-1 reads Severstal from `data/severstal` (`severstal.data_root` in config).

## Configs and Splits

- `configs/default.yaml` contains global settings.
- `configs/neu_split_a.yaml`, `configs/neu_split_b.yaml`, `configs/neu_split_c.yaml` define known/unknown splits.

Each split has **known classes** (train/val/test) and **unknown classes** (test-only).

## Pipeline Layout

- `src/pipelines/one_stage/` contains one-stage OSR pipeline:
  - `train_classifier.py`
  - `extract_embeddings.py`
  - `run_osr.py`
- `src/pipelines/two_stage/` contains two-stage cascade pipeline:
  - `train_patchcore.py`
  - `run_cascade.py`
  - `benchmark_runtime.py`
- `src/pipelines/*.py` top-level files are compatibility wrappers.

## How Thresholds are Calibrated

- **OSR distance threshold (tau)**: set on **known validation** embeddings so **95% are accepted**.
- **Classifier confidence threshold (kappa)**: optional; set on **known validation** logits so **95% are accepted**.
- **PatchCore anomaly threshold**: set on **held-out Severstal normal patches** so **95% are accepted**.

No test data is used for threshold calibration.

## Running Full Experiment

```bash
python run_all.py --config configs/default.yaml
```

This will:
- Train PatchCore once on Severstal normal patches (cached)
- For each NEU split (A/B/C):
  - Train ResNet50
  - Run OSR pipeline
  - Run cascade pipeline
  - Benchmark runtime
  - Write metrics and plots

## Running Individual Steps

```bash
python -m src.pipelines.one_stage.train_classifier --config configs/neu_split_a.yaml
python -m src.pipelines.one_stage.extract_embeddings --config configs/neu_split_a.yaml
python -m src.pipelines.one_stage.run_osr --config configs/neu_split_a.yaml
python -m src.pipelines.two_stage.train_patchcore --config configs/default.yaml
python -m src.pipelines.two_stage.run_cascade --config configs/neu_split_a.yaml
python -m src.pipelines.two_stage.benchmark_runtime --config configs/neu_split_a.yaml
```

## Running in Colab

- Notebook template: `notebooks/colab_two_stage_runner.ipynb`
- Notebook entrypoint helpers: `src/pipelines/notebook_entrypoints.py`
- The notebook calls existing pipeline functions directly:
  - `run_two_stage_stage1(...)`
  - `run_split_pipeline(...)`

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

- Two-stage Stage-1 uses Severstal normal-only filtering from `train.csv`; defective Severstal samples are excluded.
- NEU contains only defective images; there are **no normal/no-defect images**. Cascade evaluation reports a mode that **ignores the `no_defect` outcome**.

## Notes on Reproducibility

- Fixed random seeds
- CPU-first design (MPS/CUDA optional if available)
- Minimal dependencies
