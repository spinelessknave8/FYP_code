# Severstal OSR

This folder contains a standalone open-set defect pipeline using **Severstal only**:
- Stage 1: PatchCore anomaly gate (trained on Severstal normal images)
- Stage 2: ResNet-50 known-defect classifier (trained on Severstal single-label known classes)
- OSR / Cascade evaluation on held-out unknown defect classes

## Folder Layout
- `configs/`: default and split configs
- `src/`: pipeline scripts
- `notebooks/severstral_osr_full_experiment.ipynb`: end-to-end Colab notebook
- `exports/`: CSV exports for defect lists and counts

## Quick Start (local)

```bash
python severstral-osr/src/export_single_label_lists.py --config severstral-osr/configs/default.yaml
python severstral-osr/src/train_patchcore.py --config severstral-osr/configs/default.yaml
python severstral-osr/src/train_classifier.py --config severstral-osr/configs/split_a.yaml
python severstral-osr/src/extract_embeddings.py --config severstral-osr/configs/split_a.yaml
python severstral-osr/src/run_osr.py --config severstral-osr/configs/split_a.yaml
python severstral-osr/src/run_cascade.py --config severstral-osr/configs/split_a.yaml
```

Use splits `split_a.yaml`..`split_d.yaml` to rotate unknown defect classes.

## Current Split Definitions
- `split_a`: known = Class_1, Class_2, Class_3 (unknown = Class_4)
- `split_b`: known = Class_1, Class_2, Class_4 (unknown = Class_3)
- `split_c`: known = Class_1, Class_3, Class_4 (unknown = Class_2)
- `split_d`: known = Class_2, Class_3, Class_4 (unknown = Class_1)
