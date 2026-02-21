# Dataset Placement

## NEU Surface Defect Dataset (Classification)

Expected layout (your provided structure):

```
data/neu/
  train/
    images/
      crazing/
      inclusion/
      patches/
      pitted_surface/
      rolled-in_scale/
      scratches/
  validation/
    images/
      crazing/
      inclusion/
      patches/
      pitted_surface/
      rolled-in_scale/
      scratches/
```

If your dataset uses different folder names, update:
- `configs/default.yaml` under `neu.class_mapping`
- `src/datasets/neu.py` if needed

## Severstal Steel Defect Detection (PatchCore Stage 1)

Expected layout:

```
data/severstal/
  train.csv
  train_images/
    *.jpg
```

Stage-1 training filters `train.csv` to keep only **normal/no-defect** images
(`EncodedPixels` empty), then extracts non-overlapping `224x224` patches.
A validation split (default 10%) is held out for threshold calibration.
