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

## MVTec AD (PatchCore)

Expected layout:

```
data/mvtec/
  <category>/{train,test}/...
```

PatchCore uses **train/good** images only. A small validation split (e.g. 10%) is created from train/good for threshold calibration.
