# One-Stage Methodology (Locked)

This file defines the **official one-stage methodology** for this FYP so we do not drift again.

## Goal

Single-model open-set defect recognition on Severstal:
- Detect whether an image is `normal` or `defect`
- If `defect`, classify as:
  - one of the **known defect classes** (A/B/C in a split), or
  - `unknown` (held-out defect class)

---

## What One-Stage Is (and Is Not)

### One-stage = integrated embedding + distance scoring
- Backbone: ResNet feature extractor (embedding model)
- No PatchCore cascade gate in front
- No separate stage-1 + stage-2 pipeline

### Explicitly NOT part of one-stage
- PatchCore memory-bank screening
- Two-stage AND/OR fusion rules
- Separate anomaly gate threshold + classifier threshold logic

---

## Dataset Split Logic (per split)

For each split:
- Known defect classes: 3 classes (e.g., A/B/C)
- Unknown defect class: 1 held-out class (e.g., D)
- Normal images: non-defective images from Severstal

Train/val/test groups used:
- `normal_train`, `normal_val`, `normal_test`
- `known_train`, `known_val`, `known_test`
- `unknown_test` (never used in training/calibration)

Rule: `unknown_test` is strictly test-only.

---

## Model + Scoring Pipeline

1. Train classifier backbone on `known_train` (known classes only).  
   Output is used mainly for embeddings in one-stage.

2. Extract embeddings for:
- `normal_train` (for normal-vs-defect model)
- `known_train` (for known-class distribution model)
- test groups (`normal_test`, `known_test`, `unknown_test`)

3. Fit integrated one-stage scoring:
- **Defect screening score** from distance to normal embedding distribution
  (e.g., global Mahalanobis / kNN / OCSVM / IsolationForest)
- **Known-vs-unknown score** from distance to known-class embedding distributions
  (e.g., min class Mahalanobis among known classes)

4. Final one-stage decision:
- If screening score says `normal` -> predict `normal`
- Else predict `defect`, then:
  - if close to known-class distributions -> predict known class label
  - else -> predict `unknown`

This is one integrated decision flow, not a cascade.

---

## Calibration Rules

Thresholds are calibrated on validation only:
- `normal_val` and `known_val`
- no `unknown_test` leakage

Recommended reporting points:
- threshold at target `FPR_normal` = 5%, 10%, 20%

---

## Required Metrics to Report

For each split and averaged across splits:
- `AUROC_defect_screening` (normal vs defect)
- `TPR_defect @ FPR_normal={5,10,20}%`
- `TPR_unknown @ FPR_normal={5,10,20}%`
- `FPR_known_as_def @ FPR_normal={5,10,20}%`
- 3-way metrics (`normal`, `known`, `unknown`):
  - accuracy
  - macro precision / recall / F1
  - confusion matrix
- Runtime:
  - training time
  - inference time per sample

---

## Interpretation Rules

- For **known defect** samples, success requires:
  1) detected as `defect`, and
  2) assigned correct known class (A/B/C)

- For **unknown defect** samples, success requires:
  1) detected as `defect`, and
  2) assigned `unknown` (not forced into A/B/C)

If known/unknown separation is poor, that is a one-stage open-set weakness (not a bug in metric definition).

---

## Change Control

If we change anything below, we must update this file first:
- backbone/embedding method
- scoring family
- threshold calibration rule
- class split protocol
- primary metrics

This keeps experiments comparable and prevents methodology drift.
