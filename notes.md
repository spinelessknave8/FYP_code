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

---

## Deep Research Notes (for FYP write-up)

### Industry-style priority (what to optimize first)
- In industrial inspection, the default priority is usually **defect catching under a false-alarm constraint**, not raw overall accuracy.
- Practical reporting is typically:
  - `TPR_defect` / `TPR_unknown` at fixed `FPR_normal` (operating points),
  - plus threshold-free `AUROC`.
- For this FYP, primary operating points remain `FPR_normal = 5%, 10%, 20%`.

### Protocol expectations from recent literature
- Explicit known/unknown class split rules.
- No unknown-class leakage into training/calibration.
- Validation-only threshold calibration (ID-only before test).
- Multiple splits/seeds + mean/std reporting.
- Include operating-point metrics (not only AUROC).

### Dataset/benchmark reality relevant to this project
- NEU open-set benchmarks are limited/non-standard; direct 1:1 benchmark targets are scarce.
- Severstal open-set literature is also sparse; most Severstal papers are closed-set detection/segmentation.
- Therefore, method claims should be framed as:
  - protocol-correct open-set evaluation,
  - relative one-stage vs two-stage comparisons under matched constraints.

### Current one-stage exploration summary (pilot, all splits)
- Compared methods: global Mahalanobis, global kNN, OCSVM, IsolationForest, energy score, class-conditional Mahalanobis.
- Shortlist based on latest pilot:
  - `global_mahalanobis` (best balanced),
  - `global_knn` (strong unknown-rejection candidate),
  - `one_class_svm_rbf` (backup).
- `energy_score` and `class_conditional_mahalanobis` underperform in this current setup.

### Selection rule to keep final decision defensible
- Choose model/params under explicit operating constraints, e.g.:
  - maximize `TPR_unknown_within_defect`,
  - then maximize `TPR_defect`,
  - while keeping `FPR_known_as_unknown` under a fixed cap.
- Report final comparison across `FPR_normal = 5%, 10%, 20%` (not a single threshold only).

---

## Research Diary Template (append here each run/research pass)

- Date:
- Notebook / script:
- Objective:
- Data split(s):
- Main settings:
- Key metrics:
- Decision taken:
- Next step:

---

## Progress Log (Current)

### Why FPR points were 5/10/20
- Initial choice was to cover low, medium, and relaxed false-alarm operating points with minimal runtime.
- This was a practical first pass, not a hard rule.

### Updated decision
- For pilot model selection, use **FPR_normal = 5%, 10%, 15%, 20%**.
- For final thesis tables, still report at least **5%, 10%, 20%** (and include 15% as sensitivity if useful).

### Current one-stage exploration status
- Notebook: `severstral-osr/notebooks/running_new_models.ipynb`
- Evaluated one-stage methods across splits:
  - global Mahalanobis
  - global kNN
  - one-class SVM (RBF)
  - Isolation Forest
  - energy score
  - class-conditional Mahalanobis
- Current shortlist:
  - `global_mahalanobis` (best balanced)
  - `global_knn` (strong unknown-rejection candidate)
  - `one_class_svm_rbf` (backup)

### Current two-stage comparison status
- Two-stage CFLOW results exist in `severstral-osr/notebooks/cflow.ipynb` outputs.
- Next step is a fair, matched operating-point comparison against locked one-stage winner.

### Agreed next steps
1. Re-run one-stage sweep with FPR grid {5, 10, 15, 20}.
2. Run two-stage (CFLOW) with the same FPR grid and same split protocol.
3. Compare one-stage winner vs two-stage at matched operating points (per split + mean/std).

### Citation note for write-up
- Main evidence to cite for industrial priority: optimize recall under false-alarm constraints (PG2/PB2 / FPR@TPR style), not accuracy-only ranking.
- Keep direct quotes short and verify wording from the original PDFs before final thesis submission.
