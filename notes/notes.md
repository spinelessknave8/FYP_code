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

---

## Latest Run Snapshot (Working Notes)

### CFLOW two-stage full-mode status
- Notebook confirms `PILOT_MODE=False` and full-size manifests.
- Unknown split bug fixed: `split_c` now has non-zero unknown test samples.
- Full run currently shows high recall on some splits but weak false-alarm control (realized FPR much higher than target on several splits).

### One-stage vs two-stage merge status
- Comparison notebook runs and merges outputs, but some CFLOW secondary metrics were previously `NaN` because they were not written in the CFLOW summary schema.
- Action: compute secondary metrics from saved stage1/stage2 artifacts without retraining and rewrite harmonized CSV.

### Training-time note (provided by run log)
- CFLOW total training time reported by user: **5h 47m 32s**
- Converted value used for metrics: `train_sec = 20852`.
- For now, if no measured inference runtime artifact exists, `total_split_sec` for CFLOW is set from available values (train-only or train+inference when available).

### Local artifact index files in repo
- `notes/data/one_stage_integrated_methods_outputs.csv`
- `notes/data/two_stage_cflow_outputs.csv`
- `notes/data/one_stage_vs_two_stage_outputs.csv`

These CSVs map each experiment artifact group to source notebook + Drive output path + status label, so write-up can reference outputs systematically.

---

## Repository Organization (Current)

### Notes folder structure
- `notes/notes.md` -> master research log and methodology tracker.
- `notes/data/raw/` -> raw metric CSVs copied from Colab/Drive outputs.
- `notes/data/*.csv` -> artifact mapping/index tables (source notebook -> output path).
- `notes/figures/` -> reserved for copied plots/images.

### Raw metric CSVs currently present in repo
- `notes/data/raw/cflow_two_stage_summary.csv`
- `notes/data/raw/cflow_two_stage_mean_std.csv`
- `notes/data/raw/one_stage_locked_full_summary.csv`
- `notes/data/raw/two_stage_cflow_harmonized.csv`
- `notes/data/raw/two_stage_cflow_enriched_from_cache.csv`
- `notes/data/raw/one_stage_vs_two_stage_full_comparison.csv`
- `notes/data/raw/one_stage_vs_two_stage_mean_std.csv`

### Missing/Not currently copied into repo
- One-stage all-method sweep CSVs from `running_new_models.ipynb`:
  - `one_stage_scorer_sweep_all_splits.csv`
  - `one_stage_best_per_method_per_split.csv`
  - `one_stage_method_summary_mean_std.csv`
- These were missing in the latest Drive export step and should be regenerated/re-copied only if needed for final thesis appendix.

---

## End-to-End Journey (Condensed Narrative for Thesis)

1. Started with cross-domain and then Severstal-focused open-set experiments; encountered instability from runtime disconnects/cached artifacts.
2. Standardized methodology:
   - one-stage integrated OSR (embedding + distance-based scoring),
   - two-stage CFLOW pipeline,
   - strict split hygiene (known/unknown separation, no unknown leakage in calibration).
3. Built one-stage method sweep notebook and compared multiple scorers:
   - Mahalanobis, kNN, OCSVM, IsolationForest, energy, class-conditional Mahalanobis.
4. Added operating-point evaluation at fixed false-alarm budgets (`FPR_normal` grid) and unknown-threshold tuning.
5. Built two-stage CFLOW full-mode pipeline across all splits and fixed split construction issues (`unknown_test` non-zero in split_c).
6. Created unified one-stage vs two-stage comparison notebook and aligned core metrics across methods.
7. Filled missing CFLOW secondary metrics from cached artifacts (without retraining) and injected known training-time value:
   - CFLOW `train_sec = 20852` (5h 47m 32s).
8. Final result artifacts are now consolidated under `notes/data/raw/` for thesis writing and audit trail.

### Caveats to state in thesis
- CFLOW full runs show high recall on some splits but weak false-alarm control at selected operating points.
- Some secondary timing metrics for CFLOW were reconstructed from cache/inference post-processing, not end-to-end re-timed training runs.
- One-stage all-method sweep CSVs are currently not present in repo; final narrative should rely on available raw comparison CSVs unless those sweep files are restored.
