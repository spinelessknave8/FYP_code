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

### Literature-backed reporting guidance (point form + refs)
- **Industrial objective priority**
  - Prioritize high defect recall under explicit false-alarm constraints (not accuracy-only optimization).
  - Use operating-point metrics in addition to threshold-free metrics.
  - Refs: [R1], [R2]

- **Core metrics to report**
  - Threshold-free: `AUROC_defect_screening`, `AUPRC_defect_screening`
  - Operating-point: `TPR_defect@FPR_normal`, `TPR_unknown@FPR_normal`, `FPR_known_as_unknown@FPR_normal`
  - System-level: 3-way `accuracy`, `balanced_accuracy`, `macro_f1`
  - Runtime/deployability: `train_sec`, `infer_sec_per_image`, `total_split_sec`
  - Refs: [R2], [R3], [R4], [R5]

- **Recommended operating points**
  - Report at least `FPR_normal = 5%, 10%, 20%` (and `15%` as sensitivity/ablation).
  - Include stricter low-FPR commentary where possible (industry often evaluates very low false alarms).
  - Refs: [R1], [R2], [R6]

- **Protocol rules (fair comparison / no leakage)**
  - Fixed known/unknown split definition per split.
  - Unknown defects are strictly test-only (no training/calibration leakage).
  - Validation-only threshold calibration; no test-set tuning/early-stopping decisions.
  - Keep preprocessing/input size/backbone consistent across one-stage vs two-stage comparison.
  - Refs: [R1], [R2], [R3]

- **OSR-style metrics vs industrial AD metrics**
  - OSCR/FPR@95TPR are common in generic OSR literature.
  - Industrial AD papers more commonly center AUROC/AUPRO plus deployment-facing operating metrics (PG2/PB2, recall/FAR style).
  - In thesis: OSCR can be mentioned as context, but industrial metric stack should remain primary.
  - Refs: [R1], [R2], [R7], [R8]

- **Preprocessing guidance**
  - Keep AD preprocessing simple and consistent (resize/normalize); avoid aggressive crop tricks that bias false-alarm behavior.
  - Classical heavy preprocessing and defect-specific filtering are more common in closed-set steel detection literature than one-class AD benchmarks.
  - Refs: [R1], [R3], [R9]

- **Fair one-stage vs two-stage comparison checklist**
  - Same dataset splits and same FPR targets.
  - Same threshold calibration protocol.
  - Same reporting unit (per-split + mean/std).
  - Include both effectiveness metrics and runtime/latency metrics.
  - Refs: [R1], [R2], [R3], [R4]

#### References (use as citation keys in thesis draft)
- [R1] Baitieva et al., *Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection (VIAD)*, 2025. (PG2/PB2, protocol cautions, realistic industrial evaluation)
- [R2] MVTec AD / MVTec AD 2 benchmark conventions (AUROC/AUPRO and stricter FPR-capped localization emphasis).
- [R3] Gudovskiy et al., *CFLOW-AD*, WACV 2022. (industrial AD metrics + runtime/model size reporting)
- [R4] Roth et al., *PatchCore*, CVPR 2022. (strong one-class industrial baseline + runtime focus)
- [R5] FR-PatchCore, Sensors 2024. (industrial-style FAR/recall reporting on additional real datasets)
- [R6] Operating-point style metrics in industrial QA: PB2/PG2 (TNR/TPR under fixed low error constraints).
- [R7] OSR literature using OSCR/FPR@95TPR (e.g., C2AE and large-scale OSR protocol papers).
- [R8] Open-set metrics context: OSCR useful for theory, less dominant in industrial AD benchmark reporting.
- [R9] Steel-surface survey (closed-set focus): classical preprocessing is common there but should be separated from one-class open-set AD protocol choices.

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
- `notes/references.md` -> working bibliography with links for citation keys `[R1]..[R9]`.
- `notes/data/raw/` -> raw metric CSVs copied from Colab/Drive outputs.
- `notes/data/clean/` -> cleaned/renamed CSVs for analysis and thesis tables.
- `notes/data/*.csv` -> artifact mapping/index tables (source notebook -> output path).
- `notes/figures_folder/` -> copied/generated plots and figure manifests for thesis use.

### Raw metric CSVs currently present in repo
- `notes/data/raw/FYP data - onestage explorations.csv`
- `notes/data/raw/FYP data - one_stage_summarised.csv`
- `notes/data/raw/FYP data - cflow full runs.csv`
- `notes/data/raw/FYP data - cflow_mean_metrics.csv`
- `notes/data/raw/FYP data - anomaly detector model comparison.csv`
- `notes/data/raw/FYP data - one_stage_vs_tw_stage.csv`
- `notes/data/raw/FYP data - one_stage_vs_two_stage_summary.csv`
- `notes/data/raw/FYP data - classifier_model_pilot.csv`
- `notes/data/raw/FYP data - classifier_model_pilot_summary.csv`

### Cleaned/renamed CSVs (recommended to use)
- `notes/data/clean/one_stage_exploration_per_split.csv`
- `notes/data/clean/one_stage_exploration_mean_metrics.csv`
- `notes/data/clean/cflow_full_runs.csv`
- `notes/data/clean/cflow_mean_metrics.csv`
- `notes/data/clean/anomaly_detector_model_comparison.csv`
- `notes/data/clean/one_stage_vs_two_stage_full_comparison.csv`
- `notes/data/clean/one_stage_vs_two_stage_mean_std.csv`
- `notes/data/clean/classifier_model_pilot.csv`
- `notes/data/clean/classifier_model__pilot_summary.csv`
- `notes/data/clean/README_clean_files.csv` (raw->clean conversion log)

### Rename mapping (raw -> clean)
- `FYP data - onestage explorations.csv` -> `one_stage_exploration_per_split.csv`
- `FYP data - one_stage_summarised.csv` -> `one_stage_exploration_mean_metrics.csv`
- `FYP data - cflow full runs.csv` -> `cflow_full_runs.csv`
- `FYP data - cflow_mean_metrics.csv` -> `cflow_mean_metrics.csv`
- `FYP data - anomaly detector model comparison.csv` -> `anomaly_detector_model_comparison.csv`
- `FYP data - one_stage_vs_tw_stage.csv` -> `one_stage_vs_two_stage_full_comparison.csv`
- `FYP data - one_stage_vs_two_stage_summary.csv` -> `one_stage_vs_two_stage_mean_std.csv`
- `FYP data - classifier_model_pilot.csv` -> `classifier_model_pilot.csv`
- `FYP data - classifier_model_pilot_summary.csv` -> `classifier_model__pilot_summary.csv`

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

---

## Preprocessing Ablations

### Border-crop pilot (Severstal)
- Goal: test whether cropping non-black border regions improves stage-1 defect screening.
- Setup: small pilot split (`normal_train=300`, `normal_val=120`, `normal_test=200`, `defect_test=300`), ResNet18 embedding + 1-class Gaussian (Mahalanobis), calibrated at `FPR_normal=10%`.
- Result (`/content/drive/MyDrive/fyp_outputs/preprocessing_pilot/border_crop_pilot.csv`):
  - `no_crop`: AUROC `0.5190`, `TPR_defect@FPR10=0.1533`, realized `FPR_normal=0.0900`
  - `border_crop`: AUROC `0.5094`, `TPR_defect@FPR10=0.1500`, realized `FPR_normal=0.0950`
  - Delta (border - no crop): AUROC `-0.0096`, `TPR_defect@FPR10=-0.0033`, `FPR_normal=+0.0050`
- Decision: border-cropping did not improve performance and was excluded from the final pipeline.

---

## Exact Methodology Followed (Step-by-Step + Justification)

### 1) Problem framing and protocol lock
- Task locked as open-set industrial defect recognition with three outcomes: `normal`, `known defect`, `unknown defect`.
- Justification: matches deployment reality where unseen defect types appear after deployment.

### 2) Split construction
- Per split: 3 known defect classes for training/validation, 1 held-out defect class as unknown test-only.
- Normal images used across train/val/test for screening calibration and evaluation.
- Justification: enforces true open-set testing and prevents leakage.

### 3) One-stage integrated pipeline design
- Train classifier backbone on known classes only.
- Use embeddings for:
  - defect screening (normal vs defect),
  - known-vs-unknown separation inside defects.
- Candidate scorers explored: global Mahalanobis, kNN, OCSVM-RBF, IsolationForest, energy score, class-conditional Mahalanobis.
- Justification: compare classical low-cost scorers under one common embedding pipeline and pick the best trade-off.

### 4) Two-stage reference pipeline design
- Two-stage CFLOW setup used as comparator:
  - anomaly screening + defect decision pathway,
  - evaluated with same split protocol and same operating-point logic.
- Justification: direct one-stage vs two-stage comparison under matched conditions.

### 5) Threshold calibration and operating points
- Thresholds calibrated on validation only (ID-only), no unknown-class tuning.
- Operating points evaluated at target `FPR_normal` values (`5%, 10%, 15%, 20%` in sweeps; report at least `5/10/20`).
- Justification: aligns with industrial practice of maximizing recall under explicit false-alarm budgets.

### 6) Metrics collected
- Screening quality: `AUROC_defect_screening`, `AUPRC_defect_screening`.
- Operating-point metrics: `TPR_defect@FPR`, `TPR_unknown@FPR`, `FPR_known_as_unknown@FPR`, realized `FPR_normal`.
- System metrics: 3-way accuracy, balanced accuracy, macro-F1, confusion structure.
- Runtime metrics: train seconds, inference seconds per image, total split runtime.
- Justification: supports deployment trade-off analysis, not only leaderboard-style AUROC.

### 7) Data and result governance
- Raw outputs copied into `notes/data/raw/`; cleaned/renamed analysis tables in `notes/data/clean/`.
- Artifact indexes track notebook/source path to output files for traceability.
- Justification: reproducible write-up and auditable thesis appendix.

### 8) Preprocessing ablation policy
- Candidate preprocessing changes tested with small pilots before inclusion.
- Border-crop ablation run and rejected due to small negative deltas.
- Justification: avoid adding preprocessing complexity without measurable gain.

### 9) Final model selection logic used
- Prefer models that keep false alarms controlled while preserving defect/unknown recall.
- One-stage winner chosen from scorer sweep; two-stage CFLOW retained as structured comparator despite higher runtime and false-alarm issues.
- Justification: thesis compares realistic operating trade-offs, not only best single metric.

---

## Thesis Writing Map (What to show, where, and why)

This section maps your existing CSVs/plots to report sections so the final write-up is consistent and defensible.

### A) Methodology section (what was explored + selection logic)

#### A1. Candidate model exploration (one-stage)
- **Table**: model exploration summary (means/std)
  - Source: `notes/data/clean/one_stage_exploration_mean_metrics.csv`
  - Include columns: `method`, `screening_auroc_test_mean`, `tpr_defect_mean`, `fpr_normal_mean`, `tpr_unknown_within_defect_mean`, `macro_f1_3way_mean`.
- **Figure**: bar chart ranking explored one-stage methods.
- **Why this belongs here**: shows that final one-stage choice was evidence-driven, not arbitrary.

#### A2. Classifier exploration (pilot)
- **Table**: classifier pilot summary
  - Source: `notes/data/clean/classifier_model__pilot_summary.csv`
  - Include: `test_accuracy_mean`, `macro_f1_mean`, `train_sec_mean`.
- **Figure**: side-by-side bars for classifier candidates.
- **Why this belongs here**: documents backbone/classifier choice process.

#### A3. Preprocessing ablation
- **Table**: border-crop pilot result (`no_crop` vs `border_crop`)
  - Source: `/content/drive/MyDrive/fyp_outputs/preprocessing_pilot/border_crop_pilot.csv`
- **Why this belongs here**: proves preprocessing decisions were tested and justified.

### B) Experimental setup / protocol section
- **Protocol table** (small): split definition, known vs unknown rule, calibration rule, FPR targets, no leakage policy.
- **Why**: makes one-stage vs two-stage comparison fair and reproducible.

### C) Results section (main comparison one-stage vs two-stage)

#### C1. Main per-method comparison across operating points
- **Primary table**: mean/std by method and FPR target
  - Source: `notes/data/clean/one_stage_vs_two_stage_mean_std.csv`
- **Primary figure set** (line plots, same axes):
  1) `TPR_defect@FPR_mean` vs `fpr_target`
  2) `TPR_unknown@FPR_mean` vs `fpr_target`
  3) `macro_f1_3way_mean` vs `fpr_target`
  4) `balanced_acc_3way_mean` vs `fpr_target`
- **Why**: these four plots show detection/open-set quality trends under changing false-alarm budgets.

#### C2. Per-split robustness
- **Table (appendix or compact in main text)**:
  - Source: `notes/data/clean/one_stage_vs_two_stage_full_comparison.csv`
  - Include per-split values for key metrics.
- **Why**: demonstrates stability and variance across splits (not just averaged performance).

#### C3. Runtime / industrial deployment trade-offs
- **Table**: `train_sec_mean`, `infer_sec_per_image_mean`, `total_split_sec_mean`
  - Source: `notes/data/clean/one_stage_vs_two_stage_mean_std.csv`
- **Figure**: efficiency-quality scatter
  - x = `infer_sec_per_image`, y = `macro_f1_3way`
  - optional marker size by `train_sec`.
- **Why**: directly supports FYP objective (trade-offs, pros/cons, not accuracy-only).

### D) Discussion section (advantages/disadvantages)
- **Use evidence from results**:
  - one-stage strengths: better overall balance / lower false alarms (if supported by table values).
  - two-stage strengths: higher defect recall at relaxed thresholds (if supported).
  - two-stage weaknesses: false alarm inflation and higher training/runtime cost.
- **Why**: converts metric observations into engineering conclusions for industrial use.

### E) Appendix (extra but useful)
- Full metric tables per split.
- Additional exploration plots.
- Data/artifact mapping files for reproducibility:
  - `notes/data/one_stage_integrated_methods_outputs.csv`
  - `notes/data/two_stage_cflow_outputs.csv`
  - `notes/data/one_stage_vs_two_stage_outputs.csv`

---

## Metric Selection Rationale (what to say explicitly in thesis)

- `AUROC_defect_screening` / `AUPRC_defect_screening`:
  - threshold-independent quality of screening score.
- `TPR_defect@FPR`:
  - industrial priority metric (defect catch rate under fixed false-alarm budget).
- `TPR_unknown@FPR` and `FPR_known_as_unknown@FPR`:
  - open-set reliability (reject unknowns without over-rejecting known defects).
- `FPR_normal_realized`, `Specificity_normal`, `FalseAlarms_per_100`:
  - operator workload / overkill impact in production.
- `acc_3way`, `balanced_acc_3way`, `macro_f1_3way`:
  - whole-system behavior across all three labels (normal/known/unknown).
- `train_sec`, `infer_sec_per_image`, `total_split_sec`:
  - practical deployment and maintenance cost.

### Selection rule used to choose final models
- Primary: maximize `TPR_defect@FPR` and `TPR_unknown@FPR` under controlled `FPR_normal_realized`.
- Secondary: maximize `macro_f1_3way` / `balanced_acc_3way`.
- Tie-breaker: lower `infer_sec_per_image` and lower operational complexity.

---

## Deep Research Backlog (Questions to Validate with Citations)

1. What false-alarm budgets are commonly used in industrial vision QA when reporting defect recall (e.g., FPR=1/2/5/10%)?
2. For open-set industrial defect settings, which primary operating metric is preferred in papers and in practice (TPR@FPR, FNR caps, PG2/PB2, FPR@95TPR)?
3. What reporting standard is expected for unknown-defect handling (unknown recall, known-as-unknown error, OSCR, AUROC, AUPR)?
4. Are there accepted threshold-calibration rules for ID-only validation without unknown leakage?
5. What runtime/latency metrics do papers and factory case studies report for deployability (train time, per-image inference, throughput)?
6. In steel/surface defect use-cases, what preprocessing choices are commonly justified (crop borders, grayscale normalization, component filtering), and when are they discouraged?
7. For CFLOW/RD4AD/SimpleNet/PatchCore-like models, what operating-point ranges are realistic on industrial datasets beyond MVTec?
8. How should one-stage vs two-stage systems be compared fairly in literature (same splits, same FPR targets, same unknown protocol, same reporting units)?

### Ready-to-use Deep Research Prompt
Use this prompt with Perplexity/Claude/Gemini:

> I am writing an undergraduate thesis on open-set industrial defect detection (steel surface domain).  
> My protocol uses known/unknown class splits and reports metrics at fixed false-alarm budgets.  
> Please provide a source-backed synthesis (2021–latest, prioritize peer-reviewed CVPR/ICCV/ECCV/WACV, TPAMI/TII/TIM/TMM, and strong industrial benchmark papers) answering the following:
> 1) What operating priorities are standard in industrial QA: maximize defect recall under fixed FPR, or maximize balanced accuracy/F1?  
> 2) What operating-point metrics are most accepted (e.g., TPR@FPR=x, FPR@95TPR, PG2/PB2, OSCR), and what typical x values are used?  
> 3) What protocol rules are required for fair open-set evaluation (no unknown leakage, split design, calibration design, multi-split reporting)?  
> 4) What runtime/deployability metrics should be reported for factory relevance?  
> 5) For one-stage vs two-stage industrial defect systems, what are recommended fair-comparison practices?  
>  
> Output format required:
> - Section A: concise conclusions (10 bullets max)  
> - Section B: evidence table with paper, year, dataset, protocol type, metrics, key numbers, and how comparable it is to my setup  
> - Section C: recommended reporting template for my thesis (what to report in main table and appendix)  
> - Section D: direct links to primary sources  
>  
> Constraints:
> - Mark explicitly what is direct evidence vs inference/extrapolation.  
> - Prefer up-to-date papers and real industrial datasets, not only MVTec.  
> - Avoid unsupported claims; include citation links next to each claim.

### Caveats to state in thesis
- CFLOW full runs show high recall on some splits but weak false-alarm control at selected operating points.
- Some secondary timing metrics for CFLOW were reconstructed from cache/inference post-processing, not end-to-end re-timed training runs.
- One-stage all-method sweep CSVs are currently not present in repo; final narrative should rely on available raw comparison CSVs unless those sweep files are restored.

---

## Interim-to-Current Gap Notes

Source checked: `notes/FYP Interim (1).pdf` (section outline indicates one-stage integrated OSR focus with Mahalanobis and splits A/B/C).

### What changed since interim
- Scope expanded from primarily one-stage integrated OSR to explicit **one-stage vs two-stage (CFLOW)** comparison.
- Dataset focus stabilized around Severstal open-set protocol with explicit `normal / known / unknown` reporting and matched operating points.
- Evaluation moved from mostly recognition-style reporting to **industrial operating-point reporting** (`TPR@FPR`, unknown rejection behavior, runtime metrics).
- Experimentation widened from one Mahalanobis-centric setup to scorer sweeps (Mahalanobis, kNN, OCSVM, IsolationForest, etc.) and preprocessing ablations.
- Repository governance improved (raw/clean CSV tracking, artifact mapping, methodology lock notes).

### Gaps to close in final report (relative to interim)
- Add a dedicated section that explains why AUROC alone is insufficient and why operating-point metrics are primary for industrial QA.
- Add explicit fairness protocol for one-stage vs two-stage comparison (same splits, same calibration rules, no unknown leakage).
- Add deployment trade-off discussion (false alarms vs defect miss risk + latency/throughput implications).
- Add a short preprocessing ablation subsection (including border-crop pilot result and decision to exclude).
- Ensure methodology chapter includes both systems end-to-end with a unified decision flow diagram/table.

### Literature review sufficiency status
- **Partially sufficient** as a base: core open-set and integrated OSR framing exists.
- **Needs strengthening** for final thesis with:
  - recent industrial AD benchmark practice (VIAD/MVTec AD 2 style metrics/protocol),
  - explicit operating-point metric rationale (PG2/PB2 or equivalent),
  - fair-comparison protocol references for one-stage vs two-stage systems,
  - clearer separation between closed-set steel defect literature and one-class/open-set AD literature.
