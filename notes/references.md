# References (Working Bibliography)

This file expands the citation keys used in `notes/notes.md` (`[R1]` ... `[R9]`).
Use this as a base for final thesis bibliography formatting (IEEE/APA per school template).

## [R1] VIAD benchmark / industrial best practices
- N. Baitieva et al., “Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection,” 2025.
- Link (arXiv): https://arxiv.org/html/2503.23451v1
- Note: Use this as the main source for PG2/PB2 and protocol cautions (no test leakage, no biased center-crop assumptions).

## [R2] MVTec AD / MVTec AD 2 benchmark conventions
- P. Bergmann et al., “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.
- Dataset page: https://www.mvtec.com/company/research/datasets/mvtec-ad
- MVTec AD 2 (advanced scenarios): https://arxiv.org/abs/2503.21622
- Note: Cite these for AUROC/AUPRO-style evaluation conventions and stricter low-FPR localization emphasis in newer benchmark practice.

## [R3] CFLOW-AD
- D. Gudovskiy, S. Ishizaka, and K. Kozuka, “CFLOW-AD: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows,” WACV 2022.
- Paper: https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf
- Code: https://github.com/gudovskiy/cflow-ad

## [R4] PatchCore
- K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022.
- Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
- Code: https://github.com/amazon-science/patchcore-inspection

## [R5] FR-PatchCore
- (Sensors 2024) “FR-PatchCore: An Industrial Anomaly Detection Method for Improving Generalization.”
- Open-access page: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10934034/
- Note: Use for industrial-style FAR/recall reporting and runtime comparisons in additional real datasets.

## [R6] PG2/PB2 operating-point definitions
- Source: VIAD benchmark metric definitions in [R1].
- Link: https://arxiv.org/html/2503.23451v1
- Note: PG2/PB2 are used to express good-part acceptance / bad-part detection under strict low-error constraints.

## [R7] OSR metric context (OSCR/FPR@95TPR)
- V. Oza and V. M. Patel, “C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition,” CVPR 2019.
- Paper: https://arxiv.org/pdf/1904.01198.pdf
- Note: Use as OSR metric background; do not treat as industrial AD standard by itself.

## [R8] Large-scale OSR protocol context
- Large-scale OSR protocol references (ImageNet-scale OSR; OSCR usage).
- Example repository link (from prior research notes): https://www.ifi.uzh.ch/server/api/core/bitstreams/582f65e2-260e-4938-9ad4-babfd8763c22/content
- Note: Keep as contextual reference unless you add direct OSCR experiments.

## [R9] Steel surface survey (closed-set context)
- B. Ibrahim and J.-R. Tapamo, “A Survey of Vision-Based Methods for Surface Defects’ Detection and Classification in Steel Products,” *Informatics*, 2024.
- Example open copy link: https://pdfs.semanticscholar.org/74b9/44c29661aec48efcc139c8ba82151217382a.pdf
- Note: Use this to separate classical closed-set steel QA workflows from one-class/open-set anomaly detection pipelines.

---

## Optional Additional Sources (already referenced during project)

- CFLOW-AD code/paper page: https://github.com/gudovskiy/cflow-ad
- PatchCore code/paper page: https://github.com/amazon-science/patchcore-inspection
- MVTec AD dataset landing page: https://www.mvtec.com/company/research/datasets/mvtec-ad
- VIAD benchmark code: https://github.com/abc-125/viad-benchmark

