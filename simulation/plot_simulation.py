import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("simulation")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

metrics = []
with (OUT_DIR / "simulated_metrics.csv").open() as f:
    for r in csv.DictReader(f):
        for k in list(r.keys()):
            if k not in {"method", "confusion_matrix_3way"}:
                r[k] = float(r[k])
        metrics.append(r)

roc_rows = []
with (OUT_DIR / "simulated_roc_points.csv").open() as f:
    for r in csv.DictReader(f):
        r["fpr"] = float(r["fpr"])
        r["tpr"] = float(r["tpr"])
        roc_rows.append(r)

# ROC plots
plt.figure(figsize=(6, 5))
for row in metrics:
    method = row["method"]
    sub = [r for r in roc_rows if r["method"] == method and r["curve"] == "defect_roc"]
    plt.plot([r["fpr"] for r in sub], [r["tpr"] for r in sub], label=f"{method} (AUC={row['auroc_defect_screening']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Defect Screening ROC (Normal vs Defect)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_defect_screening.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
for row in metrics:
    method = row["method"]
    sub = [r for r in roc_rows if r["method"] == method and r["curve"] == "open_set_roc"]
    plt.plot([r["fpr"] for r in sub], [r["tpr"] for r in sub], label=f"{method} (AUC={row['auroc_open_set_known_vs_unknown']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Open-Set ROC (Known vs Unknown Defect)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_open_set.png", dpi=180)
plt.close()

# quality bars
methods = [r["method"] for r in metrics]
x = np.arange(len(methods))
w = 0.2
bar_cols = [
    "acc_3way",
    "macro_f1_3way",
    "tpr_unknown_at_fpr_known_10",
    "fpr_normal_at_tpr_defect_95",
]
fig, ax = plt.subplots(figsize=(10, 4))
for i, c in enumerate(bar_cols):
    vals = [r[c] for r in metrics]
    ax.bar(x + (i - 1.5) * w, vals, width=w, label=c)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1)
ax.set_title("Primary Quality Metrics (Simulated)")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "quality_bars.png", dpi=180)
plt.close()

# runtime bars
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
lat_mean = [r["latency_ms_mean"] for r in metrics]
lat_p95 = [r["latency_ms_p95"] for r in metrics]
axes[0].bar(x - 0.15, lat_mean, width=0.3, label="latency_ms_mean")
axes[0].bar(x + 0.15, lat_p95, width=0.3, label="latency_ms_p95")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods)
axes[0].set_title("Latency (ms)")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(fontsize=8)

throughput = [r["throughput_samples_per_sec"] for r in metrics]
axes[1].bar(x, throughput, width=0.45, color=["tab:green", "tab:orange"][: len(methods)])
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods)
axes[1].set_title("Throughput (samples/sec)")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(PLOT_DIR / "runtime_bars.png", dpi=180)
plt.close()

print("Wrote plots to", PLOT_DIR)
