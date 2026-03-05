import csv
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("simulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

n_normal = 1500
n_known = 1200
n_unknown = 900


def clipped_normal(mean, std, n):
    x = rng.normal(mean, std, n)
    return np.clip(x, 0.0, 1.0)


def positive_normal(mean, std, n):
    x = rng.normal(mean, std, n)
    return np.clip(x, 0.1, None)


def roc_from_scores(y_true, scores):
    # y_true in {0,1}
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]

    P = np.sum(y == 1)
    N = np.sum(y == 0)

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    # pick only score change points
    change = np.r_[True, s[1:] != s[:-1]]
    tp_c = tp[change]
    fp_c = fp[change]
    thr = s[change]

    tpr = np.r_[0.0, tp_c / max(1, P), 1.0]
    fpr = np.r_[0.0, fp_c / max(1, N), 1.0]
    thr = np.r_[thr[0] + 1e-9 if len(thr) else 1.0, thr, thr[-1] - 1e-9 if len(thr) else 0.0]

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thr, auc


def confusion_matrix_3(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_prf_from_cm(cm):
    precisions = []
    recalls = []
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


models = {
    "one_stage": {
        "defect_score": {
            "normal": clipped_normal(0.28, 0.14, n_normal),
            "known": clipped_normal(0.70, 0.18, n_known),
            "unknown": clipped_normal(0.76, 0.16, n_unknown),
        },
        "unknown_score": {
            "known": clipped_normal(0.41, 0.19, n_known),
            "unknown": clipped_normal(0.64, 0.19, n_unknown),
        },
        "latency_ms": positive_normal(9.0, 1.1, n_normal + n_known + n_unknown),
    },
    "two_stage": {
        "defect_score": {
            "normal": clipped_normal(0.23, 0.12, n_normal),
            "known": clipped_normal(0.76, 0.15, n_known),
            "unknown": clipped_normal(0.83, 0.13, n_unknown),
        },
        "unknown_score": {
            "known": clipped_normal(0.33, 0.17, n_known),
            "unknown": clipped_normal(0.72, 0.16, n_unknown),
        },
        "latency_ms": positive_normal(14.5, 1.8, n_normal + n_known + n_unknown),
    },
}

rows = []
roc_rows = []

for model_name, d in models.items():
    ds_n = d["defect_score"]["normal"]
    ds_k = d["defect_score"]["known"]
    ds_u = d["defect_score"]["unknown"]

    us_k = d["unknown_score"]["known"]
    us_u = d["unknown_score"]["unknown"]

    y_def = np.r_[np.zeros(len(ds_n), dtype=int), np.ones(len(ds_k) + len(ds_u), dtype=int)]
    s_def = np.r_[ds_n, ds_k, ds_u]
    fpr_def, tpr_def, thr_def, auroc_def = roc_from_scores(y_def, s_def)

    idx_95 = int(np.argmin(np.abs(tpr_def - 0.95)))
    defect_tau = float(thr_def[idx_95])
    fpr_normal_at_95 = float(fpr_def[idx_95])

    y_os = np.r_[np.zeros(len(us_k), dtype=int), np.ones(len(us_u), dtype=int)]
    s_os = np.r_[us_k, us_u]
    fpr_os, tpr_os, thr_os, auroc_os = roc_from_scores(y_os, s_os)

    idx_10 = int(np.argmin(np.abs(fpr_os - 0.10)))
    unk_tau = float(thr_os[idx_10])
    tpr_unknown_at_10 = float(tpr_os[idx_10])

    y3_true = np.r_[
        np.zeros(len(ds_n), dtype=int),
        np.ones(len(ds_k), dtype=int),
        np.full(len(ds_u), 2, dtype=int),
    ]

    all_def_score = np.r_[ds_n, ds_k, ds_u]
    defect_mask = all_def_score > defect_tau
    all_unknown_score = np.r_[np.zeros(len(ds_n)), us_k, us_u]
    unknown_mask = all_unknown_score > unk_tau

    pred = np.zeros_like(y3_true)
    pred[defect_mask] = 1
    pred[defect_mask & unknown_mask] = 2

    acc3 = float(np.mean(pred == y3_true))
    cm = confusion_matrix_3(y3_true, pred)
    macro_p, macro_r, macro_f1 = macro_prf_from_cm(cm)

    lat = d["latency_ms"]
    lat_mean = float(np.mean(lat))
    lat_p95 = float(np.percentile(lat, 95))
    lat_p99 = float(np.percentile(lat, 99))
    throughput = 1000.0 / lat_mean

    rows.append(
        {
            "method": model_name,
            "auroc_defect_screening": auroc_def,
            "auroc_open_set_known_vs_unknown": auroc_os,
            "fpr_normal_at_tpr_defect_95": fpr_normal_at_95,
            "tpr_unknown_at_fpr_known_10": tpr_unknown_at_10,
            "acc_3way": acc3,
            "macro_precision_3way": macro_p,
            "macro_recall_3way": macro_r,
            "macro_f1_3way": macro_f1,
            "latency_ms_mean": lat_mean,
            "latency_ms_p95": lat_p95,
            "latency_ms_p99": lat_p99,
            "throughput_samples_per_sec": float(throughput),
            "defect_threshold": defect_tau,
            "unknown_threshold": unk_tau,
            "confusion_matrix_3way": json.dumps(cm.tolist()),
        }
    )

    for f, t in zip(fpr_def, tpr_def):
        roc_rows.append({"method": model_name, "curve": "defect_roc", "fpr": float(f), "tpr": float(t)})
    for f, t in zip(fpr_os, tpr_os):
        roc_rows.append({"method": model_name, "curve": "open_set_roc", "fpr": float(f), "tpr": float(t)})

metrics_path = OUT_DIR / "simulated_metrics.csv"
roc_path = OUT_DIR / "simulated_roc_points.csv"

with metrics_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with roc_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(roc_rows[0].keys()))
    w.writeheader()
    w.writerows(roc_rows)

summary = {
    "note": "Synthetic data for planning/presentation only. Not real experiment output.",
    "generated_files": {
        "metrics_csv": str(metrics_path),
        "roc_points_csv": str(roc_path),
    },
}
(OUT_DIR / "README_simulation.json").write_text(json.dumps(summary, indent=2))

print("Wrote:")
print("-", metrics_path)
print("-", roc_path)
print("\nSimulated metrics:")
for r in rows:
    print(r)
