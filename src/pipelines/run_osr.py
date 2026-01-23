import os
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score

from ..utils.io import load_yaml, deep_update, save_json
from ..utils.metrics import compute_known_metrics
from ..osr.gaussian_mahalanobis import fit_gaussians, batch_min_mahalanobis
from ..osr.calibrate import calibrate_threshold
from ..utils.plots import plot_roc, plot_histogram
from sklearn.metrics import roc_curve


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def split_name_from_config(config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    if name.startswith("neu_"):
        return name.replace("neu_", "")
    return name


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def main(config_path: str):
    cfg = load_config(config_path)
    split_name = split_name_from_config(config_path)
    out_dir = os.path.join(cfg["output_dir"], split_name)
    emb_dir = os.path.join(out_dir, "embeddings")
    osr_dir = os.path.join(out_dir, "osr")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(osr_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    known_train = np.load(os.path.join(emb_dir, "known_train.npz"))
    known_val = np.load(os.path.join(emb_dir, "known_val.npz"))
    known_test = np.load(os.path.join(emb_dir, "known_test.npz"))
    unknown_test = np.load(os.path.join(emb_dir, "unknown_test.npz"))

    emb_train = known_train["embeddings"]
    y_train = known_train["labels"]
    emb_val = known_val["embeddings"]
    log_val = known_val["logits"]
    emb_known_test = known_test["embeddings"]
    log_known_test = known_test["logits"]
    y_known_test = known_test["labels"]
    emb_unknown_test = unknown_test["embeddings"]
    log_unknown_test = unknown_test["logits"]

    params = fit_gaussians(emb_train, y_train, cfg["osr"]["cov_reg_lambda"])
    val_scores = batch_min_mahalanobis(emb_val, params)
    tau = calibrate_threshold(val_scores, cfg["osr"]["accept_rate"])

    known_scores = batch_min_mahalanobis(emb_known_test, params)
    unknown_scores = batch_min_mahalanobis(emb_unknown_test, params)

    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    scores = np.concatenate([known_scores, unknown_scores])
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

    fpr, tpr, _ = roc_curve(y_true, scores)
    plot_roc(os.path.join(plot_dir, "roc_osr.png"), fpr, tpr)
    plot_histogram(os.path.join(plot_dir, "hist_osr.png"), known_scores, unknown_scores)

    conf_val = np.max(softmax(log_val), axis=1)
    if cfg["osr"]["use_confidence_gate"]:
        kappa = calibrate_threshold(-conf_val, cfg["osr"]["confidence_accept_rate"])
        kappa = -kappa
    else:
        kappa = None

    conf_known = np.max(softmax(log_known_test), axis=1)
    conf_unknown = np.max(softmax(log_unknown_test), axis=1)

    if kappa is not None:
        known_reject = (known_scores > tau) | (conf_known < kappa)
        unknown_reject = (unknown_scores > tau) | (conf_unknown < kappa)
    else:
        known_reject = known_scores > tau
        unknown_reject = unknown_scores > tau

    tpr_unknown = float(np.mean(unknown_reject))
    fpr_known = float(np.mean(known_reject))
    open_set_acc = (np.sum(~known_reject) + np.sum(unknown_reject)) / (len(known_reject) + len(unknown_reject))

    y_pred_known = np.argmax(log_known_test, axis=1)
    known_metrics = compute_known_metrics(y_known_test, y_pred_known)

    metrics = {
        "tau": float(tau),
        "kappa": float(kappa) if kappa is not None else None,
        "auroc_known_unknown": float(auroc),
        "tpr_unknown": tpr_unknown,
        "fpr_known": fpr_known,
        "open_set_acc": float(open_set_acc),
        "known_accuracy": known_metrics["accuracy"],
        "known_macro_f1": known_metrics["macro_f1"],
    }

    save_json(os.path.join(osr_dir, "metrics.json"), metrics)
    save_json(os.path.join(osr_dir, "thresholds.json"), {"tau": float(tau), "kappa": float(kappa) if kappa is not None else None})

    # update global metrics.json
    global_metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(global_metrics_path):
        with open(global_metrics_path, "r") as f:
            global_metrics = json.load(f)
    else:
        global_metrics = {}
    global_metrics["osr"] = metrics
    save_json(global_metrics_path, global_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
