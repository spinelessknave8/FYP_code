import numpy as np
from sklearn.metrics import roc_curve
from .calibrate import calibrate_threshold
from .gaussian_mahalanobis import fit_gaussians, batch_min_mahalanobis
from ..utils.metrics import compute_auroc, compute_known_metrics
from ..utils.plots import plot_roc, plot_histogram


def evaluate_osr(emb_train, y_train, emb_val, y_val, emb_known_test, y_known_test, emb_unknown_test, cov_reg_lambda, accept_rate, out_dir):
    params = fit_gaussians(emb_train, y_train, cov_reg_lambda)
    val_scores = batch_min_mahalanobis(emb_val, params)
    tau = calibrate_threshold(val_scores, accept_rate)

    known_scores = batch_min_mahalanobis(emb_known_test, params)
    unknown_scores = batch_min_mahalanobis(emb_unknown_test, params)

    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    scores = np.concatenate([known_scores, unknown_scores])

    auroc = compute_auroc(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    plot_roc(f"{out_dir}/roc_osr.png", fpr, tpr)
    plot_histogram(f"{out_dir}/hist_osr.png", known_scores, unknown_scores)

    # Open-set decision
    known_accept = known_scores <= tau
    unknown_accept = unknown_scores <= tau
    tpr_unknown = 1.0 - float(np.mean(unknown_accept))
    fpr_known = 1.0 - float(np.mean(known_accept))

    return {
        "tau": tau,
        "auroc": float(auroc),
        "tpr_unknown": tpr_unknown,
        "fpr_known": fpr_known,
    }
