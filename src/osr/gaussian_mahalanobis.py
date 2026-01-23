import numpy as np


def fit_gaussians(embeddings: np.ndarray, labels: np.ndarray, cov_reg_lambda: float):
    classes = sorted(list(set(labels.tolist())))
    params = {}
    for cls in classes:
        idx = labels == cls
        feats = embeddings[idx]
        mu = feats.mean(axis=0)
        cov = np.cov(feats, rowvar=False)
        cov = cov + cov_reg_lambda * np.eye(cov.shape[0])
        inv_cov = np.linalg.inv(cov)
        params[cls] = {"mu": mu, "inv_cov": inv_cov}
    return params


def mahalanobis_distance(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray):
    diff = x - mu
    return float(diff.T @ inv_cov @ diff)


def min_mahalanobis(x: np.ndarray, params: dict):
    dists = []
    for cls, p in params.items():
        d = mahalanobis_distance(x, p["mu"], p["inv_cov"])
        dists.append(d)
    return min(dists)


def batch_min_mahalanobis(X: np.ndarray, params: dict):
    return np.array([min_mahalanobis(x, params) for x in X])
