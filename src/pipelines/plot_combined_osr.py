import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from ..utils.io import load_yaml, deep_update
from ..osr.gaussian_mahalanobis import fit_gaussians, batch_min_mahalanobis


SPLITS = {
    "split_a": "configs/neu_split_a.yaml",
    "split_b": "configs/neu_split_b.yaml",
    "split_c": "configs/neu_split_c.yaml",
}


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def compute_scores(output_dir: str, split: str, cfg: dict):
    emb_dir = os.path.join(output_dir, split, "embeddings")
    known_train = np.load(os.path.join(emb_dir, "known_train.npz"))
    known_test = np.load(os.path.join(emb_dir, "known_test.npz"))
    unknown_test = np.load(os.path.join(emb_dir, "unknown_test.npz"))

    cov_reg = float(cfg["osr"]["cov_reg_lambda"])
    params = fit_gaussians(
        known_train["embeddings"],
        known_train["labels"],
        cov_reg,
    )
    known_scores = batch_min_mahalanobis(known_test["embeddings"], params)
    unknown_scores = batch_min_mahalanobis(unknown_test["embeddings"], params)
    return known_scores, unknown_scores


def plot_combined_roc(out_path: str, scores_by_split: dict):
    plt.figure()
    for split, (known_scores, unknown_scores) in scores_by_split.items():
        y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
        scores = np.concatenate([known_scores, unknown_scores])
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=split)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_combined_mahalanobis(out_path: str, scores_by_split: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, (known_scores, unknown_scores) in scores_by_split.items():
        axes[0].hist(known_scores, bins=30, density=True, histtype="step", label=split)
        axes[1].hist(unknown_scores, bins=30, density=True, histtype="step", label=split)
    axes[0].set_title("Known")
    axes[1].set_title("Unknown")
    for ax in axes:
        ax.set_xlabel("Mahalanobis score")
        ax.set_ylabel("Density")
        ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(output_dir: str, out_dir: str):
    scores_by_split = {}
    for split, cfg_path in SPLITS.items():
        cfg = load_config(cfg_path)
        scores_by_split[split] = compute_scores(output_dir, split, cfg)

    plot_combined_roc(os.path.join(out_dir, "roc_combined.png"), scores_by_split)
    plot_combined_mahalanobis(os.path.join(out_dir, "mahalanobis_combined.png"), scores_by_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--out_dir", default=os.path.join("outputs", "combined"))
    args = parser.parse_args()
    main(args.output_dir, args.out_dir)
