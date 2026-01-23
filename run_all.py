import os
import argparse
import csv

from src.pipelines.train_patchcore import main as train_patchcore
from src.pipelines.train_classifier import main as train_classifier
from src.pipelines.extract_embeddings import main as extract_embeddings
from src.pipelines.run_osr import main as run_osr
from src.pipelines.run_cascade import main as run_cascade
from src.pipelines.benchmark_runtime import main as run_benchmark
from src.utils.io import load_yaml, deep_update


def load_config(config_path: str) -> dict:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def main(config_path: str):
    cfg = load_config(config_path)
    output_dir = cfg["output_dir"]

    # Train PatchCore once
    patchcore_dir = os.path.join(output_dir, "patchcore")
    if not os.path.exists(os.path.join(patchcore_dir, "memory.npy")):
        train_patchcore(config_path)

    split_configs = [
        "configs/neu_split_a.yaml",
        "configs/neu_split_b.yaml",
        "configs/neu_split_c.yaml",
    ]

    summary_path = os.path.join(output_dir, "summary.csv")
    rows = []

    for sc in split_configs:
        train_classifier(sc)
        extract_embeddings(sc)
        run_osr(sc)
        run_cascade(sc)
        run_benchmark(sc)

        split_name = os.path.splitext(os.path.basename(sc))[0].replace("neu_", "")
        metrics_path = os.path.join(output_dir, split_name, "metrics.json")
        runtime_path = os.path.join(output_dir, split_name, "runtime.json")

        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        if os.path.exists(runtime_path):
            import json
            with open(runtime_path, "r") as f:
                runtime = json.load(f)
        else:
            runtime = {}

        if "osr" in metrics:
            rows.append({
                "split": split_name,
                "approach": "osr",
                "auroc_known_unknown": metrics["osr"].get("auroc_known_unknown"),
                "known_acc": metrics["osr"].get("known_accuracy"),
                "runtime_ms": runtime.get("osr_ms", {}).get("mean_ms"),
                "notes": "",
            })
        if "cascade" in metrics:
            rows.append({
                "split": split_name,
                "approach": "cascade",
                "auroc_known_unknown": metrics["cascade"].get("auroc_known_unknown_conf"),
                "known_acc": "n/a",
                "runtime_ms": runtime.get("cascade_ms", {}).get("mean_ms"),
                "notes": "no_defect ignored",
            })

    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "approach", "auroc_known_unknown", "known_acc", "runtime_ms", "notes"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
