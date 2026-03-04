import os
import time

from common import load_config, split_name_from_config
from train_patchcore import train_patchcore_main
from train_classifier import main as train_classifier
from extract_embeddings import main as extract_embeddings
from run_osr import main as run_osr
from run_cascade import main as run_cascade


def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)


def run_stage1(config_path: str):
    cfg = load_config(config_path)
    patchcore_dir = os.path.join(cfg["output_dir"], "patchcore")
    stage1_outputs = [
        os.path.join(patchcore_dir, "memory.npy"),
        os.path.join(patchcore_dir, "threshold.json"),
    ]
    if _all_exist(stage1_outputs):
        print(f"[severstal-osr] stage1 skip (found cached outputs): {patchcore_dir}")
        return

    t0 = time.time()
    print(f"[severstal-osr] stage1 start: {config_path}")
    train_patchcore_main(config_path)
    print(f"[severstal-osr] stage1 done in {time.time()-t0:.1f}s")


def run_split_pipeline(config_path: str, skip_if_complete: bool = False):
    cfg = load_config(config_path)
    split = split_name_from_config(config_path)
    split_dir = os.path.join(cfg["output_dir"], split)
    osr_metrics = os.path.join(split_dir, "osr", "metrics.json")
    cascade_metrics = os.path.join(split_dir, "cascade", "metrics.json")

    if skip_if_complete and os.path.exists(osr_metrics) and os.path.exists(cascade_metrics):
        print(f"[severstal-osr] skip complete split: {split}")
        return

    clf_outputs = [
        os.path.join(split_dir, "classifier", "classifier.pt"),
        os.path.join(split_dir, "classifier", "train_metrics.json"),
    ]
    emb_outputs = [
        os.path.join(split_dir, "embeddings", "known_train.npz"),
        os.path.join(split_dir, "embeddings", "known_val.npz"),
        os.path.join(split_dir, "embeddings", "known_test.npz"),
        os.path.join(split_dir, "embeddings", "unknown_test.npz"),
    ]
    osr_outputs = [
        os.path.join(split_dir, "osr", "metrics.json"),
        os.path.join(split_dir, "osr", "thresholds.json"),
    ]
    cascade_outputs = [
        os.path.join(split_dir, "cascade", "metrics.json"),
    ]

    t0 = time.time()
    print(f"[severstal-osr] split start: {split}")

    if skip_if_complete and _all_exist(clf_outputs):
        print(f"[severstal-osr] skip classifier: {split}")
    else:
        train_classifier(config_path)

    if skip_if_complete and _all_exist(emb_outputs):
        print(f"[severstal-osr] skip embeddings: {split}")
    else:
        extract_embeddings(config_path)

    if skip_if_complete and _all_exist(osr_outputs):
        print(f"[severstal-osr] skip osr: {split}")
    else:
        run_osr(config_path)

    if skip_if_complete and _all_exist(cascade_outputs):
        print(f"[severstal-osr] skip cascade: {split}")
    else:
        run_cascade(config_path)

    print(f"[severstal-osr] split done: {split} in {time.time()-t0:.1f}s")
