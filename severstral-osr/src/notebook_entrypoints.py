import os
import time

from common import load_config, split_name_from_config
from train_patchcore import train_patchcore_main
from train_classifier import main as train_classifier
from extract_embeddings import main as extract_embeddings
from run_osr import main as run_osr
from run_cascade import main as run_cascade


def run_stage1(config_path: str):
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

    t0 = time.time()
    print(f"[severstal-osr] split start: {split}")
    train_classifier(config_path)
    extract_embeddings(config_path)
    run_osr(config_path)
    run_cascade(config_path)
    print(f"[severstal-osr] split done: {split} in {time.time()-t0:.1f}s")
