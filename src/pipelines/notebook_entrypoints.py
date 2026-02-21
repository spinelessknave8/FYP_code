import json
import os
from typing import Iterable, Dict, Any
import time

from .two_stage.train_patchcore import main as train_patchcore
from .one_stage.train_classifier import main as train_classifier
from .one_stage.extract_embeddings import main as extract_embeddings
from .one_stage.run_osr import main as run_osr
from .two_stage.run_cascade import main as run_cascade


DEFAULT_SPLITS = (
    "configs/neu_split_a.yaml",
    "configs/neu_split_b.yaml",
    "configs/neu_split_c.yaml",
)


def run_two_stage_stage1(config_path: str = "configs/default.yaml") -> None:
    t0 = time.time()
    print(f"[notebook] stage1 patchcore start: {config_path}")
    train_patchcore(config_path)
    print(f"[notebook] stage1 patchcore done in {time.time() - t0:.1f}s")


def run_split_pipeline(split_config_path: str) -> None:
    t0 = time.time()
    print(f"[notebook] split pipeline start: {split_config_path}")
    t = time.time()
    print("[notebook]   train classifier...")
    train_classifier(split_config_path)
    print(f"[notebook]   classifier done in {time.time() - t:.1f}s")
    t = time.time()
    print("[notebook]   extract embeddings...")
    extract_embeddings(split_config_path)
    print(f"[notebook]   embeddings done in {time.time() - t:.1f}s")
    t = time.time()
    print("[notebook]   run one-stage osr...")
    run_osr(split_config_path)
    print(f"[notebook]   osr done in {time.time() - t:.1f}s")
    t = time.time()
    print("[notebook]   run two-stage cascade...")
    run_cascade(split_config_path)
    print(f"[notebook]   cascade done in {time.time() - t:.1f}s")
    print(f"[notebook] split pipeline done: {split_config_path} in {time.time() - t0:.1f}s")


def run_all_split_pipelines(split_configs: Iterable[str] = DEFAULT_SPLITS) -> None:
    for split_config in split_configs:
        run_split_pipeline(split_config)


def load_cascade_metrics(split_name: str, output_dir: str = "outputs") -> Dict[str, Any]:
    metrics_path = os.path.join(output_dir, split_name, "cascade", "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)
