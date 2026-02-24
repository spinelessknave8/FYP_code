import json
import os
from typing import Iterable, Dict, Any
import time

from ..utils.io import load_yaml, deep_update
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


def _load_config(config_path: str) -> Dict[str, Any]:
    default_path = os.path.join("configs", "default.yaml")
    cfg = load_yaml(default_path)
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def _split_name_from_config(config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    if name.startswith("neu_"):
        name = name.replace("neu_", "")
    if name.endswith(".colab"):
        name = name[: -len(".colab")]
    return name


def _split_output_paths(config_path: str) -> Dict[str, str]:
    cfg = _load_config(config_path)
    split_name = _split_name_from_config(config_path)
    split_dir = os.path.join(cfg["output_dir"], split_name)
    return {
        "split_dir": split_dir,
        "osr_metrics": os.path.join(split_dir, "osr", "metrics.json"),
        "cascade_metrics": os.path.join(split_dir, "cascade", "metrics.json"),
    }


def run_two_stage_stage1(config_path: str = "configs/default.yaml") -> None:
    t0 = time.time()
    print(f"[notebook] stage1 patchcore start: {config_path}")
    train_patchcore(config_path)
    print(f"[notebook] stage1 patchcore done in {time.time() - t0:.1f}s")


def run_split_pipeline(split_config_path: str, skip_if_complete: bool = False) -> None:
    if skip_if_complete:
        paths = _split_output_paths(split_config_path)
        osr_done = os.path.exists(paths["osr_metrics"])
        cascade_done = os.path.exists(paths["cascade_metrics"])
        if osr_done and cascade_done:
            print(f"[notebook] split pipeline skipped (already complete): {split_config_path}")
            print(f"[notebook]   osr metrics: {paths['osr_metrics']}")
            print(f"[notebook]   cascade metrics: {paths['cascade_metrics']}")
            return
        print(f"[notebook] split pipeline incomplete, running: {split_config_path}")
        print(f"[notebook]   osr metrics exists: {osr_done}")
        print(f"[notebook]   cascade metrics exists: {cascade_done}")

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


def run_all_split_pipelines(split_configs: Iterable[str] = DEFAULT_SPLITS, skip_if_complete: bool = False) -> None:
    for split_config in split_configs:
        run_split_pipeline(split_config, skip_if_complete=skip_if_complete)


def load_cascade_metrics(split_name: str, output_dir: str = "outputs") -> Dict[str, Any]:
    metrics_path = os.path.join(output_dir, split_name, "cascade", "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)
